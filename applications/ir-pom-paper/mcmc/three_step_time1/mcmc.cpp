#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
/*
 * Program to compute the posterior distribution
 * -- 3-step mechanism
 * -- Ir-POM chemical system
 * -- Using data from time 2.336
 */

#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <memory>
#include "sampling_custom_ode.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/conversion.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>
#include <sampleflow/consumers/covariance_matrix.h>
#include <sampleflow/consumers/count_samples.h>

#include <omp.h>


// Set the precision of the calculations
using Real = float;



// A data type we will use to convert samples to whenever we want to
// do arithmetic with them, such as if we want to compute mean values,
// covariances, etc.
using VectorType = std::valarray<Real>;


// A data type describing the linear algebra object vector that is used
// in the ODE solver.
using StateVector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;



/*
 * Create an object which holds all of the constant information for the mechanism
 */

class ConstantData
{
public:
  ConstantData();

  /*
   * For the Ir-POM system, nucleation produces a particle of size 3.
   * However, data can only be measured for particles larger than size ~100.
   * We have also seen that particles do not get larger than size 2500 in experiments.
   * We are tracking monomers.
   *
   * We want to have index 0 of vectors refer to the precursor A.
   * We want to have index 1 of vectors refer to the disassociated precursor.
   * We want to have index 2 of vectors refer to the ligand POM.
   *
   * The solvent used in the reaction has a known concentration.
   */
  unsigned int min_size = 3;
  unsigned int max_size = 2500;
  unsigned int conserved_size = 1;

  Real min_bin_size = 1.4;
  Real max_bin_size = 4.1;
  unsigned int hist_bins = 27;

  unsigned int A_index = 0;
  unsigned int As_index = 1;
  unsigned int ligand_index = 2;

  Real solvent = 11.3;


  // The raw data is provided with diameter measurements, but we want to convert that to particle size upon
  // receiving the data. We also want to keep track of the time each piece of data was collected.
  MEPBM::PomData<Real> data_diameter;
  std::vector< std::vector<Real> > data_size;
  std::vector<Real> times;

  // { kb, k1, k2, k3 }
  std::vector<Real> lower_bounds = { 0., 1000., 4800., 10., 10.};
  std::vector<Real> upper_bounds = { 1.e3, 2.e8, 1.e8, 1.e8, 1.e8};

  // Particle size cutoff should be a non-negative integer, unlike the other parameters.
  unsigned int lower_bound_cutoff = 10;
  unsigned int upper_bound_cutoff = 2000;

  // Hold the initial condition for the ODEs, i.e. the starting concentration of each species
  StateVector initial_condition;

};


// Since the data is stored inside of a container, we need to extract the desired data from the container
// and convert diameter measurements to particle size measurements.
ConstantData::ConstantData()
{
  data_size = {data_diameter.tem_diam_time1};

  times = {0., data_diameter.tem_time1};

  initial_condition = StateVector::Zero(max_size + 1);
  initial_condition(0) = 0.0012;
}



/*
 * Create an object of type Sample which holds all of the variable information for the mechanism.
 * This object must be able to interface with SampleFlow, so it must have the following functionality:
 *
 * Member variables:
 *  Whichever variables you want to vary as samples are collected must be indicated here.
 *  A ConstantData object as defined above.
 *
 * Member functions:
 *  std::vector< std::vector<Real> > return_data()
 *      -- a collection of vectors corresponding to collected data which gives particle size for each data.
 *  std::vector<Real> return_times()
 *      -- the first element is intended to be time=0 and the remaining times should correspond to when
 *      the return_data() entries were collected.
 *  Model::Model return_model()
 *      -- an object describing the right hand side of the system of differential equations
 *  std::vector<Real> return_initial_condition()
 *      -- a vector giving the concentrations of the tracked chemical species at time = 0.
 *  Histograms::Parameters return_histogram_parameters()
 *      -- an object describing the to bin together particle sizes to compare the data and simulation results.
 *  bool within_bounds()
 *      -- The intent of this function is to compare the current state of the parameters being sampled and
 *      compare to a pre-determined domain for those parameters.
 *      If any parameter lies outside of its allowed interval, return false. Otherwise, return true.
 *
 *  Operators:
 *    Sample operator = (const Sample &sample);
 *      -- A rule for setting parameters equal to one another.
 *    operator std::valarray<Real> () const;
 *      -- A rule for turning a sample into a vector of the parameters.
 *      Essentially just populating a valarray with the appropriate values.
 *    friend std::ostream & operator<< (std::ostream &out, const Sample &sample)
 *      -- A rule for what to output to a line of an external file in a way that's understood as a complete sample.
 */
class Sample
{
public:
  Real kf, kb, k1, k2, k3;
  unsigned int cutoff;
  const unsigned int dim = 6;

  // Constructors
  Sample();
  Sample(Real kf, Real kb, Real k1, Real k2, Real k3, unsigned int cutoff);

  // Functions that interface with the statistical calculations
  std::vector< std::vector<Real> > return_data() const;
  std::vector<Real> return_times() const;
  Model::Model<Real, Matrix> return_model() const;
  StateVector return_initial_condition() const;
  MEPBM::Parameters<Real> return_histogram_parameters() const;
  bool within_bounds() const;

  // Sample assignment
  Sample& operator = (const Sample &sample);

  // Conversion to std::valarray<Real> for arithmetic purposes
  explicit operator std::valarray<Real> () const;

  // What it means to output a Sample
  friend std::ostream & operator<< (std::ostream &out, const Sample &sample);

private:
  ConstantData const_parameters;
};



// Constructor defines what the unknown parameters are in the model for a given Sample.
Sample::Sample(Real kf, Real kb, Real k1, Real k2, Real k3, unsigned int cutoff)
    : kf(kf), kb(kb), k1(k1), k2(k2), k3(k3), cutoff(cutoff)
{}



// By default, make an invalid Sample. This is to ensure that the necessary values are always given
// to a Sample or otherwise the program won't run.
Sample::Sample()
    : Sample(std::numeric_limits<Real>::signaling_NaN(),
             std::numeric_limits<Real>::signaling_NaN(),
             std::numeric_limits<Real>::signaling_NaN(),
             std::numeric_limits<Real>::signaling_NaN(),
             std::numeric_limits<Real>::signaling_NaN(),
             static_cast<unsigned int>(-1))
{}



// Provides access to the measured data used in likelihood calculations
std::vector<std::vector<Real>> Sample::return_data() const
{
  return const_parameters.data_size;
}



// Provides access to the times the data was collected to be given to the ODE solver
std::vector<Real> Sample::return_times() const
{
  return const_parameters.times;
}



// Forms the model representing the mechanism being used, in this case a 3-step mechanism
Model::Model<Real, Matrix> Sample::return_model() const
{
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> nucleation =
      std::make_shared<Model::TermolecularNucleation<Real, Matrix>>(const_parameters.A_index, const_parameters.As_index,
                                                      const_parameters.ligand_index, const_parameters.min_size,
                                                      kf, kb, k1, const_parameters.solvent);

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> small_growth =
      std::make_shared<Model::Growth<Real, Matrix>>(const_parameters.A_index, const_parameters.min_size, cutoff,
                                      const_parameters.max_size, const_parameters.ligand_index,
                                      const_parameters.conserved_size, k2, const_parameters.min_size);

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> large_growth =
      std::make_shared<Model::Growth<Real, Matrix>>(const_parameters.A_index, cutoff+1, const_parameters.max_size,
                                      const_parameters.max_size, const_parameters.ligand_index,
                                      const_parameters.conserved_size, k3, cutoff+1);

  Model::Model<Real, Matrix> model(const_parameters.min_size, const_parameters.max_size);
  model.add_rhs_contribution(nucleation);
  model.add_rhs_contribution(small_growth);
  model.add_rhs_contribution(large_growth);

  return model;
}



// Provides access to the initial conditions for the ODE solver to use
StateVector Sample::return_initial_condition() const
{
  return const_parameters.initial_condition;
}



// Provides access to the parameters to be used to bin data/simulation for the likelihood calculation
MEPBM::Parameters<Real> Sample::return_histogram_parameters() const
{
  MEPBM::Parameters<Real> hist_parameters(const_parameters.hist_bins, const_parameters.min_bin_size,
                                         const_parameters.max_bin_size);
  return hist_parameters;
}



// A test to see if a perturbed sample holds reasonable values
bool Sample::within_bounds() const
{
  if (   kf < const_parameters.lower_bounds[0] || kf > const_parameters.upper_bounds[0]
         || kb < const_parameters.lower_bounds[1] || kb > const_parameters.upper_bounds[1]
         || k1 < const_parameters.lower_bounds[2] || k1 > const_parameters.upper_bounds[2]
         || k2 < const_parameters.lower_bounds[3] || k2 > const_parameters.upper_bounds[3]
         || k3 < const_parameters.lower_bounds[4] || k3 > const_parameters.upper_bounds[4]
         || cutoff < const_parameters.lower_bound_cutoff || cutoff > const_parameters.upper_bound_cutoff)
    return false;
  else
    return true;
}


// A function to perturb a sample. This generates a random sample following a normal distribution centered
// around the current sample and with the specified covariance C. I.e. new_sample ~N(sample, C). The proposal
// ratio is also returned, which is 1 in this case since the normal distribution is symmetric.
std::pair<Sample,Real> perturb(const Sample &sample,
                                 const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &C,
                                 std::mt19937 &rng)
{
  // Create a vector of random numbers following a normal distribution with mean 0 and variance 1
  Eigen::Matrix<Real, Eigen::Dynamic, 1> random_vector(sample.dim);
  for (unsigned int i=0; i < random_vector.size(); ++i)
  {
    random_vector(i) = std::normal_distribution<Real>(0,1)(rng);
  }

  // Using the covariance matrix, perform the affine transformation
  // new_prm = 2.4/sqrt(dim) * L * random_vector + old_prm
  // where LL^T = covariance matrix
  const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> L = C.llt().matrixL();
  Eigen::Matrix<Real, Eigen::Dynamic, 1> old_prm(sample.dim);
  old_prm << sample.kf, sample.kb, sample.k1, sample.k2, sample.k3, sample.cutoff;
  const auto new_prm = 2.4/std::sqrt(1.*sample.dim) * L * random_vector + old_prm;

  Sample new_sample(new_prm(0), new_prm(1), new_prm(2), new_prm(3),
                    new_prm(4), static_cast<unsigned int>(new_prm(5)));

  //std::cout << "New sample: " << new_sample << "\n";
  return {new_sample, 1.};
}



std::pair<Sample,Real> perturb_unif(const Sample &sample,
                                    std::mt19937 &rng)
{
  Eigen::Matrix<Real, Eigen::Dynamic, 1> random_vector(sample.dim);
  for (unsigned int i=0; i < random_vector.size(); ++i)
    {
      random_vector(i) = std::uniform_real_distribution<Real>(-1,1)(rng);
    }
  Eigen::Matrix<Real, Eigen::Dynamic,1> bounds(sample.dim);
  Eigen::Matrix<Real, Eigen::Dynamic,1> new_prm(sample.dim);
  bounds << 0.0025, 7.5e2, 1.e4, 1.e4, 7.5e2, 10;
  Eigen::Matrix<Real, Eigen::Dynamic, 1> old_prm(sample.dim);
  old_prm << sample.kf, sample.kb, sample.k1, sample.k2, sample.k3, sample.cutoff;
  for (unsigned int i=1; i < random_vector.size(); ++i)
    {
      new_prm(i) = random_vector(i)*bounds(i) + old_prm(i);
    }

  Sample new_sample(new_prm(1)*5.e-7, new_prm(1), new_prm(2), new_prm(3),
                    new_prm(4), static_cast<unsigned int>(new_prm(5)));

  return {new_sample, 1.};
}



// Sample assignment can be done by specifying parameter values
Sample& Sample::operator=(const Sample &sample)
{
  kf = sample.kf;
  kb = sample.kb;
  k1 = sample.k1;
  k2 = sample.k2;
  k3 = sample.k3;
  cutoff = sample.cutoff;

  return *this;
}



// When arithmetic is required, we can use a valarray containing the parameters.
Sample::operator std::valarray<Real>() const
{
  return { kf, kb, k1, k2, k3, static_cast<Real>(cutoff)};
}



// When printing to the terminal or to a file, we want comma delimited data where columns
// refer to parameters and rows refer to samples
std::ostream &operator<<(std::ostream &out, const Sample &sample)
{
  out << sample.kf << ", "
      << sample.kb << ", "
      << sample.k1 << ", "
      << sample.k2 << ", "
      << sample.k3 << ", "
      << sample.cutoff ;
  return out;
}


int main(int argc, char **argv)
{
  unsigned int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

#pragma omp parallel for
  for (unsigned int i=0;i<n_threads;++i) {
    /*
     * A previous set of samples was constructed using a slightly incorrect likelihood function.
     * These results provide a good starting guess at what the covariance matrix is, or at least
     * for what the covariance matrix would be without kf since that was set as a constant previously.
     * There is a partial run for the 4-step with kf, so we can take the variance of kf from those samples
     * and use that to fill in the gap in the covariance matrix. Hence we have
     * cov = | var_kb   0        |
     *       | 0        cov_prev |
     *
     * The values are simply hardcoded for convenience.
     */
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> initial_covariance(6, 6);
    initial_covariance <<
                       1.5e-4, 0, 0, 0, 0, 0,
        0, 1.7e8, 7.1e8, -1.9e8, 5.8e6, -6.6e4,
        0, 7.1e8, 6.3e9, 6.8e8, 3.1e8, -2.1e5,
        0, -1.9e8, 6.8e8, 1.7e9, 1.4e8, -4.3e5,
        0, 5.8e6, 3.1e8, 1.4e8, 3.2e7, 1.1e4,
        0, -6.6e4, -2.1e5, -4.3e5, 1.1e4, 2.0e3;

    // Create sample with initial values for parameters
    Sample starting_guess(8.6e3*5e-7, 8.6e3, 2.7e5, 3.0e5, 1.8e4, 97);

    // Create an output file to store the accepted samples
    std::ofstream samples("samples"
                          +
                          (argc > 1 ?
                           std::string(".") + std::to_string(atoi(argv[1]) + i) :
                           std::string(".") + std::to_string(i))
                          +
                          ".txt");

    // Create the object to conduct the metropolis hastings (MH) algorithm
    SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

    // Tell the MH algorithm where to output samples
    SampleFlow::Consumers::StreamOutput<Sample> stream_output(samples);
    stream_output.connect_to_producer(mh_sampler);

    // Since our sample object is complicated, we want to be able to turn it into a vector
    // compatible with computations whenever that is necessary
    SampleFlow::Filters::Conversion<Sample, VectorType> convert_to_vector;
    convert_to_vector.connect_to_producer(mh_sampler);

    // Update the mean value of the samples during the process
    // Updating the mean value requires computations on the samples, so we need the filter created above
    SampleFlow::Consumers::MeanValue<VectorType> mean_value;
    mean_value.connect_to_producer(convert_to_vector);

    // In order to use an adaptive proposal distribution, we need to keep track of the covariance matrix
    // corresponding to the samples generated at each step
    // Updating the covariance matrix requires computations on the samples, so we need the filter created above
    SampleFlow::Consumers::CovarianceMatrix<VectorType> covariance_matrix;
    covariance_matrix.connect_to_producer(convert_to_vector);

    // We want the covariance matrix to mimic the posterior distribution before we start using it so
    // we include a counter to allow to change the perturb function once we are comfortable using the
    // covariance matrix
    SampleFlow::Consumers::CountSamples<Sample> counter;
    counter.connect_to_producer(mh_sampler);

    // Keep track of the acceptance ratio to see how efficient our sampling process was
    SampleFlow::Consumers::AcceptanceRatio<VectorType> acceptance_ratio;
    acceptance_ratio.connect_to_producer(convert_to_vector);

    // Write to disk only on occasion to reduce load on memory
    SampleFlow::Filters::TakeEveryNth<Sample> every_100th(100);
    every_100th.connect_to_producer(mh_sampler);

    SampleFlow::Consumers::Action<Sample>
        flush_after_every_100th([&samples](const Sample &, const SampleFlow::AuxiliaryData &) {
      samples << std::flush;
    });

    flush_after_every_100th.connect_to_producer(every_100th);

    // Sample from the given distribution.
    //
    // If an argument was given on the command line,
    // use that string to create a hash value and use that has value as
    // seed for the sampler.
    const std::uint_fast32_t random_seed
        = (argc > 1 ?
           std::hash<std::string>()(std::to_string(atoi(argv[1]) + i)) :
           std::hash<std::string>()(std::to_string(i)));
    const unsigned int n_samples = 20000;

    std::mt19937 rng;
    rng.seed(random_seed);
    mh_sampler.sample(starting_guess,
                      &Statistics::log_probability<Sample, 4, Real>,
                      [&](const Sample &s) {
                        if (counter.get() < 50000)
                          return perturb_unif(s, rng);
                        else
                          return perturb(s, covariance_matrix.get(), rng);
                      },
                      n_samples,
                      random_seed);

    // Output the statistics we have computed in the process of sampling
    // everything
    std::cout << "Mean value of all samples:\n";
    for (auto x : mean_value.get())
      std::cout << x << ' ';
    std::cout << std::endl;
    std::cout << "MH acceptance ratio: "
              << acceptance_ratio.get()
              << std::endl;
  }
}
