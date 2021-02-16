/*
 * Program to compute the posterior distribution
 * -- 3-step mechanism
 * -- Ir-POM chemical system
 * -- Using data from time 0.918
 */

#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <memory>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"
#include <eigen3/Eigen/Dense>

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/conversion.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>



// A data type we will use to convert samples to whenever we want to
// do arithmetic with them, such as if we want to compute mean values,
// covariances, etc.
using VectorType = std::valarray<double>;


// A data type describing the linear algebra object vector that is used
// in the ODE solver.
using StateVector = Eigen::VectorXd;



/*
 * Create an object which holds all of the constant information for the mechanism
 */

class ConstantData
{
public:
  ConstantData();

  unsigned int min_size = 3;
  unsigned int min_reliable_size = 100;
  unsigned int max_size = 2500;
  unsigned int conserved_size = 1;

  unsigned int A_index = 0;
  unsigned int As_index = 1;
  unsigned int ligand_index = 2;

  double solvent = 11.3;

  Data::PomData data_diameter;
  std::vector< std::vector<double> > data_size;
  std::vector<double> times;

  // { kb, k1, k2, k3 }
  std::vector<double> lower_bounds = { 0., 1000., 4800., 10., 10.};
  std::vector<double> upper_bounds = { 1.e3, 2.e6, 8.e7, 8.5e5, 2.5e5};

  unsigned int lower_bound_cutoff = 10;
  unsigned int upper_bound_cutoff = 2000;

  // { kf, kb, k1, k2, k3 }
  std::vector<double> perturbation_magnitude = { 1.e-3, 1.e4, 1.e4, 5.e3, 4.e3};
  int perturbation_magnitude_cutoff = 30;

  StateVector initial_condition;

  unsigned int hist_bins = 25;
};

ConstantData::ConstantData()
{
  const std::vector<std::vector<double>> data_diam = {data_diameter.tem_diam_time1};
  for (const auto& vec : data_diam)
  {
    std::vector<double> tmp;
    for (auto diam : vec)
    {
      tmp.push_back(std::pow(diam/0.3000805, 3));
    }
    data_size.push_back(tmp);
  }

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
 *  std::vector< std::vector<double> > return_data()
 *      -- a collection of vectors corresponding to collected data which gives particle size for each data.
 *  std::vector<double> return_times()
 *      -- the first element is intended to be time=0 and the remaining times should correspond to when
 *      the return_data() entries were collected.
 *  Model::Model return_model()
 *      -- an object describing the right hand side of the system of differential equations
 *  std::vector<double> return_initial_condition()
 *      -- a vector giving the concentrations of the tracked chemical species at time = 0.
 *  Histograms::Parameters return_histogram_parameters()
 *      -- an object describing the to bin together particle sizes to compare the data and simulation results.
 *  bool within_bounds()
 *      -- The intent of this function is to compare the current state of the parameters being sampled and
 *      compare to a pre-determined domain for those parameters.
 *      If any parameter lies outside of its allowed interval, return false. Otherwise, return true.
 *  Sample perturb()
 *      -- Generates random perturbations of the parameters in the sample and returns a new object reflecting
 *      the randomly chosen new parameters.
 *  double perturb_ratio()
 *      -- Computes the ratio of probabilities parameter->new_parameters / new_parameter->parameters
 *
 *  Operators:
 *    Sample operator = (const Sample &sample);
 *      -- A rule for setting parameters equal to one another.
 *    operator std::valarray<double> () const;
 *      -- A rule for turning a sample into a vector of the parameters.
 *      Essentially just populating a valarray with the appropriate values.
 *    friend std::ostream & operator<< (std::ostream &out, const Sample &sample)
 *      -- A rule for what to output to a line of an external file in a way that's understood as a complete sample.
 */
class Sample
{
public:
  double kf, kb, k1, k2, k3;
  unsigned int cutoff;

  // Default constructor makes an invalid object
  Sample();
  Sample(double kf, double kb, double k1, double k2, double k3, unsigned int cutoff);

  std::vector< std::vector<double> > return_data() const;
  std::vector<double> return_times() const;
  Model::Model return_model() const;
  StateVector return_initial_condition() const;
  Histograms::Parameters return_histogram_parameters() const;
  bool within_bounds() const;
  Sample perturb(std::mt19937 &rng) const;
  static double perturb_ratio() ;

  Sample& operator = (const Sample &sample);
  explicit operator std::valarray<double> () const;

  friend std::ostream & operator<< (std::ostream &out, const Sample &sample);

private:
  ConstantData const_parameters;
};



Sample::Sample(double kf, double kb, double k1, double k2, double k3, unsigned int cutoff)
    : kf(kf), kb(kb), k1(k1), k2(k2), k3(k3), cutoff(cutoff)
{}



Sample::Sample()
    : Sample(std::numeric_limits<double>::signaling_NaN(),
             std::numeric_limits<double>::signaling_NaN(),
             std::numeric_limits<double>::signaling_NaN(),
             std::numeric_limits<double>::signaling_NaN(),
             std::numeric_limits<double>::signaling_NaN(),
             static_cast<unsigned int>(-1))
{}



std::vector<std::vector<double>> Sample::return_data() const
{
  return const_parameters.data_size;
}



std::vector<double> Sample::return_times() const
{
  return const_parameters.times;
}



// Here we implement a 3-Step Model
Model::Model Sample::return_model() const
{
  std::shared_ptr<Model::RightHandSideContribution> nucleation =
      std::make_shared<Model::TermolecularNucleation>(const_parameters.A_index, const_parameters.As_index,
                                                      const_parameters.ligand_index, const_parameters.min_size,
                                                      kf, kb, k1, const_parameters.solvent);

  std::shared_ptr<Model::RightHandSideContribution> small_growth =
      std::make_shared<Model::Growth>(const_parameters.A_index, const_parameters.min_size, cutoff,
                                      const_parameters.max_size, const_parameters.ligand_index,
                                      const_parameters.conserved_size, k2);

  std::shared_ptr<Model::RightHandSideContribution> large_growth =
      std::make_shared<Model::Growth>(const_parameters.A_index, cutoff+1, const_parameters.max_size,
                                      const_parameters.max_size, const_parameters.ligand_index,
                                      const_parameters.conserved_size, k3);

  Model::Model model(const_parameters.min_size, const_parameters.max_size);
  model.add_rhs_contribution(nucleation);
  model.add_rhs_contribution(small_growth);
  model.add_rhs_contribution(large_growth);

  return model;
}



StateVector Sample::return_initial_condition() const
{
  return const_parameters.initial_condition;
}



Histograms::Parameters Sample::return_histogram_parameters() const
{
  Histograms::Parameters hist_parameters(const_parameters.hist_bins, const_parameters.min_reliable_size,
                                         const_parameters.max_size);
  return hist_parameters;
}



bool Sample::within_bounds() const
{
  if (    kf < const_parameters.lower_bounds[0] || kf > const_parameters.upper_bounds[0]
          || kb < const_parameters.lower_bounds[1] || kb > const_parameters.upper_bounds[1]
          || k1 < const_parameters.lower_bounds[2] || k1 > const_parameters.upper_bounds[2]
          || k2 < const_parameters.lower_bounds[3] || k2 > const_parameters.upper_bounds[3]
          || k3 < const_parameters.lower_bounds[4] || k3 > const_parameters.upper_bounds[4]
          || cutoff < const_parameters.lower_bound_cutoff || cutoff > const_parameters.upper_bound_cutoff)
    return false;
  else
    return true;
}



Sample Sample::perturb(std::mt19937 &rng) const
{
  double new_kf = kf + std::uniform_real_distribution<>(-const_parameters.perturbation_magnitude[0],
                                                        const_parameters.perturbation_magnitude[0])(rng);
  double new_kb = kb + std::uniform_real_distribution<>(-const_parameters.perturbation_magnitude[1],
                                                        const_parameters.perturbation_magnitude[1])(rng);
  double new_k1 = k1 + std::uniform_real_distribution<>(-const_parameters.perturbation_magnitude[2],
                                                        const_parameters.perturbation_magnitude[2])(rng);
  double new_k2 = k2 + std::uniform_real_distribution<>(-const_parameters.perturbation_magnitude[3],
                                                        const_parameters.perturbation_magnitude[3])(rng);
  double new_k3 = k3 + std::uniform_real_distribution<>(-const_parameters.perturbation_magnitude[4],
                                                        const_parameters.perturbation_magnitude[4])(rng);
  unsigned int new_cutoff = cutoff + std::uniform_int_distribution<>(-const_parameters.perturbation_magnitude_cutoff,
                                                                     const_parameters.perturbation_magnitude_cutoff)(rng);

  Sample new_sample(new_kf, new_kb, new_k1, new_k2, new_k3, new_cutoff);
  std::cout << "New sample: " << new_sample << std::endl;
  return new_sample;
}



double Sample::perturb_ratio()
{
  return 1.;
}



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



Sample::operator std::valarray<double>() const
{
  return { kf, kb, k1, k2, k3, static_cast<double>(cutoff)};
}

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

  // Create sample with initial values for parameters
  Sample starting_guess(3.6e-2,7.27e4, 6.4e4, 1.61e4, 5.45e3, 265);


  std::ofstream samples ("samples"
                         +
                         (argc > 1 ?
                          std::string(".") + argv[1] :
                          std::string(""))
                         +
                         ".txt");

  SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

  SampleFlow::Consumers::StreamOutput<Sample> stream_output (samples);
  stream_output.connect_to_producer (mh_sampler);

  SampleFlow::Filters::Conversion<Sample,VectorType> convert_to_vector;
  convert_to_vector.connect_to_producer (mh_sampler);

  SampleFlow::Consumers::MeanValue<VectorType> mean_value;
  mean_value.connect_to_producer (convert_to_vector);

  SampleFlow::Consumers::AcceptanceRatio<VectorType> acceptance_ratio;
  acceptance_ratio.connect_to_producer (convert_to_vector);

  SampleFlow::Filters::TakeEveryNth<Sample> every_100th(100);
  every_100th.connect_to_producer (mh_sampler);

  SampleFlow::Consumers::Action<Sample>
      flush_after_every_100th ([&samples](const Sample &, const SampleFlow::AuxiliaryData &)
                               {
                                 samples << std::flush;
                               });

  flush_after_every_100th.connect_to_producer (every_100th);

  // Sample from the given distribution.
  //
  // If an argument was given on the command line,
  // use that string to create a hash value and use that has value as
  // seed for the sampler.
  const std::uint_fast32_t random_seed
      = (argc > 1 ?
         std::hash<std::string>()(std::string(argv[1])) :
         std::uint_fast32_t());
  const unsigned int n_samples = 5;

  std::mt19937 rng;
  rng.seed(random_seed);
  mh_sampler.sample (starting_guess,
                     &Statistics::log_probability<Sample,4>,
                     [&rng](const Sample &s)
                     {
                       return Statistics::perturb<Sample> (s, rng);
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