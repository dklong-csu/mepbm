#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include <functional>
#include <vector>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/conversion.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/histogram.h>
#include <sampleflow/consumers/acceptance_ratio.h>


// The data type that describes the samples we want to draw from some
// probability distribution.
using SampleType = Models::ThreeStepAlternative::Parameters;

// A data type we will use to convert samples to whenever we want to
// do arithmetic with them, such as if we want to compute mean values,
// covariances, etc.
using VectorType = std::valarray<double>;



// A function that given a SampleType object (i.e., a set of model
// parameters) computes the corresponding likelihood
double log_likelihood (const SampleType &prm)
{
  // data used for likelihood
  const Data::PomData raw_data;
  const std::vector<std::valarray<double>> data_atoms = { std::pow(raw_data.tem_diam_time2/0.3000805, 3) };

  // create model
  Models::ThreeStepAlternative model;

  // set up initial conditions and times for the ODE solution
  std::valarray<double> initialCondition(0.0,prm.n_variables);
  initialCondition[0] = 0.0012;
  const std::vector<double> times = {0, raw_data.tem_time2};


  // define parameters for the histogram
  const Histograms::Parameters histogramParameters(25, model.getSmallestParticleSize(prm), model.getLargestParticleSize(prm));

  // calculate log likelihood and return it
  return Statistics::log_likelihood(data_atoms, times, model, prm, initialCondition, histogramParameters);

}



// A function that given three SampleType objects (i.e., one for maximum allowable parameters,
// one for minimum allowable parameters, and one for the current set of parameters) computes
// the corresponding prior.
double log_prior (const SampleType &prm)
{
  // FIXME: It would be nice if this could automatically detect the member variables
  // we would like to optimize
  if (
      prm.k_backward < 1000. || prm.k_backward > 2000000.
      || prm.k1 < 4800. || prm.k1 > 8e+07
      || prm.k2 < 10. || prm.k2 > 850000.
      || prm.k3 < 10. || prm.k3 > 250000.
      || prm.particle_size_cutoff < 10 || prm.particle_size_cutoff > 2000
  )
    return -std::numeric_limits<double>::max();
  else
    return 0.;
}



double log_probability (const SampleType &prm)
{
  const double logPrior = log_prior(prm);

  // If the prior probability for a sample is zero, multiplication
  // with the (expensive to compute) likelihood isn't going to change
  // that. So just return what we already got.
  if (logPrior == -std::numeric_limits<double>::max())
    return logPrior;
  
  else
  // Return the probability=likelihood*prior, except of course we're
  // only dealing with logarithms.
    return log_likelihood(prm) + logPrior;
}



// A function given two real numbers returns a random number
// between those numbers based on a uniform distribution
double rand_btwn_double (const double small_num, const double big_num)
{
  static std::mt19937 gen;
  std::uniform_real_distribution<> unif(small_num,big_num);

  return unif(gen);
}



// A function given two integers returns a random number
// between those numbers based on a uniform distribution
int rand_btwn_int (const int small_num, const int big_num)
{
  static std::mt19937 gen;
  std::uniform_int_distribution<> unif(small_num,big_num);

  return unif(gen);
}



// A function given a SampleType object returns a SampleType
// object whose member variables have been randomly perturbed
// along with the ratio of the probabilities of prm->new_prm /
// new_prm->prm
std::pair<SampleType,double> perturb (const SampleType &prm)
{
  // perturb each non-constant parameter with a uniform distribution
  double new_k_backward = prm.k_backward + rand_btwn_double(-1.e4, 1.e4);
  double new_k1 = prm.k1 + rand_btwn_double(-5.e4, 5.e4);
  double new_k2 = prm.k2 + rand_btwn_double(-5.e3, 5.e3);
  double new_k3 = prm.k3 + rand_btwn_double(-4.e3, 4.e3);
  double new_part_sz_cutoff = prm.particle_size_cutoff + rand_btwn_int(-30, 30);

  SampleType new_prm(prm.k_forward,
                     new_k_backward,
                     new_k1,
                     new_k2,
                     new_k3,
                     prm.solvent,
                     prm.w,
                     prm.maxsize,
                     new_part_sz_cutoff);

  return {new_prm, 1.};
}



int main(int argc, char **argv)
{
  const SampleType
    starting_guess (3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);

  std::ofstream samples ("samples"
                         +
                         (argc > 1 ?
                          std::string(".") + argv[1] :
                          std::string(""))
                         +
                         ".txt");

  SampleFlow::Producers::MetropolisHastings<SampleType> mh_sampler;
  
  SampleFlow::Consumers::StreamOutput<SampleType> stream_output (samples);
  stream_output.connect_to_producer (mh_sampler);

  SampleFlow::Filters::Conversion<SampleType,VectorType> convert_to_vector;
  convert_to_vector.connect_to_producer (mh_sampler);
  
  SampleFlow::Consumers::MeanValue<VectorType> mean_value;
  mean_value.connect_to_producer (convert_to_vector);

  SampleFlow::Consumers::AcceptanceRatio<VectorType> acceptance_ratio;
  acceptance_ratio.connect_to_producer (convert_to_vector);
  

  SampleFlow::Filters::Conversion<SampleType,double>
    extract_k1 ([](const SampleType &prm) { return prm.k1; });
  extract_k1.connect_to_producer (mh_sampler);

  const double k1_min = 0;
  const double k1_max = 3e5;
  const unsigned int n_bins = 100;
  SampleFlow::Consumers::Histogram<double> histogram_k1 (k1_min, k1_max, n_bins);
  histogram_k1.connect_to_producer (extract_k1);
  
  
  // Sample from the given distribution.
  //
  // If an argument was given on the command line,
  // use that string to create a hash value and use that has value as
  // seed for the sampler.
  const std::uint_fast32_t random_seed
    = (argc > 1 ?
       std::hash<std::string>()(std::string(argv[1])) :
       std::uint_fast32_t());
  const unsigned int n_samples = 10;
  mh_sampler.sample (starting_guess,
                     &log_probability,
                     &perturb,
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

  // Now also output the histograms for all parameters:
  histogram_k1.write_gnuplot (std::ofstream("histogram_k1.txt"));
}
