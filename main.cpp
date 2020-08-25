#include <iostream>
#include <fstream>
#include <limits>
#include <random>
#include "models.h"
#include "histogram.h"
#include "statistics.h"

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>


// The data type that describes the samples we want to draw from some
// probability distribution.
using SampleType = Models::ThreeStepAlternative::Parameters;


// A function that given a SampleType object (i.e., a set of model
// parameters) computes the corresponding likelihood
double log_likelihood (const SampleType &prm)
{
  // S3 data
  const std::valarray<double> dataDiam = {2.98, 2.82, 1.84, 2.04, 1.56, 1.56,
                                          1.2, 1.18, 2.06, 2.27, 2.54, 2.09,
                                          1.63, 1.91, 1.96, 2.09, 2.39, 2.17,
                                          1.98, 1.69, 2.47, 1.87, 2.03, 1.5,
                                          2.73, 1.65, 2.05, 2.21, 2.38, 3.07,
                                          2.93, 2.67, 3.83, 2.95, 3.3, 2.82,
                                          2.49, 2.62, 2.09, 3.02, 2.94, 3.25,
                                          2.43, 1.92, 3.22, 2.86, 2.74, 3.09,
                                          3.19, 1.73, 1.94, 2.14, 2.91, 2.85,
                                          2.8, 2.37, 2.42, 2.68, 2.01, 1.9,
                                          2.14};


  const std::valarray<double> dataAtoms = std::pow(dataDiam/0.3000805, 3);


  // create dummy model
  Models::ThreeStepAlternative model;

  // set up dummy initial conditions, start time, and end time
  std::valarray<double> initialCondition(0.0,2500);
  initialCondition[0] = 0.0012;
  const double startTime = 0.0;
  const double endTime = 1.170;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, endTime, initialCondition);

  // define parameters for the histogram
  const Histograms::Parameters histogramParameters(25, 3.0, 2500.0);

  // calculate log likelihood and return it
  return Statistics::logLikelihood(dataAtoms, model, prm, solverParameters, histogramParameters);
}



// A function that given three SampleType objects (i.e., one for maximum allowable parameters,
// one for minimum allowable parameters, and one for the current set of parameters) computes
// the corresponding prior.
double log_prior (const SampleType &prm)
{
  // FIXME: It would be nice if this could automatically detect the member variables
  // we would like to optimize
  if (
      prm.k_backward < 1000. || prm.k_backward > 200000.
      || prm.k1 < 4800. || prm.k1 > 8e+07
      || prm.k2 < 10. || prm.k2 > 85000.
      || prm.k3 < 10. || prm.k3 > 25000.
      || prm.particle_size_cutoff < 10 || prm.particle_size_cutoff > 800
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
  double new_k_backward = prm.k_backward + rand_btwn_double(-1.e3, 1.e3);
  double new_k1 = prm.k1 + rand_btwn_double(-1.e4, 1.e4);
  double new_k2 = prm.k2 + rand_btwn_double(-1.e3, 1.e3);
  double new_k3 = prm.k3 + rand_btwn_double(-1.e3, 1.e3);
  double new_part_sz_cutoff = prm.particle_size_cutoff + rand_btwn_int(-10, 10);

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



int main()
{
  const SampleType
    starting_guess (3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);

  std::ofstream samples ("samples.txt");
  
  SampleFlow::Producers::MetropolisHastings<SampleType> mh_sampler;
  SampleFlow::Consumers::StreamOutput<SampleType> stream_output (samples);
  stream_output.connect_to_producer (mh_sampler);

  SampleFlow::Consumers::MeanValue<SampleType> mean_value;
  mean_value.connect_to_producer (mh_sampler);
  
  // Sample from the given distribution
  const unsigned int n_samples = 10;
  mh_sampler.sample (starting_guess,
                     &log_probability,
                     &perturb,
                     n_samples);

  // Output the statistics we have computed in the process of sampling
  // everything
  std::cout << "Mean value of all samples:\n"
            << mean_value.get()
            << std::endl;
}
