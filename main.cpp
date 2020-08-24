#include <iostream>
#include "models.h"
#include "histogram.h"
#include "statistics.h"

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/consumers/stream_output.h>


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


double log_probability (const SampleType &prm)
{
  const double log_like = log_likelihood(prm);
  const double log_prior = 0;   // FIXME

  // Return the probability=likelihood*prior, except of course we're
  // only dealing with logarithms.
  return log_like + log_prior;
}



std::pair<SampleType,double> perturb (const SampleType &prm)
{
  // FIXME: We need to perturb the given sample here. Then return the
  // new sample and the ratio of probabilities prm->new_sample /
  // new_sample->prm
  return {prm, 1.};
}



int main()
{
  const SampleType
    starting_guess (3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);

  SampleFlow::Producers::MetropolisHastings<SampleType> mh_sampler;
  SampleFlow::Consumers::StreamOutput<SampleType> stream_output (std::cout);
  stream_output.connect_to_producer (mh_sampler);
  
  // Sample from the given distribution
  mh_sampler.sample (starting_guess,
                     &log_probability,
                     &perturb,
                     10);
}
