#ifndef STATISTICS_H_
#define STATISTICS_H_


#include <valarray>
#include "models.h"
#include "histogram.h"



namespace Statistics
{
  // log likelihood -- one data set
  // This function solves the ODE based on odeModel, odeParameters, and solverParameters
  // Then the ODE solution and data are converted to histograms based on histogramParameters.
  // Finally the log likelihood is computed using the multinomial distribution and the two histograms.
  double logLikelihood(const std::valarray<double>& particleSizeData,
                       const Models::ModelsBase& odeModel,
                       const Models::ParametersBase& odeParameters,
                       const Models::explEulerParameters& solverParameters,
                       const Histograms::Parameters& histogramParameters);



  // log likelihood -- no ODE solve
  // A function which given a data set, a particle size distribution, and histogram parameters,
  // will compute the log likelihood that the data set occurred assuming the particle size
  // distribution provided is accurate.
  // The calculation of the likelihood is based on the multinomial distribution. The multinomial distribution says:
  // probability(data counts) = n! / ( product( b_i! ) ) * product( p_i ^ b_i )
  // where: b_i represents the number of data points within bin i
  // n = sum( b_i ) -- i.e. the total number of data points
  // p_i represents the probability of a sample being in bin i -- this is provided by the distribution
  // and ! is the factorial operator
  // For the purposes of a Markov Chain Monte Carlo simulation, the data is fixed and we only care about the ratio
  // of likelihood. Therefore, n! / ( product( b_i! ) ) does not matter to us, so we remove that from our calculation.
  // After taking the log we are left with:
  // log probability( data counts ) ~= sum( b_i * log( p_i ) )
  // This function's workflow falls out naturally to be:
  // Step 1: Turn the data into a histogram to form b_i's
  // Step 2: Turn the distribution into a histogram to form p_i's
  // Step 3: Compute the log likelihood.
  double log_likelihood(const std::valarray<double>& data,
                        const std::valarray<double>& distribution,
                        const std::valarray<double>& sizes,
                        const Histograms::Parameters& hist_prm);



  // log likelihood -- including ODE solve
  // A function which given data set(s) and corresponding time(s), an ODE model, parameters for the ODE model,
  // the initial condition, and histogram parameters will compute a particle size distribution for each time
  // point there is data for, then compute the corresponding log likelihood at each time, and finally add up
  // all of the log likelihoods, to get a log likelihood of all of the data occurring.
  // As a result, this function's workflow is:
  // Step 1: Solve ODE, saving the solution at each relevant time
  // Step 2: For each time point, calculate the log likelihood, and add to the cumulative log likelihood
  double log_likelihood(const std::vector<std::valarray<double>>& data,
                        const std::vector<double>& times,
                        const Models::ModelsBase& ode_model,
                        const Models::ParametersBase& ode_prm,
                        const std::valarray<double>& ic,
                        const Histograms::Parameters& hist_prm);
}



#endif /* STATISTICS_H_ */
