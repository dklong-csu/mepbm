#ifndef STATISTICS_H_
#define STATISTICS_H_


#include <vector>
#include "models.h"
#include "histogram.h"
#include <boost/numeric/odeint.hpp>



namespace Statistics
{
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
  double log_likelihood(const std::vector<double>& data,
                        const std::vector<double>& distribution,
                        const std::vector<double>& sizes,
                        const Histograms::Parameters& hist_prm);



  // log likelihood -- including ODE solve
  // A function which given data set(s) and corresponding time(s), an ODE model, parameters for the ODE model,
  // the initial condition, and histogram parameters will compute a particle size distribution for each time
  // point there is data for, then compute the corresponding log likelihood at each time, and finally add up
  // all of the log likelihoods, to get a log likelihood of all of the data occurring.
  // As a result, this function's workflow is:
  // Step 1: Solve ODE, saving the solution at each relevant time
  // Step 2: For each time point, calculate the log likelihood, and add to the cumulative log likelihood
  double log_likelihood(const std::vector<std::vector<double>>& data,
                        const std::vector<double>& times,
                        const Model::Model& ode_model,
                        boost::numeric::ublas::vector<double>& ic,
                        const Histograms::Parameters& hist_prm);



  // log_likelihood -- interface with SampleFlow
  // A function that given an object will compute the corresponding log likelihood by solving the system of ODEs
  // and using the data specified within the object.
  //
  // The object used must have member functions
  // std::vector< std::vector<double> > return_data() -- a collection of vectors corresponding to collected data
  //                                                     which gives particle size for each data.
  // std::vector<double> return_times() -- the first element is intended to be time=0 and the remaining times
  //                                       should correspond to when the return_data() entries were collected.
  // Model::Model return_model() -- an object describing the right hand side of the system of differential equations
  // std::vector<double> return_initial_condition() -- a vector giving the concentrations of the tracked chemical
  //                                                   species at time = 0.
  // Histograms::Parameters return_histogram_parameters() -- an object describing the way you want to bin together
  //                                                         particle sizes for comparing between data and simulation.
  template<class InputClass>
  double log_likelihood(const InputClass &my_object)
  {
    std::vector<double> ic = my_object.return_initial_condition();
    return Statistics::log_likelihood(my_object.return_data(), my_object.return_times(), my_object.return_model(),
                                      ic, my_object.return_histogram_parameters());
  }



  // A function that given an object will compute the corresponding prior under the assumption that
  // the prior is a uniform distribution on some domain specified within the input object.
  //
  // The object used must have member function
  // bool within_bounds() -- The intent of this function is to compare the current state of the parameters
  //                         being sampled and compare to a pre-determined domain for those parameters.
  //                         If any parameter lies outside of its allowed interval, return false. Else, return true.
  template<class InputClass>
  double log_prior (const InputClass &my_object)
  {
    if (my_object.within_bounds())
      return 0.;
    else
      return -std::numeric_limits<double>::max();
  }



  // A function that given an object will compute: log_probability = log_prior + log_likelihood
  // log_likelihood is an expensive calculation, so first check to see if log_prior = -infinity
  // and only calculate log_likelihood if log_prior is finite.
  //
  // The object used must meet the member function requirements of log_prior<InputClass> and log_likelihood<InputClass>
  template<class InputClass>
  double log_probability (const InputClass &my_object)
  {
    const double log_prior = Statistics::log_prior<InputClass>(my_object);

    if (log_prior == -std::numeric_limits<double>::max())
      return log_prior;

    else
      return Statistics::log_likelihood<InputClass>(my_object) + log_prior;
  }



  // A function given two real numbers returns a random number
  // between those numbers based on a uniform distribution
  double rand_btwn_double (double small_num, double big_num);



  // A function given two integers returns a random number
  // between those numbers based on a uniform distribution
  int rand_btwn_int (int small_num, int big_num);



  // A function given an object returns an object of the same type whose
  // member variables have been randomly perturbed along with the ratio of
  // the probabilities of prm->new_prm / new_prm->prm.
  // This is a template function which requires the input object, InputClass,
  // to have member functions:
  //      InputClass perturb();
  //      double perturb_ratio();
  template<class InputClass>
  std::pair<InputClass,double> perturb (const InputClass &my_object)
  {
    InputClass new_object = my_object.perturb();
    double ratio = my_object.perturb_ratio();

    return {new_object, ratio};
  }
}



#endif /* STATISTICS_H_ */
