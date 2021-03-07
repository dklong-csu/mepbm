#ifndef STATISTICS_H_
#define STATISTICS_H_


#include <vector>
#include <iostream>
#include <random>
#include "models.h"
#include "histogram.h"
#include "ode_solver.h"
#include <eigen3/Eigen/Dense>



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
  template<typename Real>
  Real log_likelihood(const std::vector<Real>& data,
                      const std::vector<Real>& distribution,
                      const std::vector<Real>& sizes,
                      const Histograms::Parameters<Real>& hist_prm);



  template<typename Real>
  Real log_likelihood(const std::vector<Real>& data,
                      const std::vector<Real>& distribution,
                      const std::vector<Real>& sizes,
                      const Histograms::Parameters<Real>& hist_prm)
  {
    // Step 1 -- Turn data into a histogram
    Histograms::Histogram<Real> hist_data(hist_prm);
    std::vector<Real> data_counts(data.size(), 1.0); // each data point occurred 1 time
    hist_data.AddToBins(data_counts, data);

    // Step 2 -- Turn distribution into a histogram
    // Step 2a -- Normalize distribution to create probability mass function (pmf)

    // We know our model and ODE solver are imperfect. Therefore, if we encounter a negative
    // value then we know this is a mathematical limitation. Fortunately, we can use physical
    // intuition to understand that zero is a more accurate value. We can extend that a step
    // further and see that instead of zero, a very small number relative to the rest of the
    // concentrations is perhaps more accurate and plays nicely with the logarithms used later.
    // To this end, we first calculate the maximum value in the distribution vector, with the
    // intention of replacing negative values with 1e-9 * max value. Then we calculate the norm
    // by summing all elements in distribution, replacing negative values as necessary. Finally,
    // the vector is normalized by the norm value, again replacing negative values as necessary.
    double max_conc = 0.;
    for (double concentration : distribution)
    {
      max_conc = std::max(max_conc, concentration);
    }

    double norm = 0.;
    for (double concentration : distribution)
    {
      if (concentration > 0)
      {
        norm += concentration;
      }
      else
      {
        norm += max_conc * 1e-9;
      }
    }

    std::vector<Real> pmf(distribution.size());
    for (unsigned int i=0; i<pmf.size(); ++i)
    {
      if (distribution[i] < 0)
      {
        pmf[i] = 1e-9 * max_conc / norm;
      }
      else
      {
        pmf[i] = distribution[i] / norm;
      }
    }

    // Step 2b -- Create histogram from pmf
    Histograms::Histogram<Real> hist_ode(hist_prm);
    hist_ode.AddToBins(pmf, sizes);

    // Step 3 -- Combine histograms to calculate log likelihood
    double likelihood = 0.0;
    for (unsigned int bin=0; bin < hist_prm.n_bins; ++bin)
    {
      // FIXME:  I think I can change the first if to just check if there are no data points
      // If the probability is zero and there are no data points, do not contribute anything
      // If the probability is zero and there are data points, then return most negative number possible
      // If the probability is non-zero, then calculate normally
      if (hist_ode.count[bin] <= 0 && hist_data.count[bin] == 0)
      {
        // do nothing
      }
      else if (hist_ode.count[bin] <= 0)
      {
        // minimum possible value
        return -std::numeric_limits<double>::max();
      }
      else
      {
        // normal calculation
        likelihood += hist_data.count[bin]*std::log(hist_ode.count[bin]);
      }
    }

    return likelihood;
  }



  // log likelihood -- including ODE solve
  // A function which given data set(s) and corresponding time(s), an ODE model, parameters for the ODE model,
  // the initial condition, and histogram parameters will compute a particle size distribution for each time
  // point there is data for, then compute the corresponding log likelihood at each time, and finally add up
  // all of the log likelihoods, to get a log likelihood of all of the data occurring.
  // As a result, this function's workflow is:
  // Step 1: Solve ODE, saving the solution at each relevant time
  // Step 2: For each time point, calculate the log likelihood, and add to the cumulative log likelihood
  template<int order, typename Real, typename Matrix>
  Real log_likelihood(const std::vector<std::vector<Real>>& data,
                      const std::vector<Real>& times,
                      const Model::Model<Real, Matrix>& ode_model,
                      Eigen::Matrix<Real, Eigen::Dynamic, 1>& ic,
                      const Histograms::Parameters<Real>& hist_prm)
  {
    // Step 1 -- Solve the ODE at each time
    ODE::StepperBDF<order, Real, Matrix> stepper(ode_model);
    std::vector< Eigen::Matrix<Real, Eigen::Dynamic, 1> > solutions;
    solutions.push_back(ic);
    for (unsigned int i = 1; i < times.size(); ++i)
    {
      // FIXME: add parameters for what dt should be and for accuracy of newton method
      auto solution = ODE::solve_ode<Real>(stepper, solutions[i-1], times[i-1], times[i], 5e-3);
      solutions.push_back(solution);
    }

    // Step 2 -- Accumulate log likelihood
    Real likelihood = 0.0;
    for (unsigned int set_num=0; set_num < data.size(); ++set_num)
    {
      // Step 2a -- Extract the particle sizes from the ODE solution
      const unsigned int smallest = ode_model.nucleation_order;
      const unsigned int largest = ode_model.max_size;

      std::vector<Real> sizes(largest - smallest + 1);
      std::vector<Real> concentration(sizes.size());

      for (unsigned int size = smallest; size < largest+1; ++size)
      {
        sizes[size - smallest] = 0.3000805 * std::pow(1.*size, 1./3);
        concentration[size - smallest] = solutions[set_num+1](size); // FIXME: particle size to index function needed
      }
      // Step 2b -- Calculate log likelihood of current data set and add to total
      likelihood += Statistics::log_likelihood<Real>(data[set_num],
                                                     concentration,
                                                     sizes,
                                                     hist_prm);
    }

    return likelihood;
  }



  // log_likelihood -- interface with SampleFlow
  // A function that given an object will compute the corresponding log likelihood by solving the system of ODEs
  // and using the data specified within the object.
  //
  // The object used must have member functions
  // std::vector< std::vector<Real> > return_data() -- a collection of vectors corresponding to collected data
  //                                                     which gives particle size for each data.
  // std::vector<Real> return_times() -- the first element is intended to be time=0 and the remaining times
  //                                       should correspond to when the return_data() entries were collected.
  // Model::Model return_model() -- an object describing the right hand side of the system of differential equations
  // std::vector<Real> return_initial_condition() -- a vector giving the concentrations of the tracked chemical
  //                                                   species at time = 0.
  // Histograms::Parameters return_histogram_parameters() -- an object describing the way you want to bin together
  //                                                         particle sizes for comparing between data and simulation.
  template<class InputClass, int order, typename Real>
  Real log_likelihood(const InputClass &my_object)
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> ic = my_object.return_initial_condition();
    return Statistics::log_likelihood<order, Real>(my_object.return_data(), my_object.return_times(), my_object.return_model(),
                                                   ic, my_object.return_histogram_parameters());
  }



  // A function that given an object will compute the corresponding prior under the assumption that
  // the prior is a uniform distribution on some domain specified within the input object.
  //
  // The object used must have member function
  // bool within_bounds() -- The intent of this function is to compare the current state of the parameters
  //                         being sampled and compare to a pre-determined domain for those parameters.
  //                         If any parameter lies outside of its allowed interval, return false. Else, return true.
  template<class InputClass, typename Real>
  Real log_prior (const InputClass &my_object)
  {
    if (my_object.within_bounds())
      return 0.;
    else
      return -std::numeric_limits<Real>::max();
  }



  // A function that given an object will compute: log_probability = log_prior + log_likelihood
  // log_likelihood is an expensive calculation, so first check to see if log_prior = -infinity
  // and only calculate log_likelihood if log_prior is finite.
  //
  // The object used must meet the member function requirements of log_prior<InputClass> and log_likelihood<InputClass>
  template<class InputClass, int order, typename Real>
  Real log_probability (const InputClass &my_object)
  {
    const Real log_prior = Statistics::log_prior<InputClass, Real>(my_object);

    if (log_prior == -std::numeric_limits<Real>::max())
      return log_prior;

    else
      return Statistics::log_likelihood<InputClass, order, Real>(my_object) + log_prior;
  }
}



#endif /* STATISTICS_H_ */
