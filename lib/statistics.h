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
  /// A function that computes the log likelihood based on a data set, a size distribution, and a histogram describing the size binning.
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



  /// A function that computes the log likelihood after solving the ODEs to compute the particle size distribution
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



  /// A function for computing the log likelihood, but interfaces with the object used in the SampleFlow algorithms
  template<class InputClass, int order, typename Real>
  Real log_likelihood(const InputClass &my_object)
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> ic = my_object.return_initial_condition();
    return Statistics::log_likelihood<order, Real>(my_object.return_data(), my_object.return_times(), my_object.return_model(),
                                                   ic, my_object.return_histogram_parameters());
  }



  /// A function for computing the log prior of a set of parameters
  template<class InputClass, typename Real>
  Real log_prior (const InputClass &my_object)
  {
    if (my_object.within_bounds())
      return 0.;
    else
      return -std::numeric_limits<Real>::max();
  }



  /// A function for computing the log probability of a set of parameters, that is, the sum of the log prior and log likelihood
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
