#include <valarray>
#include <cmath>
#include <vector>
#include <random>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "ode_solver.h"
#include <eigen3/Eigen/Dense>

#include <iostream>



using VectorType = std::vector<double>;
using StateType = Eigen::VectorXd;


// log likelihood detailed calculation
double Statistics::log_likelihood(const VectorType& data,
                                  const VectorType& distribution,
                                  const VectorType& sizes,
                                  const Histograms::Parameters& hist_prm)
{
  // Step 1 -- Turn data into a histogram
  Histograms::Histogram hist_data(hist_prm);
  VectorType data_counts(data.size(), 1.0); // each data point occurred 1 time
  hist_data.AddToBins(data_counts, data);

  // Step 2 -- Turn distribution into a histogram
    // Step 2a -- Normalize distribution to create probability mass function (pmf)
    double norm = 0.;
    for (double concentration : distribution)
    {
      norm += concentration;
    }

    VectorType pmf(distribution.size());
    for (unsigned int i=0; i<pmf.size(); ++i)
    {
      pmf[i] = distribution[i] / norm;
    }

    // Step 2b -- Create histogram from pmf
    Histograms::Histogram hist_ode(hist_prm);
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



// log likelihood ODE solve integration
double Statistics::log_likelihood(const std::vector<VectorType>& data,
                                  const std::vector<double>& times,
                                  const Model::Model& ode_model,
                                  StateType& ic,
                                  const Histograms::Parameters& hist_prm)
{
  // Step 1 -- Solve the ODE at each time
  ODE::StepperSDIRK<2> stepper(ode_model); // FIXME: make the template variable
  std::vector<StateType> solutions;
  solutions.push_back(ic);
  for (unsigned int i = 1; i < times.size(); ++i)
  {
    // FIXME: add parameters for what dt should be and for accuracy of newton method
    auto solution = ODE::solve_ode(stepper, solutions[i-1], times[i-1], times[i], 1e-2);
    solutions.push_back(solution);
  }

  // Step 2 -- Accumulate log likelihood
  double likelihood = 0.0;
  for (unsigned int set_num=0; set_num < data.size(); ++set_num)
    {
      // Step 2a -- Extract the particle sizes from the ODE solution
      const unsigned int smallest = ode_model.nucleation_order;
      const unsigned int largest = ode_model.max_size;

      VectorType sizes(largest - smallest + 1);
      VectorType concentration(sizes.size());

      for (unsigned int size = smallest; size < largest+1; ++size)
        {
          sizes[size - smallest] = size;
          concentration[size - smallest] = solutions[set_num+1](size); // FIXME: particle size to index function needed
        }
      // Step 2b -- Calculate log likelihood of current data set and add to total
      likelihood += Statistics::log_likelihood(data[set_num],
                                               concentration,
                                               sizes,
                                               hist_prm);
    }

  return likelihood;
}



double Statistics::rand_btwn_double (const double small_num, const double big_num)
{
  static std::mt19937 gen;
  std::uniform_real_distribution<> unif(small_num,big_num);

  return unif(gen);
}



int Statistics::rand_btwn_int (const int small_num, const int big_num)
{
  static std::mt19937 gen;
  std::uniform_int_distribution<> unif(small_num,big_num);

  return unif(gen);
}