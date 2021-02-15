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

    // We know our model and ODE solver are imperfect. Therefore, if we encounter a negative
    // value then we know this is a mathematical limitation. Fortunately, we can use physical
    // intuition to understand that zero is a more accurate value. We can extend that a step
    // further and see that instead of zero, a very small number relative to the rest of the
    // concentrations is perhaps more accurate and plays nicely with the logarithms used later.
    // To this end, we calculate the sum of all positive concentrations to get an idea of the
    // size scale. Then in the pmf calculation, the negative values are changed to be 1e-9 times
    // the max concentration. We could recalculate the norm adding in this adjustment to the
    // negative values, but it does not make a substantial difference, so that step is skipped.
    double norm = 0.;
    double max_conc = 0.;
    for (double concentration : distribution) {
      if (concentration > 0)
      {
        norm += concentration;
        max_conc = std::max(max_conc, concentration);
      }
    }

    VectorType pmf(distribution.size());
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
