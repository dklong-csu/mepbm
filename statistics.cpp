#include <valarray>
#include <cmath>
#include <vector>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



// log likelihood
// This function solves the ODE based on odeModel, odeParameters, and solverParameters
// Then the ODE solution and data are converted to histograms based on histogramParameters.
// Finally the log likelihood is computed using the multinomial distribution and the two histograms.
double Statistics::logLikelihood(const std::valarray<double>& particleSizeData,
                                 const Models::ModelsBase& odeModel,
                                 const Models::ParametersBase& odeParameters,
                                 const Models::explEulerParameters& solverParameters,
                                 const Histograms::Parameters& histogramParameters)
{
  // Step 1 - solve the ODE
  const std::valarray<double> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters, odeModel, odeParameters);

  // Step 2 - convert ODE solution to probability distribution on relevant particle sizes (i.e. sizes nucleationOrder - maxSize)

  // identify smallest and largest particle sizes
  const unsigned int smallestParticle = odeModel.getSmallestParticleSize(odeParameters);
  const unsigned int largestParticle = odeModel.getLargestParticleSize(odeParameters);

  // isolate the relevant particle sizes
  std::valarray<double> particleSizes (largestParticle - smallestParticle + 1);
  std::valarray<double> concentrations (particleSizes.size());
  for (unsigned int size = smallestParticle; size < largestParticle + 1; size++)
  {
    // vector holding particle sizes
    particleSizes[size - smallestParticle] = size;
    // vector holding corresponding concentrations
    concentrations[size - smallestParticle] = particleSizeDistr[odeModel.particleSizeToIndex(size, odeParameters)];
  }

  // normalize subset vector so its sum is equal to 1
  const double normalizeConst = concentrations.sum();
  const std::valarray<double> particleSizePMF = concentrations/normalizeConst;

  // create histogram from PMF to aggregate bin probability
  Histograms::Histogram odeHistogram(histogramParameters);

  odeHistogram.AddToBins(particleSizePMF, particleSizes);

  // Step 3 - convert data to histogram
  Histograms::Histogram dataHistogram(histogramParameters);

  // create vector of ones the same size as data
  std::valarray<double> dataCounts(1.0, particleSizeData.size());

  dataHistogram.AddToBins(dataCounts, particleSizeData);

  // Step 4 - use probabilities from step 2 and bin counts from step 3
  // to compute log likelihood (without proportional part) based on
  // the multinomial distribution.

  double logLikelihoodVal = 0.0;
  for (unsigned int i = 0; i < histogramParameters.n_bins; i++)
  {
    logLikelihoodVal += dataHistogram.count[i]*std::log(odeHistogram.count[i]);
  }

  return logLikelihoodVal;

}



// log likelihood detailed calculation
double Statistics::log_likelihood(const std::valarray<double>& data,
                                  const std::valarray<double>& distribution,
                                  const std::valarray<double>& sizes,
                                  const Histograms::Parameters& hist_prm)
{
  // Step 1 -- Turn data into a histogram
  Histograms::Histogram hist_data(hist_prm);
  std::valarray<double> data_counts(1.0, data.size()); // each data point occurred 1 time
  hist_data.AddToBins(data_counts, data);

  // Step 2 -- Turn distribution into a histogram
    // Step 2a -- Normalize distribution to create probability mass function (pmf)
    const double norm_const = distribution.sum();
    const std::valarray<double> pmf = distribution/norm_const;

    // Step 2b -- Create histogram from pmf
    Histograms::Histogram hist_ode(hist_prm);
    hist_ode.AddToBins(pmf, sizes);

  // Step 3 -- Combine histograms to calculate log likelihood
  double likelihood = 0.0;
  for (unsigned int bin=0; bin < hist_prm.n_bins; ++bin)
    {
      // If the probability is zero and there are no data points, do not contribute anything
      // If the probability is zero and there are data points, then return most negative number possible
      // If the probability is non-zero, then calculate normally
      if (hist_ode.count[bin] == 0 && hist_data.count[bin] == 0)
        {
          // do nothing
        }
      else if (hist_ode.count[bin] == 0)
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
double Statistics::log_likelihood(const std::vector<std::valarray<double>>& data,
                                  const std::vector<double>& times,
                                  const Models::ModelsBase& ode_model,
                                  const Models::ParametersBase& ode_prm,
                                  const std::valarray<double>& ic,
                                  const Histograms::Parameters& hist_prm)
{
  // Step 1 -- Solve the ODE at each time
  const std::vector<std::valarray<double>> solutions = Models::integrate_ode_ee_many_times(ic, ode_model, ode_prm, times);

  // Step 2 -- Accumulate log likelihood
  double likelihood = 0.0;
  for (unsigned int set_num=0; set_num < data.size(); ++set_num)
    {
      // Step 2a -- Extract the particle sizes from the ODE solution
      const unsigned int smallest = ode_model.getSmallestParticleSize(ode_prm);
      const unsigned int largest = ode_model.getLargestParticleSize(ode_prm);

      std::valarray<double> sizes(largest - smallest + 1);
      std::valarray<double> concentration(sizes.size());

      for (unsigned int size = smallest; size < largest+1; ++size)
        {
          sizes[size - smallest] = size;
          concentration[size - smallest] = solutions[set_num+1][ode_model.particleSizeToIndex(size, ode_prm)];
        }
      // Step 2b -- Calculate log likelihood of current data set and add to total
      likelihood += Statistics::log_likelihood(data[set_num],
                                               concentration,
                                               sizes,
                                               hist_prm);
    }

  return likelihood;
}
