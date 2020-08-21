#include <valarray>
#include <cmath>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



// log likelihood
// This function solves the ODE based on odeModel, odeParameters, and solverParameters
// Then the ODE solution and data are converted to histograms based on histogramParameters.
// Finally the log likelihood is computed using the multinomial distribution and the two histograms.
double Statistics::logLikelihood(std::valarray<double>& particleSizeData,
                                 Models::ModelsBase& odeModel,
                                 Models::ParametersBase& odeParameters,
                                 Models::explEulerParameters& solverParameters,
                                 Histograms::Parameters& histogramParameters)
{
  // Step 1 - solve the ODE
  std::valarray<double> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters, odeModel, odeParameters);

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
