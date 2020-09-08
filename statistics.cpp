#include <valarray>
#include <cmath>
#include <tuple>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



// log likelihood
// This function solves the ODE based on odeModel, odeParameters, and solverParameters
// Then the ODE solution and data are converted to histograms based on histogramParameters.
// Finally the log likelihood is computed using the multinomial distribution and the two histograms.
std::tuple<double, std::vector<std::valarray<double>>, std::vector<std::valarray<double>>> Statistics::logLikelihoodInternal(const std::vector<std::valarray<double>>& particleSizeData,
                                                                                                                             const Models::ModelsBase& odeModel,
                                                                                                                             const Models::ParametersBase& odeParameters,
                                                                                                                             const Models::explEulerParameters& solverParameters,
                                                                                                                             const Histograms::Parameters& histogramParameters,
                                                                                                                             const double& timeStep)
{
  // Step 1 - solve the ODE
  const std::vector<std::valarray<double>> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters, odeModel, odeParameters, timeStep);

  // Step 2 - convert ODE solution to probability distribution on relevant particle sizes (i.e. sizes nucleationOrder - maxSize)

  // identify smallest and largest particle sizes
  const unsigned int smallestParticle = odeModel.getSmallestParticleSize(odeParameters);
  const unsigned int largestParticle = odeModel.getLargestParticleSize(odeParameters);

  // isolate the relevant particle sizes
  std::valarray<double> particleSizes (largestParticle - smallestParticle + 1);
  for (unsigned int size = smallestParticle; size < largestParticle + 1; size++)
  {
    // vector holding particle sizes
    particleSizes[size - smallestParticle] = size;
  }

  // start with log likelihood at 0
  double logLikelihoodVal = 0.0;

  // instantiate vectors for future output
  std::vector<std::valarray<double>> pmfs;
  std::vector<std::valarray<double>> data_counts;

  // loop over all saved solution vectors
  for (unsigned int i = 0; i < particleSizeDistr.size(); i++)
  {
    // extract ith solution vector and instantiate vector to hold concentrations
    // of nanoparticles -- i.e. remove variables concerning the nucleation mechanism.
    std::valarray<double> distribution = particleSizeDistr[i];
    std::valarray<double> concentrations (particleSizes.size());

    // loop over all particles to population concentrations vector
    for (unsigned int size = smallestParticle; size < largestParticle + 1; size++)
    {
      concentrations[size - smallestParticle] = distribution[odeModel.particleSizeToIndex(size, odeParameters)];
    }

    // normalize subset vector so its sum is equal to 1
    double normalizeConst = concentrations.sum();
    std::valarray<double> particleSizePMF = concentrations/normalizeConst;

    // create histogram from PMF to aggregate bin probability
    Histograms::Histogram odeHistogram(histogramParameters);

    odeHistogram.AddToBins(particleSizePMF, particleSizes);
    pmfs.push_back(odeHistogram.count);

    // Step 3 - convert data to histogram
    Histograms::Histogram dataHistogram(histogramParameters);

    // create vector of ones the same size as data
    std::valarray<double> dataCounts(1.0, particleSizeData[i].size());

    dataHistogram.AddToBins(dataCounts, particleSizeData[i]);
    data_counts.push_back(dataHistogram.count);

    // Step 4 - use probabilities from step 2 and bin counts from step 3
    // to compute log likelihood (without proportional part) based on
    // the multinomial distribution.

    for (unsigned int j = 0; j < histogramParameters.n_bins; j++)
    {
      logLikelihoodVal += dataHistogram.count[j]*std::log(odeHistogram.count[j]);
    }
  }

  return std::make_tuple(logLikelihoodVal, data_counts, pmfs);

}



// log likelihood
// This function solves the ODE based on odeModel, odeParameters, and solverParameters
// Then the ODE solution and data are converted to histograms based on histogramParameters.
// Finally the log likelihood is computed using the multinomial distribution and the two histograms.
double Statistics::logLikelihood(const std::vector<std::valarray<double>>& particleSizeData,
                                 const Models::ModelsBase& odeModel,
                                 const Models::ParametersBase& odeParameters,
                                 const Models::explEulerParameters& solverParameters,
                                 const Histograms::Parameters& histogramParameters,
                                 const double& timeStep)
{
  auto likelihood_results = Statistics::logLikelihoodInternal(particleSizeData,
                                                              odeModel,
                                                              odeParameters,
                                                              solverParameters,
                                                              histogramParameters,
                                                              timeStep);

  return std::get<0>(likelihood_results);

}
