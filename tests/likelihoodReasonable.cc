#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



int main()
{
  // S3 data
  std::valarray<double> dataDiam = {2.98, 2.82, 1.84, 2.04, 1.56, 1.56,
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


  std::valarray<double> dataAtoms = std::pow(dataDiam/0.3000805, 3);


  // create dummy model
  Models::ThreeStepAlternative::Parameters prm(3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);
  Models::ThreeStepAlternative model;

  // set up dummy initial conditions, start time, and end time
  std::valarray<double> initialCondition(0.0,2500);
  initialCondition[0] = 0.0012;
  double startTime = 0.0;
  double endTime = 1.170;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, endTime, initialCondition);

  // define parameters for the histogram
  Histograms::Parameters histogramParameters(25, 3.0, 2500.0);

  // calculate log likelihood
  double logLikeliVal = Statistics::logLikelihood(dataAtoms, model, prm, solverParameters, histogramParameters);

  // print likelihood result
  std::cout << "log likelihood: "
            << logLikeliVal
            << std::endl;

  // print binned data

  Histograms::Histogram dataHistogram(histogramParameters);

  std::valarray<double> dataCounts(1.0, dataAtoms.size());

  dataHistogram.AddToBins(dataCounts, dataAtoms);

  std::cout << "Bin counts for data: "
            << std::endl;

  for (unsigned int i=0; i< dataHistogram.count.size();i++)
  {
    std::cout << dataHistogram.count[i]
              << std::endl;
  }

  // print probability mass function

  // Step 1 - solve the ODE
   std::valarray<double> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters, model, prm);
  // Step 2 - convert ODE solution to probability distribution on relevant particle sizes (i.e. sizes nucleationOrder - maxSize)

  // identify smallest and largest particle sizes
  const unsigned int smallestParticle = model.getSmallestParticleSize(prm);
  const unsigned int largestParticle = model.getLargestParticleSize(prm);

  // isolate the relevant particle sizes
  std::valarray<double> particleSizes (largestParticle - smallestParticle + 1);
  std::valarray<double> concentrations (particleSizes.size());
  for (unsigned int size = smallestParticle; size < largestParticle + 1; size++)
  {
    // vector holding particle sizes
    particleSizes[size - smallestParticle] = size;
    // vector holding corresponding concentrations
    concentrations[size - smallestParticle] = particleSizeDistr[model.particleSizeToIndex(size, prm)];
  }

  // normalize subset vector so its sum is equal to 1
  const double normalizeConst = concentrations.sum();
  const std::valarray<double> particleSizePMF = concentrations/normalizeConst;

  // create histogram from PMF to aggregate bin probability
  Histograms::Histogram odeHistogram(histogramParameters);

  odeHistogram.AddToBins(particleSizePMF, particleSizes);

  std::cout << "Probability mass function: "
            << std::endl;

  for (unsigned int i=0; i< odeHistogram.count.size();i++)
  {
    std::cout << odeHistogram.count[i]
              << std::endl;
  }
}
