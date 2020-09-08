#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"



int main()
{
  // S3 data
  /*
  const std::valarray<double> dataDiam = {2.98, 2.82, 1.84, 2.04, 1.56, 1.56,
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
*/

  const Data::PomData pom_data;
  const std::valarray<double> dataDiam = pom_data.tem_diam_time2;

  std::vector<std::valarray<double>> fitData(1);
  fitData[0] = std::pow(dataDiam/0.3000805, 3);

  std::vector<double> fitTime(1);
  fitTime[0] = pom_data.tem_time2;


  // create model
  const Models::ThreeStepAlternative::Parameters prm(3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);
  Models::ThreeStepAlternative model;

  // set up initial conditions, start time, and end time
  std::valarray<double> initialCondition(0.0,2500);
  initialCondition[0] = 0.0012;
  const double startTime = 0.0;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, fitTime, initialCondition);

  // define parameters for the histogram
  const Histograms::Parameters histogramParameters(25, 3.0, 2500.0);

  // calculate log likelihood
  const double time_step = 1e-5;
  const auto logLikeliResults = Statistics::logLikelihoodInternal(fitData, model, prm, solverParameters, histogramParameters, time_step);

  // print likelihood result
  std::cout << "log likelihood: "
            << std::get<0>(logLikeliResults)
            << std::endl;

  // print binned data

  auto binned_data = std::get<1>(logLikeliResults);

  //Histograms::Histogram dataHistogram(histogramParameters);

  //std::valarray<double> dataCounts(1.0, fitData[0].size());

  //dataHistogram.AddToBins(dataCounts, fitData[0]);

  std::cout << "Bin counts for data: "
            << std::endl;

  for (unsigned int i=0; i< binned_data[0].size();i++)
  {
    std::cout << binned_data[0][i]
              << std::endl;
  }

  // print probability mass function

  // Step 1 - solve the ODE
   //const std::vector<std::valarray<double>> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters, model, prm, time_step);
  // Step 2 - convert ODE solution to probability distribution on relevant particle sizes (i.e. sizes nucleationOrder - maxSize)

  // identify smallest and largest particle sizes
  //const unsigned int smallestParticle = model.getSmallestParticleSize(prm);
  //const unsigned int largestParticle = model.getLargestParticleSize(prm);

  // isolate the relevant particle sizes
  //std::valarray<double> particleSizes (largestParticle - smallestParticle + 1);
  //std::valarray<double> concentrations (particleSizes.size());
  //for (unsigned int size = smallestParticle; size < largestParticle + 1; size++)
  {
    // vector holding particle sizes
    //particleSizes[size - smallestParticle] = size;
    // vector holding corresponding concentrations
    //concentrations[size - smallestParticle] = particleSizeDistr[0][model.particleSizeToIndex(size, prm)];
  }

  // normalize subset vector so its sum is equal to 1
  //const double normalizeConst = concentrations.sum();
  //const std::valarray<double> particleSizePMF = concentrations/normalizeConst;

  // create histogram from PMF to aggregate bin probability
  //Histograms::Histogram odeHistogram(histogramParameters);

  //odeHistogram.AddToBins(particleSizePMF, particleSizes);

  auto binned_pmf = std::get<2>(logLikeliResults);

  std::cout << "Probability mass function: "
            << std::endl;

  for (unsigned int i=0; i< binned_pmf[0].size();i++)
  {
    std::cout << binned_pmf[0][i]
              << std::endl;
  }
}
