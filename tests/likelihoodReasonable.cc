#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"



int main()
{
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

  std::cout << "Bin counts for data: "
            << std::endl;

  for (unsigned int i=0; i< binned_data[0].size();i++)
  {
    std::cout << binned_data[0][i]
              << std::endl;
  }

  // print probability mass function

  auto binned_pmf = std::get<2>(logLikeliResults);

  std::cout << "Probability mass function: "
            << std::endl;

  for (unsigned int i=0; i< binned_pmf[0].size();i++)
  {
    std::cout << binned_pmf[0][i]
              << std::endl;
  }
}
