#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



int main()
{
  // create dummy data
  std::valarray<double> dataPre = {3,4,5,6,4,5};
  std::vector<std::valarray<double>> data(1);
  data[0] = dataPre;

  // create dummy model
  Models::ThreeStepAlternative::Parameters prm(100, 90, 80, 70, 60, 2, 3, 6, 4);
  Models::ThreeStepAlternative model;

  // set up dummy initial conditions, start time, and end time
  std::valarray<double> initialCondition = { 1,.9,.8,.7,.6,.5,.4 };
  double startTime = 0.0;
  double endTime = 2e-6;
  std::vector<double> evalTimes(1);
  evalTimes[0] = endTime;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, evalTimes, initialCondition);

  // define parameters for the histogram
  Histograms::Parameters histogramParameters(4, 3.0, 6.0);

  // calculate log likelihood
  const double dt = 1e-5;
  double logLikeliVal = Statistics::logLikelihood(data, model, prm, solverParameters, histogramParameters, dt);

  // print result
  std::cout << "log likelihood: " << logLikeliVal;
}
