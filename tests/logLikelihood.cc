#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



int main()
{
  // create dummy data
  std::valarray<double> data = {3,4,5,6,4,5};

  // create dummy model
  Models::ThreeStepAlternative::Parameters prm(100, 90, 80, 70, 60, 2, 3, 6, 4);
  Models::ThreeStepAlternative model;

  // set up dummy initial conditions, start time, and end time
  std::valarray<double> initialCondition = { 1,.9,.8,.7,.6,.5,.4 };
  double startTime = 0.0;
  double endTime = 2e-6;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, endTime, initialCondition);

  // define parameters for the histogram
  Histograms::Parameters histogramParameters(4, 3.0, 6.0);

  // calculate log likelihood
  double logLikeliVal = Statistics::logLikelihood(data, model, prm, solverParameters, histogramParameters);

  // print result
  std::cout << "log likelihood: " << logLikeliVal;
}
