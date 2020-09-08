#include <iostream>
#include <string>
#include <valarray>
#include <models.h>



int main()
{
  // create a model to test the integrator
  Models::ThreeStepAlternative::Parameters prm(100, 90, 80, 70, 60, 2, 3, 6, 4);
  Models::ThreeStepAlternative model;

  // set up initial conditions, start time, and end time
  std::valarray<double> initialCondition = { 1,.9,.8,.7,.6,.5,.4 };
  double startTime = 0.0;
  std::vector<double> evalTimes(2);
  evalTimes[0] = 1e-5;
  evalTimes[1] = 2e-5;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, evalTimes, initialCondition);

  // run one time step
  const double time_step = 1e-5;
  std::vector<std::valarray<double>> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters,
                                                                                              model,
                                                                                              prm,
                                                                                              time_step);


  // output result for checking
  std::cout << "Checking first time step:" << std::endl;
  for (unsigned int i = 0; i < particleSizeDistr[0].size(); i++)
  {
    std::cout << particleSizeDistr[0][i]
              << std::endl;
  }

  // output result for checking
  std::cout << "Checking second time step:" << std::endl;
  for (unsigned int i = 0; i < particleSizeDistr[1].size(); i++)
  {
    std::cout << particleSizeDistr[1][i]
              << std::endl;
  }
}
