#include <iostream>
#include <string>
#include <valarray>
#include <models.h>



int main()
{
  // create a model to test the integrator
  Models::ThreeStepAlternative::Parameters prm(3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);
  Models::ThreeStepAlternative model;

  // set up initial conditions, start time, and end time
  std::valarray<double> initialCondition(0.0,2500);
  initialCondition[0] = 0.0012;
  double startTime = 0.0;
  std::vector<double> evalTimes(1);
  evalTimes[0] = 3.e-5;

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
}
