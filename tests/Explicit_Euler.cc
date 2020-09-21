#include <iostream>
#include <string>
#include <valarray>
#include "models.h"



int main()
{
  // create a model to test the integrator
  Models::ThreeStepAlternative::Parameters prm(100, 90, 80, 70, 60, 2, 3, 6, 4);
  Models::ThreeStepAlternative model;

  // set up initial conditions, start time, and end time
  std::valarray<double> initialCondition = { 1,.9,.8,.7,.6,.5,.4 };
  double startTime = 0.0;
  double endTime = 1e-5;

  // set up solver parameters
  Models::explEulerParameters solverParameters(startTime, endTime, initialCondition);

  // run one time step
  std::valarray<double> particleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters,
                                                                                 model,
                                                                                 prm);


  // output result for checking
  std::cout << "Checking first time step:" << std::endl;
  for (unsigned int i = 0; i < particleSizeDistr.size(); i++)
  {
    std::cout << particleSizeDistr[i]
              << std::endl;
  }

  // run two time steps
  double newEndTime = 2e-5;

  solverParameters.endTime = newEndTime;
  std::valarray<double> newParticleSizeDistr = Models::integrate_ode_explicit_euler(solverParameters,
                                                                                    model,
                                                                                    prm);


  // output result for checking
  std::cout << "Checking second time step:" << std::endl;
  for (unsigned int i = 0; i < newParticleSizeDistr.size(); i++)
  {
    std::cout << newParticleSizeDistr[i]
              << std::endl;
  }
}
