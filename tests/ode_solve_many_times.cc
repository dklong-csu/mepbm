#include <iostream>
#include <string>
#include <valarray>
#include <vector>
#include "models.h"


int main()
{
  // create a model to test the integrator
  Models::ThreeStepAlternative::Parameters prm(100, 90, 80, 70, 60, 2, 3, 6, 4);
  Models::ThreeStepAlternative model;

  // set up initial conditions, start time, and end time
  std::valarray<double> initialCondition = { 1,.9,.8,.7,.6,.5,.4 };
  std::vector<double> sol_times = {0.0, 1e-5, 2e-5};

  // run the solver
  std::vector<std::valarray<double>> sols_all_times = Models::integrate_ode_ee_many_times(initialCondition, model, prm, sol_times);

  // output solutions
  for (unsigned int time_iter=0; time_iter < sols_all_times.size(); ++time_iter)
  {
    std::cout << "Checking time step: "
              << time_iter
              << std::endl;
    for (unsigned int entry=0; entry < sols_all_times[time_iter].size(); ++entry)
    {
      std::cout << sols_all_times[time_iter][entry]
                << std::endl;
    }
  }
}
