#include <iostream>
#include <string>
#include <valarray>
#include <vector>
#include "models.h"


int main()
{
  // create a model to test the integrator
  Models::ThreeStepAlternative::Parameters prm(3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);
  Models::ThreeStepAlternative model;

  // set up initial conditions, start time, and end time
  std::valarray<double> initialCondition(0.0, prm.n_variables);
  initialCondition[0] = 0.0012;
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
