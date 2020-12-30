#include <iostream>
#include <ode_solver.h>
#include <eigen3/Eigen/Dense>

int main()
{
  const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
  const double start_time = 0.;
  const double end_time = 1.;
  const double dt = 1e-4;

  // FIXME: this will require changes once OdeSystem is updated
  ODE::OdeSystem ode_system;
  ODE::StepperSDIRK<1> stepper(ode_system);
  auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

  // The answer should be close to exp(-10)
  std::cout << sol;
}