#include "ode_solver.h"
#include <eigen3/Eigen/Dense>



// FIXME: this is a temporary implementation until this is linked to Model::Model
Eigen::VectorXd ODE::OdeSystem::compute_rhs(double t, const Eigen::VectorXd &x) const
{
  return 10 * x;
}



// FIXME: this is a temporary implementation until this is linked to Model::Model
Eigen::MatrixXd ODE::OdeSystem::compute_jacobian(double t, const Eigen::VectorXd &x) const
{
  return 10 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
}



Eigen::VectorXd ODE::solve_ode(StepperBase &stepper, Eigen::VectorXd &ic, double t_start, double t_end, double dt)
{
  // Check for the pathological case where only 1 time step is used and make sure the time step is appropriate.
  if (t_start + dt > t_end)
    dt = t_end - t_start;

  // x0 represents the solution at the current time step.
  // x1 represents the solution at the next time step.
  auto x0 = ic;
  auto x1 = x0;

  // t is used to indicate the time at the current time step.
  double t = t_start;

  // Repeatedly step forward in time until the end time is reached.
  while (t < t_end)
  {
    // Step forward in time using the provided ODE solution method.
    x1 = stepper.step_forward(x0, t, dt);

    // If the next time step would go past the ending time, adjust the time step to end exactly on the end time.
    if (t + dt > t_end)
      dt = t_end - t;

    // Move forward one time step.
    t += dt;
    x0 = x1;
  }

  // The last x1 calculated is the desired output of the ODE solver.
  return x1;
}



std::pair<Eigen::VectorXd, unsigned int> ODE::newton_method(const FunctionBase &fcn,
                                                            const Eigen::PartialPivLU<Eigen::MatrixXd> &jac,
                                                            const Eigen::VectorXd &guess, const double tol,
                                                            const unsigned int max_iter)
{
  bool solution_not_found = true;
  unsigned int iter = 0;

  auto x0 = guess;
  auto x1 = x0;

  while (solution_not_found && iter < max_iter)
  {
    /*
     * jacobian * (x1 - x0) = -f
     * --> jacobian * d = -f
     * --> x1 = x0 + d
     */
    auto f = fcn.value(x0);
    auto d = jac.solve(-f);
    x1 = x0 + d;

    // Check the residual of the function to see if x1 is close to the root.
    f = fcn.value(x1);
    auto diff = f / std::min(x0.norm(), x1.norm());
    if (diff.norm() < tol)
      solution_not_found = false;

    // Update for the next step
    ++iter;
    x0 = x1;
  }

  // Return the pair ( x solution, number of iterations)
  return {x1, iter};
}