#include "ode_solver.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <deque>



// FIXME: this is a temporary implementation until this is linked to Model::Model
/*
Eigen::VectorXd ODE::OdeSystem::compute_rhs(double t, const Eigen::VectorXd &x) const
{
  return -10 * x;
}
*/


// FIXME: this is a temporary implementation until this is linked to Model::Model
/*
Eigen::MatrixXd ODE::OdeSystem::compute_jacobian(double t, const Eigen::VectorXd &x) const
{
  return -10 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
}
*/


Eigen::VectorXd ODE::solve_ode(StepperBase &stepper, const Eigen::VectorXd &ic, const double t_start,
                               const double t_end, double dt)
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

  Eigen::VectorXd d(guess.rows());
  
  while (solution_not_found && iter < max_iter)
  {
    /*
     * jacobian * (x1 - x0) = -f
     * --> jacobian * d = -f
     * --> x1 = x0 + d
     */
    auto f = fcn.value(x0);
    d = jac.solve(-f);
    x1 = x0 + d;

    // Check the residual of the function to see if x1 is close to the root.
    f = fcn.value(x1);
    double divisor;
    if (std::min(x0.norm(), x1.norm()) < tol)
    {
      divisor = 1.;
    }
    else
    {
      divisor = std::min(x0.norm(), x1.norm());
    }

    if (f.norm() / divisor < tol)
      solution_not_found = false;

    // Update for the next step
    ++iter;
    x0 = x1;
  }

  // Return the pair ( x solution, number of iterations)
  return {x1, iter};
}



/********************************** First order SDIRK **********************************/

ODE::StepperSDIRK<1>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                     const Eigen::VectorXd &x0)
    : ode_system(ode_system), t(t), dt(dt), x0(x0)
{}



Eigen::VectorXd ODE::StepperSDIRK<1>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  Eigen::VectorXd y = x0 + dt * x;
  return x - ode_system.rhs(y);
}



ODE::StepperSDIRK<1>::StepperSDIRK(const Model::Model &ode_system)
    : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
{}



Eigen::VectorXd ODE::StepperSDIRK<1>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  if (update_jacobian)
  {
    Eigen::MatrixXd jac = ode_system.jacobian(x0);

    Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * jac;

    jacobian_solver = newton_jacobian.partialPivLu();
  }

  StepperSDIRK<1>::NewtonFunction fcn(ode_system, t, dt, x0);
  const auto guess = Eigen::VectorXd::Zero(x0.rows());
  auto newton_result = newton_method(fcn, jacobian_solver, guess);

  auto num_newton_steps = newton_result.second;
  if (update_jacobian)
  {
    update_jacobian = false;
    num_iter_new_jac = num_newton_steps;
  }
  else if (num_newton_steps > 5 * num_iter_new_jac)
  {
    update_jacobian = true;
  }

  return x0 + dt * newton_result.first;
}



/********************************** Second order SDIRK **********************************/

ODE::StepperSDIRK<2>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                     const Eigen::VectorXd &x0)
    : ode_system(ode_system), t(t), dt(dt), x0(x0)
{}


// FIXME: add comments
Eigen::VectorXd ODE::StepperSDIRK<2>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  Eigen::VectorXd y = x0 + dt * 1./4 * x;
  return x - ode_system.rhs(y);
}



ODE::StepperSDIRK<2>::StepperSDIRK(const Model::Model &ode_system)
    : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
{}



// FIXME: add comments
Eigen::VectorXd ODE::StepperSDIRK<2>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  if (update_jacobian)
  {
    Eigen::MatrixXd jac = ode_system.jacobian(x0);

    Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * 1./4 * jac;

    jacobian_solver = newton_jacobian.partialPivLu();
  }

  StepperSDIRK<2>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
  const auto guess = Eigen::VectorXd::Zero(x0.rows());
  auto newton_result_k1 = newton_method(fcn_k1, jacobian_solver, guess);
  auto k1 = newton_result_k1.first;

  StepperSDIRK<2>::NewtonFunction fcn_k2(ode_system, t, dt, x0 + dt * 1./2 * k1);
  auto newton_result_k2 = newton_method(fcn_k2, jacobian_solver, guess);
  auto k2 = newton_result_k2.first;

  auto num_iter = std::max(newton_result_k1.second, newton_result_k2.second);
  if (update_jacobian)
  {
    update_jacobian = false;
    num_iter_new_jac = num_iter;
  }
  else if (num_iter > 5 * num_iter_new_jac)
  {
    update_jacobian = true;
  }

  return x0 + dt * 1./2 * k1 + dt * 1./2 * k2;
}



/********************************** Third order SDIRK **********************************/

ODE::StepperSDIRK<3>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                     const Eigen::VectorXd &x0)
    : ode_system(ode_system), t(t), dt(dt), x0(x0)
{}


// FIXME: add comments
Eigen::VectorXd ODE::StepperSDIRK<3>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  Eigen::VectorXd y = x0 + dt * butcher_diag * x;
  return x - ode_system.rhs(y);
}



ODE::StepperSDIRK<3>::StepperSDIRK(const Model::Model &ode_system)
    : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
{}



// FIXME: add comments
Eigen::VectorXd ODE::StepperSDIRK<3>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  if (update_jacobian)
  {
    Eigen::MatrixXd jac = ode_system.jacobian(x0);

    Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * butcher_diag * jac;

    jacobian_solver = newton_jacobian.partialPivLu();
  }

  StepperSDIRK<3>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
  const auto guess = Eigen::VectorXd::Zero(x0.rows());
  auto newton_result_k1 = newton_method(fcn_k1, jacobian_solver, guess);
  auto k1 = newton_result_k1.first;

  const double k2_coeff_k1 = (1 - butcher_diag) / 2;
  StepperSDIRK<3>::NewtonFunction fcn_k2(ode_system, t, dt, x0 + dt * k2_coeff_k1 * k1);
  auto newton_result_k2 = newton_method(fcn_k2, jacobian_solver, guess);
  auto k2 = newton_result_k2.first;

  const double k3_coeff_k1 = -3 * butcher_diag * butcher_diag / 2 + 4 * butcher_diag - 1./4;
  const double k3_coeff_k2 =  3 * butcher_diag * butcher_diag / 2 - 5 * butcher_diag + 5./4;
  StepperSDIRK<3>::NewtonFunction fcn_k3(ode_system, t, dt, x0 + dt * k3_coeff_k1 * k1 + dt * k3_coeff_k2 * k2);
  auto newton_result_k3 = newton_method(fcn_k3, jacobian_solver, guess);
  auto k3 = newton_result_k3.first;

  auto num_iter = std::max(newton_result_k1.second, newton_result_k2.second);
  num_iter = std::max(num_iter, newton_result_k3.second);
  if (update_jacobian)
  {
    update_jacobian = false;
    num_iter_new_jac = num_iter;
  }
  else if (num_iter > 5 * num_iter_new_jac)
  {
    update_jacobian = true;
  }

  const double k1_weight = -3 * butcher_diag * butcher_diag / 2 + 4 * butcher_diag - 1./4;
  const double k2_weight =  3 * butcher_diag * butcher_diag / 2 - 5 * butcher_diag + 5./4;
  const double k3_weight =  butcher_diag;

  return x0 + dt * k1_weight * k1 + dt * k2_weight * k2 + dt * k3_weight * k3;
}



/********************************** Fourth order SDIRK **********************************/

ODE::StepperSDIRK<4>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                     const Eigen::VectorXd &x0)
    : ode_system(ode_system), t(t), dt(dt), x0(x0)
{}


// FIXME: add comments
Eigen::VectorXd ODE::StepperSDIRK<4>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  Eigen::VectorXd y = x0 + dt * butcher_diag * x;
  return x - ode_system.rhs(y);
}



ODE::StepperSDIRK<4>::StepperSDIRK(const Model::Model &ode_system)
    : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
{}



// FIXME: add comments
Eigen::VectorXd ODE::StepperSDIRK<4>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  if (update_jacobian)
  {
    Eigen::MatrixXd jac = ode_system.jacobian(x0);

    Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * butcher_diag * jac;

    jacobian_solver = newton_jacobian.partialPivLu();
  }

  StepperSDIRK<4>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
  const auto guess = Eigen::VectorXd::Zero(x0.rows());
  auto newton_result_k1 = newton_method(fcn_k1, jacobian_solver, guess);
  auto k1 = newton_result_k1.first;

  const double k2_coeff_k1 = 1./2 - butcher_diag;
  StepperSDIRK<4>::NewtonFunction fcn_k2(ode_system, t, dt, x0 + dt * k2_coeff_k1 * k1);
  auto newton_result_k2 = newton_method(fcn_k2, jacobian_solver, guess);
  auto k2 = newton_result_k2.first;

  const double k3_coeff_k1 = 2 * butcher_diag;
  const double k3_coeff_k2 =  1 - 4 * butcher_diag;
  StepperSDIRK<4>::NewtonFunction fcn_k3(ode_system, t, dt, x0 + dt * k3_coeff_k1 * k1 + dt * k3_coeff_k2 * k2);
  auto newton_result_k3 = newton_method(fcn_k3, jacobian_solver, guess);
  auto k3 = newton_result_k3.first;

  auto num_iter = std::max(newton_result_k1.second, newton_result_k2.second);
  num_iter = std::max(num_iter, newton_result_k3.second);
  if (update_jacobian)
  {
    update_jacobian = false;
    num_iter_new_jac = num_iter;
  }
  else if (num_iter > 5 * num_iter_new_jac)
  {
    update_jacobian = true;
  }

  const double k1_weight = 1 / (6 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );
  const double k2_weight =  (3 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) - 1)
                            / (3 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );
  const double k3_weight =  1 / (6 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );

  return x0 + dt * k1_weight * k1 + dt * k2_weight * k2 + dt * k3_weight * k3;
}



/********************************** First order BDF **********************************/

ODE::StepperBDF<1>::StepperBDF(const Model::Model &ode_system)
  : implicit_euler(ode_system)
{}



Eigen::VectorXd ODE::StepperBDF<1>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  return implicit_euler.step_forward(x0, t, dt);
}



/********************************** Second order BDF **********************************/

ODE::StepperBDF<2>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                   const std::deque<Eigen::VectorXd> &prev_sols)
  : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
{}



Eigen::VectorXd ODE::StepperBDF<2>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  assert(prev_sols.size() == 2);
  return x - 4./3 * prev_sols[1] + 1./3 * prev_sols[0] - 2./3 * dt * ode_system.rhs(const_cast<Eigen::VectorXd &>(x));
}



ODE::StepperBDF<2>::StepperBDF(const Model::Model &ode_system)
  : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
{}



Eigen::VectorXd ODE::StepperBDF<2>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  // Add x0 as a previous solution
  prev_sols.push_back(x0);

  // If prev_sols has length < 2 right now then we cannot use the BDF method yet
  if (prev_sols.size() < 2)
  {
    // Step forward using SDIRK method
   return sdirk_stepper.step_forward(x0, t, dt);
  }
  // Otherwise we can use the BDF method
  else
  {
    /*
     * If prev_sols.size() is exactly 2 right now then this is the first time BDF is used and thus we do not
     * need to remove any of the prev_sols. Otherwise, the first entry in prev_sols is no longer needed and
     * can be discarded.
     */
    if (prev_sols.size() > 2)
    {
      prev_sols.pop_front();
    }

    if (update_jacobian)
    {
      Eigen::MatrixXd jac = ode_system.jacobian(x0);

      Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * 2/3 * jac;

      jacobian_solver = newton_jacobian.partialPivLu();
    }

    StepperBDF<2>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
    const auto guess = Eigen::VectorXd::Zero(x0.rows());
    auto newton_result = newton_method(fcn, jacobian_solver, guess);

    auto num_newton_steps = newton_result.second;
    if (update_jacobian)
    {
      update_jacobian = false;
      num_iter_new_jac = num_newton_steps;
    }
    else if (num_newton_steps > 5 * num_iter_new_jac)
    {
      update_jacobian = true;
    }

    return newton_result.first;
  }
}



/********************************** Third order BDF **********************************/

ODE::StepperBDF<3>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                   const std::deque<Eigen::VectorXd> &prev_sols)
    : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
{}



Eigen::VectorXd ODE::StepperBDF<3>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  assert(prev_sols.size() == 3);
  return x - 18./11 * prev_sols[2] + 9./11 * prev_sols[1] - 2./11 * prev_sols[0]
    - 6./11 * dt * ode_system.rhs(const_cast<Eigen::VectorXd &>(x));
}



ODE::StepperBDF<3>::StepperBDF(const Model::Model &ode_system)
    : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
{}



Eigen::VectorXd ODE::StepperBDF<3>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  // Add x0 as a previous solution
  prev_sols.push_back(x0);

  // If prev_sols has length < 3 right now then we cannot use the BDF method yet
  if (prev_sols.size() < 3)
  {
    // Step forward using SDIRK method
    return sdirk_stepper.step_forward(x0, t, dt);
  }
    // Otherwise we can use the BDF method
  else
  {
    /*
     * If prev_sols.size() is exactly 3 right now then this is the first time BDF is used and thus we do not
     * need to remove any of the prev_sols. Otherwise, the first entry in prev_sols is no longer needed and
     * can be discarded.
     */
    if (prev_sols.size() > 3)
    {
      prev_sols.pop_front();
    }

    if (update_jacobian)
    {
      Eigen::MatrixXd jac = ode_system.jacobian(x0);

      Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * 6/11 * jac;

      jacobian_solver = newton_jacobian.partialPivLu();
    }

    StepperBDF<3>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
    const auto guess = Eigen::VectorXd::Zero(x0.rows());
    auto newton_result = newton_method(fcn, jacobian_solver, guess);

    auto num_newton_steps = newton_result.second;
    if (update_jacobian)
    {
      update_jacobian = false;
      num_iter_new_jac = num_newton_steps;
    }
    else if (num_newton_steps > 5 * num_iter_new_jac)
    {
      update_jacobian = true;
    }

    return newton_result.first;
  }
}


/********************************** Fourth order BDF **********************************/

ODE::StepperBDF<4>::NewtonFunction::NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                                                   const std::deque<Eigen::VectorXd> &prev_sols)
    : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
{}



Eigen::VectorXd ODE::StepperBDF<4>::NewtonFunction::value(const Eigen::VectorXd &x) const
{
  assert(prev_sols.size() == 4);
  return x - 48./25 * prev_sols[3] + 36./25 * prev_sols[2] - 16./25 * prev_sols[1] + 3./25 * prev_sols[0]
         - 12./25 * dt * ode_system.rhs(const_cast<Eigen::VectorXd &>(x));
}



ODE::StepperBDF<4>::StepperBDF(const Model::Model &ode_system)
    : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
{}



Eigen::VectorXd ODE::StepperBDF<4>::step_forward(Eigen::VectorXd &x0, double t, double dt)
{
  // Add x0 as a previous solution
  prev_sols.push_back(x0);

  // If prev_sols has length < 4 right now then we cannot use the BDF method yet
  if (prev_sols.size() < 4)
  {
    // Step forward using SDIRK method
    return sdirk_stepper.step_forward(x0, t, dt);
  }
    // Otherwise we can use the BDF method
  else
  {
    /*
     * If prev_sols.size() is exactly 4 right now then this is the first time BDF is used and thus we do not
     * need to remove any of the prev_sols. Otherwise, the first entry in prev_sols is no longer needed and
     * can be discarded.
     */
    if (prev_sols.size() > 4)
    {
      prev_sols.pop_front();
    }

    if (update_jacobian)
    {
      Eigen::MatrixXd jac = ode_system.jacobian(x0);

      Eigen::MatrixXd newton_jacobian = Eigen::MatrixXd::Identity(jac.rows(), jac.cols()) - dt * 12/25 * jac;

      jacobian_solver = newton_jacobian.partialPivLu();
    }

    StepperBDF<4>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
    const auto guess = x0;
    auto newton_result = newton_method(fcn, jacobian_solver, guess);

    auto num_newton_steps = newton_result.second;
    if (update_jacobian)
    {
      update_jacobian = false;
      num_iter_new_jac = num_newton_steps;
    }
    else if (num_newton_steps > 5 * num_iter_new_jac)
    {
      update_jacobian = true;
    }

    return newton_result.first;
  }
}
