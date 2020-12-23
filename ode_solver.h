#ifndef MEPBM_ODE_SOLVER_H
#define MEPBM_ODE_SOLVER_H

#include <eigen3/Eigen/Dense>
#include <cmath>


namespace ODE
{
  /*
   * A function which solves an ODE in an abstract way. The details of how to proceed from one time step to the next
   * is left up to the input ODE system using its step_forward member function.
   */
  template<class OdeSystemType, class VectorType>
  VectorType solve_ode(const OdeSystemType &ode_system, const VectorType &ic,
                       const double start_time, const double end_time, double dt)
  {
    // Check for the pathological case where only 1 time step is used and make sure the time step is appropriate.
    if (start_time + dt > end_time)
      dt = end_time - start_time;

    // x0 is going to represent the current solution
    // x1 is going to represent the next solution
    // Here, these two are instantiated for use later and the first ``current solution" needs to be
    // the initial condition
    auto x0 = ic;
    VectorType x1;

    // t is used to track the current time and it needs to start at the specified start time.
    double t = start_time;
    while (t < end_time)
    {
      // OdeSystemType needs to have a step_forward method which describes how the next time step is
      // solved for -- e.g. implicit Euler method.
      x1 = ode_system.step_forward(x0, t, dt);

      // If the next time step would go past the ending time, adjust the time step to end exactly on the end time.
      if (t + dt > end_time)
        dt = end_time - t;

      // Update the time and the current solution.
      t += dt;
      x0 = x1;
    }

    return x1;
  }



  /*
   * A function which performs Newton's method to find the root of a nonlinear equation.
   * Technically, this is a modification where the Jacobian is taken as a constant for the
   * entire process. This is to save time as computing a decomposition of a Jacobian is more
   * expensive than solving with a precomputed decomposition (e.g. LU decomposition).
   */
  template<class FunctionType, class VectorType>
  VectorType perform_newtons_method(const FunctionType &function, const VectorType &guess,
                                    const double tol = 1e-6, const unsigned int max_iter = 100)
  {
    bool solution_not_found = true;
    unsigned int iter = 0;

    auto x0 = guess;
    auto x1 = guess;
    while (solution_not_found && iter < max_iter)
    {
      auto f = function.value(x0);
      // J(x_{n+1} - x_n) = -f
      auto d = function.jacobian.solve(-f);
      x1 = x0 + d;

      // Check how close to zero the Newton function is after this iteration.
      // Divide the l2 norm of the Newton function by the minimum l2 norm of x0 and x1
      // in order to solve to an accuracy appropriate to the units of x.
      f = function.value(x1);
      auto diff = f / std::min(x0.norm(), x1.norm());
      if (diff.norm() < tol)
        solution_not_found = true;

      // increment the iteration counter
      ++iter;
      // x1 we just solved for becomes the new guess
      x0 = x1;
    }
    return x1;
  }





}

#endif //MEPBM_ODE_SOLVER_H