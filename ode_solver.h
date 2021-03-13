#ifndef MEPBM_ODE_SOLVER_H
#define MEPBM_ODE_SOLVER_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "eigen3/unsupported/Eigen/src/IterativeSolvers/GMRES.h"
#include <cmath>
#include <deque>
#include <vector>
#include "models.h"
#include <iostream>


namespace ODE
{
  /*
   * Base class for ODE time stepper
   */
  template<typename Real>
  class StepperBase
  {
  public:
    virtual Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) = 0;
  };



  /*
   * A function which solves an ODE. This is intended to just lay out the basic framework for an ODE solve.
   * The stepper -- some derived class of StepperBase -- is intended to do most of the work of the ODE solve
   * within its `step_forward` method. This function simply facilitates repeatedly using `step_forward` to
   * go from the initial time to the final time.
   */
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  solve_ode(StepperBase<Real> &stepper,
           const Eigen::Matrix<Real, Eigen::Dynamic, 1> &ic,
           const Real t_start,
           const Real t_end,
           Real dt);


  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  solve_ode(StepperBase<Real> &stepper,
            const Eigen::Matrix<Real, Eigen::Dynamic, 1> &ic,
            const Real t_start,
            const Real t_end,
            Real dt)
  {
    // Check for the pathological case where only 1 time step is used and make sure the time step is appropriate.
    if (t_start + dt > t_end)
      dt = t_end - t_start;

    // x0 represents the solution at the current time step.
    // x1 represents the solution at the next time step.
    auto x0 = ic;
    auto x1 = x0;

    // t is used to indicate the time at the current time step.
    Real t = t_start;

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



  /*
   * A base class to represent a function. This simply has a rule for returning the value of the function.
   */
  template<typename Real>
  class FunctionBase
  {
  public:
    virtual Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const = 0;
  };



  /*
   * A function which performs Newton's method to find the root of a nonlinear equation.
   * Technically, this is a modification where the Jacobian is taken as a constant for the
   * entire process. This is to save time as computing a decomposition of a Jacobian is more
   * expensive than solving with a precomputed decomposition (e.g. LU decomposition).
   *
   * This is a modified Newton's method where the Jacobian is held constant for the nonlinear solve.
  */
  template<typename Real, typename Solver>
  std::pair<Eigen::Matrix<Real, Eigen::Dynamic, 1>, unsigned int>
  newton_method(const FunctionBase<Real> &fcn,
                const Solver &solver,
                const Eigen::Matrix<Real, Eigen::Dynamic, 1> &guess,
                const Real tol = 1e-6,
                const unsigned int max_iter = 100);



  template<typename Real, typename Solver>
  std::pair<Eigen::Matrix<Real, Eigen::Dynamic, 1>, unsigned int>
  newton_method(const FunctionBase<Real> &fcn,
                const Solver &solver,
                const Eigen::Matrix<Real, Eigen::Dynamic, 1> &guess,
                const Real tol,
                const unsigned int max_iter)
  {
    bool solution_not_found = true;
    unsigned int iter = 0;

    auto x0 = guess;
    auto x1 = x0;

    Eigen::Matrix<Real, Eigen::Dynamic, 1> d(guess.rows());

    while (solution_not_found && iter < max_iter)
    {
      /*
       * jacobian * (x1 - x0) = -f
       * --> jacobian * d = -f
       * --> x1 = x0 + d
       */
      auto f = fcn.value(x0);
      d = solver.solve(-f);
      x1 = x0 + d;

      // Check the residual of the function to see if x1 is close to the root.
      f = fcn.value(x1);
      Real divisor;
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



  /*
   * ===============================================================================================================
   * Base classes for SDIRK solvers and BDF solvers
   * ===============================================================================================================
   */

  /*
   * Singly diagonally implicit Runge-Kutta -- or SDIRK -- methods are methods designed to be used
   * with a modified Newton's method that uses the same Jacobian multiple times before updating it.
   * Consider a Runge-Kutta method's Butcher tableau and write the coefficients as a matrix.
   * In this matrix, anything on the upper triangular (including diagonal) part means the method is
   * implicit. "Diagonally" in SDIRK means the diagonal is non-zero but everything else in the upper
   * triangle is zero. "Singly" in SDIRK means all of the diagonal coefficients are the same. Each row
   * of this matrix corresponds to one nonlinear solve. Using the modified Newton's method, SDIRK methods
   * have the advantage of having the same Jacobian matrix for each of these nonlinear solves.
   *
   * This is easiest to see with an example:
   * Consider this second order SDIRK method
   *    1/4 | 1/4   0
   *    3/4 | 1/2   1/4
   *    ------------------
   *        | 1/2   1/2
   *
   * This can be solved by translating the Butcher tableau to
   *    k1 = f(t + h*1/4, x0 +h*1/4*k1)
   *    k2 = f(t + h*3/4, x0 + h*1/2*k1 + h*1/4*k2)
   *    x1 = x0 + h*1/2*k1 + h*1/2*k2
   * where f(...) represents the right-hand side of the ODE.
   *
   * Solving for k1 using the Newton method means
   *    J_F^{(n)}*(k1^{(n+1)} - k1^{(n)}) = -F^{(n)}
   * where the function F is defined
   *    F^{(n)} = k1^{(n)} - f(t + h*1/4, x0 +h*1/4*k1^{(n)})
   * and the chain rule means the Jacobian of F with respect to k1, J_F, is
   *    J_F^{(n)} = I - h*1/4*J_f(t + h*1/4, x0 +h*1/4*k1^{(n)})
   * where J_f is the Jacobian of the right-hand side of the ODE.
   * The advantage of SDIRK is that when solving for k2 we have
   *    J_F^{(n)} = I - h*1/4*J_f(t + h*3/4, x0 + h*1/2*k1 + h*1/4*k2).
   *
   * Computing a representation of J_F^-1 is the computationally expensive part of this process. Thus instead
   * of using Newton's method and having to compute the Jacobian many times, a modified Newton's method can be
   * used such that the Jacobian is computed infrequently. In this case, J_f remains constant for k1 and k2 within
   * a single time step and for every future time step -- and therefore J_F is constant because of the SDIRK restriction
   * that the diagonal coefficients are equal -- until some criteria for updating J_f is met.
   */

  template<int order, typename Real, typename Matrix>
  class StepperSDIRK : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(Model::Model<Real, Matrix> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0,
                                                        Real t, Real dt) override;

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Matrix> ode_system;
  };



  /*
   * Backward differentiation formula (BDF) methods are implicit linear multistep methods which use solutions from
   * previous time points to approximate the next time point and use the current time point for the derivative. The
   * first order BDF method needs one previous time step to calculate future time steps. The initial condition satisfies
   * this, so it can be used for the entire problem. The second order BDF requires two previous times, but at the start
   * of the problem only the initial condition is known. In order to preserve second order convergence, the first time
   * step must be done with a second order method which does not need previous time step -- e.g. 2nd order SDIRK -- and
   * then the remaining steps can then use the BDF method. For an n-th order BDF method, the first n-1 steps must be
   * done with an alternative method, and the BDF method can be used afterwards.
   */

  template<int order, typename Real, typename Matrix>
  class StepperBDF : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(Model::Model<Real, Matrix> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Matrix> ode_system;
    std::vector<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
  };



  /*
   * =============================================================================================================
   * Specializations for dense linear algebra
   * These specializations use dense matrices and solve linear equations using
   * a partial-pivoting LU decomposition.
   * =============================================================================================================
   */


  /*
   * =============================================================================================================
   * First order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * The first order SDIRK method used is Implicit Euler. This has Butcher tableau
   *    1 | 1
   *    -------
   *      | 1
   * which translates to
   *    k = f(t + h, x0 + h*k)
   *    x1 = x0 + h*k
   * To solve for x1 all that really needs to be done is
   *    k - f(t + h, x0 + h*k) = 0 --> solve for k using newton's method
   *    FIXME: add more details
   */

  template<typename Real>
  class StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;

  };



  template<typename Real>
  StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real,Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperSDIRK(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian
          = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * jac;

      solver = newton_jacobian.partialPivLu();
    }

    StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> > >(fcn, solver, guess);

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



  /*********************************** BDF ***********************************/

  /*
   * The first order BDF is Implicit Euler, just like the first order SDIRK method. Since
   * SDIRK is already implemented, this class will just use the SDIRK class for the calculations
   * but will have its own BDF class for compatibility.
   */

  template<typename Real>
  class StepperBDF<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

  private:
    StepperSDIRK<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> implicit_euler;
  };



  template<typename Real>
  StepperBDF<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperBDF(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : implicit_euler(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<1, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    return implicit_euler.step_forward(x0, t, dt);
  }




  /*
   * =============================================================================================================
   * Second order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * FIXME: this is not L-stable, but there is a L-stable, two-stage, 2nd order SDIRK method -- use that instead?
   * The second order SDIRK method used has Butcher tableau
   *    1/4 | 1/4   0
   *    3/4 | 1/2   1/4
   *    -----------------
   *        | 1/2   1/2
   * which translates to
   *    k1 = f(t + h*1/4, x0 + h*1/4*k1)
   *    k2 = f(t + h*3/4, x0 + h*1/2*k1 + h*1/4*k2)
   *
   *    FIXME: details of solution method
   */

  template<typename Real>
  class StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1>
    step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
  };



  template<typename Real>
  StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}


// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * 1./4 * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperSDIRK(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian
          = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * 1./4 * jac;

      solver = newton_jacobian.partialPivLu();
    }

    StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result_k1 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k1, solver, guess);
    auto k1 = newton_result_k1.first;

    StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k2(ode_system, t, dt, x0 + dt * 1./2 * k1);
    auto newton_result_k2 = newton_method<Real>(fcn_k2, solver, guess);
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



  /*********************************** BDF ***********************************/

  /*
   * The second order BDF method is defined
   *    y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n = 2/3 * h * f(t_{n+2}, y_{n+2})
   * so y_{n+2} is solved for using Newton's method to find the root of
   *    y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n - 2/3 * h * f(t_{n+2}, y_{n+2}) = 0
   */

  template<typename Real>
  class StepperBDF<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;


    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system, const Real t, const Real dt,
                     const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
    std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    StepperSDIRK<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> sdirk_stepper;
  };



  template<typename Real>
  StepperBDF<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols)
      : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    assert(prev_sols.size() == 2);
    return x - 4./3 * prev_sols[1] + 1./3 * prev_sols[0] - 2./3 * dt * ode_system.rhs(const_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1> &>(x));
  }



  template<typename Real>
  StepperBDF<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperBDF(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
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
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * 2/3 * jac;

        solver = newton_jacobian.partialPivLu();
      }

      StepperBDF<2, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
      const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
      auto newton_result = newton_method<Real,
          Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn, solver, guess);

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



  /*
   * =============================================================================================================
   * Third order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * The third order SDIRK method used has Butcher Tableau
   *          x | x                     0                     0
   *    (1+x)/2 | (1-x)/2               x                     0
   *          1 | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
   *    ---------------------------------------------------------
   *            | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
   * with x = 0.4358665215
   */

  template<typename Real>
  class StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
      const Real butcher_diag = 0.4358665215;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
    const Real butcher_diag = 0.4358665215;
  };



  template<typename Real>
  StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * butcher_diag * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperSDIRK(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * butcher_diag * jac;

      solver = newton_jacobian.partialPivLu();
    }

    StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result_k1 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k1, solver, guess);
    auto k1 = newton_result_k1.first;

    const Real k2_coeff_k1 = (1 - butcher_diag) / 2;
    StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k2(
        ode_system, t, dt, x0 + dt * k2_coeff_k1 * k1);
    auto newton_result_k2 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k2, solver, guess);
    auto k2 = newton_result_k2.first;

    const Real k3_coeff_k1 = -3 * butcher_diag * butcher_diag / 2 + 4 * butcher_diag - 1./4;
    const Real k3_coeff_k2 =  3 * butcher_diag * butcher_diag / 2 - 5 * butcher_diag + 5./4;
    StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction
        fcn_k3(ode_system, t, dt, x0 + dt * k3_coeff_k1 * k1 + dt * k3_coeff_k2 * k2);
    auto newton_result_k3 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k3, solver, guess);
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

    const Real k1_weight = -3 * butcher_diag * butcher_diag / 2 + 4 * butcher_diag - 1./4;
    const Real k2_weight =  3 * butcher_diag * butcher_diag / 2 - 5 * butcher_diag + 5./4;
    const Real k3_weight =  butcher_diag;

    return x0 + dt * k1_weight * k1 + dt * k2_weight * k2 + dt * k3_weight * k3;
  }



  /*********************************** BDF ***********************************/

  /*
   * The third order BDF method is defined
   *    y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n = 6/11 * h * f(t_{n+3}, y_{n+3})
   * so y_{n+3} is solved for using Newton's method to find the root of
   *    y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n - 6/11 * h * f(t_{n+3}, y_{n+3}) = 0
   */

  template<typename Real>
  class StepperBDF<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;


    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system, const Real t, const Real dt,
                     const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
    std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    StepperSDIRK<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> sdirk_stepper;
  };



  template<typename Real>
  StepperBDF<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols)
      : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    assert(prev_sols.size() == 3);
    return x - 18./11 * prev_sols[2] + 9./11 * prev_sols[1] - 2./11 * prev_sols[0]
           - 6./11 * dt * ode_system.rhs(const_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1> &>(x));
  }



  template<typename Real>
  StepperBDF<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperBDF(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
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
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * 6/11 * jac;

        solver = newton_jacobian.partialPivLu();
      }

      StepperBDF<3, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
      const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
      auto newton_result = newton_method<Real,
          Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn, solver, guess);

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



  /*
   * =============================================================================================================
   * Fourth order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * The fourth order SDIRK method used has Butcher Tableau
   *        x | x                 0                             0
   *      1/2 | 1/2 - x           x                             0
   *    1 - x | 2x                1 - 4x                        x
   *    -----------------------------------------------------------------------------
   *          | 1/[6*(1-2x)^2]    [3*(1-2x)^2-1]/[2*(1-2x)^2]   1/[6*(1-2x)^2]
   * with x a solution to the cubic equation
   *    x^3 - 3x^2/2 + x/2 - 1/24 = 0
   * The three roots are
   *    x1 ~= 0.128886400515720
   *    x2 ~= 0.302534578182651
   *    x3 ~= 1.06857902130163
   * x3 gives the best stability properties, so that one is used.
   */

  template<typename Real>
  class StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
      const Real butcher_diag = 1.06857902130163;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
    const Real butcher_diag = 1.06857902130163;
  };



  template<typename Real>
  StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}


// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * butcher_diag * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperSDIRK(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

      Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian
          = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * butcher_diag * jac;

      solver = newton_jacobian.partialPivLu();
    }

    StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result_k1 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k1, solver, guess);
    auto k1 = newton_result_k1.first;

    const Real k2_coeff_k1 = 1./2 - butcher_diag;
    StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k2(
        ode_system, t, dt, x0 + dt * k2_coeff_k1 * k1);
    auto newton_result_k2 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k2, solver, guess);
    auto k2 = newton_result_k2.first;

    const Real k3_coeff_k1 = 2 * butcher_diag;
    const Real k3_coeff_k2 =  1 - 4 * butcher_diag;
    StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn_k3(
        ode_system, t, dt, x0 + dt * k3_coeff_k1 * k1 + dt * k3_coeff_k2 * k2);
    auto newton_result_k3 = newton_method<Real,
        Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn_k3, solver, guess);
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

    const Real k1_weight = 1 / (6 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );
    const Real k2_weight =  (3 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) - 1)
                            / (3 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );
    const Real k3_weight =  1 / (6 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );

    return x0 + dt * k1_weight * k1 + dt * k2_weight * k2 + dt * k3_weight * k3;
  }

  /*********************************** BDF ***********************************/

  /*
   * The fourth order BDF method is defined
   *    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n = 12/25 * h * f(t_{n+4}, y_{n+4})
   * so y_{n+4} is solved for using Newton's method to find the root of
   *    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n - 12/25 * h * f(t_{n+4}, y_{n+4}) = 0
   */

  template<typename Real>
  class StepperBDF<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;


    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system, const Real t, const Real dt,
                     const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
      const Real t, dt;
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> ode_system;
    std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    StepperSDIRK<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> sdirk_stepper;
  };



  template<typename Real>
  StepperBDF<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system,
      const Real t,
      const Real dt,
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols)
      : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    assert(prev_sols.size() == 4);
    return x - 48./25 * prev_sols[3] + 36./25 * prev_sols[2] - 16./25 * prev_sols[1] + 3./25 * prev_sols[0]
           - 12./25 * dt * ode_system.rhs(const_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1> &>(x));
  }



  template<typename Real>
  StepperBDF<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::StepperBDF(
      const Model::Model<Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
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
        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> jac = ode_system.jacobian(x0);

        Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> newton_jacobian
            = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(jac.rows(), jac.cols()) - dt * 12/25 * jac;

        solver = newton_jacobian.partialPivLu();
      }

      StepperBDF<4, Real, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
      const auto guess = x0;
      auto newton_result = newton_method<Real,
          Eigen::PartialPivLU<Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>>>(fcn, solver, guess);

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




  /*
   * =============================================================================================================
   * Specializations for sparse linear algebra
   * These specializations use sparse matrices and solve linear equations using
   * the BiCGSTAB iterative solver with an incomplete LU decomposition as a preconditioner.
   * =============================================================================================================
   */


  /*
   * =============================================================================================================
   * First order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * The first order SDIRK method used is Implicit Euler. This has Butcher tableau
   *    1 | 1
   *    -------
   *      | 1
   * which translates to
   *    k = f(t + h, x0 + h*k)
   *    x1 = x0 + h*k
   * To solve for x1 all that really needs to be done is
   *    k - f(t + h, x0 + h*k) = 0 --> solve for k using newton's method
   *    FIXME: add more details
   */

  template<typename Real>
  class StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
    };

  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;

  };



  template<typename Real>
  StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real,Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperSDIRK(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

      newton_jacobian = -dt * jac;

      for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
        newton_jacobian.coeffRef(i,i) += 1;

      solver.compute(newton_jacobian);
    }

    StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real> > >(fcn, solver, guess);

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



  /*********************************** BDF ***********************************/

  /*
   * The first order BDF is Implicit Euler, just like the first order SDIRK method. Since
   * SDIRK is already implemented, this class will just use the SDIRK class for the calculations
   * but will have its own BDF class for compatibility.
   */

  template<typename Real>
  class StepperBDF<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

  private:
    StepperSDIRK<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> implicit_euler;
  };



  template<typename Real>
  StepperBDF<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperBDF(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : implicit_euler(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<1, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    return implicit_euler.step_forward(x0, t, dt);
  }




  /*
   * =============================================================================================================
   * Second order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * FIXME: this is not L-stable, but there is a L-stable, two-stage, 2nd order SDIRK method -- use that instead?
   * The second order SDIRK method used has Butcher tableau
   *    1/4 | 1/4   0
   *    3/4 | 1/2   1/4
   *    -----------------
   *        | 1/2   1/2
   * which translates to
   *    k1 = f(t + h*1/4, x0 + h*1/4*k1)
   *    k2 = f(t + h*3/4, x0 + h*1/2*k1 + h*1/4*k2)
   *
   *    FIXME: details of solution method
   */

  template<typename Real>
  class StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1>
    step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
    };

  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
  };



  template<typename Real>
  StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}


// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * 1./4 * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperSDIRK(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

      newton_jacobian = -dt * 1./4 * jac;

      for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
        newton_jacobian.coeffRef(i,i) += 1;

      solver.compute(newton_jacobian);
    }

    StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result_k1 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k1, solver, guess);
    auto k1 = newton_result_k1.first;

    StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k2(ode_system, t, dt, x0 + dt * 1./2 * k1);
    auto newton_result_k2 = newton_method<Real>(fcn_k2, solver, guess);
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



  /*********************************** BDF ***********************************/

  /*
   * The second order BDF method is defined
   *    y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n = 2/3 * h * f(t_{n+2}, y_{n+2})
   * so y_{n+2} is solved for using Newton's method to find the root of
   *    y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n - 2/3 * h * f(t_{n+2}, y_{n+2}) = 0
   */

  template<typename Real>
  class StepperBDF<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;


    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system, const Real t, const Real dt,
                     const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
    std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    StepperSDIRK<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> sdirk_stepper;
  };



  template<typename Real>
  StepperBDF<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols)
      : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    assert(prev_sols.size() == 2);
    return x - 4./3 * prev_sols[1] + 1./3 * prev_sols[0] - 2./3 * dt * ode_system.rhs(const_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1> &>(x));
  }



  template<typename Real>
  StepperBDF<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperBDF(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
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
        Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

        newton_jacobian = -dt * 2/3 * jac;

        for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
          newton_jacobian.coeffRef(i,i) += 1;

        solver.compute(newton_jacobian);
      }

      StepperBDF<2, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
      const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
      auto newton_result = newton_method<Real,
          Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn, solver, guess);

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



  /*
   * =============================================================================================================
   * Third order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * The third order SDIRK method used has Butcher Tableau
   *          x | x                     0                     0
   *    (1+x)/2 | (1-x)/2               x                     0
   *          1 | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
   *    ---------------------------------------------------------
   *            | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
   * with x = 0.4358665215
   */

  template<typename Real>
  class StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
      const Real butcher_diag = 0.4358665215;
    };

  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
    const Real butcher_diag = 0.4358665215;
  };



  template<typename Real>
  StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * butcher_diag * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperSDIRK(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

      newton_jacobian = -dt * butcher_diag * jac;

      for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
        newton_jacobian.coeffRef(i,i) += 1;

      solver.compute(newton_jacobian);
    }

    StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result_k1 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k1, solver, guess);
    auto k1 = newton_result_k1.first;

    const Real k2_coeff_k1 = (1 - butcher_diag) / 2;
    StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k2(
        ode_system, t, dt, x0 + dt * k2_coeff_k1 * k1);
    auto newton_result_k2 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k2, solver, guess);
    auto k2 = newton_result_k2.first;

    const Real k3_coeff_k1 = -3 * butcher_diag * butcher_diag / 2 + 4 * butcher_diag - 1./4;
    const Real k3_coeff_k2 =  3 * butcher_diag * butcher_diag / 2 - 5 * butcher_diag + 5./4;
    StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction
        fcn_k3(ode_system, t, dt, x0 + dt * k3_coeff_k1 * k1 + dt * k3_coeff_k2 * k2);
    auto newton_result_k3 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k3, solver, guess);
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

    const Real k1_weight = -3 * butcher_diag * butcher_diag / 2 + 4 * butcher_diag - 1./4;
    const Real k2_weight =  3 * butcher_diag * butcher_diag / 2 - 5 * butcher_diag + 5./4;
    const Real k3_weight =  butcher_diag;

    return x0 + dt * k1_weight * k1 + dt * k2_weight * k2 + dt * k3_weight * k3;
  }



  /*********************************** BDF ***********************************/

  /*
   * The third order BDF method is defined
   *    y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n = 6/11 * h * f(t_{n+3}, y_{n+3})
   * so y_{n+3} is solved for using Newton's method to find the root of
   *    y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n - 6/11 * h * f(t_{n+3}, y_{n+3}) = 0
   */

  template<typename Real>
  class StepperBDF<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;


    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system, const Real t, const Real dt,
                     const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
    std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    StepperSDIRK<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> sdirk_stepper;
  };



  template<typename Real>
  StepperBDF<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols)
      : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    assert(prev_sols.size() == 3);
    return x - 18./11 * prev_sols[2] + 9./11 * prev_sols[1] - 2./11 * prev_sols[0]
           - 6./11 * dt * ode_system.rhs(const_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1> &>(x));
  }



  template<typename Real>
  StepperBDF<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperBDF(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
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
        Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

        newton_jacobian = -dt * 6/11 * jac;

        for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
          newton_jacobian.coeffRef(i,i) += 1;

        solver.compute(newton_jacobian);
      }

      StepperBDF<3, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
      const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
      auto newton_result = newton_method<Real,
          Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn, solver, guess);

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



  /*
   * =============================================================================================================
   * Fourth order methods
   * =============================================================================================================
   */

  /********************************** SDIRK **********************************/

  /*
   * The fourth order SDIRK method used has Butcher Tableau
   *        x | x                 0                             0
   *      1/2 | 1/2 - x           x                             0
   *    1 - x | 2x                1 - 4x                        x
   *    -----------------------------------------------------------------------------
   *          | 1/[6*(1-2x)^2]    [3*(1-2x)^2-1]/[2*(1-2x)^2]   1/[6*(1-2x)^2]
   * with x a solution to the cubic equation
   *    x^3 - 3x^2/2 + x/2 - 1/24 = 0
   * The three roots are
   *    x1 ~= 0.128886400515720
   *    x2 ~= 0.302534578182651
   *    x3 ~= 1.06857902130163
   * x3 gives the best stability properties, so that one is used.
   */

  template<typename Real>
  class StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperSDIRK(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;

    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
                     const Real t, const Real dt, const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> x0;
      const Real butcher_diag = 1.06857902130163;
    };

  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
    const Real butcher_diag = 1.06857902130163;
  };



  template<typename Real>
  StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0)
      : ode_system(ode_system), t(t), dt(dt), x0(x0)
  {}


// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> y = x0 + dt * butcher_diag * x;
    return x - ode_system.rhs(y);
  }



  template<typename Real>
  StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperSDIRK(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system)
  {}



// FIXME: add comments
  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
  {
    if (update_jacobian)
    {
      Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

      newton_jacobian = -dt * butcher_diag * jac;

      for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
        newton_jacobian.coeffRef(i,i) += 1;

      solver.compute(newton_jacobian);
    }

    StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k1(ode_system, t, dt, x0);
    const auto guess = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(x0.rows());
    auto newton_result_k1 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k1, solver, guess);
    auto k1 = newton_result_k1.first;

    const Real k2_coeff_k1 = 1./2 - butcher_diag;
    StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k2(
        ode_system, t, dt, x0 + dt * k2_coeff_k1 * k1);
    auto newton_result_k2 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k2, solver, guess);
    auto k2 = newton_result_k2.first;

    const Real k3_coeff_k1 = 2 * butcher_diag;
    const Real k3_coeff_k2 =  1 - 4 * butcher_diag;
    StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn_k3(
        ode_system, t, dt, x0 + dt * k3_coeff_k1 * k1 + dt * k3_coeff_k2 * k2);
    auto newton_result_k3 = newton_method<Real,
        Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn_k3, solver, guess);
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

    const Real k1_weight = 1 / (6 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );
    const Real k2_weight =  (3 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) - 1)
                            / (3 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );
    const Real k3_weight =  1 / (6 * (1 - 2 * butcher_diag) * (1 - 2 * butcher_diag) );

    return x0 + dt * k1_weight * k1 + dt * k2_weight * k2 + dt * k3_weight * k3;
  }

  /*********************************** BDF ***********************************/

  /*
   * The fourth order BDF method is defined
   *    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n = 12/25 * h * f(t_{n+4}, y_{n+4})
   * so y_{n+4} is solved for using Newton's method to find the root of
   *    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n - 12/25 * h * f(t_{n+4}, y_{n+4}) = 0
   */

  template<typename Real>
  class StepperBDF<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> : public StepperBase<Real>
  {
  public:
    explicit StepperBDF(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system);

    Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) override;


    class NewtonFunction : public FunctionBase<Real>
    {
    public:
      NewtonFunction(const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system, const Real t, const Real dt,
                     const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols);

      Eigen::Matrix<Real, Eigen::Dynamic, 1> value(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const override;

    private:
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
      const Real t, dt;
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::SparseMatrix<Real, Eigen::RowMajor> newton_jacobian;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>> solver;
    unsigned int num_iter_new_jac;
    Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> ode_system;
    std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> prev_sols;
    StepperSDIRK<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> sdirk_stepper;
  };



  template<typename Real>
  StepperBDF<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::NewtonFunction(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system,
      const Real t,
      const Real dt,
      const std::deque<Eigen::Matrix<Real, Eigen::Dynamic, 1>> &prev_sols)
      : ode_system(ode_system), t(t), dt(dt), prev_sols(prev_sols)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction::value(
      const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x) const
  {
    assert(prev_sols.size() == 4);
    return x - 48./25 * prev_sols[3] + 36./25 * prev_sols[2] - 16./25 * prev_sols[1] + 3./25 * prev_sols[0]
           - 12./25 * dt * ode_system.rhs(const_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1> &>(x));
  }



  template<typename Real>
  StepperBDF<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::StepperBDF(
      const Model::Model<Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>> &ode_system)
      : update_jacobian(true), num_iter_new_jac(0), ode_system(ode_system), sdirk_stepper(ode_system)
  {}



  template<typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  StepperBDF<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::step_forward(
      Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt)
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
        Eigen::SparseMatrix<Real, Eigen::RowMajor> jac = ode_system.jacobian(x0);

        newton_jacobian = -dt * 12/25 * jac;

        for (unsigned int i=0; i<newton_jacobian.rows(); ++i)
          newton_jacobian.coeffRef(i,i) += 1;

        solver.compute(newton_jacobian);
      }

      StepperBDF<4, Real, Eigen::SparseMatrix<Real, Eigen::RowMajor>>::NewtonFunction fcn(ode_system, t, dt, prev_sols);
      const auto guess = x0;
      auto newton_result = newton_method<Real,
          Eigen::BiCGSTAB<Eigen::SparseMatrix<Real, Eigen::RowMajor>, Eigen::IncompleteLUT<Real>>>(fcn, solver, guess);

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
}

#endif //MEPBM_ODE_SOLVER_H