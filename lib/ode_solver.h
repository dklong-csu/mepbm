#ifndef MEPBM_ODE_SOLVER_H
#define MEPBM_ODE_SOLVER_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
//#include <eigen3/unsupported/Eigen/IterativeSolvers>
#include <cmath>
#include <deque>
#include <vector>
#include "src/models.h"
#include <iostream>


namespace ODE
{
  /// Base class for describing how an ODE solver should step forward in time
  template<typename Real>
  class StepperBase
  {
  public:
    virtual Eigen::Matrix<Real, Eigen::Dynamic, 1> step_forward(Eigen::Matrix<Real, Eigen::Dynamic, 1> &x0, Real t, Real dt) = 0;
  };



  /// A function that solves an ODE based on the time stepping rules provided
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



  /// A function that performs a modified Newton's method to find the zero of a function.
  /// The Newton's method is modified by infrequently calculating the Jacobian to save on computational expense.
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



  /// Base class for singly diagonally implicit Runge-Kutta (SDIRK) time steppers
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



  /// Base class for backward differentiation formula (BDF) time steppers
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

  /// First order SDIRK time stepper, also known as Implicit Euler
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



  /// First order BDF time stepper, also known as Implicit Euler
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

  /// Second order SDIRK time stepper with Butcher tableau
  /// 1/4 | 1/4   0
  /// 3/4 | 1/2   1/4
  /// -----------------
  ///     | 1/2   1/2
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



  /// Second order BDF time stepper defined by
  /// y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n = 2/3 * h * f(t_{n+2}, y_{n+2})
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

  /// Third order SDIRK time stepper with Butcher Tableau
  ///         x | x                     0                     0
  ///   (1+x)/2 | (1-x)/2               x                     0
  ///         1 | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
  /// ---------------------------------------------------------
  ///           | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
  /// with x = 0.4358665215
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



  /// Third order BDF time stepper defined as
  /// y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n = 6/11 * h * f(t_{n+3}, y_{n+3})
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

   /// The fourth order SDIRK method used has Butcher Tableau
   ///        x | x                 0                             0
   ///      1/2 | 1/2 - x           x                             0
   ///    1 - x | 2x                1 - 4x                        x
   ///    -----------------------------------------------------------------------------
   ///          | 1/[6*(1-2x)^2]    [3*(1-2x)^2-1]/[2*(1-2x)^2]   1/[6*(1-2x)^2]
   /// with x a solution to the cubic equation
   ///    x^3 - 3x^2/2 + x/2 - 1/24 = 0
   /// The three roots are
   ///    x1 ~= 0.128886400515720
   ///    x2 ~= 0.302534578182651
   ///    x3 ~= 1.06857902130163
   /// x3 gives the best stability properties, so that one is used.
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



  /// The fourth order BDF method is defined
  ///    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n = 12/25 * h * f(t_{n+4}, y_{n+4})
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


  /// First order SDIRK time stepper, also known as Implicit Euler
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



  /// First order BDF time stepper, also known as Implicit Euler
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

  /// Second order SDIRK time stepper with Butcher tableau
  /// 1/4 | 1/4   0
  /// 3/4 | 1/2   1/4
  /// -----------------
  ///     | 1/2   1/2
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



  /// Second order BDF time stepper defined by
  /// y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n = 2/3 * h * f(t_{n+2}, y_{n+2})
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

  /// Third order SDIRK time stepper with Butcher Tableau
  ///         x | x                     0                     0
  ///   (1+x)/2 | (1-x)/2               x                     0
  ///         1 | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
  /// ---------------------------------------------------------
  ///           | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
  /// with x = 0.4358665215
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



  /// Third order BDF time stepper defined as
  /// y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n = 6/11 * h * f(t_{n+3}, y_{n+3})
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

  /// The fourth order SDIRK method used has Butcher Tableau
  ///        x | x                 0                             0
  ///      1/2 | 1/2 - x           x                             0
  ///    1 - x | 2x                1 - 4x                        x
  ///    -----------------------------------------------------------------------------
  ///          | 1/[6*(1-2x)^2]    [3*(1-2x)^2-1]/[2*(1-2x)^2]   1/[6*(1-2x)^2]
  /// with x a solution to the cubic equation
  ///    x^3 - 3x^2/2 + x/2 - 1/24 = 0
  /// The three roots are
  ///    x1 ~= 0.128886400515720
  ///    x2 ~= 0.302534578182651
  ///    x3 ~= 1.06857902130163
  /// x3 gives the best stability properties, so that one is used.
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



  /// The fourth order BDF method is defined
  ///    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n = 12/25 * h * f(t_{n+4}, y_{n+4})
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