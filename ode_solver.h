#ifndef MEPBM_ODE_SOLVER_H
#define MEPBM_ODE_SOLVER_H

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <deque>
#include "models.h"


namespace ODE
{
  /***************** Classes -- Implementations in .cpp *****************/

  /*
   * Base class for ODE time stepper
   */
  class StepperBase
  {
  public:
    virtual Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) = 0;
  };



  /*
   * A function which solves an ODE. This is intended to just lay out the basic framework for an ODE solve.
   * The stepper -- some derived class of StepperBase -- is intended to do most of the work of the ODE solve
   * within its `step_forward` method. This function simply facilitates repeatedly using `step_forward` to
   * go from the initial time to the final time.
   */
  Eigen::VectorXd solve_ode(StepperBase &stepper, const Eigen::VectorXd &ic, const double t_start,
                            const double t_end, double dt);



  /*
   * A base class to represent a function. This simply has a rule for returning the value of the function.
   */
  class FunctionBase
  {
  public:
    virtual Eigen::VectorXd value(const Eigen::VectorXd &x) const = 0;
  };

  /*
   * A function which performs Newton's method to find the root of a nonlinear equation.
   * Technically, this is a modification where the Jacobian is taken as a constant for the
   * entire process. This is to save time as computing a decomposition of a Jacobian is more
   * expensive than solving with a precomputed decomposition (e.g. LU decomposition).
   *
   * This is a modified Newton's method where the Jacobian is held constant for the nonlinear solve.
  */
  std::pair<Eigen::VectorXd, unsigned int> newton_method(const FunctionBase &fcn, const Eigen::PartialPivLU<Eigen::MatrixXd> &jac,
                                                         const Eigen::VectorXd &guess, const double tol = 1e-6,
                                                         const unsigned int max_iter = 100);


  /********************************** Templates -- Implementation in this file **********************************/

  /********************************** SDIRK Methods **********************************/

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

  template<int order>
  class StepperSDIRK : public StepperBase
  {
  public:
    explicit StepperSDIRK(Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
  };



  /********************************** SDIRK Specializations **********************************/

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

  template<>
  class StepperSDIRK<1> : public StepperBase
  {
  public:
    explicit StepperSDIRK(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt, const Eigen::VectorXd &x0);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const Eigen::VectorXd x0;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    const Model::Model ode_system;

  };



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

  template<>
  class StepperSDIRK<2> : public StepperBase
  {
  public:
    explicit StepperSDIRK(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt, const Eigen::VectorXd &x0);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const Eigen::VectorXd x0;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
  };


  /*
   * The third order SDIRK method used has Butcher Tableau
   *          x | x                     0                     0
   *    (1+x)/2 | (1-x)/2               x                     0
   *          1 | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
   *    ---------------------------------------------------------
   *            | -3x^2/2 + 4x - 1/4    3x^2/2 - 5x + 5/4     x
   * with x = 0.4358665215
   */

  template<>
  class StepperSDIRK<3> : public StepperBase {
  public:
    explicit StepperSDIRK(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

    class NewtonFunction : public FunctionBase {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt, const Eigen::VectorXd &x0);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const Eigen::VectorXd x0;
      const double butcher_diag = 0.4358665215;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
    const double butcher_diag = 0.4358665215;
  };

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

  template<>
  class StepperSDIRK<4> : public StepperBase
  {
  public:
    explicit StepperSDIRK(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt, const Eigen::VectorXd &x0);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const Eigen::VectorXd x0;
      const double butcher_diag = 1.06857902130163;
    };

  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
    const double butcher_diag = 1.06857902130163;
  };



  /********************************** BDF Methods **********************************/

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

  template<int order>
  class StepperBDF : public StepperBase
  {
  public:
    explicit StepperBDF(Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
    std::vector<Eigen::VectorXd> prev_sols;
  };



  /********************************** BDF Specializations **********************************/

  /*
   * The first order BDF is Implicit Euler, just like the first order SDIRK method. Since
   * SDIRK is already implemented, this class will just use the SDIRK class for the calculations
   * but will have its own BDF class for compatibility.
   */

  template<>
  class StepperBDF<1> : public StepperBase
  {
  public:
    explicit StepperBDF(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;

  private:
    StepperSDIRK<1> implicit_euler;
  };



  /*
   * The second order BDF method is defined
   *    y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n = 2/3 * h * f(t_{n+2}, y_{n+2})
   * so y_{n+2} is solved for using Newton's method to find the root of
   *    y_{n+2} - 4/3 * y_{n+1} + 1/3 * y_n - 2/3 * h * f(t_{n+2}, y_{n+2}) = 0
   */

  template<>
  class StepperBDF<2> : public StepperBase
  {
  public:
    explicit StepperBDF(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;


    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                     const std::deque<Eigen::VectorXd> &prev_sols);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const std::deque<Eigen::VectorXd> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
    std::deque<Eigen::VectorXd> prev_sols;
    StepperSDIRK<2> sdirk_stepper;
  };



  /*
   * The third order BDF method is defined
   *    y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n = 6/11 * h * f(t_{n+3}, y_{n+3})
   * so y_{n+3} is solved for using Newton's method to find the root of
   *    y_{n+3} - 18/11 * y_{n+2} + 9/11 * y_{n+1} - 2/11 * y_n - 6/11 * h * f(t_{n+3}, y_{n+3}) = 0
   */

  template<>
  class StepperBDF<3> : public StepperBase
  {
  public:
    explicit StepperBDF(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;


    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                     const std::deque<Eigen::VectorXd> &prev_sols);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const std::deque<Eigen::VectorXd> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
    std::deque<Eigen::VectorXd> prev_sols;
    StepperSDIRK<3> sdirk_stepper;
  };



  /*
   * The fourth order BDF method is defined
   *    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n = 12/25 * h * f(t_{n+4}, y_{n+4})
   * so y_{n+4} is solved for using Newton's method to find the root of
   *    y_{n+4} - 48/25 * y_{n+3} + 36/25 * y_{n+2} - 16/25 * y_{n+1} + 3/25 * y_n - 12/25 * h * f(t_{n+4}, y_{n+4}) = 0
   */

  template<>
  class StepperBDF<4> : public StepperBase
  {
  public:
    explicit StepperBDF(const Model::Model &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) override;


    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const Model::Model &ode_system, const double t, const double dt,
                     const std::deque<Eigen::VectorXd> &prev_sols);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const Model::Model ode_system;
      const double t, dt;
      const std::deque<Eigen::VectorXd> prev_sols;
    };


  private:
    bool update_jacobian;
    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
    unsigned int num_iter_new_jac;
    Model::Model ode_system;
    std::deque<Eigen::VectorXd> prev_sols;
    StepperSDIRK<4> sdirk_stepper;
  };
}

#endif //MEPBM_ODE_SOLVER_H