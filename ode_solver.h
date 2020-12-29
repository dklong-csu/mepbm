#ifndef MEPBM_ODE_SOLVER_H
#define MEPBM_ODE_SOLVER_H

#include <eigen3/Eigen/Dense>
#include <cmath>


namespace ODE
{
  /***************** Classes -- Implementations in .cpp *****************/

  // FIXME: This is a placeholder for the moment. Eventually I will integrate this class
  // FIXME: with the objects in models.h but that first requires a bunch of things to be
  // FIXME: converted from Boost data types to Eigen data types.
  class OdeSystem
  {
  public:
    Eigen::VectorXd compute_rhs(double t, const Eigen::VectorXd &x) const;

    Eigen::MatrixXd compute_jacobian(double t, const Eigen::VectorXd &x) const;

    Eigen::PartialPivLU<Eigen::MatrixXd> jacobian_solver;
  };



  /*
   * Base class for ODE time stepper
   */
  class StepperBase
  {
  public:
    virtual Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt) const = 0;
  };



  /*
   * A function which solves an ODE. This is intended to just lay out the basic framework for an ODE solve.
   * The stepper -- some derived class of StepperBase -- is intended to do most of the work of the ODE solve
   * within its `step_forward` method. This function simply facilitates repeatedly using `step_forward` to
   * go from the initial time to the final time.
   */
  Eigen::VectorXd solve_ode(StepperBase &stepper, Eigen::VectorXd &ic, double t_start, double t_end, double dt);



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
    explicit StepperSDIRK(OdeSystem &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt);

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    OdeSystem ode_system;
  };



  /********************************** SDIRK Specializations **********************************/

  /*
   * The first order SDIRK method used is Implict Euler. This has Butcher tableau
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
    explicit StepperSDIRK(OdeSystem &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt);

    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const OdeSystem &ode_system, const double t, const double dt, const Eigen::VectorXd &x0);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const OdeSystem ode_system;
      const double t, dt;
      const Eigen::VectorXd x0;
    };

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    OdeSystem ode_system;

  };



  /*
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
    explicit StepperSDIRK(OdeSystem &ode_system);

    Eigen::VectorXd step_forward(Eigen::VectorXd &x0, double t, double dt);

  private:
    bool update_jacobian;
    unsigned int num_iter_new_jac;
    OdeSystem ode_system;

    class NewtonFunction : public FunctionBase
    {
    public:
      NewtonFunction(const OdeSystem &ode_system, const double t, const double dt, const Eigen::VectorXd &x0);

      Eigen::VectorXd value(const Eigen::VectorXd &x) const override;

    private:
      const OdeSystem ode_system;
      const double t, dt;
      const Eigen::VectorXd x0;
    };
  };



}

#endif //MEPBM_ODE_SOLVER_H