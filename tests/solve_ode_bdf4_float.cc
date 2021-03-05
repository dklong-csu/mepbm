#include <iostream>
#include <iomanip>
#include <ode_solver.h>
#include "models.h"
#include <eigen3/Eigen/Dense>



using Real = float;



class SimpleOde : public Model::RightHandSideContribution<Real>
{
  void add_contribution_to_rhs(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x, Eigen::Matrix<Real, Eigen::Dynamic, 1> &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &x, Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> &jacobi)
  {
    jacobi += -10 * Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(x.rows(), x.rows());
  }
};

int main()
{
  const Eigen::Matrix<Real, Eigen::Dynamic, 1> ic = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Ones(1);
  const Real start_time = 0.;
  const Real end_time = 1.;
  const Real dt = 1e-5;

  std::shared_ptr<Model::RightHandSideContribution<Real>> my_ode
      = std::make_shared<SimpleOde>();
  Model::Model<Real> ode_system(0, 0);
  ode_system.add_rhs_contribution(my_ode);
  ODE::StepperBDF<4, Real> stepper(ode_system);
  auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

  // The answer should be close to exp(-10)
  std::cout << std::setprecision(20) << std::fixed << sol;
}