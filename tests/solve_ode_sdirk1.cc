#include <iostream>
#include <ode_solver.h>
#include "models.h"
#include <eigen3/Eigen/Dense>



class SimpleOde : public Model::RightHandSideContribution
{
  void add_contribution_to_rhs(const Eigen::VectorXd &x, Eigen::VectorXd &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Eigen::VectorXd &x, Eigen::MatrixXd &jacobi)
  {
    jacobi += -10 * Eigen::MatrixXd::Identity(x.rows(), x.rows());
  }
};

int main()
{
  const Eigen::VectorXd ic = Eigen::VectorXd::Ones(1);
  const double start_time = 0.;
  const double end_time = 1.;
  const double dt = 1e-4;

  std::shared_ptr<Model::RightHandSideContribution> my_ode
    = std::make_shared<SimpleOde>();
  Model::Model ode_system(0, 0);
  ode_system.add_rhs_contribution(my_ode);
  ODE::StepperSDIRK<1> stepper(ode_system);
  auto sol = ODE::solve_ode(stepper, ic, start_time, end_time, dt);

  // The answer should be close to exp(-10)
  std::cout << sol;
}