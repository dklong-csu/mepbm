#include <iostream>
#include <iomanip>
#include <ode_solver.h>
#include "models.h"
#include <eigen3/Eigen/Dense>


using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

class SimpleOde : public Model::RightHandSideContribution<Real, Matrix>
{
  void add_contribution_to_rhs(const Vector &x, Vector &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Vector &x, Matrix &jacobi)
  {
    for (unsigned int i=0; i<jacobi.rows(); ++i)
      jacobi.coeffRef(i,i) += -10 ;
  }
};

int main()
{
  const Vector ic = Vector::Ones(1);
  const Real start_time = 0.;
  const Real end_time = 1.;
  const Real dt = 1e-5;

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
      = std::make_shared<SimpleOde>();
  Model::Model<Real, Matrix> ode_system(0, 0);
  ode_system.add_rhs_contribution(my_ode);
  ODE::StepperBDF<1, Real, Matrix> stepper(ode_system);
  auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

  // The answer should be exactly the same as in the solve_ode_sdirk1 test.
  std::cout << std::setprecision(20) << std::fixed << sol;
}