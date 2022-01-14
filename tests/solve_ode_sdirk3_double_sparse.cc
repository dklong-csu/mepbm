#include <iostream>
#include <iomanip>
#include <ode_solver.h>
#include "src/models.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>




using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
const int order = 3;



class SimpleOde : public Model::RightHandSideContribution<Real, Matrix>
{
  void add_contribution_to_rhs(const Vector &x, Vector &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Vector &x, Matrix &jacobi)
  {
    jacobi.coeffRef(0, 0) -= 10;
    jacobi.makeCompressed();
  }

  void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list)
  {
    triplet_list.push_back(Eigen::Triplet<Real>(0, 0));
  }

  void update_num_nonzero(unsigned int &num_nonzero)
  {
    num_nonzero += 1;
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
  ODE::StepperSDIRK<order, Real, Matrix> stepper(ode_system);
  auto sol = ODE::solve_ode<Real>(stepper, ic, start_time, end_time, dt);

  // The answer should be close to exp(-10)
  std::cout << std::setprecision(20) << std::fixed << sol;
}