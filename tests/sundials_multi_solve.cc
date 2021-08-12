#include "sundials_solvers.h"
#include "linear_solver_eigen.h"
#include <eigen3/Eigen/Sparse>
#include "models.h"
#include <iostream>
#include <iomanip>
#include <vector>



using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<realtype, Eigen::RowMajor>;

class SimpleOde : public Model::RightHandSideContribution<realtype, Matrix>
    {
  void add_contribution_to_rhs(const Vector &x, Vector &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Vector &x, Matrix &jacobi)
  {
    for (unsigned int i=0; i<jacobi.rows(); ++i)
    {
      jacobi.coeffRef(i,i) -= 10;
    }

    jacobi.makeCompressed();
  }

  void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<realtype>> &triplet_list) {}

  void update_num_nonzero(unsigned int &num_nonzero) {}
    };



int main ()
{
  // Setup problem constants
  auto initial_condition = create_eigen_nvector<Vector>(1);
  auto ic_vec = static_cast<Vector*>(initial_condition->content);
  *ic_vec = Vector::Ones(1);

  const realtype start_time = 0.;
  const realtype end_time = 1.;
  const realtype abs_tol = 1e-7;
  const realtype rel_tol = 1e-7;

  // Create ode model
  std::shared_ptr<Model::RightHandSideContribution<realtype, Matrix>> my_ode
  = std::make_shared<SimpleOde>();
  Model::Model<realtype, Matrix> ode_system(0, 0);
  ode_system.add_rhs_contribution(my_ode);

  // Create templates of vectors and matrices
  auto vector_template = initial_condition->ops->nvclone(initial_condition);
  auto matrix_template = create_eigen_sunmatrix<Matrix>(1,1);

  // Create the linear solver
  auto linear_solver = create_eigen_linear_solver<Matrix, realtype>();

  // Create settings for CVODE solver
  sundials::CVodeParameters<realtype> param(start_time,
                                            end_time,
                                            abs_tol,
                                            rel_tol,
                                            CV_BDF);

  // Setup CVODE solver
  sundials::CVodeSolver<Matrix, realtype> ode_solver(param,
                                                     ode_system,
                                                     initial_condition,
                                                     vector_template,
                                                     matrix_template,
                                                     linear_solver);

  // Solve the ODE
  std::vector<N_Vector> solutions;
  std::vector<realtype> times = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1};
  ode_solver.solve_ode_incrementally(solutions, times);

  // The answers should be close to exp(-10*times[i])
  for (auto vec : solutions)
  {
    auto sol = static_cast<Vector*>(vec->content);
    std::cout << std::setprecision(20) << std::fixed << *sol << std::endl;
  }
}