#include <iostream>
#include <iomanip>
#include <ode_solver.h>
#include "models.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "sundials_solvers.h"
#include <nvector/nvector_serial.h>

#include "limits"


using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;


/*********************************************
 * Data for Eigen implementation of ODE solver
 ********************************************/

class SimpleOde : public Model::RightHandSideContribution<Real, Matrix>
{
  void add_contribution_to_rhs(const Vector &x, Vector &rhs)
  {
    rhs += -10 * x;
  }

  void add_contribution_to_jacobian(const Vector &x, Matrix &jacobi)
  {
    jacobi += -10 * Matrix::Identity(x.rows(), x.rows());
  }

  void add_nonzero_to_jacobian(std::vector<Eigen::Triplet<Real>> &triplet_list) {}

  void update_num_nonzero(unsigned int &num_nonzero) {}
};

/************************************************
 * Data for SUNDIALS implementation of ODE Solver
 ***********************************************/

int main()
{
  N_Vector ic = N_VNew_Serial(1);
  auto ic_data = N_VGetArrayPointer(ic);
  ic_data[0] = 1.;

  const Real start_time = 0.;
  const Real end_time = 1.;

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> my_ode
      = std::make_shared<SimpleOde>();
  Model::Model<Real, Matrix> ode_system(0, 0);
  ode_system.add_rhs_contribution(my_ode);

  const sundials::CVodeParameters<Real> cvode_settings(sundials::SPARSE, sundials::SPFGMR,
                                                       start_time, end_time, false, true,
                                                       true, 1e-7, 1e-7,
                                                       std::numeric_limits<Real>::epsilon(), CSC_MAT);
  sundials::CVodeSolver<Matrix, Real> cvode_solver(cvode_settings, ode_system, ic);

  N_Vector solution = N_VNew_Serial(1);
  cvode_solver.solve_ode(solution, end_time);

  // The answer should be close to exp(-10)
  auto solution_data = N_VGetArrayPointer(solution);

  std::cout << std::setprecision(20) << std::fixed << solution_data[0];
}
