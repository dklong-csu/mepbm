#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "src/models.h"
#include "sundials_solvers.h"
#include "nvector_eigen.h"
#include "sunmatrix_eigen.h"
#include "linear_solver_eigen.h"
#include "ode_solver.h"

#include <chrono>
#include <fstream>

using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using SolverType = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<Real> >;



int main()
{
  /******************************
   * Create a model of the 3 step
   *****************************/
  const unsigned int max_size = 2500;
  const unsigned int nucleation_order = 3;
  const Real solvent = 11.3;
  const unsigned int conserved_size = 1;

  const unsigned int A_index = 0;
  const unsigned int As_index = 1;
  const unsigned int POM_index = 2;
  const unsigned int nucleation_index = 3;

  const Real kf = 3.3e-3;
  const Real kb = 6.67e3;
  const Real k1 = 1.26e5;
  const Real k2 = 2.57e5;
  const Real k3 = 6.29e3;
  const Real cutoff = 107;

  const Real start_time = 0.;
  const Real end_time = 4.838;

  // Nucleation
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> nucleation
  = std::make_shared<Model::TermolecularNucleation<Real, Matrix>>(A_index, As_index, POM_index,nucleation_index,
      kf, kb, k1, solvent);

  // Small Growth
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> small_growth
  = std::make_shared<Model::Growth<Real, Matrix>>(A_index, nucleation_order, cutoff, max_size,
      POM_index, conserved_size, k2, nucleation_order);

  // Large Growth
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> large_growth
  = std::make_shared<Model::Growth<Real, Matrix>>(A_index, cutoff+1, max_size, max_size,
      POM_index, conserved_size, k3, cutoff+1);

  // Create Model
  Model::Model<Real, Matrix> three_step_alt(nucleation_order, max_size);
  three_step_alt.add_rhs_contribution(nucleation);
  three_step_alt.add_rhs_contribution(small_growth);
  three_step_alt.add_rhs_contribution(large_growth);

  // set up initial condition
  Vector ic = Vector::Zero(max_size+1);
  ic(0) = 0.0012;

  N_Vector ic_sundials = create_eigen_nvector<Vector>(ic.size());
  auto ic_sundials_vec = static_cast<Vector*>(ic_sundials->content);
  (*ic_sundials_vec)(0) = ic(0);

  std::cout << "3-step Mechanism performance:" << std::endl;
  std::cout << "-----------------------------" << std::endl;



  /*******************************************
   * Solve using Eigen as an accuracy baseline
   ******************************************/
  ODE::StepperBDF<4, Real, Matrix> stepper_acc(three_step_alt);
  auto begin = std::chrono::steady_clock::now();

  auto eigen_accurate_solution = ODE::solve_ode(stepper_acc, ic, start_time, end_time, 5e-5);

  auto end = std::chrono::steady_clock::now();
  std::cout << "Accurate Eigen sparse solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds"
            << std::endl;

  std::ofstream file_acc("./acc_3step.m");

  file_acc << "sol_acc_3 = ["
           << std::endl
           << eigen_accurate_solution
           << "];"
           << std::endl;
  file_acc.close();


  /*******************************************
   * Solve using Eigen as a time baseline
   ******************************************/
  ODE::StepperBDF<4, Real, Matrix> stepper_fast(three_step_alt);
  begin = std::chrono::steady_clock::now();

  auto eigen_fast_solution = ODE::solve_ode(stepper_fast, ic, start_time, end_time, 5e-3);

  end = std::chrono::steady_clock::now();
  std::cout << "Fast Eigen sparse solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds"
            << std::endl;

  std::ofstream file_fast("./fast_3step.m");

  file_fast << "sol_fast_3 = ["
            << std::endl
            << eigen_fast_solution
            << "];"
            << std::endl;
  file_fast.close();



  /*********************************
   * Solve using SUNDIALS
   ********************************/

  // Problem constants
  const realtype abs_tol = 1e-13;
  const realtype rel_tol = 1e-6;
  const unsigned int dim = ic.size();

  // Vector and Matrix templates for CVODE to use
  auto vector_template = create_eigen_nvector<Vector>(dim);
  auto matrix_template = create_eigen_sunmatrix<Matrix>(dim,dim);

  // Linear solver for CVODE to use
  auto linear_solver = create_eigen_iterative_linear_solver<Matrix, realtype, SolverType>();

  // Settings for CVODE
  sundials::CVodeParameters<realtype> param(start_time,
                                            end_time,
                                            abs_tol,
                                            rel_tol,
                                            CV_BDF);

  // Setup the CVODE solver
  sundials::CVodeSolver<Matrix, realtype, SolverType> ode_solver(param,
                                                     three_step_alt,
                                                     ic_sundials,
                                                     vector_template,
                                                     matrix_template,
                                                     linear_solver);

  // Solve the ODE
  auto solution = create_eigen_nvector<Vector>(dim);

  begin = std::chrono::steady_clock::now();

  ode_solver.solve_ode(solution, end_time);

  end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds"
            << std::endl;

  auto sundials_solution = *static_cast<Vector*>(solution->content);

  std::ofstream file_sun("./sun_3step.m");

  file_sun << "sol_sun_3 = ["
           << std::endl
           << sundials_solution
           << "];"
           << std::endl;
  file_sun.close();

}