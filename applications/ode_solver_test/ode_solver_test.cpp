#include <iostream>
#include <eigen3/Eigen/Sparse>
#include "models.h"
#include "sundials_solvers.h"
#include <sundials/sundials_nvector.h>
#include "ode_solver.h"

#include <chrono>

using Real = double;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;





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

  const Real kf = 3.6e-2;
  const Real kb = 7.27e4;
  const Real k1 = 6.40e4;
  const Real k2 = 1.61e4;
  const Real k3 = 5.45e3;
  const Real cutoff = 265;

  const Real start_time = 0.;
  const Real end_time = 0.001;

  // Nucleation
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> nucleation
      = std::make_shared<Model::TermolecularNucleation<Real, Matrix>>(A_index, As_index, POM_index,nucleation_index,
                                                                      kf, kb, k1, solvent);

  // Small Growth
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> small_growth
      = std::make_shared<Model::Growth<Real, Matrix>>(A_index, nucleation_order, cutoff, max_size,
                                                      POM_index, conserved_size, k2);

  // Large Growth
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> large_growth
      = std::make_shared<Model::Growth<Real, Matrix>>(A_index, cutoff+1, max_size, max_size,
                                                      POM_index, conserved_size, k3);

  // Create Model
  Model::Model<Real, Matrix> three_step_alt(nucleation_order, max_size);
  three_step_alt.add_rhs_contribution(nucleation);
  three_step_alt.add_rhs_contribution(small_growth);
  three_step_alt.add_rhs_contribution(large_growth);

  // set up initial condition
  Vector ic = Vector::Zero(max_size+1);
  ic(0) = 0.0012;

  N_Vector ic_sundials = N_VNew_Serial(max_size+1);
  for (unsigned int i=0; i<max_size+1; ++i)
  {
    NV_Ith_S(ic_sundials, i) = ic(i);
  }



  /*******************************************
   * Solve using Eigen as an accuracy baseline
   ******************************************/
  ODE::StepperBDF<4, Real, Matrix> stepper(three_step_alt);
  auto begin = std::chrono::steady_clock::now();

  auto eigen_accurate_solution = ODE::solve_ode(stepper, ic, start_time, end_time, 1e-4);

  auto end = std::chrono::steady_clock::now();
  std::cout << "Accurate Eigen sparse solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds"
            << std::endl;

  auto nvec_accurate_eigen = N_VNew_Serial(max_size+1);
  for (unsigned int i=0; i<max_size+1; ++i)
  {
    NV_Ith_S(nvec_accurate_eigen, i) = eigen_accurate_solution(i);
  }


  /*******************************************
   * Solve using Eigen as a time baseline
   ******************************************/
  begin = std::chrono::steady_clock::now();

  auto eigen_fast_solution = ODE::solve_ode(stepper, ic, start_time, end_time, 5e-3);

  end = std::chrono::steady_clock::now();
  std::cout << "Fast Eigen sparse solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds";

  auto nvec_fast_eigen = N_VNew_Serial(max_size+1);
  for (unsigned int i=0; i<max_size+1; ++i)
  {
    NV_Ith_S(nvec_fast_eigen, i) = eigen_fast_solution(i);
  }

  auto eigen_fast_error = N_VNew_Serial(max_size+1);
  N_VLinearSum(1., nvec_accurate_eigen, -1., nvec_fast_eigen, eigen_fast_error);

  std::cout << "; Infinity-norm = "
            << N_VMaxNorm(eigen_fast_error)
            << std::endl;



  /*********************************
   * Solve using SUNDIALS dense
   ********************************/
  const sundials::CVodeParameters<Real> dense_settings(sundials::DENSE, sundials::DIRECTSOLVE,
                                                       start_time, end_time,false,
                                                       true,true,1e-7,
                                                       1e-4,std::numeric_limits<Real>::epsilon(),CSC_MAT);

  sundials::CVodeSolver<Matrix, Real> dense_solver(dense_settings, three_step_alt, ic_sundials);
  N_Vector dense_solution = N_VNew_Serial(max_size+1);

  begin = std::chrono::steady_clock::now();

  dense_solver.solve_ode(dense_solution, end_time);

  end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS dense solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds";

  auto dense_error = N_VNew_Serial(max_size+1);
  N_VLinearSum(1., nvec_accurate_eigen, -1., dense_solution, dense_error);

  std::cout << "; Infinity-norm = "
            << N_VMaxNorm(dense_error)
            << std::endl;


  /*********************************
   * Solve using SUNDIALS SPBCGS
   ********************************/
  /*const sundials::CVodeParameters<Real> spbcgs_settings(sundials::SPARSE, sundials::SPBCGS,
                                                        start_time, end_time,false,
                                                        true,true,1e-7,
                                                        1e-4,std::numeric_limits<Real>::epsilon(),CSC_MAT);

  sundials::CVodeSolver<Matrix, Real> spbcgs_solver(spbcgs_settings, three_step_alt, ic_sundials);
  N_Vector spbcgs_solution = N_VNew_Serial(max_size+1);

  begin = std::chrono::steady_clock::now();

  spbcgs_solver.solve_ode(spbcgs_solution, end_time);

  end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS SPBCGS solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds";

  auto spbcgs_error = N_VNew_Serial(max_size+1);
  N_VLinearSum(1., nvec_accurate_eigen, -1., spbcgs_solution, spbcgs_error);

  std::cout << "; Infinity-norm = "
            << N_VMaxNorm(spbcgs_error)
            << std::endl;*/


  /*********************************
   * Solve using SUNDIALS SPFGMR
   ********************************/
  const sundials::CVodeParameters<Real> spfgmr_settings(sundials::SPARSE, sundials::SPFGMR,
                                                        start_time, end_time,false,
                                                        true,true,1e-7,
                                                        1e-4,std::numeric_limits<Real>::epsilon(),CSC_MAT);

  sundials::CVodeSolver<Matrix, Real> spfgmr_solver(spfgmr_settings, three_step_alt, ic_sundials);
  N_Vector spfgmr_solution = N_VNew_Serial(max_size+1);

  begin = std::chrono::steady_clock::now();

  spfgmr_solver.solve_ode(spfgmr_solution, end_time);

  end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS SPFGMR solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds";

  auto spfgmr_error = N_VNew_Serial(max_size+1);
  N_VLinearSum(1., nvec_accurate_eigen, -1., spfgmr_solution, spfgmr_error);

  std::cout << "; Infinity-norm = "
            << N_VMaxNorm(spfgmr_error)
            << std::endl;


  /*********************************
   * Solve using SUNDIALS SPGMR
   ********************************/
  /*const sundials::CVodeParameters<Real> spgmr_settings(sundials::SPARSE, sundials::SPGMR,
                                                        start_time, end_time,false,
                                                        true,true,1e-7,
                                                        1e-4,std::numeric_limits<Real>::epsilon(),CSC_MAT);

  sundials::CVodeSolver<Matrix, Real> spgmr_solver(spgmr_settings, three_step_alt, ic_sundials);
  N_Vector spgmr_solution = N_VNew_Serial(max_size+1);

  begin = std::chrono::steady_clock::now();

  spgmr_solver.solve_ode(spgmr_solution, end_time);

  end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS SPGMR solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds";

  auto spgmr_error = N_VNew_Serial(max_size+1);
  N_VLinearSum(1., nvec_accurate_eigen, -1., spgmr_solution, spgmr_error);

  std::cout << "; Infinity-norm = "
            << N_VMaxNorm(spgmr_error)
            << std::endl;*/


  /*********************************
   * Solve using SUNDIALS SPTFQMR
   ********************************/
  /*const sundials::CVodeParameters<Real> sptfqmr_settings(sundials::SPARSE, sundials::SPTFQMR,
                                                       start_time, end_time,false,
                                                       true,true,1e-7,
                                                       1e-4,std::numeric_limits<Real>::epsilon(),CSC_MAT);

  sundials::CVodeSolver<Matrix, Real> sptfqmr_solver(sptfqmr_settings, three_step_alt, ic_sundials);
  N_Vector sptfqmr_solution = N_VNew_Serial(max_size+1);

  begin = std::chrono::steady_clock::now();

  sptfqmr_solver.solve_ode(sptfqmr_solution, end_time);

  end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS SPTFQMR solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds";

  auto sptfqmr_error = N_VNew_Serial(max_size+1);
  N_VLinearSum(1., nvec_accurate_eigen, -1., sptfqmr_solution, sptfqmr_error);

  std::cout << "; Infinity-norm = "
            << N_VMaxNorm(sptfqmr_error)
            << std::endl;*/
}