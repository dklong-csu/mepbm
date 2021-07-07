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


  /*********************************
   * Solve using SUNDIALS dense
   ********************************/
  const sundials::CVodeParameters<Real> dense_settings(sundials::DENSE, sundials::DIRECTSOLVE,
                                                       start_time, end_time,false,
                                                       true,true,1e-7,
                                                       1e-4,std::numeric_limits<Real>::epsilon(),CSC_MAT);

  sundials::CVodeSolver<Matrix, Real> dense_solver(dense_settings, three_step_alt, ic_sundials);
  N_Vector dense_solution = N_VNew_Serial(max_size+1);

  auto begin = std::chrono::steady_clock::now();

  dense_solver.solve_ode(dense_solution, end_time);

  auto end = std::chrono::steady_clock::now();
  std::cout << "SUNDIALS dense solve: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count()
            << " milliseconds"
            << std::endl;
}