#include "sampling_sundials.h"
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include <vector>
#include <utility>
#include <cmath>


using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<Real> >;


/*
 * This is testing conservation of mass using the 3-step mechanism on the Ir-POM system.
 * There should be a constant number of Iridium monomers (up to rounding error or small numerical error).
 */


Real growth_kernel(const unsigned int size)
{
  return (1.0 * size) * (2.677 * std::pow(1.0*size, -0.28));
}



/*
 * The mechanism is
 *    A + 2solv <-> Asolv + L
 *    2Asolv + A -> B + L
 *    A + B -> C + L
 *    A + C -> C + L
 */
int rhs(Real t, N_Vector x, N_Vector x_dot, void * user_data)
{
  // Parameters
  const Real kb = 6.62e3;
  const Real kf = kb * 5.e-7;
  const Real k1 = 1.24e5;
  const Real k2 = 2.6e5;
  const Real k3 = 6.22e3;
  const unsigned int M = 107;

  const unsigned int max_size = 2500;

  // Form the mechanism
  // Index the vector in the order:
  // [A->0, Asolv->1, L->2, Particles->3-2501]
  MEPBM::Species A(0);
  MEPBM::Species Asolv(1);
  MEPBM::Species L(2);
  MEPBM::Particle B(3, M, 3);
  MEPBM::Particle C(M+1, max_size, M+1);

  // Chemical reactions
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAf({ {A,1} }, { {Asolv,1}, {L,1} },11.3*11.3*kf);
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAb({ {Asolv,1}, {L,1} }, { {A,1} }, kb);
  auto B_nucleated = B.species(3);
  MEPBM::ChemicalReaction<Real, Matrix> nucleationB({ {Asolv,2}, {A,1} }, { {B_nucleated,1}, {L,1} }, k1);
  MEPBM::ParticleGrowth<Real, Matrix> small_growth(B, k2, 1, max_size, &growth_kernel, { {A,1} }, { {L,1} });
  MEPBM::ParticleGrowth<Real, Matrix> large_growth(C, k3, 1, max_size, &growth_kernel, { {A,1} }, { {L,1} });

  // Get the functions
  auto rhs_nAf = nucleationAf.rhs_function();
  auto rhs_nAb = nucleationAb.rhs_function();
  auto rhs_nB  = nucleationB.rhs_function();
  auto rhs_sg  = small_growth.rhs_function();
  auto rhs_lg  = large_growth.rhs_function();

  // Apply all functions
  int err = 0;
  err = rhs_nAf(t, x, x_dot, user_data);
  err = rhs_nAb(t, x, x_dot, user_data);
  err = rhs_nB(t, x, x_dot, user_data);
  err = rhs_sg(t, x, x_dot, user_data);
  err = rhs_lg(t, x, x_dot, user_data);

  return err;
}



int jac(Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // Parameters
  const Real kb = 6.62e3;
  const Real kf = kb * 5.e-7;
  const Real k1 = 1.24e5;
  const Real k2 = 2.6e5;
  const Real k3 = 6.22e3;
  const unsigned int M = 107;

  const unsigned int max_size = 2500;

  // Form the mechanism
  // Index the vector in the order:
  // [A->0, Asolv->1, L->2, Particles->3-2501]
  MEPBM::Species A(0);
  MEPBM::Species Asolv(1);
  MEPBM::Species L(2);
  MEPBM::Particle B(3, M, 3);
  MEPBM::Particle C(M+1, max_size, M+1);

  // Chemical reactions
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAf({ {A,1} }, { {Asolv,1}, {L,1} },11.3*11.3*kf);
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAb({ {Asolv,1}, {L,1} }, { {A,1} }, kb);
  auto B_nucleated = B.species(3);
  MEPBM::ChemicalReaction<Real, Matrix> nucleationB({ {Asolv,2}, {A,1} }, { {B_nucleated,1}, {L,1} }, k1);
  MEPBM::ParticleGrowth<Real, Matrix> small_growth(B, k2, 1, max_size, &growth_kernel, { {A,1} }, { {L,1} });
  MEPBM::ParticleGrowth<Real, Matrix> large_growth(C, k3, 1, max_size, &growth_kernel, { {A,1} }, { {L,1} });

  // Get the functions
  auto jac_nAf = nucleationAf.jacobian_function();
  auto jac_nAb = nucleationAb.jacobian_function();
  auto jac_nB  = nucleationB.jacobian_function();
  auto jac_sg  = small_growth.jacobian_function();
  auto jac_lg  = large_growth.jacobian_function();

  // Apply all functions
  int err = 0;
  err = jac_nAf(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
  err = jac_nAb(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
  err = jac_nB(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
  err = jac_sg(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
  err = jac_lg(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);

  return err;
}


int main ()
{
  // Initial condition
  auto ic = MEPBM::create_eigen_nvector<Vector>(2501);
  Vector* ic_vec = static_cast<Vector*>(ic->content);
  (*ic_vec)(0) = 0.0012;
  for (unsigned int i=1; i<2501; ++i)
    (*ic_vec)(i) = 0;

  // State start and end times
  const Real t0 = 0;
  const Real t1 = 0.01;

  // Create the matrix template
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(2501,2501);

  // Create the linear solver
  auto linear_solver = MEPBM::create_sparse_direct_solver<Matrix, Real, Solver>();

  // Create the CVODE object
  MEPBM::CVODE<Real> ode_solver(ic,template_matrix, linear_solver,&rhs,&jac,t0,t1);

  // Solve at t=1
  auto solution = ode_solver.solve(t1);
  const Vector sol = *static_cast<Vector*>(solution->content);

  // Multiply solution vector by number of Iridium atoms in each entry
  Real mass = 0.0;
  mass += sol(0);
  mass += sol(1);
  for (unsigned int i=3; i<sol.size(); ++i)
    mass += sol(i) * i;

  // Compare to initial number of Ir atoms
  std::cout << (0.0012 - mass) << std::endl;
}