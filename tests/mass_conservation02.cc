#include "sampling_sundials.h"
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include <cmath>


using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<Real> >;


/*
 * This is testing conservation of mass using the 4-step mechanism on the Ir-POM system.
 * There should be a constant number of Iridium monomers (up to rounding error or small numerical error).
 */


Real growth_kernel(const unsigned int size, const Real k)
{
  return k * (1.0 * size) * (2.677 * std::pow(1.0*size, -0.28));
}



/*
 * The mechanism is
 *    A + 2solv <-> Asolv + L
 *    2Asolv + A -> B + L
 *    A + B -> C + L
 *    A + C -> C + L
 */
MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_mechanism()
{
  // Parameters
  const Real kb = 1.37e5;
  const Real kf = kb * 5.e-7;
  const Real k1 = 7.69e4;
  const Real k2 = 1.40e4;
  const Real k3 = 7.15e3;
  const Real k4 = 1.74e3;
  const unsigned int M = 111;

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
  MEPBM::ParticleGrowth<Real, Matrix> small_growth(B, 1, max_size, [&](const unsigned int size){return growth_kernel(size, k2);}, { {A,1} }, { {L,1} });
  MEPBM::ParticleGrowth<Real, Matrix> large_growth(C, 1, max_size, [&](const unsigned int size){return growth_kernel(size, k3);}, { {A,1} }, { {L,1} });
  MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B, B, max_size, [&](const unsigned int sizeA, const unsigned int sizeB){return k4 * growth_kernel(sizeA, 1) * growth_kernel(sizeB, 1);}, {},{});

  MEPBM::ChemicalReactionNetwork<Real, Matrix> network({nucleationAf, nucleationAb, nucleationB},
                                                       {small_growth, large_growth},
                                                       {agglom});
  return network;
}


int rhs(Real t, N_Vector x, N_Vector x_dot, void * user_data)
{
  auto mech = create_mechanism();

  // Get the function
  auto rhs = mech.rhs_function();

  // Apply the function
  int err = rhs(t, x, x_dot, user_data);

  return err;
}



int jac(Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  auto mech = create_mechanism();

  // Get the function
  auto jac = mech.jacobian_function();

  // Apply the function
  int err = jac(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);

  return err;
}


int main ()
{
  // Initial condition
  auto ic = MEPBM::create_eigen_nvector<Vector>(2501);
  auto ic_vec = static_cast<Vector*>(ic->content);
  (*ic_vec)(0) = 0.0012;
  for (unsigned int i=1; i<2501; ++i)
    (*ic_vec)(i) = 0;

  // State start and end times
  const Real t0 = 0;
  const Real t1 = 0.1;

  // Create the matrix template
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(2501,2501);

  // Create the linear solver
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();

  // Create the CVODE object
  MEPBM::CVODE<Real> ode_solver(ic,template_matrix, linear_solver,&rhs,&jac,t0,t1);

  // Solutions
  const int n_solutions = 10;
  const Real dt = (t1-t0)/n_solutions;
  for (unsigned int s=0; s<n_solutions; ++s)
  {
    auto solution_pair = ode_solver.solve(dt + s*dt);
    auto end_ode = std::chrono::steady_clock::now();
    auto solution = solution_pair.first;
    const Vector sol = *static_cast<Vector*>(solution->content);

    // Multiply solution vector by number of Iridium atoms in each entry
    Real mass = 0.0;
    mass += sol(0);
    mass += sol(1);
    for (unsigned int i=3; i<sol.size(); ++i)
      mass += sol(i) * i;

    // Compare to initial number of Ir atoms
    std::cout << (0.0012 - mass) << std::endl;

    solution->ops->nvdestroy(solution);
  }

  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);
}