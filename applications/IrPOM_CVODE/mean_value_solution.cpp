#include "sampling_sundials.h"
#include <eigen3/Eigen/Eigen>

#include <iostream>
#include <fstream>

using Real = realtype;
using Sample = Eigen::Matrix<Real, 1, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IncompleteLUT<Real>>;


Real growth_kernel(const unsigned int size)
{
  return (1.0 * size) * (2.677 * std::pow(1.0*size, -0.28));
}



MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_mechanism(const Sample & sample)
{
  // Give names to elements of the sample
  assert(sample.size() == 5);
  const Real kb = sample(0);
  const Real kf = kb * 5e-7;
  const Real k1 = sample(1);
  const Real k2 = sample(2);
  const Real k3 = sample(3);
  const unsigned int M = sample(4);

  const unsigned int max_size = 2500;
  const Real S = 11.3;

  // Form the mechanism
  // Index the vector in the order:
  // [A->0, Asolv->1, L->2, Particles->3-2501]
  MEPBM::Species A(0);
  MEPBM::Species Asolv(1);
  MEPBM::Species L(2);
  MEPBM::Particle B(3, M, 3);
  MEPBM::Particle C(M+1, max_size, M+1);

  // Chemical reactions
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAf({ {A,1} }, { {Asolv,1}, {L,1} },S*S*kf);
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAb({ {Asolv,1}, {L,1} }, { {A,1} }, kb);
  auto B_nucleated = B.species(3);
  MEPBM::ChemicalReaction<Real, Matrix> nucleationB({ {Asolv,2}, {A,1} }, { {B_nucleated,1}, {L,1} }, k1);
  MEPBM::ParticleGrowth<Real, Matrix> small_growth(B, k2, 1, max_size, &growth_kernel, { {A,1} }, { {L,1} });
  MEPBM::ParticleGrowth<Real, Matrix> large_growth(C, k3, 1, max_size, &growth_kernel, { {A,1} }, { {L,1} });

  MEPBM::ChemicalReactionNetwork<Real, Matrix> network({nucleationAf, nucleationAb, nucleationB},
                                                       {small_growth, large_growth},
                                                       {});
  return network;
}



int
rhs(Real t, N_Vector x, N_Vector x_dot, void* user_data)
{
  const Sample sample = *static_cast<Sample*>(user_data);
  auto mechanism = create_mechanism(sample);

  auto rhs = mechanism.rhs_function();

  int err = 0;
  err = rhs(t,x,x_dot, user_data);

  return err;
}



int
jac(Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  const Sample sample = *static_cast<Sample*>(user_data);
  auto mechanism = create_mechanism(sample);

  auto jac = mechanism.jacobian_function();

  int err = 0;
  err = jac(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);

  return err;
}



void
solve_ode(const Sample & sample, const std::string & file_name)
{
  // Create the ODE solver
  auto ic = MEPBM::create_eigen_nvector<Vector>(2501);
  auto ic_vec = static_cast<Vector*>(ic->content);
  (*ic_vec)(0) = 0.0012;
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(2501,2501);
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();
  const Real t_start = 0;
  const Real t_end = 4.838;
  MEPBM::CVODE<Real> ode_solver(ic,
                                template_matrix,
                                linear_solver,
                                &rhs,
                                &jac,
                                t_start,
                                t_end);
  auto sample_ptr = &sample;
  auto user_data = (void *)sample_ptr;
  ode_solver.set_user_data(user_data);

  ode_solver.set_tolerance(1e-7,1e-13);

  std::ofstream out_file;
  out_file.open(file_name, std::ios::app);
  std::vector<Real> times = {0.918, 1.17, 2.336, 4.838};
  for (unsigned int i=0; i<times.size();++i)
  {
    auto sol = ode_solver.solve(times[i]);
    // output solution to a file in Matlab format
    std::string var_name = "sol" + std::to_string(i);
    out_file << var_name << " = [";
    auto sol_vec = static_cast<Vector*>(sol->content);
    out_file << *sol_vec << "];" << std::endl;
    // Delete sol from the heap
    sol->ops->nvdestroy(sol);
  }
  out_file.close();

  // Perform cleanup of SUNDIALS objects
  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);

  std::cout << "Solutions generated!" << std::endl;
}



int main ()
{
  Sample mean_prm_new(5);
  mean_prm_new << 6697.366638, 121179.063303, 249454.291397, 6097.416300, 108;
  Sample mean_prm_old_partial(5);
  mean_prm_old_partial << 6632.423355, 126060.762120, 265205.395442, 6286.615366, 106;
  Sample mean_prm_old_all(5);
  mean_prm_old_all << 6674.992925, 125555.551951, 256546.987552, 6289.311830, 106;

  solve_ode(mean_prm_new, "mean_sol_new.m");
  solve_ode(mean_prm_old_partial, "mean_sol_old_partial.m");
  solve_ode(mean_prm_old_all, "mean_sol_old_all.m");
}