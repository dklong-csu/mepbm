#include "sampling_sundials.h"
#include "src/ir_hpo4_data.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <utility>
#include <cassert>



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
  assert(sample.size() == 7);
  const Real kf = sample(0);
  const Real kb = sample(1);
  const Real k1 = sample(2);
  const Real k2 = sample(3);
  const Real k3 = sample(4);
  const Real k4 = sample(5);
  const unsigned int M = sample(6);

  const unsigned int max_size = 450;
  const Real S = 11.7;
  const unsigned int growth_amount = 2;

  // Form the mechanism
  // Index the vector in the order:
  // [A->0, Asolv->1, L->2, Particles->3-2501]
  MEPBM::Species A(0);
  MEPBM::Species Asolv(1);
  MEPBM::Species L(2);
  const unsigned int first_size = 2;
  const unsigned int first_index = 3;
  const unsigned int M_index = M + (first_index-first_size);
  MEPBM::Particle B(first_index, M_index, first_size);
  const unsigned int last_index = max_size + (first_index - first_size);
  MEPBM::Particle C(M_index+1, last_index, M+1);

  // Chemical reactions
  MEPBM::ChemicalReaction<Real, Matrix> nucleationAf({ {A,1} },
                                                     { {Asolv,1}, {L,1} },
                                                     S*S*kf);

  MEPBM::ChemicalReaction<Real, Matrix> nucleationAb({ {Asolv,1}, {L,1} },
                                                     { {A,1} },
                                                     kb);

  auto B_nucleated = B.species(B.index(first_size));
  MEPBM::ChemicalReaction<Real, Matrix> nucleationB({ {Asolv,1} },
                                                    { {B_nucleated,1}, {L,1} },
                                                    k1);

  MEPBM::ParticleGrowth<Real, Matrix> small_growth(B,
                                                   k2,
                                                   growth_amount,
                                                   max_size,
                                                   &growth_kernel,
                                                   { {A,1} },
                                                   { {L,2} });

  MEPBM::ParticleGrowth<Real, Matrix> large_growth(C,
                                                   k3,
                                                   growth_amount,
                                                   max_size,
                                                   &growth_kernel,
                                                   { {A,1} },
                                                   { {L,2} });

  MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B,
                                                    B,
                                                    k4,
                                                    max_size,
                                                    &growth_kernel,
                                                    {},
                                                    {});

  MEPBM::ChemicalReactionNetwork<Real, Matrix> network({nucleationAf, nucleationAb, nucleationB},
                                                       {small_growth, large_growth},
                                                       {agglom});
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



std::pair<std::vector<Vector>, std::vector<Real>>
solve_ode(const Sample & sample)
{
  // Get the data
  const MEPBM::HPO4Data<Real> data;
  const std::vector<Real> tem_times = {data.time1, data.time2, data.time3, data.time4};
  unsigned int tem_index = 0;
  unsigned int precursor_index = 1; // index=0 is time 0 which we don't need to check

  // Create the ODE solver
  const unsigned int first_size = 2;
  const unsigned int first_index = 3;
  const unsigned int max_size = 450;
  const unsigned int last_index = max_size + (first_index - first_size);
  auto ic = MEPBM::create_eigen_nvector<Vector>(last_index+1);
  auto ic_vec = static_cast<Vector*>(ic->content);
  // Initial A concentration = 0.0025 (index 0)
  // Initial HPO4 concentration = 0.0625 (index 2)
  (*ic_vec)(0) = 0.0025;
  (*ic_vec)(2) = 0.0625;
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic->ops->nvgetlength(ic),ic->ops->nvgetlength(ic));
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();
  const Real t_start = 0;
  const Real t_end = std::max(tem_times.back(), data.precursor_times.back());
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
  ode_solver.set_tolerance(1e-7,1e-13); // Based on visual inspection this tends to give non-oscillitory solutions

  // Create output objects that will be populated in subsequent steps
  std::vector<Vector> tem_solutions;
  std::vector<Real> precursor_concentrations;

  // Loop through all TEM and Precursor Curve times
  Real solve_time = std::min(tem_times[tem_index], data.precursor_times[precursor_index]);
  while (tem_index < tem_times.size() || precursor_index < data.precursor_times.size())
  {
    auto sol = ode_solver.solve(solve_time);
    // See if this is a precursor time
    if (std::abs(solve_time - data.precursor_times[precursor_index]) < 1e-6)
    {
      const auto conc_A = (*static_cast<Vector *>(sol->content))(0);
      precursor_concentrations.push_back(conc_A);
      ++precursor_index;
    }
    // See if this is a TEM time
    if (std::abs(solve_time - tem_times[tem_index]) < 1e-6)
    {
      // Only extract the particles
      const Vector particles = MEPBM::get_subset<Real>(sol, first_index, last_index);
      tem_solutions.push_back(particles);
      ++tem_index;
    }

    // Garbage collect the solution vector
    sol->ops->nvdestroy(sol);

    // Update the solve time
    if (tem_index >= tem_times.size())
    {
      if (precursor_index >= data.precursor_times.size())
      {
        // No more calculations will occur and the while loop will end
      }
      else
      {
        solve_time = data.precursor_times[precursor_index];
      }
    }
    else
    {
      if (precursor_index >= data.precursor_times.size())
      {
        solve_time = tem_times[tem_index];
      }
      else
      {
        solve_time = std::min(tem_times[tem_index], data.precursor_times[precursor_index]);
      }
    }
  }


  // Perform cleanup of SUNDIALS objects
  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);

  return {tem_solutions, precursor_concentrations};
}



int main ()
{
  // Solve the ODE for a set of parameters
  Sample parameters(7);
  //parameters << 2.6e-1, 2.e4, 2.2, 5.4e4, 1e3, 1.6e6, 23;
  parameters << 0.280176,    11256.3,    1.17277,    95293.3,    1057.67, 2.4607e+06,    18.5371 ;

  const auto sol_tem_and_precursor = solve_ode(parameters);
  const auto tem_sols = sol_tem_and_precursor.first;
  const auto precursor_conc = sol_tem_and_precursor.second;

  // Create a Matlab file with the solutions for visualization
  std::ofstream precursor_file;
  precursor_file.open("precursor_simulation.m");
  precursor_file << "A_conc_sim = [0.0025" << std::endl;
  for (auto c : precursor_conc)
    precursor_file << c << std::endl;
  precursor_file << "];" << std::endl;
  precursor_file.close();

  std::ofstream tem_file;
  tem_file.open("tem_simulation.m");
  for (unsigned int i=0; i<tem_sols.size(); ++i)
  {
    tem_file << "sol_time" << i+1 << "= [";
    tem_file << tem_sols[i];
    tem_file << "];\n" << std::endl;
  }

  std::vector<Real> particle_diameters;
  const unsigned int first_size = 2;
  const unsigned int max_size = 450;
  for (unsigned int s=first_size; s<=max_size; ++s)
    particle_diameters.push_back(MEPBM::atoms_to_diameter<Real>(s));
  tem_file << "particle_diameters = [";
  for (const auto d : particle_diameters)
    tem_file << d << std::endl;
  tem_file << "];" << std::endl;

  tem_file.close();

}