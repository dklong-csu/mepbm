#include "sample.h"
#include <iostream>
#include "nvector_eigen.h"
#include <eigen3/Eigen/Sparse>
#include "sampling_algorithm.h"
#include <string>
#include "data.h"
#include "models.h"
#include <vector>
#include "chemical_reaction.h"
#include <utility>
#include "sundials_statistics.h"
#include <limits>



using Real = realtype;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using SolverType = Eigen::SparseLU<Matrix>; // ODE solver test found this was fastest on average
using Sample = Sampling::Sample<Real>;



Real
growth_kernel(const unsigned int size)
{
  return size * 2.677 * std::pow(size * 1., -0.28);
}



Model::Model<Real, Matrix>
four_step(const std::vector<Real> real_prm,
          const std::vector<int> int_prm)
{
  /*
   * Mechanism is
   *    A <-> Asolv + L  (1)
   *    Asolv -> B_2 + L         (2)
   *    A + B_i -> B_{i+2} + L   (3)
   *    A + C_i -> C_{i+2} + L   (4)
   *    B_i + B_j -> B_{i+j}     (5)
   */

  static unsigned int A_index = 0;
  static unsigned int Asolv_index = 1;
  static unsigned int L_index = 2;
  static unsigned int small_particle_start_index = 3;
  static unsigned int smallest_particle_size = 2;
  static unsigned int largest_particle_size = 800;

  const unsigned int small_particle_end_index = int_prm[0];
  const unsigned int large_particle_start_index = small_particle_end_index + 1;
  const unsigned int large_particle_end_index = (largest_particle_size - smallest_particle_size) + small_particle_start_index;
  const unsigned int large_particle_first_size = (large_particle_start_index - small_particle_start_index) + smallest_particle_size;


  Model::Species A(A_index);
  Model::Species Asolv(Asolv_index);
  Model::Species L(L_index);
  Model::Particle B(small_particle_start_index, small_particle_end_index, smallest_particle_size);
  Model::Particle C(large_particle_start_index, large_particle_end_index, large_particle_first_size);

  // Eqn (1)
  const std::vector<std::pair<Model::Species, unsigned int>> eqn1_reactants = { {A,1}};
  const std::vector<std::pair<Model::Species, unsigned int>> eqn1_products = { {Asolv,1}, {L, 1}};
  const Real kf = real_prm[0];
  std::shared_ptr< Model::RightHandSideContribution<Real, Matrix> > eqn1f =
      std::make_shared< Model::ChemicalReaction<Real, Matrix> >(eqn1_reactants,eqn1_products, kf);

  const Real kb = real_prm[1];
  std::shared_ptr< Model::RightHandSideContribution<Real, Matrix> > eqn1b =
      std::make_shared< Model::ChemicalReaction<Real, Matrix> >(eqn1_products,eqn1_reactants, kb);

  // Eqn (2)
  const std::vector<std::pair<Model::Species, unsigned int>> eqn2_reactants = { {Asolv, 1}};
  auto B_2 = B.get_particle_species(3);
  const std::vector<std::pair<Model::Species, unsigned int>> eqn2_products = { {B_2, 1}, {L, 1}};
  const Real k1 = real_prm[2];
  std::shared_ptr< Model::RightHandSideContribution<Real, Matrix> > eqn2 =
      std::make_shared< Model::ChemicalReaction<Real, Matrix> >(eqn2_reactants,eqn2_products, k1);

  // Eqn (3)
  static unsigned int growth_amount = 2;
  const Real k2 = real_prm[3];
  const std::vector<std::pair<Model::Species, unsigned int>> eqn3_reactants = { {A, 1} };
  const std::vector<std::pair<Model::Species, unsigned int>> eqn3_products  = { {L, 2} };

  std::shared_ptr< Model::RightHandSideContribution<Real, Matrix> > eqn3 =
      std::make_shared< Model::ParticleGrowth<Real, Matrix> >(B,
                                                              k2,
                                                              growth_amount,
                                                              largest_particle_size,
                                                              &growth_kernel,
                                                              eqn3_reactants,
                                                              eqn3_products);

  // Eqn (4)
  const Real k3 = real_prm[4];
  const std::vector<std::pair<Model::Species, unsigned int>> eqn4_reactants = { {A, 1} };
  const std::vector<std::pair<Model::Species, unsigned int>> eqn4_products  = { {L, 2} };

  std::shared_ptr< Model::RightHandSideContribution<Real, Matrix> > eqn4 =
      std::make_shared< Model::ParticleGrowth<Real, Matrix> >(C,
                                                              k3,
                                                              growth_amount,
                                                              largest_particle_size,
                                                              &growth_kernel,
                                                              eqn4_reactants,
                                                              eqn4_products);

  // Eqn (5)
  const Real k4 = real_prm[5];
  const std::vector<std::pair<Model::Species, unsigned int>> eqn5_reactants = {};
  const std::vector<std::pair<Model::Species, unsigned int>> eqn5_products = {};

  std::shared_ptr< Model::RightHandSideContribution<Real, Matrix> > eqn5 =
      std::make_shared< Model::ParticleAgglomeration<Real, Matrix> >(B,
                                                                     B,
                                                                     k4,
                                                                     largest_particle_size,
                                                                     &growth_kernel,
                                                                     eqn5_reactants,
                                                                     eqn5_products);

  Model::Model<Real, Matrix> m(smallest_particle_size,largest_particle_size);
  m.add_rhs_contribution(eqn1f);
  m.add_rhs_contribution(eqn1b);
  m.add_rhs_contribution(eqn2);
  m.add_rhs_contribution(eqn3);
  m.add_rhs_contribution(eqn4);
  m.add_rhs_contribution(eqn5);

  return m;
}


std::vector<N_Vector>
solve_ode_during_likelihood(const Sample & s)
{
  // Model setup
  std::vector<std::pair<Real, Real> > real_prm_bounds =
      {
          {0, 1e12},
          {0, 1e12},
          {0, 1e12},
          {0, 1e12},
          {0, 1e12},
          {0, 1e12}
      };


  std::vector<std::pair<int, int> > int_prm_bounds =
      {
          {5, 700}
      };

    std::function<Model::Model<Real, Matrix>(const std::vector<Real>, const std::vector<int>)> create_model_fcn
        = four_step;

    // Calculate the number of dimensions
    // Track species: A, A_s, L, A_1/2, sizes 1, 2, ... , 799, 800
    const unsigned int max_size = 800;
    const unsigned int n_vars = 3 + max_size;

    N_Vector initial_condition = create_eigen_nvector<Vector>(n_vars);

    auto ic_vec = static_cast<Vector *>(initial_condition->content);
    // Initial precursor (dimer) concentration is 0.0025 -- index 0
    // Initial ligand (HPO4) concentration is 0.0625 -- index 2
    for (unsigned int i = 0; i < ic_vec->size(); ++i) {
      if (i == 0)
        (*ic_vec)(i) = 0.0025;
      else if (i == 2)
        (*ic_vec)(i) = 0.0625;
      else
        (*ic_vec)(i) = 0.;
    }
    Real start_time = 0;
    Real end_time = 10.0;
    Real abs_tol = 1e-13;
    Real rel_tol = 1e-6;

    Data::IrHPO4Data<Real> data;
    std::vector<Real> times = {data.time1, data.time2, data.time3, data.time4};

    unsigned int first_particle_index = 3;
    unsigned int last_particle_index = n_vars - 1;

    Histograms::Parameters<Real> binning_parameters(25, 0.3, 2.8);
    unsigned int first_particle_size = 2;
    unsigned int particle_size_increase = 2;
    std::vector<std::vector<Real> > data_sets = {data.tem_time1, data.tem_time2, data.tem_time3, data.tem_time4};

    Sampling::ModelingParameters<Real, Matrix> user_data(real_prm_bounds,
                                                              int_prm_bounds,
                                                              create_model_fcn,
                                                              initial_condition,
                                                              start_time,
                                                              end_time,
                                                              abs_tol,
                                                              rel_tol,
                                                              times,
                                                              first_particle_index,
                                                              last_particle_index,
                                                              binning_parameters,
                                                              first_particle_size,
                                                              particle_size_increase,
                                                              data_sets);


    auto solver_output = SUNDIALS_Statistics::Internal::solve_ODE_from_sample<Real, Matrix, SolverType, DIRECT>(s, user_data);
    auto solutions = solver_output.first;
    
    

  // Turn ODE solution(s) into a distribution
  std::vector< Histograms::Histogram<Real> > probabilities;

  std::vector<Real> particle_diameters(user_data.last_particle_index - user_data.first_particle_index + 1);
  for (unsigned int i=0; i<particle_diameters.size(); ++i)
  {
    particle_diameters[i]
        = SUNDIALS_Statistics::Internal::convert_particle_size_to_diameter<Real>(user_data.first_particle_size + i*user_data.particle_size_increase);
  }

  for (unsigned int i=0; i<solutions.size(); ++i)
  {
    auto sol = SUNDIALS_Statistics::Internal::convert_solution_to_vector<Real>(solutions[i]);
    auto concentrations = SUNDIALS_Statistics::Internal::strip_nanoparticles_from_vector<Real>(sol,
                                                                              user_data.first_particle_index,
                                                                              user_data.last_particle_index);
    auto p = SUNDIALS_Statistics::Internal::convert_concentrations_to_pmf(concentrations,
                                                     user_data.binning_parameters,
                                                     particle_diameters);
    probabilities.push_back(p);
  }

  // Turn data into a distribution
  std::vector< Histograms::Histogram<Real> > measurements;
  for (unsigned int i=0; i<solutions.size(); ++i)
  {
    auto m = SUNDIALS_Statistics::Internal::TEMData::bin_TEM_data(user_data.data[i], user_data.binning_parameters);
    measurements.push_back(m);
  }

  // Calculate the log likelihood
  Real likelihood = 0.;

  for (unsigned int i=0; i<probabilities.size(); ++i)
  {
    likelihood = SUNDIALS_Statistics::Internal::TEMData::compute_likelihood_from_binned_data(measurements[i], probabilities[i]);
    std::cout << likelihood << std::endl;
  }
    return solutions;
}



int main ()
{
  std::vector<Real> real_prm = {0.00428731, 101798, 8713.2, 98429.9, 9730.21, 153.977};
  std::vector<int> int_prm = {346};

  Sample s(real_prm, int_prm);

  auto sol = solve_ode_during_likelihood(s);

  std::ofstream output_file;
  output_file.open("ode_solutions.m");
  for (unsigned int i=0;i<sol.size();++i)
  {
    output_file << "sol" << i << " = [";
    auto vec = *static_cast<Vector *>(sol[i]->content);
    output_file << vec;
    output_file << "];" << std::endl;
  }
  output_file.close();
}