#include "sample.h"
#include <iostream>
#include "nvector_eigen.h"
#include <eigen3/Eigen/Sparse>
#include "sampling_algorithm.h"
#include <string>
#include "data.h"
#include "src/models.h"
#include <vector>
#include "chemical_reaction.h"
#include <utility>



using Real = realtype;
using Matrix = Eigen::SparseMatrix<realtype, Eigen::RowMajor>;
using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;
using SolverType = Eigen::SparseLU<Matrix>; // ODE solver test found this was fastest on average



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




Real
log_likelihood(const Sampling::Sample<Real> & s)
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
          {5, 790}
      };

  if (Sampling::sample_is_valid(s, real_prm_bounds, int_prm_bounds) == false)
  {
    return -std::numeric_limits<Real>::max();
  }
  else {
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
    Real abs_tol = 1e-12;
    Real rel_tol = 1e-6;

    Data::IrHPO4Data<Real> data;
    std::vector<Real> times = {data.time1, data.time2, data.time3, data.time4};

    unsigned int first_particle_index = 3;
    unsigned int last_particle_index = n_vars - 1;

    Histograms::Parameters<Real> binning_parameters(25, 0.3, 2.8);
    unsigned int first_particle_size = 2;
    unsigned int particle_size_increase = 2;
    std::vector<std::vector<Real> > data_sets = {data.tem_time1, data.tem_time2, data.tem_time3, data.tem_time4};

    Sampling::ModelingParameters<Real, Matrix> model_settings(real_prm_bounds,
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
    auto log_likelihood = SUNDIALS_Statistics::compute_likelihood_TEM_only<Real, Matrix,SolverType,DIRECT>(s, model_settings);
    return log_likelihood;
  }
}



int main (int argc, char **argv)
{
  unsigned int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

  // Don't have a good idea of what a "good" set of parameters is so just randomly generate parameters for each thread
  // FIXME - if successive iterations of this code are run, the starting samples can be chosen as the mean of the previous run

#pragma omp parallel for
  for (unsigned int thread=0; thread<n_threads; ++thread) {
    // Baseline sample
    /*std::vector<Real> real_prm = {1e-3, 1e5, 1e4, 1e5, 1e4, 1e2};
    std::vector<int> int_prm = {100};*/
    // Sample update after previous run
    std::vector<Real> real_prm = {0.00168761, 100129, 9942.68, 100012, 9982.08, 98.4339};
    std::vector<int> int_prm = {155};
    Sampling::Sample<Real> sample(real_prm, int_prm);

    // Create the sampler
    auto d = sample.real_valued_parameters.size() + sample.integer_valued_parameters.size();
    // FIXME - if multiple iterations of this code are run, replace the starting covariance with the covariance matrix from the last run.
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> covariance(d,d);
    // Baseline covariance
/*
    covariance <<
               1e-3,       0,      0,      0,       0,       0,     0,
        0,  1e5, 0, 0,  0,  0,  0,
        0, 0,  1e4,  0, 0,  0, 0,
        0, 0,  0,   1e5, 0, 0, 0,
        0,  0, 0, 0,  1e4,  0,    0,
        0,  0,  0, 0,  0,   1e2,  0,
        0,  0, 0, 0,    0,  0,       1e1;
*/
    // Covariance based on previous run

    covariance <<
     3.17022e-08, 0.000515745, -0.000564883, 0.00049721, 1.30706e-05, 1.04134e-05, -0.00200345,
0.000515745, 1475.18, -1023.56, -1502.72, -224.115, 50.0783, 590.389,
-0.000564883, -1023.56, 727.707, 1043.53, 153.797, -35.5918, -307.723,
0.00049721, -1502.72, 1043.53, 1584.71, 233.461, -52.287, -610.144,
1.30706e-05, -224.115, 153.797, 233.461, 35.0768, -7.68912, -99.3635,
1.04134e-05, 50.0783, -35.5918, -52.287, -7.68912, 1.78074, 14.7378,
-0.00200345, 590.389, -307.723, -610.144, -99.3635, 14.7378, 952.095;


    covariance *= 2.38*2.38/d;
    covariance *= 100;

    auto chain_num = (argc > 1 ?
        atoi(argv[1] + thread) :
        thread);

    Sampling::AdaptiveMetropolisSampler<Real, Sampling::Sample<Real>> sampler(&log_likelihood,
                                                                              covariance,
                                                                              chain_num);

    // Create a seed for the random number generator to ensure consistent results.
    const std::uint_fast32_t random_seed = std::hash<std::string>()(std::to_string(chain_num));

    // Generate samples
    const unsigned int n_burn_in = 0;
    const unsigned int n_samples = 100;
    const unsigned int adaptation_period = 100;
    const unsigned int adaptation_frequency = 100;

    sampler.sample_with_burn_in(sample,
                                n_burn_in,
                                n_samples,
                                adaptation_period,
                                adaptation_frequency,
                                random_seed);
  }
  std::string command = "./summarize_samples";
  const auto first_chain = (argc > 1 ?
      atoi(argv[1]) :
      0);
  command += " " + std::to_string(first_chain);
  command += " " + std::to_string(first_chain + n_threads - 1);
  auto err = system(command.c_str());
}