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



int main (int argc, char **argv)
{
  // Create the covariance matrix before the loop so that all chains contribute to the same covariance matrix
  auto covariance_matrix = std::make_shared< SampleFlow::Consumers::CovarianceMatrix< std::valarray< Real > > >(); // use a valarray instead of Sample<> to support arithmetic
  unsigned int n_threads = 6;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

  // Don't have a good idea of what a "good" set of parameters is so just randomly generate parameters for each thread
  // FIXME - if successive iterations of this code are run, the starting samples can be chosen as the mean of the previous run

#pragma omp parallel for
  for (unsigned int thread=0; thread<n_threads; ++thread) {
    // create a sample
    // derek            -- kf, kb, k1, k2, k3, k4, M
    // convert to danny -- kf, kb, k1, k2, k4, k3, M
    std::vector<Real> real_prm = {1e-2, 1e4, 1e5, 1e5, 1e3, 1e2};
    std::vector<int> int_prm = {20};

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
    const unsigned int n_vars = 4 + max_size;

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

    Sampling::Sample<Real> sample1(real_prm, int_prm);


    // Create the sampler
    auto d = sample1.real_valued_parameters.size() + sample1.integer_valued_parameters.size();
    // FIXME - if multiple iterations of this code are run, replace the starting covariance with the covariance matrix from the last run.
    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> starting_covariance(d,d);
    starting_covariance <<
        8.9073e-06,   0.0198409,   0.0340872,    0.179625,    0.263132,   0.0329012, -0.00757917,
        0.0198409,      60.356,     70.1265,     547.348,     756.801,     93.1149,    -21.7613,
        0.0340872,     70.1265,     135.891,     635.528,     957.772,     118.748,    -27.2615,
        0.179625,     547.348,     635.528,     5169.47,     7062.75,     855.706,    -198.793,
        0.263132,     756.801,     957.772,     7062.75,       10104,     1228.84,    -282.243,
        0.0329012,     93.1149,     118.748,     855.706,     1228.84,     159.439,    -35.8455,
        -0.00757917,    -21.7613,    -27.2615,    -198.793,    -282.243,    -35.8455,     8.18516;

    std::vector<Real> variances = {1e-4, 1e2, 1e3, 1e3, 1e2, 1e1, 1e1};
    for (unsigned int i=0; i<d; ++i)
    {
      for (unsigned int j=0; j<d; ++j)
      {
        if (i==j)
        {
          starting_covariance(i,j) = variances[i];
        }
        else
        {
          starting_covariance(i,j) = 0.;
        }
      }
    }

    // The adaptive algorithm uses a proposal distribution of
    // (1-beta)*N(x, alpha*2.38^2/d * sample covariance) + beta*N(x, alpha*gamma^2/d * I)
    // alpha is calculated internally but beta and gamma need to be specified.
    const Real beta = 0.05;
    const Real gamma = 0.01;

    Sampling::AdaptiveMHSampler<Real, Matrix, Sampling::UniformPrior, Sampling::DataTEMOnly, SolverType, DIRECT>
        sampler(sample1, covariance_matrix, starting_covariance, model_settings, beta, gamma);

    // Create a seed for the random number generator to ensure consistent results.
    const std::uint_fast32_t random_seed
        = (argc > 1 ?
           std::hash<std::string>()(std::to_string(atoi(argv[1]) + thread)) :
           std::hash<std::string>()(std::to_string(thread)));

    // Create an output file to print all of the samples.
    std::ofstream samples_file("samples"
                               +
                               (argc > 1 ?
                                std::string(".") + std::to_string(atoi(argv[1]) + thread) :
                                std::string(".") + std::to_string(thread))
                               +
                               ".txt");

    const unsigned int n_samples = 1000;
    const unsigned int adaptive_start_sample = 100;
    const unsigned int ar_check_interval = 100;
    const Real goal_ar = .234;
    const Real ar_check_buffer = .05;
    sampler.generate_samples(n_samples, samples_file, random_seed, adaptive_start_sample, ar_check_interval, goal_ar, ar_check_buffer);
  }

  std::cout << "Covariance matrix is: " << std::endl;
  // Normal printout of matrix does not include comma separation, but comma separation is useful for pasting
  // a previous run's covariance matrix in for the next run.
  for (unsigned int r=0; r<covariance_matrix->get().rows();++r)
  {
    for (unsigned int c=0; c<covariance_matrix->get().cols();++c)
    {
      std::cout << covariance_matrix->get()(r,c);
      if (!(r==covariance_matrix->get().rows()-1 && c==covariance_matrix->get().cols()-1))
        std::cout << ", ";
    }
    std::cout << std::endl;
  }

}