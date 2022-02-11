#include "sampling_sundials.h"
#include "src/ir_hpo4_data.h"

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>
#include <sampleflow/consumers/covariance_matrix.h>
#include <sampleflow/consumers/count_samples.h>
#include <sampleflow/filters/discard_first_n.h>
#include <sampleflow/producers/differential_evaluation_mh.h>

#include <omp.h>

#include <utility>
#include <fstream>
#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>


using Real = realtype;
using Sample = Eigen::Matrix<Real, 1, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IncompleteLUT<Real>>;



std::pair<Sample, Real>
perturb(const Sample & sample,
        const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> & cov,
        std::mt19937 &rng)
{
  Vector perturb_std_normal(sample.size());
  for (unsigned int i=0;i<perturb_std_normal.size(); ++i)
    perturb_std_normal(i) = std::normal_distribution<Real>(0,1)(rng);

  const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> L = cov.llt().matrixL();
  Sample perturbation = (L * perturb_std_normal).transpose();

  Sample new_sample = sample + perturbation;
  return {new_sample, 1.};
}



bool
within_bounds(const Sample & sample)
{
  // Parameters are: kf, kb, k1, k2, k3, k4, M
  Vector lower_bounds(7);
  Vector upper_bounds(7);
  lower_bounds << 0, 0, 0, 0, 0, 0, 5;
  upper_bounds << 1e3, 1e10, 1e10, 1e10, 1e10, 1e10, 200;

  assert(sample.size() == lower_bounds.size());
  assert(sample.size() == upper_bounds.size());

  // Check each bound
  for (unsigned int i=0; i<sample.size(); ++i)
  {
    if (sample(i) < lower_bounds(i) ||
        sample(i) > upper_bounds(i))
      return false;
  }

  // If we get here then the sample is within all bounds
  return true;
}



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



Real
log_likelihood(const N_Vector solution_vector, const std::vector<Real> data_vector)
{
  // Reduce the solution vector to only the concentrations of particles
  const unsigned int first_size = 2;
  const unsigned int first_index = 3;
  const unsigned int max_size = 450;
  const unsigned int last_index = max_size + (first_index - first_size);
  auto only_particles = MEPBM::get_subset<Real>(solution_vector, first_index, last_index);

  // Normalize the concentrations to make them a probability distribution
  auto particle_probability = MEPBM::normalize_concentrations(only_particles);

  // Convert to a std::vector to work with Histogram
  auto probability_vec = MEPBM::to_vector(particle_probability);

  // Create the diameter vector corresponding to the particle vector
  std::vector<unsigned int> particle_sizes(only_particles.size());
  unsigned int size = first_size;
  for (auto & s : particle_sizes)
  {
    s = size;
    ++size;
  }


  std::vector<Real> particle_diameters(particle_sizes.size());
  for (unsigned int i=0; i<particle_diameters.size(); ++i)
  {
    particle_diameters[i] = MEPBM::atoms_to_diameter<Real>(particle_sizes[i]);
  }

  // Convert to a Histogram to create the Probability Mass Function
  // Want bins to be -> 0.4:0.05:2.3, which is 38 bins
  const MEPBM::Parameters<Real> hist_prm(38, 0.4, 2.3);
  const auto pmf = MEPBM::create_histogram(probability_vec, particle_diameters,hist_prm);

  // Create the data count histogram
  std::vector<Real> ones(data_vector.size());
  for (auto & val : ones)
    val = 1.;

  const auto data_counts = MEPBM::create_histogram(ones, data_vector, hist_prm);

  // Compute the likelihood
  Real likelihood = MEPBM::log_multinomial(pmf, data_counts);
  return likelihood;
}



Real
log_probability(const Sample & sample)
{
  std::stringstream msg;
  msg << "Proposed sample: " << sample << "; Log probability: ";
  if (!within_bounds(sample)) {
    msg << -std::numeric_limits<Real>::infinity() << "\n";
    std::cout << msg.str();
    return -std::numeric_limits<Real>::infinity();
  }

  // Get the data
  const MEPBM::HPO4Data<Real> data;
  const std::vector<Real> tem_times = {data.time1, data.time2, data.time3, data.time4};
  const std::vector<std::vector<Real>> tem_data = {data.tem_data_t1, data.tem_data_t2, data.tem_data_t3, data.tem_data_t4};

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
  const Real t_end = tem_times.back();
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
  Real log_prob = 0;
  // Loop through all TEM times
  for (unsigned int t=0; t<tem_times.size(); ++t)
  {
    auto sol = ode_solver.solve(tem_times[t]);
    log_prob += log_likelihood(sol, tem_data[t]);
    sol->ops->nvdestroy(sol);
  }

  // Perform cleanup of SUNDIALS objects
  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);

  msg << log_prob << "\n";
  std::cout << msg.str();
  return log_prob;
}



Sample
crossover(const Sample &current_sample, const Sample & sample_a, const Sample & sample_b)
{
  return current_sample + (2.38 / std::sqrt(2*current_sample.size())) * ( sample_a - sample_b);
}



int main(int argc, char **argv)
{
  /*
   * Settings for the sampling algorithm
   * 1) How many chains should be used
   * 2) How long the burn-in should be
   * 3) How many total samples are desired
   * 4) What should the starting sample be
   * 5) How much should the starting sample be perturbed for the start of each chain
   */

  // 1)
  const unsigned int n_chains = 7;

  // 2)
  const unsigned int n_burn_in = 0;

  // 3)
  const unsigned int n_samples = 22;

  // 4)
  const int prm_dim = 7;
  Sample starting_guess(prm_dim);
  // kf, kb, k1, k2, k3, k4, M
  starting_guess << 2.6e-1, 2.e4, 2.2, 5.4e4, 1e3, 1.6e6, 23;

  // 5)
  // sigma^2 = k*^2/9 means 99.7% of draws will be +-100% of the parameter value
  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> perturb_cov(prm_dim, prm_dim);
  perturb_cov <<
      std::pow(starting_guess(0)/3., 2.), 0, 0, 0, 0, 0, 0,
      0, std::pow(starting_guess(1)/3., 2.), 0, 0, 0, 0, 0,
      0, 0, std::pow(starting_guess(2)/3., 2.), 0, 0, 0, 0,
      0, 0, 0, std::pow(starting_guess(3)/3., 2.), 0, 0, 0,
      0, 0, 0, 0, std::pow(starting_guess(4)/3., 2.), 0, 0,
      0, 0, 0, 0, 0, std::pow(starting_guess(5)/3., 2.), 0,
      0, 0, 0, 0, 0, 0, std::pow(starting_guess(6)/3., 2.);

  /*
   * ***************************************
   * End of settings, beginning of algorithm
   * ***************************************
   */

  // Create and seed random number generator used throughout the code
  const std::uint_fast32_t random_seed
      = (argc > 1 ?
         std::hash<std::string>()(std::to_string(atoi(argv[1]))) :
         std::hash<std::string>()(std::to_string(0)));

  std::mt19937 rng;
  rng.seed(random_seed);



  // Generate starting parameters. Make sure they are valid parameters. Redraw parameters if necessary.
  std::vector<Sample> starting_samples(n_chains);
  for (auto & s : starting_samples)
  {
    bool invalid_sample = true;
    while (invalid_sample)
    {
      auto sample = perturb(starting_guess, 0.0*perturb_cov, rng);
      if (within_bounds(sample.first))
      {
        s = sample.first;
        invalid_sample = false;
      }
    }
  }



  // Setup sampler
  SampleFlow::Producers::DifferentialEvaluationMetropolisHastings<Sample> demc;

  // Filter out burn-in
  SampleFlow::Filters::DiscardFirstN<Sample> burn_in(n_burn_in);
  burn_in.connect_to_producer(demc);

  // Track the mean value. Only track after burn-in, so filter through the burn_in object instead of directly to the sampler.
  SampleFlow::Consumers::MeanValue<Sample> mean_value;
  mean_value.connect_to_producer(burn_in);

  // Track the covariance. Only track after burn-in, so filter through the burn_in object instead of directly to the sampler.
  SampleFlow::Consumers::CovarianceMatrix<Sample> covariance_matrix;
  covariance_matrix.connect_to_producer(burn_in);

  // Track the acceptance ratio. Only track after burn-in, so filter through the burn_in object instead of directly to the sampler.
  SampleFlow::Consumers::AcceptanceRatio<Sample> acceptance_ratio;
  acceptance_ratio.connect_to_producer(burn_in);

  // Output the samples to a file. Don't output the burn-in period.
  std::ofstream samples("samples"
                        +
                        (argc > 1 ?
                         std::string(".") + std::to_string(atoi(argv[1])) :
                         std::string(".") + std::to_string(0))
                        +
                        ".txt"
  );
  MEPBM::MyStreamOutput<Sample> stream_output(samples);
  stream_output.connect_to_producer(burn_in);

  // Define the perturb function the sampler will use
  auto sampler_perturb = [&](const Sample &s)
  {
    return perturb(s, 0.001 * perturb_cov, rng);
  };



  // Generate samples
  demc.sample(starting_samples,
              &log_probability,
              sampler_perturb,
              &crossover,
              1,
              n_burn_in + n_samples);



  std::cout << "Mean value: " << mean_value.get() << std::endl;
  std::cout << "Acceptance ratio: " << acceptance_ratio.get() << std::endl;
  std::cout << "Covariance matrix:\n" << covariance_matrix.get() << std::endl;
}