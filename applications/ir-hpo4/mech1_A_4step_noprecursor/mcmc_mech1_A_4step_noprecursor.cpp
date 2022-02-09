#include "sampling_sundials.h"
#include "src/ir_hpo4_data.h"

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>
#include <sampleflow/consumers/covariance_matrix.h>
#include <sampleflow/consumers/count_samples.h>
#include <sampleflow/filters/discard_first_n.h>

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
  lower_bounds << 0, 0, 0, 0, 0, 0,5;
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
  if (!within_bounds(sample)) {
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

  return log_prob;
}




int main(int argc, char **argv)
{
  unsigned int n_threads = 1;
#ifdef _OPENMP
  n_threads = omp_get_max_threads();
#endif

#pragma omp parallel for
  for (unsigned int i=0;i<n_threads;++i)
  {
    const int prm_dim = 7;
    Sample starting_guess(prm_dim);
    // kf, kb, k1, k2, k3, M
    starting_guess << 1e-5, 1e5, 1e3, 1e3, 1e4, 1e2, 50;

    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> starting_cov(prm_dim, prm_dim);
    starting_cov <<
                 std::pow(starting_guess(0)/10, 2), 0, 0, 0, 0, 0, 0,
        0, std::pow(starting_guess(1)/10, 2), 0, 0, 0, 0, 0,
        0, 0, std::pow(starting_guess(2)/10, 2), 0, 0, 0, 0,
        0, 0, 0, std::pow(starting_guess(3)/10, 2), 0, 0, 0,
        0, 0, 0, 0, std::pow(starting_guess(4)/10, 2), 0, 0,
        0, 0, 0, 0, 0, std::pow(starting_guess(5)/10, 2), 0,
        0, 0, 0, 0, 0, 0, std::pow(starting_guess(6)/10, 2);


    // Use the Metropolis-Hastings sampling algorithm
    SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

    // We want a burn-in period
    const unsigned int samples_desired = 20000;
    const unsigned int burn_in_period = 0.02 * samples_desired;
    SampleFlow::Filters::DiscardFirstN<Sample> burn_in(burn_in_period);
    burn_in.connect_to_producer(mh_sampler);


    // Output samples to a file
    std::ofstream samples("samples"
                          +
                          (argc > 1 ?
                           std::string(".") + std::to_string(atoi(argv[1])+i) :
                           std::string(".") + std::to_string(i))
                          +
                          ".txt"
    );


    // Output the auxilary data as well as the parameter values
    MEPBM::MyStreamOutput<Sample> stream_output(samples);
    stream_output.connect_to_producer(burn_in);

    // Keep track of the mean
    SampleFlow::Consumers::MeanValue<Sample> mean_value;
    mean_value.connect_to_producer(burn_in);

    // Keep track of the acceptance_ratio
    SampleFlow::Consumers::AcceptanceRatio<Sample> acceptance_ratio;
    acceptance_ratio.connect_to_producer(burn_in);

    // Only write to disk every 100 samples
    SampleFlow::Filters::TakeEveryNth<Sample> every_100th(100);
    every_100th.connect_to_producer(mh_sampler);
    SampleFlow::Consumers::Action<Sample>
        flush_sample([&samples](const Sample &, const SampleFlow::AuxiliaryData &){samples << std::flush;});
    flush_sample.connect_to_producer(every_100th);

    // Keep track of the covariance matrix
    SampleFlow::Consumers::CovarianceMatrix<Sample> covariance_matrix;
    covariance_matrix.connect_to_producer(burn_in);

    // Count the number of samples
    SampleFlow::Consumers::CountSamples<Sample> count_samples;
    count_samples.connect_to_producer(burn_in);

    /*
     * The following is found in Algorithm 1 of https://doi.org/10.1111/anzs.12344
     * Robbins-Monro algorithm scales the covariance matrix to target a particular acceptance ratio
     * Each proposal is of the form: X_{n+1} ~ N(X_n, \lambda_n^2 * c * C_n)
     * where X_n are samples,
     * c = 2.38^2/dim,
     * C_n is the sample covariance matrix,
     * and \lambda_n^2 is the factor the Robbins-Monro algorithm updates.
     *
     * For \theta_n = log(\lambda_n), the update is defined as
     *    \theta_{n+1} = { \theta_n + \delta/n * (1-a), if nth trial is accepted
     *                   { \theta_n - \delta/n * a    , if nth trial is rejected
     * where \delta is a step-size constant stated below,
     * n is the sample number,
     * and a is the target acceptance ratio.
     * Thus, E(\theta_{n+1}-\theta_n) = 0 iff success probability is a. So an acceptance ratio of a is targeted.
     * From https://doi.org/10.1080/03610926.2014.936562
     *    \delta = (1 - 1/dim) * (A^{-1} * 2^{-1/2} * \sqrt(\pi) * exp(A^2/2)) + (dim * a * (1-a) )^{-1}
     *    where A = -f^{-1}(a/2) and f is the cdf of a standard normal distribution.
     */
    const Real c = std::pow(2.38,2.)/starting_guess.size();
    const Real target_ar = 0.234;
    // From Matlab command icdf('Normal',0.234/2,0,1)
    // If target_ar is changed, then this value also needs to be changed.
    const Real A = -1.19011804189642322882036751252599060535430908203125;
    const Real pi = 2 * std::acos(0.0); // arccos(0) = pi/2
    const Real delta = (1 - 1/starting_guess.size())
                       *(1/(A * std::sqrt(2.0)) * std::sqrt(pi) * std::exp(A*A/2.))
                       + 1/(starting_guess.size() * target_ar * (1-target_ar));
    Real lambda_start = 1;
    Real lambda = lambda_start;
    const Real lambda_min = 1e-2;
    // Sample to start at to avoid rapid changes in scale at the beginning
    int n_start = 5/( (1-target_ar)*target_ar );
    // If lambda exceeds lambda_start by this factor, then restart the Robbins-Monro algorithm
    const Real restart_factor = 3;

    // The final scale to multiply the covariance matrix by.
    Real scale = c * lambda * lambda;

    // theta = ln(lambda)
    Real theta = std::log(lambda);

    // Maximum times the Robbins-Monro algorithm is allowed to reset
    // Theoretical convergence results are with finite resets
    const unsigned int max_n_resets = 1e6;
    unsigned int times_reset = 0;

    // Apply the Robbins-Monro algorithm to the sampler
    SampleFlow::Filters::TakeEveryNth<Sample> robbins_monro(1);
    robbins_monro.connect_to_producer(mh_sampler);
    SampleFlow::Consumers::Action<Sample>
        rm_update(
        [&](const Sample &, const SampleFlow::AuxiliaryData & aux_data)
        {
          // Update the scale
          const bool repeated_sample = boost::any_cast<bool>(aux_data.at("sample is repeated"));
          if (repeated_sample)
            theta += (delta / ( n_start + count_samples.get() ) ) * (1 - target_ar);
          else
            theta -= (delta / ( n_start + count_samples.get() ) ) * target_ar;
          lambda = std::max(lambda_min, std::exp(theta));
          scale = lambda * lambda * c;
          // Check if the algorithm should be restarted
          if (times_reset < max_n_resets) {
            if (std::abs(std::log(lambda) - std::log(lambda_start)) > std::log(restart_factor)) {
              lambda_start = lambda;
              n_start = 5 / (target_ar * (1 - target_ar)) - count_samples.get();
              ++times_reset;
            }
          }
        }
    );
    rm_update.connect_to_producer(robbins_monro);



    const std::uint_fast32_t random_seed
        = (argc > 1 ?
           std::hash<std::string>()(std::to_string(atoi(argv[1]) + i)) :
           std::hash<std::string>()(std::to_string(i)));

    const unsigned int n_samples = samples_desired + burn_in_period;

    std::mt19937 rng;
    rng.seed(random_seed);
    auto sampler_perturb = [&](const Sample &s){
      if (count_samples.get() < 1)
        return perturb(s, starting_cov, rng);
      else
        return perturb(s, scale * (covariance_matrix.get() + std::max(0.001, 1.0/count_samples.get()) * starting_cov), rng);
    };
    mh_sampler.sample(starting_guess,
                      &log_probability,
                      sampler_perturb,
                      n_samples,
                      random_seed);

    std::cout << "Mean value of all samples: " << mean_value.get() << std::endl;
    std::cout << "MH acceptance ratio: " << acceptance_ratio.get() << std::endl;
  }
}