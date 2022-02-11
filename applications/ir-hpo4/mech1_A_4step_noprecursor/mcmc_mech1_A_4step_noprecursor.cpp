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
#include <sampleflow/consumers/maximum_probability_sample.h>

#include <omp.h>

#include <utility>
#include <fstream>
#include <vector>
#include <limits>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <boost/math/special_functions/erf.hpp>


using Real = realtype;
using Sample = Eigen::Matrix<Real, 1, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
//using Matrix = Eigen::SparseMatrix<Real>;
using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IncompleteLUT<Real>>;
//using Solver = Eigen::SparseLU<Matrix>;


Real
inv_normal_cdf(const Real quantile, const Real mu, const Real sigma)
{
  return mu + sigma * std::sqrt(2.0) * boost::math::erf_inv(2 * quantile - 1);
}



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
  lower_bounds << 0, 0, 0, 0, 0, 0,10;
  upper_bounds << 1e3, 1e10, 1e10, 1e10, 1e10, 1e10, 60;

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
log_likelihood(const N_Vector solution_vector, const std::vector<Real> & data_vector)
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
  const std::vector<Real> tem_times = {MEPBM::HPO4Data<Real>::time1, MEPBM::HPO4Data<Real>::time2, MEPBM::HPO4Data<Real>::time3, MEPBM::HPO4Data<Real>::time4};
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
    if (t==3)
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

  /*
   * 1) Conduct a burn-in period
   */
  const int prm_dim = 7;
  Sample starting_guess(prm_dim);
  starting_guess << 2.6e-1, 2.e4, 2.2, 5.4e4, 1e3, 1.6e6, 50;

  Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> cov(prm_dim, prm_dim);
  cov <<
      std::pow(starting_guess(0)/3, 2), 0, 0, 0, 0, 0, 0,
      0, std::pow(starting_guess(1)/3, 2), 0, 0, 0, 0, 0,
      0, 0, std::pow(starting_guess(2)/3, 2), 0, 0, 0, 0,
      0, 0, 0, std::pow(starting_guess(3)/3, 2), 0, 0, 0,
      0, 0, 0, 0, std::pow(starting_guess(4)/3, 2), 0, 0,
      0, 0, 0, 0, 0, std::pow(starting_guess(5)/3, 2), 0,
      0, 0, 0, 0, 0, 0, std::pow(starting_guess(6)/3, 2);

  const unsigned int n_burn_in = 500;

  std::vector<Sample> max_prob_samples_burnin(n_threads);
  std::vector<Real> max_prob_values(n_threads);

#pragma omp parallel for
  for (unsigned int i=0; i<n_threads; ++i)
  {
    SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

    // Keep track of the maximum likelihood sample so we can start the next step with that sample.
    SampleFlow::Consumers::MaximumProbabilitySample<Sample> max_prob;
    max_prob.connect_to_producer(mh_sampler);

    SampleFlow::Consumers::AcceptanceRatio<Sample> ar_burnin;
    ar_burnin.connect_to_producer(mh_sampler);

    const std::uint_fast32_t random_seed
        = (argc > 1 ?
           std::hash<std::string>()(std::to_string(atoi(argv[1]) + i)) :
           std::hash<std::string>()(std::to_string(i)));

    std::mt19937 rng;
    rng.seed(random_seed);

    auto sampler_perturb = [&](const Sample &s){
        return perturb(s, cov, rng);
    };

    mh_sampler.sample(starting_guess,
                      &log_probability,
                      sampler_perturb,
                      n_burn_in,
                      random_seed);

    // Report the maximum probability sample
    max_prob_samples_burnin[i] = max_prob.get().first;
    max_prob_values[i] = boost::any_cast<Real>(max_prob.get().second.at("relative log likelihood"));
    std::cout << "AR: " << ar_burnin.get() << std::endl;
  }
  auto max_prob_iterator = std::max_element(max_prob_values.begin(), max_prob_values.end());
  auto max_prob_index = std::distance(max_prob_values.begin(), max_prob_iterator);
  starting_guess = max_prob_samples_burnin[max_prob_index];
  std::cout << "After burn-in: sample [" << starting_guess << "] has the maximum probability of " << max_prob_values[max_prob_index] << std::endl;


  /*
   * 2) Construct an initial estimate of the covariance matrix
   */
  unsigned int n_iter = 0;
  const unsigned int max_iter = 10;
  Real upper_scale = 2;
  Real lower_scale = 0;
  Real scale = (upper_scale + lower_scale)/2;

  const Real max_ar = 0.4;
  const Real min_ar = 0.1;
  Real avg_ar = 0.;

  const unsigned int n_adapt = 500;
  while ( (n_iter < max_iter) && (avg_ar < min_ar || avg_ar > max_ar) )
  {
    std::vector<Sample> max_prob_samples_adapt(n_threads);
    std::vector<Real> max_prob_value_adapt(n_threads);
    std::vector<Real> ar_adapt(n_threads);

    // Share the covariance matrix between chains for the most information
    SampleFlow::Consumers::CovarianceMatrix<Sample> cov_adaptation;
#pragma omp parallel for
    for (unsigned int i=0; i<n_threads; ++i)
    {
      SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

      cov_adaptation.connect_to_producer(mh_sampler);

      SampleFlow::Consumers::AcceptanceRatio<Sample> acceptance_ratio;
      acceptance_ratio.connect_to_producer(mh_sampler);

      SampleFlow::Consumers::MaximumProbabilitySample<Sample> max_prob;
      max_prob.connect_to_producer(mh_sampler);

      SampleFlow::Consumers::CountSamples<Sample> count_samples;
      count_samples.connect_to_producer(mh_sampler);


      const std::uint_fast32_t random_seed
          = (argc > 1 ?
             std::hash<std::string>()(std::to_string(atoi(argv[1]) + i)) :
             std::hash<std::string>()(std::to_string(i)));

      std::mt19937 rng;
      rng.seed(random_seed);

      auto sampler_perturb = [&](const Sample &s){
        return perturb(s, scale * cov, rng);
      };

      mh_sampler.sample(starting_guess,
                        &log_probability,
                        sampler_perturb,
                        n_adapt,
                        random_seed);

      max_prob_samples_adapt[i] = max_prob.get().first;
      max_prob_value_adapt[i] = boost::any_cast<Real>(max_prob.get().second.at("relative log likelihood"));
      ar_adapt[i] = acceptance_ratio.get();
    }
    // Get the average AR
    avg_ar = std::accumulate(ar_adapt.begin(), ar_adapt.end(), 0.0)/ ar_adapt.size();
    std::cout << "Average AR: " << avg_ar << std::endl;
    if (avg_ar < min_ar)
      upper_scale = scale;
    else if (avg_ar > max_ar)
      lower_scale = scale;
    else
      cov = cov_adaptation.get();

    scale = (upper_scale + lower_scale)/2;
    std::cout << "Covariance scale: " << scale << std::endl;

    // Increase iteration count
    ++n_iter;
    if (n_iter == max_iter)
      cov = cov_adaptation.get();

    // Start next iteration at the best sample
    auto max_iter_adapt = std::max_element(max_prob_value_adapt.begin(), max_prob_value_adapt.end());
    auto max_index_adapt = std::distance(max_prob_value_adapt.begin(), max_iter_adapt);
    starting_guess = max_prob_samples_adapt[max_prob_index];
    std::cout << "After adaptation: sample [" << starting_guess << "] has the maximum probability of " << max_prob_value_adapt[max_index_adapt] << std::endl;
  }

  std::cout << "Covariance matrix after adaptation:\n"
            << cov << std::endl;





/*
 * 3) Sample
 */
SampleFlow::Consumers::CovarianceMatrix<Sample> covariance_matrix;
#pragma omp parallel for
for (unsigned int i=0; i<n_threads; ++i)
{
  SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

  covariance_matrix.connect_to_producer(mh_sampler);

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
  stream_output.connect_to_producer(mh_sampler);

  // Keep track of the mean
  SampleFlow::Consumers::MeanValue<Sample> mean_value;
  mean_value.connect_to_producer(mh_sampler);

  // Keep track of the acceptance_ratio
  SampleFlow::Consumers::AcceptanceRatio<Sample> acceptance_ratio;
  acceptance_ratio.connect_to_producer(mh_sampler);

  // Track the number of samples
  SampleFlow::Consumers::CountSamples<Sample> count_samples;
  count_samples.connect_to_producer(mh_sampler);

  // Only write to disk every 100 samples
  SampleFlow::Filters::TakeEveryNth<Sample> every_100th(100);
  every_100th.connect_to_producer(mh_sampler);
  SampleFlow::Consumers::Action<Sample>
      flush_sample([&samples](const Sample &, const SampleFlow::AuxiliaryData &){samples << std::flush;});
  flush_sample.connect_to_producer(every_100th);

  const std::uint_fast32_t random_seed
      = (argc > 1 ?
         std::hash<std::string>()(std::to_string(atoi(argv[1]) + i)) :
         std::hash<std::string>()(std::to_string(i)));

  const unsigned int n_samples = 50000;

  std::mt19937 rng;
  rng.seed(random_seed);
  const Real c = 2.38 * 2.38 / starting_guess.size();
  auto sampler_perturb = [&](const Sample &s){
    if (count_samples.get() > 1)
      return perturb(s, c * covariance_matrix.get() + 0.01 * cov, rng);
    else
      return perturb(s, c * cov, rng);
  };

  mh_sampler.sample(starting_guess,
                    &log_probability,
                    sampler_perturb,
                    n_samples,
                    random_seed);

  std::stringstream msg;
  msg << "Acceptance ratio for chain " << i << ": " << acceptance_ratio.get() << "\n";
  std::cout << msg.str();
}


}