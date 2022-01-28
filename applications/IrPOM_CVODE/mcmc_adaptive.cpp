/*
 * A test to redo the sampling in the IrPOM paper but with the new ODE solver that uses SUNDIALS.
 * This is simply a validation that using the sampling methodology in the IrPOM alongside SUNDIALS
 * results in a similar posterior distribution.
 */


#include "sampling_sundials.h"
#include "src/ir_pom_data.h"

#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>
#include <sampleflow/consumers/covariance_matrix.h>

#include <omp.h>

#include <utility>
#include <fstream>
#include <vector>
#include <limits>


using Real = realtype;
using Sample = Eigen::Matrix<Real, 1, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IncompleteLUT<Real>>;



std::pair<Sample, Real>
perturb(const Sample & sample,
        std::mt19937 &rng)
{
  Vector bounds(sample.size());
  bounds << 3e2, 3e3, 3e3, 5e2, 10;
  Sample perturbation(sample.size());
  for (unsigned int i=0; i<perturbation.size()-1; ++i)
  {
    perturbation(i) = std::uniform_real_distribution<Real>(-1.0*bounds(i),bounds(i))(rng);
  }
  // Last parameter is an integer
  perturbation(perturbation.size()-1)
      = std::uniform_int_distribution<>(-1*bounds(perturbation.size()-1),bounds(perturbation.size()-1))(rng);
  Sample new_sample = sample + perturbation;
  return {new_sample, 1.};
}



bool
within_bounds(const Sample & sample)
{
  Vector lower_bounds(5);
  Vector upper_bounds(5);
  lower_bounds << 1000, 4800, 10, 10, 10;
  upper_bounds << 2e8, 1e8, 1e8, 1e8, 2000;

  if(sample(0) >= lower_bounds(0) &&
     sample(0) <= upper_bounds(0) &&
     sample(1) >= lower_bounds(1) &&
     sample(1) <= upper_bounds(1) &&
     sample(2) >= lower_bounds(2) &&
     sample(2) <= upper_bounds(2) &&
     sample(3) >= lower_bounds(3) &&
     sample(3) <= upper_bounds(3) &&
     sample(4) >= lower_bounds(4) &&
     sample(4) <= upper_bounds(4))
  {
    return true;
  }
  else
  {
    return false;
  }
}



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
  const unsigned int M = std::floor(sample(4));

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



Real
log_likelihood(const N_Vector solution_vector, const std::vector<Real> data_vector)
{
  // Reduce the solution vector to only the concentrations of particles
  auto only_particles = MEPBM::get_subset<Real>(solution_vector, 3, 2500);

  // Normalize the concentrations to make them a probability distribution
  auto particle_probability = MEPBM::normalize_concentrations(only_particles);

  // Convert to a std::vector to work with Histogram
  auto probability_vec = MEPBM::to_vector(particle_probability);

  // Create the diameter vector corresponding to the particle vector
  std::vector<unsigned int> particle_sizes(2500-3+1);
  unsigned int size = 3;
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
  const MEPBM::Parameters<Real> hist_prm(27, 1.4, 4.1);
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
  if (!within_bounds(sample))
    return -std::numeric_limits<Real>::infinity();

  // Get the data
  const MEPBM::PomData<Real> data;
  const std::vector<Real> times = {data.tem_time1, data.tem_time2, data.tem_time3, data.tem_time4};
  const std::vector<std::vector<Real>> tem_data = {data.tem_diam_time1, data.tem_diam_time2, data.tem_diam_time3, data.tem_diam_time4};

  // Create the ODE solver
  auto ic = MEPBM::create_eigen_nvector<Vector>(2501);
  auto ic_vec = static_cast<Vector*>(ic->content);
  (*ic_vec)(0) = 0.0012;
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(2501,2501);
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();
  const Real t_start = 0;
  const Real t_end = times.back();
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
  Real log_prob = 0;
  for (unsigned int i=0; i<times.size();++i)
  {
    auto sol = ode_solver.solve(times[i]);
    log_prob += log_likelihood(sol, tem_data[i]);
  }
  return log_prob;
}




int main(int argc, char **argv)
{
  unsigned int n_threads = 1;
#ifdef _OPENMP
  //n_threads = omp_get_max_threads();
#endif

#pragma omp parallel for
  for (unsigned int i=0;i<n_threads;++i)
  {
    const int prm_dim = 5;
    Sample starting_guess(prm_dim);
    starting_guess << 1.2e4, 1.8e5, 1.9e5, 1.7e4, 97;

    SampleFlow::Producers::MetropolisHastings<Sample> mh_sampler;

    std::ofstream samples("samples"
                          +
                          (argc > 1 ?
                           std::string(".") + std::to_string(atoi(argv[1])+i) :
                           std::string(".") + std::to_string(i))
                          +
                          ".txt"
    );

    SampleFlow::Consumers::StreamOutput<Sample> stream_output(samples);
    stream_output.connect_to_producer(mh_sampler);

    SampleFlow::Consumers::MeanValue<Sample> mean_value;
    mean_value.connect_to_producer(mh_sampler);

    SampleFlow::Consumers::AcceptanceRatio<Sample> acceptance_ratio;
    acceptance_ratio.connect_to_producer(mh_sampler);

    SampleFlow::Filters::TakeEveryNth<Sample> every_100th(100);
    every_100th.connect_to_producer(mh_sampler);
    SampleFlow::Consumers::Action<Sample>
        flush_sample([&samples](const Sample &, const SampleFlow::AuxiliaryData &){samples << std::flush;});
    flush_sample.connect_to_producer(every_100th);

    const std::uint_fast32_t random_seed
        = (argc > 1 ?
           std::hash<std::string>()(std::to_string(atoi(argv[1]) + i)) :
           std::hash<std::string>()(std::to_string(i)));

    const unsigned int n_samples = 5;

    std::mt19937 rng;
    rng.seed(random_seed);
    auto sampler_perturb = [&](const Sample &s){
      return perturb(s,rng);
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