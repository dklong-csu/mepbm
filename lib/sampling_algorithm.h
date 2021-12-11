#ifndef MEPBM_SAMPLING_ALGORITHM_H
#define MEPBM_SAMPLING_ALGORITHM_H


#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/filters/conversion.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>
#include <sampleflow/consumers/covariance_matrix.h>
#include <sampleflow/consumers/count_samples.h>

#include "perturb_sample.h"
#include "sundials_statistics.h"
#include "sampling_parameters.h"

#include <iostream>
#include <fstream>
#include <valarray>
#include <random>
#include <limits>


namespace Sampling
{
  /**
   * An enum describing the distribution for the proposal.
   */
   enum ProposalType {UniformProposal, NormalProposal};



   /**
    * An enum describing the algorithm used for the adaptive proposal distribution.
    */
    enum AdaptiveType {None};



  /**
   * An enum describing the distribution for the prior.
   */
   enum PriorType {UniformPrior, NormalPrior};



   /**
    * An enum describing the type of data used in the likelihood function
    */
    enum DataType {DataTEMOnly};



  /**
   * An object that contains the necessary information to generate a chain of samples.
   * This sampler uses the same proposal distribution for the entire sampling process.
   */
   template<typename RealType, typename Matrix, ProposalType Proposal, PriorType Prior, DataType Data, typename SolverType, LinearSolverClass SolverClass>
   class Sampler
   {
   public:
     /// Generates a chain of samples.
     void
     generate_samples(const unsigned int num_samples, std::ofstream &samples_file);
   };



   /**
    * Partial specialization of the Sample class for when a Uniform distribution is used.
    */
    template<typename RealType, typename Matrix, typename SolverType, LinearSolverClass SolverClass>
    class Sampler<RealType, Matrix, UniformProposal, UniformPrior, DataTEMOnly, SolverType, SolverClass>
    {
    public:
      /// Constructor
      Sampler(const Sample<RealType> &starting_guess,
              const std::vector<RealType> &perturb_magnitude_real,
              const std::vector<int> &perturb_magnitude_integer,
              const ModelingParameters<RealType, Matrix> &user_data);

      /// Generates a chain of samples.
      void
      generate_samples(const unsigned int num_samples,
                       std::ofstream &samples_file,
                       const std::uint_fast32_t &random_seed);

    private:
      const Sample<RealType> starting_guess;

      const std::vector<RealType> perturb_magnitude_real;
      const std::vector<int> perturb_magnitude_integer;
      const ModelingParameters<RealType, Matrix> user_data;

    };


    template<typename RealType, typename Matrix, typename SolverType, LinearSolverClass SolverClass>
    Sampler<RealType, Matrix, UniformProposal, UniformPrior, DataTEMOnly, SolverType, SolverClass>::Sampler(
        const Sample<RealType> &starting_guess,
        const std::vector<RealType> &perturb_magnitude_real,
        const std::vector<int> &perturb_magnitude_integer,
        const ModelingParameters<RealType, Matrix> &user_data)
      : starting_guess(starting_guess),
        perturb_magnitude_real(perturb_magnitude_real),
        perturb_magnitude_integer(perturb_magnitude_integer),
        user_data(user_data)
    {}


    template<typename RealType, typename Matrix, typename SolverType, LinearSolverClass SolverClass>
    void
    Sampler<RealType, Matrix, UniformProposal, UniformPrior, DataTEMOnly, SolverType, SolverClass>::generate_samples(
        const unsigned int num_samples,
        std::ofstream &samples_file,
        const std::uint_fast32_t &random_seed)
    {
      // Create the object to conduct the metropolis hastings (MH) algorithm
      SampleFlow::Producers::MetropolisHastings<Sample<RealType>> mh_sampler;

      // Tell the MH algorithm where to output samples
      SampleFlow::Consumers::StreamOutput<Sample<RealType>> stream_output(samples_file);
      stream_output.connect_to_producer(mh_sampler);

      // Since our sample object is complicated, we want to be able to turn it into a vector
      // compatible with computations whenever that is necessary
      SampleFlow::Filters::Conversion<Sample<RealType>, std::valarray<RealType> > convert_to_vector;
      convert_to_vector.connect_to_producer(mh_sampler);

      // Update the mean value of the samples during the process
      // Updating the mean value requires computations on the samples, so we need the filter created above
      SampleFlow::Consumers::MeanValue< std::valarray<RealType> > mean_value;
      mean_value.connect_to_producer(convert_to_vector);

      // Keep track of the acceptance ratio to see how efficient our sampling process was
      SampleFlow::Consumers::AcceptanceRatio< std::valarray<RealType> > acceptance_ratio;
      acceptance_ratio.connect_to_producer(convert_to_vector);

      // Write to disk only on occasion to reduce load on memory
      SampleFlow::Filters::TakeEveryNth<Sample<RealType>> every_100th(100);
      every_100th.connect_to_producer(mh_sampler);

      SampleFlow::Consumers::Action<Sample<RealType>>
          flush_after_every_100th([&samples_file](const Sample<RealType> &, const SampleFlow::AuxiliaryData &) {
        samples_file << std::flush;
      });

      flush_after_every_100th.connect_to_producer(every_100th);

      // Sample from the given distribution.
      // Since the prior distribution is uniform then we can simply check if the parameters are within the
      // prior bounds. If not, then we have a probability of 0. If we do, then the prior gives the same
      // contribution to the probability, so we only need to calculate the likelihood
      std::mt19937 rng;
      rng.seed(random_seed);
      mh_sampler.sample(starting_guess,
                        [&](const Sample<RealType> &s) {
                          if (sample_is_valid(s, user_data.real_parameter_bounds, user_data.integer_parameter_bounds) == false)
                            return -std::numeric_limits<RealType>::max();
                          else
                            return SUNDIALS_Statistics::compute_likelihood_TEM_only<RealType, Matrix, SolverType, SolverClass>(s, user_data);
                        },
                        [&](const Sample<RealType> &s) {
                          return perturb_uniform(s, rng, perturb_magnitude_real, perturb_magnitude_integer);
                        },
                        num_samples,
                        random_seed);

      // Output the statistics we have computed in the process of sampling
      // everything
      std::cout << "Mean value of all samples: ";
      for (auto x : mean_value.get())
        std::cout << x << ' ';
      std::cout << std::endl;
      std::cout << "MH acceptance ratio: "
                << acceptance_ratio.get()
                << std::endl;

    }






  /**
 * An object that contains the necessary information to generate a chain of samples.
 * This sampler employs an adaptive Metropolis-Hastings algorithm. Specifically, after a specified number of samples,
 * the proposal distribution (1-beta)*N(x_n, (2.38)^2/d * Sigma_n) + beta*N(x, gamma^2/d * I_d)
 * is used.
 */
  template<typename RealType, typename Matrix, PriorType Prior, DataType Data, typename SolverType, LinearSolverClass SolverClass>
  class AdaptiveMHSampler
  {
  public:
    /// Constructor
    AdaptiveMHSampler(const Sample<RealType> starting_guess,
                      SampleFlow::Consumers::CovarianceMatrix< std::valarray<RealType> >* sample_covariance_matrix,
                      const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> starting_covariance_matrix,
                      const ModelingParameters<RealType, Matrix> user_data,
                      const RealType beta,
                      const RealType gamma);

    /// Generates a chain of samples.
    void
    generate_samples(const unsigned int num_samples,
                     std::ofstream &samples_file,
                     const std::uint_fast32_t &random_seed,
                     const unsigned int adaptive_start_sample);

  private:
    const Sample<RealType> starting_guess;
    std::shared_ptr< SampleFlow::Consumers::CovarianceMatrix<std::valarray< RealType > > > sample_covariance_matrix;
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> starting_covariance_matrix;
    const ModelingParameters<RealType, Matrix> user_data;
    const RealType beta;
    const RealType gamma;
  };



  /// Partial specialization for TEM data only
  template<typename RealType, typename Matrix, PriorType Prior, typename SolverType, LinearSolverClass SolverClass>
  class AdaptiveMHSampler<RealType, Matrix, Prior, DataTEMOnly, SolverType, SolverClass>
  {
  public:
    /// Constructor
    AdaptiveMHSampler(const Sample<RealType> starting_guess,
                      std::shared_ptr< SampleFlow::Consumers::CovarianceMatrix<std::valarray< RealType > > > sample_covariance_matrix,
                      const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> starting_covariance_matrix,
                      const ModelingParameters<RealType, Matrix> user_data,
                      const RealType beta,
                      const RealType gamma);


    /// Generates a chain of samples.
    void
    generate_samples(const unsigned int num_samples,
                     std::ofstream &samples_file,
                     const std::uint_fast32_t &random_seed,
                     const unsigned int adaptive_start_sample);

  private:
    const Sample<RealType> starting_guess;
    std::shared_ptr< SampleFlow::Consumers::CovarianceMatrix<std::valarray< RealType > > > sample_covariance_matrix;
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> starting_covariance_matrix;
    const ModelingParameters<RealType, Matrix> user_data;
    const RealType beta;
    const RealType gamma;
  };



  template<typename RealType, typename Matrix, PriorType Prior, typename SolverType, LinearSolverClass SolverClass>
  AdaptiveMHSampler<RealType, Matrix, Prior, DataTEMOnly, SolverType, SolverClass>::AdaptiveMHSampler(
      const Sample<RealType> starting_guess,
      std::shared_ptr< SampleFlow::Consumers::CovarianceMatrix<std::valarray< RealType > > > sample_covariance_matrix,
      const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> starting_covariance_matrix,
      const ModelingParameters<RealType, Matrix> user_data,
      const RealType beta,
      const RealType gamma)
      : starting_guess(starting_guess),
        sample_covariance_matrix(sample_covariance_matrix),
        starting_covariance_matrix(starting_covariance_matrix),
        user_data(user_data),
        beta(beta),
        gamma(gamma)
      {}



  template<typename RealType, typename Matrix, PriorType Prior, typename SolverType, LinearSolverClass SolverClass>
  void AdaptiveMHSampler<RealType, Matrix, Prior, DataTEMOnly, SolverType, SolverClass>::generate_samples(
      const unsigned int num_samples,
      std::ofstream &samples_file,
      const std::uint_fast32_t &random_seed,
      const unsigned int adaptive_start_sample)
  {
    // Create the object to conduct the metropolis hastings (MH) algorithm
    SampleFlow::Producers::MetropolisHastings<Sample<RealType>> mh_sampler;

    // Tell the MH algorithm where to output samples
    // FIXME replace with MyStreamOutput so I can calculate Bayes Factor
    SampleFlow::Consumers::StreamOutput<Sample<RealType>> stream_output(samples_file);
    stream_output.connect_to_producer(mh_sampler);

    // Since our sample object is complicated, we want to be able to turn it into a vector
    // compatible with computations whenever that is necessary
    SampleFlow::Filters::Conversion<Sample<RealType>, std::valarray<RealType> > convert_to_vector;
    convert_to_vector.connect_to_producer(mh_sampler);

    // Update the mean value of the samples during the process
    // Updating the mean value requires computations on the samples, so we need the filter created above
    SampleFlow::Consumers::MeanValue< std::valarray<RealType> > mean_value;
    mean_value.connect_to_producer(convert_to_vector);

    // Update the covariance matrix of the samples during the process
    // Updating the covariance matrix requires computations on the samples, so we need the filter created above
    sample_covariance_matrix->connect_to_producer(convert_to_vector);

    // Keep track of the acceptance ratio to see how efficient our sampling process was
    SampleFlow::Consumers::AcceptanceRatio< std::valarray<RealType> > acceptance_ratio;
    acceptance_ratio.connect_to_producer(convert_to_vector);

    // Write to disk only on occasion to reduce load on memory
    SampleFlow::Filters::TakeEveryNth<Sample<RealType>> every_100th(100);
    every_100th.connect_to_producer(mh_sampler);

    SampleFlow::Consumers::Action<Sample<RealType>>
        flush_after_every_100th([&samples_file](const Sample<RealType> &, const SampleFlow::AuxiliaryData &) {
      samples_file << std::flush;
    });

    flush_after_every_100th.connect_to_producer(every_100th);

    // Count the number of samples so I know when to perform adaptations.
    SampleFlow::Consumers::CountSamples<Sample<RealType>> sample_count;
    sample_count.connect_to_producer(mh_sampler);

    // Sample from the given distribution.
    // Since the prior distribution is uniform then we can simply check if the parameters are within the
    // prior bounds. If not, then we have a probability of 0. If we do, then the prior gives the same
    // contribution to the probability, so we only need to calculate the likelihood
    std::mt19937 rng;
    rng.seed(random_seed);
    auto problem_dimension = starting_guess.real_valued_parameters.size() + starting_guess.integer_valued_parameters.size();

    auto log_probability = [&](const Sample<RealType> &s) {
      if (sample_is_valid(s, user_data.real_parameter_bounds, user_data.integer_parameter_bounds) == false)
        return -std::numeric_limits<RealType>::max();
      else
        return SUNDIALS_Statistics::compute_likelihood_TEM_only<RealType, Matrix, SolverType, SolverClass>(s, user_data);
    };


    auto proposal_distribution = [&](const Sample<RealType> &s) {
      if (sample_count.get() < adaptive_start_sample)
      {
        return perturb_normal(s, rng, starting_covariance_matrix, 1.);
      }
      else
      {
        auto sample_x = perturb_normal(s, rng, sample_covariance_matrix->get(), 2.38/std::sqrt(problem_dimension));
        const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> identity_matrix
          = Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic>::Identity(problem_dimension, problem_dimension);
        auto sample_y = perturb_normal(s, rng, identity_matrix , gamma/std::sqrt(problem_dimension));
        std::vector<RealType> perturbed_real_prm;
        for (unsigned int i=0; i<sample_x.first.real_valued_parameters.size(); ++i)
        {
          auto prm = (1-beta)*sample_x.first.real_valued_parameters[i] + beta*sample_y.first.real_valued_parameters[i];
          perturbed_real_prm.push_back(prm);
        }

        std::vector<int> perturbed_int_prm;
        for (unsigned int i=0; i<sample_x.first.integer_valued_parameters.size(); ++i)
        {
          auto prm = (1-beta)*sample_x.first.integer_valued_parameters[i] + beta*sample_y.first.integer_valued_parameters[i];
          perturbed_int_prm.push_back(prm);
        }
        Sample<RealType> perturbed_sample(perturbed_real_prm, perturbed_int_prm);
        std::pair<Sample<RealType>, RealType> sample_and_ratio = {perturbed_sample, 1.};
        return sample_and_ratio;
      }
    };



    mh_sampler.sample(starting_guess,
                      log_probability,
                      proposal_distribution,
                      num_samples,
                      random_seed);

    // Output the statistics we have computed in the process of sampling
    // everything
    std::cout << "Mean value of all samples: ";
    for (auto x : mean_value.get())
      std::cout << x << ' ';
    std::cout << std::endl;
    std::cout << "MH acceptance ratio: "
              << acceptance_ratio.get()
              << std::endl;
  }









}

#endif //MEPBM_SAMPLING_ALGORITHM_H
