#ifndef MEPBM_SAMPLING_ALGORITHM_H
#define MEPBM_SAMPLING_ALGORITHM_H


#include <sampleflow/producers/metropolis_hastings.h>
#include <sampleflow/producers/differential_evaluation_mh.h>
#include <sampleflow/filters/conversion.h>
#include <sampleflow/filters/take_every_nth.h>
#include <sampleflow/consumers/stream_output.h>
#include <sampleflow/consumers/mean_value.h>
#include <sampleflow/consumers/acceptance_ratio.h>
#include <sampleflow/consumers/action.h>
#include <sampleflow/consumers/covariance_matrix.h>
#include <sampleflow/consumers/count_samples.h>
#include <sampleflow/consumers/last_sample.h>

#include "src/perturb_sample.h"
#include "sundials_statistics.h"
#include "sampling_parameters.h"

#include <iostream>
#include <fstream>
#include <valarray>
#include <random>
#include <limits>


namespace Sampling
{
  // Like the StreamOutput class, but not only output the sample itself
  // but also its properties (as recorded by the AuxiliaryData object)
  template <typename InputType>
  class MyStreamOutput : public SampleFlow::Consumer<InputType>
  {
  public:
    MyStreamOutput (std::ostream &output_stream)
        :
        output_stream (output_stream)
    {}

    ~MyStreamOutput ()
    {
      this->disconnect_and_flush();
    }


    virtual
    void
    consume (InputType sample, SampleFlow::AuxiliaryData aux_data) override
    {
      std::lock_guard<std::mutex> lock(mutex);

      output_stream << "Sample: " << sample << std::endl;
      for (const auto &data : aux_data)
      {
        // Output the key of each pair:
        output_stream << "   " << data.first;

        // Then see if we can interpret the value via a known type:
        if (const bool *p = boost::any_cast<bool>(&data.second))
          output_stream << " -> " << (*p ? "true" : "false") << std::endl;
        else if (const double *p = boost::any_cast<double>(&data.second))
          output_stream << " -> " << *p << std::endl;
        else
          output_stream << std::endl;
      }
    }


  private:
    mutable std::mutex mutex;
    std::ostream &output_stream;
  };



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






  /**
   * An object to run the Adaptive Metropolis-Hastings sampling algorithm
   */
   template<typename Real, typename SampleType>
   class AdaptiveMetropolisSampler
   {
   public:
     /// Constructor
     AdaptiveMetropolisSampler(const std::function<Real(const SampleType &)> log_likelihood,
                       const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> covariance,
                       const int chain_num)
                       : log_likelihood(log_likelihood), covariance(covariance), chain_num(chain_num)
     {}


     /// Function to run the burn-in period for a chain.
     SampleType
     burn_in(const SampleType & starting_sample, const unsigned int n_burn_in, const std::uint_fast32_t &random_seed)
     {
       // The only thing we care about is the last sample produced
       SampleFlow::Producers::MetropolisHastings<SampleType> mh_sampler;
       SampleFlow::Consumers::LastSample<SampleType> last_sample;
       last_sample.connect_to_producer(mh_sampler);

       // Run the burn-in period
       std::mt19937 rng;
       rng.seed(random_seed);
       const auto proposal = [&](const SampleType &s){
         return perturb_normal(s, rng, covariance, 1.);
       };
       mh_sampler.sample(starting_sample,
                         log_likelihood,
                         proposal,
                         n_burn_in,
                         random_seed);

       return last_sample.get();
     }


     /// Function to generate samples
     void
     sample(const SampleType & starting_sample,
            const unsigned int n_samples,
            const unsigned int adaptation_period,
            const unsigned int adaptation_frequency,
            const std::uint_fast32_t & random_seed)
     {
       // First perform the adaptation period
       // Require the acceptance ratio be within a certain range, otherwise adjust the covariance and try again

       bool adaptation_finished = false;
       SampleType first_sample = starting_sample;
       int adaptation_attempts = 0;

       if (adaptation_period == 0)
         adaptation_finished = true;

       while (!adaptation_finished) {
         SampleFlow::Producers::MetropolisHastings<SampleType> mh_sampler_adaptation;

         // Since our sample object is complicated, we want to be able to turn it into a vector
         // compatible with computations whenever that is necessary
         SampleFlow::Filters::Conversion<SampleType, std::valarray<Real> > convert_to_vector;
         convert_to_vector.connect_to_producer(mh_sampler_adaptation);

         SampleFlow::Consumers::LastSample<SampleType> last_sample;
         last_sample.connect_to_producer(mh_sampler_adaptation);

         // Keep track of the acceptance ratio to see how efficient our sampling process was
         SampleFlow::Consumers::AcceptanceRatio< std::valarray<Real> > acceptance_ratio;
         acceptance_ratio.connect_to_producer(convert_to_vector);

         std::mt19937 rng;
         rng.seed(random_seed);
         const auto proposal = [&](const SampleType &s){
           return perturb_normal(s, rng, covariance, 1.);
         };

         mh_sampler_adaptation.sample(first_sample,
                           log_likelihood,
                           proposal,
                           adaptation_period,
                           random_seed);

         // FIXME: think about making this an input parameter
         const Real adaptation_ar_lower_bound = .2;
         const Real adaptation_ar_upper_bound = .5;
         // FIXME: enforce max number of redo attempts and abort if limit is reached
         // if the acceptance ratio is too small then reduce the covariance and try again
         if (acceptance_ratio.get() < adaptation_ar_lower_bound)
         {
           covariance *= 0.1;
         }
         // if the acceptance ratio too large then increase the covariance and try again
         else if (acceptance_ratio.get() > adaptation_ar_upper_bound)
         {
           covariance *= 10;
         }
         // otherwise the adaptation period is over
         else
         {
           adaptation_finished = true;
         }
         // Regardless of outcome, set the first sample of the next sampling procedure to the most recent sample
         first_sample = last_sample.get();
         ++adaptation_attempts;
         if (adaptation_attempts > 1)
         {
           // Don't waste too much time on the adaptation period. Move on if too many attempts have been made.
           adaptation_finished = true;
         }
       }
       std::cout << "Chain " << chain_num << " is ready for AMH after  " << adaptation_attempts << " adaptation cycles." << std::endl;


       //Perform the sampling now that the adaptation period is finished
       SampleFlow::Producers::MetropolisHastings<SampleType> mh_sampler;

       // Tell the MH algorithm where to output samples
       // FIXME replace with MyStreamOutput so I can calculate Bayes Factor
       std::ofstream samples_file("samples."
                                  +
                                  std::to_string(chain_num)
                                  +
                                  ".txt");

       //MyStreamOutput<SampleType> stream_output(samples_file);
       SampleFlow::Consumers::StreamOutput<SampleType> stream_output(samples_file);
       stream_output.connect_to_producer(mh_sampler);

       // Since our sample object is complicated, we want to be able to turn it into a vector
       // compatible with computations whenever that is necessary
       SampleFlow::Filters::Conversion<SampleType, std::valarray<Real> > convert_to_vector;
       convert_to_vector.connect_to_producer(mh_sampler);

       // Update the mean value of the samples during the process
       // Updating the mean value requires computations on the samples, so we need the filter created above
       SampleFlow::Consumers::MeanValue< std::valarray<Real> > mean_value;
       mean_value.connect_to_producer(convert_to_vector);

       // Update the covariance matrix of the samples during the process
       // Updating the covariance matrix requires computations on the samples, so we need the filter created above
       SampleFlow::Consumers::CovarianceMatrix< std::valarray<Real> > sample_covariance;
       sample_covariance.connect_to_producer(convert_to_vector);

       // Keep track of the acceptance ratio to see how efficient our sampling process was
       SampleFlow::Consumers::AcceptanceRatio< std::valarray<Real> > acceptance_ratio;
       acceptance_ratio.connect_to_producer(convert_to_vector);

       // Write to disk only on occasion to reduce load on memory
       SampleFlow::Filters::TakeEveryNth<SampleType> every_100th(100);
       every_100th.connect_to_producer(mh_sampler);

       SampleFlow::Consumers::Action<SampleType>
           flush_after_every_100th([&samples_file](const SampleType &, const SampleFlow::AuxiliaryData &) {
         samples_file << std::flush;
       });

       flush_after_every_100th.connect_to_producer(every_100th);

       // Count the number of samples so I know when to perform adaptations.
       //SampleFlow::Consumers::CountSamples<SampleType> sample_count;
       //sample_count.connect_to_producer(mh_sampler);

       // How frequently should the covariance matrix be updated with new adaptation
       SampleFlow::Filters::TakeEveryNth<SampleType> adaptation_update(adaptation_frequency);
       adaptation_update.connect_to_producer(mh_sampler);

       SampleFlow::Consumers::Action<SampleType>
           update_covariance_matrix([&](const SampleType &, const SampleFlow::AuxiliaryData &){

             covariance = sample_covariance.get()*2.38*2.38/(covariance.rows());
             // FIXME: consider making these bounds user input
             // FIXME: consider making covariance modifier stack
             /*if (acceptance_ratio.get() < 0.15)
               covariance *= 0.1;
             else if (acceptance_ratio.get() > 0.4)
               covariance *= 10;*/
           }
       );

       update_covariance_matrix.connect_to_producer(adaptation_update);

       // Run the sampler
       std::mt19937 rng;
       rng.seed(random_seed);
       const auto proposal = [&](const SampleType &s){
         return perturb_normal(s, rng, covariance, 1.);
       };

       mh_sampler.sample(starting_sample,
                         log_likelihood,
                         proposal,
                         n_samples,
                         random_seed);
       std::cout << "Chain " << chain_num << " acceptance ratio: " << acceptance_ratio.get() << std::endl;
     }



     // Function to run a burn-in and then generate samples
     void
     sample_with_burn_in(const SampleType & starting_sample,
                         const unsigned int n_burn_in,
                         const unsigned int n_samples,
                         const unsigned int adaptation_period,
                         const unsigned int adaptation_frequency,
                         const std::uint_fast32_t & random_seed)
     {
       if (n_burn_in > 0) {
         auto s = burn_in(starting_sample,
                          n_burn_in,
                          random_seed);
         sample(s,
                n_samples,
                adaptation_period,
                adaptation_frequency,
                random_seed);
       }
       else {
         sample(starting_sample,
                n_samples,
                adaptation_period,
                adaptation_frequency,
                random_seed);
       }
     }


   private:
     const std::function<Real(const SampleType &)> log_likelihood;
     Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> covariance;
     const int chain_num;
   };



  /**
   * An object to run the Differential Evolution sampling algorithm
   */
   template<typename Real, typename SampleType>
   class DifferentialEvolutionSampler
   {
   public:
     DifferentialEvolutionSampler(const std::function<Real(const SampleType &)> log_likelihood,
                                  const std::function< std::pair< SampleType,Real >(const SampleType &) > perturb,
                                  const std::function< SampleType(const SampleType &, const SampleType &, const SampleType &, const Real gamma) > crossover,
                                  std::string file_name)
                                  : log_likelihood(log_likelihood),
                                    perturb(perturb),
                                    crossover(crossover),
                                    file_name(file_name)
     {
        conversion.connect_to_producer(sampler);
        mean_value.connect_to_producer(conversion);
        sample_count.connect_to_producer(sampler);
        cov.connect_to_producer(conversion);
     }

     void generate_samples(std::vector<SampleType> & starting_samples,
                           const unsigned int crossover_gap,
                           const unsigned int n_samples)
     {
       std::ofstream samples_file(file_name);
       MyStreamOutput<SampleType> output_file(samples_file);
       output_file.connect_to_producer(sampler);

       auto crossover_sampler = [&](const SampleType & s, const SampleType & s1, const SampleType & s2) {
         // FIXME: maybe make the 10 a variable
         if (sample_count.get() % 10 == 0)
         {
           return crossover(s, s1, s2, 1.0);
         }
         else
         {
           return crossover(s, s1, s2, 2.38/std::sqrt(2.0 * s.get_dimension()));
         }
       };

       sampler.sample(starting_samples,
                      log_likelihood,
                      perturb,
                      crossover_sampler,
                      crossover_gap,
                      n_samples);

       std::cout << "Mean value of all samples: ";
       for (auto x : mean_value.get())
         std::cout << x << ' ';
       std::cout << std::endl;
       std::cout << "Covariance matrix of samples in all chains:\n";
       std::cout << cov.get() << std::endl;
     }

   private:
     const std::function<Real(const SampleType &)> log_likelihood;
     const std::function< std::pair< SampleType,Real >(const SampleType &) > perturb;
     const std::function< SampleType(const SampleType &, const SampleType &, const SampleType &, const Real &) > crossover;
     SampleFlow::Producers::DifferentialEvaluationMetropolisHastings<SampleType> sampler;
     SampleFlow::Filters::Conversion<SampleType, std::valarray<Real>> conversion;
     SampleFlow::Consumers::MeanValue<std::valarray<Real>> mean_value;
     const std::string file_name;
     SampleFlow::Consumers::CountSamples<Sample<Real>> sample_count;
     SampleFlow::Consumers::CovarianceMatrix<std::valarray<Real>> cov;
   };













}

#endif //MEPBM_SAMPLING_ALGORITHM_H
