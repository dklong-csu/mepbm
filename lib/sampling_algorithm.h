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




}

#endif //MEPBM_SAMPLING_ALGORITHM_H
