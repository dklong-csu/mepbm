#ifndef MEPBM_SUNDIALS_STATISTICS_H
#define MEPBM_SUNDIALS_STATISTICS_H

#include "sample.h"
#include "sundials_solvers.h"
#include "linear_solver_eigen.h"
#include "nvector_eigen.h"
#include "sunmatrix_eigen.h"
#include <vector>
#include "histogram.h"
#include <iostream>
#include <stdexcept>
#include "sampling_parameters.h"



namespace SUNDIALS_Statistics {
  namespace Internal
  {
    /**
     * A function to solve the ODE associated with a sample
     */
    template<typename RealType, typename Matrix>
    std::vector<N_Vector>
    solve_ODE_from_sample(const Sampling::Sample<RealType> &sample,
                          const Sampling::ModelingParameters<RealType, Matrix> &user_data)
    {
      // Create ODE solver
      auto vector_template = user_data.initial_condition->ops->nvclone(user_data.initial_condition);
      auto N = user_data.initial_condition->ops->nvgetlength(user_data.initial_condition);
      auto matrix_template = create_eigen_sunmatrix<Matrix>(N,N);

      auto linear_solver = create_eigen_linear_solver<Matrix, realtype>();

      sundials::CVodeParameters<RealType> param(user_data.start_time,
                                                user_data.end_time,
                                                user_data.abs_tol,
                                                user_data.rel_tol,
                                                CV_BDF);

      auto ode_system = user_data.create_model(sample.real_valued_parameters, sample.integer_valued_parameters);

      sundials::CVodeSolver<Matrix, RealType> ode_solver(param,
                                                         ode_system,
                                                         user_data.initial_condition,
                                                         vector_template,
                                                         matrix_template,
                                                         linear_solver);

      // Solve the ODE
      std::vector<N_Vector> solutions;
      ode_solver.solve_ode_incrementally(solutions, user_data.times);

      return solutions;
    }



    /**
     * A function to take in an N_Vector and return a std::vector.
     */
    template<typename RealType>
    std::vector<RealType>
    convert_solution_to_vector(N_Vector sol)
    {
      auto n = sol->ops->nvgetlength(sol);
      std::vector<RealType> vec(n);

      auto sol_array_pointer = sol->ops->nvgetarraypointer(sol);

      for (unsigned int i=0; i<n; ++i)
      {
        vec[i] = (sol_array_pointer)[i];
      }

      return vec;
    }



    /**
     * A function to strip a vector of elements that are not particles.
     * It is assumed that the indices corresponding to particles are stored contiguously.
     */
     template<typename RealType>
     std::vector<RealType>
     strip_nanoparticles_from_vector(const std::vector<RealType> &vec,
                                     const unsigned int start_index,
                                     const unsigned int end_index)
    {
       std::vector<RealType> vec_particles(end_index - start_index + 1);
       for (unsigned int i=start_index; i<=end_index; ++i)
         vec_particles[i - start_index] = vec[i];

       return vec_particles;
    }



    /**
     * A function to bin a vector of concentrations into a probability mass function (PMF).
     * In this context, negative values are unphysical and seen as a limitation in the simulation.
     * From a modeling perspective, it is more accurate to convert negative values to a small number.
     */
     template<typename RealType>
     Histograms::Histogram<RealType>
     convert_concentrations_to_pmf(const std::vector<RealType> &concentrations,
                                   const Histograms::Parameters<RealType> &histogram_parameters,
                                   const std::vector<RealType> &particle_sizes)
    {
      // Make sure the concentration and size vectors are the same length
      if (concentrations.size() != particle_sizes.size())
        throw std::domain_error("The concentrations and particle_sizes vectors are different lengths.");

      // Find the maximum concentration
      RealType max_conc = 0;
      for (auto conc : concentrations)
      {
        max_conc = std::max(max_conc, conc);
      }

      // Find the sum of the vector components with non-positive numbers adjusted to be more physically correct.
      RealType norm = 0;
      for (auto conc : concentrations)
      {
        if (conc > 0)
        {
          norm += conc;
        }
        else
        {
          norm += max_conc * 1e-9;
        }
      }

      // Create a pmf
      std::vector<RealType> pmf(concentrations.size());
      for (unsigned int i=0; i<pmf.size(); ++i)
      {
        if (concentrations[i] < 0)
        {
          pmf[i] = 1e-9 * max_conc / norm;
        }
        else
        {
          pmf[i] = concentrations[i] / norm;
        }
      }

      // Create the binned pmf in the form of a histogram
      Histograms::Histogram<RealType> histogram(histogram_parameters);
      histogram.AddToBins(pmf, particle_sizes);
      return histogram;
    }



    /**
     * A function to take in the particle size (i.e. number of atoms) and convert to a diameter measurement.
     */
    template<typename RealType>
    RealType
    convert_particle_size_to_diameter(const int num_atoms)
    {
      // multiply num_atoms by 1. to make sure the RealType implementation is used instead of integer.
      return 0.3000805 * std::pow(1.*num_atoms, 1./3);
    }


    namespace TEMData
    {
      /**
       * A function to compute the likelihood based a single set of TEM data after the data has already been binned.
       */
       template<typename RealType>
       RealType
       compute_likelihood_from_binned_data(const Histograms::Histogram<RealType> &measurements,
                                           const Histograms::Histogram<RealType> &probabilities)
      {
         // Check to make sure both histograms have the same number of bins
         if (measurements.num_bins != probabilities.num_bins)
           throw std::domain_error("There must be an equal number of measurement and probability bins.");


         // Loop through each bin to accumulate likelihood
         RealType likelihood = 0;
         for (unsigned int i=0; i<measurements.num_bins; ++i)
         {
           // If no measured data, then no likelihood contribution
           if (measurements.count[i] > 0)
           {
             // If the bin probability is 0 but there are measurements, then the likelihood should be as small as possible
             if (probabilities.count[i] <= 0)
               return -std::numeric_limits<RealType>::max();

             // Otherwise, perform a normal likelihood calculation
             likelihood += measurements.count[i]*std::log(probabilities.count[i]);
           }
         }

         return likelihood;
      }



      /**
       * A function to bin TEM data into measurement counts.
       */
       template<typename RealType>
       Histograms::Histogram<RealType>
       bin_TEM_data(const std::vector<RealType> &data,
                    const Histograms::Parameters<RealType> &histogram_parameters)
      {
        Histograms::Histogram<RealType> histogram(histogram_parameters);
        std::vector<RealType> data_counts(data.size(), 1.0); // data points occur 1 time
        histogram.AddToBins(data_counts, data);
        return histogram;
      }

    }
  }



  /**
   * A function to calculate the log likelihood of a sample based solely on the particle size distribution (PSD) data.
   * For the \f$i\f$th set of data and parameters \f$K\f$ we have the likelihood function is
   * \f[
   *      L_i(K) = N_i \prod_{\ell=1}^{N_\text{bins}} (p_{i,\ell}(K))^{\beta_{i,\ell}^\text{measured}
   * \f]
   * where \f$N_i\f$ is some normalization constant that is of no consequence for our algorithms; \f$N_\text{bins}\f$
   * is the number of bins the data is aggregated into; \f$p_{i,\ell}\f$ is the probability of finding a particle in
   * bin \f$\ell\f$ in data set \f$i\f$ as a function of the parameters \f$K\f$; and \f$\beta_{i,\ell}^\text{measured}\f$
   * is the number of measured particles in data set \f$i\f$ in bin \f$\ell\f$. We don't care about \f$N_i\f$ for our
   * algorithms, so we drop it. Moreover, these sorts of computations are easier as logarithms, so we consider
   * \f[
   *    \log L_i(K) = \sum_[\ell=1}^{N_\text{bins}} \beta_{i,\ell}^\text{measured} \log p_{i,\ell}(K).
   * \f]
   * We assume independence between data sets, so the product of \f$L_i\f$ gives the full likelihood. Or, using logs,
   * \f[
   *    \log L(K) = \sum_{i=1}^{N_\text{datasets}} \log L_i(K).
   * \f]
   * This function takes in a Sample (which contains all of the information about bins, measured data, parameters, etc.)
   * and solves the ODE to calculate all \f$p_{i,\ell}\f$. Then \f$\log L(K)\f$ is calculated as described above. The
   * output is of type RealType.
   */
  template<typename RealType, typename Matrix>
  RealType compute_likelihood_TEM_only(const Sampling::Sample<RealType> &sample,
                                       const Sampling::ModelingParameters<RealType, Matrix> &user_data)
  {
    // Solve the ODE
    std::vector<N_Vector> solutions = Internal::solve_ODE_from_sample<RealType, Matrix>(sample, user_data);

    // Turn ODE solution(s) into a distribution
    std::vector< Histograms::Histogram<RealType> > probabilities;

    std::vector<RealType> particle_diameters(user_data.last_particle_index - user_data.first_particle_index + 1);
    for (unsigned int i=0; i<particle_diameters.size(); ++i)
    {
      particle_diameters[i]
          = Internal::convert_particle_size_to_diameter<RealType>(user_data.first_particle_size + i*user_data.particle_size_increase);
    }

    for (unsigned int i=0; i<solutions.size(); ++i)
    {
      auto sol = Internal::convert_solution_to_vector<RealType>(solutions[i]);
      auto concentrations = Internal::strip_nanoparticles_from_vector<RealType>(sol,
                                                                                user_data.first_particle_index,
                                                                                user_data.last_particle_index);
      auto p = Internal::convert_concentrations_to_pmf(concentrations,
                                                       user_data.binning_parameters,
                                                       particle_diameters);
      probabilities.push_back(p);
    }

    // Turn data into a distribution
    std::vector< Histograms::Histogram<RealType> > measurements;
    for (unsigned int i=0; i<solutions.size(); ++i)
    {
      auto m = Internal::TEMData::bin_TEM_data(user_data.data[i], user_data.binning_parameters);
      measurements.push_back(m);
    }

    // Calculate the log likelihood
    RealType likelihood = 0;

    for (unsigned int i=0; i<probabilities.size(); ++i)
    {
      likelihood += Internal::TEMData::compute_likelihood_from_binned_data(measurements[i], probabilities[i]);
      // if one of the likelihood computations returns negative "infinity" then simply return that value
      if (likelihood == -std::numeric_limits<RealType>::max())
        return likelihood;
    }

    return likelihood;
  }
}









#endif //MEPBM_SUNDIALS_STATISTICS_H
