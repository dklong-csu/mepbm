#ifndef MEPBM_PERTURB_SAMPLE_H
#define MEPBM_PERTURB_SAMPLE_H

#include "sample.h"
#include <utility>
#include <random>
#include <eigen3/Eigen/Dense>
#include <vector>


namespace Sampling
{

/**
 * A function to perturb a sample based on a uniform distribution. This function returns
 * both the new sample and the proposal ratio.
 */
 template<typename RealType>
 std::pair<Sample<RealType>, RealType>
 perturb_uniform(const Sample<RealType> &sample,
                 std::mt19937 &rng,
                 const std::vector<RealType> &perturb_magnitude_real,
                 const std::vector<int> &perturb_magnitude_int)
 {
   // perturb the real-valued parameters
   std::vector<RealType> perturbed_real_parameters(sample.real_valued_parameters.size());
   for (unsigned int i=0; i < sample.real_valued_parameters.size(); ++i)
   {
     RealType perturb
      = std::uniform_real_distribution<RealType>(-perturb_magnitude_real[i],perturb_magnitude_real[i])(rng);

     perturbed_real_parameters[i] = sample.real_valued_parameters[i] + perturb;
   }

   // perturb the integer-valued parameters
   std::vector<int> perturbed_int_parameters(sample.integer_valued_parameters.size());
   for (unsigned int i=0; i < sample.integer_valued_parameters.size(); ++i)
   {
     int perturb
      = std::uniform_int_distribution<int>(-perturb_magnitude_int[i],perturb_magnitude_int[i])(rng);

     perturbed_int_parameters[i] = sample.integer_valued_parameters[i] + perturb;
   }

   Sample<RealType> new_sample(perturbed_real_parameters,
                               perturbed_int_parameters);

   // since the uniform distribution is symmetric, the proposal ratio is always 1.
   std::cout << "New sample: ";
   for (auto val : new_sample.real_valued_parameters)
   {
     std::cout << val << "     ";
   }
   for (auto val : new_sample.integer_valued_parameters)
   {
     std::cout << val << "     ";
   }
   std::cout << std::endl;
   std::pair< Sample<RealType>, RealType > sample_and_ratio(new_sample, 1.);
   return sample_and_ratio;
 }



 /**
  * A function to perturb a sample based on a normal distribution. This function returns
  * both the new sample and the proposal ratio.
  */
  template<typename RealType>
  std::pair<Sample<RealType>, RealType>
  perturb_normal(const Sample<RealType> &sample,
                 std::mt19937 &rng,
                 const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &covariance,
                 const RealType factor)
 {
    // Create a vector following a normal distribution with mean 0 and variance 1
    const auto dim = sample.real_valued_parameters.size() + sample.integer_valued_parameters.size();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> random_vector(dim);
    for (unsigned int i=0; i < dim; ++i)
    {
      random_vector(i) = std::normal_distribution<RealType>(0,1)(rng);
    }

    /*
     * Use the covariance matrix and multiplicative factor to transform the random vector
     * to follow the desired distribution. Performing a Cholesky decomposition, C = LL^T,
     * we can modify the distribution to follow a normal distribution centered around the
     * current sample with covariance = factor^2*C by performing the transformation
     *    y = factor * L * x + current parameters
     */
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> L = covariance.llt().matrixL();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> perturbed_vector(dim);

    for (unsigned int i=0; i < sample.real_valued_parameters.size(); ++i)
    {
      perturbed_vector(i) = sample.real_valued_parameters[i];
    }

    for (unsigned int i=sample.real_valued_parameters.size(); i < dim; ++i)
    {
      perturbed_vector(i) = sample.integer_valued_parameters[i];
    }

    perturbed_vector += factor * L * random_vector;

    // Organize perturbed parameters and create a new sample
    std::vector<RealType> perturbed_real_parameters(sample.real_valued_parameters.size());
    for (unsigned int i=0; i < sample.real_valued_parameters.size(); ++i)
    {
      perturbed_real_parameters[i] = perturbed_vector(i);
    }

    std::vector<int> perturbed_int_parameters(sample.integer_valued_parameters.size());
    for (unsigned int i=0; i < sample.integer_valued_parameters.size(); ++i)
    {
      perturbed_int_parameters[i] = perturbed_vector(i + sample.real_valued_parameters.size());
    }

   Sample<RealType> new_sample(perturbed_real_parameters,
                               perturbed_int_parameters);

   // since the normal distribution is symmetric, the proposal ratio is always 1.
   return {new_sample, 1.};
 }








}










#endif //MEPBM_PERTURB_SAMPLE_H
