#ifndef MEPBM_PERTURB_SAMPLE_H
#define MEPBM_PERTURB_SAMPLE_H


#include <utility>
#include <random>
#include <eigen3/Eigen/Dense>
#include <vector>


namespace MEPBM
{
 /**
  * A function to perturb a sample based on a normal distribution. This function returns
  * both the new sample and the proposal ratio.
  */
  template<typename RealType>
  std::pair<Eigen::Matrix<RealType,Eigen::Dynamic, 1>, RealType>
  perturb_normal(const Eigen::Matrix<RealType,Eigen::Dynamic, 1> &sample,
                 std::mt19937 &rng,
                 const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> &covariance)
 {
    // FIXME -- I don't really know how to write a test for this.
    // Create a vector following a normal distribution with mean 0 and variance 1
    const auto dim = sample.size();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> random_vector(dim);
    for (unsigned int i=0; i < dim; ++i)
    {
      random_vector(i) = std::normal_distribution<RealType>(0,1)(rng);
    }

    /*
     * If C = L*L^T then X~N(0,1) can be transformed to Y~(s_0,C) with
     *    y = s_0 + L * x
     * where x,y are realizations of X,Y.
     */
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic> L = covariance.llt().matrixL();
    Eigen::Matrix<RealType, Eigen::Dynamic, 1> new_sample =
        sample + L * random_vector;

   // since the normal distribution is symmetric, the proposal ratio is always 1.
   return {new_sample, 1.};
 }








}










#endif //MEPBM_PERTURB_SAMPLE_H
