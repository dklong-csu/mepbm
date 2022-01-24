#ifndef MEPBM_NORMALIZE_CONCENTRATIONS_H
#define MEPBM_NORMALIZE_CONCENTRATIONS_H


#include <eigen3/Eigen/Dense>


namespace MEPBM {
  /**
   * A function that takes a vector and normalizes it such that the sum of the entries is one. The intended use
   * of this function is to act on a vector representing particle concentrations. In this case, negative values
   * are physically inaccurate and are replaced with small numbers. The output vector is intended to be interpreted
   * as a probability mass function for each particle included in the vector.
   */
  template <typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  normalize_concentrations(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &vec)
  {
    // The values in the vector *should* be strictly positive. If any negative values occur, this can be attributed
    // to error from the ODE solver. From a modeling standpoint, it is more accurate to replace the negative values
    // with positive numbers that are small relative to the other values in the vector.

    // Find the maximum concentration
    auto max_conc = vec.maxCoeff();

    // Replace nonpositive values with a small number
    Eigen::Matrix<Real, Eigen::Dynamic, 1> vec_corrected = vec;
    for (unsigned int i=0; i<vec_corrected.size(); ++i)
    {
      if (vec_corrected(i) <= 0)
      {
        vec_corrected(i) = 1e-9 * max_conc;
      }
    }

    // Normalize the vector
    auto vec_sum = vec_corrected.sum();
    Eigen::Matrix<Real, Eigen::Dynamic, 1> vec_norm = vec_corrected / vec_sum;

    return vec_norm;
  }
}


#endif //MEPBM_NORMALIZE_CONCENTRATIONS_H
