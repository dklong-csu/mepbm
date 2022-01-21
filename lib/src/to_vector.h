#ifndef MEPBM_TO_VECTOR_H
#define MEPBM_TO_VECTOR_H

#include <vector>
#include <eigen3/Eigen/Dense>


namespace MEPBM {
  /**
   * A function that takes an Eigen vector and converts it to a std::vector.
   */
   template<typename Real>
   std::vector<Real>
   to_vector(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &e_vec)
  {
     std::vector<Real> out_vec(e_vec.size());
     Eigen::Matrix<Real, Eigen::Dynamic, 1>::Map(&out_vec[0], e_vec.size()) = e_vec;
     return out_vec;
  }
}

#endif //MEPBM_TO_VECTOR_H
