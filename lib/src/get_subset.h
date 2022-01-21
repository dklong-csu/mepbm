#ifndef MEPBM_GET_SUBSET_H
#define MEPBM_GET_SUBSET_H

#include <eigen3/Eigen/Dense>
#include <sundials/sundials_nvector.h>

namespace MEPBM {
  /**
   * A function that takes an `N_Vector` and returns a specified subset in the form of an Eigen vector.
   */
  template <typename Real>
  Eigen::Matrix<Real, Eigen::Dynamic, 1>
  get_subset(const N_Vector solution, const unsigned int first_index, const unsigned int last_index)
  {
    assert(last_index >= first_index);
    assert(last_index < solution->ops->nvgetlength(solution));
    const unsigned int n = last_index - first_index + 1;
    const auto sol_vec = *static_cast<Eigen::Matrix<Real, Eigen::Dynamic, 1>*>(solution->content);
    Eigen::Matrix<Real, Eigen::Dynamic, 1> particles = sol_vec.segment(first_index,n);
    return particles;
  }
}

#endif //MEPBM_GET_SUBSET_H
