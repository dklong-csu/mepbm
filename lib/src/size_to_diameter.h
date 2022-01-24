#ifndef MEPBM_SIZE_TO_DIAMETER_H
#define MEPBM_SIZE_TO_DIAMETER_H

#include <cmath>
#include <cassert>

namespace MEPBM {
  /**
   * A function to convert the size of an iridium particle to a diameter comparable to the results of a TEM measurement.
   */
  template<typename RealType>
  RealType
  iridium_size_to_diameter(const int num_atoms)
  {
    assert(num_atoms > 0);
    return 0.3000805 * std::pow(1.*num_atoms, 1./3);
  }
}

#endif //MEPBM_SIZE_TO_DIAMETER_H
