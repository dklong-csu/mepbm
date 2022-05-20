#ifndef MEPBM_R_FUNCTION_H
#define MEPBM_R_FUNCTION_H



#include <cmath>
#include <algorithm>



namespace MEPBM {
  /**
     * Each atom in a nanoparticle will be able to bind with other species, so long as it is on the outside
     * of the particle. The reaction thus will be sped up in proportion to how many cluster are on the surface
     * of the particle. Adapted from the work in https://doi.org/10.1007/s11244-005-9261-4.
     *
     * @param size - The particle size
     * @return - The amount the base reaction rate should be multiplied by to account for the particle size.
     */
  template<typename Real>
  Real r_func(const unsigned int size) {
    return (1.0*size) * 2.677 * std::pow(1.0*size, -0.28);
  }



  /**
   * Similar to r_func but caps the regression formula from https://doi.org/10.1007/s11244-005-9261-4 at 1.
   * The original formula from the reference results in values above 1 for small particles sizes. Once
   * multiplied by the size, the interpretation is "how many atoms are on the surface". Giving a number
   * larger than the particle size (i.e. total number of atoms) is unphysical. So capping the number at surface
   * atoms at the total number of atoms has a stronger physical interpretation.
   *
   * @tparam Real - The type representing real-valued numbers (e.g. double, float)
   * @param size  - The particle size
   * @return - The amount the base reaction rate should be multiplied by to account for the particle size.
   */
  template<typename Real>
  Real r_func_capped(const unsigned int size) {
    return std::min(1.0*size, r_func<Real>(size));
  }
}

#endif //MEPBM_R_FUNCTION_H
