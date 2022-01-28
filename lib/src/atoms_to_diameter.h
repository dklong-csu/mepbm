#ifndef MEPBM_ATOMS_TO_DIAMETER_H
#define MEPBM_ATOMS_TO_DIAMETER_H


#include <cmath>


namespace MEPBM {
  template<typename Real>
  Real
  atoms_to_diameter(const unsigned int num_atoms)
  {
    return 0.3000805 * std::pow(1. * num_atoms, 1./3.);
  }
}

#endif //MEPBM_ATOMS_TO_DIAMETER_H
