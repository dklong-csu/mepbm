#ifndef MEPBM_SPECIES_H
#define MEPBM_SPECIES_H

namespace MEPBM {
  /**
   * A Species is a container that associates an index in a vector with a chemical species to facilitate more readable code.
   */
   class Species {
   public:
     Species(const unsigned int index)
     : index(index) {}

     unsigned int index;
   };
}

#endif //MEPBM_SPECIES_H
