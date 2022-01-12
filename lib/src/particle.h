#ifndef MEPBM_PARTICLE_H
#define MEPBM_PARTICLE_H

#include "species.h"
#include <cassert>

namespace MEPBM {
/**
 * A Particle describes the location in a vector of a group of nanoparticles that react in a similar manner.
 * Their indices in the vector are assumed to be contiguous and the first size of the particle is used to determine
 * the size of all particles contained.
 */
 class Particle {
 public:
   /// Constructor.
   Particle(const unsigned int index_start, const unsigned int index_end, const unsigned int first_size)
    : index_start(index_start),
    index_end(index_end),
    first_size(first_size)
   {
     assert(index_end >= index_start);
   }

   unsigned int index_start; /// The vector index for the smallest particle in this group.
   unsigned int index_end; /// The vector index for the largest particle in this group.
   unsigned int first_size; /// The particle size of the smallest particle in this group.

   /// Extracts the Species associated with the particular particle associated with the given index.
   Species species(const unsigned int index) const {
     return Species(index);
   }

   /// Returns the size of the particle at the indicated index.
   int size(const unsigned int index) const {
     int particle_size = (index - index_start) + first_size;
     assert(particle_size > 0);
     return particle_size;
   }

   /// Returns the index of a particle given its size
   int index(const unsigned int size) const {
     assert(size > 0);
     int particle_index = (size - first_size) + index_start;
     assert(particle_index >= 0);
     return particle_index;
   }


   /// Returns the number of particles in this group.
   int n_particles() const {
     return (index_end - index_start) + 1;
   }
 };
}

#endif //MEPBM_PARTICLE_H
