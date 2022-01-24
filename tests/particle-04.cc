#include "src/particle.h"
#include <iostream>

// This tests the n_particles function of the Particle class.

int main ()
{
  MEPBM::Particle my_particle(1,5,3);

  // Should be 5 particles
  std::cout << my_particle.n_particles() << std::endl;
}