#include "src/particle.h"
#include <iostream>

// This tests the species function of the Particle class

int main ()
{
  MEPBM::Particle my_particle(1,5,3);

  for (unsigned int i=my_particle.index_start; i<=my_particle.index_end; ++i)
  {
    auto my_species = my_particle.species(i);
    std::cout << std::boolalpha << (my_species.index == i) << std::endl;
  }
}