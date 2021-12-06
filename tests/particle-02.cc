#include "chemical_reaction.h"
#include <iostream>

/*
 * This tests the get_particle_species function of the Particle class
 */

int main ()
{
  Model::Particle my_particle(1,5,3);

  for (unsigned int i=my_particle.index_start; i<=my_particle.index_end; ++i)
  {
    auto my_species = my_particle.get_particle_species(i);
    std::cout << std::boolalpha << (my_species.index == i) << std::endl;
  }
}