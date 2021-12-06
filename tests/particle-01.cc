#include "chemical_reaction.h"
#include <iostream>

/*
 * This tests the member variables of the Particle class
 */


int main ()
{
  Model::Particle my_particle(1,10,3);

  std::cout << std::boolalpha << (my_particle.index_start == 1) << std::endl;
  std::cout << std::boolalpha << (my_particle.index_end == 10) << std::endl;
  std::cout << std::boolalpha << (my_particle.first_size == 3) << std::endl;
}