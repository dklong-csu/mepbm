#include "src/particle.h"
#include <iostream>

// This tests the size function of the Particle class.

int main ()
{
  MEPBM::Particle my_particle(1,5,3);

  for (unsigned int i=my_particle.index_start; i<=my_particle.index_end; ++i)
  {
    auto s = my_particle.size(i);
    // sizes should be: 3, 4, 5, 6, 7
    std::cout << s << std::endl;
  }
}