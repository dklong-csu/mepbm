#include "src/particle.h"
#include <iostream>
#include <vector>

// This tests the index function of the Particle class.

int main ()
{
  MEPBM::Particle my_particle(1,5,3);

  // Should be able to extrapolate index outside the particle for applications of growth and agglomeration
  // sizes: 3, 4, 5, 6, 7, 8
  // index: 1, 2, 3, 4, 5, 6
  std::vector<int> sizes = {3, 4, 5, 6, 7, 8};
  for (const auto s : sizes)
  {
    std::cout << my_particle.index(s) << std::endl;
  }
}