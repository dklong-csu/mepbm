#include "sundials_statistics.h"
#include <vector>
#include <iostream>


int main ()
{
  const std::vector<double> all_species = {0,1,2,3,4};
  const unsigned int index0 = 1;
  const unsigned int index1 = 3;

  const auto particles = SUNDIALS_Statistics::Internal::strip_nanoparticles_from_vector(all_species, index0, index1);
  for (const auto & val : particles)
  {
    std::cout << val << std::endl;
  }
}