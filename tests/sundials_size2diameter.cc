#include "sundials_statistics.h"
#include <iostream>
#include <iomanip>
#include <vector>



int main ()
{
  std::vector<int> sizes(2500);
  int s = 1;
  for (auto & val : sizes)
  {
    val = s;
    s += 1;
  }

  for (const auto & val : sizes)
  {
    const auto diam = SUNDIALS_Statistics::Internal::convert_particle_size_to_diameter<double>(val);
    std::cout << std::setprecision(20) << diam << std::endl;
  }
}