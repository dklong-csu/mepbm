#include "src/atoms_to_diameter.h"
#include <iostream>
#include <iomanip>

int main()
{
  for (unsigned int s=3; s<=2500; ++s)
    std::cout << std::setprecision(40) << MEPBM::atoms_to_diameter<double>(s) << std::endl;
}