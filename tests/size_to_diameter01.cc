#include "src/size_to_diameter.h"
#include <iostream>
#include <iomanip>

int main ()
{
  const auto diam = MEPBM::iridium_size_to_diameter<double>(5);
  std::cout << std::setprecision(40) << diam << std::endl;
}