#include "IrHPO4.h"
#include <iostream>
#include <iomanip>

int main () {
  for (unsigned int i=1; i<=200; ++i)
    std::cout << std::setprecision(20) << MEPBM::IrHPO4::r_func<double>(i) << std::endl;
}