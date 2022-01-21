#include "src/normalize_concentrations.h"
#include <iostream>

using Vector = Eigen::Matrix<double, Eigen::Dynamic,1>;

int main ()
{
  Vector conc(3);
  conc << 2, 4, 10;
  auto conc_norm = MEPBM::normalize_concentrations(conc);
  // conc_norm = [2/16, 4/16, 10/16] = [0.125, 0.25, 0.625] if done correctly
  std::cout << conc_norm << std::endl;
}