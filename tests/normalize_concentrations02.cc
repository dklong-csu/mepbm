#include "src/normalize_concentrations.h"
#include <iostream>
#include <iomanip>

using Vector = Eigen::Matrix<double, Eigen::Dynamic,1>;

int main ()
{
  Vector conc(4);
  conc << 1, 2, 0, -1;
  auto conc_norm = MEPBM::normalize_concentrations(conc);
  // The 0 and -1 should be replaced with 2e-9 and so the result should be
  // conc_norm = 1/(3+4e-9) * [1, 2, 2e-9, 2e-9]
  std::cout << std::setprecision(40) << conc_norm << std::endl;
}