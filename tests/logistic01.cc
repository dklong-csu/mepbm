#include "src/logistic_curve.h"
#include <iostream>
#include <iomanip>
#include <vector>



int main () {
  MEPBM::BackwardsLogisticCurve<double> curve(5,3,2);
  std::vector<double> pts = {0, 1, 2, 3, 4, 5, 6};
  for (auto p : pts)
    std::cout << std::setprecision(20) << curve.evaluate(p) << std::endl;
}