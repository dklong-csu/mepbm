#include "src/growth_kernel.h"
#include <iostream>
#include <iomanip>

int main () {

  // First test for the surface atoms function just returning 1
  auto one_fcn = [](const unsigned int ) {return 1.0;};

  // Sample to use in tests
  std::vector<double> sample = {3, .5, 2, .25, 1};

  // Test points
  std::vector<unsigned int> pts = {1, 2, 3, 4};



  // Test for one logistic curve
  MEPBM::LogisticCurveGrowthKernel<double, std::vector<double>> kern1(one_fcn,
                                                                              {0, 2},
                                                                              {2},
                                                                              {1});
  auto kern1_fcn = kern1.get_function(sample);
  for (auto p : pts)
    std::cout << std::setprecision(20) << kern1_fcn(p) << std::endl;



  // Test for two logistic curves
  MEPBM::LogisticCurveGrowthKernel<double, std::vector<double>> kern2(one_fcn,
                                                                              {0, 2, 4},
                                                                              {2, 3},
                                                                              {1, 3});
  auto kern2_fcn = kern2.get_function(sample);
  for (auto p : pts)
    std::cout << std::setprecision(20) << kern2_fcn(p) << std::endl;



  // Test for one logistic curve + non-trivial surface atoms function
  auto interesting_fcn = [](const unsigned int s) {return 1.0 * s;};
  MEPBM::LogisticCurveGrowthKernel<double, std::vector<double>> kern3(interesting_fcn,
                                                                              {0, 2},
                                                                              {2},
                                                                              {1});
  auto kern3_fcn = kern3.get_function(sample);
  for (auto p : pts)
    std::cout << std::setprecision(20) << kern3_fcn(p) << std::endl;
}