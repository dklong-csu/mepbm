#include "IrHPO4.h"
#include <iostream>

int main () {
  // Sample to use for all of the tests
  std::vector<double> sample = {1, 2, 3};

  // First test for the surface atoms function just returning 1
  auto one_fcn = [](const unsigned int ) {return 1.0;};

  // Test points for each function
  std::vector<unsigned int> test_pts = {1, 2, 3, 4};

  // Test for 1 step
  MEPBM::IrHPO4::StepGrowthKernel<double, std::vector<double>> kern1(one_fcn, {0,1}, {2});
  auto kern1_fcn = kern1.get_function(sample);
  for (const auto s : test_pts)
    std::cout << kern1_fcn(s) << std::endl;


  // Test for 2 steps
  MEPBM::IrHPO4::StepGrowthKernel<double, std::vector<double>> kern2(one_fcn, {0,1,2}, {2,3});
  auto kern2_fcn = kern2.get_function(sample);
  for (const auto s : test_pts)
    std::cout << kern2_fcn(s) << std::endl;


  // Test with non-identity surface atoms function
  auto interesting_fcn = [](const unsigned int s) {return 1.0 * s;};
  MEPBM::IrHPO4::StepGrowthKernel<double, std::vector<double>> kern3(interesting_fcn, {0,1},{2});
  auto kern3_fcn = kern3.get_function(sample);
  for (const auto s : test_pts)
    std::cout << kern3_fcn(s) << std::endl;
}