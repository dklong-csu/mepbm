#include "IrHPO4.h"
#include <iostream>

int main () {
  // Sample to use for all the tests
  std::vector<double> sample = {1, 2, 3};

  // First test for the surface atoms function just returning 1
  auto one_fcn = [](const unsigned int ) {return 1.0;};

  // Test points for each function
  std::vector<unsigned int> test_pts = {1, 2, 3};


  // Test for 1 step
  MEPBM::IrHPO4::StepAgglomerationKernel<double, std::vector<double>> kern1(one_fcn,
                                                                            {0,1},
                                                                            {2});
  auto kern1_fcn = kern1.get_function(sample);
  for (const auto s1 : test_pts)
    for (const auto s2 : test_pts)
      std::cout << kern1_fcn(s1, s2) << std::endl;


  // Test for 2 steps
  MEPBM::IrHPO4::StepAgglomerationKernel<double, std::vector<double>> kern2(one_fcn,
                                                                            {0,1,2},
                                                                            {2,3});
  auto kern2_fcn = kern2.get_function(sample);
  for (const auto s1 : test_pts)
    for (const auto s2 : test_pts)
      std::cout << kern2_fcn(s1, s2) << std::endl;

  // Test with non-identity surface atoms function
  auto interesting_fcn = [](const unsigned int s) {return 1.0 * s;};
  MEPBM::IrHPO4::StepAgglomerationKernel<double, std::vector<double>> kern3(interesting_fcn,
                                                                            {0,1},
                                                                            {2});
  auto kern3_fcn = kern3.get_function(sample);
  for (const auto s1 : test_pts)
    for (const auto s2 : test_pts)
      std::cout << kern3_fcn(s1, s2) << std::endl;
}