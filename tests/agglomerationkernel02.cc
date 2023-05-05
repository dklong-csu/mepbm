#include "src/agglomeration_kernel.h"
#include <iostream>

int main () {
  // Test points for all
  std::vector<double> test_pts = {0,1,2,3,4,5,6,7};

  // First test for the surface atoms function just returning 1
  auto one_fcn = [](const unsigned int ) {return 1.0;};

  
  // Test for 1 point provided (global constant)
  std::cout << "Global constant test\n";
  {
    std::vector<unsigned int> x = {0};
    std::vector<unsigned int> y_idx = {0};
    std::vector<double> sample = {10};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(one_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }

  // Test for 2 points provided (global linear)
  std::cout << "Global linear test\n";
  {
    std::vector<unsigned int> x = {0,2};
    std::vector<unsigned int> y_idx = {0, 1};
    std::vector<double> sample = {10, 1};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(one_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }

  // Test for 3 points provided (piecewise linear)
  std::cout << "Piecewise linear test (3 points)\n";
  {
    std::vector<unsigned int> x = {0,2,4};
    std::vector<unsigned int> y_idx = {0, 1, 2};
    std::vector<double> sample = {10, 1, 9};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(one_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }

  // Test for 4 points provided (pievewise linear)
  std::cout << "Piecewise linear test (4 points)\n";
  {
    std::vector<unsigned int> x = {0,2,4,6};
    std::vector<unsigned int> y_idx = {0, 1, 2, 3};
    std::vector<double> sample = {10, 1, 9, 2};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(one_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }

  // Test with non-identity surface atoms function
  auto interesting_fcn = [](const unsigned int s) {return 1.0 * s;};
  std::cout << "Non-identity test (global constant)\n";
  {
    std::vector<unsigned int> x = {0};
    std::vector<unsigned int> y_idx = {0};
    std::vector<double> sample = {10};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(interesting_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }

  std::cout << "Non-identity test (global linear)\n";
  {
    std::vector<unsigned int> x = {0,2};
    std::vector<unsigned int> y_idx = {0,1};
    std::vector<double> sample = {10,1};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(interesting_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }

  std::cout << "Non-identity test (piecewise linear)\n";
  {
    std::vector<unsigned int> x = {0, 2, 4};
    std::vector<unsigned int> y_idx = {0, 1, 2};
    std::vector<double> sample = {10, 1, 9};
    MEPBM::PiecewiseLinearAgglomerationKernel<double, std::vector<double>> kern(interesting_fcn, y_idx, x);
    auto kern_fcn = kern.get_function(sample);
    for (const auto s1 : test_pts)
      for (const auto s2 : test_pts)
        std::cout << kern_fcn(s1, s2) << std::endl;
  }
}