#include "src/get_subset.h"
#include "src/create_nvector.h"
#include <iostream>

using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


int main ()
{
  auto vector = MEPBM::create_eigen_nvector<Vector>(5);
  auto vec_ptr = static_cast<Vector*>(vector->content);
  (*vec_ptr) << 0,1,2,3,4;

  auto subset = MEPBM::get_subset<double>(vector, 1, 3);

  // subset = [1,2,3] if done correctly
  std::cout << subset << std::endl;
}