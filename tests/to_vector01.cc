#include "src/to_vector.h"
#include <iostream>

int main()
{
  Eigen::Matrix<double, Eigen::Dynamic, 1> e_vec(4);
  e_vec << 0, 1, 2, 3;
  auto vec = MEPBM::to_vector(e_vec);
  for (const auto v : vec)
    std::cout << v << std::endl;
}