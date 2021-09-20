#include "sundials_statistics.h"
#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>


using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;

int main ()
{
  auto x = create_eigen_nvector<Vector>(5);
  auto x_vector = static_cast<Vector*>(x->content);
  *x_vector << 0, 1, 2, 3, 4;

  const auto y = SUNDIALS_Statistics::Internal::convert_solution_to_vector<double>(x);
  for (const auto & val : y)
  {
    std::cout << val << std::endl;
  }
}