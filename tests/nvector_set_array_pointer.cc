#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  Vector w(2);
  N_Vector v = create_eigen_nvector<Vector>(&w);

  realtype* new_data;
  v->ops->nvsetarraypointer(new_data, v);
  auto v_ptr = v->ops->nvgetarraypointer(v);
  auto result = (v_ptr == new_data);
  std::cout << result;
}