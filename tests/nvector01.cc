#include "src/create_nvector.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector v = MEPBM::create_eigen_nvector<Vector>(2);
  auto v_vec = static_cast<Vector*>(v->content);
  *v_vec << -1, 2;

  N_Vector x = v->ops->nvclone(v);
  v->ops->nvabs(v, x);

  auto vec = *(static_cast<Vector *>(x->content));
  std::cout << vec << std::endl;
  v->ops->nvdestroy(v);
  x->ops->nvdestroy(x);
}
