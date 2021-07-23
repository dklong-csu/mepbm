#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector v = create_eigen_nvector<Vector>(2);


  v->ops->nvdestroy(v);
  // FIXME For some reason v->ops and v->content are not set to nullptr. I don't understand why.
  std::cout << (v->ops == nullptr) << std::endl;
  std::cout << (v->content == nullptr) << std::endl;
  // FIXME I'm not sure if this is a complete test. How do I know ops and content were deleted from the heap. I could just set their values to nullptr and not delete.
}