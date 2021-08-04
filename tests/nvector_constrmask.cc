#include "nvector_eigen.h"
#include "eigen3/Eigen/Dense"
#include <iostream>

using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;

int main ()
{
  N_Vector c = create_eigen_nvector<Vector>(4);
  auto c_vec = static_cast<Vector*>(c->content);
  *c_vec << 2, 1, -1, -2;


  N_Vector x = create_eigen_nvector<Vector>(4);
  auto x_vec = static_cast<Vector*>(x->content);

  
  N_Vector m = create_eigen_nvector<Vector>(4);
  auto m_vec = static_cast<Vector*>(m->content);

  // we expect all to pass
  *x_vec << 1, 0, 0, -1;
  booleantype result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << *m_vec << std::endl;

  // we expect c=2 to fail
  *x_vec << 0, 0, 0, -1;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << *m_vec << std::endl;

  // we expect c=1 to fail
  *x_vec << 1, -1, 0, -1;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << *m_vec << std::endl;

  // we expect c=-1 to fail
  *x_vec << 1, 0, 1, -1;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << *m_vec << std::endl;

  // we expect c=-2 to fail
  *x_vec << 1, 0, 0, 0;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << *m_vec << std::endl;

  // we expect all to fail
  *x_vec << 0, -1, 1, 0;
  result = c->ops->nvconstrmask(c,x,m);
  std::cout << result << std::endl;
  std::cout << *m_vec << std::endl;


}