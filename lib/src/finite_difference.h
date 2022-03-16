#ifndef MEPBM_FINITE_DIFFERENCE_H
#define MEPBM_FINITE_DIFFERENCE_H

#include <functional>

namespace MEPBM
{
  template<typename Real, typename Vector>
  Vector
  finite_difference_one_sided(const std::function<Real(const Vector &)> f, const Vector & x, const Vector & h)
  {
    Vector grad(x.size());
    const Real f_x = f(x);

    // Perform in parallel if OpenMP is being used
#pragma omp parallel for
    for (unsigned int i=0; i<grad.size(); ++i)
    {
      const Real dx = h(i);
      Vector y = x;
      y(i) += dx;
      Real f_xph = f(y);
      grad(i) = (f_xph - f_x) / dx;
    }

    return grad;
  }


  template<typename Real, typename Vector>
  Vector
  finite_difference_central(const std::function<Real(const Vector &)> f, const Vector & x, const Vector & h)
  {
    Vector grad(x.size());
    Vector f_xh(2*x.size());

    // Perform in parallel if OpenMP is being used
#pragma omp parallel for
    for (unsigned int i=0; i<f_xh.size(); ++i)
    {
      // If even i, use f(x+h/2); odd i, compute f(x-h/2)
      unsigned int h_index = i/2; // Will round down so that consecutive even-odd i get same h_index
      if (i % 2 == 0)
      {
        Vector y = x;
        y(h_index) += h(h_index)/2;
        f_xh(i) = f(y);
      }
      else
      {
        Vector y = x;
        y(h_index) -= h(h_index)/2;
        f_xh(i) = f(y);
      }
    }

    // All function evaluations finished, so now apply differencing
    for (unsigned int i=0; i<grad.size(); ++i)
    {
      grad(i) = (f_xh(2*i) - f_xh(2*i+1)) / h(i);
    }

    return grad;
  }
}


#endif //MEPBM_FINITE_DIFFERENCE_H
