#ifndef MEPBM_LINE_SEARCH_WOLFE_H
#define MEPBM_LINE_SEARCH_WOLFE_H

#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>

namespace MEPBM
{
  /*
   * An implementation of an inexact line search finding a point that satisfies the strong Wolfe Conditions
   * From p. 60-61 of Numerical Optimization Ed. 2 by Nocedal & Wright
   * and p. 62 of Algorithms for Optimization by Kochenderfer & Wheeler
   */
  template<typename Real>
  Real
  line_search_wolfe(const std::function< Real(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &) > f,
                    const std::function< Eigen::Matrix<Real, Eigen::Dynamic, 1>(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &) > grad_f,
                    const Eigen::Matrix<Real, Eigen::Dynamic, 1> & x,
                    const Eigen::Matrix<Real, Eigen::Dynamic, 1> & d,
                    Real a = 1.0,
                    const Real b = 1e-4,
                    const Real s = 0.9,
                    const unsigned int max_iter = 100)
  {
    Real y0 = f(x);
    Real g0 = grad_f(x).transpose() * d;
    Real y_prev = 0;
    Real a_prev = 0;
    Real alo = 0;
    Real ahi = 0;

    unsigned int n_iter = 0;
    while (n_iter < max_iter)
    {
      Real y = f(x + a*d);
      if (y > y0 + b*a*g0 ||
          ( y >= y_prev && n_iter > 0))
      {
        alo = a_prev;
        ahi = a;
        break; // Enter Zoom
      }

      Real g = grad_f(x + a*d).transpose() * d;
      if (std::abs(g) <= - s*g0)
      {
        return a;
      }

      if (g >= 0)
      {
        alo = a;
        ahi = a_prev;
        break; // Enter Zoom
      }

      y_prev = y;
      a_prev = a;
      a *= 2;
      ++n_iter;
      // Go back through loop
    }

    // Zoom as described in p. 60-61 of Numerical Optimization Ed. 2 by Nocedal & Wright
    Real ylo = f(x + alo*d);
    n_iter = 0;
    while (n_iter < max_iter)
    {
      a = (alo + ahi)/2;
      Real y = f(x + a*d);


      if (y > y0 + b*a*g0 ||
          y >= ylo)
      {
        ahi = a;
      }
      else
      {
        Real g = grad_f(x + a*d).transpose() * d;
        if (std::abs(g) <= -s*g0)
        {
          return a;
        }
        else if (g*(ahi - alo) >= 0)
        {
          ahi = alo;
        }
        alo = a;
      }
      ++n_iter;
    }
    return a;
  }
}

#endif //MEPBM_LINE_SEARCH_WOLFE_H
