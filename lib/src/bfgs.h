#ifndef MEPBM_BFGS_H
#define MEPBM_BFGS_H

#include <functional>
#include <eigen3/Eigen/Dense>
#include "src/line_search_wolfe.h"
#include <cmath>


namespace MEPBM
{
  /**
   * A class implementing the Quasi-Newton Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm
   * for unconstrained optimization.
   */
  template<typename Real>
  class BFGS
  {
  public:
    BFGS(std::function<Real(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &)> f,
         std::function<Eigen::Matrix<Real, Eigen::Dynamic, 1>(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &)> grad_f)
       : f(f), grad_f(grad_f)
    {}

    Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> H_inv;
    std::function<Real(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &)> f;
    std::function<Eigen::Matrix<Real, Eigen::Dynamic, 1>(const Eigen::Matrix<Real, Eigen::Dynamic, 1> &)> grad_f;


    Eigen::Matrix<Real, Eigen::Dynamic, 1> minimize(const Eigen::Matrix<Real, Eigen::Dynamic, 1> & x, const Real tol, const unsigned int max_iter)
    {
      /*
       * First iteration
       */
      Eigen::Matrix<Real, Eigen::Dynamic, 1> x0 = x;
      // Initialize H_inv as identity, making the first step gradient descent
      const int dim = x0.size();
      H_inv = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(dim, dim);

      // Compute gradient
      Eigen::Matrix<Real, Eigen::Dynamic, 1> g0 = grad_f(x0);

      // Compute search direction
      Eigen::Matrix<Real, Eigen::Dynamic, 1> d = - H_inv * g0;

      // Line search to find step size
      Real a = line_search_wolfe(f, grad_f, x0, d);

      // Compute x_{k+1}
      Eigen::Matrix<Real, Eigen::Dynamic, 1> s = a * d;
      Eigen::Matrix<Real, Eigen::Dynamic, 1> x1 = x0 + s;

      // Compute g_{k+1}
      Eigen::Matrix<Real, Eigen::Dynamic, 1> g1 = grad_f(x1);

      // Compute H_{k+1}^{-1}
      Eigen::Matrix<Real, Eigen::Dynamic, 1> y = g1 - g0;

        // Heuristic to scale initial Hessian estimate -- for the first step only
        H_inv *= y.dot(s) / y.dot(y);

      H_inv = H_inv
          - (1/s.dot(y)) * (s * y.transpose() * H_inv + H_inv * y * s.transpose())
          + (1 + y.dot(H_inv*y)/s.dot(y)) * (s * s.transpose()) / s.dot(y);

      unsigned int n_iter = 1;


      /*
       * Subsequent iterations
       */
      while (g1.norm() > tol && n_iter < max_iter)
      {
        // Updates for iteration count
        x0 = x1;
        g0 = g1;

        // Compute search direction
        d = - H_inv * g0;

        // Line search to find step size
        Real a = line_search_wolfe(f, grad_f, x0, d);

        // Compute x_{k+1}
        s = a * d;
        x1 = x0 + s;

        // Compute g_{k+1}
        g1 = grad_f(x1);

        // Compute H_{k+1}^{-1}
        y = g1 - g0;
        H_inv = H_inv
                - (1/s.dot(y)) * (s * y.transpose() * H_inv + H_inv * y * s.transpose())
                + (1 + y.dot(H_inv*y)/s.dot(y)) * (s * s.transpose()) / s.dot(y);

        ++n_iter;
      }

      // FIXME: Some output to console to describe convergence?

      return x1;
    }
  };
}

#endif //MEPBM_BFGS_H
