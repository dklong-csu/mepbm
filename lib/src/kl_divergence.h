#ifndef MEPBM_KL_DIVERGENCE_H
#define MEPBM_KL_DIVERGENCE_H



#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <vector>
#include "src/histogram.h"
#include "src/normalize_concentrations.h"
#include "src/to_vector.h"



namespace MEPBM {
  // FIXME: documentation
  template<typename Real, typename Vector>
  Real
  kl_divergence(const Vector & distribution, const Vector & ref_distribution) {
    // Make sure distributions have the same number of categories
    assert(distribution.size() == ref_distribution.size());

    // Check normalization
    const Real distr_sum = std::accumulate(distribution.begin(), distribution.end(), 0.0);
    if (std::abs(distr_sum - 1.0) >= 1e-8) {
      throw std::invalid_argument("The sum of distribution elements is not equal to 1. Check the input parameter `distribution`.");
    }

    const Real ref_sum = std::accumulate(ref_distribution.begin(), ref_distribution.end(), 0.0);
    if (std::abs(ref_sum - 1.0) >= 1e-8) {
      throw std::invalid_argument("The sum of distribution elements is not equal to 1. Check the input parameter `ref_distribution`.");
    }

    // Compute the Kullback-Leibler divergence D(distribution || ref_distribution)
    Real kl_div = 0.0;
    for (unsigned int i=0; i<distribution.size(); ++i) {
      // If distribution[i] == 0, the contribution is interpreted to be zero.
      if (distribution[i] > 0 && ref_distribution[i] > 0) {
        kl_div += distribution[i] * std::log(distribution[i] / ref_distribution[i]);
      }
      else if (distribution[i] > 0 && ref_distribution[i] <= 0) {
        // If the reference distribution has a category with 0 probability but the comparison distribution does not
        // then the KL divergence is infinity
        return std::numeric_limits<Real>::max();
      }
    }
    return kl_div;
  }



  // FIXME: documentation
  template<typename Real, typename Vector>
  Real
  kl_divergence(const Vector & particles,
                const std::vector<Real> & diams,
                const std::vector<Real> & data_diameters,
                const MEPBM::Parameters<Real> & hist_prm) {
    // Format solution
    auto particle_vec = MEPBM::to_vector(particles);
    const auto sim_pmf = MEPBM::create_histogram(particle_vec, diams, hist_prm);
    auto Q = sim_pmf.count;
    const auto Q_sum = std::accumulate(Q.begin(), Q.end(), 0.0);
    for (auto & q : Q){
      q /= Q_sum;
    }



    // Format data
    std::vector<Real> weights(data_diameters.size(), 1.0 / data_diameters.size());
    const auto data_pmf = MEPBM::create_histogram(weights, data_diameters, hist_prm);
    const auto P = data_pmf.count;


    return kl_divergence< Real, std::vector<Real> >(P, Q);
  }



  // FIXME: documentation
  template<typename Real, typename Vector>
  Real
  js_divergence(const Vector & particles,
                const std::vector<Real> & diams,
                const std::vector<Real> & data_diameters,
                const MEPBM::Parameters<Real> & hist_prm) {
    // Format solution
    auto particle_vec = MEPBM::to_vector(particles);
    const auto sim_pmf = MEPBM::create_histogram(particle_vec, diams, hist_prm);
    auto Q = sim_pmf.count;
    const auto Q_sum = std::accumulate(Q.begin(), Q.end(), 0.0);
    for (auto & q : Q){
      q /= Q_sum;
    }



    // Format data
    std::vector<Real> weights(data_diameters.size(), 1.0 / data_diameters.size());
    const auto data_pmf = MEPBM::create_histogram(weights, data_diameters, hist_prm);
    const auto P = data_pmf.count;



    // Average distribution
    std::vector<Real> M(P.size());
    for (unsigned int i=0; i<P.size(); ++i){
      M[i] = 0.5*(Q[i]+P[i]);
    }


    return 0.5*( kl_divergence< Real, std::vector<Real> >(P, M) + kl_divergence< Real, std::vector<Real> >(Q, M) );
  }
}


#endif //MEPBM_KL_DIVERGENCE_H