#ifndef MEPBM_KL_DIVERGENCE_H
#define MEPBM_KL_DIVERGENCE_H



#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>



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
    auto particles_normed = MEPBM::normalize_concentrations(particles);
    auto particle_prob = MEPBM::to_vector(particles_normed);
    const auto sim_pmf = MEPBM::create_histogram(particle_prob, diams, hist_prm);
    const auto Q = sim_pmf.count;


    // Format data
    std::vector<Real> weights(data_diameters.size(), 1.0 / data_diameters.size());
    const auto data_pmf = MEPBM::create_histogram(weights, data_diameters, hist_prm);
    const auto P = data_pmf.count;


    return kl_divergence< Real, std::vector<Real> >(P, Q);
  }
}


#endif //MEPBM_KL_DIVERGENCE_H