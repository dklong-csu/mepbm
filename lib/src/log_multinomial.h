#ifndef MEPBM_LOG_MULTINOMIAL_H
#define MEPBM_LOG_MULTINOMIAL_H


#include <cmath>
#include "src/histogram.h"
#include <cassert>
#include "src/normalize_concentrations.h"
#include "src/to_vector.h"


namespace MEPBM {
  /**
   * A function to calculate the logarithm of the (un-normalized) probability of a set of counts
   * based on a multinomial distribution with the given probabilities.
   */
  template<typename Real>
  Real
  log_multinomial(const MEPBM::Histogram<Real> &probability, const MEPBM::Histogram<Real> &counts)
  {
    // Get the probability mass function
    const auto pmf = probability.count;
    // Get the number of occurrences for each bin
    const auto occurrences = counts.count;
    // Make sure they are the same size
    assert(pmf.size() == occurrences.size());

    // Calculate the (un-normalized) probability of getting "occurrences" based on "pmf"
    Real log_probability = 0.0;
    for (unsigned int i=0; i<pmf.size(); ++i)
    {
      log_probability += occurrences[i] * std::log(pmf[i]);
    }
    return log_probability;
  }



  /**
   * A function to calculate the logarithm of the (un-normalized) probability of a set of counts
   * based on a multinomial distribution with probabilites taken from the solution vector
   */
   template<typename Real, typename Vector>
   Real
   log_multinomial(const Vector & particles,
                   const std::vector<Real> & diams,
                   const std::vector<Real> & data_diameters,
                   const MEPBM::Parameters<Real> & hist_prm) {
     // Format solution
     auto particles_normed = MEPBM::normalize_concentrations(particles);
     auto particle_prob = MEPBM::to_vector(particles_normed);
     const auto pmf = MEPBM::create_histogram(particle_prob, diams, hist_prm);



     // Format data
     std::vector<Real> ones(data_diameters.size(), 1.0);
     const auto data_binned = MEPBM::create_histogram(ones, data_diameters, hist_prm);



     return log_multinomial<Real>(pmf, data_binned);
   }
}


#endif //MEPBM_LOG_MULTINOMIAL_H
