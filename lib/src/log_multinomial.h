#ifndef MEPBM_LOG_MULTINOMIAL_H
#define MEPBM_LOG_MULTINOMIAL_H


#include <cmath>
#include "src/histogram.h"
#include <cassert>


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
}


#endif //MEPBM_LOG_MULTINOMIAL_H
