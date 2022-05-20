#ifndef MEPBM_LOGISTIC_CURVE_H
#define MEPBM_LOGISTIC_CURVE_H



#include <cmath>



namespace MEPBM {
  /**
   * A class that provides the means to evaluate a logistic curve that goes from high to low (i.e. "backwards").
   * @tparam Real - The floating type number (e.g. `double` or `float`)
   */
  template<typename Real>
  class BackwardsLogisticCurve {
  public:
    /**
     * Constructor which takes in the necessary parameters for a logistic curve.
     * @param height - The height (in the limit) of the logistic curve.
     * @param midpoint - The point where the drop is centered around.
     * @param rate - The coefficient for how quickly the curve drops to zero.
     */
    BackwardsLogisticCurve(const Real height, const Real midpoint, const Real rate)
        : height(height),
          midpoint(midpoint),
          rate(rate)
    {}



    /**
     * Evaluates the backwards logistic curve at a specified point.
     * @param x - The point to evaluate the function at.
     * @return
     */
    Real evaluate(const Real x) const {
      return height - height / (1 + std::exp(-rate * (x - midpoint)));
    };

  private:
    const Real height;
    const Real midpoint;
    const Real rate;
  };
}

#endif //MEPBM_LOGISTIC_CURVE_H
