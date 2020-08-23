#ifndef STATISTICS_H_
#define STATISTICS_H_


#include <valarray>
#include "models.h"
#include "histogram.h"



namespace Statistics
{
  // log likelihood
  // This function solves the ODE based on odeModel, odeParameters, and solverParameters
  // Then the ODE solution and data are converted to histograms based on histogramParameters.
  // Finally the log likelihood is computed using the multinomial distribution and the two histograms.
  double logLikelihood(const std::valarray<double>& particleSizeData,
                       const Models::ModelsBase& odeModel,
                       const Models::ParametersBase& odeParameters,
                       const Models::explEulerParameters& solverParameters,
                       const Histograms::Parameters& histogramParameters);
}



#endif /* STATISTICS_H_ */
