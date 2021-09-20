#include "histogram.h"
#include "sundials_statistics.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main ()
{
  // Create histogram parameters
  const Histograms::Parameters<double> prm(3,0,3);

  // Create histogram for measurements
  const std::vector<double> x_vals_data = {0.5, 1.5, 2.5};
  const std::vector<double> y_vals_data = {2, 4, 1};
  const auto measurements = Histograms::create_histogram(y_vals_data, x_vals_data, prm);

  // Create histogram for probabilities
  const std::vector<double> x_vals_prob = {0.5, 1.5, 2.5};
  const std::vector<double> y_vals_prob = {.4, .5, .1};
  const auto probabilities = Histograms::create_histogram(y_vals_prob, x_vals_prob, prm);

  // Compute likelihood
  auto likelihood = SUNDIALS_Statistics::Internal::TEMData::compute_likelihood_from_binned_data(measurements, probabilities);

  std::cout << std::setprecision(20) << likelihood << std::endl;
}