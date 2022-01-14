#include "src/histogram.h"
#include "sundials_statistics.h"
#include <vector>
#include <iostream>
#include <iomanip>


int main ()
{
  // Convert concentration to pmf with all positive values
  const std::vector<double> conc = {2e-6, 3e-6, 5e-6};
  const Histograms::Parameters<double> prm(3,0,3);
  const std::vector<double> sizes = {.5, 1.5, 2.5};

  auto pmf_positive = SUNDIALS_Statistics::Internal::convert_concentrations_to_pmf(conc, prm, sizes);
  for (const auto & val : pmf_positive.count)
  {
    std::cout << std::setprecision(20) << val << std::endl;
  }

  // Concentrations with negative value
  const std::vector<double> conc_neg = {-1, 4e-6, 6e-6};
  auto pmf_negative = SUNDIALS_Statistics::Internal::convert_concentrations_to_pmf(conc_neg, prm, sizes);
  for (const auto & val : pmf_negative.count)
  {
    std::cout << std::setprecision(20) << val << std::endl;
  }
}