#include <iostream>
#include "src/histogram.h"
#include "src/statistics.h"



using Real = double;



int main()
{
  // create dummy data
  std::vector<Real> data = {3,4,5,6,4,5};

  // create dummy distribution
  std::vector<Real> sizes = {3, 4, 5, 6};
  std::vector<Real> distr = {1, 2, 3, 4};

  // create histogram parameters
  Histograms::Parameters<Real> hist_prm(4, 3, 6);

  // calculate log likelihood
  Real likelihood = Statistics::log_likelihood<Real>(data, distr, sizes, hist_prm);

  // print result
  std::cout << "log likelihood: " << likelihood;
}
