#include <iostream>
#include <string>
#include <boost/numeric/odeint.hpp>
#include "models.h"
#include "histogram.h"
#include "statistics.h"



int main()
{
  // create dummy data
  std::vector<double> data = {3,4,5,6,4,5};

  // create dummy distribution
  std::vector<double> sizes = {3, 4, 5, 6};
  std::vector<double> distr = {1, 2, 3, 4};

  // create histogram parameters
  Histograms::Parameters hist_prm(4, 3, 6);

  // calculate log likelihood
  double likelihood = Statistics::log_likelihood(data, distr, sizes, hist_prm);

  // print result
  std::cout << "log likelihood: " << likelihood;
}
