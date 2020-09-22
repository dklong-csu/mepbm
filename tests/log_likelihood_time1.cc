#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"



int main()
{
  // create data
  const Data::PomData all_data;
  const std::vector<std::valarray<double>> data = {std::pow(all_data.tem_diam_time1/0.3000805, 3)};
  const std::vector<double> times = {all_data.tem_time1};

  // create ODE model
  const Models::ThreeStepAlternative::Parameters ode_prm(3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);
  const Models::ThreeStepAlternative model;

  // set up initial condition
  std::valarray<double> ic(0.0, ode_prm.n_variables);
  ic[0] = 0.0012;

  // set up histogram parameters
  const Histograms::Parameters hist_prm(25, 3.0, 2500.0);

  // calculate log likelihood
  const double likelihood = Statistics::log_likelihood(data, times, model, ode_prm, ic, hist_prm);

  // print result
  std::cout << "log likelihood: " << likelihood;
}
