#include "src/log_multinomial.h"
#include "src/histogram.h"
#include "src/to_vector.h"
#include "src/normalize_concentrations.h"
#include <iostream>
#include <iomanip>


using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;


int main()
{
  // Histogram category
  const MEPBM::Parameters<double> hist_prm(3,0,6);
  // bins = { [0,2), [2, 4), [4,6] }
  const std::vector<double> labels = {0, 1, 2, 3, 4, 5};


  // Create pmf
  Vector conc(6);
  conc << 1, 1, 2, 4, 8, 16;
  auto prob = MEPBM::normalize_concentrations(conc);
  auto prob_vec = MEPBM::to_vector(prob);
  auto pmf = MEPBM::create_histogram(prob_vec, labels, hist_prm);

  // Create counts
  std::vector<double> data = {1,2,3,4,5,6};
  auto counts = MEPBM::create_histogram(data, labels, hist_prm);

  // Calculate
  const auto log_prob = MEPBM::log_multinomial(pmf,counts);
  std::cout << std::setprecision(40) << log_prob << std::endl;

}