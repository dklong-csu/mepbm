#include <src/kl_divergence.h>
#include <vector>
#include <iostream>
#include <iomanip>


int main () {
  const std::vector<double> P = {.2, .5, .3};
  const std::vector<double> Q = {.15, .45, .4};

  double kl = MEPBM::kl_divergence< double, std::vector<double> >(P, Q);

  std::cout << std::setprecision(20) << kl << std::endl;
}