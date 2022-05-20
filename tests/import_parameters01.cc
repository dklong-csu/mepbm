#include "src/import_parameters.h"
#include <vector>
#include <iostream>


int main () {
  auto prm = MEPBM::import_parameters< std::vector<double> >("import_parameters01.input", 3);
  for (auto & p : prm)
    std::cout << p << std::endl;
}