#include "src/histogram.h"
#include "sundials_statistics.h"
#include <vector>
#include <iostream>

int main ()
{
  /**
   * This test attempts to bin a vector in the way TEM data should be binned.
   * We will bin the vector
   *    1, 2, 3, 4, 5, 6, 7
   * into bins holding values
   *    1-3, 3-5, 5-7
   * The resulting histogram should have vectors:
   *    interval_pts = 1, 3, 5, 7
   *    counts = 2, 2, 3
   */
  Histograms::Parameters<double> prm(3,1,7);
  std::vector<double> data = {1,2,3,4,5,6,7};

  auto hist = SUNDIALS_Statistics::Internal::TEMData::bin_TEM_data(data, prm);

  for (auto & val : hist.interval_pts)
    std::cout << val << std::endl;

  for (auto & val : hist.count)
    std::cout << val << std::endl;

}