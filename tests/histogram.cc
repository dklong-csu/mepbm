#include <valarray>
#include <iostream>
#include <string>
#include "histogram.h"

using HistVector = std::valarray<double>;

int main()
{
  // check we can make proper histogram parameters
  Histograms::Parameters prm(25, 0., 4.5);

  std::cout << "Checking for Histograms::Parameters"
            << std::endl;

  std::cout << "Parameter n_bins is: "
            << prm.n_bins
            << std::endl;

  std::cout << "Parameter x_start is: "
            << prm.x_start
            << std::endl;

  std::cout << "Parameter x_end is: "
            << prm.x_end
            << std::endl;



  // check if we can make a histogram
  Histograms::Histogram hist(prm);

  std::cout << "Checking for Histograms::Histogram"
            << std::endl;

  std::cout << "Interval points are: "
            << std::endl;

  for (unsigned int i = 0; i < hist.interval_pts.size(); i++)
  {
    std::cout << hist.interval_pts[i]
              << ' ';
  }
  std::cout << std::endl;


  std::cout << "Maximum x value is: "
            << hist.max_x
            << std::endl;

  std::cout << "Minimum x value is: "
            << hist.min_x
            << std::endl;

  std::cout << "Number of bins is: "
            << hist.num_bins
            << std::endl;



  // check AddToBin
  std::cout << "Checking for: AddToBin method"
            << std::endl;

  HistVector x = { 0, 4.5, 1.7, -1, 5, 1.7 };
  HistVector y = { 1, 1.1, 3, 20, 30, 2 };
  hist.AddToBins(y, x);

  std::cout << "Updated bin counts are: "
            << std::endl;

  for (unsigned int i = 0; i < hist.count.size(); i++)
  {
    std::cout << hist.count[i]
              << ' ';
  }
  std::cout << std::endl;



  // Check create histogram function
  HistVector labels = { 0,4.5,1.7, 1.7 };
  HistVector counts = { 1,1.1,3,2 };

  Histograms::Histogram hist2 = Histograms::create_histogram(counts, labels, prm);
  std::cout << "Created histogram properties:"
            << std::endl;

  std::cout << "Interval points are: "
            << std::endl;
  for (unsigned int i = 0; i < hist2.interval_pts.size(); i++)
  {
    std::cout << hist2.interval_pts[i]
              << ' ';
  }
  std::cout << std::endl;

  std::cout << "Maximum x value is: "
            << hist2.max_x
            << std::endl;

  std::cout << "Minimum x value is: "
            << hist2.min_x
            << std::endl;

  std::cout << "Number of bins is: "
            << hist2.num_bins
            << std::endl;

  std::cout << "Bin counts are: "
            << std::endl;

  for (unsigned int i = 0; i < hist2.count.size(); i++)
  {
    std::cout << hist2.count[i]
              << ' ';
  }
  std::cout << std::endl;
}
