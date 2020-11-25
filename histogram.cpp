#include "histogram.h"
#include <vector>
#include <stdexcept>
#include <iostream>


using HistVector = std::vector<double>;



// constructor -- require all parameters specified when creating this object.
Histograms::Parameters::Parameters(const unsigned int n_bins_value,
                                   const double x_start_value,
                                   const double x_end_value)
{
  n_bins = n_bins_value;
  x_start = x_start_value;
  x_end = x_end_value;
}



// constructor -- require parameters be passed in upon creation of this object.
Histograms::Histogram::Histogram(const Histograms::Parameters& parameters)
{
  // receive values from parameters
  min_x = parameters.x_start;
  max_x = parameters.x_end;
  num_bins = parameters.n_bins;

  // initialize interval points array
  HistVector pts(num_bins + 1);

  // create interval points with uniform spaces between bins
  pts[0] = min_x;
  pts[num_bins] = max_x;
  const double dx = (max_x - min_x) / num_bins;
  for (unsigned int i = 1; i < num_bins; i++)
  {
    pts[i] = min_x + i*dx;
  }

  interval_pts = pts;

  // initialize count array -- set all counts to zero since no data is included yet
  HistVector c(num_bins, 0.);
  count = c;
}



void Histograms::Histogram::AddToBins(const HistVector& y,
                                      const HistVector& x)
{
  // ensure x and y are the same size
  if (x.size() != y.size())
    throw std::domain_error("There must be an equal number of labels and counts.");

  // loop through supplied data
  for (unsigned int i = 0; i < x.size(); i++)
  {
    // throw away if x is outside of [min_x, max_x]
    if (x[i] > max_x || x[i] < min_x)
    {
      continue;
    }
    // last bin is on a closed interval, whereas all other bins are half-open - i.e. [a, b)
    else if (x[i] == max_x)
    {
      count[num_bins-1] += y[i];
    }
    else
    {
      unsigned int interval_index = bisection_method(x[i]);
      count[interval_index] += y[i];
    }
  }
}



unsigned int Histograms::Histogram::bisection_method(double x)
{
  // set right and left endpoints of search
  // searching for the index value i where x in [interval_pts[i], interval_pts[i+1])
  // data is assumed unstructured, so we always start by considering all bins
  unsigned int index_left = 0;
  unsigned int index_right = num_bins;
  unsigned int middle = (index_right + index_left) / 2;

  // check middle index
  bool found = false;
  while (!found)
  {
    // check if x is in the middle bin -- if it is, we're done!
    if (x >= interval_pts[middle] && x < interval_pts[middle + 1])
    {
      found = true;
      break;
    }
    // if x is larger than the middle bin, then we do not need to check bins left,...,middle anymore
    else if (x >= interval_pts[middle + 1])
    {
      index_left = middle + 1;
    }
    // if x is smaller than the middle bin, then we do not need to check bins middle,...,right anymore
    else
    {
      index_right = middle;
    }
    // determine midpoint of considered values
    middle = (index_right + index_left) / 2;
  }
  return middle;
}



// This function creates a histogram.
// Given the x-values (labels) and y-values (counts) of the user's data, along with the appropriate parameters,
// this function creates a histogram object and adds the data to populate the bins.
Histograms::Histogram Histograms::create_histogram(const std::vector<double>& counts,
                                                   const std::vector<double>& labels,
                                                   const Parameters& parameters)
{
  // create Histogram object using supplied Parameters
  Histograms::Histogram histogram(parameters);

  // add data to the Histogram
  histogram.AddToBins(counts, labels);

  return histogram;
}

