#include "histogram.h"
#include <valarray>
#include <stdexcept>


using HistVector = std::valarray<double>;


namespace Histograms
{
  // This class describes the parameters for a histogram
  // A histogram needs to know the number of bins it has,
  // the smallest x-value, and the largest x-value.
  class Parameters
  {
  public:
    unsigned int n_bins;
    double x_start, x_end;

    // constructor -- require all parameters specified when creating this object.
    Parameters(const unsigned int n_bins_value,
      const double x_start_value,
      const double x_end_value)
    {
      n_bins = n_bins_value;
      x_start = x_start_value;
      x_end = x_end_value;
    }
  };



  //This class describes a histogram.
  // A histogram requires specification of the minimum/maximum x-value and the number of bins.
  // From these parameters, the points that define the boundaries of each bin are defined as interval_pts.
  // For example, interval_pts = { 0, 1, 2} corresponds to bin 0: x in [0,1); bin 1: x in [1,2); and bin 2: x in [2,3].
  // Currently, interval_pts is created based off uniform spacing.
  // A histogram also needs to keep track of how many items are in each bin.
  // This is accomplished through the variable count.
  // For example, count[i] indicates the number of items in the ith bin, i.e. x in [interval_pts[i], interval_pts[i+1]) (starting from bin 0).
  // The subroutine "AddToBins" adds the ability to add data to the histogram. The user provides two vectors corresponding to an x-value
  // and the corresponding amount of "stuff" (given as the y-value) that x-value has. The subroutine uses the x-value to decide which bin of the histogram
  // the data belongs to and adds the y-value to the appropriate entry of counts.
  class Histogram
  {
  public:
    double min_x, max_x;
    unsigned int num_bins;
    HistVector interval_pts, count;

    // constructor -- require parameters be passed in upon creation of this object.
    Histogram(const Histograms::Parameters& parameters)
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
      HistVector c(0., num_bins);
      count = c;
    }


    void AddToBins(const HistVector& y,
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


  private:
    unsigned int bisection_method(double x)
    {
      // set right and left endpoints of search
      // searching for the index value i where x in [interval_pts[i], interval_pts[i+1])
      // data is assumed unstructured, so we always start by considering all bins
      unsigned int index_left = 0;
      unsigned int index_right = num_bins;

      // check middle index
      bool not_found = true;
      while (not_found)
      {

        // determine midpoint of considered values
        unsigned int middle = (index_right + index_left) / 2;

        // check if x is in the middle bin -- if it is, we're done!
        if (x >= interval_pts[middle] && x < interval_pts[middle + 1])
        {
          not_found = false;
          return middle;
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

      }
    }
  };



  // This function creates a histogram.
  // Given the x-values (labels) and y-values (counts) of the user's data, along with the appropriate parameters,
  // this function creates a histogram object and adds the data to populate the bins.
  Histogram create_histogram(const std::valarray<double>& counts,
    const std::valarray<double>& labels,
    const Parameters& parameters)
  {
    // create Histogram object using supplied Parameters
    Histogram histogram(parameters);

    // add data to the Histogram
    histogram.AddToBins(counts, labels);

    return histogram;
  }
}
