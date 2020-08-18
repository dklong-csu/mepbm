#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include <valarray>
#include <stdexcept>


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
      const double x_end_value);
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
    std::valarray<double> interval_pts, count;

    // constructor -- require parameters be passed in upon creation of this object.
    Histogram(const Histograms::Parameters& parameters);


    void AddToBins(const std::valarray<double>& y,
             const std::valarray<double>& x);


  private:
    unsigned int bisection_method(double x);
  };




  // This function creates a histogram.
  // Given the x-values (labels) and y-values (counts) of the user's data, along with the appropriate parameters,
  // this function creates a histogram object and adds the data to populate the bins.
  Histogram create_histogram(const std::valarray<double>& counts,
    const std::valarray<double>& labels,
    const Parameters& parameters);
}


#endif /* HISTOGRAM_H_ */
