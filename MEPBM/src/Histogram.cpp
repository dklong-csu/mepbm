// Define histogram and related classes
#include <vector>
#include <valarray>
#include <stdexcept>

namespace Histograms
{
	// define parameters for the histogram
	// n_bins = number of bins to group data into
	// x_start = value at which the bins start
	// x_end = value at which the bins end
	class Parameters
	{
	public:
		unsigned int n_bins;
		double x_start, x_end;
		// constructor
		Parameters(const unsigned int n_bins_value,
			const double x_start_value,
			const double x_end_value)
		{
			n_bins = n_bins_value;
			x_start = x_start_value;
			x_end = x_end_value;

		}
	};
	

	/*
	This class describes a histogram.
	A histogram is defined to be a vector composed of bins.
	Each bin holds a minimum and maximum value, as well as a count.
	The minimum and maximum can be thought of as a half open interval [minimum, maximum),
	where the purpose of the bin is to accumulate contributions of objects whose label
	is in the interval [minimum, maximum).
	Count describes how many of the object is in the bin.
	
	To create a histogram, specify the number of bins and the start and end of the domain (i.e. the minimum and maximum value a label can take).
	The count of each bin is initialized to be zero.

	To add items to a bin, use the AddToBin method.
	This method takes in counts and labels (whose indices match appropriately) and distributes the counts to the appropriate bin.
		
	*/
	class Histogram
	{
	public:
		class Bin
		{
		public:
			double x_min, x_max, count;

			// constructor
			Bin(const double minimum_x,
				const double maximum_x)
			{
				x_min = minimum_x;
				x_max = maximum_x;
				count = 0;
			}
		};

		std::vector<Bin> bins;
		unsigned int num_bins;

		// constructor
		Histogram(const Parameters &parameters)
		{
			// initialize bins to have number_of_bins entries		
			num_bins = parameters.n_bins;
			std::vector<Bin> bins(num_bins);
			
			// define the first interval and calculate the width of the interval
			double x_left = parameters.x_start;
			double dx = (parameters.x_end - parameters.x_start) / parameters.n_bins;
			double x_right = x_left + dx;

			for (unsigned int i = 0; i < num_bins; i++)
			{
				// construct each bin
				bins[i] = Bin(x_left, x_right);
				// update intervals for each bin
				x_left = x_right;
				x_right += dx;
			}
		}

		// method to add to bins given arrays specifying counts and corresponding particle sizes
		void AddToBin(const std::valarray<double>& counts,
					  const std::valarray<double>& labels)
		{
			// check to see if counts and labels are the same size
			if (counts.size() != labels.size())
				throw std::domain_error("There must be an equal number of labels and counts.");

			// loop over x and y and add counts to the appropriate bin
			for (unsigned int i = 0; i < counts.size(); i++)
			{
				for (unsigned int j = 0; j < num_bins; j++)
				{
					if ((labels[i] >= bins[j].x_min) && (labels[i] < bins[j].x_max))
					{
						bins[j].count += counts[i];
						break;
					}
				}
			}
		}
		
	};

	/*
	This is a function to create a histogram

	Supply all of the labels and associated counts you would like to include in the histogram.
	Also supply the parameters which define the histogram.

	The function will create a Histogram object and add the desired data to the histogram.
	*/
	Histogram create_histogram(const std::valarray<double> &counts,
							   const std::valarray<double> &labels,
							   const Parameters &parameters)
	{
		// create Histogram object using supplied Parameters
		Histogram histogram(parameters); 

		// add data to the Histogram
		histogram.AddToBin(counts, labels);

		return histogram;
	}
}