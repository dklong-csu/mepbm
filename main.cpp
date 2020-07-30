#include <vector>
#include <valarray>
#include <stdexcept>
#include <cmath>
#include <string>
#include <iostream>


using RateVector = std::valarray<double>;
using StateVector = std::valarray<double>;
using GrowthVector = std::valarray<double>;
using HistVector = std::valarray<double>;

namespace Histograms
{
	/*
	This class describes the parameters for a histogram
	*/
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


	*/
	
	class Histogram
	{
	public:
		double min_x, max_x;
		unsigned int num_bins;
		HistVector interval_pts, count;

		// constructor
		Histogram(const Histograms::Parameters& parameters)
		{
			// receive values from parameters
			min_x = parameters.x_start;
			max_x = parameters.x_end;
			num_bins = parameters.n_bins;

			// initialize interval points array
			HistVector pts(num_bins + 1);

			pts[0] = min_x;
			pts[num_bins] = max_x;
			const double dx = (max_x - min_x) / num_bins;
			for (unsigned int i = 1; i < num_bins; i++)
			{
				pts[i] = min_x + i*dx;
			}

			interval_pts = pts;

			// initialize count array
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
			unsigned int index_left = 0;
			unsigned int index_right = num_bins;

			// check middle index
			bool not_found = true;
			while (not_found)
			{
				unsigned int middle = (index_right + index_left) / 2;
				if (x >= interval_pts[middle] && x < interval_pts[middle + 1])
				{
					not_found = false;
					return middle;
				}
				else if (x >= interval_pts[middle + 1])
				{
					index_left = middle + 1;
				}
				else
				{
					index_right = middle;
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

namespace Models
{
	double available_atoms(const double& size)
	{
		return size * 2.677 * std::pow(size, -0.28);
	}

	class ParametersBase
	{
	public:
		// what does this do?
		virtual ~ParametersBase() = default;
	};

	class ModelsBase
	{
	public:
		virtual RateVector right_hand_side(const StateVector& x,
										   const ParametersBase& parameters) const = 0;

	};
	

	class TwoStep : public ModelsBase
	{
	public:
		class Parameters : public ParametersBase
		{
		public:
			double k1, k2;
			unsigned int w, maxsize, n_variables;

			// constructor
			Parameters(const double k1_value,
					   const double k2_value,
					   const unsigned int nucleation_order,
					   const unsigned int maxsize_value)
			{
				k1 = k1_value;
				k2 = k2_value;
				w = nucleation_order;
				maxsize = maxsize_value;
				n_variables = maxsize - w + 2;
			}
		};

		virtual RateVector right_hand_side(const StateVector& x,
										   const ParametersBase& parameters) const
		{
			const Models::TwoStep::Parameters& two_step_parameters = dynamic_cast<const Models::TwoStep::Parameters&>(parameters);

			/*
			f[0] = dn_0
			f[1] = dn_w
			f[2] = dn_{w+1}
			...
			f[two_step_parameters.n_variable - 1) = dn_{maxsize}
			*/
			RateVector f(two_step_parameters.n_variables);
			// loss from nucleation
			f[0] = -((int)two_step_parameters.w * (int)two_step_parameters.k1 * std::pow(x[0], 1. * (int)two_step_parameters.w));
			// gain from nucleation
			f[1] = two_step_parameters.k1 * std::pow(x[0], 1. * two_step_parameters.w);
			for (unsigned int i = 2; i < two_step_parameters.n_variables; i++)
			{
				// gain from growth
				f[i] = two_step_parameters.k2 * x[0] * available_atoms(two_step_parameters.w + i - 2) * x[i - 1];
				f[0] -= f[i];
			}

			for (unsigned int i = 1; i < two_step_parameters.n_variables - 1; i++)
			{
				// loss from growth
				f[i] -= f[i + 1];
			}

			// loss from growth on largest particle -- I'm torn about including this
			f[two_step_parameters.n_variables - 1] -= two_step_parameters.k2 * x[0] * available_atoms(two_step_parameters.maxsize) * x[two_step_parameters.n_variables - 1];
			f[0] -= two_step_parameters.k2 * x[0] * available_atoms(two_step_parameters.maxsize) * x[two_step_parameters.n_variables - 1];
			return f;
		}
	};
	
	class TwoStepAlternative : public ModelsBase
	{
	public:
		class Parameters : public ParametersBase
		{
		public:
			double k1, k2, k_forward, k_backward, solvent;
			unsigned int w, maxsize, n_variables;

			// constructor
			Parameters(const double k_forward_value,
					   const double k_backward_value,
					   const double k1_value,
					   const double k2_value,
					   const double solvent_value,
					   const unsigned int nucleation_order,
					   const unsigned int maxsize_value)
			{
				k_forward = k_forward_value;
				k_backward = k_backward_value;
				k1 = k1_value;
				k2 = k2_value;
				w = nucleation_order;
				maxsize = maxsize_value;
				n_variables = maxsize - w + 4;
				solvent = solvent_value;
			}
		};

		virtual RateVector right_hand_side(const StateVector& x,
										   const ParametersBase& parameters) const
		{
			const Models::TwoStepAlternative::Parameters& two_step_alt_parameters = dynamic_cast<const Models::TwoStepAlternative::Parameters&>(parameters);

			/*
			f[0] = dn_0
			f[1] = dn_s
			f[2] = dp
			f[3] = dn_w
			f[4] = dn_{w+1}
			...
			f[two_step_alt_parameters.n_variables - 1] = dn_{maxsize}
			*/
			RateVector f(two_step_alt_parameters.n_variables);
			// precursor -- loss from dissociative step, gain from dissociative step
			f[0] = -two_step_alt_parameters.k_forward * x[0] * two_step_alt_parameters.solvent * two_step_alt_parameters.solvent + two_step_alt_parameters.k_backward*x[1]*x[2];
			// dissasociated precursor -- opposite effect from the dissociative step as the precursor
			f[1] = -f[0];
			// precursor -- loss from nucleation
			f[0] -= two_step_alt_parameters.k1 * x[0] * x[1] * x[1];
			// dissasociated precursor -- loss from nucleation
			f[1] -= 2* two_step_alt_parameters.k1 * x[0] * x[1] * x[1];
			// skip ligand (p) for now since f[2] = - f[0] and we still need to update f[0]

			// nucleated particle -- gain from nucleation
			f[3] = two_step_alt_parameters.k1 * x[0] * x[1] * x[1];
			for (unsigned int i = 4; i < two_step_alt_parameters.n_variables; i++)
			{
				// particle gain from growth
				f[i] = two_step_alt_parameters.k2 * x[0] * available_atoms(i - 1) * x[i - 1];
				// precursor loss from growth
				f[0] -= f[i];
			}

			for (unsigned int i = 3; i < two_step_alt_parameters.n_variables - 1; i++)
			{
				// loss from growth
				f[i] -= f[i + 1];
			}

			// loss from growth on largest particle -- I'm torn about including this
			f[two_step_alt_parameters.n_variables - 1] -= two_step_alt_parameters.k2 * x[0] * available_atoms(two_step_alt_parameters.maxsize) * x[two_step_alt_parameters.n_variables - 1];
			f[0] -= two_step_alt_parameters.k2 * x[0] * available_atoms(two_step_alt_parameters.maxsize) * x[two_step_alt_parameters.n_variables - 1];
			// assign ligand rate now that precursor rate is final
			f[2] = -f[0];

			return f;
		}
	};

	class ThreeStep : public ModelsBase
	{
	public:
		class Parameters : public ParametersBase
		{
		public:
			double k1, k2, k3;
			unsigned int w, maxsize, n_variables, particle_size_cutoff;

			// constructor
			Parameters(const double k1_value,
					   const double k2_value,
					   const double k3_value,
					   const unsigned int nucleation_order,
					   const unsigned int maxsize_value,
					   const unsigned int particle_size_cutoff_value)
			{
				k1 = k1_value;
				k2 = k2_value;
				k3 = k3_value;
				w = nucleation_order;
				maxsize = maxsize_value;
				n_variables = maxsize - w + 2;
				particle_size_cutoff = particle_size_cutoff_value;
			}
		};

		double rate_constant(const unsigned int& size,
									 const ParametersBase& parameters) const
		{
			const Models::ThreeStep::Parameters& three_step_parameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);
			if (size <= three_step_parameters.particle_size_cutoff)
				return three_step_parameters.k2;
			else
				return three_step_parameters.k3;
		}

		virtual RateVector right_hand_side(const StateVector& x,
										   const ParametersBase& parameters) const
		{
			const Models::ThreeStep::Parameters& three_step_parameters = dynamic_cast<const Models::ThreeStep::Parameters&>(parameters);

			/*
			f[0] = dn_0
			f[1] = dn_w
			f[2] = dn_{w+1}
			...
			f[three_step_parameters.n_variable - 1) = dn_{maxsize}
			*/
			RateVector f(three_step_parameters.n_variables);
			// precursor -- loss from nucleation
			f[0] = -(int)three_step_parameters.w * three_step_parameters.k1 * std::pow(x[0], 1. * three_step_parameters.w);
			// nucleated particle -- gain from nucleation
			f[1] = three_step_parameters.k1 * std::pow(x[0], 1. * three_step_parameters.w);
			for (unsigned int i = 2; i < three_step_parameters.n_variables; i++)
			{
				// particle gain from growth
				f[i] = rate_constant(three_step_parameters.w + i - 2,three_step_parameters)*x[0]* available_atoms(three_step_parameters.w + i - 2) * x[i - 1];
				f[0] -= f[i];
			}

			for (unsigned int i = 1; i < three_step_parameters.n_variables - 1; i++)
			{
				// particle loss from growth
				f[i] -= f[i + 1];
			}

			// loss from growth on largest particle -- I'm torn about including this
			f[three_step_parameters.n_variables - 1] -= rate_constant(three_step_parameters.maxsize,three_step_parameters)* x[0] * available_atoms(three_step_parameters.maxsize) * x[three_step_parameters.n_variables - 1];
			f[0] -= rate_constant(three_step_parameters.maxsize, three_step_parameters) * x[0] * available_atoms(three_step_parameters.maxsize) * x[three_step_parameters.n_variables - 1];
			return f;
		}
	};

	class ThreeStepAlternative : public ModelsBase
	{
	public:
		class Parameters : public ParametersBase
		{
		public:
			double k1, k2, k3, k_forward, k_backward, solvent;
			unsigned int w, maxsize, n_variables, particle_size_cutoff;

			// constructor
			Parameters(const double k_forward_value,
					   const double k_backward_value,
					   const double k1_value,
					   const double k2_value,
					   const double k3_value,
					   const double solvent_value,
					   const unsigned int nucleation_order,
					   const unsigned int maxsize_value,
					   const unsigned int particle_size_cutoff_value)
			{
				k_forward = k_forward_value;
				k_backward = k_backward_value;
				k1 = k1_value;
				k2 = k2_value;
				k3 = k3_value;
				w = nucleation_order;
				maxsize = maxsize_value;
				n_variables = maxsize - w + 4;
				particle_size_cutoff = particle_size_cutoff_value;
				solvent = solvent_value;
			}
		};

		double rate_constant(const unsigned int& size,
			const ParametersBase& parameters) const
		{
			const Models::ThreeStepAlternative::Parameters& three_step_alt_parameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);
			if (size <= three_step_alt_parameters.particle_size_cutoff)
				return three_step_alt_parameters.k2;
			else
				return three_step_alt_parameters.k3;
		}

		virtual RateVector right_hand_side(const StateVector& x,
			const ParametersBase& parameters) const
		{
			const Models::ThreeStepAlternative::Parameters& three_step_alt_parameters = dynamic_cast<const Models::ThreeStepAlternative::Parameters&>(parameters);

			/*
			f[0] = dn_0
			f[1] = dn_s
			f[2] = dp
			f[3] = dn_w
			f[4] = dn_{w+1}
			...
			f[three_step_alt_parameters.n_variable - 1) = dn_{maxsize}
			*/
			RateVector f(three_step_alt_parameters.n_variables);
			// precursor -- loss from dissociative step, gain from dissociative step
			f[0] = -three_step_alt_parameters.k_forward * x[0] * three_step_alt_parameters.solvent * three_step_alt_parameters.solvent + three_step_alt_parameters.k_backward * x[1]*x[2];
			// dissasociated precursor -- opposite effect from the dissociative step as the precursor
			f[1] = -f[0];
			// precursor -- loss from nucleation
			f[0] -= three_step_alt_parameters.k1 * x[0] * x[1] * x[1];
			// dissasociated precursor -- loss from nucleation
			f[1] -= 2 * three_step_alt_parameters.k1 * x[0] * x[1] * x[1];
			// skip ligand (p) for now since f[2] = - f[0] and we still need to update f[0]

			// nucleated particle -- gain from nucleation
			f[3] = three_step_alt_parameters.k1 * x[0] * x[1] * x[1];

			for (unsigned int i = 4; i < three_step_alt_parameters.n_variables; i++)
			{
				// particle gain from growth
				f[i] = rate_constant(three_step_alt_parameters.w + i - 4, three_step_alt_parameters) * x[0] * available_atoms(three_step_alt_parameters.w + i - 4) * x[i - 1];
				f[0] -= f[i];
			}

			for (unsigned int i = 3; i < three_step_alt_parameters.n_variables - 1; i++)
			{
				// loss from growth
				f[i] -= f[i + 1];
			}

			// loss from growth on largest particle -- I'm torn about including this
			f[three_step_alt_parameters.n_variables - 1] -= rate_constant(three_step_alt_parameters.maxsize,three_step_alt_parameters) * x[0] * available_atoms(three_step_alt_parameters.maxsize) * x[three_step_alt_parameters.n_variables - 1];
			f[0] -= rate_constant(three_step_alt_parameters.maxsize, three_step_alt_parameters) * x[0] * available_atoms(three_step_alt_parameters.maxsize) * x[three_step_alt_parameters.n_variables - 1];
			// assign ligand rate now that precursor rate is final
			f[2] = -f[0];

			return f;
		}
	};

	// Function to integrate the ODE
	// Explicit Euler
	StateVector integrate_ode_explicit_euler(const StateVector& x0,
											 const Models::ModelsBase& model,
											 const Models::ParametersBase& parameters,
											 const double start_time,
											 const double end_time)
	{
		StateVector x = x0;
		double time_step = 1e-6;

		double time = start_time;
		while (time <= end_time)
		{
			// advance to next time
			if (time + time_step > end_time)
			{
				time_step = end_time - time;
				time = end_time;
			}
			else
			{
				time += time_step;
			}

			// explicit euler update step
			x += time_step * model.right_hand_side(x, parameters);
		}

		return x;
	}
}

namespace Debugging
{
	// function for successive right hand side calls to ensure it works
	void calc_rhs(Models::ModelsBase& model, StateVector& state, Models::ParametersBase& prm)
	{
		RateVector rhs_output = model.right_hand_side(state, prm);

		std::cout << "Right hand side evaluates to: " << std::endl;
		for (unsigned int i = 0; i < rhs_output.size(); i++)
		{
			std::cout << rhs_output[i] << ' ';
		}
		std::cout << std::endl;
	}
}
int main()
{
	// histogram debugging
	/*
	// check we can make proper histogram parameters
	Histograms::Parameters prm(25, 0., 4.5);

	std::cout << "Checking for Histograms::Parameters" << std::endl;
	std::cout << "Parameter n_bins is:" << prm.n_bins << std::endl;
	std::cout << "Parameter x_start is:" << prm.x_start << std::endl;
	std::cout << "Parameter x_end is:" << prm.x_end << std::endl;

	// check if we can make a histogram
	Histograms::Histogram hist(prm);

	std::cout << "Checking for Histograms::Histogram" << std::endl;
	std::cout << "Initial bin counts are: " << std::endl;

	for (unsigned int i = 0; i < hist.count.size(); i++)
	{
		std::cout << hist.count[i] << ' ';
	}
	std::cout << std::endl;

	std::cout << "Interval points are: " << std::endl;
	for (unsigned int i = 0; i < hist.interval_pts.size(); i++)
	{
		std::cout << hist.interval_pts[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << "Maximum x value is: " << hist.max_x << std::endl;
	std::cout << "Minimum x value is: " << hist.min_x << std::endl;
	std::cout << "Number of bins is: " << hist.num_bins << std::endl;

	// check AddToBin
	std::cout << "Checking for: AddToBin method" << std::endl;

	HistVector x = { 0, 4.5, 1.7, -1, 5, 1.7 };
	HistVector y = { 1, 1.1, 3, 20, 30, 2 };
	hist.AddToBins(y, x);

	HistVector x2;
	HistVector y2;

	// I wanted a domain error thrown in a case like this and I don't see anything

	HistVector x3 = { 1 };
	HistVector y3 = { 1, 2 };

	std::cout << "Updated bin counts are: " << std::endl;

	for (unsigned int i = 0; i < hist.count.size(); i++)
	{
		std::cout << hist.count[i] << ' ';
	}
	std::cout << std::endl;

	// Check create histogram function
	HistVector labels = { 0,4.5,1.7, 1.7 };
	HistVector counts = { 1,1.1,3,2 };

	Histograms::Histogram hist2 = Histograms::create_histogram(counts, labels, prm);
	std::cout << "Created histogram properties:" << std::endl;

	std::cout << "Interval points are: " << std::endl;
	for (unsigned int i = 0; i < hist2.interval_pts.size(); i++)
	{
		std::cout << hist2.interval_pts[i] << ' ';
	}
	std::cout << std::endl;
	std::cout << "Maximum x value is: " << hist2.max_x << std::endl;
	std::cout << "Minimum x value is: " << hist2.min_x << std::endl;
	std::cout << "Number of bins is: " << hist2.num_bins << std::endl;

	std::cout << "Bin counts are: " << std::endl;

	for (unsigned int i = 0; i < hist2.count.size(); i++)
	{
		std::cout << hist2.count[i] << ' ';
	}
	std::cout << std::endl;
	*/

	// models debugging

	/*
	// two step
	std::cout << "Two-step testing" << std::endl;

	Models::TwoStep::Parameters prm2step(100, 10, 3, 5);
	Models::TwoStep model2step;

	StateVector ic1 = { 1,0.5,0.3,0.2 };

	Debugging::calc_rhs(model2step, ic1, prm2step);
	// confirmed with calculations in excel
	*/

	/*
	// two step alt
	std::cout << "Two-step alt testing" << std::endl;

	Models::TwoStepAlternative::Parameters prm2stepalt(100, 70,60,40,2,3,5);
	Models::TwoStepAlternative model2stepalt;

	StateVector ic2 = { 1,.8,.6,.4,.2,.1 };

	Debugging::calc_rhs(model2stepalt, ic2, prm2stepalt);
	// confirmed with calculations in excel
	*/

	/*
	// three step
	std::cout << "Three-step testing" << std::endl;

	Models::ThreeStep::Parameters prm3step(100, 10, 5, 3, 6, 4);
	Models::ThreeStep model3step;

	StateVector ic3 = { 1,.8,.6,.4,.2 };

	Debugging::calc_rhs(model3step, ic3, prm3step);
	// confirmed with calculations in excel
	*/

	/*
	// three step alt
	std::cout << "Three-step alt testing" << std::endl;

	Models::ThreeStepAlternative::Parameters prm3stepalt(100, 90, 80, 70, 60, 2, 3, 6, 4);
	Models::ThreeStepAlternative model3stepalt;

	StateVector ic4 = { 1,.9,.8,.7,.6,.5,.4 };

	Debugging::calc_rhs(model3stepalt, ic4, prm3stepalt);
	// confirmed with calculations in excel
	*/


}