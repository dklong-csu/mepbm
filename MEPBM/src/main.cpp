// need to include Histogram.cpp and models.cpp --> header files?
#include <iostream>
#include <string>
#include <cctype>
#include <valarray>

using StateVector = std::valarray<double>;

void main()
{
	// ask for desired model
	std::cout << "Which model would you like to run? (Please enter Two Step or Three Step) "; // include more models as they are coded

	// read the choice
	std::string model_choice;
	std::cin >> model_choice;

	// request more information based on model
	if (model_choice == "Two Step")
	{
		// ask for parameters
		std::cout << "Enter values for k1, k2, and the order of nucleation: ";

		// read the choices
		double k1, k2;
		unsigned int w;
		std::cin >> k1;
		std::cin >> k2;
		std::cin >> w;

		// ask for maximum particle size
		std::cout << "Enter the maximum particle size (in number of atoms): ";

		// read the choice
		unsigned int max_size;
		std::cin >> max_size;

		// ask for initial condition
		std::cout << "Enter the initial concentration of the precursor: ";

		// read the choice
		double ic;
		std::cin >> ic;

		// create initial particle size distribution based on initial condition
		StateVector x0(0., max_size - w + 2);
		x0[0] = ic;

		// create model based on input
		const Models::TwoStep::Parameters prm(k1, k2, w);

		const Models::TwoStep model();

		// once model is established, integrate and create a histogram

		// ask for end time of ODE
		std::cout << "Enter the end time for the reaction: ";

		// read end time
		double end_time;
		std::cin >> end_time;

		// integrate ODE
		const StateVector x = Models::integrate_ode(x0, model, prm, 0, end_time);

		// create labels based on particle sizes
		StateVector sizes(x.size());
		sizes[0] = 1;
		for (unsigned int i = 0; i < sizes.size(); ++i)
		{
			sizes[i] = i + w - 1;
		}

		// ask for parameters for the histogram
		std::cout << "Enter the minimum particles size, maximum particle size, and number of bins for the histogram: ";

		// read in the choices
		double max_size, min_size;
		unsigned int num_bins;
		std::cin >> max_size;
		std::cin >> min_size;
		std::cin >> num_bins;

		const Histogram::Parameters prm_hist(num_bins, min_size, max_size);

		const Histogram create_histogram(x, sizes, prm_hist);

		// graphing utilities
	}
	else if (model_choice == "Three Step")
	{

	}
	else
	{
		// throw error
	}
}