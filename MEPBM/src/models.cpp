// Define model and related classes
#include <valarray>
#include <cmath>

using RateVector = std::valarray<double>;
using StateVector = std::valarray<double>;
using GrowthVector = std::valarray<double>;

namespace Models
{
	class ParametersBase
	{
	public:
		// what does = default do?
		virtual ~ParametersBase() = default;
	};

	class ModelsBase
	{
	public:
		double available_atoms(const unsigned int size)
		{
			return size * 2.677 * std::pow(size, -0.28);
		}
		// what does const = 0 do?
		virtual unsigned int n_variables() const = 0;

		virtual RateVector right_hand_side(const StateVector &x,
										   const ParametersBase &parameters) const = 0;
	};
	/*
	Would it make more sense to have a class hierarchy such as:
	ModelBase
	--> Classic Nucleation || Alternative Nucleation || Sintering || etc.
		--> Agglomeration || No Agglomeration
			--> Variable growth kernel to reflect 2-step, 3-step, etc.?
	*/

	class TwoStep : public ModelsBase
	{
	public:
		class Parameters : public ParametersBase
		{
		public:
			double k1, k2;
			unsigned int w;

			// constructor
			Parameters(const double k1_value,
					   const double k2_value,
					   const unsigned int nucleation_order)
			{
				k1 = k1_value;
				k2 = k2_value;
				w = nucleation_order;
			}
		};

		// confused by this
		virtual unsigned int n_variables() const
		{
			return 2500;
		}

		virtual RateVector right_hand_side(const StateVector &x,
										   const ParametersBase &parameters) const
		{
			// why is there an error on dynamic_cast?
			const Models::TwoStep::Parameters & two_step_parameters = dynamic_cast<Models::TwoStep::Parameters&>(parameters);

			RateVector f(n_variables());
			f[0] = -two_step_parameters.w * two_step_parameters.k1 * std::pow(x[1], 1. * two_step_parameters.w);
			f[1] = two_step_parameters.k1 * std::pow(x[1], 1. * two_step_parameters.w);
			for (unsigned int i = 2; i < n_variables(); i++)
			{
				// gain from growth
				// why is there an error on available_atoms?
				f[i] = two_step_parameters.k2 * x[0] * available_atoms(i - 1) * x[i - 1];
				f[0] -= f[i];
			}

			for (unsigned int i = 1; i < n_variables() - 1; i++)
			{
				// loss from growth
				f[i] -= f[i + 1];
			}
		}
	};

	class ThreeStep : public ModelsBase
	{
	public:
		class Parameters : public ParametersBase
		{
		public:
			double k1, k2, k3;
			unsigned int w, m;

			// constructor
			Parameters(const double k1_value,
				const double k2_value,
				const double k3_value,
				const unsigned int nucleation_order,
				const unsigned int particle_size_cutoff)
			{
				k1 = k1_value;
				k2 = k2_value;
				k3 = k3_value;
				w = nucleation_order;
				m = particle_size_cutoff;
			}
		};

		// confused by this
		virtual unsigned int n_variables() const
		{
			return 2500;
		}

		virtual RateVector right_hand_side(const StateVector& x,
			const ParametersBase& parameters) const
		{
			// why is there an error on dynamic_cast?
			const Models::ThreeStep::Parameters& three_step_parameters = dynamic_cast<Models::ThreeStep::Parameters&>(parameters);

			RateVector f(n_variables());
			f[0] = -three_step_parameters.w * three_step_parameters.k1 * std::pow(x[1], 1. * three_step_parameters.w);
			f[1] = three_step_parameters.k1 * std::pow(x[1], 1. * three_step_parameters.w);
			for (unsigned int i = 2; i < n_variables(); i++)
			{
				// gain from growth
				// why is there an error on available_atoms?
				if (i - 1 < three_step_parameters.m)
				{
					f[i] = three_step_parameters.k2 * x[0] * available_atoms(i - 1) * x[i - 1];
				}
				else
				{
					f[i] = three_step_parameters.k3 * x[0] * available_atoms(i - 1) * x[i - 1];
				}
				f[0] -= f[i];
			}

			for (unsigned int i = 1; i < n_variables() - 1; i++)
			{
				// loss from growth
				f[i] -= f[i + 1];
			}
		}
	};

	// Function to integrate the ODE
	// Explicit Euler
	StateVector integrate_ode(const StateVector& x0,
		const Models::ModelsBase& model,
		const Models::Parameters& parameters /*I'm confused how this works since it's defined within the model*/,
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
	}
}