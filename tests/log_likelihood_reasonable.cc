#include <iostream>
#include <string>
#include <valarray>
#include "models.h"
#include "histogram.h"
#include "statistics.h"
#include "data.h"


using DataVector = std::valarray<double>;



int main()
{
  // Start with Three-Step alternative nucleation model for Ir-POM system
    // Model creation
    Models::ThreeStepAlternative::Parameters prm(3.6e-2, 7.27e4, 6.45e4, 1.63e4, 5.56e3, 11.3, 3, 2500, 274);
    Models::ThreeStepAlternative model;

    // Data for the model
    const Data::PomData pom_data;

    // set up initial conditions, start time, and end time
    std::valarray<double> ic(0.0, prm.n_variables);
    ic[0] = 0.0012;
    const std::vector<double> sol_times = {0.0, pom_data.tem_time1, pom_data.tem_time2, pom_data.tem_time3, pom_data.tem_time4};


  // Solve the ODE and save the solution for all times we have data for
  const std::vector<std::valarray<double>> solutions = Models::integrate_ode_ee_many_times(ic, model, prm, sol_times);


  // loop over all times to compute log likelihood at each time
    // put all data into a single vector
    const std::vector<DataVector> tem_data = {pom_data.tem_diam_time1,
                                              pom_data.tem_diam_time2,
                                              pom_data.tem_diam_time3,
                                              pom_data.tem_diam_time4};

    for (unsigned int time = 0; time < tem_data.size(); ++time)
    {
      // Isolate particles from the ODE -- i.e. eliminate precursor and other tracked quantities other than particles
        // Determine smallest and largest particle size
        // These quantities will determine how big of an array we need
        const unsigned int smallest_particle = model.getSmallestParticleSize(prm);
        const unsigned int largest_particle = model.getLargestParticleSize(prm);


        // Initialize arrays which describe all of the particle sizes we work with and their corresponding concentrations
        std::valarray<double> sizes(largest_particle - smallest_particle + 1);
        std::valarray<double> concentrations(sizes.size());

        // Loop through all possible particle sizes, add the particle size to the sizes array, and find that particle
        // size's concentration
        for (unsigned int size = smallest_particle; size < largest_particle+1; ++size)
        {
          sizes[size - smallest_particle] = size;
          // solutions[0] corresponds to the initial condition
          // we want solutions[time+1] because solutions[0] is the initial condition,
          // solutions[1] is the solution for time1, etc.
          concentrations[size - smallest_particle] = solutions[time+1][model.particleSizeToIndex(size, prm)];

        }


      // Compute the log likelihood
        // We want to compute the log likelihood for the Ir-POM data at the first available time point
        // and we need to convert from the data being measured in diameter to number of particles
        const std::valarray<double> data_sizes = std::pow(tem_data[time]/0.3000805, 3);

        // Define the parameters for the histogram -- i.e. how many bins and their total span
        const Histograms::Parameters hist_prm(25, smallest_particle, largest_particle);

        double likelihood = Statistics::log_likelihood(data_sizes, concentrations, sizes, hist_prm);


      // print result
      std::cout << "log likelihood: "
                << likelihood
                << std::endl;
    }
}
