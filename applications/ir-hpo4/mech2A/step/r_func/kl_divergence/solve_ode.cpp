#include "sampling_sundials.h"
#include "src/ir_hpo4_data.h"
#include "src/mechanism_IrHPO4.h"



using Real = double;
using Vector = Eigen::VectorXd;
using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IncompleteLUT<double>>;
using SampleType = Eigen::Matrix<double, 1, Eigen::Dynamic>;



int main (int argc, char** argv)
{
  /*
   * Check to make sure command line arguments are passed
   * This should not need adjustment
   */
  if (argc < 2){
    std::cout << "Usage: " << argv[0] << " <input filename>" << std::endl;
    std::cout << "Exiting..." << std::endl;
    return EXIT_FAILURE;
  }



  /*
   * Define the model and parameters
   * This needs to be updated for each analysis
   */
  const std::string input_file = argv[1];
  const auto sample = MEPBM::import_parameters<SampleType>(input_file, 6);

  MEPBM::ExperimentalDesign<Vector, Real> design(450,
                                                 11.7,
                                                 0.0025,
                                                 0.0625,
                                                 10.0);

  MEPBM::IrHPO4::Mech2A<Vector, Matrix, Real, SampleType> mech(design);

  const std::vector<unsigned int> growth_height_indices = {3, 4};
  const std::vector<unsigned int> growth_midpoint_sizes = {13};
  MEPBM::StepGrowthKernel<Real, SampleType> growth_kernel(&MEPBM::r_func<Real>,
                                                          growth_height_indices,
                                                          growth_midpoint_sizes);

  const std::vector<unsigned int> agglom_height_indices = {5};
  const std::vector<unsigned int> agglom_midpoint_indices = {13};
  MEPBM::StepAgglomerationKernel<Real, SampleType> agglomeration_kernel(&MEPBM::r_func<Real>,
                                                                        agglom_height_indices,
                                                                        agglom_midpoint_indices);



  /*
   * Setup the ODE solver
   * The `max_steps` argument for `ode_solver` and the arguments to `set_tolerance` might need to be updated
   */
  auto ic = mech.make_IC();
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic->ops->nvgetlength(ic), ic->ops->nvgetlength(ic));
  auto linear_solver = MEPBM::create_sparse_iterative_solver< Matrix, Real, Solver >();
  const Real t_start = 0;
  const Real t_end = design.get_end_time();

  auto rhs = [](Real t, N_Vector x, N_Vector x_dot, void* user_data){
    auto rxns = *static_cast<MEPBM::ChemicalReactionNetwork<Real, Matrix>*>(user_data);
    auto rhs_fcn = rxns.rhs_function();
    auto err = rhs_fcn(t, x, x_dot, user_data);
    return err;
  };

  auto jac = [](Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    auto rxns = *static_cast<MEPBM::ChemicalReactionNetwork<Real, Matrix>*>(user_data);
    auto jac_fcn = rxns.jacobian_function();
    auto err = jac_fcn(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
    return err;
  };

  MEPBM::CVODE<Real> ode_solver(ic,
                                template_matrix,
                                linear_solver,
                                rhs,
                                jac,
                                t_start,
                                t_end,
                                5000);

  ode_solver.set_tolerance(1e-7, 1e-13);
  auto rxns = mech.make_rxns(sample,
                             growth_kernel,
                             agglomeration_kernel);
  void * user_data = (void *)&rxns;
  ode_solver.set_user_data(user_data);



  /*
   * Load data including TEM sets and times for each data set and set up the binning histogram
   * This needs to be updated for each analysis
   */
  const MEPBM::HPO4Data<Real> data;
  const MEPBM::Parameters<Real> hist_prm(38, 0.4, 2.1);

  const std::vector<Real> times = {data.time1,
                                   data.time2,
                                   data.time3,
                                   data.time4};
  const std::vector< std::vector<Real> > tem_data = {
      data.tem_data_t1,
      data.tem_data_t2,
      data.tem_data_t3,
      data.tem_data_t4
  };



  /*
   * Solve the ODE at each time and calculate the log likelihood
   * Update `log_likelihood += ...` if a different log likelihood function is desired
   */
  const std::string matlab_output = input_file.substr(0, input_file.find('.')) + ".m";
  std::ofstream outputfile(matlab_output.c_str());

  Real log_likelihood = 0.0;
  for (unsigned int i=0; i<times.size(); ++i) {
    // If the log likelihood is at lowest() then it must have been that the ODE solver failed and we want
    // to give a final value of lowest(). Hence we can skip the remaining ODE solves to save computation time.
    if (log_likelihood > std::numeric_limits<Real>::lowest()) {
      auto solution = ode_solver.solve(times[i]);
      if (solution.second != 0) {
        // ODE solver required many time steps which means the solution is probably unphysical
        log_likelihood = std::numeric_limits<Real>::lowest();
        std::cout << "WARNING: ODE solver failed. Results will not be meaningful." << std::endl;
      }
      else {
        auto particles = mech.extract_particles(solution.first);

        outputfile << "tem_sol" << i << " = ["
                   << particles
                   << "];"
                   << std::endl;

        auto diams = mech.get_particle_diameters();
        log_likelihood -= MEPBM::js_divergence(particles,
                                                 diams,
                                                 tem_data[i],
                                                 hist_prm);
      }
    }
  }




  /*
   * Output the final log likelihood value to a file
   * No updates should be needed
   */
  const std::string output_file = input_file.substr(0, input_file.find('.')) + ".out";
  MEPBM::output_result(log_likelihood, output_file);

  return EXIT_SUCCESS;
}