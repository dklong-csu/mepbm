#include "sampling_sundials.h"
#include "src/ir_pom_data.h"
#include "src/get_subset.h"

#include <omp.h>



using Real = double;
using Vector = Eigen::VectorXd;
using Matrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB<Matrix, Eigen::IncompleteLUT<double>>;
using SampleType = Eigen::Matrix<double, 1, Eigen::Dynamic>;


Real growth_kernel(const unsigned int size, const Real k)
{
  return k * (1.0 * size) * (2.677 * std::pow(1.0*size, -0.28));
}



MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_ode_4step(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const Real k4, const unsigned int M)
{
  constexpr unsigned int A_index = 0;
  constexpr unsigned int As_index = 1;
  constexpr unsigned int ligand_index = 2;
  constexpr unsigned int max_size = 2500;
  constexpr unsigned int conserved_size = 1;
  constexpr Real solvent = 11.3;


  MEPBM::Species A(A_index);
  MEPBM::Species As(As_index);
  MEPBM::Species L(ligand_index);
  MEPBM::Particle B(3,M,3);
  MEPBM::Particle C(M+1,max_size,M+1);

  MEPBM::ChemicalReaction<Real, Matrix> rxn1({ {A,1} },
                                             { {As,1}, {L,1} },
                                             solvent*solvent*kf);
  MEPBM::ChemicalReaction<Real, Matrix> rxn2({ {As, 1}, {L,1} },
                                             { {A,1} },
                                             kb);
  auto B_nuc = B.species(3);
  MEPBM::ChemicalReaction<Real, Matrix> rxn3({{As,2}, {A,1}},
                                             {{B_nuc,1}, {L,1}},
                                             k1);
  MEPBM::ParticleGrowth<Real, Matrix> rxn4(B,conserved_size,max_size,[&](const unsigned int size){return growth_kernel(size, k2);},{{A,1}},{{L,1}});
  MEPBM::ParticleGrowth<Real, Matrix> rxn5(C, conserved_size, max_size, [&](const unsigned int size){return growth_kernel(size, k3);}, {{A,1}},{{L,1}});

  MEPBM::ParticleAgglomeration<Real, Matrix> rxn6(B, B, max_size, 
      [&](const unsigned int sizeA, const unsigned int sizeB){return k4*growth_kernel(sizeA,1.0)*growth_kernel(sizeB, 1.0);},
      {}, {});

  MEPBM::ChemicalReactionNetwork<Real, Matrix> mech({rxn1,rxn2,rxn3},{rxn4,rxn5},{rxn6});
  return mech;
}




MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_ode_3step(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const unsigned int M)
{
  constexpr unsigned int A_index = 0;
  constexpr unsigned int As_index = 1;
  constexpr unsigned int ligand_index = 2;
  constexpr unsigned int max_size = 2500;
  constexpr unsigned int conserved_size = 1;
  constexpr Real solvent = 11.3;


  MEPBM::Species A(A_index);
  MEPBM::Species As(As_index);
  MEPBM::Species L(ligand_index);
  MEPBM::Particle B(3,M,3);
  MEPBM::Particle C(M+1,max_size,M+1);

  MEPBM::ChemicalReaction<Real, Matrix> rxn1({ {A,1} },
                                             { {As,1}, {L,1} },
                                             solvent*solvent*kf);
  MEPBM::ChemicalReaction<Real, Matrix> rxn2({ {As, 1}, {L,1} },
                                             { {A,1} },
                                             kb);
  auto B_nuc = B.species(3);
  MEPBM::ChemicalReaction<Real, Matrix> rxn3({{As,2}, {A,1}},
                                             {{B_nuc,1}, {L,1}},
                                             k1);
  MEPBM::ParticleGrowth<Real, Matrix> rxn4(B,conserved_size,max_size,[&](const unsigned int size){return growth_kernel(size, k2);},{{A,1}},{{L,1}});
  MEPBM::ParticleGrowth<Real, Matrix> rxn5(C, conserved_size, max_size, [&](const unsigned int size){return growth_kernel(size, k3);}, {{A,1}},{{L,1}});

  MEPBM::ChemicalReactionNetwork<Real, Matrix> mech({rxn1,rxn2,rxn3},{rxn4,rxn5},{});
  return mech;
}



// Convert an Eigen vector to a N_Vector
N_Vector
create_nvec_from_vec(const Vector & e_vec)
{
  auto n_vec = MEPBM::create_eigen_nvector<Vector>(e_vec.size());
  auto v = static_cast<Vector*>(n_vec->content);
  for (unsigned int i=0;i<v->size();++i)
  {
    (*v)(i) = e_vec(i);
  }
  return n_vec;
}



N_Vector
create_ic()
{
  constexpr unsigned int max_size = 2500;
  Vector initial_condition = Vector::Zero(max_size + 1);
  initial_condition(0) = 0.0012;
  auto ic = create_nvec_from_vec(initial_condition);

  return ic;
}



int main (int argc, char** argv)
{
  // argv[1] = root file name
  // argv[2] = number of chains i.e. # of parallel processes that need to run

  const std::string root_file = argv[1];
  const int n_chains = std::atoi(argv[2]);
  std::vector<Real> log_likelihood_values(n_chains);

  #pragma omp parallel for
    for (unsigned int chain=0; chain<n_chains; ++chain){

    /*
    * Define the parameters
    * This needs to be updated for each analysis
    */
    const std::string input_file = root_file + "-" + std::to_string(chain+1) + ".inp";
    // std::cout << input_file << std::endl;
    const auto sample = MEPBM::import_parameters<SampleType>(input_file, 14);



    /*
    * Setup the ODE solver
    * The `max_steps` argument for `ode_solver` and the arguments to `set_tolerance` might need to be updated
    */
    auto ic = create_ic();
    auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic->ops->nvgetlength(ic), ic->ops->nvgetlength(ic));
    auto linear_solver = MEPBM::create_sparse_iterative_solver< Matrix, Real, Solver >();
    const Real t_start = 0;
    const Real t_end = 4.838;

    auto rhs3 = [](Real t, N_Vector x, N_Vector x_dot, void* user_data){
      const auto prm = *static_cast<SampleType*>(user_data); 
      const auto kf3 = prm[0];
      const auto kb3 = prm[1];
      const auto k13 = prm[2];
      const auto k23 = prm[3];
      const auto k33 = prm[4];
      const auto M3 = prm[5];
      const auto kf4 = prm[6];
      const auto kb4 = prm[7];
      const auto k14 = prm[8];
      const auto k24 = prm[9];
      const auto k34 = prm[10];
      const auto k44 = prm[11];
      const auto M4 = prm[12];
      const auto omega = prm[13];
      const auto rxns = create_ode_3step(kf3, kb3, k13, k23, k33, M3);
      auto rhs_fcn = rxns.rhs_function();
      auto err = rhs_fcn(t, x, x_dot, user_data);
      return err;
    };

    auto jac3 = [](Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
      const auto prm = *static_cast<SampleType*>(user_data); 
      const auto kf3 = prm[0];
      const auto kb3 = prm[1];
      const auto k13 = prm[2];
      const auto k23 = prm[3];
      const auto k33 = prm[4];
      const auto M3 = prm[5];
      const auto kf4 = prm[6];
      const auto kb4 = prm[7];
      const auto k14 = prm[8];
      const auto k24 = prm[9];
      const auto k34 = prm[10];
      const auto k44 = prm[11];
      const auto M4 = prm[12];
      const auto omega = prm[13];
      const auto rxns = create_ode_3step(kf3, kb3, k13, k23, k33, M3);
      auto jac_fcn = rxns.jacobian_function();
      auto err = jac_fcn(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
      return err;
    };


    auto rhs4 = [](Real t, N_Vector x, N_Vector x_dot, void* user_data){
      const auto prm = *static_cast<SampleType*>(user_data); 
      const auto kf3 = prm[0];
      const auto kb3 = prm[1];
      const auto k13 = prm[2];
      const auto k23 = prm[3];
      const auto k33 = prm[4];
      const auto M3 = prm[5];
      const auto kf4 = prm[6];
      const auto kb4 = prm[7];
      const auto k14 = prm[8];
      const auto k24 = prm[9];
      const auto k34 = prm[10];
      const auto k44 = prm[11];
      const auto M4 = prm[12];
      const auto omega = prm[13];
      const auto rxns = create_ode_4step(kf4, kb4, k14, k24, k34, k44, M4);
      auto rhs_fcn = rxns.rhs_function();
      auto err = rhs_fcn(t, x, x_dot, user_data);
      return err;
    };

    auto jac4 = [](Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
      const auto prm = *static_cast<SampleType*>(user_data); 
      const auto kf3 = prm[0];
      const auto kb3 = prm[1];
      const auto k13 = prm[2];
      const auto k23 = prm[3];
      const auto k33 = prm[4];
      const auto M3 = prm[5];
      const auto kf4 = prm[6];
      const auto kb4 = prm[7];
      const auto k14 = prm[8];
      const auto k24 = prm[9];
      const auto k34 = prm[10];
      const auto k44 = prm[11];
      const auto M4 = prm[12];
      const auto omega = prm[13];
      const auto rxns = create_ode_4step(kf4, kb4, k14, k24, k34, k44, M4);
      auto jac_fcn = rxns.jacobian_function();
      auto err = jac_fcn(t, x, x_dot, J, user_data, tmp1, tmp2, tmp3);
      return err;
    };

    MEPBM::CVODE<Real> ode_solver3(ic,
                                    template_matrix,
                                    linear_solver,
                                    rhs3,
                                    jac3,
                                    t_start,
                                    t_end,
                                    5000);

    ode_solver3.set_tolerance(1e-7, 1e-13);
    void * user_data = (void *)&sample;
    ode_solver3.set_user_data(user_data);


    MEPBM::CVODE<Real> ode_solver4(ic,
                                    template_matrix,
                                    linear_solver,
                                    rhs4,
                                    jac4,
                                    t_start,
                                    t_end,
                                    5000);

    ode_solver4.set_tolerance(1e-7, 1e-13);
    ode_solver4.set_user_data(user_data);



    /*
    * Load data including TEM sets and times for each data set and set up the binning histogram
    * This needs to be updated for each analysis
    */
    const MEPBM::PomData<Real> data;
    const MEPBM::Parameters<Real> hist_prm(27, 1.4, 4.1);

    const std::vector<Real> times = {data.tem_time1,
                                    data.tem_time2,
                                    data.tem_time3,
                                    data.tem_time4};
    const std::vector< std::vector<Real> > tem_data = {
        data.tem_diam_time1,
        data.tem_diam_time2,
        data.tem_diam_time3,
        data.tem_diam_time4
    };



    /*
    * Solve the ODE at each time and calculate the log likelihood
    * Update `log_likelihood += ...` if a different log likelihood function is desired
    */
    Real log_likelihood = 0.0;
    for (unsigned int i=0; i<times.size(); ++i) {
      // If the log likelihood is at lowest() then it must have been that the ODE solver failed and we want
      // to give a final value of lowest(). Hence we can skip the remaining ODE solves to save computation time.
      if (log_likelihood > std::numeric_limits<Real>::lowest()) {
        // std::cout << "about to solve" << std::endl;
        auto solution3 = ode_solver3.solve(times[i]);
        // std::cout << "solve 3 step" << std::endl;
        auto solution4 = ode_solver4.solve(times[i]);
        // std::cout << "solve 4 step" << std::endl;
        if (solution3.second != 0 || solution4.second != 0) {
          // ODE solver required many time steps which means the solution is probably unphysical
          log_likelihood = std::numeric_limits<Real>::lowest();
        }
        else {
          auto particles3 = MEPBM::get_subset<Real>(solution3.first, 3, 2500);
          auto particles4 = MEPBM::get_subset<Real>(solution4.first, 3, 2500);
          const Real omega = sample[13];
          Vector particles = omega*particles3 + (1.- omega)*particles4;
          std::vector<Real> diams;
          for (unsigned int i=3; i<=2500; ++i){
            diams.push_back(MEPBM::atoms_to_diameter<Real>(i));
          }
          log_likelihood += MEPBM::log_multinomial(particles,
                                                  diams,
                                                  tem_data[i],
                                                  hist_prm);
        }
      }
    }

    log_likelihood_values[chain] = log_likelihood;
  }




  /*
   * Output the final log likelihood value to a file
   * No updates should be needed
   */
  const std::string out = root_file + ".out";
  std::ofstream out_file;
  out_file.open(out);
  for (const auto ll : log_likelihood_values){
    out_file << ll << "\n";
  }
  out_file.close();

  return EXIT_SUCCESS;
}