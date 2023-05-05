#include "sampling_sundials.h"
#include "src/ir_pom_data.h"
#include "src/get_subset.h"


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
create_ode(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const unsigned int M)
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
  // argv[1] = file with parameters
  // argv[2] = output file name

  const std::string root_file = argv[1];
  const std::string root_output_name = argv[2];

  const auto sample = MEPBM::import_parameters<SampleType>(root_file, 6);

  /*
    * Setup the ODE solver
    * The `max_steps` argument for `ode_solver` and the arguments to `set_tolerance` might need to be updated
    */
    auto ic = create_ic();
    auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic->ops->nvgetlength(ic), ic->ops->nvgetlength(ic));
    auto linear_solver = MEPBM::create_sparse_iterative_solver< Matrix, Real, Solver >();
    const Real t_start = 0;
    const Real t_end = 4.838;

    auto rhs = [](Real t, N_Vector x, N_Vector x_dot, void* user_data){
      const auto prm = *static_cast<SampleType*>(user_data); 
      const auto kf = prm[0];
      const auto kb = prm[1];
      const auto k1 = prm[2];
      const auto k2 = prm[3];
      const auto k3 = prm[4];
      const auto M = prm[5];
      const auto rxns = create_ode(kf, kb, k1, k2, k3, M);
      auto rhs_fcn = rxns.rhs_function();
      auto err = rhs_fcn(t, x, x_dot, user_data);
      return err;
    };

    auto jac = [](Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
      const auto prm = *static_cast<SampleType*>(user_data); 
      const auto kf = prm[0];
      const auto kb = prm[1];
      const auto k1 = prm[2];
      const auto k2 = prm[3];
      const auto k3 = prm[4];
      const auto M = prm[5];
      const auto rxns = create_ode(kf, kb, k1, k2, k3, M);
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
    void * user_data = (void *)&sample;
    ode_solver.set_user_data(user_data);

    
    const MEPBM::PomData<Real> data;
    const std::vector<Real> times = {data.tem_time1,
                                    data.tem_time2,
                                    data.tem_time3,
                                    data.tem_time4};
    for (unsigned int i=0; i<times.size(); ++i){
      // solve ODE
      auto solution = ode_solver.solve(times[i]);
      // get particle concentrations only
      auto particles = MEPBM::get_subset<Real>(solution.first, 3, 2500);
      // output concentrations to file
      const std::string out = root_output_name + "-" + std::to_string(i+1) + ".out";
      std::ofstream out_file;
      out_file.open(out);
      out_file << particles;
      out_file.close();
    }


  return EXIT_SUCCESS;
}