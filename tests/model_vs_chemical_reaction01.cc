#include "sampling_sundials.h"
#include "sampling_custom_ode.h"
#include <iostream>



using Real = realtype;
using Vector = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
using Matrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;
using Solver = Eigen::BiCGSTAB< Matrix, Eigen::IncompleteLUT<Real> >;



Real growth_kernel(const unsigned int size, const Real k)
{
  return k * (1.0 * size) * (2.677 * std::pow(1.0*size, -0.28));
}



// Old way of solving the ODE which I know works
Model::Model<Real, Matrix>
create_old_ode(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const unsigned int M)
{
  constexpr unsigned int A_index = 0;
  constexpr unsigned int As_index = 1;
  constexpr unsigned int ligand_index = 2;
  constexpr unsigned int min_size = 3;
  constexpr unsigned int max_size = 2500;
  constexpr unsigned int conserved_size = 1;
  constexpr Real solvent = 11.3;

  // Create the model
  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> nucleation =
      std::make_shared<Model::TermolecularNucleation<Real, Matrix>>(A_index, As_index,
                                                                    ligand_index, min_size,
                                                                    kf, kb, k1, solvent);

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> small_growth =
      std::make_shared<Model::Growth<Real, Matrix>>(A_index, min_size, M,
                                                    max_size, ligand_index,
                                                    conserved_size, k2, min_size);

  std::shared_ptr<Model::RightHandSideContribution<Real, Matrix>> large_growth =
      std::make_shared<Model::Growth<Real, Matrix>>(A_index, M+1, max_size,
                                                    max_size, ligand_index,
                                                    conserved_size, k3, M+1);


  Model::Model<Real, Matrix> model(min_size, max_size);
  model.add_rhs_contribution(nucleation);
  model.add_rhs_contribution(small_growth);
  model.add_rhs_contribution(large_growth);

  return model;
}



// Old way of making initial condition
Vector
create_old_ic()
{
  constexpr unsigned int max_size = 2500;
  Vector initial_condition = Vector::Zero(max_size + 1);
  initial_condition(0) = 0.0012;

  return initial_condition;
}



// Old way of solving the ODE
Vector
solve_ode_old(const Model::Model<Real, Matrix> & model, const Real t)
{
  ODE::StepperBDF<4, Real, Matrix> stepper(model);
  auto ic = create_old_ic();
  auto solution = ODE::solve_ode<Real>(stepper, ic, 0, t, 1e-5); // pretty accurate time step
  return solution;
}



// New way of creating the ODE
MEPBM::ChemicalReactionNetwork<Real, Matrix>
create_new_ode(const Real kf, const Real kb, const Real k1, const Real k2, const Real k3, const unsigned int M)
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


// rhs function for ODE solver using mean value parameters
int cvode_rhs_func(Real t, N_Vector x, N_Vector x_dot, void * user_data)
{
  // values used
  const Real kf = 3.31e-3;
  const Real kb = 6.62e3;
  const Real k1 = 1.24e5;
  const Real k2 = 2.6e5;
  const Real k3 = 6.22e3;
  const unsigned int M = 107;

  auto mech = create_new_ode(kf,kb,k1,k2,k3,M);
  auto rhs = mech.rhs_function();
  auto err = rhs(t,x,x_dot,user_data);
  return err;
}



// jacobian function for ODE solver using mean value parameters
int cvode_jac_func(Real t, N_Vector x, N_Vector x_dot, SUNMatrix J, void * user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3)
{
  // values used
  const Real kf = 3.31e-3;
  const Real kb = 6.62e3;
  const Real k1 = 1.24e5;
  const Real k2 = 2.6e5;
  const Real k3 = 6.22e3;
  const unsigned int M = 107;

  auto mech = create_new_ode(kf,kb,k1,k2,k3,M);
  auto jac = mech.jacobian_function();
  auto err = jac(t,x,x_dot,J,user_data,tmp1, tmp2, tmp3);
  return err;
}



// New way of solving the ODE
Vector
solve_ode_new(const Real t)
{
  auto ic_evec = create_old_ic();
  auto ic = create_nvec_from_vec(ic_evec);
  auto template_matrix = MEPBM::create_eigen_sunmatrix<Matrix>(ic_evec.size(), ic_evec.size());
  auto linear_solver = MEPBM::create_sparse_iterative_solver<Matrix, Real, Solver>();
  MEPBM::CVODE<Real> ode_solver(ic, template_matrix, linear_solver,&cvode_rhs_func,&cvode_jac_func,0,1);
  ode_solver.set_tolerance(1e-7,1e-13);
  auto sol_pair = ode_solver.solve(t);
  auto sol = sol_pair.first;
  auto s = *static_cast<Vector*>(sol->content);

  ic->ops->nvdestroy(ic);
  template_matrix->ops->destroy(template_matrix);
  linear_solver->ops->free(linear_solver);
  sol->ops->nvdestroy(sol);

  return s;
}


// Reduce small entries to zero
Vector
prune_vector(const Vector & vec, const Real cutoff)
{
  const auto dim = vec.size();
  auto pruned = vec;
  for (unsigned int i=0; i<dim; ++i)
  {
    if (std::abs(vec(i)) < cutoff)
      pruned(i) = 0;
  }
  return pruned;
}



// Calculate percent difference between each entry in vector
// FIXME: make this relative to size of vector
Vector
percent_diff(const Vector & x, const Vector & y)
{
  assert(x.size() == y.size());
  Vector diff(x.size());
  for (unsigned int i=0; i<x.size();++i)
  {
    if (x(i) == 0)
      diff(i) = y(i);
    else if (y(i)==0)
      diff(i) = x(i);
    else
      diff(i) = std::abs( (x(i)-y(i))/x(i) );
  }
  return diff;
}



int main ()
{
  // values used
  const Real kf = 3.31e-3;
  const Real kb = 6.62e3;
  const Real k1 = 1.24e5;
  const Real k2 = 2.6e5;
  const Real k3 = 6.22e3;
  const unsigned int M = 107;

  const Real t_solv = 0.1;



  auto old_3step = create_old_ode(kf,kb,k1,k2,k3,M);

  auto new_3step = create_new_ode(kf, kb, k1, k2, k3, M);

  auto sol_old = solve_ode_old(old_3step, t_solv);
  auto sol_new = solve_ode_new(t_solv);

  // Prune the vectors so that "basically zero" values are set to zero to not throw off comparisons
  // e.g. 10^-13 and 10^-20 are both "basically zero" in this context but the % error is massive.
  auto pruned_old = prune_vector(sol_old, 1e-10);
  auto pruned_new = prune_vector(sol_new, 1e-10);

  auto diff = percent_diff(pruned_old, pruned_new);
  auto max_diff = diff.maxCoeff();
  // Check if within 5%
  std::cout << std::boolalpha << (max_diff < 0.05) << std::endl;


}