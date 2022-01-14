#include "sample.h"
#include <iostream>
#include "nvector_eigen.h"
#include <eigen3/Eigen/Sparse>
#include "sampling_algorithm.h"
#include <string>
#include "data.h"
#include "src/models.h"
#include <vector>
#include "chemical_reaction.h"
#include <utility>
#include <chrono>


using Vector = Eigen::Matrix<realtype, Eigen::Dynamic, 1>;


realtype
growth_kernel(const unsigned int size)
{
  return size * 2.677 * std::pow(size * 1., -0.28);
}



template<typename Matrix>
Model::Model<realtype, Matrix>
four_step(const std::vector<realtype> real_prm,
          const std::vector<int> int_prm)
{
  /*
   * Mechanism is
   *    A <-> Asolv + L  (1)
   *    Asolv -> B_2 + L         (2)
   *    A + B_i -> B_{i+2} + L   (3)
   *    A + C_i -> C_{i+2} + L   (4)
   *    B_i + B_j -> B_{i+j}     (5)
   */

  static unsigned int A_index = 0;
  static unsigned int Asolv_index = 1;
  static unsigned int L_index = 2;
  static unsigned int small_particle_start_index = 3;
  static unsigned int smallest_particle_size = 2;
  static unsigned int largest_particle_size = 800;

  const unsigned int small_particle_end_index = int_prm[0];
  const unsigned int large_particle_start_index = small_particle_end_index + 1;
  const unsigned int large_particle_end_index = (largest_particle_size - smallest_particle_size) + small_particle_start_index;
  const unsigned int large_particle_first_size = (large_particle_start_index - small_particle_start_index) + smallest_particle_size;


  Model::Species A(A_index);
  Model::Species Asolv(Asolv_index);
  Model::Species L(L_index);
  Model::Particle B(small_particle_start_index, small_particle_end_index, smallest_particle_size);
  Model::Particle C(large_particle_start_index, large_particle_end_index, large_particle_first_size);

  // Eqn (1)
  const std::vector<std::pair<Model::Species, unsigned int>> eqn1_reactants = { {A,1}};
  const std::vector<std::pair<Model::Species, unsigned int>> eqn1_products = { {Asolv,1}, {L, 1}};
  const realtype kf = real_prm[0];
  std::shared_ptr< Model::RightHandSideContribution<realtype, Matrix> > eqn1f =
      std::make_shared< Model::ChemicalReaction<realtype, Matrix> >(eqn1_reactants,eqn1_products, kf);

  const realtype kb = real_prm[1];
  std::shared_ptr< Model::RightHandSideContribution<realtype, Matrix> > eqn1b =
      std::make_shared< Model::ChemicalReaction<realtype, Matrix> >(eqn1_products,eqn1_reactants, kb);

  // Eqn (2)
  const std::vector<std::pair<Model::Species, unsigned int>> eqn2_reactants = { {Asolv, 1}};
  auto B_2 = B.get_particle_species(3);
  const std::vector<std::pair<Model::Species, unsigned int>> eqn2_products = { {B_2, 1}, {L, 1}};
  const realtype k1 = real_prm[2];
  std::shared_ptr< Model::RightHandSideContribution<realtype, Matrix> > eqn2 =
      std::make_shared< Model::ChemicalReaction<realtype, Matrix> >(eqn2_reactants,eqn2_products, k1);

  // Eqn (3)
  static unsigned int growth_amount = 2;
  const realtype k2 = real_prm[3];
  const std::vector<std::pair<Model::Species, unsigned int>> eqn3_reactants = { {A, 1} };
  const std::vector<std::pair<Model::Species, unsigned int>> eqn3_products  = { {L, 2} };

  std::shared_ptr< Model::RightHandSideContribution<realtype, Matrix> > eqn3 =
      std::make_shared< Model::ParticleGrowth<realtype, Matrix> >(B,
                                                              k2,
                                                              growth_amount,
                                                              largest_particle_size,
                                                              &growth_kernel,
                                                              eqn3_reactants,
                                                              eqn3_products);

  // Eqn (4)
  const realtype k3 = real_prm[4];
  const std::vector<std::pair<Model::Species, unsigned int>> eqn4_reactants = { {A, 1} };
  const std::vector<std::pair<Model::Species, unsigned int>> eqn4_products  = { {L, 2} };

  std::shared_ptr< Model::RightHandSideContribution<realtype, Matrix> > eqn4 =
      std::make_shared< Model::ParticleGrowth<realtype, Matrix> >(C,
                                                              k3,
                                                              growth_amount,
                                                              largest_particle_size,
                                                              &growth_kernel,
                                                              eqn4_reactants,
                                                              eqn4_products);

  // Eqn (5)
  const realtype k4 = real_prm[5];
  const std::vector<std::pair<Model::Species, unsigned int>> eqn5_reactants = {};
  const std::vector<std::pair<Model::Species, unsigned int>> eqn5_products = {};

  std::shared_ptr< Model::RightHandSideContribution<realtype, Matrix> > eqn5 =
      std::make_shared< Model::ParticleAgglomeration<realtype, Matrix> >(B,
                                                                     B,
                                                                     k4,
                                                                     largest_particle_size,
                                                                     &growth_kernel,
                                                                     eqn5_reactants,
                                                                     eqn5_products);

  Model::Model<realtype, Matrix> m(smallest_particle_size,largest_particle_size);
  m.add_rhs_contribution(eqn1f);
  m.add_rhs_contribution(eqn1b);
  m.add_rhs_contribution(eqn2);
  m.add_rhs_contribution(eqn3);
  m.add_rhs_contribution(eqn4);
  m.add_rhs_contribution(eqn5);

  return m;
}




template<typename _Real, typename _Matrix, typename _Solver, LinearSolverClass _SolverClass>
std::vector<N_Vector>
ode_solve_test(const int n_solves)
{
  // Sets of parameters selected randomly (but the same for every instance of function)
  std::uint_fast32_t random_seed = std::hash<std::string>()(std::to_string(0));
  std::mt19937 rng;
  rng.seed(random_seed);

  std::vector<std::pair<_Real, _Real>> test_prm_bounds_real =
      {
          {1e-4, 1e-2}, /*kf*/
          {1e3, 1e5},   /*kb*/
          {1e4,1e6},    /*k1*/
          {1e4, 1e6},   /*k2*/
          {1e3, 1e5},   /*k3*/
          {1e1, 1e2}    /*k4*/
      };

  std::vector<std::pair<int,int>> test_prm_bounds_int =
      {
          {100, 700} /*M*/
      };

  std::cout << "M: ";
  std::vector<Sampling::Sample<_Real>> test_solve_samples;
  for (unsigned int n=0; n<n_solves; ++n)
  {
    std::vector<_Real> r_prm;
    // Loop through each parameter
    for (unsigned int k=0; k<test_prm_bounds_real.size(); ++k)
    {
      std::uniform_real_distribution<> dis(test_prm_bounds_real[k].first, test_prm_bounds_real[k].second);
      r_prm.push_back(dis(rng));
    }

    std::uniform_int_distribution<> dis(test_prm_bounds_int[0].first, test_prm_bounds_int[0].second);
    std::vector<int> i_prm;
    auto m_val = dis(rng);
    std::cout << m_val << " ";
    i_prm.push_back(m_val);
    Sampling::Sample<_Real> my_sample(r_prm, i_prm);
    test_solve_samples.push_back(my_sample);
  }
  std::cout << std::endl;

  std::vector<N_Vector> all_solutions;

  // Time the code
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::milliseconds(0));
  for (auto s : test_solve_samples) {


    std::vector<std::pair<_Real, _Real> > real_prm_bounds =
        {
            {0, 1e12},
            {0, 1e12},
            {0, 1e12},
            {0, 1e12},
            {0, 1e12},
            {0, 1e12}
        };


    std::vector<std::pair<int, int> > int_prm_bounds =
        {
            {5, 700}
        };

    std::function<Model::Model<_Real, _Matrix>(const std::vector<_Real>, const std::vector<int>)> create_model_fcn
        = four_step<_Matrix>;

    // Calculate the number of dimensions
    // Track species: A, A_s, L, A_1/2, sizes 1, 2, ... , 799, 800
    const unsigned int max_size = 800;
    const unsigned int n_vars = 4 + max_size;

    N_Vector initial_condition = create_eigen_nvector<Vector>(n_vars);

    auto ic_vec = static_cast<Vector *>(initial_condition->content);
    // Initial precursor (dimer) concentration is 0.0025 -- index 0
    // Initial ligand (HPO4) concentration is 0.0625 -- index 2
    for (unsigned int i = 0; i < ic_vec->size(); ++i) {
      if (i == 0)
        (*ic_vec)(i) = 0.0025;
      else if (i == 2)
        (*ic_vec)(i) = 0.0625;
      else
        (*ic_vec)(i) = 0.;
    }
    _Real start_time = 0;
    _Real end_time = 10.0;
    _Real abs_tol = 1e-13;
    _Real rel_tol = 1e-6;

    Data::IrHPO4Data<_Real> data;
    std::vector<_Real> times = {data.time4};

    unsigned int first_particle_index = 3;
    unsigned int last_particle_index = n_vars - 1;

    Histograms::Parameters<_Real> binning_parameters(25, 0.3, 2.8);
    unsigned int first_particle_size = 2;
    unsigned int particle_size_increase = 2;
    std::vector<std::vector<_Real> > data_sets = {data.tem_time4};

    Sampling::ModelingParameters<_Real, _Matrix> model_settings(real_prm_bounds,
                                                              int_prm_bounds,
                                                              create_model_fcn,
                                                              initial_condition,
                                                              start_time,
                                                              end_time,
                                                              abs_tol,
                                                              rel_tol,
                                                              times,
                                                              first_particle_index,
                                                              last_particle_index,
                                                              binning_parameters,
                                                              first_particle_size,
                                                              particle_size_increase,
                                                              data_sets);



    // Solve the ODE
    auto start = std::chrono::high_resolution_clock::now();
    auto solutions = SUNDIALS_Statistics::Internal::solve_ODE_from_sample<_Real,_Matrix,_Solver,_SolverClass>(s, model_settings);
    auto end = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    all_solutions.push_back(solutions.first[0]);
  }
  std::cout << duration.count() << " milliseconds" << std::endl;
  return all_solutions;
}



void
compare_solution_accuracy(const std::vector<N_Vector> inaccurate_sol, const std::vector<N_Vector> accurate_sol)
{
  realtype l2norm = 0;
  for (unsigned int i=0; i<inaccurate_sol.size();++i)
  {
    auto x = *static_cast<Vector*>(inaccurate_sol[0]->content);
    auto y = *static_cast<Vector*>(accurate_sol[0]->content);
    l2norm += (x-y).norm();
  }

  std::cout << l2norm/inaccurate_sol.size() << std::endl;
}


using RealTypeAllTests = realtype;
using MatrixTypeSparseTests = Eigen::SparseMatrix<realtype>;
using MatrixTypeSparseRowMajorTests = Eigen::SparseMatrix<realtype, Eigen::RowMajor>;
using MatrixTypeDenseTests = Eigen::Matrix<realtype, Eigen::Dynamic, Eigen::Dynamic>;

int main ()
{
  const int n_tests = 10;

  // Full Pivot LU
  std::cout << "Full Pivot LU solve time for " << n_tests << " solves: " << std::endl;
  auto fplu_sols = ode_solve_test<realtype, MatrixTypeDenseTests, Eigen::FullPivLU<MatrixTypeDenseTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Partial pivot LU
  std::cout << "Partial Pivot LU solve time for " << n_tests << " solves: " << std::endl;
  auto pplu_sols = ode_solve_test<realtype, MatrixTypeDenseTests, Eigen::PartialPivLU<MatrixTypeDenseTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Householder QR
  std::cout << "Householder QR solve time for " << n_tests << " solves: " << std::endl;
  auto hqr_sols = ode_solve_test<realtype, MatrixTypeDenseTests, Eigen::HouseholderQR<MatrixTypeDenseTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Column pivot Householder QR
  std::cout << "Column pivot Householder QR solve time for " << n_tests << " solves: " << std::endl;
  auto cphqr_sols = ode_solve_test<realtype, MatrixTypeDenseTests, Eigen::ColPivHouseholderQR<MatrixTypeDenseTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Full pivot Householder QR
  std::cout << "Full Pivot Householder QR solve time for " << n_tests << " solves: " << std::endl;
  auto fphqr_sols = ode_solve_test<realtype, MatrixTypeDenseTests, Eigen::FullPivHouseholderQR<MatrixTypeDenseTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Complete orthogonal decomposition
  std::cout << "Complete orthogonal decomposition solve time for " << n_tests << " solves: " << std::endl;
  auto cod_sols = ode_solve_test<realtype, MatrixTypeDenseTests, Eigen::CompleteOrthogonalDecomposition<MatrixTypeDenseTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Sparse LU
  std::cout << "Sparse LU solve time for " << n_tests << " solves: " << std::endl;
  auto slu_sols = ode_solve_test<realtype, MatrixTypeSparseRowMajorTests, Eigen::SparseLU<MatrixTypeSparseRowMajorTests>, DIRECT>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;

  // Sparse QR -- FIXME doesn't work for some reason
  //std::cout << "Sparse QR solve time for " << n_tests << " solves: ";
  //auto sqr_sols = ode_solve_test<realtype, MatrixTypeSparseTests, Eigen::SparseQR<MatrixTypeSparseTests, Eigen::COLAMDOrdering<int>>, DIRECT>(n_tests);

  // BiCGSTAB
  std::cout << "BiCGSTAB solve time for " << n_tests << " solves: " << std::endl;
  auto bicgstab_sols = ode_solve_test<realtype, MatrixTypeSparseRowMajorTests, Eigen::BiCGSTAB<MatrixTypeSparseRowMajorTests, Eigen::IncompleteLUT<realtype>>, ITERATIVE>(n_tests);
  std::cout << "--------------------------------------------------" << std::endl;


  // FullPivLU, ColPivHouseholderQR, FullPivHouseholderQR, CompleteOrthogonalDecomposition are rated by Eigen as the most accurate
  // Compare to each of them to get an idea of accuracy of the other methods

  std::cout << std::endl;
  // Compare to FullPivLU
  {
    auto comparison = fplu_sols;
    std::cout << "Comparisons to Full Pivot LU" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "Full pivot LU: ";
    compare_solution_accuracy(fplu_sols, comparison);

    std::cout << "Partial pivot LU: ";
    compare_solution_accuracy(pplu_sols, comparison);

    std::cout << "Householder QR: ";
    compare_solution_accuracy(hqr_sols, comparison);

    std::cout << "Column Pivot Householder QR: ";
    compare_solution_accuracy(cphqr_sols, comparison);

    std::cout << "Full Pivot Householder QR: ";
    compare_solution_accuracy(fphqr_sols, comparison);

    std::cout << "Complete orthogonal decomposition: ";
    compare_solution_accuracy(cod_sols, comparison);

    std::cout << "Sparse LU: ";
    compare_solution_accuracy(slu_sols, comparison);

    std::cout << "BiCGSTAB: ";
    compare_solution_accuracy(bicgstab_sols, comparison);
  }


  std::cout << std::endl;
  // Compare to ColPivHouseholderQR
  {
    auto comparison = cphqr_sols;
    std::cout << "Comparisons to Col pivot Householder QR" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "Full pivot LU: ";
    compare_solution_accuracy(fplu_sols, comparison);

    std::cout << "Partial pivot LU: ";
    compare_solution_accuracy(pplu_sols, comparison);

    std::cout << "Householder QR: ";
    compare_solution_accuracy(hqr_sols, comparison);

    std::cout << "Column Pivot Householder QR: ";
    compare_solution_accuracy(cphqr_sols, comparison);

    std::cout << "Full Pivot Householder QR: ";
    compare_solution_accuracy(fphqr_sols, comparison);

    std::cout << "Complete orthogonal decomposition: ";
    compare_solution_accuracy(cod_sols, comparison);

    std::cout << "Sparse LU: ";
    compare_solution_accuracy(slu_sols, comparison);

    std::cout << "BiCGSTAB: ";
    compare_solution_accuracy(bicgstab_sols, comparison);
  }


  std::cout << std::endl;
  // Compare to FullPivHouseholderQR
  {
    auto comparison = fphqr_sols;
    std::cout << "Comparisons to Full Pivot Householder QR" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "Full pivot LU: ";
    compare_solution_accuracy(fplu_sols, comparison);

    std::cout << "Partial pivot LU: ";
    compare_solution_accuracy(pplu_sols, comparison);

    std::cout << "Householder QR: ";
    compare_solution_accuracy(hqr_sols, comparison);

    std::cout << "Column Pivot Householder QR: ";
    compare_solution_accuracy(cphqr_sols, comparison);

    std::cout << "Full Pivot Householder QR: ";
    compare_solution_accuracy(fphqr_sols, comparison);

    std::cout << "Complete orthogonal decomposition: ";
    compare_solution_accuracy(cod_sols, comparison);

    std::cout << "Sparse LU: ";
    compare_solution_accuracy(slu_sols, comparison);

    std::cout << "BiCGSTAB: ";
    compare_solution_accuracy(bicgstab_sols, comparison);
  }


  std::cout << std::endl;
  // Compare to CompleteOrthogonalDecomposition
  {
    auto comparison = cod_sols;
    std::cout << "Comparisons to Complete Orthogonal Decomposition" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "Full pivot LU: ";
    compare_solution_accuracy(fplu_sols, comparison);

    std::cout << "Partial pivot LU: ";
    compare_solution_accuracy(pplu_sols, comparison);

    std::cout << "Householder QR: ";
    compare_solution_accuracy(hqr_sols, comparison);

    std::cout << "Column Pivot Householder QR: ";
    compare_solution_accuracy(cphqr_sols, comparison);

    std::cout << "Full Pivot Householder QR: ";
    compare_solution_accuracy(fphqr_sols, comparison);

    std::cout << "Complete orthogonal decomposition: ";
    compare_solution_accuracy(cod_sols, comparison);

    std::cout << "Sparse LU: ";
    compare_solution_accuracy(slu_sols, comparison);

    std::cout << "BiCGSTAB: ";
    compare_solution_accuracy(bicgstab_sols, comparison);
  }

}