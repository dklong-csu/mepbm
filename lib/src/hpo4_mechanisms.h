#ifndef MEPBM_HPO4_MECHANISMS_H
#define MEPBM_HPO4_MECHANISMS_H

#include "chemical_reaction_network.h"
#include <eigen3/Eigen/Dense>
#include <functional>
#include <iostream>
#include <cmath>

namespace MEPBM {
  namespace HPO4 {
    /**
     * FIXME
     */
     enum Mechanism {mech1A, mech1B, mech2A, mech2B};

    /**
     * FIXME
     */
     enum Kernel {Step, Logistic};



     /**
      * FIXME
      */
    double r_func(const unsigned int size)
    {
      return (1.0*size) * 2.677 * std::pow(1.0*size, -0.28);
    }



    /**
     * FIXME
     */
     double logistic(const double L, const double k, const double x0, const double x)
    {
       return L/(1.0 + std::exp(-k*(x-x0)));
    }




    /**
     *  Represents the mechanism
     *      A_2 + 2solv     <->[kf,kb]          A_2(solv) + L
     *      A_2(solv)        ->[k1]             B_2 + L
     *      A_2 + B_i        ->[growth(i)]      B_{i+2} + 2L
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */
    template<typename Real, typename Matrix, MEPBM::HPO4::Kernel K>
    MEPBM::ChemicalReactionNetwork<Real, Matrix>
    create_mech1A(const Eigen::Matrix<Real, 1, Eigen::Dynamic> & sample)
    {
      // Constants specific to HPO4
      const unsigned int max_size = 450;
      const Real S = 11.7;
      const unsigned int growth_amount = 2;
      const unsigned int first_size = 2;
      const unsigned int first_index = 3;
      const unsigned int last_index = max_size + (first_index - first_size);

      // Parameters that will always be in Sample
      const Real kf = sample(0);
      const Real kb = sample(1);
      const Real k1 = sample(2);

      // Form the mechanism
      // Vector indexing is:
      // [A->0, Asolv->1, L->2, particles->3,4,...]
      MEPBM::Species A(0);
      MEPBM::Species Asolv(1);
      MEPBM::Species L(2);
      MEPBM::Particle B(first_index, last_index, first_size);

      // Chemical reactions
      MEPBM::ChemicalReaction<Real, Matrix> nucAf(
          { {A, 1} },
          { {Asolv, 1}, {L, 1} },
          S*S*kf
      );

      MEPBM::ChemicalReaction<Real, Matrix> nucAb(
          { {Asolv, 1}, {L, 1} },
          { {A, 1} },
          kb
      );

      auto B_nuc = B.species(B.index(first_size));
      MEPBM::ChemicalReaction<Real, Matrix> nucB(
          { {Asolv, 1} },
          { {B_nuc, 1}, {L, 1} },
          k1
      );

      std::function<Real(const unsigned int)> growth_fcn;

      switch (K)
      {
        case MEPBM::HPO4::Kernel::Step:
        {
          // For the step function
          // sample = [kf, kb, k1, k2, k3, k4, M]
          growth_fcn = [&](const unsigned int size){
            if (size <= sample(6))
              return sample(3) * r_func(size);
            else
              return sample(4) * r_func(size);
          };
          break;
        }
        case MEPBM::HPO4::Kernel::Logistic:
        {
          // For the logistic function
          // sample = [kf, kb, k1, L1, L2, L3, r1, r2, k4]
          growth_fcn = [&](const unsigned int size){
            // logistic curve centered at 13
            auto logistic13 = sample(3) - logistic(sample(3), sample(6), 13, size);
            // logistic curve centered at 55
            auto logistic55 = sample(4) - logistic(sample(4), sample(7), 55, size);
            // logistic13 + logistic55 + L3 = kernel
            Real kern = logistic13 + logistic55 + sample(5);
            /*std::cout << "size=" << size
                      << "\nM=" << M
                      << "\nr=" << drop_rate
                      << "\nk2=" << k2
                      << "\nk3=" << k3
                      << "\nk=" << k3+logistic << std::endl;*/
            return r_func(size) * kern;
          };
          break;
        }
        default:
        {
          std::cerr << "Invalid growth kernel!" << std::endl;
          break;
        }
      }

      //auto growth_fcn = create_growth_fcn<Step, Real>(sample);
      MEPBM::ParticleGrowth<Real, Matrix> growth(B,
                                                 growth_amount,
                                                 max_size,
                                                 growth_fcn,
                                                 { {A, 1} },
                                                 { {L, 2}}
      );

      std::function<Real(const unsigned int, const unsigned int)> agglom_fcn;
      unsigned int agglom_size;

      switch (K)
      {
        case Step:
        {
          // For the step function
          // sample = [kf, kb, k1, k2, k3, k4, M]
          agglom_fcn = [&](const unsigned int s1, const unsigned int s2)
          {
            if (s1 <= sample(6) && s2 <= sample(6))
              return sample(5) * r_func(s1) * r_func(s2);
            else
              return 0.0;
          };

          agglom_size = sample(6);
          break;
        }
        case Logistic:
        {
          agglom_size = 13;
          // For the logistic function
          // sample = [kf, kb, k1, L1, L2, L3, r1, r2, k4]
          agglom_fcn = [&](const unsigned int s1, const unsigned int s2)
          {
            if (s1 <= agglom_size && s2 <= agglom_size)
              return sample(8) * r_func(s1) * r_func(s2);
            else
              return 0.0;
          };
          break;
        }
        default:
        {
          std::cerr << "Invalid growth kernel!" << std::endl;
          break;
        }
      }

      const auto agglom_index = B.index(agglom_size);
      MEPBM::Particle B_agglom(first_index, agglom_index, first_size);

      //auto agglom_fcn = create_agglom_fcn<Step, Real>(sample);
      MEPBM::ParticleAgglomeration<Real, Matrix> agglom(B_agglom,
                                                        B_agglom,
                                                        max_size,
                                                        agglom_fcn,
                                                        {},
                                                        {});

      MEPBM::ChemicalReactionNetwork<Real, Matrix> network({nucAf, nucAb, nucB},
                                                           {growth},
                                                           {agglom});

      return network;
    }



    /**
     *  Represents the mechanism
     *      A_2 + 2solv     <->[kf,kb]          A_2(solv) + L
     *      A_2(solv)        ->[k1]             B_2 + L
     *      A_2(solv) + B_i  ->[growth(i)]      B_{i+2} + 2L FIXME: confirm the L part of this
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */




    /**
     *  Represents the mechanism
     *      A_2 + 4solv     <->[kf,kb]          2A_1(solv) + 2L
     *      2A_1(solv)       ->[k1]             B_2
     *      A_2 + B_i        ->[growth(i)]      B_{i+2} + 2L FIXME: confirm the L part of this
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */




    /**
     *  Represents the mechanism
     *      A_2 + 4solv     <->[kf,kb]          2A_1(solv) + 2L
     *      2A_1(solv)       ->[k1]             B_2
     *      A_1(solv) + B_i  ->[growth(i)]      B_{i+1} FIXME: confirm the L part of this
     *      B_i + B_j        ->[agglom(i,j)]    B_{i+j}
     *  where growth(i) is a function giving the reaction rate for the growth of a particle of size i
     *  and agglom(i,j) is a function giving the reaction rate for agglomeration between a particle of size i and j
     */



    /**
     * FIXME
     */
    template<typename Real, typename Matrix, MEPBM::HPO4::Kernel K, MEPBM::HPO4::Mechanism M>
    MEPBM::ChemicalReactionNetwork<Real, Matrix>
    create_mechanism(const Eigen::Matrix<Real, 1, Eigen::Dynamic> & sample)
    {
      switch (M) {
        case mech1A : return create_mech1A<Real, Matrix, K>(sample);
      }
    }


  }
}

#endif //MEPBM_HPO4_MECHANISMS_H
