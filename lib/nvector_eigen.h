#ifndef MEPBM_NVECTOR_EIGEN_H
#define MEPBM_NVECTOR_EIGEN_H

#include "sundials/sundials_nvector.h"
#include "sundials/sundials_types.h"
#include "eigen3/Eigen/Sparse"
#include "eigen3/Eigen/Dense"
#include <memory>
#include <cmath>



/**************************************************************************
 * Functions whose memory locations are given to the N_Vector Ops structure
 *************************************************************************/

namespace NVectorOperations
{
  /// Returns the length of the vector
  template <typename VectorType>
  sunindextype
  N_VGetLength( N_Vector v)
  {
    return (static_cast<VectorType*>(v->content))->size();
  }



  /// Function to copy all ops fields from vector y to vector x.
  void
  copy_ops_pointers(N_Vector x, N_Vector y)
  {
    // Pointers are either NULL or point to functions, so there is not a concern about ownership.
    x->ops->nvgetlength         = y->ops->nvgetlength;
    x->ops->nvclone             = y->ops->nvclone;
    x->ops->nvcloneempty        = y->ops->nvcloneempty;
    x->ops->nvdestroy           = y->ops->nvdestroy;
    x->ops->nvspace             = y->ops->nvspace;
    x->ops->nvgetarraypointer   = y->ops->nvgetarraypointer;
    x->ops->nvsetarraypointer   = y->ops->nvsetarraypointer;
    x->ops->nvlinearsum         = y->ops->nvlinearsum;
    x->ops->nvconst             = y->ops->nvconst;
    x->ops->nvprod              = y->ops->nvprod;
    x->ops->nvdiv               = y->ops->nvdiv;
    x->ops->nvscale             = y->ops->nvscale;
    x->ops->nvabs               = y->ops->nvabs;
    x->ops->nvinv               = y->ops->nvinv;
    x->ops->nvaddconst          = y->ops->nvaddconst;
    x->ops->nvmaxnorm           = y->ops->nvmaxnorm;
    x->ops->nvwrmsnorm          = y->ops->nvwrmsnorm;
    x->ops->nvmin               = y->ops->nvmin;
    x->ops->nvminquotient       = y->ops->nvminquotient;
    x->ops->nvconstrmask        = y->ops->nvconstrmask;
    x->ops->nvcompare           = y->ops->nvcompare;
    x->ops->nvinvtest           = y->ops->nvinvtest;
    x->ops->nvlinearcombination = y->ops->nvlinearcombination;
    x->ops->nvscaleaddmulti     = y->ops->nvscaleaddmulti;
    x->ops->nvdotprodmulti      = y->ops->nvdotprodmulti;
    x->ops->nvscalevectorarray  = y->ops->nvscalevectorarray;
  }



  /// Creates a new N_Vector with the same ops field as an existing N_Vector. Allocates storage for the vector.
  template <typename VectorType>
  N_Vector
  N_VClone( N_Vector w)
  {
    N_Vector v = N_VCloneEmpty(w);

    // create memory for the vector in the new N_Vector
    // the corresponding delete is called in N_VDestroy()
    auto cloned = new VectorType(w->ops->nvgetlength(w));
    v->content = cloned;
    return v;
  }



  /// Creates a new N_Vector with the same ops field as an existing N_Vector. Does not allocate storage for the vector.
  N_Vector
  N_VCloneEmpty( N_Vector w)
  {
    N_Vector v = N_VNewEmpty();

    copy_ops_pointers(v, w);
    v->content = nullptr;

    return v;
  }



  /// Destroys the N_Vector and frees allocated memory
  template<typename VectorType>
  void
  N_VDestroy( N_Vector v)
  {
    if (v->content != nullptr)
    {
      auto *content = static_cast<VectorType *>(v->content);
      delete content;
      v->content = nullptr;
    }

    N_VFreeEmpty(v);
    v = nullptr;
  }



  /// Returns storage requirements for one N_Vector. This is a dummy function because it is not of interest.
  void
  N_VSpace(N_Vector v, sunindextype* lrw, sunindextype* liw)
  {
    *lrw = v->ops->nvgetlength(v);
    *liw = 0;
  }



  /// Returns a pointer to an array from the N_Vector.
  template<typename VectorType>
  realtype*
  N_VGetArrayPointer(N_Vector v)
  {
    auto vector = static_cast<VectorType*>(v->content);
    return vector->data();
  }



  /// Overwrites the pointer to the data in an N_Vector.
  template<typename VectorType>
  void
  N_VSetArrayPointer(realtype *v_data, N_Vector v)
  {
    assert(false);
  }



  /// Performs the operation z = ax + by
  template<typename VectorType>
  void
  N_VLinearSum(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto y_vec = static_cast<VectorType*>(y->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = a * (*x_vec) + b * (*y_vec);
  }



  /// Sets all components of z to c.
  template<typename VectorType>
  void
  N_VConst(realtype c, N_Vector z)
  {
    auto z_vec = static_cast<VectorType*>(z->content);
    z_vec->setConstant(z_vec->size(), c);
  }



  /// Sets z to the component-wise product of x and y.
  template<typename VectorType>
  void
  N_VProd(N_Vector x, N_Vector y, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto y_vec = static_cast<VectorType*>(y->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = ( x_vec->array() ) * ( y_vec->array() ) ;
  }



  /// Sets z to the component-wise ratio of x to y. Does not test for zero values.
  template<typename VectorType>
  void
  N_VDiv(N_Vector x, N_Vector y, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto y_vec = static_cast<VectorType*>(y->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = ( x_vec->array() ) / ( y_vec->array() ) ;
  }



  /// Scales x by c and stores the result in z.
  template<typename VectorType>
  void
  N_VScale(realtype c, N_Vector x, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = c * (*x_vec);
  }



  /// Sets the components of z to be the absolute value of the components of x.
  template<typename VectorType>
  void
  N_VAbs(N_Vector x, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = (*x_vec).array().abs();
  }



  /// Sets the components of z to be the inverses of the components of x.
  template<typename VectorType>
  void
  N_VInv(N_Vector x, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = (*x_vec).array().inverse();
  }



  /// Adds the scalar b to all components of x and returns the result in z.
  template<typename VectorType>
  void
  N_VAddConst(N_Vector x, realtype b, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    *z_vec = (*x_vec).array() + b;
  }



  /// Returns the maximum norm of x.
  template<typename VectorType>
  realtype
  N_VMaxNorm(N_Vector x)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    return (*x_vec).array().abs().maxCoeff();
  }



  /// Returns the weighted root-mean-square norm of x.
  template<typename VectorType>
  realtype
  N_VWrmsNorm(N_Vector x, N_Vector w)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto w_vec = static_cast<VectorType*>(w->content);
    auto n = (*x_vec).size();
    auto sum = ( ( (*x_vec).array() * (*w_vec).array() ).pow(2) ).mean();
    return std::sqrt(sum);
  }



  /// Returns the smallest element of x.
  template<typename VectorType>
  realtype
  N_VMin(N_Vector x)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    return (*x_vec).minCoeff();
  }



  /// Returns the minimum of the quotients obtained by the component-wise ratio. A zero element in denom is skipped
  /// and if no quotients are found then the value BIG_REAL is returned.
  template<typename VectorType>
  realtype
  N_VMinQuotient(N_Vector num, N_Vector denom)
  {
    auto num_vec = static_cast<VectorType*>(num->content);
    auto denom_vec = static_cast<VectorType*>(denom->content);

    realtype result = BIG_REAL;
    for (unsigned int i = 0; i<num_vec->size(); ++i)
    {
      if ( (*denom_vec)(i) != 0 )
      {
        auto ratio = (*num_vec)(i) / (*denom_vec)(i);
        if (ratio < result)
        {
          result = ratio;
        }
      }
    }

    return result;
  }



  /// Performs the test: xi > 0 if ci = 2, xi >= 0 if ci = 1, xi <= 0 if ci = -1, xi < 0 if ci = -2
  template<typename VectorType>
  booleantype
  N_VConstrMask(N_Vector c, N_Vector x, N_Vector m)
  {
    auto c_vec = static_cast<VectorType*>(c->content);
    auto x_vec = static_cast<VectorType*>(x->content);
    auto m_vec = static_cast<VectorType*>(m->content);

    booleantype result = SUNTRUE;

    for (unsigned int i=0; i<x->ops->nvgetlength(x); ++i)
    {
      auto constraint = (int)(*c_vec)(i);
      auto val = (*x_vec)(i);
      switch(constraint)
      {
        case 2:
          if (val > 0)
          {
            (*m_vec)(i) = 0;
          }
          else
          {
            (*m_vec)(i) = 1;
            result = SUNFALSE;
          }
          break;
        case 1:
          if (val >= 0)
          {
            (*m_vec)(i) = 0;
          }
          else
          {
            (*m_vec)(i) = 1;
            result = SUNFALSE;
          }
          break;
        case -1:
          if (val <= 0 )
          {
            (*m_vec)(i) = 0;
          }
          else
          {
            (*m_vec)(i) = 1;
            result = SUNFALSE;
          }
          break;
        case -2:
          if (val < 0)
          {
            (*m_vec)(i) = 0;
          }
          else
          {
            (*m_vec)(i) = 1;
            result = SUNFALSE;
          }
          break;
        default:
          (*m_vec)(i) = 0;
      }
    }

    return result;
  }



  /// Compares the components of x to the scalar c and returns a vector z such that zi = 1 if abs(x_i) >= c and 0 otherwise.
  template<typename VectorType>
  void
  N_VCompare(realtype c, N_Vector x, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    for (unsigned int i=0; i<x->ops->nvgetlength(x); ++i)
    {
      auto val = std::abs((*x_vec)(i));
      (val > c) ? (*z_vec)(i) = 1. : (*z_vec)(i) = 0.;
    }
  }



  /// Sets the components of z to be the inverses of the components of x, with testing for zero values.
  template<typename VectorType>
  booleantype
  N_VInvTest(N_Vector x, N_Vector z)
  {
    auto x_vec = static_cast<VectorType*>(x->content);
    auto z_vec = static_cast<VectorType*>(z->content);

    booleantype result = SUNTRUE;

    for (unsigned int i=0; i<x->ops->nvgetlength(x); ++i)
    {
      auto val = (*x_vec)(i);
      if (val != 0)
        (*z_vec)(i) = 1./val;
      else
        result = SUNFALSE;
    }
    return result;
  }



  /// Computes the linear combination of nv vectors X.
  template<typename VectorType>
  int
  N_VLinearCombination(int nv, realtype* c, N_Vector* X, N_Vector z)
  {
    // invalid number of vectors
    if (nv < 1) return -1;

    // should have called N_VScale in this case
    if (nv == 1)
    {
      N_VScale<VectorType>(c[0], X[0], z);
      return 0;
    }

    // should have called N_VLinearSum
    if (nv == 2)
    {
      N_VLinearSum<VectorType>(c[0], X[0], c[1], X[1], z);
      return 0;
    }

    // when nv > 2, start with linear sum and then keep adding to z
    N_VLinearSum<VectorType>(c[0], X[0], c[1], X[1], z);
    for (unsigned int i=2; i<nv; ++i)
      N_VLinearSum<VectorType>(1, z, c[i], X[i], z);

    return 0;
  }



  /// Scales and adds one vector to nv vectors.
  template<typename VectorType>
  int
  N_VScaleAddMulti(int nv, realtype* c, N_Vector x, N_Vector* Y, N_Vector* Z)
  {
    // invalid number of vectors
    if (nv < 1) return -1;

    for (unsigned int i=0; i<nv; ++i)
    {
      N_VLinearSum<VectorType>(c[i], x, 1, Y[i], Z[i]);
    }

    return 0;
  }



  /// Computes the dot product of a vector with n_v other vectors.
  template<typename VectorType>
  int
  N_VDotProdMulti(int nv, N_Vector x, N_Vector* Y, realtype* d)
  {
    // invalid number of vectors
    if (nv < 1) return -1;

    auto x_vec = static_cast<VectorType*>(x->content);

    for (unsigned int i=0; i<nv; ++i)
    {
      auto y_vec = static_cast<VectorType*>(Y[i]->content);
      d[i] = x_vec->dot(*y_vec);
    }
    return 0;
  }



  /// Scales each vector by a potentially different constant.
  template<typename VectorType>
  int
  N_VScaleVectorArray(int nv, realtype* c, N_Vector* X, N_Vector* Z)
  {
    // invalid number of vectors
    if (nv < 1) return -1;

    for (unsigned int i=0; i<nv; ++i)
    {
      N_VScale<VectorType>(c[i], X[i], Z[i]);
    }
    return 0;
  }

}



/// Function to set all ops fields to the correct function pointer
template<typename VectorType>
void
set_ops_pointers(N_Vector v)
{
  v->ops->nvgetlength         = NVectorOperations::N_VGetLength<VectorType>;
  v->ops->nvclone             = NVectorOperations::N_VClone<VectorType>;
  v->ops->nvcloneempty        = NVectorOperations::N_VCloneEmpty;
  v->ops->nvdestroy           = NVectorOperations::N_VDestroy<VectorType>;
  v->ops->nvspace             = NVectorOperations::N_VSpace;
  v->ops->nvgetarraypointer   = NVectorOperations::N_VGetArrayPointer<VectorType>;
  v->ops->nvsetarraypointer   = NVectorOperations::N_VSetArrayPointer<VectorType>;
  v->ops->nvlinearsum         = NVectorOperations::N_VLinearSum<VectorType>;
  v->ops->nvconst             = NVectorOperations::N_VConst<VectorType>;
  v->ops->nvprod              = NVectorOperations::N_VProd<VectorType>;
  v->ops->nvdiv               = NVectorOperations::N_VDiv<VectorType>;
  v->ops->nvscale             = NVectorOperations::N_VScale<VectorType>;
  v->ops->nvabs               = NVectorOperations::N_VAbs<VectorType>;
  v->ops->nvinv               = NVectorOperations::N_VInv<VectorType>;
  v->ops->nvaddconst          = NVectorOperations::N_VAddConst<VectorType>;
  v->ops->nvmaxnorm           = NVectorOperations::N_VMaxNorm<VectorType>;
  v->ops->nvwrmsnorm          = NVectorOperations::N_VWrmsNorm<VectorType>;
  v->ops->nvmin               = NVectorOperations::N_VMin<VectorType>;
  v->ops->nvminquotient       = NVectorOperations::N_VMinQuotient<VectorType>;
  v->ops->nvconstrmask        = NVectorOperations::N_VConstrMask<VectorType>;
  v->ops->nvcompare           = NVectorOperations::N_VCompare<VectorType>;
  v->ops->nvinvtest           = NVectorOperations::N_VInvTest<VectorType>;
  v->ops->nvlinearcombination = NVectorOperations::N_VLinearCombination<VectorType>;
  v->ops->nvscaleaddmulti     = NVectorOperations::N_VScaleAddMulti<VectorType>;
  v->ops->nvdotprodmulti      = NVectorOperations::N_VDotProdMulti<VectorType>;
  v->ops->nvscalevectorarray  = NVectorOperations::N_VScaleVectorArray<VectorType>;
}



/// Function to create an N_Vector without allocating memory for the vector.
template<typename VectorType>
N_Vector
create_empty_eigen_nvector()
{
  N_Vector v = N_VNewEmpty();

  set_ops_pointers<VectorType>(v);

  return v;
}



/// Function to create an N_Vector using Eigen for the backend linear algebra
template<typename VectorType>
N_Vector
create_eigen_nvector(unsigned int dim)
{
  N_Vector v = create_empty_eigen_nvector<VectorType>();
  VectorType* vec = new VectorType(dim);
  v->content = (void*)vec;

  return v;

}


#endif //MEPBM_NVECTOR_EIGEN_H
