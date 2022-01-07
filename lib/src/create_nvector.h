#ifndef MEPBM_CREATE_NVECTOR_H
#define MEPBM_CREATE_NVECTOR_H

#include "sundials/sundials_nvector.h"
#include "sundials/sundials_types.h"
#include "nvector_operations.h"



namespace MEPBM {
  /// Function to set all ops fields to the correct function pointer
  template<typename VectorType>
  void
  set_ops_pointers(N_Vector v)
  {
    v->ops->nvgetlength         = MEPBM::N_VGetLength<VectorType>;
    v->ops->nvclone             = MEPBM::N_VClone<VectorType>;
    v->ops->nvcloneempty        = MEPBM::N_VCloneEmpty;
    v->ops->nvdestroy           = MEPBM::N_VDestroy<VectorType>;
    v->ops->nvspace             = MEPBM::N_VSpace;
    v->ops->nvgetarraypointer   = MEPBM::N_VGetArrayPointer<VectorType>;
    v->ops->nvsetarraypointer   = MEPBM::N_VSetArrayPointer<VectorType>;
    v->ops->nvlinearsum         = MEPBM::N_VLinearSum<VectorType>;
    v->ops->nvconst             = MEPBM::N_VConst<VectorType>;
    v->ops->nvprod              = MEPBM::N_VProd<VectorType>;
    v->ops->nvdiv               = MEPBM::N_VDiv<VectorType>;
    v->ops->nvscale             = MEPBM::N_VScale<VectorType>;
    v->ops->nvabs               = MEPBM::N_VAbs<VectorType>;
    v->ops->nvinv               = MEPBM::N_VInv<VectorType>;
    v->ops->nvaddconst          = MEPBM::N_VAddConst<VectorType>;
    v->ops->nvmaxnorm           = MEPBM::N_VMaxNorm<VectorType>;
    v->ops->nvwrmsnorm          = MEPBM::N_VWrmsNorm<VectorType>;
    v->ops->nvmin               = MEPBM::N_VMin<VectorType>;
    v->ops->nvminquotient       = MEPBM::N_VMinQuotient<VectorType>;
    v->ops->nvconstrmask        = MEPBM::N_VConstrMask<VectorType>;
    v->ops->nvcompare           = MEPBM::N_VCompare<VectorType>;
    v->ops->nvinvtest           = MEPBM::N_VInvTest<VectorType>;
    v->ops->nvlinearcombination = MEPBM::N_VLinearCombination<VectorType>;
    v->ops->nvscaleaddmulti     = MEPBM::N_VScaleAddMulti<VectorType>;
    v->ops->nvdotprodmulti      = MEPBM::N_VDotProdMulti<VectorType>;
    v->ops->nvscalevectorarray  = MEPBM::N_VScaleVectorArray<VectorType>;
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
}

#endif //MEPBM_CREATE_NVECTOR_H
