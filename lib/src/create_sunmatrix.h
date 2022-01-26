#ifndef MEPBM_CREATE_SUNMATRIX_H
#define MEPBM_CREATE_SUNMATRIX_H

#include "sundials/sundials_matrix.h"
#include "sunmatrix_operations.h"

namespace MEPBM {
  /// Function to set the ops pointers
  template<typename MatrixType>
  void
  set_ops_pointers(SUNMatrix A)
  {
    A->ops->getid     = MEPBM::SUNMatGetID;
    A->ops->clone     = MEPBM::SUNMatClone<MatrixType>;
    A->ops->destroy   = MEPBM::SUNMatDestroy<MatrixType>;
    A->ops->zero      = MEPBM::SUNMatZero<MatrixType>;
    A->ops->copy      = MEPBM::SUNMatCopy<MatrixType>;
    A->ops->scaleaddi = MEPBM::SUNMatScaleAddI<MatrixType>;
  }



  /// Function to create a SUNMatrix without allocating memory for the matrix
  template<typename MatrixType>
  SUNMatrix
  create_empty_eigen_sunmatrix()
  {
    SUNMatrix A = SUNMatNewEmpty();

    MEPBM::set_ops_pointers<MatrixType>(A);

    A->content = nullptr;

    return A;
  }



  /// Function to create a SUNMatrix and allocate memory for the matrix
  template<typename MatrixType>
  SUNMatrix
  create_eigen_sunmatrix(sunindextype rows, sunindextype cols)
  {
    SUNMatrix A = create_empty_eigen_sunmatrix<MatrixType>();
    MatrixType* mat = new MatrixType(rows, cols);
    mat->setZero();
    A->content = (void*)mat;

    return A;
  }
}

#endif //MEPBM_CREATE_SUNMATRIX_H
