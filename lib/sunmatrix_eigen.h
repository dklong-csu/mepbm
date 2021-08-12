#ifndef MEPBM_SUNMATRIX_EIGEN_H
#define MEPBM_SUNMATRIX_EIGEN_H

#include "sundials/sundials_matrix.h"
#include "eigen3/Eigen/Sparse"



namespace SUNMatrixOperations
{
  /// Returns the type identifier for the matrix.
  SUNMatrix_ID
  SUNMatGetID(SUNMatrix A)
  {
    return SUNMATRIX_CUSTOM;
  }



  /// Function to copy all ops fields from matrix A to matrix B
  void
  copy_ops_pointers(SUNMatrix A, SUNMatrix B)
  {
    B->ops->getid = A->ops->getid;
    B->ops->clone = A->ops->clone;
    B->ops->destroy = A->ops->destroy;
    B->ops->zero = A->ops->zero;
    B->ops->copy = A->ops->copy;
    B->ops->scaleaddi = A->ops->scaleaddi;
  }



  /// Creates a new SUNMatrix of the same type as an existing matrix and sets the ops field. Allocates storage for the new matrix.
  template<typename MatrixType>
  SUNMatrix
  SUNMatClone(SUNMatrix A)
  {
    SUNMatrix B = SUNMatNewEmpty();
    copy_ops_pointers(A, B);

    auto A_ptr = static_cast<MatrixType*>(A->content);
    auto n_cols = A_ptr->cols();
    auto n_rows = A_ptr->rows();
    auto cloned = new MatrixType(n_rows,n_cols);
    B->content = cloned;
    return B;
  }



  /// Destroys the matrix and frees memory allocated for its internal data.
  template<typename MatrixType>
  void
  SUNMatDestroy(SUNMatrix A)
  {
    if (A->content != nullptr)
    {
      auto content = static_cast<MatrixType *>(A->content);
      delete content;
    }

    SUNMatFreeEmpty(A);
    A->content = nullptr;
    A->ops = nullptr;
  }



  /// Performs the operation A_{ij} = 0 for all entries of A.
  template<typename MatrixType>
  int
  SUNMatZero(SUNMatrix A)
  {
    auto A_ptr = static_cast<MatrixType*>(A->content);
    A_ptr->setZero();
    return SUNMAT_SUCCESS;
  }



  /// Performs the operation $B_{ij} = A_{ij} for all entries of A, B.
  template<typename MatrixType>
  int
  SUNMatCopy(SUNMatrix A, SUNMatrix B)
  {
    auto A_ptr = static_cast<MatrixType*>(A->content);
    if (B->content == nullptr)
    {
      auto rows = A_ptr->rows();
      auto cols = A_ptr->cols();
      MatrixType* mat = new MatrixType(rows, cols);
      B->content = (void*)mat;
    }
    auto B_ptr = static_cast<MatrixType*>(B->content);

    *B_ptr = *A_ptr;

    return SUNMAT_SUCCESS;
  }



  /// Performs the operation A = cA + I
  template<typename MatrixType>
  int
  SUNMatScaleAddI(realtype c, SUNMatrix A)
  {
    auto A_ptr = static_cast<MatrixType*>(A->content);

    *A_ptr = c * (*A_ptr);
    auto n = A_ptr->cols();
    for (unsigned int i=0; i<n; ++i)
    {
      A_ptr->coeffRef(i,i) += 1.;
    }

    return SUNMAT_SUCCESS;
  }



  /// Function to set the ops pointers
  template<typename MatrixType>
  void
  set_ops_pointers(SUNMatrix A)
  {
    A->ops->getid     = SUNMatrixOperations::SUNMatGetID;
    A->ops->clone     = SUNMatrixOperations::SUNMatClone<MatrixType>;
    A->ops->destroy   = SUNMatrixOperations::SUNMatDestroy<MatrixType>;
    A->ops->zero      = SUNMatrixOperations::SUNMatZero<MatrixType>;
    A->ops->copy      = SUNMatrixOperations::SUNMatCopy<MatrixType>;
    A->ops->scaleaddi = SUNMatrixOperations::SUNMatScaleAddI<MatrixType>;
  }
}



  /// Function to create a SUNMatrix without allocating memory for the matrix
  template<typename MatrixType>
  SUNMatrix
  create_empty_eigen_sunmatrix()
  {
    SUNMatrix A = SUNMatNewEmpty();

    SUNMatrixOperations::set_ops_pointers<MatrixType>(A);

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
    A->content = (void*)mat;

    return A;
  }


#endif //MEPBM_SUNMATRIX_EIGEN_H