#pragma once

// Public API: materialize — write the dense representation of a linear operator
// into a caller-provided buffer.
//
// Generic fallback: multiply by the identity matrix.
// Overloaded specializations avoid that cost when the underlying storage is
// directly accessible (DenseLinOp, SparseLinOp).

#include "rl_concepts.hh"
#include "rl_dense_linop.hh"
#include "rl_sparse_linop.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <cstdint>


namespace RandLAPACK {

/// Materialize a linear operator into a dense column-major buffer.
///
/// Generic fallback: forms buf = A * I  by applying the operator to the
/// identity matrix.  Works for any type satisfying the LinearOperator concept.
///
/// @param[in]  A    Linear operator (m-by-n).
/// @param[in]  m    Number of rows.
/// @param[in]  n    Number of columns.
/// @param[out] buf  Pre-allocated buffer of length at least m * n.
/// @param[in]  ldb  Leading dimension of buf (>= m).
template <typename LinOp>
void materialize(LinOp& A, int64_t m, int64_t n, typename LinOp::scalar_t* buf, int64_t ldb) {
    using T = typename LinOp::scalar_t;
    randblas_require(ldb >= m);
    // Zero the output buffer.
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            buf[i + j * ldb] = (T)0.0;
    // Build identity matrix.
    T* Eye = new T[n * n]();
    RandLAPACK::util::eye(n, n, Eye);
    // buf = 1.0 * A * I + 0.0 * buf
    A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      m, n, n, (T)1.0, Eye, n, (T)0.0, buf, ldb);
    delete[] Eye;
}

/// Specialization for DenseLinOp: copy directly via lacpy (ColMajor) or
/// fall back to the generic path (RowMajor).
template <typename T>
void materialize(linops::DenseLinOp<T>& A, int64_t m, int64_t n, T* buf, int64_t ldb) {
    randblas_require(m == A.n_rows);
    randblas_require(n == A.n_cols);
    randblas_require(ldb >= m);
    if (A.buff_layout == blas::Layout::ColMajor) {
        lapack::lacpy(lapack::MatrixType::General, m, n, A.A_buff, A.lda, buf, ldb);
    } else {
        // RowMajor: transpose-copy element by element.
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                buf[i + j * ldb] = A.A_buff[i * A.lda + j];
    }
}

/// Specialization for SparseLinOp: zero-init + sparse_to_dense.
template <RandBLAS::sparse_data::SparseMatrix SpMat>
void materialize(linops::SparseLinOp<SpMat>& A, int64_t m, int64_t n,
                 typename SpMat::scalar_t* buf, int64_t ldb) {
    using T = typename SpMat::scalar_t;
    randblas_require(m == A.n_rows);
    randblas_require(n == A.n_cols);
    randblas_require(ldb >= m);
    // Zero the output buffer.
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            buf[i + j * ldb] = (T)0.0;
    if (ldb == m) {
        // Contiguous: sparse_to_dense can write directly.
        RandLAPACK::util::sparse_to_dense(A.A_sp, blas::Layout::ColMajor, buf);
    } else {
        // Non-contiguous leading dimension: materialize into temp, then copy.
        T* tmp = new T[m * n]();
        RandLAPACK::util::sparse_to_dense(A.A_sp, blas::Layout::ColMajor, tmp);
        lapack::lacpy(lapack::MatrixType::General, m, n, tmp, m, buf, ldb);
        delete[] tmp;
    }
}

} // end namespace RandLAPACK
