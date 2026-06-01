#pragma once

// RandLAPACK Extras - Sparse axpy
//
// Helper: C := alpha * A + beta * B for two RandBLAS::CSRMatrix inputs whose
// nonzero patterns (rowptr + colidxs) are bit-identical.
//
// Use case: forming X = K - omega*M in the reduced spectral application, where
// K and M come from a single FEM mesh and therefore share sparsity exactly.
//
// This is the fast O(nnz) value-only path.  A general-purpose `sparse_axpby`
// supporting different sparsity patterns belongs upstream in RandBLAS (or
// punts to MKL `mkl_sparse_d_add` / cuSPARSE `cusparseDcsrgeam2`); we don't
// need it for the current use case and would just be replicating those libs.

#include <RandBLAS.hh>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace RandLAPACK_extras {

/// Compute C := alpha * A + beta * B with A, B sharing sparsity pattern.
///
/// Requirements:
///   - A.n_rows == B.n_rows, A.n_cols == B.n_cols, A.nnz == B.nnz
///   - A.index_base == B.index_base
///   - A.rowptr[i] == B.rowptr[i] for all i in [0, n_rows]
///   - A.colidxs[i] == B.colidxs[i] for all i in [0, nnz)
///
/// Returns a newly-owned CSRMatrix C whose sparsity pattern is copied from A.
template <typename T, typename sint_t>
RandBLAS::sparse_data::CSRMatrix<T, sint_t> sparse_axpby_shared_pattern(
    T alpha,
    const RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    T beta,
    const RandBLAS::sparse_data::CSRMatrix<T, sint_t>& B)
{
    randblas_require(A.n_rows == B.n_rows);
    randblas_require(A.n_cols == B.n_cols);
    randblas_require(A.nnz    == B.nnz);
    randblas_require(A.index_base == B.index_base);

    // Verify the rowptr arrays match.  This catches user error early when
    // someone forgets the "shared sparsity" precondition.
    for (int64_t i = 0; i <= A.n_rows; ++i)
        randblas_require(A.rowptr[i] == B.rowptr[i]);
    // colidxs comparison: O(nnz) but cheap relative to any downstream sparse op.
    for (int64_t i = 0; i < A.nnz; ++i)
        randblas_require(A.colidxs[i] == B.colidxs[i]);

    RandBLAS::sparse_data::CSRMatrix<T, sint_t> C(A.n_rows, A.n_cols);
    C.reserve(A.nnz);
    C.index_base = A.index_base;

    std::memcpy(C.rowptr,  A.rowptr,  sizeof(sint_t) * (A.n_rows + 1));
    std::memcpy(C.colidxs, A.colidxs, sizeof(sint_t) * A.nnz);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < A.nnz; ++i)
        C.vals[i] = alpha * A.vals[i] + beta * B.vals[i];

    return C;
}

} // namespace RandLAPACK_extras
