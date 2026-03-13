#pragma once

/// Shared utility functions for solver-based linear operators (CholSolverLinOp, LUSolverLinOp).
///
/// These free functions handle common tasks: reading Matrix Market headers, applying
/// row permutations via LAPACK, copying op(B) into work buffers, and accumulating
/// alpha * W + beta * C results.

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <string>
#include <fstream>
#include <sstream>
#include <vector>

namespace RandLAPACK_extras {
namespace solver_util {

using blas::Layout;
using blas::Op;

/// Read a single dimension (rows or cols) from a Matrix Market file header.
/// @param dim_index  0 for rows, 1 for cols.
inline int64_t read_matrix_dimension(const std::string& filename, int dim_index) {
    std::ifstream file(filename);
    randblas_require(file.is_open());

    // Skip comment lines (start with '%')
    std::string line;
    do {
        std::getline(file, line);
    } while (line[0] == '%');

    // First non-comment line contains: rows cols nnz
    std::istringstream iss(line);
    int64_t rows, cols, nnz;
    iss >> rows >> cols >> nnz;
    file.close();

    return (dim_index == 0) ? rows : cols;
}

/// Apply a row permutation to an m x ncols matrix X, in-place.
/// After this call, row i of X contains what was previously row perm[i].
///
/// Uses a gather-based implementation: copies permuted data into a temporary
/// buffer column-by-column (ColMajor) or row-by-row (RowMajor), then writes
/// back. This avoids LAPACK's lapmr/lapmt which are not available on all
/// platforms (e.g., macOS Accelerate).
///
/// @param layout  Storage layout of X.
/// @param m       Number of rows of X (= length of perm).
/// @param ncols   Number of columns of X.
/// @param perm    0-based permutation vector: result row i = source row perm[i].
/// @param X       Matrix to permute, modified in-place.
/// @param ldx     Leading dimension of X.
template <typename T>
void apply_row_perm(Layout layout, int64_t m, int64_t ncols,
                    const int* perm, T* X, int64_t ldx) {
    if (layout == Layout::ColMajor) {
        // Process column-by-column. Each column is contiguous with stride 1.
        // Temporary: one column (m elements).
        std::vector<T> temp(m);
        for (int64_t j = 0; j < ncols; ++j) {
            T* col = X + j * ldx;
            for (int64_t i = 0; i < m; ++i)
                temp[i] = col[perm[i]];
            for (int64_t i = 0; i < m; ++i)
                col[i] = temp[i];
        }
    } else {
        // RowMajor: row i occupies X[i*ldx .. i*ldx + ncols - 1].
        // Gather rows according to perm into temp, then copy back.
        std::vector<T> temp(m * ncols);
        for (int64_t i = 0; i < m; ++i)
            blas::copy(ncols, X + (int64_t)perm[i] * ldx, 1,
                       temp.data() + i * ncols, 1);
        for (int64_t i = 0; i < m; ++i)
            blas::copy(ncols, temp.data() + i * ncols, 1,
                       X + i * ldx, 1);
    }
}

/// Copy op(B) into a contiguous work buffer W.
///
/// Both B (or its transpose) and W use the same layout. After the copy,
/// W contains op(B) with dimensions rows_op x cols_op and leading dimension ldw.
///
/// This is needed because RandBLAS sparse TRSM overwrites its B argument in-place,
/// and the caller's B is const.
///
/// For NoTrans, lapack::lacpy handles the copy in one call (supports both layouts
/// and different leading dimensions). For Trans, we fall back to strided blas::copy
/// loops since lacpy does not support transpose.
///
/// @param layout   Storage layout for both B and W.
/// @param trans_B  Whether to apply B or B^T.
/// @param rows_op  Number of rows of op(B).
/// @param cols_op  Number of columns of op(B).
/// @param B        Source matrix (const, not modified).
/// @param ldb      Leading dimension of B.
/// @param W        Destination work buffer (must be pre-allocated).
/// @param ldw      Leading dimension of W.
template <typename T>
void copy_op_B(Layout layout, Op trans_B, int64_t rows_op, int64_t cols_op,
               const T* B, int64_t ldb, T* W, int64_t ldw) {
    if (trans_B == Op::NoTrans) {
        // No transpose: B and op(B) have the same shape (rows_op x cols_op).
        // lacpy copies an m x n submatrix from one leading dimension to another.
        // It is always ColMajor, so for RowMajor we swap dimensions: a RowMajor
        // rows_op x cols_op matrix is the same memory as ColMajor cols_op x rows_op.
        if (layout == Layout::ColMajor) {
            lapack::lacpy(lapack::MatrixType::General, rows_op, cols_op, B, ldb, W, ldw);
        } else {
            lapack::lacpy(lapack::MatrixType::General, cols_op, rows_op, B, ldb, W, ldw);
        }
    } else {
        // Transpose: B is cols_op x rows_op before transpose, op(B) = B^T is rows_op x cols_op.
        // lacpy cannot transpose, so we copy column-by-column (ColMajor) or row-by-row (RowMajor).
        // Each iteration reads a strided slice of B and writes a contiguous slice of W.
        if (layout == Layout::ColMajor) {
            // Column j of op(B) = row j of B.
            // Row j of B (ColMajor cols_op x rows_op): starts at B + j, stride ldb.
            // Column j of W: contiguous at W + j*ldw.
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t j = 0; j < cols_op; ++j)
                blas::copy(rows_op, B + j, ldb, W + j * ldw, 1);
        } else {
            // Row i of op(B) = column i of B.
            // Column i of B (RowMajor cols_op x rows_op): starts at B + i, stride ldb.
            // Row i of W: contiguous at W + i*ldw.
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < rows_op; ++i)
                blas::copy(cols_op, B + i, ldb, W + i * ldw, 1);
        }
    }
}

/// Accumulate: C := beta * C + alpha * W.
///
/// Both C and W are m x n in the given layout. Operates column-by-column (ColMajor)
/// or row-by-row (RowMajor) to correctly handle non-contiguous C (when ldc > m or ldc > n).
///
/// @param layout  Storage layout for both C and W.
/// @param m       Number of rows.
/// @param n       Number of columns.
/// @param alpha   Scalar multiplier for W.
/// @param W       Source matrix (m x n).
/// @param ldw     Leading dimension of W.
/// @param beta    Scalar multiplier for C (applied before adding alpha * W).
/// @param C       Destination matrix (m x n), modified in-place.
/// @param ldc     Leading dimension of C.
template <typename T>
void accumulate(Layout layout, int64_t m, int64_t n, T alpha,
                const T* W, int64_t ldw, T beta, T* C, int64_t ldc) {
    if (layout == Layout::ColMajor) {
        // Each column is contiguous: scale then add.
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < n; ++j) {
            blas::scal(m, beta, C + j * ldc, 1);
            blas::axpy(m, alpha, W + j * ldw, 1, C + j * ldc, 1);
        }
    } else {
        // Each row is contiguous: scale then add.
#if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < m; ++i) {
            blas::scal(n, beta, C + i * ldc, 1);
            blas::axpy(n, alpha, W + i * ldw, 1, C + i * ldc, 1);
        }
    }
}

} // namespace solver_util
} // namespace RandLAPACK_extras
