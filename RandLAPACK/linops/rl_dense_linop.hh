#pragma once

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                     DenseLinOp                        */
/*                                                       */
/*********************************************************/

/// @brief Dense linear operator for matrix multiplication operations
///
/// @details Provides a linear operator interface for dense matrices, supporting
/// multiplication with both dense and sparse matrices. The operator supports:
/// - Left and right multiplication modes (Side::Left and Side::Right)
/// - Both ColMajor and RowMajor memory layouts
/// - Transpose operations (Op::NoTrans, Op::Trans)
/// - Dense-dense and dense-sparse multiplications
///
/// The operator uses optimized BLAS routines (gemm) for dense operations and
/// RandBLAS sparse BLAS routines (right_spmm, left_spmm) for sparse operations.
///
/// @tparam T Scalar type (e.g., float, double)
template <typename T>
struct DenseLinOp {
    using scalar_t = T;
    const int64_t n_rows;      ///< Number of rows in the operator matrix
    const int64_t n_cols;      ///< Number of columns in the operator matrix
    const T* A_buff;           ///< Pointer to the dense matrix data
    const int64_t lda;         ///< Leading dimension of A (layout-dependent)
    const Layout buff_layout;  ///< Memory layout of A (ColMajor or RowMajor)

    /// @brief Construct a dense linear operator
    /// @param n_rows Number of rows in the dense matrix
    /// @param n_cols Number of columns in the dense matrix
    /// @param A_buff Pointer to dense matrix data
    /// @param lda Leading dimension of A (ColMajor: lda >= n_rows, RowMajor: lda >= n_cols)
    /// @param buff_layout Memory layout of A (ColMajor or RowMajor)
    DenseLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        const T* A_buff,
        int64_t lda,
        Layout buff_layout
    ) : n_rows(n_rows), n_cols(n_cols), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {
        // Validate leading dimension based on layout
        if (buff_layout == Layout::ColMajor) {
            randblas_require(lda >= n_rows);
        } else {  // RowMajor
            randblas_require(lda >= n_cols);
        }
    }

    /// @brief Compute the Frobenius norm of the dense operator matrix
    /// @return Frobenius norm
    T fro_nrm(
    ) {
        return lapack::lange(Norm::Fro, n_rows, n_cols, A_buff, lda);
    }

    /// @brief Dense-dense matrix multiplication: C := alpha * op(A) * op(B) + beta * C
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        const T* B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        randblas_require(layout == buff_layout);
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
        randblas_require(rows_A == n_rows);
        randblas_require(cols_A == n_cols);

        // Layout-aware dimension checks
        if (layout == Layout::ColMajor) {
            randblas_require(ldb >= rows_B);
            randblas_require(ldc >= m);
        } else {  // RowMajor
            randblas_require(ldb >= cols_B);
            randblas_require(ldc >= n);
        }

        blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A_buff, lda, B, ldb, beta, C, ldc);
    }

    /// @brief Dense-dense multiplication with explicit side specification
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        const T* B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if (side == Side::Left) {
            // Left multiplication: delegate to non-sided dense operator
            (*this)(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        } else {  // Side::Right
            // Right multiplication: C := alpha * op(B) * op(A) + beta * C
            randblas_require(layout == buff_layout);
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_A == n_rows);
            randblas_require(cols_A == n_cols);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            // Compute C := alpha * op(B) * op(A) + beta * C by swapping operand order in GEMM
            blas::gemm(layout, trans_B, trans_A, m, n, k, alpha, B, ldb, A_buff, lda, beta, C, ldc);
        }
    }

    /// @brief Dense-sparse matrix multiplication: C := alpha * op(A) * op(B_sp) + beta * C
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SpMatB &B_sp,
        T beta,
        T* C,
        int64_t ldc
    ) {
        // Validate layout and input dimensions
        randblas_require(layout == buff_layout);
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
        randblas_require(rows_A == n_rows);
        randblas_require(cols_A == n_cols);

        // Layout-aware dimension checks
        if (layout == Layout::ColMajor) {
            randblas_require(ldc >= m);
        } else {  // RowMajor
            randblas_require(ldc >= n);
        }

        // Use RandBLAS right_spmm: C = alpha * op(A) * op(B_sp) + beta * C
        RandBLAS::sparse_data::right_spmm(layout, trans_A, trans_B, m, n, k, alpha, A_buff, lda, B_sp, 0, 0, beta, C, ldc);
    }

    /// @brief Dense-sparse multiplication with explicit side specification
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SpMatB &B_sp,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if (side == Side::Left) {
            // Left multiplication: delegate to non-sided sparse operator
            (*this)(layout, trans_A, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);
        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B_sp) * op(A) + beta * C
            // Use RandBLAS left_spmm: sparse × dense

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_A == n_rows);
            randblas_require(cols_A == n_cols);
            randblas_require(layout == buff_layout);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldc >= n);
            }

            // left_spmm computes: C = alpha * op(B_sp) * op(A_buff) + beta * C
            RandBLAS::sparse_data::left_spmm(layout, trans_B, trans_A, m, n, k, alpha, B_sp, 0, 0, A_buff, lda, beta, C, ldc);
        }
    }

    /// Sketching operator multiplication without explicit Side parameter.
    /// Computes C = alpha * op(S) * op(A) + beta * C (equivalent to Side::Right).
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_S,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        (*this)(Side::Right, layout, trans_A, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    /// Sketching operator multiplication with dense linear operator.
    ///   Side::Left:  C = alpha * op(A) * op(S) + beta * C
    ///   Side::Right: C = alpha * op(S) * op(A) + beta * C
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_S,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        randblas_require(layout == buff_layout);

        if (side == Side::Left) {
            // C = alpha * op(A) * op(S) + beta * C
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randblas_require(rows_A == n_rows);
            randblas_require(cols_A == n_cols);

            // Layout-aware ldc check
            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldc >= n);
            }

            // Right sketch in RandBLAS terms: B = alpha * op(A) * op(S) + beta * B
            RandBLAS::sketch_general(layout, trans_A, trans_S, m, n, k, alpha, A_buff, lda, S, beta, C, ldc);

        } else {  // Side::Right
            // C = alpha * op(S) * op(A) + beta * C
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_A == n_rows);
            randblas_require(cols_A == n_cols);

            // Layout-aware ldc check
            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldc >= n);
            }

            // Left sketch in RandBLAS terms: B = alpha * op(S) * op(A) + beta * B
            RandBLAS::sketch_general(layout, trans_S, trans_A, m, n, k, alpha, S, A_buff, lda, beta, C, ldc);
        }
    }

    // =====================================================================
    //  Block view methods
    // =====================================================================

    /// @brief Create a view of a contiguous row range [row_start, row_start + row_count).
    DenseLinOp<T> row_block(int64_t row_start, int64_t row_count) const {
        randblas_require(row_start >= 0);
        randblas_require(row_count > 0);
        randblas_require(row_start + row_count <= n_rows);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + row_start
            : A_buff + row_start * lda;
        return DenseLinOp<T>(row_count, n_cols, offset, lda, buff_layout);
    }

    /// @brief Create a view of a contiguous column range [col_start, col_start + col_count).
    DenseLinOp<T> col_block(int64_t col_start, int64_t col_count) const {
        randblas_require(col_start >= 0);
        randblas_require(col_count > 0);
        randblas_require(col_start + col_count <= n_cols);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + col_start * lda
            : A_buff + col_start;
        return DenseLinOp<T>(n_rows, col_count, offset, lda, buff_layout);
    }

    /// @brief Create a view of a submatrix starting at (row_start, col_start).
    DenseLinOp<T> submatrix(int64_t row_start, int64_t col_start,
                            int64_t row_count, int64_t col_count) const {
        randblas_require(row_start >= 0);
        randblas_require(col_start >= 0);
        randblas_require(row_count > 0);
        randblas_require(col_count > 0);
        randblas_require(row_start + row_count <= n_rows);
        randblas_require(col_start + col_count <= n_cols);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + row_start + col_start * lda
            : A_buff + row_start * lda + col_start;
        return DenseLinOp<T>(row_count, col_count, offset, lda, buff_layout);
    }
};

} // end namespace RandLAPACK::linops
