#pragma once

// Public API: DenseLinOp — linear operator backed by a dense matrix buffer.

#include "rl_exceptions.hh"
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
            randlapack_error_if_msg(lda < n_rows, "lda=%lld < n_rows=%lld (lda must be >= n_rows under ColMajor)", (long long)lda, (long long)n_rows);
        } else {  // RowMajor
            randlapack_error_if_msg(lda < n_cols, "lda=%lld < n_cols=%lld (lda must be >= n_cols under RowMajor)", (long long)lda, (long long)n_cols);
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
        (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    /// @brief Dense-dense multiplication with explicit side specification.
    /// Side refers to the side on which this operator appears.
    ///   Side::Left:  C = alpha * op(A) * op(B) + beta * C
    ///   Side::Right: C = alpha * op(B) * op(A) + beta * C
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
        randlapack_error_if_msg(layout != buff_layout, "operation layout must match the operator storage layout (buff_layout)");

        if (side == Side::Left) {
            // C := alpha * op(A) * op(B) + beta * C
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randlapack_error_if_msg(rows_A != n_rows, "op(A) row dim inferred from (m, k, trans_A) is %lld but operator n_rows=%lld", (long long)rows_A, (long long)n_rows);
            randlapack_error_if_msg(cols_A != n_cols, "op(A) col dim inferred from (m, k, trans_A) is %lld but operator n_cols=%lld", (long long)cols_A, (long long)n_cols);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randlapack_error_if_msg(ldb < rows_B, "ldb=%lld < rows_B=%lld (ldb must be >= rows of op(B) under ColMajor)", (long long)ldb, (long long)rows_B);
                randlapack_error_if_msg(ldc < m, "ldc=%lld < m=%lld (ldc must be >= m under ColMajor)", (long long)ldc, (long long)m);
            } else {  // RowMajor
                randlapack_error_if_msg(ldb < cols_B, "ldb=%lld < cols_B=%lld (ldb must be >= cols of op(B) under RowMajor)", (long long)ldb, (long long)cols_B);
                randlapack_error_if_msg(ldc < n, "ldc=%lld < n=%lld (ldc must be >= n under RowMajor)", (long long)ldc, (long long)n);
            }

            blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A_buff, lda, B, ldb, beta, C, ldc);
        } else {  // Side::Right
            // C := alpha * op(B) * op(A) + beta * C
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randlapack_error_if_msg(rows_A != n_rows, "op(A) row dim inferred from (m, k, trans_A) is %lld but operator n_rows=%lld", (long long)rows_A, (long long)n_rows);
            randlapack_error_if_msg(cols_A != n_cols, "op(A) col dim inferred from (m, k, trans_A) is %lld but operator n_cols=%lld", (long long)cols_A, (long long)n_cols);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randlapack_error_if_msg(ldb < rows_B, "ldb=%lld < rows_B=%lld (ldb must be >= rows of op(B) under ColMajor)", (long long)ldb, (long long)rows_B);
                randlapack_error_if_msg(ldc < m, "ldc=%lld < m=%lld (ldc must be >= m under ColMajor)", (long long)ldc, (long long)m);
            } else {  // RowMajor
                randlapack_error_if_msg(ldb < cols_B, "ldb=%lld < cols_B=%lld (ldb must be >= cols of op(B) under RowMajor)", (long long)ldb, (long long)cols_B);
                randlapack_error_if_msg(ldc < n, "ldc=%lld < n=%lld (ldc must be >= n under RowMajor)", (long long)ldc, (long long)n);
            }

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
        (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);
    }

    /// @brief Dense-sparse multiplication with explicit side specification.
    /// Side refers to the side on which this operator appears.
    ///   Side::Left:  C = alpha * op(A) * op(B_sp) + beta * C
    ///   Side::Right: C = alpha * op(B_sp) * op(A) + beta * C
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
        randlapack_error_if_msg(layout != buff_layout, "operation layout must match the operator storage layout (buff_layout)");

        if (side == Side::Left) {
            // C = alpha * op(A) * op(B_sp) + beta * C
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randlapack_error_if_msg(rows_A != n_rows, "op(A) row dim inferred from (m, k, trans_A) is %lld but operator n_rows=%lld", (long long)rows_A, (long long)n_rows);
            randlapack_error_if_msg(cols_A != n_cols, "op(A) col dim inferred from (m, k, trans_A) is %lld but operator n_cols=%lld", (long long)cols_A, (long long)n_cols);

            if (layout == Layout::ColMajor) {
                randlapack_error_if_msg(ldc < m, "ldc=%lld < m=%lld (ldc must be >= m under ColMajor)", (long long)ldc, (long long)m);
            } else {
                randlapack_error_if_msg(ldc < n, "ldc=%lld < n=%lld (ldc must be >= n under RowMajor)", (long long)ldc, (long long)n);
            }

            RandBLAS::sparse_data::right_spmm(layout, trans_A, trans_B, m, n, k, alpha, A_buff, lda, B_sp, 0, 0, beta, C, ldc);
        } else {  // Side::Right
            // C = alpha * op(B_sp) * op(A) + beta * C
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randlapack_error_if_msg(rows_A != n_rows, "op(A) row dim inferred from (m, k, trans_A) is %lld but operator n_rows=%lld", (long long)rows_A, (long long)n_rows);
            randlapack_error_if_msg(cols_A != n_cols, "op(A) col dim inferred from (m, k, trans_A) is %lld but operator n_cols=%lld", (long long)cols_A, (long long)n_cols);

            if (layout == Layout::ColMajor) {
                randlapack_error_if_msg(ldc < m, "ldc=%lld < m=%lld (ldc must be >= m under ColMajor)", (long long)ldc, (long long)m);
            } else {
                randlapack_error_if_msg(ldc < n, "ldc=%lld < n=%lld (ldc must be >= n under RowMajor)", (long long)ldc, (long long)n);
            }

            RandBLAS::sparse_data::left_spmm(layout, trans_B, trans_A, m, n, k, alpha, B_sp, 0, 0, A_buff, lda, beta, C, ldc);
        }
    }

    /// Left-multiply this by a sketching operator.
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
    /// Side refers to the side on which this operator appears.
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
        randlapack_error_if_msg(layout != buff_layout, "operation layout must match the operator storage layout (buff_layout)");

        if (side == Side::Left) {
            // C = alpha * op(A) * op(S) + beta * C
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randlapack_error_if_msg(rows_A != n_rows, "op(A) row dim inferred from (m, k, trans_A) is %lld but operator n_rows=%lld", (long long)rows_A, (long long)n_rows);
            randlapack_error_if_msg(cols_A != n_cols, "op(A) col dim inferred from (m, k, trans_A) is %lld but operator n_cols=%lld", (long long)cols_A, (long long)n_cols);

            // Layout-aware ldc check
            if (layout == Layout::ColMajor) {
                randlapack_error_if_msg(ldc < m, "ldc=%lld < m=%lld (ldc must be >= m under ColMajor)", (long long)ldc, (long long)m);
            } else {
                randlapack_error_if_msg(ldc < n, "ldc=%lld < n=%lld (ldc must be >= n under RowMajor)", (long long)ldc, (long long)n);
            }

            // Right sketch in RandBLAS terms: B = alpha * op(A) * op(S) + beta * B
            RandBLAS::sketch_general(layout, trans_A, trans_S, m, n, k, alpha, A_buff, lda, S, beta, C, ldc);

        } else {  // Side::Right
            // C = alpha * op(S) * op(A) + beta * C
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randlapack_error_if_msg(rows_A != n_rows, "op(A) row dim inferred from (m, k, trans_A) is %lld but operator n_rows=%lld", (long long)rows_A, (long long)n_rows);
            randlapack_error_if_msg(cols_A != n_cols, "op(A) col dim inferred from (m, k, trans_A) is %lld but operator n_cols=%lld", (long long)cols_A, (long long)n_cols);

            // Layout-aware ldc check
            if (layout == Layout::ColMajor) {
                randlapack_error_if_msg(ldc < m, "ldc=%lld < m=%lld (ldc must be >= m under ColMajor)", (long long)ldc, (long long)m);
            } else {
                randlapack_error_if_msg(ldc < n, "ldc=%lld < n=%lld (ldc must be >= n under RowMajor)", (long long)ldc, (long long)n);
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
        randlapack_error_if_msg(row_start < 0, "row_start=%lld must be >= 0", (long long)row_start);
        randlapack_error_if_msg(row_count <= 0, "row_count=%lld must be > 0", (long long)row_count);
        randlapack_error_if_msg(row_start + row_count > n_rows, "row_start=%lld + row_count=%lld exceeds n_rows=%lld", (long long)row_start, (long long)row_count, (long long)n_rows);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + row_start
            : A_buff + row_start * lda;
        return DenseLinOp<T>(row_count, n_cols, offset, lda, buff_layout);
    }

    /// @brief Create a view of a contiguous column range [col_start, col_start + col_count).
    DenseLinOp<T> col_block(int64_t col_start, int64_t col_count) const {
        randlapack_error_if_msg(col_start < 0, "col_start=%lld must be >= 0", (long long)col_start);
        randlapack_error_if_msg(col_count <= 0, "col_count=%lld must be > 0", (long long)col_count);
        randlapack_error_if_msg(col_start + col_count > n_cols, "col_start=%lld + col_count=%lld exceeds n_cols=%lld", (long long)col_start, (long long)col_count, (long long)n_cols);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + col_start * lda
            : A_buff + col_start;
        return DenseLinOp<T>(n_rows, col_count, offset, lda, buff_layout);
    }

    /// @brief Create a view of a submatrix starting at (row_start, col_start).
    DenseLinOp<T> submatrix(int64_t row_start, int64_t col_start,
                            int64_t row_count, int64_t col_count) const {
        randlapack_error_if_msg(row_start < 0, "row_start=%lld must be >= 0", (long long)row_start);
        randlapack_error_if_msg(col_start < 0, "col_start=%lld must be >= 0", (long long)col_start);
        randlapack_error_if_msg(row_count <= 0, "row_count=%lld must be > 0", (long long)row_count);
        randlapack_error_if_msg(col_count <= 0, "col_count=%lld must be > 0", (long long)col_count);
        randlapack_error_if_msg(row_start + row_count > n_rows, "row_start=%lld + row_count=%lld exceeds n_rows=%lld", (long long)row_start, (long long)row_count, (long long)n_rows);
        randlapack_error_if_msg(col_start + col_count > n_cols, "col_start=%lld + col_count=%lld exceeds n_cols=%lld", (long long)col_start, (long long)col_count, (long long)n_cols);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + row_start + col_start * lda
            : A_buff + row_start * lda + col_start;
        return DenseLinOp<T>(row_count, col_count, offset, lda, buff_layout);
    }
};

} // end namespace RandLAPACK::linops
