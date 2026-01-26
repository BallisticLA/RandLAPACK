#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "RandBLAS/sparse_data/base.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <concepts>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*            Abstract Linear Operator Concept          */
/*                                                       */
/*********************************************************/
// Abstract linear operator concept
template<typename LinOp, typename T = LinOp::scalar_t>
concept LinearOperator = requires(LinOp A) {
    { A.n_rows }  -> std::same_as<const int64_t&>;
    { A.n_cols }  -> std::same_as<const int64_t&>;
} && requires(LinOp A, Layout layout, Op trans_A, Op trans_B, int64_t m, int64_t n, int64_t k, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
    // A Matmul-like function that updates C := alpha A*B + beta C, where
    // B and C have n columns and are stored in layout order with strides (ldb, ldc).
    { A(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc) } -> std::same_as<void>;
};

/*********************************************************/
/*                                                       */
/*              CompositeOperator                        */
/*                                                       */
/*********************************************************/
// Composite linear operator representing the product of two linear operators.
// Represents the implicit product LinOp1 * LinOp2.
//
// Template parameters:
//   LinOp1 - Left operator type satisfying LinearOperator concept
//   LinOp2 - Right operator type satisfying LinearOperator concept
//
// Functionality:
//   - Supports multiplication with both dense and sparse matrices B
//   - Supports both left and right multiplication via Side parameter
//
// Strategy:
//   All operations use a two-step process with an intermediate buffer:
//     Step 1: temp = LinOp2 * B  (LinOp2 handles dense/sparse B internally)
//     Step 2: result = LinOp1 * temp
//
// Sparse matrix handling:
//   Delegated to the individual operators (DenseLinOp, SparseLinOp, etc.)
//   - DenseLinOp: Uses RandBLAS::right_spmm for dense * sparse
//   - SparseLinOp: Densifies sparse B, then uses left_spmm for sparse * dense
template <LinearOperator LinOp1, LinearOperator LinOp2>
struct CompositeOperator {
    using T = typename LinOp1::scalar_t;
    using scalar_t = T;
    const int64_t n_rows;    // Number of rows in LinOp1 * LinOp2
    const int64_t n_cols;    // Number of columns in LinOp1 * LinOp2
    LinOp1 &left_op;         // Reference to the left operator
    LinOp2 &right_op;        // Reference to the right operator

    CompositeOperator(
        const int64_t n_rows,
        const int64_t n_cols,
        LinOp1 &left_op,
        LinOp2 &right_op
    ) : n_rows(n_rows), n_cols(n_cols), left_op(left_op), right_op(right_op) {
        // Validate dimensions: left_op is (n_rows x right_op.n_rows), right_op is (right_op.n_rows x n_cols)
        randblas_require(left_op.n_cols == right_op.n_rows);
        randblas_require(right_op.n_cols == n_cols);
    }

    // Non-sided dense matrix multiplication operator (required by LinearOperator concept).
    // Defaults to left multiplication: C := alpha * (LinOp1 * LinOp2) * op(B) + beta * C
    void operator()(
        Layout layout,
        Op trans_comp,  // Transposition of the composite operator (LinOp1 * LinOp2)
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
        // Delegate to sided version with Side::Left
        (*this)(Side::Left, layout, trans_comp, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    // Non-sided sparse matrix multiplication operator (required by LinearOperator concept).
    // Defaults to left multiplication: C := alpha * (LinOp1 * LinOp2) * op(B_sp) + beta * C
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Layout layout,
        Op trans_comp,  // Transposition of the composite operator (LinOp1 * LinOp2)
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
        // Delegate to sided version with Side::Left
        (*this)(Side::Left, layout, trans_comp, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);
    }

    // Dense matrix multiplication operator with Side parameter.
    // Handles both left and right multiplication with a dense matrix B.
    //
    // Side::Left:  C := alpha * (LinOp1 * LinOp2) * op(B) + beta * C
    //              Applies operators right-to-left: LinOp2 first, then LinOp1
    //
    // Side::Right: C := alpha * op(B) * (LinOp1 * LinOp2) + beta * C
    //              Applies operators left-to-right: LinOp1 first, then LinOp2
    //              Step 1: temp = op(B) * LinOp1
    //              Step 2: C = alpha * temp * LinOp2 + beta * C
    //
    // Note: We cannot use the transpose trick (C = B*A <=> C^T = A^T * B^T) for
    // Side::Right because it requires swapping layout (ColMajor <-> RowMajor),
    // and some operators (e.g., CholSolverLinOp) only support one layout.
    // Instead, we explicitly implement Side::Right by calling constituent operators
    // with their Side::Right parameter. This is why we have separate Side::Left and
    // Side::Right implementations rather than reusing code via the transpose trick.
    void operator()(
        Side side,
        Layout layout,
        Op trans_comp,  // Transposition of the composite operator (LinOp1 * LinOp2)
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
            // Left multiplication: C := alpha * (LinOp1 * LinOp2) * op(B) + beta * C
            // Compute right-to-left via intermediate buffer
            //   Step 1: temp = LinOp2 * op(B)
            //   Step 2: C = alpha * LinOp1 * temp + beta * C

            // Validate input dimensions
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
            auto [rows_comp, cols_comp] = RandBLAS::dims_before_op(m, k, trans_comp);
            randblas_require(rows_comp <= n_rows);
            randblas_require(cols_comp <= n_cols);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            if (trans_comp == Op::NoTrans) {
                // C := alpha * (LinOp1 * LinOp2) * op(B) + beta * C
                // Intermediate dimension: right_op.n_rows = left_op.n_cols
                int64_t temp_rows = right_op.n_rows;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp := LinOp2 * op(B), dimension (right_op.n_rows x n)
                right_op(Side::Left, layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, const_cast<T*>(B), ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * LinOp1 * temp + beta * C
                left_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {
                // trans_comp == Op::Trans
                // C := alpha * (LinOp1 * LinOp2)^T * op(B) + beta * C
                //    = alpha * LinOp2^T * LinOp1^T * op(B) + beta * C
                // Note: left_op.n_cols is the intermediate dimension
                int64_t temp_rows = left_op.n_cols;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp := LinOp1^T * op(B), dimension (left_op.n_cols x n)
                left_op(Side::Left, layout, Op::Trans, trans_B, temp_rows, n, k, (T)1.0, const_cast<T*>(B), ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * LinOp2^T * temp + beta * C
                right_op(Side::Left, layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }

        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B) * (LinOp1 * LinOp2) + beta * C
            // Compute left-to-right (opposite order from Side::Left):
            //
            //   If trans_comp == NoTrans:
            //     C := alpha * op(B) * (LinOp1 * LinOp2) + beta * C
            //        = alpha * (op(B) * LinOp2) * LinOp1 + beta * C
            //     Step 1: temp := op(B) * LinOp2
            //     Step 2: C := alpha * temp * LinOp1 + beta * C
            //
            //   If trans_comp == Trans:
            //     C := alpha * op(B) * (LinOp1 * LinOp2)^T + beta * C
            //        = alpha * op(B) * LinOp2^T * LinOp1^T + beta * C
            //        = alpha * (op(B) * LinOp2^T) * LinOp1^T + beta * C
            //     Step 1: temp := op(B) * LinOp2^T
            //     Step 2: C := alpha * temp * LinOp1^T + beta * C

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_comp, cols_comp] = RandBLAS::dims_before_op(k, n, trans_comp);
            randblas_require(rows_comp <= n_rows);
            randblas_require(cols_comp <= n_cols);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            if (trans_comp == Op::NoTrans) {
                // C := alpha * op(B) * (LinOp1 * LinOp2) + beta * C
                // Compute left-to-right in operator names: (op(B) * LinOp1) * LinOp2
                // This applies LinOp1 first, then LinOp2
                // Intermediate dimension: left_op.n_cols = right_op.n_rows
                int64_t temp_cols = left_op.n_cols;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                // Step 1: temp := op(B) * LinOp1, dimension (m x left_op.n_cols)
                //   op(B) is (m x k), LinOp1 is (k x left_op.n_cols)
                left_op(Side::Right, layout, Op::NoTrans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, const_cast<T*>(B), ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * temp * LinOp2 + beta * C
                //   temp is (m x left_op.n_cols), LinOp2 is (left_op.n_cols x n)
                right_op(Side::Right, layout, Op::NoTrans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {  // trans_comp == Op::Trans
                // C := alpha * op(B) * (LinOp1 * LinOp2)^T + beta * C
                //    = alpha * op(B) * LinOp2^T * LinOp1^T + beta * C
                // Compute left-to-right in operator names: (op(B) * LinOp1^T) * LinOp2^T
                // This applies LinOp1^T first, then LinOp2^T
                // Intermediate dimension: right_op.n_rows (between op(B) and LinOp1^T)
                int64_t temp_cols = right_op.n_rows;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                // Step 1: temp := op(B) * LinOp1^T, dimension (m x right_op.n_rows)
                //   op(B) is (m x k), LinOp1^T is (k x right_op.n_rows)
                left_op(Side::Right, layout, Op::Trans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, const_cast<T*>(B), ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * temp * LinOp2^T + beta * C
                //   temp is (m x right_op.n_rows), LinOp2^T is (right_op.n_rows x n)
                right_op(Side::Right, layout, Op::Trans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }
        }
    }

    // Sparse matrix multiplication operator with Side parameter.
    // Handles both left and right multiplication with a sparse matrix B_sp.
    //
    // Side::Left:  C := alpha * (LinOp1 * LinOp2) * op(B_sp) + beta * C
    //              Applies operators right-to-left: LinOp2 first, then LinOp1
    //
    // Side::Right: C := alpha * op(B_sp) * (LinOp1 * LinOp2) + beta * C
    //              Applies operators left-to-right: LinOp1 first, then LinOp2
    //
    // Note: We cannot use the transpose trick for Side::Right because it requires
    // layout swapping, which some operators don't support. See dense version above.
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Side side,
        Layout layout,
        Op trans_comp,  // Transposition of the composite operator (LinOp1 * LinOp2)
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
            // Left multiplication: C := alpha * (LinOp1 * LinOp2) * op(B_sp) + beta * C
            // Compute right-to-left via intermediate buffer (same as dense case).
            //   Step 1: temp = LinOp2 * op(B_sp)  [LinOp2 handles sparse B_sp internally]
            //   Step 2: C = alpha * LinOp1 * temp + beta * C
            //
            // Note: Both DenseLinOp and SparseLinOp support sparse matrix multiplication.
            //   - DenseLinOp uses RandBLAS::right_spmm (dense * sparse)
            //   - SparseLinOp densifies B_sp then uses left_spmm (sparse * dense)

            // Validate input dimensions
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
            auto [rows_comp, cols_comp] = RandBLAS::dims_before_op(m, k, trans_comp);
            randblas_require(rows_comp <= n_rows);
            randblas_require(cols_comp <= n_cols);
            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldc >= n);
            }

            if (trans_comp == Op::NoTrans) {
                // C := alpha * (LinOp1 * LinOp2) * op(B_sp) + beta * C
                // Intermediate dimension: right_op.n_rows = left_op.n_cols
                int64_t temp_rows = right_op.n_rows;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp := LinOp2 * op(B_sp), dimension (right_op.n_rows x n)
                // LinOp2's operator handles sparse B_sp internally
                right_op(Side::Left, layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * LinOp1 * temp + beta * C
                left_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {
                // trans_comp == Op::Trans
                // C := alpha * (LinOp1 * LinOp2)^T * op(B_sp) + beta * C
                //    = alpha * LinOp2^T * LinOp1^T * op(B_sp) + beta * C
                // Note: left_op.n_cols is the intermediate dimension
                int64_t temp_rows = left_op.n_cols;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp := LinOp1^T * op(B_sp), dimension (left_op.n_cols x n)
                // LinOp1's operator handles sparse B_sp internally
                left_op(Side::Left, layout, Op::Trans, trans_B, temp_rows, n, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * LinOp2^T * temp + beta * C
                right_op(Side::Left, layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }

        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B_sp) * (LinOp1 * LinOp2) + beta * C
            // Same strategy as dense Side::Right (left-to-right computation)

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_comp, cols_comp] = RandBLAS::dims_before_op(k, n, trans_comp);
            randblas_require(rows_comp <= n_rows);
            randblas_require(cols_comp <= n_cols);
            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldc >= n);
            }

            if (trans_comp == Op::NoTrans) {
                // C := alpha * op(B_sp) * (LinOp1 * LinOp2) + beta * C
                // Compute left-to-right in operator names: (op(B_sp) * LinOp1) * LinOp2
                // This applies LinOp1 first, then LinOp2
                // Intermediate dimension: left_op.n_cols = right_op.n_rows
                int64_t temp_cols = left_op.n_cols;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                // Step 1: temp := op(B_sp) * LinOp1, dimension (m x left_op.n_cols)
                //   op(B_sp) is (m x k), LinOp1 is (k x left_op.n_cols)
                left_op(Side::Right, layout, Op::NoTrans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * temp * LinOp2 + beta * C
                //   temp is (m x left_op.n_cols), LinOp2 is (left_op.n_cols x n)
                right_op(Side::Right, layout, Op::NoTrans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {  // trans_comp == Op::Trans
                // C := alpha * op(B_sp) * (LinOp1 * LinOp2)^T + beta * C
                //    = alpha * op(B_sp) * LinOp2^T * LinOp1^T + beta * C
                // Compute left-to-right in operator names: (op(B_sp) * LinOp1^T) * LinOp2^T
                // This applies LinOp1^T first, then LinOp2^T
                // Intermediate dimension: right_op.n_rows (between op(B_sp) and LinOp1^T)
                int64_t temp_cols = right_op.n_rows;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                // Step 1: temp := op(B_sp) * LinOp1^T, dimension (m x right_op.n_rows)
                //   op(B_sp) is (m x k), LinOp1^T is (k x right_op.n_rows)
                left_op(Side::Right, layout, Op::Trans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * temp * LinOp2^T + beta * C
                //   temp is (m x right_op.n_rows), LinOp2^T is (right_op.n_rows x n)
                right_op(Side::Right, layout, Op::Trans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }
        }
    }

    /// @brief Composite-sketch multiplication with explicit side specification
    ///
    /// @tparam SkOp RandBLAS sketching operator type (DenseSkOp or SparseSkOp)
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for this composite operator (NoTrans or Trans)
    /// @param trans_S Transpose operation for sketching operator S (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(LinOp1 * LinOp2) * op(S) + beta * C
    ///   Computed as: temp = LinOp2 * op(S), then C = alpha * LinOp1 * temp + beta * C
    ///   (or appropriate transpose variant)
    ///
    /// - Side::Right: C := alpha * op(S) * op(LinOp1 * LinOp2) + beta * C
    ///   Computed as: temp = op(S) * LinOp1, then C = alpha * temp * LinOp2 + beta * C
    ///   (or appropriate transpose variant)
    ///
    /// This delegates sketching to constituent operators, avoiding full materialization.
    template <typename SkOp>
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
        if (side == Side::Left) {
            // C = alpha * op(LinOp1 * LinOp2) * op(S) + beta * C
            // op(A) is m×k, op(S) is k×n, C is m×n

            if (trans_A == Op::NoTrans) {
                // C = alpha * (LinOp1 * LinOp2) * op(S) + beta * C
                // Strategy: temp = LinOp2 * op(S), then C = alpha * LinOp1 * temp + beta * C
                // LinOp2 is (right_op.n_rows × n_cols), op(S) is k×n
                // temp is (right_op.n_rows × n)
                int64_t temp_rows = right_op.n_rows;
                T* temp = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp = LinOp2 * op(S), dimension (right_op.n_rows × n)
                right_op(Side::Left, layout, Op::NoTrans, trans_S, temp_rows, n, k, (T)1.0, S, (T)0.0, temp, ldt);

                // Step 2: C = alpha * LinOp1 * temp + beta * C
                // LinOp1 is (n_rows × left_op.n_cols), temp is (left_op.n_cols × n)
                left_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;

            } else {  // trans_A == Op::Trans
                // C = alpha * (LinOp1 * LinOp2)^T * op(S) + beta * C
                //   = alpha * LinOp2^T * LinOp1^T * op(S) + beta * C
                // Strategy: temp = LinOp1^T * op(S), then C = alpha * LinOp2^T * temp + beta * C
                int64_t temp_rows = left_op.n_cols;
                T* temp = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp = LinOp1^T * op(S), dimension (left_op.n_cols × n)
                left_op(Side::Left, layout, Op::Trans, trans_S, temp_rows, n, k, (T)1.0, S, (T)0.0, temp, ldt);

                // Step 2: C = alpha * LinOp2^T * temp + beta * C
                right_op(Side::Left, layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;
            }

        } else {  // Side::Right
            // C = alpha * op(S) * op(LinOp1 * LinOp2) + beta * C
            // op(S) is m×k, op(A) is k×n, C is m×n

            if (trans_A == Op::NoTrans) {
                // C = alpha * op(S) * (LinOp1 * LinOp2) + beta * C
                // Strategy: temp = op(S) * LinOp1, then C = alpha * temp * LinOp2 + beta * C
                int64_t temp_cols = left_op.n_cols;
                T* temp = new T[m * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? m : temp_cols;

                // Step 1: temp = op(S) * LinOp1, dimension (m × left_op.n_cols)
                left_op(Side::Right, layout, Op::NoTrans, trans_S, m, temp_cols, k, (T)1.0, S, (T)0.0, temp, ldt);

                // Step 2: C = alpha * temp * LinOp2 + beta * C
                // temp is (m × left_op.n_cols), LinOp2 is (left_op.n_cols × n_cols)
                right_op(Side::Right, layout, Op::NoTrans, Op::NoTrans, m, n, temp_cols, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;

            } else {  // trans_A == Op::Trans
                // C = alpha * op(S) * (LinOp1 * LinOp2)^T + beta * C
                //   = alpha * op(S) * LinOp2^T * LinOp1^T + beta * C
                // Strategy: temp = op(S) * LinOp2^T, then C = alpha * temp * LinOp1^T + beta * C
                int64_t temp_cols = right_op.n_rows;
                T* temp = new T[m * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? m : temp_cols;

                // Step 1: temp = op(S) * LinOp2^T, dimension (m × right_op.n_rows)
                right_op(Side::Right, layout, Op::Trans, trans_S, m, temp_cols, k, (T)1.0, S, (T)0.0, temp, ldt);

                // Step 2: C = alpha * temp * LinOp1^T + beta * C
                left_op(Side::Right, layout, Op::Trans, Op::NoTrans, m, n, temp_cols, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;
            }
        }
    }
};

/*********************************************************/
/*                                                       */
/*                   SparseLinOp                         */
/*                                                       */
/*********************************************************/

/// @brief Sparse linear operator for matrix multiplication operations
///
/// @details Provides a linear operator interface for sparse matrices, supporting
/// multiplication with both dense and sparse matrices. The operator supports:
/// - Left and right multiplication modes (Side::Left and Side::Right)
/// - Both ColMajor and RowMajor memory layouts
/// - Transpose operations (Op::NoTrans, Op::Trans)
/// - Sparse-dense and sparse-sparse multiplications
///
/// The operator delegates to RandBLAS sparse BLAS routines for optimal performance.
///
/// @tparam SpMat Sparse matrix type (CSC, CSR, or COO format)
///
/// @note For sparse-sparse multiplication, the current implementation densifies
/// one operand, which may not be optimal for very sparse matrices.
template <RandBLAS::sparse_data::SparseMatrix SpMat>
struct SparseLinOp {
    using T = typename SpMat::scalar_t;
    using scalar_t = T;
    const int64_t n_rows;  ///< Number of rows in the operator matrix
    const int64_t n_cols;  ///< Number of columns in the operator matrix
    SpMat &A_sp;          ///< Reference to the sparse matrix data

    /// @brief Construct a sparse linear operator
    /// @param n_rows Number of rows in the sparse matrix
    /// @param n_cols Number of columns in the sparse matrix
    /// @param A_sp Reference to the sparse matrix (CSC, CSR, or COO format)
    SparseLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        SpMat &A_sp
    ) : n_rows(n_rows), n_cols(n_cols), A_sp(A_sp) {

    }

    /// @brief Compute the Frobenius norm of the sparse operator matrix
    /// @return Frobenius norm (computed as L2 norm of nonzero values)
    T fro_nrm(
    ) {
        return blas::nrm2(A_sp.nnz, A_sp.vals, 1);
    }

    /// @brief Sparse-dense matrix multiplication: C := alpha * op(A_sp) * op(B) + beta * C
    ///
    /// @param layout Memory layout of B and C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in op(A) and C
    /// @param n Number of columns in op(B) and C
    /// @param k Inner dimension: columns of op(A), rows of op(B)
    /// @param alpha Scalar multiplier for the product
    /// @param B Pointer to dense matrix B
    /// @param ldb Leading dimension of B (layout-dependent)
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @note Uses RandBLAS::left_spmm for sparse × dense multiplication
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
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(m, k, trans_A);
        randblas_require(rows_submat_A <= n_rows);
        randblas_require(cols_submat_A <= n_cols);

        // Layout-aware dimension checks
        if (layout == Layout::ColMajor) {
            randblas_require(ldb >= rows_B);
            randblas_require(ldc >= m);
        } else {  // RowMajor
            randblas_require(ldb >= cols_B);
            randblas_require(ldc >= n);
        }

        RandBLAS::sparse_data::left_spmm(layout, trans_A, trans_B, m, n, k, alpha, A_sp, 0, 0, B, ldb, beta, C, ldc);
    }

    /// @brief Sparse-dense multiplication with explicit side specification
    ///
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of B and C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param B Pointer to dense matrix B
    /// @param ldb Leading dimension of B (layout-dependent)
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A_sp) * op(B) + beta * C
    /// - Side::Right: C := alpha * op(B) * op(A_sp) + beta * C
    ///
    /// Side::Left delegates to the non-sided operator for efficiency.
    /// Side::Right uses RandBLAS::right_spmm for dense × sparse multiplication.
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        T* B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if (side == Side::Left) {
            // Left multiplication: delegate to non-sided dense operator
            (*this)(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        } else {
            // Side::Right: C := alpha * op(B) * op(A_sp) + beta * C
            // Use RandBLAS::right_spmm for dense × sparse multiplication

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            // Layout-aware dimension checks
            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            // Use RandBLAS right_spmm: C := alpha * op(B) * op(A_sp) + beta * C
            RandBLAS::sparse_data::right_spmm(layout, trans_B, trans_A, m, n, k, alpha, B, ldb, A_sp, 0, 0, beta, C, ldc);
        }
    }

    /// @brief Sparse-sparse matrix multiplication: C := alpha * op(A_sp) * op(B_sp) + beta * C
    ///
    /// @tparam SpMatB Sparse matrix type for B (CSC, CSR, or COO format)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in op(A) and C
    /// @param n Number of columns in op(B) and C
    /// @param k Inner dimension: columns of op(A), rows of op(B)
    /// @param alpha Scalar multiplier for the product
    /// @param B_sp Reference to sparse matrix B
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @note Current implementation densifies B_sp and delegates to sparse-dense operator.
    /// This is suboptimal for very sparse matrices. A true sparse-sparse implementation
    /// would be more efficient but is not yet available in RandBLAS.
    ///
    /// @warning Duplicate (row, col) entries in sparse matrices are correctly summed.
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
        // Validate input dimensions
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(m, k, trans_A);
        randblas_require(rows_submat_A <= n_rows);
        randblas_require(cols_submat_A <= n_cols);

        // Layout-aware ldc validation
        if (layout == Layout::ColMajor) {
            randblas_require(ldc >= m);
        } else {  // RowMajor
            randblas_require(ldc >= n);
        }

        std::cerr << "For now, sparse * sparse is done via densifying the rhs matrix. This is suboptimal." << std::endl;

        // TODO: Implement sparse * sparse multiplication without explicit densification
        // Current approach densifies B_sp, which is not ideal for performance

        // Densify B_sp
        int64_t dense_rows = rows_B;
        int64_t dense_cols = cols_B;
        T* B_dense = new T[dense_rows * dense_cols]();
        int64_t ldb = (layout == Layout::ColMajor) ? dense_rows : dense_cols;

        // Convert sparse to dense, summing duplicates to match spmm semantics
        RandLAPACK::util::sparse_to_dense_summing_duplicates(B_sp, layout, B_dense);

        // Use existing sparse-dense operator
        (*this)(layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);
        delete[] B_dense;
    }

    /// @brief Sparse-sparse multiplication with explicit side specification
    ///
    /// @tparam SpMatB Sparse matrix type for B (CSC, CSR, or COO format)
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param B_sp Reference to sparse matrix B
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A_sp) * op(B_sp) + beta * C
    /// - Side::Right: C := alpha * op(B_sp) * op(A_sp) + beta * C
    ///
    /// Side::Left delegates to the non-sided sparse-sparse operator.
    /// Side::Right densifies B_sp and uses RandBLAS::right_spmm.
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
            // Left multiplication: delegate to default sparse operator
            (*this)(layout, trans_A, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);
        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B_sp) * op(A_sp) + beta * C
            // Strategy: Densify A_sp and use RandBLAS::right_spmm
            // This computes: C := alpha * op(B_sp) * op(A_dense) + beta * C

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            // Layout-aware ldc validation
            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {  // RowMajor
                randblas_require(ldc >= n);
            }

            // Densify B_sp (the input) to avoid densifying the operator A_sp
            T* B_dense = new T[rows_B * cols_B]();
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_sp, layout, B_dense);

            // Compute: C := alpha * op(B_dense) * op(A_sp) + beta * C using right_spmm
            RandBLAS::sparse_data::right_spmm(layout, trans_B, trans_A, m, n, k, alpha, B_dense, ldb, A_sp, 0, 0, beta, C, ldc);

            delete[] B_dense;
        }
    }

    /// @brief Sparse-sketch multiplication with explicit side specification
    ///
    /// @tparam SkOp RandBLAS sketching operator type (DenseSkOp or SparseSkOp)
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for this operator A (NoTrans or Trans)
    /// @param trans_S Transpose operation for sketching operator S (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A) * op(S) + beta * C
    /// - Side::Right: C := alpha * op(S) * op(A) + beta * C
    ///
    /// @note Current implementation densifies the sparse matrix A and uses
    /// RandBLAS::sketch_general. A future optimization could use sparse sketching directly.
    template <typename SkOp>
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
        // Densify the sparse matrix A
        T* A_dense = new T[n_rows * n_cols]();
        int64_t lda = (layout == Layout::ColMajor) ? n_rows : n_cols;
        RandLAPACK::util::sparse_to_dense_summing_duplicates(A_sp, layout, A_dense);

        if (side == Side::Left) {
            // C = alpha * op(A) * op(S) + beta * C
            // op(A) is m×k, op(S) is k×n, C is m×n
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
            RandBLAS::sketch_general(layout, trans_A, trans_S, m, n, k, alpha, A_dense, lda, S, beta, C, ldc);

        } else {  // Side::Right
            // C = alpha * op(S) * op(A) + beta * C
            // op(S) is m×k, op(A) is k×n, C is m×n
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
            RandBLAS::sketch_general(layout, trans_S, trans_A, m, n, k, alpha, S, A_dense, lda, beta, C, ldc);
        }

        delete[] A_dense;
    }
};

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
    ///
    /// @param layout Memory layout of B and C (ColMajor or RowMajor, must match buff_layout)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in op(A) and C
    /// @param n Number of columns in op(B) and C
    /// @param k Inner dimension: columns of op(A), rows of op(B)
    /// @param alpha Scalar multiplier for the product
    /// @param B Pointer to dense matrix B
    /// @param ldb Leading dimension of B (layout-dependent)
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @note Uses BLAS gemm for optimal performance
    /// @note Layout must match buff_layout (the layout of operator matrix A)
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        T* const B,
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
    ///
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of B and C (ColMajor or RowMajor, must match buff_layout)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param B Pointer to dense matrix B
    /// @param ldb Leading dimension of B (layout-dependent)
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A) * op(B) + beta * C
    /// - Side::Right: C := alpha * op(B) * op(A) + beta * C
    ///
    /// Side::Left delegates to the non-sided operator for efficiency.
    /// Side::Right swaps operand order in BLAS gemm call.
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        T* B,
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
    ///
    /// @tparam SpMatB Sparse matrix type for B (CSC, CSR, or COO format)
    /// @param layout Memory layout of C (ColMajor or RowMajor, must match buff_layout)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in op(A) and C
    /// @param n Number of columns in op(B) and C
    /// @param k Inner dimension: columns of op(A), rows of op(B)
    /// @param alpha Scalar multiplier for the product
    /// @param B_sp Reference to sparse matrix B
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @note Uses RandBLAS::right_spmm for dense × sparse multiplication
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
    ///
    /// @tparam SpMatB Sparse matrix type for B (CSC, CSR, or COO format)
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of C (ColMajor or RowMajor, must match buff_layout)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_B Transpose operation for B (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param B_sp Reference to sparse matrix B
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A) * op(B_sp) + beta * C
    /// - Side::Right: C := alpha * op(B_sp) * op(A) + beta * C
    ///
    /// Side::Left delegates to the non-sided operator for efficiency.
    /// Side::Right uses RandBLAS::left_spmm for sparse × dense multiplication.
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

    /// @brief Dense-sketch multiplication with explicit side specification
    ///
    /// @tparam SkOp RandBLAS sketching operator type (DenseSkOp or SparseSkOp)
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of C (ColMajor or RowMajor, must match buff_layout)
    /// @param trans_A Transpose operation for this operator A (NoTrans or Trans)
    /// @param trans_S Transpose operation for sketching operator S (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A) * op(S) + beta * C
    /// - Side::Right: C := alpha * op(S) * op(A) + beta * C
    ///
    /// Uses RandBLAS::sketch_general for efficient sketching.
    template <typename SkOp>
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
            // op(A) is m×k, op(S) is k×n, C is m×n
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
            // op(S) is m×k, op(A) is k×n, C is m×n
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
};

/*********************************************************/
/*                                                       */
/*        Symmetric Linear Operator Concept             */
/*                                                       */
/*********************************************************/
template<typename LinOp, typename T = LinOp::scalar_t>
concept SymmetricLinearOperator = requires(LinOp A) {
    { A.dim }  -> std::same_as<const int64_t&>;
    // It's recommended that A also have const int64_t members n_rows and n_cols,
    // both equal to A.dim.
} && requires(LinOp A, Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
    // A SYMM-like function that updates C := alpha A*B + beta C, where
    // B and C have n columns and are stored in layout order with strides (ldb, ldc).
    //
    // If layout is ColMajor then an error will be thrown if min(ldb, ldc) < A.dim.
    //
    { A(layout, n, alpha, B, ldb, beta, C, ldc) } -> std::same_as<void>;
};

/*********************************************************/
/*                                                       */
/*                  ExplicitSymLinOp                     */
/*                                                       */
/*********************************************************/
template <typename T>
struct ExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const Uplo uplo;
    const T* A_buff;
    const int64_t lda;
    const Layout buff_layout;

    ExplicitSymLinOp(
        int64_t dim,
        Uplo uplo,
        const T* A_buff,
        int64_t lda,
        Layout buff_layout
    ) : m(dim), dim(dim), uplo(uplo), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {}

    // Note: the "layout" parameter here is interpreted for (B and C).
    // If layout conflicts with this->buff_layout then we manipulate
    // parameters to blas::symm to reconcile the different layouts of
    // A vs (B, C).
    void operator()(
        Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        randblas_require(ldb >= dim);
        randblas_require(ldc >= dim);
        auto blas_call_uplo = this->uplo;
        if (layout != this->buff_layout)
            blas_call_uplo = (this->uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        // Reading the "blas_call_uplo" triangle of "this->A_buff" in "layout" order is the same
        // as reading the "this->uplo" triangle of "this->A_buff" in "this->buff_layout" order.
        blas::symm(
            layout, Side::Left, blas_call_uplo, dim, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    }

    inline T operator()(int64_t i, int64_t j) {
        randblas_require(this->uplo == Uplo::Upper && this->buff_layout == Layout::ColMajor);
        if (i > j) {
            return A_buff[j + i*lda];
        } else {
            return A_buff[i + j*lda];
        }
    }
};

/*********************************************************/
/*                                                       */
/*               RegExplicitSymLinOp                     */
/*                                                       */
/*********************************************************/
template <typename T>
struct RegExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const T* A_buff;
    const int64_t lda;
    int64_t num_ops = 1;
    T* regs = nullptr;
    bool _eval_includes_reg;

    static const Uplo uplo = Uplo::Upper;
    static const Layout buff_layout = Layout::ColMajor;

    RegExplicitSymLinOp(
        int64_t dim, const T* A_buff, int64_t lda, T* arg_regs, int64_t arg_num_ops
    ) : m(dim), dim(dim), A_buff(A_buff), lda(lda) {
        randblas_require(lda >= dim);
        _eval_includes_reg = false;
        num_ops = arg_num_ops;
        num_ops = std::max(num_ops, (int64_t) 1);
        regs = new T[num_ops]{};
        std::copy(arg_regs, arg_regs + arg_num_ops, regs);
    }

    RegExplicitSymLinOp(
        int64_t dim, const T* A_buff, int64_t lda, std::vector<T> &arg_regs
    ) : RegExplicitSymLinOp<T>(dim, A_buff, lda, arg_regs.data(), static_cast<int64_t>(arg_regs.size())) {}

    ~RegExplicitSymLinOp() {
        if (regs != nullptr) delete [] regs;
    }

    void set_eval_includes_reg(bool eir) {
        _eval_includes_reg = eir;
    }

    void operator()(Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randblas_require(layout == this->buff_layout);
        randblas_require(ldb >= dim);
        randblas_require(ldc >= dim);
        blas::symm(layout, blas::Side::Left, this->uplo, dim, n, alpha, this->A_buff, this->lda, B, ldb, beta, C, ldc);

        if (_eval_includes_reg) {
            if (num_ops != 1) randblas_require(n == num_ops);
            for (int64_t i = 0; i < n; ++i) {
                T coeff =  alpha * regs[std::min(i, num_ops - 1)];
                blas::axpy(dim, coeff, B + i*ldb, 1, C +  i*ldc, 1);
            }
        }
        return;
    }

    inline T operator()(int64_t i, int64_t j) {
        T val;
        if (i > j) {
            val = A_buff[j + i*lda];
        } else {
            val = A_buff[i + j*lda];
        }
        if (_eval_includes_reg && i == j) {
            randblas_require(num_ops == 1);
            val += regs[0];
        }
        return val;
    }

};

/*********************************************************/
/*                                                       */
/*                  SpectralPrecond                      */
/*                                                       */
/*********************************************************/
template<typename T>
struct SpectralPrecond {

    using scalar_t = T; 
    const int64_t m; // an alias for dim, keeping for backward compatibility reasons.
    const int64_t dim;
    int64_t dim_pre;
    int64_t num_rhs;
    T* V = nullptr;
    T* D = nullptr;
    T* W = nullptr;
    int64_t num_ops = 0;

    /* Suppose we want to precondition a positive semidefinite matrix G_mu = G + mu*I.
     *
     * Once properly preparred, this preconditioner represents a linear operator of the form
     *      P = V diag(D) V' + I.
     * The columns of V approximate the top dim_pre eigenvectors of G, while the 
     * entries of D are *functions of* the corresponding approximate eigenvalues.
     * 
     * The specific form of the entries of D are as follows. Suppose we start with
     * (V, lambda) as approximations of the top dim_pre eigenpairs of G, and define the vector
     *      D0 = (min(lambda) + mu) / (lambda + mu).
     * From a mathematical perspective, this preconditioner represents the linear operator
     *      P = V diag(D0) V' + (I - VV').
     * The action of this linear operator can be computed with two calls to GEMM
     * instead of three if we store D = D0 - 1 instead of D0 itself.
     */

    SpectralPrecond(int64_t dim) : m(dim), dim(dim), dim_pre(0), num_rhs(0) {}

    // Move constructor
    // Call as SpectralPrecond<T> spc(std::move(other)) when we want to transfer the
    // contents of "other" to "this". 
    SpectralPrecond(SpectralPrecond &&other) noexcept
        : m(other.dim), dim(other.dim), dim_pre(other.dim_pre), num_rhs(other.num_rhs), num_ops(other.num_ops)
    {
        std::swap(V, other.V);
        std::swap(D, other.D);
        std::swap(W, other.W);
    }

    // Copy constructor
    // Call as SpectralPrecond<T> spc(other) when we want to copy "other".
    SpectralPrecond(const SpectralPrecond &other)
        : m(other.dim), dim(other.dim), dim_pre(other.dim_pre), num_rhs(other.num_rhs),  num_ops(other.num_ops)
     {
        reset_owned_buffers(dim_pre, num_rhs, num_ops);
        std::copy(other.V, other.V + dim * dim_pre,        V);
        std::copy(other.D, other.D + dim_pre * num_ops, D);
     } 

    ~SpectralPrecond() {
        if (D != nullptr) delete [] D;
        if (V != nullptr) delete [] V;
        if (W != nullptr) delete [] W;
    }

    void reset_owned_buffers(int64_t arg_dim_pre, int64_t arg_num_rhs, int64_t arg_num_ops) {
        randblas_require(arg_num_rhs == arg_num_ops || arg_num_ops == 1);

        if (arg_dim_pre * arg_num_ops > dim_pre * num_ops) {
            if (D != nullptr) delete [] D;
            D = new T[arg_dim_pre * arg_num_ops]{};
        } 
        if (arg_dim_pre > dim_pre) {
            if (V != nullptr) delete [] V;
            V = new T[dim * arg_dim_pre];
        }
        if (arg_dim_pre * arg_num_rhs > dim_pre * num_rhs) {
            if (W != nullptr) delete [] W;
            W = new T[arg_dim_pre * arg_num_rhs];
        }

        dim_pre = arg_dim_pre;
        num_rhs = arg_num_rhs;
        num_ops = arg_num_ops;
    }

    void set_D_from_eigs_and_regs(T* eigvals, T* mus) {
        for (int64_t r = 0; r < num_ops; ++r) {
            T  mu_r = mus[r];
            T* D_r  = D + r*dim_pre;
            T  numerator = eigvals[dim_pre-1] + mu_r;
            for (int i = 0; i < dim_pre; ++i) {
                D_r[i] = (numerator / (eigvals[i] + mu_r)) - 1.0;
            }
        }
        return;
    }

    void prep(std::vector<T> &eigvecs, std::vector<T> &eigvals, std::vector<T> &mus, int64_t arg_num_rhs) {
        // assume eigvals are positive numbers sorted in decreasing order.
        int64_t arg_num_ops = mus.size();
        int64_t arg_dim_pre  = eigvals.size();
        reset_owned_buffers(arg_dim_pre, arg_num_rhs, arg_num_ops);
        set_D_from_eigs_and_regs(eigvals.data(), mus.data());
        std::copy(eigvecs.begin(), eigvecs.end(), V);
        return;
    }

    void operator()(
        Layout layout, int64_t n, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc
    ) {
        randblas_require(layout == Layout::ColMajor);
        randblas_require(ldb >= dim);
        randblas_require(ldc >= dim);
        if (this->num_ops != 1) {
            randblas_require(n == num_ops);
        } else {
            randblas_require(this->num_rhs >= n);
        }
        // update C = alpha*(V diag(D) V' + I)B + beta*C
        //      Step 1: w = V'B                    with blas::gemm
        //      Step 2: w = D w                    with our own kernel
        //      Step 3: C = beta * C + alpha * B   with blas::copy or blas::scal + blas::axpy
        //      Step 4: C = alpha * V w + C        with blas::gemm
        blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, dim_pre, n, dim, (T) 1.0, V, dim, B, ldb, (T) 0.0, W, dim_pre);
 
        // -----> start step 2
        #define mat_D(_i, _j)  ((num_ops == 1) ? D[(_i)] : D[(_i) + dim_pre*(_j)])
        #define mat_W(_i, _j)  W[(_i) + dim_pre*(_j)]
        for (int64_t j = 0; j < n; j++) {
            for (int64_t i = 0; i < dim_pre; i++) {
                mat_W(i, j) = mat_D(i, j) * mat_W(i, j);
            }
        }
        #undef mat_D
        #undef mat_W
        // <----- end step 2

        // -----> start step 3
        int64_t i;
        #define colB(_i) &B[(_i)*ldb]
        #define colC(_i) &C[(_i)*ldb]
        if (beta == (T) 0.0 && alpha == (T) 1.0) {
            for (i = 0; i < n; ++i)
                blas::copy(dim, colB(i), 1, colC(i), 1);
        } else {
            for (i = 0; i < n; ++i) {
                T* Ci = colC(i);
                blas::scal(dim, beta, Ci, 1);
                blas::axpy(dim, alpha, colB(i), 1, Ci, 1);
            }
        }
        #undef colB
        #undef colC
        // <----- end step 3
    
        blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, dim, n, dim_pre, (T) 1.0, V, dim, W, dim_pre, 1.0, C, ldc);
        return;
    }
};

} // end namespace RandLAPACK::linops
