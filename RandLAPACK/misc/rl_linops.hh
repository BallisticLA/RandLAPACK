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

private:
    // ---- Owned operands for block-derived CompositeOperators ----
    //
    // When this CompositeOperator is produced by a block method (row_block,
    // col_block, submatrix), one or both of these shared_ptrs hold the
    // block-derived operand(s).  For externally-constructed operators,
    // both are null and left_op/right_op reference caller-owned objects.
    //
    // IMPORTANT: Declared BEFORE left_op/right_op so that C++ destruction
    // order (reverse of declaration) destroys the references first, then
    // releases the owned operand data they may point into.
    std::shared_ptr<LinOp1> owned_left_;
    std::shared_ptr<LinOp2> owned_right_;

    /// @brief Private constructor: owns left operand, borrows right.
    /// Used by row_block() which extracts a row block from the left operator.
    CompositeOperator(int64_t n_rows, int64_t n_cols,
                      std::shared_ptr<LinOp1> owned_L, LinOp2& right_op)
        : n_rows(n_rows), n_cols(n_cols),
          owned_left_(std::move(owned_L)), owned_right_(),
          left_op(*owned_left_), right_op(right_op) {}

    /// @brief Private constructor: borrows left operand, owns right.
    /// Used by col_block() which extracts a column block from the right operator.
    CompositeOperator(int64_t n_rows, int64_t n_cols,
                      LinOp1& left_op, std::shared_ptr<LinOp2> owned_R)
        : n_rows(n_rows), n_cols(n_cols),
          owned_left_(), owned_right_(std::move(owned_R)),
          left_op(left_op), right_op(*owned_right_) {}

    /// @brief Private constructor: owns both operands.
    /// Used by submatrix() which extracts from both left and right operators.
    CompositeOperator(int64_t n_rows, int64_t n_cols,
                      std::shared_ptr<LinOp1> owned_L,
                      std::shared_ptr<LinOp2> owned_R)
        : n_rows(n_rows), n_cols(n_cols),
          owned_left_(std::move(owned_L)), owned_right_(std::move(owned_R)),
          left_op(*owned_left_), right_op(*owned_right_) {}

public:
    LinOp1 &left_op;         // Reference to the left operator
    LinOp2 &right_op;        // Reference to the right operator

    CompositeOperator(
        const int64_t n_rows,
        const int64_t n_cols,
        LinOp1 &left_op,
        LinOp2 &right_op
    ) : n_rows(n_rows), n_cols(n_cols),
        owned_left_(), owned_right_(),
        left_op(left_op), right_op(right_op) {
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
                right_op(Side::Left, layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

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
                left_op(Side::Left, layout, Op::Trans, trans_B, temp_rows, n, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

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
                        temp_rows, temp_cols, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

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
                        temp_rows, temp_cols, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

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

    /// Non-sided sketching operator multiplication (defaults to left multiplication).
    /// Delegates to sided version with Side::Left.
    ///
    /// @tparam SkOp Sketching operator type (DenseSkOp or SparseSkOp from RandBLAS)
    /// @param layout Memory layout (ColMajor or RowMajor)
    /// @param trans_comp Transposition of the composite operator (LinOp1 * LinOp2)
    /// @param trans_S Transposition of the sketching operator
    /// @param m Number of rows in output C
    /// @param n Number of columns in output C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    template <typename SkOp>
    void operator()(
        Layout layout,
        Op trans_comp,
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
        // Delegate to sided version with Side::Right (S * A multiplication)
        (*this)(Side::Right, layout, trans_comp, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    /// Sketching operator multiplication (for RandBLAS sketching operators).
    /// Handles both left and right sketching with composite operators.
    ///
    /// @param side Side::Left or Side::Right (determines if sketching is on left or right)
    /// @param layout Memory layout (ColMajor or RowMajor)
    /// @param trans_A Transposition of the composite operator (LinOp1 * LinOp2)
    /// @param trans_S Transposition of the sketching operator
    /// @param m Number of rows in output C
    /// @param n Number of columns in output C
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

    // =====================================================================
    //  Block view methods
    // =====================================================================
    //
    //  Row partitioning touches only the left operator (L):
    //      (L * R)[rows, :] = L[rows, :] * R
    //
    //  Column partitioning touches only the right operator (R):
    //      (L * R)[:, cols] = L * R[:, cols]
    //
    //  Submatrix partitioning touches both:
    //      (L * R)[rows, cols] = L[rows, :] * R[:, cols]
    //
    //  The inner dimension (L.n_cols == R.n_rows) is never partitioned,
    //  because blocking the inner dimension would require summing partial
    //  products, which is a different decomposition pattern.
    //
    //  Block-derived CompositeOperators own their block operand(s) via
    //  shared_ptr members (owned_left_, owned_right_).  The non-blocked
    //  operand is still borrowed by reference from this operator's operand.
    //

    /// @brief Extract a row block [row_start, row_start + row_count).
    ///
    /// Only the left operator is partitioned.  The right operator is
    /// borrowed unchanged.  The caller's left operand must outlive this
    /// operator (since right_op is borrowed by reference from it).
    ///
    /// @param row_start  First row of the block (0-indexed)
    /// @param row_count  Number of rows in the block
    /// @return CompositeOperator representing L[rows,:] * R
    CompositeOperator row_block(int64_t row_start, int64_t row_count) const {
        auto block_L = std::make_shared<LinOp1>(
            left_op.row_block(row_start, row_count));
        return CompositeOperator(row_count, n_cols,
                                 std::move(block_L), right_op);
    }

    /// @brief Extract a column block [col_start, col_start + col_count).
    ///
    /// Only the right operator is partitioned.  The left operator is
    /// borrowed unchanged.  The caller's right operand must outlive this
    /// operator (since left_op is borrowed by reference from it).
    ///
    /// @param col_start  First column of the block (0-indexed)
    /// @param col_count  Number of columns in the block
    /// @return CompositeOperator representing L * R[:,cols]
    CompositeOperator col_block(int64_t col_start, int64_t col_count) const {
        auto block_R = std::make_shared<LinOp2>(
            right_op.col_block(col_start, col_count));
        return CompositeOperator(n_rows, col_count,
                                 left_op, std::move(block_R));
    }

    /// @brief Extract a submatrix at (row_start, col_start) with dimensions
    /// row_count x col_count.
    ///
    /// Both operators are partitioned: row block from the left, column block
    /// from the right.  The returned CompositeOperator owns both blocked
    /// operands and is fully self-contained.
    ///
    /// @param row_start  First row of the submatrix (0-indexed)
    /// @param col_start  First column of the submatrix (0-indexed)
    /// @param row_count  Number of rows in the submatrix
    /// @param col_count  Number of columns in the submatrix
    /// @return CompositeOperator representing L[rows,:] * R[:,cols]
    CompositeOperator submatrix(int64_t row_start, int64_t col_start,
                                int64_t row_count, int64_t col_count) const {
        auto block_L = std::make_shared<LinOp1>(
            left_op.row_block(row_start, row_count));
        auto block_R = std::make_shared<LinOp2>(
            right_op.col_block(col_start, col_count));
        return CompositeOperator(row_count, col_count,
                                 std::move(block_L), std::move(block_R));
    }
};

/*********************************************************/
/*                                                       */
/*          Sparse Block View Utilities                  */
/*                                                       */
/*********************************************************/

/// @brief Non-owning view of a contiguous row range of a CSR matrix.
template <typename T, typename sint_t = int64_t>
struct CSRRowBlockView {
    std::vector<sint_t> rowptr;
    T* vals;
    sint_t* colidxs;
    int64_t n_rows;
    int64_t n_cols;
    int64_t nnz;
    RandBLAS::sparse_data::CSRMatrix<T, sint_t> as_csr() {
        return RandBLAS::sparse_data::CSRMatrix<T, sint_t>(
            n_rows, n_cols, nnz, vals, rowptr.data(), colidxs);
    }
};

template <typename T, typename sint_t>
CSRRowBlockView<T, sint_t> csr_row_block(
    RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    int64_t row_start, int64_t row_count
) {
    randblas_require(row_start >= 0 && row_count > 0);
    randblas_require(row_start + row_count <= A.n_rows);
    sint_t base = A.rowptr[row_start];
    int64_t block_nnz = A.rowptr[row_start + row_count] - base;
    std::vector<sint_t> rebased(row_count + 1);
    for (int64_t i = 0; i <= row_count; ++i)
        rebased[i] = A.rowptr[row_start + i] - base;
    return CSRRowBlockView<T, sint_t>{std::move(rebased), A.vals + base, A.colidxs + base, row_count, A.n_cols, block_nnz};
}

template <typename T, typename sint_t = int64_t>
struct CSRColBlock {
    std::vector<T> vals;
    std::vector<sint_t> rowptr;
    std::vector<sint_t> colidxs;
    int64_t n_rows, n_cols, nnz;
    RandBLAS::sparse_data::CSRMatrix<T, sint_t> as_csr() {
        return RandBLAS::sparse_data::CSRMatrix<T, sint_t>(n_rows, n_cols, nnz, vals.data(), rowptr.data(), colidxs.data());
    }
};

template <typename T, typename sint_t>
CSRColBlock<T, sint_t> csr_col_block(
    RandBLAS::sparse_data::CSRMatrix<T, sint_t>& A,
    int64_t col_start, int64_t col_count
) {
    randblas_require(col_start >= 0 && col_count > 0 && col_start + col_count <= A.n_cols);
    int64_t col_end = col_start + col_count;
    std::vector<sint_t> rowptr(A.n_rows + 1, 0);
    for (int64_t i = 0; i < A.n_rows; ++i)
        for (sint_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k)
            if (A.colidxs[k] >= col_start && A.colidxs[k] < col_end) rowptr[i + 1]++;
    for (int64_t i = 0; i < A.n_rows; ++i) rowptr[i + 1] += rowptr[i];
    int64_t block_nnz = rowptr[A.n_rows];
    std::vector<T> vals(block_nnz);
    std::vector<sint_t> colidxs(block_nnz);
    for (int64_t i = 0; i < A.n_rows; ++i) {
        sint_t write_pos = rowptr[i];
        for (sint_t k = A.rowptr[i]; k < A.rowptr[i + 1]; ++k)
            if (A.colidxs[k] >= col_start && A.colidxs[k] < col_end) {
                vals[write_pos] = A.vals[k];
                colidxs[write_pos] = A.colidxs[k] - col_start;
                write_pos++;
            }
    }
    return CSRColBlock<T, sint_t>{std::move(vals), std::move(rowptr), std::move(colidxs), A.n_rows, col_count, block_nnz};
}

template <typename T, typename sint_t = int64_t>
struct CSCColBlockView {
    std::vector<sint_t> colptr;
    T* vals;
    sint_t* rowidxs;
    int64_t n_rows, n_cols, nnz;
    RandBLAS::sparse_data::CSCMatrix<T, sint_t> as_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, sint_t>(n_rows, n_cols, nnz, vals, rowidxs, colptr.data());
    }
};

template <typename T, typename sint_t>
CSCColBlockView<T, sint_t> csc_col_block(
    RandBLAS::sparse_data::CSCMatrix<T, sint_t>& A,
    int64_t col_start, int64_t col_count
) {
    randblas_require(col_start >= 0 && col_count > 0 && col_start + col_count <= A.n_cols);
    sint_t base = A.colptr[col_start];
    int64_t block_nnz = A.colptr[col_start + col_count] - base;
    std::vector<sint_t> rebased(col_count + 1);
    for (int64_t i = 0; i <= col_count; ++i)
        rebased[i] = A.colptr[col_start + i] - base;
    return CSCColBlockView<T, sint_t>{std::move(rebased), A.vals + base, A.rowidxs + base, A.n_rows, col_count, block_nnz};
}

template <typename T, typename sint_t = int64_t>
struct CSCRowBlock {
    std::vector<T> vals;
    std::vector<sint_t> colptr;
    std::vector<sint_t> rowidxs;
    int64_t n_rows, n_cols, nnz;
    RandBLAS::sparse_data::CSCMatrix<T, sint_t> as_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, sint_t>(n_rows, n_cols, nnz, vals.data(), rowidxs.data(), colptr.data());
    }
};

template <typename T, typename sint_t>
CSCRowBlock<T, sint_t> csc_row_block(
    RandBLAS::sparse_data::CSCMatrix<T, sint_t>& A,
    int64_t row_start, int64_t row_count
) {
    randblas_require(row_start >= 0 && row_count > 0 && row_start + row_count <= A.n_rows);
    int64_t row_end = row_start + row_count;
    std::vector<sint_t> colptr(A.n_cols + 1, 0);
    for (int64_t j = 0; j < A.n_cols; ++j)
        for (sint_t k = A.colptr[j]; k < A.colptr[j + 1]; ++k)
            if (A.rowidxs[k] >= row_start && A.rowidxs[k] < row_end) colptr[j + 1]++;
    for (int64_t j = 0; j < A.n_cols; ++j) colptr[j + 1] += colptr[j];
    int64_t block_nnz = colptr[A.n_cols];
    std::vector<T> vals(block_nnz);
    std::vector<sint_t> rowidxs(block_nnz);
    for (int64_t j = 0; j < A.n_cols; ++j) {
        sint_t write_pos = colptr[j];
        for (sint_t k = A.colptr[j]; k < A.colptr[j + 1]; ++k)
            if (A.rowidxs[k] >= row_start && A.rowidxs[k] < row_end) {
                vals[write_pos] = A.vals[k];
                rowidxs[write_pos] = A.rowidxs[k] - row_start;
                write_pos++;
            }
    }
    return CSCRowBlock<T, sint_t>{std::move(vals), std::move(colptr), std::move(rowidxs), row_count, A.n_cols, block_nnz};
}

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
/// @note For sparse-sparse multiplication, uses RandBLAS::spgemm (MKL-accelerated)
/// when available. Falls back to densifying one operand when MKL is absent or
/// when transpose configurations are not directly supported by spgemm.
template <RandBLAS::sparse_data::SparseMatrix SpMat>
struct SparseLinOp {
    using T = typename SpMat::scalar_t;
    using scalar_t = T;
    using sint_t = typename SpMat::index_t;
    const int64_t n_rows;  ///< Number of rows in the operator matrix
    const int64_t n_cols;  ///< Number of columns in the operator matrix

private:
    // ---- Block data ownership (type-erased) ----
    //
    // When this SparseLinOp is produced by a block method (row_block,
    // col_block, submatrix), block_owner_ holds the block/view struct
    // (e.g., CSRRowBlockView, CSRColBlock, CSCColBlockView, CSCRowBlock)
    // that owns the underlying sparse data arrays.  For SparseLinOps
    // constructed directly from an external SpMat&, block_owner_ is null.
    //
    // Uses std::shared_ptr<void> for type-erased ownership: the correct
    // destructor is captured at construction time via shared_ptr's deleter,
    // so a single member can hold any block type without exposing it in the
    // class template signature.
    //
    // IMPORTANT: block_owner_ is declared BEFORE A_sp so that C++ destruction
    // order (reverse of declaration) destroys A_sp (the non-owning view)
    // first, then block_owner_ releases the data A_sp may point into.
    std::shared_ptr<void> block_owner_;

    /// @brief Create a non-owning SpMat view of an existing sparse matrix.
    ///
    /// Uses the SpMat expert constructor (own_memory=false) to produce a
    /// lightweight view that borrows all pointer data from src.  The source
    /// matrix must outlive any SparseLinOp built from this view.
    ///
    /// Dispatches at compile time via if constexpr for CSR, CSC, and COO.
    static SpMat make_view(SpMat& src) {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        using COO = RandBLAS::sparse_data::COOMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            return CSR(src.n_rows, src.n_cols, src.nnz,
                       src.vals, src.rowptr, src.colidxs);
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            return CSC(src.n_rows, src.n_cols, src.nnz,
                       src.vals, src.rowidxs, src.colptr);
        } else if constexpr (std::is_same_v<SpMat, COO>) {
            return COO(src.n_rows, src.n_cols, src.nnz,
                       src.vals, src.rows, src.cols);
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "make_view only supports CSR, CSC, and COO sparse formats");
        }
    }

    /// @brief Private constructor for block-derived SparseLinOps.
    ///
    /// Used by row_block(), col_block(), and submatrix() to return a
    /// SparseLinOp that owns its block data via the type-erased owner.
    ///
    /// @param n_rows  Number of rows in the block
    /// @param n_cols  Number of columns in the block
    /// @param view    Non-owning SpMat whose pointers reference data in owner
    /// @param owner   Type-erased shared_ptr holding the block/view struct
    SparseLinOp(int64_t n_rows, int64_t n_cols,
                SpMat&& view, std::shared_ptr<void> owner)
        : n_rows(n_rows), n_cols(n_cols),
          block_owner_(std::move(owner)), A_sp(std::move(view)) {}

public:
    // ---- Sparse matrix data ----
    //
    // Non-owning view (own_memory == false) of the sparse matrix, created
    // via the SpMat expert constructor.  The actual data resides either in
    // the caller's original SpMat (for externally-constructed operators) or
    // in block_owner_ (for block-derived operators).
    //
    // Declared mutable because const block methods (row_block, col_block,
    // submatrix) must pass A_sp to free functions (e.g., csr_row_block)
    // that take SpMat& (non-const reference).  The underlying matrix data
    // is never modified through A_sp; the mutable qualifier only relaxes
    // the top-level const on the SpMat object itself.
    mutable SpMat A_sp;

    /// @brief Construct a sparse linear operator from an existing sparse matrix.
    ///
    /// Creates a non-owning view of the source matrix via its expert
    /// constructor (own_memory=false).  The source matrix must outlive this
    /// operator and any natural-direction block views derived from it
    /// (which borrow vals/index pointers from the source).
    ///
    /// @param n_rows  Number of rows in the sparse matrix
    /// @param n_cols  Number of columns in the sparse matrix
    /// @param src     Source sparse matrix (CSR, CSC, or COO). Not modified.
    SparseLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        SpMat &src
    ) : n_rows(n_rows), n_cols(n_cols), block_owner_(), A_sp(make_view(src)) {

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
        const T* B,
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
    /// @note When MKL is available and trans_B == NoTrans, uses RandBLAS::spgemm
    /// (sparse × sparse → dense) directly. Otherwise falls back to densifying B_sp.
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

        // C := alpha * op(A_sp) * op(B_sp) + beta * C
        // spgemm supports op(A) * B (opA only), so we can use it when trans_B == NoTrans.
        #if defined(RandBLAS_HAS_MKL)
        if (trans_B == Op::NoTrans) {
            RandBLAS::spgemm(layout, trans_A, alpha, A_sp, B_sp, beta, C, ldc);
            return;
        }
        #endif

        // Fallback: densify B_sp and use sparse-dense multiplication.
        T* B_dense = new T[rows_B * cols_B]();
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);
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
    /// Side::Right uses spgemm when MKL is available and trans_A == NoTrans;
    /// otherwise falls back to densifying B_sp.
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

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldc >= n);
            }

            // spgemm supports op(A) * B, so we can use it when trans_A == NoTrans
            // (B_sp is first arg with opA=trans_B, A_sp is second arg with no op).
            #if defined(RandBLAS_HAS_MKL)
            if (trans_A == Op::NoTrans) {
                RandBLAS::spgemm(layout, trans_B, alpha, B_sp, A_sp, beta, C, ldc);
                return;
            }
            #endif

            // Fallback: densify B_sp and use sparse-dense multiplication.
            T* B_dense = new T[rows_B * cols_B]();
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
            RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);
            RandBLAS::sparse_data::right_spmm(layout, trans_B, trans_A, m, n, k, alpha, B_dense, ldb, A_sp, 0, 0, beta, C, ldc);
            delete[] B_dense;
        }
    }

    /// Non-sided sketching operator multiplication (defaults to left multiplication).
    /// Delegates to sided version with Side::Left.
    ///
    /// @tparam SkOp Sketching operator type (DenseSkOp or SparseSkOp from RandBLAS)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A_sp (NoTrans or Trans)
    /// @param trans_S Transpose operation for S (NoTrans or Trans)
    /// @param m Number of rows in output C
    /// @param n Number of columns in output C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    template <typename SkOp>
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
        // Delegate to sided version with Side::Right (S * A multiplication)
        (*this)(Side::Right, layout, trans_A, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    /// Sketching operator multiplication with sparse linear operator.
    /// Handles both left and right sketching operations.
    ///
    /// @tparam SkOp Sketching operator type (DenseSkOp or SparseSkOp from RandBLAS)
    /// @param side Side::Left or Side::Right (determines if sketching is on left or right)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A_sp (NoTrans or Trans)
    /// @param trans_S Transpose operation for S (NoTrans or Trans)
    /// @param m Number of rows in output C
    /// @param n Number of columns in output C
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
    /// @note When MKL is available, uses spgemm (sparse × sparse → dense) for
    /// sparse sketch operators when the second operand is NoTrans. Otherwise
    /// falls back to densifying A_sp and using sketch_general.
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
        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            // Dense sketch operator: extract buffer and use SpMM directly
            // (avoids full densification of the sparse matrix A).
            if (S.buff == nullptr) {
                RandBLAS::fill_dense(S);
            }
            // Handle layout mismatch: a RowMajor m×n buffer is the same memory
            // as a ColMajor n×m buffer, so we flip the transpose flag.
            Op adjusted_trans = trans_S;
            if (S.layout != layout) {
                adjusted_trans = (trans_S == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            }
            (*this)(side, layout, trans_A, adjusted_trans, m, n, k, alpha, S.buff, S.dist.dim_major, beta, C, ldc);
        } else {
            // Sparse sketch operator: use spgemm (sparse × sparse → dense) when available.
            // spgemm supports op(A) * B (opA only), so we can use it when the second
            // operand has no transpose.

            #if defined(RandBLAS_HAS_MKL)
            {
                if (S.nnz < 0)
                    RandBLAS::fill_sparse(S);
                auto S_coo = RandBLAS::coo_view_of_skop(S);

                if (side == Side::Left && trans_S == Op::NoTrans) {
                    // C = alpha * op(A_sp) * S_coo + beta * C
                    auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
                    randblas_require(rows_A == n_rows);
                    randblas_require(cols_A == n_cols);
                    if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                    else randblas_require(ldc >= n);
                    RandBLAS::spgemm(layout, trans_A, alpha, A_sp, S_coo, beta, C, ldc);
                    return;
                }
                if (side == Side::Right && trans_A == Op::NoTrans) {
                    // C = alpha * op(S_coo) * A_sp + beta * C
                    auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
                    randblas_require(rows_A == n_rows);
                    randblas_require(cols_A == n_cols);
                    if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                    else randblas_require(ldc >= n);
                    RandBLAS::spgemm(layout, trans_S, alpha, S_coo, A_sp, beta, C, ldc);
                    return;
                }
            }
            #endif

            // Fallback: densify A_sp and use sketch_general.
            T* A_dense = new T[n_rows * n_cols]();
            int64_t lda = (layout == Layout::ColMajor) ? n_rows : n_cols;
            RandLAPACK::util::sparse_to_dense(A_sp, layout, A_dense);

            if (side == Side::Left) {
                auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
                randblas_require(rows_A == n_rows);
                randblas_require(cols_A == n_cols);
                if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                else randblas_require(ldc >= n);
                RandBLAS::sketch_general(layout, trans_A, trans_S, m, n, k, alpha, A_dense, lda, S, beta, C, ldc);
            } else {
                auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
                randblas_require(rows_A == n_rows);
                randblas_require(cols_A == n_cols);
                if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                else randblas_require(ldc >= n);
                RandBLAS::sketch_general(layout, trans_S, trans_A, m, n, k, alpha, S, A_dense, lda, beta, C, ldc);
            }

            delete[] A_dense;
        }
    }

    /// @brief Extract a row block [row_start, row_start + row_count).
    ///
    /// For CSR: natural-direction O(row_count) view via CSRRowBlockView.
    ///          Parent matrix must outlive the returned operator.
    /// For CSC: cross-direction O(nnz) owning copy via CSCRowBlock.
    ///          Returned operator is self-contained.
    ///
    /// @param row_start First row of the block (0-indexed)
    /// @param row_count Number of rows in the block
    /// @return SparseLinOp viewing the extracted row block
    SparseLinOp row_block(int64_t row_start, int64_t row_count) const {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            // Natural direction: CSRRowBlockView owns rebased rowptr,
            // borrows vals/colidxs from this operator's source.
            auto blk = std::make_shared<CSRRowBlockView<T, sint_t>>(
                csr_row_block(A_sp, row_start, row_count));
            auto view = blk->as_csr();
            return SparseLinOp(row_count, A_sp.n_cols,
                               std::move(view), std::move(blk));
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            // Cross direction: CSCRowBlock owns all arrays (full copy).
            auto blk = std::make_shared<CSCRowBlock<T, sint_t>>(
                csc_row_block(A_sp, row_start, row_count));
            auto view = blk->as_csc();
            return SparseLinOp(row_count, A_sp.n_cols,
                               std::move(view), std::move(blk));
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "row_block is only supported for CSR and CSC sparse formats");
        }
    }

    /// @brief Extract a column block [col_start, col_start + col_count).
    ///
    /// For CSR: cross-direction O(nnz) owning copy via CSRColBlock.
    ///          Returned operator is self-contained.
    /// For CSC: natural-direction O(col_count) view via CSCColBlockView.
    ///          Parent matrix must outlive the returned operator.
    ///
    /// @param col_start First column of the block (0-indexed)
    /// @param col_count Number of columns in the block
    /// @return SparseLinOp viewing the extracted column block
    SparseLinOp col_block(int64_t col_start, int64_t col_count) const {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            // Cross direction: CSRColBlock owns all arrays (full copy).
            auto blk = std::make_shared<CSRColBlock<T, sint_t>>(
                csr_col_block(A_sp, col_start, col_count));
            auto view = blk->as_csr();
            return SparseLinOp(A_sp.n_rows, col_count,
                               std::move(view), std::move(blk));
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            // Natural direction: CSCColBlockView owns rebased colptr,
            // borrows vals/rowidxs from this operator's source.
            auto blk = std::make_shared<CSCColBlockView<T, sint_t>>(
                csc_col_block(A_sp, col_start, col_count));
            auto view = blk->as_csc();
            return SparseLinOp(A_sp.n_rows, col_count,
                               std::move(view), std::move(blk));
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "col_block is only supported for CSR and CSC sparse formats");
        }
    }

    /// @brief Extract a submatrix at (row_start, col_start) with dimensions
    /// row_count x col_count.
    ///
    /// Implemented as a two-step extraction: natural-direction block first
    /// (cheap O(block_size)), then cross-direction extraction on the result
    /// (O(block_nnz)).  The returned operator is always self-contained
    /// because the cross-direction step produces a full owning copy.
    ///
    /// For CSR: row block then col extraction -> CSRColBlock.
    /// For CSC: col block then row extraction -> CSCRowBlock.
    ///
    /// @param row_start First row of the submatrix (0-indexed)
    /// @param col_start First column of the submatrix (0-indexed)
    /// @param row_count Number of rows in the submatrix
    /// @param col_count Number of columns in the submatrix
    /// @return SparseLinOp viewing the extracted submatrix
    SparseLinOp submatrix(int64_t row_start, int64_t col_start,
                          int64_t row_count, int64_t col_count) const {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            // Step 1: natural row-block (O(row_count), temporary view)
            auto row_view = csr_row_block(A_sp, row_start, row_count);
            auto row_csr = row_view.as_csr();
            // Step 2: cross-direction col extraction (O(block_nnz), full copy)
            auto blk = std::make_shared<CSRColBlock<T, sint_t>>(
                csr_col_block(row_csr, col_start, col_count));
            auto view = blk->as_csr();
            return SparseLinOp(row_count, col_count,
                               std::move(view), std::move(blk));
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            // Step 1: natural col-block (O(col_count), temporary view)
            auto col_view = csc_col_block(A_sp, col_start, col_count);
            auto col_csc = col_view.as_csc();
            // Step 2: cross-direction row extraction (O(block_nnz), full copy)
            auto blk = std::make_shared<CSCRowBlock<T, sint_t>>(
                csc_row_block(col_csc, row_start, row_count));
            auto view = blk->as_csc();
            return SparseLinOp(row_count, col_count,
                               std::move(view), std::move(blk));
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "submatrix is only supported for CSR and CSC sparse formats");
        }
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

    /// Non-sided sketching operator multiplication (defaults to left multiplication).
    /// Delegates to sided version with Side::Left.
    ///
    /// @tparam SkOp Sketching operator type (DenseSkOp or SparseSkOp from RandBLAS)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_S Transpose operation for S (NoTrans or Trans)
    /// @param m Number of rows in output C
    /// @param n Number of columns in output C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    template <typename SkOp>
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
        // Delegate to sided version with Side::Right (S * A multiplication)
        (*this)(Side::Right, layout, trans_A, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    /// Sketching operator multiplication with dense linear operator.
    /// Handles both left and right sketching operations.
    ///
    /// @tparam SkOp Sketching operator type (DenseSkOp or SparseSkOp from RandBLAS)
    /// @param side Side::Left or Side::Right (determines if sketching is on left or right)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for A (NoTrans or Trans)
    /// @param trans_S Transpose operation for S (NoTrans or Trans)
    /// @param m Number of rows in output C
    /// @param n Number of columns in output C
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

    // =====================================================================
    //  Block view methods
    // =====================================================================

    /// @brief Create a view of a contiguous row range [row_start, row_start + row_count).
    ///
    /// Returns a non-owning DenseLinOp whose A_buff points into this operator's memory.
    /// The parent must outlive the returned view. The view preserves the original lda
    /// and is directly usable in BLAS/LAPACK calls.
    ///
    /// @param row_start First row of the block (0-indexed)
    /// @param row_count Number of rows in the block
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
    ///
    /// Returns a non-owning DenseLinOp whose A_buff points into this operator's memory.
    /// The parent must outlive the returned view. The view preserves the original lda
    /// and is directly usable in BLAS/LAPACK calls.
    ///
    /// @param col_start First column of the block (0-indexed)
    /// @param col_count Number of columns in the block
    DenseLinOp<T> col_block(int64_t col_start, int64_t col_count) const {
        randblas_require(col_start >= 0);
        randblas_require(col_count > 0);
        randblas_require(col_start + col_count <= n_cols);
        const T* offset = (buff_layout == Layout::ColMajor)
            ? A_buff + col_start * lda
            : A_buff + col_start;
        return DenseLinOp<T>(n_rows, col_count, offset, lda, buff_layout);
    }

    /// @brief Create a view of a submatrix starting at (row_start, col_start)
    /// with dimensions row_count x col_count.
    ///
    /// Returns a non-owning DenseLinOp whose A_buff points into this operator's memory.
    /// The parent must outlive the returned view. The view preserves the original lda
    /// and is directly usable in BLAS/LAPACK calls.
    ///
    /// @param row_start First row of the submatrix (0-indexed)
    /// @param col_start First column of the submatrix (0-indexed)
    /// @param row_count Number of rows in the submatrix
    /// @param col_count Number of columns in the submatrix
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