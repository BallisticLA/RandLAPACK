#pragma once

// Public API: CompositeOperator — implicit product of two linear operators.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <memory>
#include <algorithm>


namespace RandLAPACK::linops {

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
    // CompositeOperator stores references (left_op, right_op) to its two
    // operands, but doesn't always own them. Who owns what depends on how
    // the CompositeOperator was created:
    //
    //   Example 1 — User-constructed (both borrowed, shared_ptrs are null):
    //
    //       DenseLinOp<double> A(...);
    //       SparseLinOp<double> B(...);
    //       CompositeOperator comp(m, n, A, B);
    //       // comp.left_op and comp.right_op are references to A and B.
    //       // The caller is responsible for keeping A and B alive.
    //
    //   Example 2 — row_block() (left operand owned, right borrowed):
    //
    //       auto sub = comp.row_block(0, 100);
    //       // sub needs a new LinOp1 for the row-sliced left operand.
    //       // That object is heap-allocated and held by owned_left_.
    //       // sub.right_op still references the original comp.right_op.
    //
    //   Example 3 — submatrix() (both owned):
    //
    //       auto sub = comp.submatrix(0, 0, 100, 50);
    //       // Both operands are new heap-allocated block views,
    //       // held by owned_left_ and owned_right_.
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

    /// Block size for blocked materialization of the composite operator.
    /// 0 (default) disables blocking and materializes all columns at once.
    int64_t block_size = 0;

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
    void operator()(
        Layout layout,
        Op trans_comp,
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

    // Non-sided sparse matrix multiplication operator.
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Layout layout,
        Op trans_comp,
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
    void operator()(
        Side side,
        Layout layout,
        Op trans_comp,
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
                int64_t temp_rows = right_op.n_rows;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp := LinOp2 * op(B)
                right_op(Side::Left, layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * LinOp1 * temp + beta * C
                left_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {
                // trans_comp == Op::Trans
                // C := alpha * LinOp2^T * LinOp1^T * op(B) + beta * C
                int64_t temp_rows = left_op.n_cols;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                // Step 1: temp := LinOp1^T * op(B)
                left_op(Side::Left, layout, Op::Trans, trans_B, temp_rows, n, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * LinOp2^T * temp + beta * C
                right_op(Side::Left, layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }

        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B) * (LinOp1 * LinOp2) + beta * C

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
                int64_t temp_cols = left_op.n_cols;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                // Step 1: temp := op(B) * LinOp1
                left_op(Side::Right, layout, Op::NoTrans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * temp * LinOp2 + beta * C
                right_op(Side::Right, layout, Op::NoTrans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {  // trans_comp == Op::Trans
                int64_t temp_cols = right_op.n_rows;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                // Step 1: temp := op(B) * LinOp1^T
                left_op(Side::Right, layout, Op::Trans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * temp * LinOp2^T + beta * C
                right_op(Side::Right, layout, Op::Trans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }
        }
    }

    // Sparse matrix multiplication operator with Side parameter.
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Side side,
        Layout layout,
        Op trans_comp,
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
                int64_t temp_rows = right_op.n_rows;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                right_op(Side::Left, layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);
                left_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {
                int64_t temp_rows = left_op.n_cols;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                left_op(Side::Left, layout, Op::Trans, trans_B, temp_rows, n, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);
                right_op(Side::Left, layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }

        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B_sp) * (LinOp1 * LinOp2) + beta * C

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
                int64_t temp_cols = left_op.n_cols;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                left_op(Side::Right, layout, Op::NoTrans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);
                right_op(Side::Right, layout, Op::NoTrans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;

            } else {  // trans_comp == Op::Trans
                int64_t temp_cols = right_op.n_rows;
                int64_t temp_rows = m;
                T* temp_buffer = new T[temp_rows * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : temp_cols;

                left_op(Side::Right, layout, Op::Trans, trans_B,
                        temp_rows, temp_cols, k, (T)1.0, B_sp, (T)0.0, temp_buffer, ldt);
                right_op(Side::Right, layout, Op::Trans, Op::NoTrans,
                         m, n, temp_cols, alpha, temp_buffer, ldt, beta, C, ldc);

                delete[] temp_buffer;
            }
        }
    }

    /// Left-multiply this by a sketching operator.
    /// Computes C = alpha * op(S) * op(L1 * L2) + beta * C (equivalent to Side::Right),
    /// where L1 and L2 are the left and right operators of this composite.
    template <RandBLAS::SketchingOperator SkOp>
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
        (*this)(Side::Right, layout, trans_comp, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    /// Sketching operator multiplication with composite linear operator.
    /// Side refers to the side on which this operator appears.
    ///   Side::Left:  C = alpha * op(L1 * L2) * op(S) + beta * C
    ///   Side::Right: C = alpha * op(S) * op(L1 * L2) + beta * C
    /// where L1 and L2 are the left and right operators of this composite.
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
        if (side == Side::Left) {
            // C = alpha * op(LinOp1 * LinOp2) * op(S) + beta * C

            if (trans_A == Op::NoTrans) {
                int64_t temp_rows = right_op.n_rows;
                T* temp = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                right_op(Side::Left, layout, Op::NoTrans, trans_S, temp_rows, n, k, (T)1.0, S, (T)0.0, temp, ldt);
                left_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;

            } else {  // trans_A == Op::Trans
                int64_t temp_rows = left_op.n_cols;
                T* temp = new T[temp_rows * n]();
                int64_t ldt = (layout == Layout::ColMajor) ? temp_rows : n;

                left_op(Side::Left, layout, Op::Trans, trans_S, temp_rows, n, k, (T)1.0, S, (T)0.0, temp, ldt);
                right_op(Side::Left, layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;
            }

        } else {  // Side::Right
            // C = alpha * op(S) * op(LinOp1 * LinOp2) + beta * C

            if (trans_A == Op::NoTrans) {
                bool use_blocked = (block_size > 0 && layout == Layout::ColMajor);

                if (use_blocked) {
                    int64_t m_inner = right_op.n_rows;  // = left_op.n_cols
                    int64_t n_rows_left = left_op.n_rows;
                    int64_t b_max = std::min(block_size, n);

                    T* eye_block = new T[n_cols * b_max];
                    T* V_block   = new T[m_inner * b_max];
                    T* W         = new T[n_rows_left * b_max];

                    for (int64_t j = 0; j < n; j += block_size) {
                        int64_t b = std::min(block_size, n - j);

                        std::fill_n(eye_block, n_cols * b, (T)0.0);
                        for (int64_t i = 0; i < b; ++i)
                            eye_block[(j + i) + i * n_cols] = (T)1.0;

                        right_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                 m_inner, b, n_cols, (T)1.0, eye_block, n_cols,
                                 (T)0.0, V_block, m_inner);

                        left_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                n_rows_left, b, m_inner, (T)1.0, V_block, m_inner,
                                (T)0.0, W, n_rows_left);

                        T* C_col = C + j * ldc;
                        RandBLAS::sketch_general(
                            Layout::ColMajor, trans_S, Op::NoTrans,
                            m, b, n_rows_left, alpha, S, W, n_rows_left,
                            beta, C_col, ldc);
                    }

                    delete[] eye_block;
                    delete[] V_block;
                    delete[] W;

                } else {
                    int64_t temp_cols = left_op.n_cols;
                    T* temp = new T[m * temp_cols]();
                    int64_t ldt = (layout == Layout::ColMajor) ? m : temp_cols;

                    left_op(Side::Right, layout, Op::NoTrans, trans_S, m, temp_cols, k, (T)1.0, S, (T)0.0, temp, ldt);
                    right_op(Side::Right, layout, Op::NoTrans, Op::NoTrans, m, n, temp_cols, alpha, temp, ldt, beta, C, ldc);

                    delete[] temp;
                }

            } else {  // trans_A == Op::Trans
                int64_t temp_cols = right_op.n_rows;
                T* temp = new T[m * temp_cols]();
                int64_t ldt = (layout == Layout::ColMajor) ? m : temp_cols;

                right_op(Side::Right, layout, Op::Trans, trans_S, m, temp_cols, k, (T)1.0, S, (T)0.0, temp, ldt);
                left_op(Side::Right, layout, Op::Trans, Op::NoTrans, m, n, temp_cols, alpha, temp, ldt, beta, C, ldc);

                delete[] temp;
            }
        }
    }

    // =====================================================================
    //  Block view methods
    // =====================================================================

    /// @brief Extract a row block [row_start, row_start + row_count).
    CompositeOperator row_block(int64_t row_start, int64_t row_count) const {
        auto block_L = std::make_shared<LinOp1>(
            left_op.row_block(row_start, row_count));
        return CompositeOperator(row_count, n_cols,
                                 std::move(block_L), right_op);
    }

    /// @brief Extract a column block [col_start, col_start + col_count).
    CompositeOperator col_block(int64_t col_start, int64_t col_count) const {
        auto block_R = std::make_shared<LinOp2>(
            right_op.col_block(col_start, col_count));
        return CompositeOperator(n_rows, col_count,
                                 left_op, std::move(block_R));
    }

    /// @brief Extract a submatrix at (row_start, col_start).
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

} // end namespace RandLAPACK::linops
