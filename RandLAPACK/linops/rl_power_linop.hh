#pragma once

// Public API: PowerOp: implicit j-th power of a square linear operator.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_exceptions.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <utility>
#include <type_traits>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                      PowerOp                          */
/*                                                       */
/*********************************************************/
// Generic LinearOperator wrapper that represents A^j for a square base operator A.
//
// Template parameter:
//   InnerOp - Square base operator satisfying LinearOperator concept (dense, sparse,
//             composite, sparse-solver-inverse, etc.)
//
// Strategy:
//   Each application chains j calls to the base operator with two ping-pong scratch
//   buffers.  A^j is never materialized.
//
//   j == 1: single base call, no scratch.
//   j >= 2: one scratch for the first apply; a second scratch for the middle applies
//           (j == 2 only allocates the first).  The final apply writes to C and
//           respects the user's alpha/beta.
//
// Restrictions:
//   - base.n_rows must equal base.n_cols.  PowerOp is only well-defined for square base.
//   - Side::Left only (square A^j on the left of B).  Side::Right could be added by
//     mirroring the loop, but no current consumer needs it.
//
// Op::Trans semantics:
//   trans_A == Op::Trans applies (A^T)^j == (A^j)^T: each iteration dispatches the
//   base op with Op::Trans.  Intermediate scratch dispatches always use Op::NoTrans.
//
template <LinearOperator InnerOp>
struct PowerOp {
    using T = typename InnerOp::scalar_t;
    using scalar_t = T;

    InnerOp& base;
    const int j;
    const int64_t n_rows;
    const int64_t n_cols;

    PowerOp(InnerOp& base_op, int power)
        : base(base_op), j(power),
          n_rows(base_op.n_rows), n_cols(base_op.n_cols)
    {
        randlapack_require(base.n_rows == base.n_cols)
            << "PowerOp: base must be square, got n_rows=" << base.n_rows
            << " n_cols=" << base.n_cols;
        randlapack_require(power >= 1)
            << "PowerOp: power=" << power << " must be >= 1 (identity j=0 not supported; caller can lacpy)";
    }

    // Concept-required 12-arg overload (no Side); delegates to Side::Left.
    void operator()(
        Layout layout, Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Left, layout, trans_A, trans_B,
                m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    // C := alpha * (base^j)^{trans_A} * op_{trans_B}(B) + beta * C
    //
    // Since base is square (N x N), m == k == N for any valid Side::Left call.
    // op_{trans_B}(B) has shape N x n; C has shape N x n.
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left) << "PowerOp supports Side::Left only";
        randlapack_require(m == n_rows) << "PowerOp: m=" << m << " must equal n_rows=" << n_rows;
        randlapack_require(k == n_rows) << "PowerOp: k=" << k << " must equal n_rows=" << n_rows;

        // side is fixed to Left above, so base is dispatched through the concept-
        // required 12-arg overload (no Side param needed); this is the form every
        // LinearOperator is guaranteed to provide.
        if (j == 1) {
            base(layout, trans_A, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
            return;
        }

        // j >= 2: ping-pong scratch.
        // Layout-aware leading dimension: m for ColMajor (m x n stored), n for RowMajor.
        int64_t ldt = (layout == Layout::ColMajor) ? m : n;
        T* buf_a = new T[(size_t)m * (size_t)n]();
        T* buf_b = (j >= 3) ? new T[(size_t)m * (size_t)n]() : nullptr;

        // 1st apply: buf_a := base^{trans_A} * op_{trans_B}(B)
        base(layout, trans_A, trans_B,
             m, n, k, (T)1.0, B, ldb, (T)0.0, buf_a, ldt);

        // Middle applies (only run when j >= 3): ping-pong buf_a <-> buf_b.
        T* in_buf  = buf_a;
        T* out_buf = buf_b;
        for (int it = 1; it < j - 1; ++it) {
            base(layout, trans_A, Op::NoTrans,
                 m, n, m, (T)1.0, in_buf, ldt, (T)0.0, out_buf, ldt);
            std::swap(in_buf, out_buf);
        }

        // Last apply writes to C with the user's alpha/beta.
        base(layout, trans_A, Op::NoTrans,
             m, n, m, alpha, in_buf, ldt, beta, C, ldc);

        delete[] buf_a;
        if (buf_b) delete[] buf_b;
    }

    // SkOp overload: materialize S as a dense matrix, then delegate to the dense apply.
    // Square base means op_{trans_A}(base^j) is square N x N, so op(S) must be N x n,
    // i.e. S is k x n or n x k. Side::Left only (checked before any allocation).
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S,
        T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left) << "PowerOp SkOp overload supports Side::Left only";

        // SparseSkOp materialization only supports ColMajor (see below). Checked
        // before S_dense is allocated so the reject path cannot leak it.
        if constexpr (!std::is_same_v<typename SkOp::distribution_t, RandBLAS::DenseDist>) {
            randlapack_require(layout == Layout::ColMajor)
                << "PowerOp SkOp overload materializes SparseSkOp in ColMajor only, got RowMajor";
        }

        int64_t S_rows = S.n_rows;
        int64_t S_cols = S.n_cols;
        int64_t lds = (layout == Layout::ColMajor) ? S_rows : S_cols;

        T* S_dense = new T[(size_t)S_rows * (size_t)S_cols];

        if constexpr (std::is_same_v<typename SkOp::distribution_t, RandBLAS::DenseDist>) {
            // Materialize directly in the caller's layout: no sketch-by-identity
            // GEMM, no S_cols^2 identity buffer.
            RandBLAS::fill_dense_unpacked(layout, S.dist, S_rows, S_cols, 0, 0, S_dense, S.seed_state);
        } else {
            T* I_block = new T[(size_t)S_cols * (size_t)S_cols]();
            for (int64_t i = 0; i < S_cols; ++i) I_block[i + i * S_cols] = (T)1.0;
            RandBLAS::sketch_general(
                Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                S_rows, S_cols, S_cols,
                (T)1.0, S, I_block, S_cols,
                (T)0.0, S_dense, S_rows);
            delete[] I_block;
        }

        // Delegate to the dense overload's concept-required 12-arg form (side is
        // fixed to Left above).
        (*this)(layout, trans_A, trans_S,
                m, n, k, alpha, S_dense, lds, beta, C, ldc);

        delete[] S_dense;
    }
};

} // namespace RandLAPACK::linops
