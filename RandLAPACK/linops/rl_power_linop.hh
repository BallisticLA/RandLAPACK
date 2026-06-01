#pragma once

// Public API: PowerOp — implicit j-th power of a square linear operator.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <algorithm>


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
//             composite, sparse-LU-inverse, etc.)
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
//   trans_A == Op::Trans applies (A^T)^j == (A^j)^T — each iteration dispatches the
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
        randblas_require(base.n_rows == base.n_cols);   // PowerOp requires square base
        randblas_require(power >= 1);                   // identity (j=0) not supported; caller can lacpy
    }

    // Concept-required 12-arg overload (no Side); delegates to Side::Left.
    void operator()(
        Layout layout, Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, T* const B, int64_t ldb,
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
        T alpha, T* const B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        randblas_require(side == Side::Left);
        randblas_require(m == n_rows);
        randblas_require(k == n_rows);

        if (j == 1) {
            base(side, layout, trans_A, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
            return;
        }

        // j >= 2: ping-pong scratch.
        // Layout-aware leading dimension: m for ColMajor (m x n stored), n for RowMajor.
        int64_t ldt = (layout == Layout::ColMajor) ? m : n;
        T* buf_a = new T[(size_t)m * (size_t)n]();
        T* buf_b = (j >= 3) ? new T[(size_t)m * (size_t)n]() : nullptr;

        // 1st apply: buf_a := base^{trans_A} * op_{trans_B}(B)
        base(side, layout, trans_A, trans_B,
             m, n, k, (T)1.0, B, ldb, (T)0.0, buf_a, ldt);

        // Middle applies (only run when j >= 3): ping-pong buf_a <-> buf_b.
        T* in_buf  = buf_a;
        T* out_buf = buf_b;
        for (int it = 1; it < j - 1; ++it) {
            base(side, layout, trans_A, Op::NoTrans,
                 m, n, m, (T)1.0, in_buf, ldt, (T)0.0, out_buf, ldt);
            std::swap(in_buf, out_buf);
        }

        // Last apply writes to C with the user's alpha/beta.
        base(side, layout, trans_A, Op::NoTrans,
             m, n, m, alpha, in_buf, ldt, beta, C, ldc);

        delete[] buf_a;
        if (buf_b) delete[] buf_b;
    }
};

} // namespace RandLAPACK::linops
