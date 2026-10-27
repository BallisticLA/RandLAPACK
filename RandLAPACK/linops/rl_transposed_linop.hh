#pragma once

// Public API: TransposedLinOp, the implicit transpose of a linear operator.

#include "rl_concepts.hh"
#include "rl_exceptions.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                  TransposedLinOp                      */
/*                                                       */
/*********************************************************/
// LinearOperator wrapper representing A^T for any LinearOperator A, as a zero-cost view:
// each apply flips trans_A and delegates to A. n_rows and n_cols are A's swapped.
template <LinearOperator BaseLinOp>
struct TransposedLinOp {
    using T = typename BaseLinOp::scalar_t;
    using scalar_t = T;

    BaseLinOp& base;
    const int64_t n_rows;   // = base.n_cols
    const int64_t n_cols;   // = base.n_rows

    explicit TransposedLinOp(BaseLinOp& base_op)
        : base(base_op), n_rows(base_op.n_cols), n_cols(base_op.n_rows) {}

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

    // Side::Left:  C := alpha * op_{trans_A}(base^T) * op_{trans_B}(B) + beta * C.
    // Side::Right: C := alpha * op_{trans_B}(B) * op_{trans_A}(base^T) + beta * C, which needs
    // base's Side-taking overload (not part of the LinearOperator concept).
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        Op base_trans = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;
        if (side == Side::Left) {
            base(layout, base_trans, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
        } else if constexpr (requires { base(side, layout, base_trans, trans_B,
                                             m, n, k, alpha, B, ldb, beta, C, ldc); }) {
            base(side, layout, base_trans, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
        } else {
            randlapack_require(false)
                << "TransposedLinOp: Side::Right needs a base operator with a Side-taking overload";
        }
    }

    // SkOp overload; requires base's own Side-taking SkOp overload (not part of the concept).
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S,
        T beta, T* C, int64_t ldc)
    {
        Op base_trans = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;
        base(side, layout, base_trans, trans_S,
             m, n, k, alpha, S, beta, C, ldc);
    }
};

} // namespace RandLAPACK::linops
