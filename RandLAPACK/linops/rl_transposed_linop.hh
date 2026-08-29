#pragma once

// Public API: TransposedOp: implicit transpose view of a LinearOperator.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                    TransposedOp                       */
/*                                                       */
/*********************************************************/
// Generic LinearOperator wrapper that represents A^T for an inner operator A.
//
// Template parameter:
//   InnerOp - Any type satisfying the LinearOperator concept (dense, sparse,
//             composite, solver-based, PowerOp, another TransposedOp, ...).
//             InnerOp must implement operator() with both Op::NoTrans and
//             Op::Trans dispatch on its first matrix argument (this is part
//             of the LinearOperator concept).
//
// Semantics:
//   TransposedOp simply flips the user-supplied trans_A flag before delegating
//   to the inner op.  No data is materialized; this is a zero-cost view.
//
//     user calls T_op(..., trans_A = NoTrans, ...)  →  base(..., Trans,   ...)
//     user calls T_op(..., trans_A = Trans,   ...)  →  base(..., NoTrans, ...)
//
//   n_rows and n_cols are swapped from base so non-square wrapping works
//   (CompositeOperator etc. read these for dimension checks).
//
// Why this is fully generic:
//   The LinearOperator concept already requires both Op::NoTrans and Op::Trans
//   dispatch on the first matrix argument. Every implementation that satisfies
//   the concept supports being transposed by the right dispatch flag, so this
//   wrapper just inverts the mapping. Composite chains, sparse solvers,
//   PowerOp, and nested TransposedOps all flow through the same one-line body.
//
template <LinearOperator InnerOp>
struct TransposedOp {
    using T = typename InnerOp::scalar_t;
    using scalar_t = T;

    InnerOp& base;
    const int64_t n_rows;   // = base.n_cols
    const int64_t n_cols;   // = base.n_rows

    explicit TransposedOp(InnerOp& base_op)
        : base(base_op), n_rows(base_op.n_cols), n_cols(base_op.n_rows) {}

    // Concept-required 12-arg overload (no Side); delegates to Side::Left.
    void operator()(
        Layout layout, Op trans_self, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Left, layout, trans_self, trans_B,
                m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    // C := alpha * op_{trans_self}(base^T) * op_{trans_B}(B) + beta * C
    //
    // op_{NoTrans}(base^T) = base^T, dispatched to base() with Op::Trans.
    // op_{Trans}(base^T)   = base,   dispatched to base() with Op::NoTrans.
    //
    // side == Side::Left dispatches through the concept-required 12-arg overload
    // (every LinearOperator is guaranteed to provide it). side == Side::Right needs
    // the extended Side-taking overload, which the LinearOperator concept does not
    // guarantee; InnerOp must supply it if this wrapper is ever used on the right.
    void operator()(
        Side side, Layout layout,
        Op trans_self, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        Op base_trans = (trans_self == Op::NoTrans) ? Op::Trans : Op::NoTrans;
        if (side == Side::Left) {
            base(layout, base_trans, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
        } else {
            base(side, layout, base_trans, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
        }
    }

    // SkOp overload: delegate to base by flipping trans_self. SkOp support is not
    // part of the LinearOperator concept at all, so InnerOp must provide this
    // Side-taking SkOp overload itself; there is no concept-guaranteed fallback.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout,
        Op trans_self, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S,
        T beta, T* C, int64_t ldc)
    {
        Op base_trans = (trans_self == Op::NoTrans) ? Op::Trans : Op::NoTrans;
        base(side, layout, base_trans, trans_S,
             m, n, k, alpha, S, beta, C, ldc);
    }
};

} // namespace RandLAPACK::linops
