#pragma once

// Public API: VStackOp — vertical concatenation [Top; Bot] of two linear operators.

#include "rl_exceptions.hh"
#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                      VStackOp                          */
/*                                                       */
/*********************************************************/
// Vertical (row-wise) concatenation of two linear operators that share a column
// dimension:
//
//        A_hat = [ Top ]      (Top.n_rows + Bot.n_rows)  x  n_cols
//                [ Bot ]
//
// with  Top.n_cols == Bot.n_cols == n_cols.  Apply rules:
//
//   NoTrans:  A_hat * X = [ Top*X ; Bot*X ]      (each block written to its rows)
//   Trans:    A_hat^T * Y = Top^T*Y_1 + Bot^T*Y_2,  Y = [Y_1; Y_2] split at Top.n_rows
//
// The headline use is the mu-regularized augmented operator for Cholesky-QR
// preconditioning:  A_hat = VStackOp(A, ScaledIdentityOp(mu, n)).  Because the
// Gram path forms  A_hat^T (A_hat * E)  block by block, it yields exactly
// A^T A + mu^2 I with no change to any Cholesky-QR driver, so the resulting R is
// the regularized factor used as a right preconditioner in IterRefineLSQ.
//
// The dense path (Side::Left, trans_B == NoTrans) covers the Cholesky-QR Gram,
// IterRefineLSQ, and the orthogonality check. A sketching overload (Side::Right)
// is also provided so sketch-based drivers (CQRRT) can be handed A_hat directly:
// it is BLOCKED — for each output column block it forms W = A_hat * I_block (an
// (n_rows x b) slice, via this operator's own NoTrans) and sketches that small
// block with the full S, so no (n_rows x d) intermediate is ever materialized and
// S is never partitioned. Works for sparse and dense sketches alike.
template <LinearOperator TopOp, LinearOperator BotOp>
struct VStackOp {
    using T = typename TopOp::scalar_t;
    using scalar_t = T;

    TopOp& top;
    BotOp& bot;
    const int64_t n_rows;   // = top.n_rows + bot.n_rows
    const int64_t n_cols;   // = top.n_cols == bot.n_cols

    /// Block size for the blocked sketch path (0 -> default 256). Caps the width of
    /// the (n_rows x b) slice formed per output column block.
    int64_t block_size = 0;

    VStackOp(TopOp& top_op, BotOp& bot_op)
        : top(top_op), bot(bot_op),
          n_rows(top_op.n_rows + bot_op.n_rows),
          n_cols(top_op.n_cols)
    {
        randlapack_require(top_op.n_cols == bot_op.n_cols)
            << "VStackOp: top.n_cols=" << top_op.n_cols
            << " must match bot.n_cols=" << bot_op.n_cols;
    }

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

    void operator()(
        Side side, Layout layout,
        Op trans_self, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left) << "VStackOp supports Side::Left only";
        randlapack_require(trans_B == Op::NoTrans) << "VStackOp supports trans_B == NoTrans only";

        const int64_t r_top = top.n_rows;
        const int64_t r_bot = bot.n_rows;

        if (trans_self == Op::NoTrans) {
            // C (n_rows x n) := alpha * [Top; Bot] * B + beta * C.
            // Top fills output rows [0, r_top); Bot fills [r_top, r_top + r_bot).
            // The two row-blocks are disjoint, so each applies beta to its own region.
            randlapack_require(m == n_rows) << "VStackOp NoTrans: m=" << m
                << " must equal n_rows=" << n_rows;
            const int64_t bot_off = (layout == Layout::ColMajor) ? r_top : r_top * ldc;
            top(side, layout, Op::NoTrans, Op::NoTrans, r_top, n, k,
                alpha, B, ldb, beta, C, ldc);
            bot(side, layout, Op::NoTrans, Op::NoTrans, r_bot, n, k,
                alpha, B, ldb, beta, C + bot_off, ldc);
        } else {
            // C (n_cols x n) := alpha * [Top; Bot]^T * B + beta * C
            //                 = alpha * (Top^T * B_top + Bot^T * B_bot) + beta * C,
            // where B (n_rows x n) splits row-wise at r_top.
            randlapack_require(k == n_rows) << "VStackOp Trans: k=" << k
                << " must equal n_rows=" << n_rows;
            const int64_t bot_off = (layout == Layout::ColMajor) ? r_top : r_top * ldb;
            // First block sets C (applies beta); second block accumulates (beta = 1).
            top(side, layout, Op::Trans, Op::NoTrans, m, n, r_top,
                alpha, B, ldb, beta, C, ldc);
            bot(side, layout, Op::Trans, Op::NoTrans, m, n, r_bot,
                alpha, B + bot_off, ldb, (T)1.0, C, ldc);
        }
    }

    // Blocked sketch:  C := alpha * op(S) * [Top; Bot] + beta * C,  with S a
    // d x n_rows sketching operator (Side::Right means S multiplies on the left of
    // this operator). For each output column block we form W = [Top;Bot] * I_block
    // (an n_rows x b slice, via this operator's own NoTrans) and sketch it with the
    // full S, so the only buffers are O(n_rows x b) -- no n_rows x d intermediate,
    // and S is never partitioned. This is how CQRRT can be handed A_hat directly.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout,
        Op trans_self, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S, T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Right) << "VStackOp sketch overload supports Side::Right only";
        randlapack_require(trans_self == Op::NoTrans) << "VStackOp sketch overload supports trans_self == NoTrans only";
        randlapack_require(layout == Layout::ColMajor) << "VStackOp sketch overload supports ColMajor only";
        randlapack_require(k == n_rows) << "VStackOp sketch: k=" << k << " must equal n_rows=" << n_rows;

        const int64_t d     = m;   // sketch output dimension
        const int64_t b_blk = (block_size > 0) ? std::min(block_size, n) : std::min<int64_t>(256, n);
        T* eye = new T[n_cols * b_blk]();
        T* W   = new T[n_rows * b_blk];
        for (int64_t j = 0; j < n; j += b_blk) {
            int64_t b = std::min(b_blk, n - j);
            std::fill_n(eye, n_cols * b, (T)0);
            for (int64_t i = 0; i < b; ++i) eye[(j + i) + i * n_cols] = (T)1;
            // W = [Top; Bot] * I_block  (n_rows x b), via the dense NoTrans path.
            (*this)(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    n_rows, b, n_cols, (T)1, eye, n_cols, (T)0, W, n_rows);
            // C[:, j:j+b] := alpha * op(S) * W + beta * C[:, j:j+b]
            RandBLAS::sketch_general(Layout::ColMajor, trans_S, Op::NoTrans,
                                     d, b, n_rows, alpha, S, W, n_rows, beta, C + j * ldc, ldc);
        }
        delete[] eye;
        delete[] W;
    }
};

} // namespace RandLAPACK::linops
