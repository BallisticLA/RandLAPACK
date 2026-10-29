#pragma once

// Public API: VStackLinOp, the vertical concatenation [Top; Bot] of two linear operators.

#include "rl_exceptions.hh"
#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <memory>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                     VStackLinOp                       */
/*                                                       */
/*********************************************************/
// Row-wise concatenation A_hat = [Top; Bot] of two operators with the same column count:
//   NoTrans:  A_hat * X   = [Top * X; Bot * X]
//   Trans:    A_hat^T * Y = Top^T * Y_1 + Bot^T * Y_2, with Y split at Top.n_rows.
// The headline use is A_hat = [A; mu*I], with Bot a ScaledIdentityLinOp(n, mu): a Cholesky QR
// of A_hat forms A^T A + mu^2 I with no change to the QR driver. The sketch overload lets
// sketch-based drivers take A_hat directly. When both blocks provide materialize_cols, so does
// the stack, and the sketch overload reads its column blocks through it.
template <LinearOperator TopLinOp, LinearOperator BotLinOp>
struct VStackLinOp {
    using T = typename TopLinOp::scalar_t;
    using scalar_t = T;

    TopLinOp& top;
    BotLinOp& bot;
    const int64_t n_rows;   // = top.n_rows + bot.n_rows
    const int64_t n_cols;   // = top.n_cols == bot.n_cols

    /// Column-block width of the sketch overload's n_rows x b slices. 0 selects 256 (unlike
    /// CompositeOperator, whose 0 means unblocked).
    int64_t block_size = 0;

    VStackLinOp(TopLinOp& top_op, BotLinOp& bot_op)
        : top(top_op), bot(bot_op),
          n_rows(top_op.n_rows + bot_op.n_rows),
          n_cols(top_op.n_cols)
    {
        randlapack_require(top_op.n_cols == bot_op.n_cols)
            << "VStackLinOp: top.n_cols=" << top_op.n_cols
            << " must match bot.n_cols=" << bot_op.n_cols;
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

    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left) << "VStackLinOp supports Side::Left only";
        randlapack_require(trans_B == Op::NoTrans) << "VStackLinOp supports trans_B == NoTrans only";

        const int64_t r_top = top.n_rows;
        const int64_t r_bot = bot.n_rows;

        if (trans_A == Op::NoTrans) {
            // Top fills output rows [0, r_top) and Bot the rest, each applying beta to its rows.
            randlapack_require(m == n_rows) << "VStackLinOp NoTrans: m=" << m
                << " must equal n_rows=" << n_rows;
            randlapack_require(k == n_cols) << "VStackLinOp NoTrans: k=" << k
                << " must equal n_cols=" << n_cols;
            randlapack_require(layout == Layout::ColMajor ? ldc >= m : ldc >= n)
                << "VStackLinOp NoTrans: ldc=" << ldc << " too small for the stacked output";
            const int64_t bot_off = (layout == Layout::ColMajor) ? r_top : r_top * ldc;
            top(layout, Op::NoTrans, Op::NoTrans, r_top, n, k,
                alpha, B, ldb, beta, C, ldc);
            bot(layout, Op::NoTrans, Op::NoTrans, r_bot, n, k,
                alpha, B, ldb, beta, C + bot_off, ldc);
        } else {
            // B splits row-wise at r_top.
            randlapack_require(k == n_rows) << "VStackLinOp Trans: k=" << k
                << " must equal n_rows=" << n_rows;
            randlapack_require(m == n_cols) << "VStackLinOp Trans: m=" << m
                << " must equal n_cols=" << n_cols;
            randlapack_require(layout == Layout::ColMajor ? ldb >= k : ldb >= n)
                << "VStackLinOp Trans: ldb=" << ldb << " too small for the stacked input";
            const int64_t bot_off = (layout == Layout::ColMajor) ? r_top : r_top * ldb;
            // First block sets C (applies beta); second block accumulates (beta = 1).
            top(layout, Op::Trans, Op::NoTrans, m, n, r_top,
                alpha, B, ldb, beta, C, ldc);
            bot(layout, Op::Trans, Op::NoTrans, m, n, r_bot,
                alpha, B + bot_off, ldb, (T)1.0, C, ldc);
        }
    }

    /// Columns [col_start, col_start + col_count) of [Top; Bot] into the column-major buf
    /// (ldb >= n_rows), from both blocks' materialize_cols; present only when both have one.
    void materialize_cols(int64_t col_start, int64_t col_count, T* buf, int64_t ldb) const
        requires HasMaterializeCols<TopLinOp> && HasMaterializeCols<BotLinOp>
    {
        randlapack_require(col_start >= 0 && col_start <= n_cols
                           && col_count >= 0 && col_count <= n_cols - col_start)
            << "VStackLinOp::materialize_cols: col_start=" << col_start << ", col_count="
            << col_count << " must select columns within [0, " << n_cols << ")";
        randlapack_require(ldb >= n_rows) << "VStackLinOp::materialize_cols: ldb=" << ldb
                                          << " must be >= n_rows=" << n_rows;
        top.materialize_cols(col_start, col_count, buf, ldb);
        bot.materialize_cols(col_start, col_count, buf + top.n_rows, ldb);
    }

    // Sketch: C := alpha * op(S) * [Top; Bot] + beta * C (Side::Right: this operator is on
    // the right). Each output column block forms W = [Top; Bot] * I_block (n_rows x b), by
    // materialize_cols when both blocks provide it, and sketches it with the full S, so scratch
    // stays O(n_rows * b) for any sketch type.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S, T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Right) << "VStackLinOp sketch overload supports Side::Right only";
        randlapack_require(trans_A == Op::NoTrans) << "VStackLinOp sketch overload supports trans_A == NoTrans only";
        randlapack_require(layout == Layout::ColMajor) << "VStackLinOp sketch overload supports ColMajor only";
        randlapack_require(k == n_rows) << "VStackLinOp sketch: k=" << k << " must equal n_rows=" << n_rows;
        randlapack_require(n == n_cols) << "VStackLinOp sketch: n=" << n << " must equal n_cols=" << n_cols;
        randlapack_require(ldc >= m) << "VStackLinOp sketch: ldc=" << ldc << " must be >= m=" << m;
        auto [rows_S, cols_S] = RandBLAS::dims_before_op(m, k, trans_S);
        randlapack_require(rows_S == S.n_rows && cols_S == S.n_cols)
            << "VStackLinOp sketch: op(S) must be " << m << " x " << k << ", got S of size "
            << S.n_rows << " x " << S.n_cols;

        const int64_t d     = m;   // sketch output dimension
        const int64_t b_blk = (block_size > 0) ? std::min(block_size, n) : std::min<int64_t>(256, n);
        constexpr bool direct = HasMaterializeCols<TopLinOp> && HasMaterializeCols<BotLinOp>;
        // Sample S once here; RandBLAS would otherwise regenerate it for every block.
        if constexpr (requires { S.buff; }) {
            if (S.buff == nullptr) RandBLAS::fill_dense(S);
        } else {
            if (S.nnz < 0) RandBLAS::fill_sparse(S);
        }
        std::unique_ptr<T[]> eye_buf, W_buf;
        if constexpr (direct) {
            W_buf.reset(new T[n_rows * b_blk]);   // materialize_cols writes every entry
        } else {
            eye_buf.reset(new T[n_cols * b_blk]());
            W_buf.reset(new T[n_rows * b_blk]());   // a block operator may read C even when beta == 0
        }
        T* W = W_buf.get();
        for (int64_t j = 0; j < n; j += b_blk) {
            int64_t b = std::min(b_blk, n - j);
            // W = [Top; Bot] * I_block  (n_rows x b)
            if constexpr (direct) {
                materialize_cols(j, b, W, n_rows);
            } else {
                T* eye = eye_buf.get();
                if (j > 0)   // clear the previous block's ones
                    for (int64_t i = 0; i < b_blk; ++i) eye[(j - b_blk + i) + i * n_cols] = (T)0;
                for (int64_t i = 0; i < b; ++i) eye[(j + i) + i * n_cols] = (T)1;
                (*this)(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        n_rows, b, n_cols, (T)1, eye, n_cols, (T)0, W, n_rows);
            }
            // C[:, j:j+b] := alpha * op(S) * W + beta * C[:, j:j+b]
            RandBLAS::sketch_general(Layout::ColMajor, trans_S, Op::NoTrans,
                                     d, b, n_rows, alpha, S, W, n_rows, beta, C + j * ldc, ldc);
        }
    }
};

} // namespace RandLAPACK::linops
