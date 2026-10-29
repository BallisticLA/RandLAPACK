#pragma once

// Public API: PowerLinOp, the implicit p-th power of a square linear operator.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_exceptions.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <memory>
#include <utility>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                     PowerLinOp                        */
/*                                                       */
/*********************************************************/
// LinearOperator wrapper representing A^p for a square base operator A (dense, sparse,
// composite, solver-based, ...). Each apply chains p base applies through two ping-pong
// scratch buffers (one when p == 2, none when p == 1); A^p is never formed. The last apply
// writes C with the caller's alpha and beta. Op::Trans applies (A^T)^p = (A^p)^T.
// Side::Left only.
template <LinearOperator BaseLinOp>
struct PowerLinOp {
    using T = typename BaseLinOp::scalar_t;
    using scalar_t = T;

    BaseLinOp& base;
    const int power;
    const int64_t n_rows;
    const int64_t n_cols;

    PowerLinOp(BaseLinOp& base_op, int power)
        : base(base_op), power(power),
          n_rows(base_op.n_rows), n_cols(base_op.n_cols)
    {
        randlapack_require(base.n_rows == base.n_cols)
            << "PowerLinOp: base must be square, got n_rows=" << base.n_rows
            << " n_cols=" << base.n_cols;
        randlapack_require(power >= 1)
            << "PowerLinOp: power=" << power << " must be >= 1";
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

    // C := alpha * (base^p)^{trans_A} * op_{trans_B}(B) + beta * C, with m == k == n_rows.
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left) << "PowerLinOp supports Side::Left only";
        randlapack_require(m == n_rows) << "PowerLinOp: m=" << m << " must equal n_rows=" << n_rows;
        randlapack_require(k == n_rows) << "PowerLinOp: k=" << k << " must equal n_rows=" << n_rows;

        // Base applies go through the concept-required 12-arg overload.
        if (power == 1) {
            base(layout, trans_A, trans_B,
                 m, n, k, alpha, B, ldb, beta, C, ldc);
            return;
        }

        // Zero-initialized: a base operator may read C even when beta == 0.
        int64_t ldt = (layout == Layout::ColMajor) ? m : n;
        std::unique_ptr<T[]> buf_a(new T[(size_t)m * (size_t)n]());
        std::unique_ptr<T[]> buf_b((power >= 3) ? new T[(size_t)m * (size_t)n]() : nullptr);

        base(layout, trans_A, trans_B,
             m, n, k, (T)1.0, B, ldb, (T)0.0, buf_a.get(), ldt);

        T* in_buf  = buf_a.get();
        T* out_buf = buf_b.get();
        for (int it = 1; it < power - 1; ++it) {
            base(layout, trans_A, Op::NoTrans,
                 m, n, m, (T)1.0, in_buf, ldt, (T)0.0, out_buf, ldt);
            std::swap(in_buf, out_buf);
        }

        base(layout, trans_A, Op::NoTrans,
             m, n, m, alpha, in_buf, ldt, beta, C, ldc);
    }

    // SkOp overload: C := alpha * (base^p)^{trans_A} * op_{trans_S}(S) + beta * C. S is
    // handed to the dense apply (a dense S through its buffer, a sparse S densified), so any
    // base operator works.
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S,
        T beta, T* C, int64_t ldc)
    {
        randlapack_require(side == Side::Left) << "PowerLinOp SkOp overload supports Side::Left only";
        randlapack_require(m == n_rows) << "PowerLinOp: m=" << m << " must equal n_rows=" << n_rows;
        randlapack_require(k == n_rows) << "PowerLinOp: k=" << k << " must equal n_rows=" << n_rows;
        auto [rows_S, cols_S] = RandBLAS::dims_before_op(k, n, trans_S);
        randlapack_require(rows_S == S.n_rows && cols_S == S.n_cols)
            << "PowerLinOp: op(S) must be " << k << " x " << n << ", got S of size "
            << S.n_rows << " x " << S.n_cols;

        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            if (S.buff == nullptr)
                RandBLAS::fill_dense(S);
            // S.buff is stored in S.layout; read in the other layout it is S^T.
            Op trans_eff = trans_S;
            if (S.layout != layout)
                trans_eff = (trans_S == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            (*this)(layout, trans_A, trans_eff,
                    m, n, k, alpha, S.buff, S.dist.dim_major, beta, C, ldc);
        } else {
            if (S.nnz < 0)
                RandBLAS::fill_sparse(S);
            std::unique_ptr<T[]> S_dense(new T[(size_t)S.n_rows * (size_t)S.n_cols]);
            RandLAPACK::util::sparse_to_dense(RandBLAS::coo_view_of_skop(S), layout, S_dense.get());
            int64_t lds = (layout == Layout::ColMajor) ? S.n_rows : S.n_cols;
            (*this)(layout, trans_A, trans_S,
                    m, n, k, alpha, S_dense.get(), lds, beta, C, ldc);
        }
    }
};

} // namespace RandLAPACK::linops
