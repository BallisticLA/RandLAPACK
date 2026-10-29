#pragma once

// Public API: ScaledIdentityLinOp, the matrix-free scaled identity mu*I_n.

#include "rl_exceptions.hh"
#include "rl_blaspp.hh"

#include <algorithm>
#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                ScaledIdentityLinOp                    */
/*                                                       */
/*********************************************************/
// Matrix-free n x n operator mu * I_n. Its main use is as the bottom block of a VStackLinOp,
// giving the regularized operator A_hat = [A; mu*I] whose Gram is A^T A + mu^2 I.
// Symmetric, so trans_A is ignored. Side::Left and trans_B == NoTrans only.
template <typename T>
struct ScaledIdentityLinOp {
    using scalar_t = T;
    const int64_t n_rows;   // = n
    const int64_t n_cols;   // = n
    const T mu;

    ScaledIdentityLinOp(int64_t n, T mu_)
        : n_rows(n), n_cols(n), mu(mu_) {}

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

    // C := alpha * mu * B + beta * C, with k == m == n_rows.
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        (void)trans_A;  // mu*I is symmetric.
        randlapack_require(side == Side::Left) << "ScaledIdentityLinOp supports Side::Left only";
        randlapack_require(trans_B == Op::NoTrans) << "ScaledIdentityLinOp supports trans_B == NoTrans only";
        randlapack_require(k == m) << "ScaledIdentityLinOp: contracted dim k=" << k
                                   << " must equal output rows m=" << m << " (square identity)";
        randlapack_require(m == n_rows) << "ScaledIdentityLinOp: m=" << m
                                        << " must equal n_rows=" << n_rows;
        const int64_t min_ld = (layout == Layout::ColMajor) ? m : n;
        randlapack_require(ldb >= min_ld && ldc >= min_ld)
            << "ScaledIdentityLinOp: ldb=" << ldb << " and ldc=" << ldc << " must be >= " << min_ld;

        // Contiguous vectors: the columns (ColMajor) or rows (RowMajor) of B and C.
        const T am = alpha * mu;
        const int64_t nvec = (layout == Layout::ColMajor) ? n : m;
        for (int64_t v = 0; v < nvec; ++v) {
            const T* bv = B + v * ldb;
            T*       cv = C + v * ldc;
            if (beta == (T)0) {
                for (int64_t i = 0; i < min_ld; ++i) cv[i] = am * bv[i];
            } else {
                for (int64_t i = 0; i < min_ld; ++i) cv[i] = beta * cv[i] + am * bv[i];
            }
        }
    }

    /// Columns [col_start, col_start + col_count) of mu * I_n into the column-major buf
    /// (ldb >= n), written directly (see HasMaterializeCols).
    void materialize_cols(int64_t col_start, int64_t col_count, T* buf, int64_t ldb) const {
        randlapack_require(col_start >= 0 && col_start <= n_cols
                           && col_count >= 0 && col_count <= n_cols - col_start)
            << "ScaledIdentityLinOp::materialize_cols: col_start=" << col_start << ", col_count="
            << col_count << " must select columns within [0, " << n_cols << ")";
        randlapack_require(ldb >= n_rows) << "ScaledIdentityLinOp::materialize_cols: ldb=" << ldb
                                          << " must be >= n_rows=" << n_rows;
        for (int64_t j = 0; j < col_count; ++j) {
            T* col = buf + j * ldb;
            std::fill(col, col + n_rows, (T)0);
            col[col_start + j] = mu;
        }
    }
};

} // namespace RandLAPACK::linops
