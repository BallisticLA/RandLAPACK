#pragma once

// Public API: ScaledIdentityOp — matrix-free scaled identity mu*I_n.

#include "rl_exceptions.hh"
#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <cstdint>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                  ScaledIdentityOp                     */
/*                                                       */
/*********************************************************/
// Matrix-free n x n scaled identity operator mu * I_n.
//
// Its main use is as the bottom block of a VStackOp to build the regularized
// augmented operator  A_hat = [A; mu*I],  whose Gram is  A^T A + mu^2 I.  A
// Cholesky-QR of A_hat therefore yields R = chol(A^T A + mu^2 I), a regularized
// right preconditioner that is well defined even when A is rank-deficient or
// extremely ill-conditioned (see rl_iter_refine_lsq.hh).
//
// mu*I is symmetric, so Op::NoTrans and Op::Trans behave identically. Only
// Side::Left and trans_B == Op::NoTrans are supported — that is all the
// Cholesky-QR Gram path, IterRefineLSQ, and the orthogonality check ever use.
template <typename T>
struct ScaledIdentityOp {
    using scalar_t = T;
    const int64_t n_rows;   // = n
    const int64_t n_cols;   // = n
    const T mu;

    ScaledIdentityOp(int64_t n, T mu_)
        : n_rows(n), n_cols(n), mu(mu_) {}

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

    // C := alpha * (mu I) * B + beta * C   (mu I is symmetric, so trans_self is moot).
    // The identity action requires the contracted dim k to equal the output row
    // dim m (square identity), so C[i,j] = alpha*mu*B[i,j] + beta*C[i,j].
    void operator()(
        Side side, Layout layout,
        Op trans_self, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        (void)trans_self;  // mu*I is symmetric.
        randlapack_require(side == Side::Left) << "ScaledIdentityOp supports Side::Left only";
        randlapack_require(trans_B == Op::NoTrans) << "ScaledIdentityOp supports trans_B == NoTrans only";
        randlapack_require(k == m) << "ScaledIdentityOp: contracted dim k=" << k
                                   << " must equal output rows m=" << m << " (square identity)";

        const T am = alpha * mu;
        if (layout == Layout::ColMajor) {
            for (int64_t j = 0; j < n; ++j) {
                const T* bcol = B + j * ldb;
                T*       ccol = C + j * ldc;
                if (beta == (T)0) {
                    for (int64_t i = 0; i < m; ++i) ccol[i] = am * bcol[i];
                } else {
                    for (int64_t i = 0; i < m; ++i) ccol[i] = beta * ccol[i] + am * bcol[i];
                }
            }
        } else {  // RowMajor
            for (int64_t i = 0; i < m; ++i) {
                const T* brow = B + i * ldb;
                T*       crow = C + i * ldc;
                if (beta == (T)0) {
                    for (int64_t j = 0; j < n; ++j) crow[j] = am * brow[j];
                } else {
                    for (int64_t j = 0; j < n; ++j) crow[j] = beta * crow[j] + am * brow[j];
                }
            }
        }
    }
};

} // namespace RandLAPACK::linops
