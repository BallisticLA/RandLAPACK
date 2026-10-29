#pragma once

// Public API: lsqr, matrix-free LSQR (Paige and Saunders) for min_x ||b - A x||_2 with an
// optional upper-triangular right preconditioner R.
//
// A is a tall LinearOperator (m x n, m >= n), used only through A v + beta u (which reads u
// when beta != 0) and A^T u. LSQR runs on A~ = A R^{-1} (A~ = A without R) and returns
// x = R^{-1} y; with a sketch-QR preconditioner (Blendenpik) A~ is nearly orthonormal and LSQR
// converges in a few iterations independently of kappa(A). Stopping tests, on the recurrence's
// estimates for A~ and r = b - A~ y:
// S1 ||r|| <= btol ||b||, and S2 ||A~^T r|| <= atol ||A~||_F ||r||. (Paige and Saunders' S1
// also adds atol ||A~|| ||y||; this one is residual-only.)
//
// Reference: C. C. Paige and M. A. Saunders, "LSQR: An algorithm for sparse linear equations
// and sparse least squares", ACM TOMS 8(1), 1982.

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"
#include "rl_thread_caps.hh"
#include "rl_concepts.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>


namespace RandLAPACK {

namespace detail {
/// v := R^{-1} v (op = NoTrans) or R^{-T} v (op = Trans) for R n x n upper triangular, at the
/// level-2 thread cap; adds the elapsed microseconds to *t_us when t_us is non-null.
template <typename T>
inline void capped_trsv(blas::Op op, int64_t n, const T* R, int64_t ldr, T* v, long* t_us = nullptr) {
    auto ts = std::chrono::steady_clock::now();
    { Blas2ThreadGuard tg(n);
        blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper, op, blas::Diag::NonUnit, n, R, ldr, v, 1);
    }
    if (t_us)
        *t_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - ts).count();
}
} // namespace detail


/// @brief Matrix-free LSQR for min ||b - A x||, optional right preconditioner R.
///
/// @param[in]  A          tall LinearOperator (m x n), applied as A v + beta u and A^T u (see above).
/// @param[in]  m,n        dimensions (m >= n).
/// @param[in]  R          upper-triangular right preconditioner (n x n, ColMajor), or
///                        nullptr for none. Must be nonsingular when supplied.
/// @param[in]  ldr        leading dimension of R.
/// @param[in]  b          right-hand side (length m).
/// @param[out] x          solution (length n).
/// @param[in]  atol       tolerance for the S2 test.
/// @param[in]  btol       tolerance for the S1 test.
/// @param[in]  max_iters  iteration cap.
/// @param[out] iters_done iterations run.
/// @param[out] times      optional [fwd_us, adj_us, trsv_us, total_us] (may be nullptr).
/// @param[out] final_relres optional ||b - A x|| / ||b|| at termination, as estimated by the
///                        recurrence (not recomputed).
/// @param[out] stop_test  optional: 1 = S1 met, 2 = S2 met (it fires at the least-squares
///                        floor even when the residual is far above btol), 0 = cap hit.
/// @returns 0 if a stopping test was met; 1 if the iteration cap was hit.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int lsqr(
    GLO& A, int64_t m, int64_t n,
    const T* R, int64_t ldr,
    const T* b, T* x,
    T atol, T btol, int max_iters,
    int& iters_done,
    long* times = nullptr,
    T* final_relres = nullptr,
    int* stop_test = nullptr)
{
    randlapack_require(m == A.n_rows && n == A.n_cols && m >= n)
        << "lsqr: (m, n) = (" << m << ", " << n << ") must equal A's (" << A.n_rows << ", "
        << A.n_cols << ") with m >= n";
    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    SolveWidthScope solve_scope(n);
    long t_fwd = 0, t_adj = 0, t_trsv = 0;
    auto total_start = clock::now();

    const bool prec = (R != nullptr);
    // Operator outputs (v, atu) are zero-initialized: an operator may read its output even
    // when beta == 0. The others are overwritten before use.
    std::unique_ptr<T[]> u_buf(new T[m]);
    std::unique_ptr<T[]> v_buf(new T[n]()), w_buf(new T[n]), atu_buf(new T[n]()), sc_buf(new T[n]);
    T* u = u_buf.get();     // left bidiagonalization vector
    T* v = v_buf.get();     // right bidiagonalization vector
    T* w = w_buf.get();     // update direction
    T* atu = atu_buf.get(); // A~^T u
    T* sc = sc_buf.get();   // triangular-solve scratch

    // out = A~ v + beta out = A (R^{-1} v) + beta out, one operator call
    auto apply_Atilde = [&](const T* vin, T beta_out, T* out) {
        const T* fwd_in = vin;
        if (prec) {
            std::copy(vin, vin + n, sc);
            detail::capped_trsv(blas::Op::NoTrans, n, R, ldr, sc, &t_trsv);
            fwd_in = sc;
        }
        auto tf = clock::now();
        A(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, (T)1.0, fwd_in, n, beta_out, out, m);
        t_fwd += duration_cast<microseconds>(clock::now() - tf).count();
    };

    // A~^T u = R^{-T} (A^T u)
    auto apply_AtildeT = [&](const T* uin, T* out) {
        auto ta = clock::now();
        A(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, uin, m, (T)0.0, out, n);
        t_adj += duration_cast<microseconds>(clock::now() - ta).count();
        if (prec) detail::capped_trsv(blas::Op::Trans, n, R, ldr, out, &t_trsv);
    };

    auto finish = [&](int status, int test, T relres) {
        if (stop_test) *stop_test = test;
        if (final_relres) *final_relres = relres;
        if (times) {
            times[0] = t_fwd; times[1] = t_adj; times[2] = t_trsv;
            times[3] = duration_cast<microseconds>(clock::now() - total_start).count();
        }
        return status;
    };

    // Bidiagonalization init: beta u = b, alpha v = A~^T u.
    std::fill(x, x + n, (T)0);
    iters_done = 0;
    std::copy(b, b + m, u);
    T beta = blas::nrm2(m, u, 1);
    const T bnorm = beta;
    if (beta == (T)0)
        return finish(0, 1, (T)0);
    blas::scal(m, (T)1.0 / beta, u, 1);

    apply_AtildeT(u, v);
    T alpha = blas::nrm2(n, v, 1);
    if (alpha == (T)0)   // b is orthogonal to range(A): x = 0 is the minimizer
        return finish(0, 2, (T)1);
    blas::scal(n, (T)1.0 / alpha, v, 1);
    std::copy(v, v + n, w);

    T phibar = beta, rhobar = alpha;
    T anorm = alpha;   // Frobenius estimate of ||A~||, accumulated with hypot to avoid overflow
    int status = 1, test = 0;

    for (int it = 1; it <= max_iters; ++it) {
        // u = A~ v - alpha u, beta = ||u||
        apply_Atilde(v, -alpha, u);
        beta = blas::nrm2(m, u, 1);
        if (beta > (T)0) blas::scal(m, (T)1.0 / beta, u, 1);

        // v = A~^T u - beta v, alpha = ||v||
        apply_AtildeT(u, atu);
        blas::scal(n, -beta, v, 1);
        blas::axpy(n, (T)1.0, atu, 1, v, 1);
        alpha = blas::nrm2(n, v, 1);
        if (alpha > (T)0) blas::scal(n, (T)1.0 / alpha, v, 1);

        // Plane rotation
        T rho = std::hypot(rhobar, beta);
        T c   = rhobar / rho;
        T s   = beta / rho;
        T theta  = s * alpha;
        rhobar   = -c * alpha;
        T phi    = c * phibar;
        phibar   = s * phibar;

        // y += (phi/rho) w;  w = v - (theta/rho) w
        blas::axpy(n, phi / rho, w, 1, x, 1);
        blas::scal(n, -theta / rho, w, 1);
        blas::axpy(n, (T)1.0, v, 1, w, 1);

        anorm = std::hypot(std::hypot(anorm, alpha), beta);   // 2-argument hypot is scaled everywhere
        T rnorm = phibar;   // ||b - A~ y||; ||A~^T r|| = rnorm * alpha * |c|
        iters_done = it;
        if (rnorm <= btol * bnorm) { status = 0; test = 1; break; }
        if (rnorm > (T)0 && alpha * std::abs(c) <= atol * anorm) { status = 0; test = 2; break; }
    }

    // x = R^{-1} y
    if (prec) detail::capped_trsv(blas::Op::NoTrans, n, R, ldr, x, &t_trsv);
    return finish(status, test, phibar / bnorm);
}


} // namespace RandLAPACK
