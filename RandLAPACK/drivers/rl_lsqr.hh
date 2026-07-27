#pragma once

// Public API: lsqr — matrix-free LSQR (Paige & Saunders) least-squares solver,
//                    with an optional upper-triangular RIGHT preconditioner R.
//
// Solves  min_x ||b - A x||_2  for a tall LinearOperator A (m x n, m >= n) using
// the Golub-Kahan bidiagonalization LSQR recurrence. Only matrix-vector products
// A*v and A^T*u are used (via the LinearOperator), so A is never materialized.
//
// If R (upper triangular, n x n, ColMajor) is supplied, LSQR is run on the
// right-preconditioned operator  Ã = A R^{-1}  (so it solves min_y ||b - Ã y||),
// and the returned x is R^{-1} y. This is exactly the Blendenpik use case: with a
// sketch-QR preconditioner R the operator Ã is nearly orthonormal (kappa(Ã) ~ 1),
// so LSQR converges in O(log(1/eps)) iterations independent of kappa(A). Pass
// R = nullptr for the unpreconditioned solve.
//
// Reference: C. C. Paige and M. A. Saunders, "LSQR: An algorithm for sparse
//   linear equations and sparse least squares", ACM TOMS 8(1), 1982. Stopping
//   tests S1 (||r||/||b|| <= btol) and S2 (||A^T r||/(||A|| ||r||) <= atol).

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "../linops/rl_concepts.hh"

#include <chrono>
#include <cmath>
#include <cstdint>


namespace RandLAPACK {


/// @brief Matrix-free LSQR for min ||b - A x||, optional right preconditioner R.
///
/// @param[in]  A        tall LinearOperator (m x n), applied as A*v and A^T*u.
/// @param[in]  m,n      dimensions (m >= n).
/// @param[in]  R        upper-triangular right preconditioner (n x n, ColMajor)
///                      or nullptr for none. Must be nonsingular when supplied.
/// @param[in]  ldr      leading dimension of R.
/// @param[in]  b        right-hand side (length m).
/// @param[out] x        solution (length n).
/// @param[in]  atol     tolerance for the ||A^T r|| stopping test (S2).
/// @param[in]  btol     tolerance for the ||r||/||b|| stopping test (S1).
/// @param[in]  max_iters iteration cap.
/// @param[out] iters_done number of iterations actually run.
/// @param[out] times    optional [fwd_us, adj_us, trsm_us, total_us] (may be nullptr).
/// @param[out] final_relres optional: the solver's own ||b - Ã y|| / ||b|| at
///                      termination (the Paige-Saunders S1 estimate). For a right
///                      preconditioner this equals ||b - A x|| / ||b|| since Ã y = A x.
/// @returns 0 if a stopping test was met; 1 if the iteration cap was hit.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int lsqr(
    GLO& A, int64_t m, int64_t n,
    const T* R, int64_t ldr,
    const T* b, T* x,
    T atol, T btol, int max_iters,
    int& iters_done,
    long* times = nullptr,
    T* final_relres = nullptr)
{
    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    long t_fwd = 0, t_adj = 0, t_trsm = 0;
    auto total_start = clock::now();

    const bool prec = (R != nullptr);

    // Workspaces (raw T*, freed before every return).
    T* u    = new T[m]();     // left bidiag vector (length m)
    T* v    = new T[n]();     // right bidiag vector (length n)
    T* w    = new T[n]();     // update direction (length n)
    T* av   = new T[m]();     // holds Ã v (length m)
    T* atu  = new T[n]();     // holds Ã^T u (length n)
    T* sc   = new T[n]();     // trsm scratch (length n)
    auto cleanup = [&]() { delete[] u; delete[] v; delete[] w; delete[] av; delete[] atu; delete[] sc; };

    // Ã v  =  A (R^{-1} v)     (out has length m)
    auto apply_Atilde = [&](const T* vin, T* out) {
        const T* fwd_in = vin;
        if (prec) {
            std::copy(vin, vin + n, sc);                      // sc = v
            auto ts = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, sc, 1);
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
            fwd_in = sc;                                       // sc = R^{-1} v
        }
        auto tf = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, (T)1.0, fwd_in, n, (T)0.0, out, m);
        t_fwd += duration_cast<microseconds>(clock::now() - tf).count();
    };

    // Ã^T u  =  R^{-T} (A^T u)   (out has length n)
    auto apply_AtildeT = [&](const T* uin, T* out) {
        auto ta = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, uin, m, (T)0.0, out, n);
        t_adj += duration_cast<microseconds>(clock::now() - ta).count();
        if (prec) {
            auto ts = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, out, 1);
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
    };

    // ---- Bidiagonalization init:  beta u = b,  alpha v = Ã^T u ----
    std::copy(b, b + m, u);
    T beta = blas::nrm2(m, u, 1);
    T bnorm = beta;
    for (int64_t i = 0; i < n; ++i) x[i] = (T)0;
    if (beta == (T)0) { iters_done = 0; cleanup(); if (times) { times[0]=t_fwd; times[1]=t_adj; times[2]=t_trsm; times[3]=duration_cast<microseconds>(clock::now()-total_start).count(); } if (final_relres) *final_relres = (T)0; return 0; }
    blas::scal(m, (T)1.0 / beta, u, 1);

    apply_AtildeT(u, v);
    T alpha = blas::nrm2(n, v, 1);
    if (alpha > (T)0) blas::scal(n, (T)1.0 / alpha, v, 1);
    std::copy(v, v + n, w);

    T phibar = beta, rhobar = alpha;
    T anorm2 = (T)0;
    iters_done = 0;
    int status = 1;

    for (int it = 1; it <= max_iters; ++it) {
        // u ← Ã v - alpha u ;  beta = ||u|| ;  normalize
        apply_Atilde(v, av);
        blas::scal(m, -alpha, u, 1);
        blas::axpy(m, (T)1.0, av, 1, u, 1);
        beta = blas::nrm2(m, u, 1);
        if (beta > (T)0) blas::scal(m, (T)1.0 / beta, u, 1);

        // v ← Ã^T u - beta v ;  alpha = ||v|| ;  normalize
        apply_AtildeT(u, atu);
        blas::scal(n, -beta, v, 1);
        blas::axpy(n, (T)1.0, atu, 1, v, 1);
        alpha = blas::nrm2(n, v, 1);
        if (alpha > (T)0) blas::scal(n, (T)1.0 / alpha, v, 1);

        // Orthogonal transformation (plane rotation)
        T rho = std::hypot(rhobar, beta);
        T c   = rhobar / rho;
        T s   = beta / rho;
        T theta  = s * alpha;
        rhobar   = -c * alpha;
        T phi    = c * phibar;
        phibar   = s * phibar;

        // y ← y + (phi/rho) w ;  w ← v - (theta/rho) w
        blas::axpy(n, phi / rho, w, 1, x, 1);
        blas::scal(n, -theta / rho, w, 1);
        blas::axpy(n, (T)1.0, v, 1, w, 1);

        // Stopping tests (Paige-Saunders estimates)
        anorm2 += alpha * alpha + beta * beta;
        T anorm = std::sqrt(anorm2);
        T rnorm = phibar;                       // ||b - Ã y||
        T arnorm = phibar * alpha * std::abs(c); // ||Ã^T r||
        iters_done = it;
        if (rnorm <= btol * bnorm) { status = 0; break; }
        if (anorm * rnorm > (T)0 && arnorm <= atol * anorm * rnorm) { status = 0; break; }
    }

    // Undo the preconditioner: x = R^{-1} y  (y currently in x).
    if (prec) {
        auto ts = clock::now();
        blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                   blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, x, 1);
        t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
    }

    if (times) { times[0]=t_fwd; times[1]=t_adj; times[2]=t_trsm; times[3]=duration_cast<microseconds>(clock::now()-total_start).count(); }
    // phibar holds the last ||b - Ã y|| estimate; bnorm is ||b||.
    if (final_relres) *final_relres = (bnorm > (T)0) ? (phibar / bnorm) : (T)0;
    cleanup();
    return status;
}


} // namespace RandLAPACK
