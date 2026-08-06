#pragma once

// Public API: restarted_pcg_ne, restarted PCG on the right-preconditioned normal
//             equations, with an optional upper-triangular RIGHT preconditioner R.
//
// Solves  min_x ||b - A x||_2  for a tall LinearOperator A (m x n, m >= n) by
// running conjugate gradients on the preconditioned normal equations
//
//     H z = g,    H = R^{-T} A^T A R^{-1},    g = R^{-T} A^T b,
//
// and recovering x = R^{-1} z. Pass R = nullptr for the unpreconditioned normal
// equations (H = A^T A). This is the second solver of the reference Toeplitz
// least-squares benchmark (ar_sysid_toeplitz_qless_qr_benchmark.m,
// restarted_pcg_preconditioned_normal_eq), ported with the same semantics:
//
//   * Each outer round runs CG on the correction equation H dz = r_ne with a
//     deliberately LOOSE relative tolerance `restart_drop` (default 1e-2, i.e.
//     stop after a 100x residual drop) and an inner cap of
//     min(restart_maxit, max_iters - iters_so_far). Since 2026-08-06 the inner
//     solver is the shared instrumented kernel (rl_pcg_inner.hh) -- the same CG
//     that IterRefineLSQ runs, with stagnation window and best-iterate return.
//     That is a deliberate deviation from the reference's plain MATLAB pcg: a
//     round whose target is unreachable exits at its residual floor with the
//     best iterate instead of grinding out the full cap, and stagnation is not
//     treated as terminal (the next round's true-residual restart decides).
//   * After the round, z += dz and BOTH residuals are recomputed exactly: the
//     true least-squares residual b - A x, and the normal-equation residual in
//     the STABLE form R^{-T}(A^T(b - A x)) (Epperly et al. Alg. 1 line 5) rather
//     than the reference's g - H z. The two are mathematically identical, but
//     g - H z subtracts large kappa-contaminated quantities and floors the
//     achievable accuracy on hard problems (second deliberate deviation from the
//     reference; measured A/B in the round-residual comment below).
//   * The loop exits when ||b - A x|| / ||b|| <= tol (success), when the TOTAL
//     inner-iteration budget max_iters is exhausted, or when the inner CG breaks
//     down (indefinite H apply, a sign the factor R is unusable).
//
// The convergence test is on the true LS residual, matching the reference; the
// inner drop tolerance only paces the restarts.

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"
#include "rl_pcg_inner.hh"
#include "../linops/rl_concepts.hh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>


namespace RandLAPACK {


/// @brief Restarted PCG on the right-preconditioned normal equations for
///        min ||b - A x||, optional upper-triangular right preconditioner R.
///
/// @param[in]  A        tall LinearOperator (m x n), applied as A*v and A^T*u.
/// @param[in]  m,n      dimensions (m >= n).
/// @param[in]  R        upper-triangular right preconditioner (n x n, ColMajor)
///                      or nullptr for none. Must be nonsingular when supplied.
/// @param[in]  ldr      leading dimension of R.
/// @param[in]  b        right-hand side (length m).
/// @param[out] x        solution (length n).
/// @param[in]  tol      target on the true LS relative residual ||b - A x|| / ||b||.
/// @param[in]  max_iters TOTAL inner CG iteration budget, shared across restarts.
/// @param[out] iters_done total inner CG iterations actually run.
/// @param[in]  restart_maxit inner CG cap per restart (reference default 200).
/// @param[in]  restart_drop  inner CG relative residual drop per restart, in (0,1).
/// @param[in]  max_restarts  additional outer rounds allowed after the first, the
///                      IterRefineLSQ inner_restarts convention: 0 means a single
///                      round, 3 means up to four rounds, negative means unlimited
///                      (the reference behaviour, rounds bounded only by max_iters).
/// @param[out] restarts_done optional: number of outer restart rounds (may be nullptr).
/// @param[out] times    optional [fwd_us, adj_us, trsm_us, total_us] (may be nullptr).
/// @param[out] final_relres optional: the true LS relative residual at termination.
/// @returns 0 if the LS tolerance was met; 1 if the iteration budget was exhausted;
///          2 if the inner CG broke down or made no progress (reference flag 2).
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int restarted_pcg_ne(
    GLO& A, int64_t m, int64_t n,
    const T* R, int64_t ldr,
    const T* b, T* x,
    T tol, int max_iters,
    int& iters_done,
    int restart_maxit = 200,
    T restart_drop = (T)1e-2,
    int max_restarts = -1,
    int* restarts_done = nullptr,
    long* times = nullptr,
    T* final_relres = nullptr)
{
    randlapack_require(restart_drop > (T)0 && restart_drop < (T)1)
        << "restarted_pcg_ne: restart_drop must lie in (0,1)";
    randlapack_require(restart_maxit >= 1)
        << "restarted_pcg_ne: restart_maxit must be >= 1";

    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    long t_fwd = 0, t_adj = 0, t_trsm = 0;
    auto total_start = clock::now();

    const bool prec = (R != nullptr);

    // Workspaces (raw T*, freed on every return path via cleanup()).
    T* z    = new T[n]();     // preconditioned solution, x = R^{-1} z
    T* g    = new T[n]();     // normal-equation right-hand side R^{-T} A^T b
    T* r_ne = new T[n]();     // true NE residual g - H z
    T* dz   = new T[n]();     // inner CG correction
    T* p    = new T[n]();     // CG direction
    T* q    = new T[n]();     // holds H p
    T* r    = new T[n]();     // inner CG (recursive) residual
    T* zb   = new T[n]();     // kernel best-iterate snapshot
    T* sc   = new T[n]();     // trsv scratch
    T* wm   = new T[m]();     // length-m scratch (A applies)
    auto cleanup = [&]() { delete[] z; delete[] g; delete[] r_ne; delete[] dz;
                           delete[] p; delete[] q; delete[] r; delete[] zb;
                           delete[] sc; delete[] wm; };

    // H v  =  R^{-T} (A^T (A (R^{-1} v)))      (out has length n; out may not alias v)
    auto apply_H = [&](const T* vin, T* out) {
        const T* fwd_in = vin;
        if (prec) {
            std::copy(vin, vin + n, sc);                       // sc = v
            auto ts = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, sc, 1);
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
            fwd_in = sc;                                        // sc = R^{-1} v
        }
        auto tf = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, (T)1.0, fwd_in, n, (T)0.0, wm, m);
        t_fwd += duration_cast<microseconds>(clock::now() - tf).count();
        auto ta = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, wm, m, (T)0.0, out, n);
        t_adj += duration_cast<microseconds>(clock::now() - ta).count();
        if (prec) {
            auto ts = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, out, 1);
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
    };

    // True LS relative residual ||b - A x|| / ||b|| for the CURRENT z (recovers x too).
    T bnorm = blas::nrm2(m, b, 1);
    T bden = std::max(bnorm, std::numeric_limits<T>::min());
    auto recover_x_and_relres = [&]() -> T {
        std::copy(z, z + n, x);
        if (prec) {
            auto ts = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, x, 1);
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
        auto tf = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, (T)1.0, x, n, (T)0.0, wm, m);
        t_fwd += duration_cast<microseconds>(clock::now() - tf).count();
        blas::scal(m, (T)-1.0, wm, 1);
        blas::axpy(m, (T)1.0, b, 1, wm, 1);                    // wm = b - A x
        return blas::nrm2(m, wm, 1) / bden;
    };

    // g = R^{-T} A^T b.
    {
        auto ta = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, b, m, (T)0.0, g, n);
        t_adj += duration_cast<microseconds>(clock::now() - ta).count();
        if (prec) {
            auto ts = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, g, 1);
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
    }

    // z = 0, so r_ne = g and the initial x is 0.
    std::copy(g, g + n, r_ne);
    T relres = recover_x_and_relres();

    iters_done = 0;
    int restarts = 0;
    int status = 1;

    while (relres > tol && iters_done < max_iters) {
        if (max_restarts >= 0 && restarts > max_restarts) break;   // round budget spent
        T ne_norm = blas::nrm2(n, r_ne, 1);
        if (ne_norm == (T)0) { status = 0; break; }     // exact NE solution reached

        ++restarts;
        int inner_cap = std::min(restart_maxit, max_iters - iters_done);

        // ---- Inner CG on H dz = r_ne, from dz = 0 (shared instrumented kernel):
        //      loose target restart_drop * ||r_ne||, stagnation window + best-
        //      iterate return. Breakdown maps to the reference's terminal flag 2;
        //      Stagnated/HitCap continue to the next true-residual round. ----
        PCGInnerControls<T> ctl;
        ctl.tol       = restart_drop;
        ctl.max_iters = inner_cap;
        ctl.tag       = "[PCG-NE]";
        PCGInnerReport<T> rep;
        int kret = pcg_inner<T>(apply_H, r_ne, n, dz, r, p, q, zb, ctl, rep,
                                /*warm_start=*/false);
        int inner_iters = rep.iters;
        int inner_flag = (kret != 0) ? 2
                       : (rep.status == InnerCGStatus::Converged ? 0 : 1);

        blas::axpy(n, (T)1.0, dz, 1, z, 1);
        iters_done += inner_iters;

        // Recompute BOTH residuals exactly; this is the restart that removes
        // recursive-residual drift (the point of the algorithm).
        relres = recover_x_and_relres();
        // STABLE residual form (2026-08-06, measured). recover_x_and_relres left
        // wm = b - A x (the SMALL true LS residual); map it through R^{-T} A^T
        // (Epperly et al. Alg. 1 line 5) instead of the reference's g - H z, which
        // subtracts two large kappa-contaminated quantities. On the m=800 prolate
        // benchmark case (FFT operator, lambda_rel 1e-20) the two forms were A/B'd
        // with everything else identical: g - H z stalls every method's LS relres
        // at 1.75e-6 with garbage recovery (1.75e4); this form reaches the 1e-10
        // noise floor with recovery 1.9e-3, matching warm Blendenpik. Dense
        // replicas of the same problem do NOT reproduce the stall (both forms
        // converge to kappa ~ 1e11 and beyond), so the FFT apply's rounding is a
        // necessary ingredient; the benchmark is the regression harness for this.
        {
            A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
              n, 1, m, (T)1.0, wm, m, (T)0.0, r_ne, n);
            if (prec) {
                blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, r_ne, 1);
            }
        }

        if (relres <= tol) { status = 0; break; }
        if (inner_flag == 2) { status = 2; break; }      // breakdown: R unusable
        if (inner_iters == 0) { status = 2; break; }     // no progress possible
    }

    if (relres <= tol) status = 0;

    if (restarts_done) *restarts_done = restarts;
    if (times) { times[0] = t_fwd; times[1] = t_adj; times[2] = t_trsm;
                 times[3] = duration_cast<microseconds>(clock::now() - total_start).count(); }
    if (final_relres) *final_relres = relres;
    cleanup();
    return status;
}


} // namespace RandLAPACK
