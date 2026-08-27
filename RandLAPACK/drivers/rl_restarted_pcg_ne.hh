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
//     deliberately LOOSE relative tolerance `restart_drop` (default 1e-4, i.e.
//     stop after a 1e4x residual drop; Oleg's restart pacing, 2026-08-07 --
//     was 1e-2 until then) and an inner cap of
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
//     inner-iteration budget max_iters is exhausted, when the round budget
//     max_restarts is spent, when the inner CG breaks down (indefinite H apply,
//     a sign the factor R is unusable), or when outer_stag_window consecutive
//     rounds fail to improve the true LS residual (the LS-floor exit). Each exit
//     has its own status code; see @returns.
//
// The convergence test is on the true LS residual, matching the reference; the
// inner drop tolerance only paces the restarts.

#include "rl_blaspp.hh"
#include "rl_blas2_threads.hh"
#include "rl_exceptions.hh"
#include "rl_pcg_inner.hh"
#include "../linops/rl_concepts.hh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>


namespace RandLAPACK {


/// Per-round records of one restarted_pcg_ne run, for callers that need more
/// than the aggregate outputs (IterRefineLSQ delegates here and republishes
/// these as its per-step diagnostics). All vectors have one entry per round.
/// The t_* fields separate work done INSIDE the inner CG kernel from the
/// restart loop's own residual recomputations, so a non-overlapping timing
/// breakdown can be assembled by the caller.
template <typename T>
struct PCGRoundHistory {
    std::vector<int> iters;          ///< inner CG iterations of the round
    std::vector<int> status;         ///< InnerCGStatus of the round (as int)
    std::vector<T>   relres;         ///< kernel relres of the RETURNED iterate (the
                                     ///< best-iterate relres on Stagnated/HitCap exits)
    std::vector<T>   best_relres;    ///< best kernel relres seen in the round
    std::vector<int> best_iter;      ///< iteration achieving best_relres
    std::vector<T>   ls_relres;      ///< true LS relres after the round
    long t_inner_us      = 0;        ///< wallclock inside pcg_inner
    long t_fwd_inner_us  = 0;        ///< A applies inside the kernel
    long t_adj_inner_us  = 0;        ///< A^T applies inside the kernel
    long t_trsm_inner_us = 0;        ///< trsv time inside the kernel
    void clear() {
        iters.clear(); status.clear(); relres.clear();
        best_relres.clear(); best_iter.clear(); ls_relres.clear();
        t_inner_us = t_fwd_inner_us = t_adj_inner_us = t_trsm_inner_us = 0;
    }
};


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
/// @param[in]  stag_window,stag_rel_improve  stagnation exit knobs, forwarded to
///                      the inner kernel (see PCGInnerControls).
/// @param[in]  inner_abs_tol  ABSOLUTE inner target, relative to the INITIAL
///                      normal-equation right-hand side ||g||. When > 0, a
///                      round whose NE residual has already fallen to
///                      inner_abs_tol * ||g|| stops immediately instead of
///                      grinding for a further restart_drop factor -- Oleg's
///                      "CG still terminates once below the target" guard
///                      (2026-08-07). 0 disables the guard.
/// @param[out] history  optional per-round records (see PCGRoundHistory).
/// @param[in]  x0       optional initial guess (length n). nullptr = cold start
///                      (x = 0), the historical behaviour and the policy for every
///                      Q-less method. Supplied, the solver refines THAT iterate:
///                      z is seeded with R x0 so that x = R^{-1} z reproduces it,
///                      and the first round's normal-equation residual is taken
///                      from the true residual b - A x0 rather than from g.
///                      Added 2026-08-10 so a solver's own answer (e.g.
///                      Blendenpik's) can be handed to iterative refinement,
///                      separating preconditioner quality from solver structure.
/// @param[in]  outer_stag_window  consecutive rounds without a stag_rel_improve
///                      drop of the TRUE LS residual (measured against the last
///                      significant improvement, mirroring the inner kernel) that
///                      end the loop as an LS-floor exit. <= 0 disables the outer
///                      exit. Decoupled from stag_window (2026-08-27); previously
///                      one knob silently controlled both mechanisms.
/// @returns 0 if the LS tolerance was met;
///          1 if the total inner-iteration budget was exhausted;
///          2 if the inner CG broke down or made no progress (reference flag 2);
///          3 if the outer round budget (max_restarts) was spent;
///          4 if the run ended at its LS floor (outer stagnation exit, or an
///            exactly-zero NE residual with the LS tolerance still unmet).
///          Codes 3 and 4 were folded into 1 before 2026-08-27; callers that only
///          test zero/nonzero are unaffected.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int restarted_pcg_ne(
    GLO& A, int64_t m, int64_t n,
    const T* R, int64_t ldr,
    const T* b, T* x,
    T tol, int max_iters,
    int& iters_done,
    int restart_maxit = 200,
    T restart_drop = (T)1e-4,
    int max_restarts = -1,
    int* restarts_done = nullptr,
    long* times = nullptr,
    T* final_relres = nullptr,
    int stag_window = 20,
    T stag_rel_improve = (T)1e-3,
    T inner_abs_tol = (T)0,
    PCGRoundHistory<T>* history = nullptr,
    const T* x0 = nullptr,
    int outer_stag_window = 2)
{
    randlapack_require(restart_drop > (T)0 && restart_drop < (T)1)
        << "restarted_pcg_ne: restart_drop must lie in (0,1)";
    randlapack_require(restart_maxit >= 1)
        << "restarted_pcg_ne: restart_maxit must be >= 1";

    using clock = std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    // One width for the whole solve: narrows width-capped kernels (the Toeplitz
    // FFT) to the trsv width so the inner loop pays no team re-formation per
    // width transition. No-op for operators that do not consult the context.
    SolveWidthScope solve_scope(n);
    long t_fwd = 0, t_adj = 0, t_trsm = 0;
    // Inner/outer attribution: apply_H is called both inside the CG kernel and
    // by the restart loop's residual recomputations. The flag routes each op's
    // time into the inner-only counters too, so history (when requested) can
    // report a non-overlapping breakdown.
    bool in_kernel = false;
    long t_fwd_in = 0, t_adj_in = 0, t_trsm_in = 0, t_kernel = 0;
    auto total_start = clock::now();
    if (history) history->clear();

    const bool prec = (R != nullptr);

    // Workspaces (raw T*, freed on every return path via cleanup()).
    T* z    = new T[n]();     // preconditioned solution, x = R^{-1} z
    T* g    = new T[n]();     // normal-equation right-hand side R^{-T} A^T b
    T* r_ne = new T[n]();     // NE residual in the stable form R^{-T} A^T (b - A x)
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
        long dt;
        const T* fwd_in = vin;
        if (prec) {
            std::copy(vin, vin + n, sc);                       // sc = v
            auto ts = clock::now();
            { Blas2ThreadGuard tg(n);   // cap threads: see rl_blas2_threads.hh
                blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, sc, 1);
            }
            dt = duration_cast<microseconds>(clock::now() - ts).count();
            t_trsm += dt; if (in_kernel) t_trsm_in += dt;
            fwd_in = sc;                                        // sc = R^{-1} v
        }
        auto tf = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, (T)1.0, fwd_in, n, (T)0.0, wm, m);
        dt = duration_cast<microseconds>(clock::now() - tf).count();
        t_fwd += dt; if (in_kernel) t_fwd_in += dt;
        auto ta = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, wm, m, (T)0.0, out, n);
        dt = duration_cast<microseconds>(clock::now() - ta).count();
        t_adj += dt; if (in_kernel) t_adj_in += dt;
        if (prec) {
            auto ts = clock::now();
            { Blas2ThreadGuard tg(n);   // cap threads: see rl_blas2_threads.hh
                blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, out, 1);
            }
            dt = duration_cast<microseconds>(clock::now() - ts).count();
            t_trsm += dt; if (in_kernel) t_trsm_in += dt;
        }
    };

    // True LS relative residual ||b - A x|| / ||b|| for the CURRENT z (recovers x too).
    T bnorm = blas::nrm2(m, b, 1);
    T bden = std::max(bnorm, std::numeric_limits<T>::min());
    auto recover_x_and_relres = [&]() -> T {
        std::copy(z, z + n, x);
        if (prec) {
            auto ts = clock::now();
            { Blas2ThreadGuard tg(n);   // cap threads: see rl_blas2_threads.hh
                blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, x, 1);
            }
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
            { Blas2ThreadGuard tg(n);   // cap threads: see rl_blas2_threads.hh
                blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, g, 1);
            }
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
    }

    // Cold start: z = 0, so x = 0 and the NE residual is exactly g.
    // Warm start: seed z = R x0 so that x = R^{-1} z recovers x0, then take the
    // NE residual from the TRUE residual b - A x0 in the same stable form the
    // restart loop uses (g - H z would reintroduce the cancellation that the
    // 2026-08-06 change removed).
    T relres;
    if (x0 == nullptr) {
        // z = 0, so x = 0 exactly and ||b - A x|| / ||b|| = 1 with no arithmetic:
        // the full apply the recovery lambda would burn here (one FFT-class
        // operator apply plus a trsv on a zero vector) is skipped (2026-08-27).
        std::copy(g, g + n, r_ne);
        std::fill(x, x + n, (T)0);
        relres = (bnorm > (T)0) ? (T)1 : (T)0;
    } else {
        std::copy(x0, x0 + n, z);
        if (prec) {
            auto ts = clock::now();
            { Blas2ThreadGuard tg(n);   // trmv degrades unguarded like trsv
                blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, z, 1);
            }
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
        relres = recover_x_and_relres();          // leaves wm = b - A x
        auto ta = clock::now();
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, wm, m, (T)0.0, r_ne, n);
        t_adj += duration_cast<microseconds>(clock::now() - ta).count();
        if (prec) {
            auto ts = clock::now();
            { Blas2ThreadGuard tg(n);
                blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                           blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, r_ne, 1);
            }
            t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
        }
    }

    // NE-space reference scale for the inner_abs_tol guard: the initial NE
    // right-hand side ||g||.
    T g0_norm = blas::nrm2(n, g, 1);

    iters_done = 0;
    int restarts = 0;
    int status = 1;

    // Outer stagnation state (2026-08-07; reference semantics fixed 2026-08-27):
    // when tol sits below the problem's achievable LS floor (e.g. a data noise
    // floor above the requested tolerance), every round reaches the floor and
    // further rounds burn iterations without progress. outer_stag_window
    // consecutive rounds without a significant CUMULATIVE improvement over the
    // last significant drop end the loop as an LS-floor exit (status 4). The
    // reference advances only on a significant improvement, mirroring the inner
    // kernel: the old per-round reference update killed steady slow descent
    // (e.g. 0.05% per round) after two rounds. Disabled when
    // outer_stag_window <= 0.
    T   ls_stag_ref    = relres;
    int ls_flat_rounds = 0;

    while (relres > tol && iters_done < max_iters) {
        if (max_restarts >= 0 && restarts > max_restarts) { status = 3; break; }  // round budget spent
        T ne_norm = blas::nrm2(n, r_ne, 1);
        if (ne_norm == (T)0) {
            // Exactly-zero NE residual: x is the LS minimizer. That meets the
            // caller's tolerance only if the LS residual itself does; otherwise
            // this is the LS floor, not convergence (fixed 2026-08-27).
            status = (relres <= tol) ? 0 : 4;
            break;
        }

        ++restarts;
        int inner_cap = std::min(restart_maxit, max_iters - iters_done);

        // ---- Inner CG on H dz = r_ne, from dz = 0 (shared instrumented kernel):
        //      loose target restart_drop * ||r_ne||, stagnation window + best-
        //      iterate return. Breakdown maps to the reference's terminal flag 2;
        //      Stagnated/HitCap continue to the next true-residual round. ----
        PCGInnerControls<T> ctl;
        ctl.tol       = restart_drop;
        // Absolute-target guard: once ||r_ne|| has fallen to inner_abs_tol * ||g||,
        // the round's effective target is already met (or nearly so) and grinding
        // out a further restart_drop factor is wasted work at the noise floor. The
        // kernel tolerance is relative to THIS round's RHS, so rescale.
        if (inner_abs_tol > (T)0 && ne_norm > (T)0) {
            T floor_rel = inner_abs_tol * g0_norm / ne_norm;
            if (floor_rel > ctl.tol) ctl.tol = floor_rel;
        }
        ctl.max_iters        = inner_cap;
        ctl.stag_window      = stag_window;
        ctl.stag_rel_improve = stag_rel_improve;
        ctl.tag              = "[PCG-NE]";
        PCGInnerReport<T> rep;
        in_kernel = true;
        auto tk0 = clock::now();
        int kret = pcg_inner<T>(apply_H, r_ne, n, dz, r, p, q, zb, ctl, rep,
                                /*warm_start=*/false);
        t_kernel += duration_cast<microseconds>(clock::now() - tk0).count();
        in_kernel = false;
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
            auto ta = clock::now();
            A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
              n, 1, m, (T)1.0, wm, m, (T)0.0, r_ne, n);
            t_adj += duration_cast<microseconds>(clock::now() - ta).count();
            if (prec) {
                auto ts = clock::now();
                { Blas2ThreadGuard tg(n);   // cap threads: see rl_blas2_threads.hh
                    blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                               blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, r_ne, 1);
                }
                t_trsm += duration_cast<microseconds>(clock::now() - ts).count();
            }
        }

        if (history) {
            history->iters.push_back(inner_iters);
            history->status.push_back(static_cast<int>(rep.status));
            history->relres.push_back(rep.relres);
            history->best_relres.push_back(rep.best_relres);
            history->best_iter.push_back(rep.best_iter);
            history->ls_relres.push_back(relres);
        }

        if (relres <= tol) { status = 0; break; }
        if (inner_flag == 2) { status = 2; break; }      // breakdown: R unusable
        // A zero-iteration round changes nothing, so the loop must end either
        // way; gate the MEANING on the kernel status, not the count (2026-08-27):
        // a round whose target was already met at entry is the LS floor (the NE
        // target cannot improve the true residual further), not a breakdown.
        if (inner_iters == 0) {
            status = (rep.status == InnerCGStatus::Converged) ? 4 : 2;
            break;
        }
        if (outer_stag_window > 0) {
            if (relres >= ls_stag_ref * ((T)1 - stag_rel_improve)) {
                if (++ls_flat_rounds >= outer_stag_window) {
                    status = 4; break;                   // LS floor reached: stop
                }
            } else {
                ls_flat_rounds = 0;
                ls_stag_ref = relres;   // reference advances on significant drops only
            }
        }
    }

    if (relres <= tol) status = 0;

    if (restarts_done) *restarts_done = restarts;
    if (times) { times[0] = t_fwd; times[1] = t_adj; times[2] = t_trsm;
                 times[3] = duration_cast<microseconds>(clock::now() - total_start).count(); }
    if (history) {
        history->t_inner_us      = t_kernel;
        history->t_fwd_inner_us  = t_fwd_in;
        history->t_adj_inner_us  = t_adj_in;
        history->t_trsm_inner_us = t_trsm_in;
    }
    if (final_relres) *final_relres = relres;
    cleanup();
    return status;
}


} // namespace RandLAPACK
