#pragma once

// Internal component: pcg_inner, the instrumented conjugate-gradient kernel shared
// by IterRefineLSQ (rl_iter_refine_lsq.hh) and restarted_pcg_ne
// (rl_restarted_pcg_ne.hh). Extracted 2026-08-06 so the two least-squares drivers
// run the SAME inner solver: stagnation window with best-iterate return (2026-07-29,
// from ISAAC diagnostic evidence), warm-start entry against the true residual
// (2026-08-05), and per-solve diagnosis reporting.
//
// The kernel solves the SPD system M z = c on R^n given only a matvec callable
// apply_M(v, out). Tolerances are relative to ||c||; the caller decides what M, c,
// and the tolerance mean (IterRefineLSQ: correction equation at fixed inner_tol;
// restarted_pcg_ne: correction equation at the loose per-round drop).

#include "rl_blaspp.hh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>


namespace RandLAPACK {


/// Exit condition of one inner-CG solve. Recorded per step/round so a capped,
/// non-converged solve is distinguishable from a converged one.
enum class InnerCGStatus : int {
    Converged = 0,   ///< reached the relative-residual target
    HitCap    = 1,   ///< exhausted max_iters without reaching the target
    Breakdown = 2,   ///< p^T M p <= 0 (loss of orthogonality / non-SPD M)
    Stagnated = 3    ///< residual stopped descending; exited early with the best iterate
};

/// What one inner-CG solve did, for diagnosis.
template <typename T>
struct PCGInnerReport {
    int           iters       = 0;
    InnerCGStatus status      = InnerCGStatus::Converged;
    T             relres      = (T)0;   ///< ||r||/||c|| at exit
    T             best_relres = (T)0;   ///< smallest ||r||/||c|| seen
    int           best_iter   = 0;      ///< iteration achieving best_relres
};

/// Knobs of one inner-CG solve. Defaults match the values validated on the FEM2
/// campaigns (stagnation window 20 at 0.1% improvement).
template <typename T>
struct PCGInnerControls {
    T   tol;                            ///< stop when ||r|| <= tol * ||c||
    int max_iters;                      ///< hard iteration cap
    int stag_window      = 20;          ///< <= 0 disables the stagnation exit
    T   stag_rel_improve = (T)1e-3;     ///< drop counting as progress for the window
    bool verbose         = false;
    const char* tag      = "[PCG]";     ///< verbose-output prefix
};

/// @brief Instrumented CG on the SPD system M z = c.
///
/// @param apply_M    callable void(const T* v, T* out): out = M v. May use its own
///                   scratch; must not alias v/out.
/// @param c          right-hand side (length n).
/// @param n          system dimension.
/// @param z          solution buffer (length n). warm_start = false: initialized to
///                   0. warm_start = true: holds the starting iterate on entry; the
///                   TRUE residual c - M z is computed (one extra M apply) and CG
///                   runs from there, so a genuinely converged incoming iterate
///                   returns immediately with 0 iterations.
/// @param cg_r,cg_p,cg_Mp,cg_zbest  caller-allocated length-n workspaces.
/// @param ctl        tolerances and caps (see PCGInnerControls).
/// @param rep        filled with the solve's diagnosis (see PCGInnerReport).
/// @returns 0 on Converged/HitCap/Stagnated (the caller reads rep.status to tell
///          them apart); 1 on Breakdown.
///
/// On Stagnated and HitCap exits z holds the BEST iterate seen, not the last one:
/// best_relres <= final relres by construction, and the 07-29 `bigcap` diagnostic
/// measured the last iterate to be up to 11x worse in outer solution error.
template <typename T, typename FApplyM>
int pcg_inner(FApplyM&& apply_M, const T* c, int64_t n,
              T* z, T* cg_r, T* cg_p, T* cg_Mp, T* cg_zbest,
              const PCGInnerControls<T>& ctl, PCGInnerReport<T>& rep,
              bool warm_start = false)
{
    T c_norm = blas::nrm2(n, c, 1);
    T tol_abs = ctl.tol * c_norm;
    if (c_norm == (T)0) {
        // M is SPD, so M z = 0 has the unique solution z = 0. On a warm start the
        // incoming z is already the previous attempt's answer to the same c = 0
        // system, i.e. 0 -- so writing 0 is correct on both paths.
        std::fill(z, z + n, (T)0);
        rep.iters = 0;
        rep.status = InnerCGStatus::Converged;
        rep.relres = (T)0; rep.best_relres = (T)0; rep.best_iter = 0;
        return 0;
    }

    rep.best_iter = 0;

    if (!warm_start) {
        // Initial guess z = 0; r = c - M*z = c.
        std::fill(z, z + n, (T)0);
        std::fill(cg_zbest, cg_zbest + n, (T)0);
        std::copy(c, c + n, cg_r);
        rep.best_relres = (T)1;
    } else {
        // TRUE residual at the incoming iterate: r = c - M z. The best-iterate
        // snapshot starts at z itself, so a restart can never end worse than
        // where it began.
        apply_M(z, cg_Mp);
        for (int64_t i = 0; i < n; ++i) cg_r[i] = c[i] - cg_Mp[i];
        std::copy(z, z + n, cg_zbest);
        T r0 = blas::nrm2(n, cg_r, 1);
        rep.best_relres = r0 / c_norm;
        if (r0 <= tol_abs) {
            rep.iters  = 0;
            rep.status = InnerCGStatus::Converged;
            rep.relres = rep.best_relres;
            return 0;
        }
    }
    std::copy(cg_r, cg_r + n, cg_p);

    // Stagnation state: `stag_ref` is the residual at the last SIGNIFICANT
    // improvement (a drop of at least stag_rel_improve), and `last_improve_it` when
    // it happened. A merely-noisy decrease does not count as progress: the
    // pathological case descends by ~0 for hundreds of iterations.
    T   stag_ref        = std::numeric_limits<T>::max();
    int last_improve_it = 0;

    T rs_old = blas::dot(n, cg_r, 1, cg_r, 1);

    for (int it = 0; it < ctl.max_iters; ++it) {
        apply_M(cg_p, cg_Mp);

        T pMp = blas::dot(n, cg_p, 1, cg_Mp, 1);
        if (!(pMp > 0)) {
            rep.iters  = it;
            rep.status = InnerCGStatus::Breakdown;
            rep.relres = std::sqrt(rs_old) / c_norm;
            return 1;  // CG breakdown (loss of orthogonality / non-SPD M)
        }
        T alpha = rs_old / pMp;

        blas::axpy(n,  alpha, cg_p, 1, z, 1);      // z ← z + alpha p
        blas::axpy(n, -alpha, cg_Mp, 1, cg_r, 1);  // r ← r - alpha Mp

        T rs_new = blas::dot(n, cg_r, 1, cg_r, 1);
        T r_norm = std::sqrt(rs_new);
        T relres = r_norm / c_norm;
        if (relres < rep.best_relres) {
            rep.best_relres = relres;
            rep.best_iter   = it + 1;
            std::copy(z, z + n, cg_zbest);   // snapshot for the stagnation exit
        }
        if (relres < stag_ref * ((T)1 - ctl.stag_rel_improve)) {
            stag_ref        = relres;
            last_improve_it = it + 1;
        }

        if (ctl.verbose) {
            std::printf("%s   inner CG iter %d: ||r||/||c|| = %.4e\n",
                        ctl.tag, it + 1, (double)relres);
        }
        // Convergence is checked BEFORE stagnation: a solve that reaches the target
        // reports Converged even if its last few steps were flat.
        if (r_norm <= tol_abs) {
            rep.iters  = it + 1;
            rep.status = InnerCGStatus::Converged;
            rep.relres = relres;
            return 0;
        }
        if (ctl.stag_window > 0 &&
            (it + 1) - last_improve_it >= ctl.stag_window) {
            // Residual has flatlined. More iterations cannot reach the target, and
            // are measurably harmful to the outer solution, so stop and hand back
            // the best iterate rather than the last one.
            std::copy(cg_zbest, cg_zbest + n, z);
            rep.iters  = it + 1;
            rep.status = InnerCGStatus::Stagnated;
            rep.relres = rep.best_relres;
            if (ctl.verbose) {
                std::printf("%s   inner CG STAGNATED at iter %d "
                            "(no %.1e improvement in %d iters); returning best "
                            "iterate from iter %d, relres %.4e\n",
                            ctl.tag, it + 1, (double)ctl.stag_rel_improve,
                            ctl.stag_window, rep.best_iter, (double)rep.best_relres);
            }
            return 0;
        }

        T beta = rs_new / rs_old;
        for (int64_t i = 0; i < n; ++i) cg_p[i] = cg_r[i] + beta * cg_p[i];
        rs_old = rs_new;
    }
    // Exhausted the budget without reaching the target. Still returns 0, because a
    // capped solve is not necessarily an error for the caller -- the status records
    // it. Hand back the best iterate here too.
    std::copy(cg_zbest, cg_zbest + n, z);
    rep.iters  = ctl.max_iters;
    rep.status = InnerCGStatus::HitCap;
    rep.relres = rep.best_relres;
    return 0;
}


} // namespace RandLAPACK
