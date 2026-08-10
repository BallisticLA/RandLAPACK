#pragma once

// Public API: IterRefineLSQ — Q-less, sketch-and-precondition iterative-refinement
//                             least-squares solver.
//
// Solves min_x ||b - J x||_2 for a tall LinearOperator J using a precomputed
// triangular preconditioner R (e.g., the R-factor from CQRRT_linops on J or on
// a sketch SJ). R is treated as a right preconditioner on the normal equations.
//
// SINCE 2026-08-07 THIS CLASS IS A THIN ADAPTER over the shared restarted
// engine restarted_pcg_ne (rl_restarted_pcg_ne.hh): the FEM2 and Toeplitz
// benchmarks previously ran two separately-implemented copies of the same
// algorithm (rounds of CG on the right-preconditioned normal equations,
// separated by exact recomputations of the true residual). The 2026-08-06
// stable-residual change made even the round right-hand sides identical, so
// the outer loops were unified here. What this class adds over the raw engine
// call is the historical field names, the per-step diagnostic vectors, and the
// cold-start policy.
//
// RESTART PACING (Oleg's proposition, 2026-08-07). Each round's inner CG stops
// after its residual has dropped by the factor `round_drop` (default 1e-4)
// relative to the round's own right-hand side, rather than grinding to a fixed
// tiny tolerance: only the between-round recomputation of the true residual
// injects new information, so deep inner solves polish a stale right-hand side
// (measured 2026-08-06: inner tolerance is not the accuracy lever, outer
// rounds are). `inner_tol` survives as the ABSOLUTE inner target: a round
// whose normal-equation residual has already fallen to inner_tol * ||g_0||
// terminates immediately (the "CG still stops once below the target" guard).
// Set round_drop <= 0 to restore the legacy fixed-tolerance rounds.
//
// Reference: E. N. Epperly, M. Meier, and Y. Nakatsukasa,
//   "Fast randomized least-squares solvers can be just as accurate and stable
//    as classical direct solvers," arXiv:2406.03468v3 (2025), Algorithm 1
//    + Theorem 6.1 (master theorem on backward stability of two-step IR).

#include "rl_blaspp.hh"
#include "rl_exceptions.hh"
#include "rl_pcg_inner.hh"
#include "rl_restarted_pcg_ne.hh"
#include "../linops/rl_concepts.hh"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>


namespace RandLAPACK {


/*********************************************************/
/*                                                       */
/*                    IterRefineLSQ                      */
/*                                                       */
/*********************************************************/

/// @brief Iterative-refinement least-squares solver with right preconditioner R.
///
/// Solves min_x ||b - J x||_2 by up to `n_refine_steps` rounds of
///
///   r_i ← b - J x_i                       (true residual, working precision)
///   c_i ← R^{-T} (J^T r_i)
///   z_i ← CG on M z = c_i                 with M = R^{-T} J^T J R^{-1},
///                                         stopped after a `round_drop` residual
///                                         drop (or at the inner_tol floor)
///   x_{i+1} ← x_i + R^{-1} z_i
///
/// exiting early once ||b - J x|| / ||b|| <= outer_tol. Executed by the shared
/// engine restarted_pcg_ne; see the file header for the pacing rationale.
// InnerCGStatus lives in rl_pcg_inner.hh since the 2026-08-06 kernel extraction
// (same name, same namespace, same values -- callers are unaffected).

template <typename T>
struct IterRefineLSQ {
    // ------------- Configuration -------------
    /// ABSOLUTE inner-CG target, relative to the initial normal-equation
    /// right-hand side ||R^{-T} J^T b||: a round whose NE residual is already
    /// below inner_tol * ||g_0|| stops immediately. In legacy mode
    /// (round_drop <= 0) this is instead the per-round relative tolerance,
    /// the pre-2026-08-07 contract.
    T inner_tol;
    /// Hard cap on inner CG iterations per round. The TOTAL budget across all
    /// rounds is max_inner_iters * n_refine_steps.
    int max_inner_iters;
    /// STAGNATION EXIT (added 2026-07-29 from ISAAC diagnostic evidence).
    ///
    /// Stop the inner CG when its residual has not improved significantly for
    /// `inner_stag_window` consecutive iterations, and return the BEST iterate seen
    /// rather than the last one. Set `inner_stag_window <= 0` to disable.
    ///
    /// WHY, and why this rather than a bigger cap. On the FEM2 operator at
    /// kappa^colnorm = 1e10 the CholQR preconditioner is unusable (measured
    /// cond(J R^-1) = 7.8e4 against ~1.000 for the other methods, orthogonality error
    /// 0.56 against 1e-6). Its inner CG reached its floor at iteration 17 of each
    /// 200-iteration step and then ground out the remaining ~183 with no progress. A
    /// paired diagnostic run with a 10x larger cap settled the mechanism:
    ///
    ///     cap 200/step (400 total):  best_relres 3.483414e-09 @ iter 17, solution error 48.7
    ///     cap 2000/step (4000 total): best_relres 3.483414e-09 @ iter 17, solution error 547.0
    ///
    /// Ten times the budget left the best residual BIT-IDENTICAL and made the outer
    /// solution 11x WORSE, while spending 166 s instead of 13 s. So the cap was never the
    /// binding constraint, raising it is actively harmful, and the last iterate is worse
    /// than the best one -- hence both halves of this fix.
    ///
    /// A converging solve is unaffected: it returns Converged before the window elapses.
    /// The window is deliberately generous (and the improvement threshold small) so that
    /// slow-but-real descent is not mistaken for stagnation.
    int inner_stag_window;
    /// Relative residual drop that counts as progress for the stagnation test
    /// (default 1e-3, i.e. the residual must fall by at least 0.1%).
    T inner_stag_rel_improve;
    /// RESTART PACING (Oleg's proposition, 2026-08-07): per-round relative
    /// residual drop at which the inner CG stops and control returns to the
    /// outer loop for a true-residual restart. Default 1e-4. <= 0 restores the
    /// legacy contract (each round runs to the fixed inner_tol).
    ///
    /// This replaces the 2026-08-05 `inner_restarts` verification pass: under
    /// the restarted scheme every round IS a restart against the true
    /// residual, so a separate drift check inside the round is redundant.
    T round_drop;
    /// Maximum outer rounds (default benchmark setting: 20). With outer_tol
    /// enabled the loop reads "refine until done, capped at n_refine_steps";
    /// well-preconditioned methods exit after a few rounds, and the cap gives
    /// weakly-preconditioned configurations room to keep descending.
    int n_refine_steps;
    /// OUTER EARLY EXIT (added 2026-08-06, structure unification with
    /// restarted_pcg_ne): stop refining once the TRUE residual ||b - Jx|| / ||b||
    /// is at or below this value, checked between rounds where the residual is
    /// recomputed anyway. 0 (the default) disables the check: run exactly
    /// n_refine_steps rounds.
    T outer_tol;
    /// Optional initial iterate to refine (length n), or nullptr for the cold
    /// start that every Q-less method uses. Added 2026-08-10 so another solver's
    /// answer (Blendenpik's) can be handed to refinement, which separates
    /// preconditioner quality from solver structure in the benchmark suite.
    const T* warm_x0 = nullptr;
    /// Enable per-step / per-substep timing breakdown.
    bool timing;
    /// Print convergence info to stdout.
    bool verbose;

    // ------------- Outputs (filled by call) -------------
    /// Number of outer rounds actually executed.
    int outer_iters_done;
    /// CG iteration counts for each round.
    std::vector<int> inner_iters_per_step;
    /// Exit condition of each round's inner CG (see InnerCGStatus).
    std::vector<int> inner_status_per_step;
    /// Relative CG residual ||M z - c|| / ||c|| at exit, per round.
    std::vector<T> inner_relres_per_step;
    /// Smallest relative CG residual seen during that round, and the iteration at
    /// which it occurred. Together with inner_relres_per_step these separate the two
    /// ways a solve can burn its budget:
    ///   best_iter << iters and best ~= final  -> converged then STAGNATED (the
    ///       tolerance is below the attainable floor; more iterations cannot help)
    ///   best_iter ~= iters                    -> still descending at the cap (the
    ///       preconditioner is weak; more iterations would help)
    std::vector<T>   inner_best_relres_per_step;
    std::vector<int> inner_best_iter_per_step;
    /// Final relative residual ||b - J x|| / ||b|| (or ||b - J x|| if ||b|| == 0).
    T final_residual_norm;
    /// Per-substep wall-clock breakdown (microseconds), populated when timing == true.
    /// Entries: [0]=outer_total, [1]=inner_cg_total, [2]=trsm_total,
    ///          [3]=fwd_total, [4]=adj_total, [5]=other.
    std::vector<long> times;

    IterRefineLSQ(T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85),
                  int max_inner = 200,
                  int n_steps = 2,
                  bool timing_on = false,
                  bool verbose_on = false)
        : inner_tol(tol),
          max_inner_iters(max_inner),
          inner_stag_window(20),
          inner_stag_rel_improve((T)1e-3),
          round_drop((T)1e-4),
          n_refine_steps(n_steps),
          outer_tol((T)0),
          timing(timing_on),
          verbose(verbose_on),
          outer_iters_done(0),
          final_residual_norm((T)0)
    {}

    /// @brief Solve min ||b - J x||_2 with right preconditioner R.
    ///
    /// @tparam J_LO  A LinearOperator type (must satisfy linops::LinearOperator).
    ///
    /// @param J     Forward operator (m × n, m >= n; n_rows == m, n_cols == n).
    /// @param R     n × n upper triangular ColMajor; leading dim ldr >= n.
    /// @param ldr   Leading dimension of R.
    /// @param b     Right-hand side, length m.
    /// @param m     Number of rows of J / length of b.
    /// @param x     Solution buffer, length n. Always COLD-STARTED: any incoming
    ///              content is zeroed (the 2026-08-05 policy; the sketch-and-solve
    ///              warm start is Blendenpik-only).
    /// @param n     Number of columns of J / length of x.
    ///
    /// @returns 0 on success; nonzero on inner-CG breakdown.
    template <linops::LinearOperator J_LO>
    int call(J_LO& J, const T* R, int64_t ldr,
             const T* b, int64_t m, T* x, int64_t n)
    {
        randlapack_require(n_refine_steps >= 1)
            << "IterRefineLSQ: n_refine_steps must be >= 1";
        randlapack_require(max_inner_iters >= 1)
            << "IterRefineLSQ: max_inner_iters must be >= 1";

        using clock = std::chrono::steady_clock;
        auto t_start = clock::now();

        // Cold start (2026-08-05 policy) unless the caller supplied warm_x0.
        std::fill(x, x + n, (T)0);

        // Legacy mode (round_drop <= 0): rounds run to the fixed inner_tol
        // relative to their own right-hand side, no absolute guard.
        const bool paced = (round_drop > (T)0);
        T drop      = paced ? round_drop : inner_tol;
        T abs_guard = paced ? inner_tol : (T)0;

        PCGRoundHistory<T> hist;
        int iters_total = 0, rounds = 0;
        T final_rel = (T)0;
        long times4[4] = {0, 0, 0, 0};

        int st = restarted_pcg_ne<T>(J, m, n, R, ldr, b, x,
            /*tol=*/outer_tol,
            /*max_iters=*/max_inner_iters * n_refine_steps,
            iters_total,
            /*restart_maxit=*/max_inner_iters,
            /*restart_drop=*/drop,
            /*max_restarts=*/n_refine_steps - 1,
            &rounds,
            timing ? times4 : nullptr,
            &final_rel,
            inner_stag_window, inner_stag_rel_improve,
            abs_guard, &hist, warm_x0);

        // Republish the engine's per-round records under the historical names.
        inner_iters_per_step       = hist.iters;
        inner_status_per_step      = hist.status;
        inner_relres_per_step      = hist.relres;
        inner_best_relres_per_step = hist.best_relres;
        inner_best_iter_per_step   = hist.best_iter;
        outer_iters_done    = rounds;
        final_residual_norm = final_rel;

        if (verbose) {
            static const char* kNames[] = {"converged", "HIT CAP", "breakdown", "STAGNATED"};
            for (size_t s = 0; s < hist.iters.size(); ++s) {
                std::printf("[IR-LSQ] round %zu: inner CG %s after %d iters, "
                            "relres=%.4e (best %.4e at iter %d); LS relres %.4e\n",
                            s, kNames[hist.status[s]], hist.iters[s],
                            (double)hist.relres[s], (double)hist.best_relres[s],
                            hist.best_iter[s], (double)hist.ls_relres[s]);
            }
        }

        if (timing) {
            long total = std::chrono::duration_cast<std::chrono::microseconds>(
                             clock::now() - t_start).count();
            // Non-overlapping [outer_total, inner_cg_total, trsm, fwd, adj, other]:
            // op totals are all-inclusive; "other" subtracts the kernel wallclock
            // and the outer-only op time so nothing is counted twice.
            long op_outer = (times4[0] - hist.t_fwd_inner_us)
                          + (times4[1] - hist.t_adj_inner_us)
                          + (times4[2] - hist.t_trsm_inner_us);
            long other = total - hist.t_inner_us - op_outer;
            if (other < 0) other = 0;
            times = {total, hist.t_inner_us, times4[2], times4[0], times4[1], other};
        }

        // Historical contract: a capped or stagnated solve is not an error
        // return; only a CG breakdown is.
        return (st == 2) ? 1 : 0;
    }
};


} // namespace RandLAPACK
