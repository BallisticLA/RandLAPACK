#pragma once

// Public API: IterRefineLSQ: Q-less, sketch-and-precondition iterative-refinement
//                             least-squares solver.
//
// Solves min_x ||b - J x||_2 for a tall LinearOperator J using a precomputed
// triangular preconditioner R (e.g., the R-factor from CQRRT_linops on J or on
// a sketch SJ). R is treated as a right preconditioner on the normal equations.
//
// This class is a thin adapter over the shared restarted engine
// restarted_pcg_ne (rl_restarted_pcg_ne.hh); it adds the historical field
// names, the per-step diagnostic vectors, and the cold-start policy.
//
// RESTART PACING. Each round's inner CG stops after its residual has dropped
// by the factor `round_drop` (default 1e-4) relative to the round's own
// right-hand side, rather than grinding to a fixed tiny tolerance: only the
// between-round recomputation of the true residual injects new information,
// so deep inner solves polish a stale right-hand side. `inner_tol` survives
// as the ABSOLUTE inner target: a round whose normal-equation residual has
// already fallen to inner_tol * ||g_0|| terminates immediately (the "CG
// still stops once below the target" guard). Set round_drop <= 0 to restore
// the legacy fixed-tolerance rounds.
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
// InnerCGStatus lives in rl_pcg_inner.hh (same name, same namespace, same
// values; callers are unaffected).

template <typename T>
struct IterRefineLSQ {
    // ------------- Configuration -------------
    /// ABSOLUTE inner-CG target, relative to the initial normal-equation
    /// right-hand side ||R^{-T} J^T b||: a round whose NE residual is already
    /// below inner_tol * ||g_0|| stops immediately. In legacy mode
    /// (round_drop <= 0) this is instead the per-round relative tolerance,
    /// the legacy contract.
    T inner_tol;
    /// Hard cap on inner CG iterations per round. The TOTAL budget across all
    /// rounds is max_inner_iters * n_refine_steps.
    int max_inner_iters;
    /// STAGNATION EXIT. Stop the inner CG when its residual has not improved
    /// significantly for `inner_stag_window` consecutive iterations, and
    /// return the BEST iterate seen rather than the last one (on an
    /// ill-conditioned preconditioner, raising the iteration cap does not
    /// improve the best residual but does let the last iterate drift worse).
    /// Set `inner_stag_window <= 0` to disable. A converging solve is
    /// unaffected: it returns Converged before the window elapses.
    int inner_stag_window;
    /// Relative residual drop that counts as progress for the stagnation test
    /// (default 1e-3, i.e. the residual must fall by at least 0.1%).
    T inner_stag_rel_improve;
    /// RESTART PACING: per-round relative residual drop at which the inner CG
    /// stops and control returns to the outer loop for a true-residual
    /// restart. Default 1e-4. <= 0 restores the legacy contract (each round
    /// runs to the fixed inner_tol).
    T round_drop;
    /// Maximum outer rounds (default benchmark setting: 20). With outer_tol
    /// enabled the loop reads "refine until done, capped at n_refine_steps";
    /// well-preconditioned methods exit after a few rounds, and the cap gives
    /// weakly-preconditioned configurations room to keep descending.
    int n_refine_steps;
    /// OUTER EARLY EXIT: stop refining once the TRUE residual ||b - Jx|| / ||b||
    /// is at or below this value, checked between rounds where the residual is
    /// recomputed anyway. 0 (the default) disables THIS check only; the run can
    /// still end before n_refine_steps rounds via the engine's LS-floor
    /// stagnation exit (see outer_stag_window) or a terminal inner-CG condition.
    T outer_tol;
    /// Consecutive rounds without significant true-residual improvement that end
    /// the run as an LS-floor exit (forwarded to restarted_pcg_ne; see its
    /// documentation). <= 0 disables the floor exit. Decoupled from
    /// inner_stag_window: the two mechanisms are independent.
    int outer_stag_window = 2;
    /// Optional initial iterate to refine (length n), or nullptr for the cold
    /// start that every Q-less method uses. Lets another solver's answer
    /// (Blendenpik's) be handed to refinement, separating preconditioner
    /// quality from solver structure in the benchmark suite.
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
    /// True LS relative residual after each round (engine ls_relres, kept for
    /// the per-round campaign sidecar records).
    std::vector<T> ls_relres_per_step;
    /// Final relative residual ||b - J x|| / ||b|| (or ||b - J x|| if ||b|| == 0).
    T final_residual_norm;
    /// The engine's exit status, verbatim (see restarted_pcg_ne @returns:
    /// 0 tol met, 1 budget, 2 breakdown, 3 round budget, 4 LS floor). Recorded
    /// so callers can report WHY a run ended, not just whether it converged.
    int engine_status = 1;
    /// Total inner CG iterations summed over all rounds, for callers that report a
    /// single iteration count.
    int inner_iters_total() const {
        int t = 0; for (int v : inner_iters_per_step) t += v; return t;
    }
    /// Per-substep wall-clock breakdown (microseconds), populated when timing == true.
    /// Entries: [0]=outer_total, [1]=inner_cg_total, [2]=trsm_total,
    ///          [3]=fwd_total, [4]=adj_total, [5]=other.
    /// Slot order (total first, trsm before fwd/adj) intentionally differs from
    /// the engine's times[] (rl_restarted_pcg_ne.hh: [fwd, adj, trsm, total]);
    /// this struct predates the engine unification. Benchmarks read both by
    /// name, not by matching index, so the divergence is safe to keep.
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
    /// @param x     Solution buffer, length n. Incoming content is ignored: the
    ///              start is cold (the policy for Q-less methods) unless
    ///              warm_x0 is set, in which case THAT iterate is refined
    ///              (the Blendenpik handoff).
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

        // Cold start unless the caller supplied warm_x0: no
        // zero-fill needed here, restarted_pcg_ne unconditionally overwrites x
        // (cold start or warm_x0 refinement, both handled inside the engine).

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
            abs_guard, &hist, warm_x0, outer_stag_window);
        engine_status = st;

        // Republish the engine's per-round records under the historical names.
        inner_iters_per_step       = hist.iters;
        inner_status_per_step      = hist.status;
        inner_relres_per_step      = hist.relres;
        inner_best_relres_per_step = hist.best_relres;
        inner_best_iter_per_step   = hist.best_iter;
        ls_relres_per_step         = hist.ls_relres;
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
