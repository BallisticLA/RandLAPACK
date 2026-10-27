#pragma once

// Public API: IterRefineLSQ, iterative-refinement least squares with a precomputed
// triangular right preconditioner R (for example the R factor of a Q-less QR of J or of a
// sketch S J), or nullptr for none (unpreconditioned normal equations; ldr is then ignored).
//
// A member-configured adapter over restarted_pcg_ne.
//
// Reference: E. N. Epperly, M. Meier and Y. Nakatsukasa, "Fast randomized least-squares
// solvers can be just as accurate and stable as classical direct solvers",
// arXiv:2406.03468, Algorithm 1 and Theorem 6.1.

#include "rl_exceptions.hh"
#include "rl_pcg_inner.hh"
#include "rl_restarted_pcg_ne.hh"
#include "rl_concepts.hh"

#include <climits>
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
///   r_i <- b - J x_i                       (true residual, working precision)
///   c_i <- R^{-T} (J^T r_i)
///   z_i <- CG on M z = c_i                 with M = R^{-T} J^T J R^{-1}, stopped after a
///                                          `round_drop` residual drop (or at the inner_abs_tol floor)
///   x_{i+1} <- x_i + R^{-1} z_i
///
/// exiting early once ||b - J x|| / ||b|| <= outer_tol.
template <typename T>
struct IterRefineLSQ {
    // ------------- Configuration -------------
    /// Run-wide inner-CG floor: a round also stops once its normal-equation residual is below
    /// inner_abs_tol * ||c_0||, c_0 = R^{-T} J^T b, unlike round_drop, which is relative to
    /// each round's own c_i. 0 disables it.
    T inner_abs_tol;
    /// Inner CG iteration cap per round; the total budget is max_inner_iters * n_refine_steps.
    int max_inner_iters;
    /// Iterations without significant inner-CG progress after which a round stops and keeps
    /// its best iterate; <= 0 disables the exit.
    int inner_stag_window = RestartedPCGNEControls<T>{}.inner_stag_window;
    /// Relative drop that counts as progress, both for the inner stagnation test and for the
    /// outer floor test on the true residual.
    T stag_rel_improve = RestartedPCGNEControls<T>{}.stag_rel_improve;
    /// Per-round relative residual drop that ends a round, in (0,1).
    T round_drop = RestartedPCGNEControls<T>{}.round_drop;
    /// Maximum rounds.
    int n_refine_steps;
    /// Stop once ||b - J x|| / ||b|| <= outer_tol (finite, >= 0), checked between rounds. The
    /// default 0 is met only by an exactly zero residual, so the run then ends on the round
    /// budget, the floor exit (outer_stag_window) or the oracle.
    T outer_tol;
    /// Rounds without significant true-residual improvement that end the run at the
    /// least-squares floor (see RestartedPCGNEControls); <= 0 disables it.
    int outer_stag_window = RestartedPCGNEControls<T>{}.outer_stag_window;
    /// Optional convergence measure (see BackwardErrorOracle); active when set and be_tol >= 0.
    BackwardErrorOracle<T> oracle;
    T be_tol = RestartedPCGNEControls<T>{}.be_tol;
    /// In-round oracle poll period (see RestartedPCGNEControls); 0 = off.
    int be_poll_every = RestartedPCGNEControls<T>{}.be_poll_every;
    /// Fill `times` with the breakdown below.
    bool timing;
    /// Print one line per round to stdout.
    bool verbose;

    // ------------- Outputs (filled by call) -------------
    /// The run's totals and per-round records (see RestartedPCGNEReport).
    RestartedPCGNEReport<T> report;
    /// How the run ended.
    RestartedPCGNETermination termination_reason = RestartedPCGNETermination::iteration_budget;
    /// Wall-clock breakdown in microseconds when timing is on, empty otherwise:
    /// [0] total, [1] inner CG, [2] triangular solves, [3] J applies, [4] J^T applies,
    /// [5] other. [2]-[4] are totals including the inner CG's share; [5] = [0] - [1] -
    /// (applies and solves outside the inner CG). The oracle's time is excluded throughout.
    std::vector<long> times;

    IterRefineLSQ(T abs_tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85),
                  int max_inner = 200,
                  int n_steps = 2,
                  bool timing_on = false,
                  bool verbose_on = false)
        : inner_abs_tol(abs_tol),
          max_inner_iters(max_inner),
          n_refine_steps(n_steps),
          outer_tol((T)0),
          timing(timing_on),
          verbose(verbose_on)
    {}

    /// @brief Solve min ||b - J x||_2 with right preconditioner R.
    ///
    /// @param J     LinearOperator (m x n, m >= n).
    /// @param R     n x n upper triangular, ColMajor, leading dimension ldr >= n, or nullptr
    ///              for none (unpreconditioned normal equations; ldr is then ignored).
    /// @param ldr   Leading dimension of R.
    /// @param b     Right-hand side, length m.
    /// @param m     Rows of J.
    /// @param x     Solution, length n (output).
    /// @param n     Columns of J.
    /// @param x0    Optional initial guess, length n; nullptr = cold start.
    ///
    /// @returns 0, or 1 on an inner-CG breakdown; termination_reason says why the run ended.
    template <linops::LinearOperator J_LO>
    int call(J_LO& J, const T* R, int64_t ldr,
             const T* b, int64_t m, T* x, int64_t n, const T* x0 = nullptr)
    {
        randlapack_require(m == J.n_rows && n == J.n_cols && m >= n)
            << "IterRefineLSQ: (m, n) = (" << m << ", " << n << ") must equal J's ("
            << J.n_rows << ", " << J.n_cols << ") with m >= n";
        randlapack_require(n_refine_steps >= 1)
            << "IterRefineLSQ: n_refine_steps must be >= 1 (got " << n_refine_steps << ")";
        randlapack_require(max_inner_iters >= 1)
            << "IterRefineLSQ: max_inner_iters must be >= 1 (got " << max_inner_iters << ")";
        randlapack_require(std::isfinite(outer_tol) && outer_tol >= (T)0)
            << "IterRefineLSQ: outer_tol must be finite and >= 0 (got " << outer_tol << ")";
        randlapack_require(max_inner_iters <= INT_MAX / n_refine_steps)
            << "IterRefineLSQ: max_inner_iters * n_refine_steps overflows int";

        RestartedPCGNEControls<T> ctl;
        ctl.tol                 = outer_tol;
        ctl.max_iters           = max_inner_iters * n_refine_steps;
        ctl.max_iters_per_round = max_inner_iters;
        ctl.round_drop          = round_drop;
        ctl.max_restarts        = n_refine_steps - 1;
        ctl.inner_stag_window   = inner_stag_window;
        ctl.stag_rel_improve    = stag_rel_improve;
        ctl.inner_abs_tol       = inner_abs_tol;
        ctl.outer_stag_window   = outer_stag_window;
        ctl.oracle              = oracle;
        ctl.be_tol              = be_tol;
        ctl.be_poll_every       = be_poll_every;
        termination_reason = restarted_pcg_ne<T>(J, m, n, R, ldr, b, x, ctl, report, x0);

        if (verbose) {
            for (size_t s = 0; s < report.cg_iters.size(); ++s) {
                std::printf("[IterRefineLSQ] round %zu: inner CG %s after %d iters, "
                            "relres=%.4e (best %.4e at iter %d); LS relres %.4e\n",
                            s, to_string(report.cg_status[s]), report.cg_iters[s],
                            (double)report.cg_relres[s], (double)report.cg_best_relres[s],
                            report.cg_best_iter[s], (double)report.ls_relres[s]);
            }
        }

        if (timing) {
            const auto& t4 = report.times;   // [fwd, adj, trsv, total]
            long total = t4[3];
            long op_outer = (t4[0] - report.t_fwd_inner_us)
                          + (t4[1] - report.t_adj_inner_us)
                          + (t4[2] - report.t_trsv_inner_us);
            long other = total - report.t_inner_us - op_outer;
            if (other < 0) other = 0;
            times = {total, report.t_inner_us, t4[2], t4[0], t4[1], other};
        } else {
            times.clear();
        }

        return (termination_reason == RestartedPCGNETermination::breakdown) ? 1 : 0;
    }
};


} // namespace RandLAPACK
