#pragma once

// Public API: IterRefineLSQ — Q-less, sketch-and-precondition iterative-refinement
//                             least-squares solver.
//
// Solves min_x ||b - J x||_2 for a tall LinearOperator J using a precomputed
// triangular preconditioner R (e.g., the R-factor from CQRRT_linops on J or on
// a sketch SJ). R is treated as a right preconditioner on the normal equations,
// and two iterative-refinement steps are performed; under standard hypotheses
// two steps suffice for backward stability. The inner solver is CG on the
// symmetric-positive-definite preconditioned normal-equation matrix
//
//     M = R^{-T} J^T J R^{-1}.
//
// Reference: E. N. Epperly, M. Meier, and Y. Nakatsukasa,
//   "Fast randomized least-squares solvers can be just as accurate and stable
//    as classical direct solvers," arXiv:2406.03468v3 (2025), Algorithm 1
//    + Theorem 6.1 (master theorem on backward stability of two-step IR).

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_pcg_inner.hh"
#include "../linops/rl_concepts.hh"

#include <chrono>
#include <cmath>
#include <cstdint>
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
/// Solves min_x ||b - J x||_2 by performing `n_refine_steps` outer steps of
///
///   r_i ← b - J x_i
///   g_i ← J^T r_i
///   c_i ← R^{-T} g_i
///   z_i ← inner_solve(M, c_i)            with M = R^{-T} J^T J R^{-1}
///   x_{i+1} ← x_i + R^{-1} z_i
///
/// where R is upper triangular (n × n, ColMajor) and is held constant. The
/// inner solver is preconditioner-free conjugate gradients on the SPD matrix
/// M; each inner matvec costs two TRSMs, one J apply (forward), and one J^T
/// apply (adjoint). The default `n_refine_steps = 2` and inner CG stopping
/// rule (relative residual `< inner_tol`) suffice for backward stability
/// under the conditions of Epperly et al. (2025) Theorem 6.1.
// InnerCGStatus lives in rl_pcg_inner.hh since the 2026-08-06 kernel extraction
// (same name, same namespace, same values -- callers are unaffected).

template <typename T>
struct IterRefineLSQ {
    // ------------- Configuration -------------
    /// Inner-CG residual tolerance: stop when ||M z - c|| <= inner_tol * ||c||.
    T inner_tol;
    /// Hard cap on inner CG iterations per outer refinement step.
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
    /// SINGLE RESTART (added 2026-08-05, Max's proposition).
    ///
    /// After the inner CG terminates (converged, stagnated, or capped), restart it
    /// once from the iterate it returned: recompute the TRUE residual r = c - M z
    /// and run a fresh CG from there. Number of restarts per outer step; 0 disables.
    ///
    /// Why this helps: CG tracks its residual through the recurrence
    /// r <- r - alpha M p, which drifts from the true residual c - M z in finite
    /// precision. Both the convergence test and the stagnation window read the
    /// RECURSIVE residual, so a solve can stop at a floor (or declare convergence)
    /// that the true residual does not corroborate. The restart discards the drifted
    /// recurrence, re-measures the truth, and gives CG fresh conjugacy from the
    /// returned iterate; a genuinely converged solve costs only one extra M apply
    /// (the entry check sees the true residual already under tolerance and returns
    /// immediately with 0 iterations).
    int inner_restarts;
    /// Outer refinement steps (Algorithm 1 of Epperly et al. uses 2).
    int n_refine_steps;
    /// OUTER EARLY EXIT (added 2026-08-06, structure unification with
    /// restarted_pcg_ne): stop refining once the TRUE residual ||b - Jx|| / ||b||
    /// is at or below this value, checked at the top of every step where the
    /// residual is computed anyway. 0 (the default) disables the check and
    /// preserves the historical fixed-step behaviour: run exactly n_refine_steps.
    /// With it enabled the loop reads "refine until done, capped at
    /// n_refine_steps" -- the same contract as restarted_pcg_ne's tol + round cap.
    T outer_tol;
    /// Enable per-step / per-substep timing breakdown.
    bool timing;
    /// Print convergence info to stdout.
    bool verbose;

    // ------------- Outputs (filled by call) -------------
    /// Number of outer refinement steps actually executed.
    int outer_iters_done;
    /// CG iteration counts for each outer step.
    std::vector<int> inner_iters_per_step;
    /// Exit condition of each outer step's inner CG (see InnerCGStatus).
    std::vector<int> inner_status_per_step;
    /// Relative CG residual ||M z - c|| / ||c|| at exit, per outer step.
    std::vector<T> inner_relres_per_step;
    /// Smallest relative CG residual seen during that step, and the iteration at
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
          inner_restarts(1),
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
    /// @param x     Solution buffer, length n. On entry: initial guess (use zeros
    ///              for cold start). On exit: the refined LS solution.
    /// @param n     Number of columns of J / length of x.
    ///
    /// @returns 0 on success; nonzero on inner-CG breakdown.
    template <linops::LinearOperator J_LO>
    int call(J_LO& J, const T* R, int64_t ldr,
             const T* b, int64_t m, T* x, int64_t n)
    {
        using clock = std::chrono::steady_clock;
        auto outer_start = clock::now();

        // Separate outer-only (excluding inner CG) and inner-CG-only counters so
        // we can report a non-overlapping breakdown. The total wallclock spent
        // inside inner_cg() is tracked via t_inner_total; the per-op time
        // inside inner_cg is captured by the t_inner_* counters and would
        // otherwise be triple-counted (once in t_inner_total, once in the
        // outer totals) if we shared counters.
        long t_inner_total = 0;
        long t_outer_trsm = 0, t_outer_fwd = 0, t_outer_adj = 0;
        long t_inner_trsm = 0, t_inner_fwd = 0, t_inner_adj = 0;

        inner_iters_per_step.clear();
        inner_status_per_step.clear();
        inner_relres_per_step.clear();
        inner_best_relres_per_step.clear();
        inner_best_iter_per_step.clear();
        outer_iters_done = 0;

        // Per-call workspace buffers. Allocated once up front, reused across outer steps.
        T* r     = new T[m]();   // residual
        T* g     = new T[n]();   // J^T r
        T* c     = new T[n]();   // R^{-T} g
        T* z     = new T[n]();   // inner-solve output
        T* dx    = new T[n]();   // R^{-1} z
        T* cg_r  = new T[n]();   // CG residual
        T* cg_p  = new T[n]();   // CG search direction
        T* cg_Mp = new T[n]();   // M * p inside CG
        T* tmp_n = new T[n]();   // R^{-1} p scratch
        T* tmp_m = new T[m]();   // J * v scratch (m-length)
        // Best-iterate snapshot for the stagnation exit. One extra n-vector, and one
        // n-copy per improvement -- negligible against a CG iteration's two operator
        // applies and two triangular solves.
        T* cg_zbest = new T[n]();
        auto free_workspace = [&]() {
            delete[] r; delete[] g; delete[] c; delete[] z; delete[] dx;
            delete[] cg_r; delete[] cg_p; delete[] cg_Mp;
            delete[] tmp_n; delete[] tmp_m; delete[] cg_zbest;
        };

        T b_norm = blas::nrm2(m, b, 1);
        if (b_norm == (T)0) b_norm = (T)1;  // avoid div-by-zero in residual reporting

        for (int step = 0; step < n_refine_steps; ++step) {
            // r = b - J*x
            //   tmp_m = J*x
            auto t0 = clock::now();
            J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              m, 1, n, (T)1.0, x, n, (T)0.0, tmp_m, m);
            t_outer_fwd += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            //   r = b - tmp_m
            for (int64_t i = 0; i < m; ++i) r[i] = b[i] - tmp_m[i];

            T r_norm = blas::nrm2(m, r, 1);
            if (verbose) {
                std::printf("[IR-LSQ] step %d: ||r||/||b|| = %.4e\n", step, (double)(r_norm / b_norm));
            }
            // Outer early exit: the residual just computed IS the true one, so the
            // check costs nothing extra. Exits before spending this step's inner CG.
            if (outer_tol > (T)0 && r_norm <= outer_tol * b_norm) {
                if (verbose) {
                    std::printf("[IR-LSQ] step %d: outer_tol %.1e met; stopping early\n",
                                step, (double)outer_tol);
                }
                break;
            }

            // g = J^T r
            t0 = clock::now();
            J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
              n, 1, m, (T)1.0, r, m, (T)0.0, g, n);
            t_outer_adj += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            // c = R^{-T} g  (in-place TRSM on a copy of g)
            std::copy(g, g + n, c);
            t0 = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, c, 1);
            t_outer_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            // Inner CG on M*z = c, then `inner_restarts` restarts from the returned
            // iterate (attempt > 0 recomputes the TRUE residual c - M z and runs a
            // fresh CG; see the inner_restarts field comment for why). The per-step
            // report aggregates the attempts: iters summed, best tracked across all,
            // status/relres from the final attempt.
            InnerCGReport rep{};
            rep.best_relres = (T)1;
            int cg_status = 0;
            auto t_in0 = clock::now();
            for (int attempt = 0; attempt <= std::max(0, inner_restarts); ++attempt) {
                InnerCGReport att{};
                cg_status = inner_cg(J, R, ldr, c, n, m,
                                     z, cg_r, cg_p, cg_Mp, tmp_n, tmp_m, cg_zbest,
                                     att, t_inner_trsm, t_inner_fwd, t_inner_adj,
                                     /*warm_start=*/attempt > 0);
                if (att.best_relres < rep.best_relres) {
                    rep.best_relres = att.best_relres;
                    rep.best_iter   = rep.iters + att.best_iter;
                }
                rep.iters += att.iters;
                rep.status  = att.status;
                rep.relres  = att.relres;
                if (cg_status != 0) break;   // breakdown: no restart can help
                if (verbose && attempt < std::max(0, inner_restarts)) {
                    std::printf("[IR-LSQ] step %d: restarting inner CG from its "
                                "returned iterate (attempt %d)\n", step, attempt + 2);
                }
            }
            t_inner_total += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_in0).count();
            inner_iters_per_step.push_back(rep.iters);
            inner_status_per_step.push_back(static_cast<int>(rep.status));
            inner_relres_per_step.push_back(rep.relres);
            inner_best_relres_per_step.push_back(rep.best_relres);
            inner_best_iter_per_step.push_back(rep.best_iter);
            if (verbose) {
                static const char* kNames[] = {"converged", "HIT CAP", "breakdown", "STAGNATED"};
                std::printf("[IR-LSQ] step %d: inner CG %s after %d iters, "
                            "relres=%.4e (best %.4e at iter %d)\n",
                            step, kNames[static_cast<int>(rep.status)], rep.iters,
                            (double)rep.relres, (double)rep.best_relres, rep.best_iter);
            }
            if (cg_status != 0) {
                outer_iters_done = step;
                final_residual_norm = r_norm / b_norm;
                if (timing) populate_times(outer_start, t_inner_total,
                                            t_outer_trsm, t_outer_fwd, t_outer_adj,
                                            t_inner_trsm, t_inner_fwd, t_inner_adj);
                free_workspace();
                return cg_status;
            }

            // dx = R^{-1} z
            std::copy(z, z + n, dx);
            t0 = clock::now();
            blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, dx, 1);
            t_outer_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            // x ← x + dx
            blas::axpy(n, (T)1.0, dx, 1, x, 1);
            outer_iters_done = step + 1;
        }

        // Final residual report
        {
            auto t0 = clock::now();
            J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              m, 1, n, (T)1.0, x, n, (T)0.0, tmp_m, m);
            t_outer_fwd += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
            for (int64_t i = 0; i < m; ++i) tmp_m[i] = b[i] - tmp_m[i];
            final_residual_norm = blas::nrm2(m, tmp_m, 1) / b_norm;
        }

        if (timing) populate_times(outer_start, t_inner_total,
                                    t_outer_trsm, t_outer_fwd, t_outer_adj,
                                    t_inner_trsm, t_inner_fwd, t_inner_adj);
        free_workspace();
        return 0;
    }

public:
    /// What one inner-CG solve did, for diagnosis (see the per-step output vectors).
    /// Alias of the shared kernel's report since the 2026-08-06 extraction.
    using InnerCGReport = PCGInnerReport<T>;

private:
    // One application of M = R^{-T} J^T J R^{-1}: out = M * v. Shared by the CG
    // iteration body and the warm-start entry residual, so the two can never
    // compute M differently. Clobbers tmp_n / tmp_m.
    template <linops::LinearOperator J_LO>
    void apply_M(J_LO& J, const T* R, int64_t ldr,
                 const T* v, T* out, int64_t n, int64_t m,
                 T* tmp_n, T* tmp_m,
                 long& t_trsm, long& t_fwd, long& t_adj)
    {
        using clock = std::chrono::steady_clock;
        //   tmp_n = R^{-1} v
        std::copy(v, v + n, tmp_n);
        auto t0 = clock::now();
        blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                   blas::Op::NoTrans, blas::Diag::NonUnit, n, R, ldr, tmp_n, 1);
        t_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

        //   tmp_m = J * tmp_n
        t0 = clock::now();
        J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, (T)1.0, tmp_n, n, (T)0.0, tmp_m, m);
        t_fwd += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

        //   out = J^T * tmp_m
        t0 = clock::now();
        J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, (T)1.0, tmp_m, m, (T)0.0, out, n);
        t_adj += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

        //   out ← R^{-T} out   (in-place TRSM)
        t0 = clock::now();
        blas::trsv(blas::Layout::ColMajor, blas::Uplo::Upper,
                   blas::Op::Trans, blas::Diag::NonUnit, n, R, ldr, out, 1);
        t_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();
    }

    // Inner CG: solve M z = c, where M = R^{-T} J^T J R^{-1}, on ℝ^n.
    // Since 2026-08-06 this is a thin wrapper over the shared instrumented kernel
    // (rl_pcg_inner.hh), so IterRefineLSQ and restarted_pcg_ne run the SAME inner
    // solver: stagnation window, best-iterate return, warm-start entry against the
    // true residual. Workspaces as before; tmp_n / tmp_m feed apply_M only.
    template <linops::LinearOperator J_LO>
    int inner_cg(J_LO& J, const T* R, int64_t ldr,
                 const T* c, int64_t n, int64_t m,
                 T* z,
                 T* cg_r, T* cg_p, T* cg_Mp,
                 T* tmp_n, T* tmp_m, T* cg_zbest,
                 InnerCGReport& rep,
                 long& t_trsm, long& t_fwd, long& t_adj,
                 bool warm_start = false)
    {
        PCGInnerControls<T> ctl;
        ctl.tol              = inner_tol;
        ctl.max_iters        = max_inner_iters;
        ctl.stag_window      = inner_stag_window;
        ctl.stag_rel_improve = inner_stag_rel_improve;
        ctl.verbose          = verbose;
        ctl.tag              = "[IR-LSQ]";
        auto Mv = [&](const T* v, T* out) {
            apply_M(J, R, ldr, v, out, n, m, tmp_n, tmp_m, t_trsm, t_fwd, t_adj);
        };
        return pcg_inner<T>(Mv, c, n, z, cg_r, cg_p, cg_Mp, cg_zbest,
                            ctl, rep, warm_start);
    }

    void populate_times(std::chrono::steady_clock::time_point outer_start,
                        long t_inner_total,
                        long t_outer_trsm, long t_outer_fwd, long t_outer_adj,
                        long t_inner_trsm, long t_inner_fwd, long t_inner_adj)
    {
        using clock = std::chrono::steady_clock;
        long outer_total = std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now() - outer_start).count();
        // Non-overlapping breakdown of outer_total:
        //   inner_ex   = inner CG wallclock minus its TRSM/fwd/adj portions
        //   total_trsm = outer-loop TRSMs + inner-CG TRSMs
        //   total_fwd  = outer-loop J·v  + inner-CG J·v
        //   total_adj  = outer-loop J^T·v + inner-CG J^T·v
        //   other      = residual setup, axpy/copy/nrm2 bookkeeping, clock overhead
        long inner_ex   = t_inner_total - t_inner_trsm - t_inner_fwd - t_inner_adj;
        long total_trsm = t_outer_trsm + t_inner_trsm;
        long total_fwd  = t_outer_fwd  + t_inner_fwd;
        long total_adj  = t_outer_adj  + t_inner_adj;
        long other      = outer_total - inner_ex - total_trsm - total_fwd - total_adj;
        times = { outer_total, inner_ex, total_trsm, total_fwd, total_adj, other };
    }
};


} // namespace RandLAPACK
