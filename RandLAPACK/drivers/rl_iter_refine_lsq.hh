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
template <typename T>
struct IterRefineLSQ {
    // ------------- Configuration -------------
    /// Inner-CG residual tolerance: stop when ||M z - c|| <= inner_tol * ||c||.
    T inner_tol;
    /// Hard cap on inner CG iterations per outer refinement step.
    int max_inner_iters;
    /// Outer refinement steps (Algorithm 1 of Epperly et al. uses 2).
    int n_refine_steps;
    /// Enable per-step / per-substep timing breakdown.
    bool timing;
    /// Print convergence info to stdout.
    bool verbose;

    // ------------- Outputs (filled by call) -------------
    /// Number of outer refinement steps actually executed.
    int outer_iters_done;
    /// CG iteration counts for each outer step.
    std::vector<int> inner_iters_per_step;
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
          n_refine_steps(n_steps),
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
        auto free_workspace = [&]() {
            delete[] r; delete[] g; delete[] c; delete[] z; delete[] dx;
            delete[] cg_r; delete[] cg_p; delete[] cg_Mp;
            delete[] tmp_n; delete[] tmp_m;
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

            // g = J^T r
            t0 = clock::now();
            J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
              n, 1, m, (T)1.0, r, m, (T)0.0, g, n);
            t_outer_adj += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            // c = R^{-T} g  (in-place TRSM on a copy of g)
            std::copy(g, g + n, c);
            t0 = clock::now();
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit,
                       n, 1, (T)1.0, R, ldr, c, n);
            t_outer_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            // Inner CG on M*z = c
            int inner_iters = 0;
            auto t_in0 = clock::now();
            int cg_status = inner_cg(J, R, ldr, c, n, m,
                                     z, cg_r, cg_p, cg_Mp, tmp_n, tmp_m,
                                     inner_iters, t_inner_trsm, t_inner_fwd, t_inner_adj);
            t_inner_total += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t_in0).count();
            inner_iters_per_step.push_back(inner_iters);
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
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, 1, (T)1.0, R, ldr, dx, n);
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

private:
    // Inner CG: solve M z = c, where M = R^{-T} J^T J R^{-1}, on ℝ^n.
    // Workspaces (caller-allocated, length n unless noted): cg_r, cg_p, cg_Mp,
    // tmp_n (for R^{-1} v), tmp_m (m-length, for J v_pre).
    template <linops::LinearOperator J_LO>
    int inner_cg(J_LO& J, const T* R, int64_t ldr,
                 const T* c, int64_t n, int64_t m,
                 T* z,
                 T* cg_r, T* cg_p, T* cg_Mp,
                 T* tmp_n, T* tmp_m,
                 int& iters_out,
                 long& t_trsm, long& t_fwd, long& t_adj)
    {
        using clock = std::chrono::steady_clock;
        // Initial guess z = 0.
        std::fill(z, z + n, (T)0);

        // r = c - M*z = c (since z=0)
        std::copy(c, c + n, cg_r);
        std::copy(c, c + n, cg_p);

        T c_norm = blas::nrm2(n, c, 1);
        T tol_abs = inner_tol * c_norm;
        if (c_norm == (T)0) {
            iters_out = 0;
            return 0;
        }

        T rs_old = blas::dot(n, cg_r, 1, cg_r, 1);

        for (int it = 0; it < max_inner_iters; ++it) {
            // Mp = M * p =  R^{-T} J^T J R^{-1} p
            //   tmp_n = R^{-1} p
            std::copy(cg_p, cg_p + n, tmp_n);
            auto t0 = clock::now();
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, 1, (T)1.0, R, ldr, tmp_n, n);
            t_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            //   tmp_m = J * tmp_n
            t0 = clock::now();
            J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
              m, 1, n, (T)1.0, tmp_n, n, (T)0.0, tmp_m, m);
            t_fwd += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            //   cg_Mp = J^T * tmp_m
            t0 = clock::now();
            J(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
              n, 1, m, (T)1.0, tmp_m, m, (T)0.0, cg_Mp, n);
            t_adj += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            //   cg_Mp ← R^{-T} cg_Mp   (in-place TRSM)
            t0 = clock::now();
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit,
                       n, 1, (T)1.0, R, ldr, cg_Mp, n);
            t_trsm += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t0).count();

            T pMp = blas::dot(n, cg_p, 1, cg_Mp, 1);
            if (!(pMp > 0)) {
                iters_out = it;
                return 1;  // CG breakdown (loss of orthogonality / non-SPD M)
            }
            T alpha = rs_old / pMp;

            // z ← z + alpha p
            blas::axpy(n, alpha, cg_p, 1, z, 1);
            // r ← r - alpha Mp
            blas::axpy(n, -alpha, cg_Mp, 1, cg_r, 1);

            T rs_new = blas::dot(n, cg_r, 1, cg_r, 1);
            T r_norm = std::sqrt(rs_new);

            if (verbose) {
                std::printf("[IR-LSQ]   inner CG iter %d: ||r||/||c|| = %.4e\n",
                            it + 1, (double)(r_norm / c_norm));
            }
            if (r_norm <= tol_abs) {
                iters_out = it + 1;
                return 0;
            }

            T beta = rs_new / rs_old;
            // p ← r + beta p
            for (int64_t i = 0; i < n; ++i) cg_p[i] = cg_r[i] + beta * cg_p[i];
            rs_old = rs_new;
        }
        iters_out = max_inner_iters;
        return 0;  // hit cap; not necessarily an error — caller can inspect inner_iters
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
