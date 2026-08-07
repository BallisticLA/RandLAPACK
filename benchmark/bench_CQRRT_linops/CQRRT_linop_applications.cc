// Unified Q-less QR benchmark — IR-LSQ application, plus rspec (Algorithm 4).
//
// Pipeline:
//   1. Load matrices (FEM mode: K, M, V .mtx files; sparse mode: a single A.mtx).
//   2. (FEM only) Cholesky-factorize M = L L^T via CholSolverLinOp(half_solve=true).
//   3. (FEM only) Build J = L^{-1} K V as a doubly-nested CompositeOperator
//      J = CompositeOperator(L_inv_op, CompositeOperator(K_op, V_op)).
//   4. Run Q-less QR via one of 5 variants (CQRRT_linop, CholQR, sCholQR3,
//      sCholQR3_basic, CholQR2), selected by method_mask.
//   5. Post-processing dictated by <mode>:
//        irlsq — IterRefineLSQ from x_0 = 0 (no sketch-and-solve initial guess;
//                the only sketch is S_1 inside Q-less QR, which produces R)
//        rspec — reduced spectral approximation (Algorithm 4): Rayleigh–Ritz on
//                range(C^j V_FEM), C = L^T (K - ω M)^{-1} L. FEM-only.
//
// Usage:
//   ./CQRRT_linop_applications <prec> <outdir> <runs> <mode>
//          sparse <A.mtx> <d_factor> [nnz] [b] [compute_cond] [method_mask] [noise_level]
//   ./CQRRT_linop_applications <prec> <outdir> <runs> <mode>
//          <K.mtx> <M.mtx> <V.mtx> <d_factor> [nnz] [b] [compute_cond] [method_mask] [noise_level] [omega] [power_j]
//
// mode        = "irlsq_reg" | "rspec"
//   NOTE: these are the ONLY accepted values, and an unrecognized mode is NOT diagnosed --
//   it matches neither dispatch branch, so the run allocates its node, performs no
//   experiment, writes no CSV, and exits 0. Six SLURM scripts passing the long-removed
//   "irlsq" were archived on 2026-07-29 for exactly this reason. If you add a mode, add it
//   here and consider erroring on the unknown case.
// method_mask = bitmask of Q-less QR variants (default 0b11111 = 31)
//                 bit 0 ( 1): CQRRT_linop (TRSM_IDENTITY)
//                 bit 1 ( 2): CholQR
//                 bit 2 ( 4): sCholQR3
//                 bit 3 ( 8): sCholQR3_basic
//                 bit 4 (16): CholQR2
//                 bit 5 (32): Blendenpik  (NOT in the default mask; pass 63 to include it)
//   CQRRT_linop_bqrrp (legacy bit 6 / 64) was removed in the 2026-06-05 rework.
//
// Trailing optional args after precond_prec (irlsq / irlsq_reg), added 2026-07-27:
//   [ir_max_inner] inner-CG iteration cap per outer refinement step (default 200).
//                  With 2 outer steps this is what produced the fixed 400-iteration
//                  ceiling in earlier CSVs. Pass <= 0 to keep the default.
//   [ir_inner_tol] inner-CG relative-residual tolerance (default: eps^0.85 in the
//                  working precision, ~4.9e-14 in double). Pass < 0 to keep it.
//   [ir_round_drop] per-round inner-CG residual drop (default 1e-4; Oleg's restart
//                  pacing, 2026-08-07, replacing [ir_inner_restarts] in this slot).
//                  Each round's CG stops after this relative drop and the outer
//                  loop restarts against the TRUE residual; ir_inner_tol survives
//                  as the absolute floor at which rounds stop immediately. Pass 0
//                  for legacy fixed-tolerance rounds. Values >= 1 are rejected so
//                  stale scripts passing the old restart count fail loudly.
//   [ir_n_steps]   outer-round cap (default 20 since 2026-08-07; was 4). Under the
//                  paced scheme rounds are shallow and ir_outer_tol exits early,
//                  so strong preconditioners use a few rounds and weak ones get
//                  room to descend instead of being budget-truncated.
//   [ir_outer_tol] outer early-exit tolerance on ||b - Jx||/||b|| (default < 0 =>
//                  10*eps of the solve precision; pass 0 to always run all steps).
//                  Makes the outer loop "refine until done, capped at ir_n_steps",
//                  the same contract as the Toeplitz benchmark's pcg_ne solver.
// These exist so a diagnostic sweep can separate "CG stagnates below an unreachable
// tolerance" from "CG is still converging when the cap stops it" without a rebuild.
// The per-run answer is written to the CSV as ir_inner_capped / ir_inner_relres /
// ir_inner_best_relres / ir_inner_best_iter.
//
// Warm-start policy (2026-08-05, Max; supersedes the 07-30 2x2 ablation knobs):
// the sketch-and-solve x0 warm start is Blendenpik-only. Method mask bit 32 runs
// TWO variants -- "Blendenpik" (its own warm start, the published configuration)
// and "Blendenpik_cold" (x0 = 0) -- and IterRefineLSQ always starts from x0 = 0
// (per collaborator request 2026-06-09). The former [ir_warm_start] and
// [bp_warm_start] CLI knobs are removed; the 07-30 2x2 that motivated them is
// answered (warm x0 = Blendenpik's forward-error edge, see the 07-30 session log).

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <functional>
#include <numeric>
#include <random>

// Extras utilities (Eigen-dependent)
#include "../../extras/misc/ext_util.hh"
#include "../../extras/misc/ext_sparse_axpy.hh"
#include "../../extras/linops/ext_cholsolver_linop.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrt_bench_common.hh"

// Linops algorithms
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "rl_blendenpik.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ============================================================================
// Condition-number injection + precision-casting helpers (irlsq_reg mode)
// ============================================================================

// Geometric column-scaling diagonal d[j] = kappa^(j/(n-1)), j = 0..n-1, so the
// column-norm spread injected into J = L^{-1} K (V D) is kappa (d[0]=1, d[n-1]=kappa).
// kappa <= 1 (or n <= 1) means no scaling (all ones), i.e. native conditioning.
static std::vector<double> geometric_colscale(int64_t n, double kappa) {
    std::vector<double> d(n, 1.0);
    if (kappa > 1.0 && n > 1) {
        for (int64_t j = 0; j < n; ++j)
            d[j] = std::pow(kappa, (double)j / (double)(n - 1));
    }
    return d;
}

// Right-multiply a CSR matrix by diag(d): column j is scaled by d[j].
// In CSR, nonzero p sits in column colidxs[p], so vals[p] *= d[colidxs[p]].
template <typename T, typename sint_t>
static void scale_csr_columns(RandBLAS::sparse_data::csr::CSRMatrix<T, sint_t>& A,
                              const std::vector<double>& d) {
    for (int64_t p = 0; p < A.nnz; ++p)
        A.vals[p] = (T)((double)A.vals[p] * d[(int64_t)A.colidxs[p]]);
}

// Cast a CSR matrix to a different value precision (structure copied, values cast).
template <typename Tdst, typename Tsrc, typename sint_t>
static RandBLAS::sparse_data::csr::CSRMatrix<Tdst, sint_t>
csr_cast(const RandBLAS::sparse_data::csr::CSRMatrix<Tsrc, sint_t>& src) {
    RandBLAS::sparse_data::csr::CSRMatrix<Tdst, sint_t> dst(src.n_rows, src.n_cols);
    if (src.nnz > 0) {
        dst.reserve(src.nnz);
        std::copy(src.rowptr,  src.rowptr  + src.n_rows + 1, dst.rowptr);
        std::copy(src.colidxs, src.colidxs + src.nnz,        dst.colidxs);
        for (int64_t p = 0; p < src.nnz; ++p) dst.vals[p] = (Tdst)src.vals[p];
    }
    return dst;
}

// Unit roundoff u = eps/2 in precision P (collaborator's mu = mu_factor * u).
template <typename P>
static P unit_roundoff() { return std::numeric_limits<P>::epsilon() / (P)2; }

// ============================================================================
// Result struct (unified — sentinel values for fields irrelevant to the mode)
// ============================================================================

template <typename T>
struct bench_result {
    int64_t m, n;
    int64_t run_idx;
    std::string alg_name;
    T noise_level;

    long chol_time_us;     // FEM: shared, measured once. Sparse: 0.
    int qr_status;         // 0 on success
    long qr_time_us;       // -1 if QR failed
    int  chol_retries = 0; // CholeskyQR adaptive-shift retries (0 = clean; N/A for Blendenpik)

    // Q-factor orthogonality: ||Q^T Q - I||_F / sqrt(n), computed for all methods.
    T orth_error;

    // IR-LSQ-mode fields
    long ir_total_us;
    long ir_setup_us = 0;  // warm-start x0 build time INSIDE ir_total_us (0 = cold start)
    int  ir_outer_iters;
    int  ir_inner_iters_total;
    T    ls_residual_norm;
    T    ls_solution_error;   // -1 sentinel when undefined (FEM irlsq)

    // Inner-CG diagnosis (2026-07-27). ir_inner_iters_total alone cannot say whether a
    // solve converged or merely exhausted its budget -- both used to report success.
    //   ir_inner_capped   : 1 if ANY outer step hit max_inner (0 = all converged,
    //                       2 = a CG breakdown occurred). -1 for Blendenpik (no IR).
    //   ir_inner_relres   : worst per-step achieved ||Mz-c||/||c||.
    //   ir_inner_best_relres / ir_inner_best_iter : the smallest residual seen in the
    //       worst step and the iteration it happened at. best_iter far below the
    //       iteration count means the solve STAGNATED (tolerance below the attainable
    //       floor); best_iter near the count means it was still converging when capped.
    int  ir_inner_capped      = -1;
    T    ir_inner_relres      = (T)-1;
    T    ir_inner_best_relres = (T)-1;
    int  ir_inner_best_iter   = -1;

    // cond(J R^-1): the number that actually says whether the preconditioner works.
    // -1 when not computed (too large, or not requested).
    T    cond_precond = (T)-1;

    // irlsq_reg only: kappa(A) estimate from the regularized R diagonal
    // (max|R_ii| / min|R_ii|; floored near sigma_max/mu when sigma_min < mu).
    // -1 sentinel for the plain irlsq path.
    T    kappa_measured = (T)-1;

    // QR timing breakdown (from algo.times[])
    std::vector<long> qr_breakdown;
    std::vector<long> ir_breakdown;

    long peak_rss_kb;
    long analytical_kb;
};

// ---- CLI-configurable inner-CG controls -------------------------------------
// File-scope rather than threaded through the runners, whose signatures already take
// 14 parameters. Both are set once in main() from argv before any runner is called and
// are read-only thereafter.
//
// Why they are configurable at all (2026-07-27): the inner-CG budget was hard-coded at
// 200 per outer step, which with 2 outer steps produced the fixed 400-iteration ceiling
// seen in the CSVs, and the tolerance was fixed at eps^0.85 (~4.9e-14 in double) -- a
// relative-residual target close enough to the floating-point stagnation floor that CG
// can be unable to reach it regardless of preconditioner quality. Exposing both lets a
// diagnostic sweep separate those two effects without a rebuild.
static int    g_ir_max_inner = 200;    // <= 0 => keep the IterRefineLSQ default
static double g_ir_inner_tol = -1.0;   // <  0 => eps^0.85 in the working precision
static double g_ir_round_drop = 1e-4;  // per-round CG drop (Oleg 2026-08-07); 0 = legacy fixed-tol rounds
static int    g_ir_n_steps = 20;       // outer-round cap (2026-08-07, was 4; outer_tol exits early)
static double g_ir_outer_tol = -1.0;   // <0 => 10*eps(solve precision); 0 disables early exit
// (g_ir_warm_start / g_bp_warm_start removed 2026-08-05: IR methods are always
//  cold; Blendenpik runs as two mask-32 variants, warm and cold.)

// Summarize an IterRefineLSQ run's inner-CG behavior into the CSV fields.
//
// Reports the WORST outer step, since one capped step is enough to make the reported
// iteration count meaningless as a convergence measure. `capped` is 0 if every step
// converged, 1 if any step exhausted max_inner, 2 if any step broke down.
template <typename T>
static void record_inner_cg_diagnosis(const RandLAPACK::IterRefineLSQ<T>& ir,
                                      bench_result<T>& res) {
    if (ir.inner_status_per_step.empty()) return;
    // Rank by SEVERITY, not by the enum's numeric value. The codes are not ordered by
    // severity: Stagnated = 3 was added after Breakdown = 2, so a plain `>` comparison would
    // let a clean stagnation (which exits early WITH the best iterate) mask a genuine CG
    // breakdown in another step. Severity order, worst first:
    //   Breakdown (2) -- solver failed outright
    //   HitCap    (1) -- ran out of budget while still descending
    //   Stagnated (3) -- reached its floor and stopped; benign, best iterate returned
    //   Converged (0) -- met the tolerance
    auto severity = [](int status) -> int {
        switch (status) {
            case 2:  return 3;   // Breakdown
            case 1:  return 2;   // HitCap
            case 3:  return 1;   // Stagnated
            default: return 0;   // Converged
        }
    };
    int worst = ir.inner_status_per_step[0];
    size_t worst_idx = 0;
    for (size_t i = 1; i < ir.inner_status_per_step.size(); ++i) {
        if (severity(ir.inner_status_per_step[i]) > severity(worst)) {
            worst = ir.inner_status_per_step[i];
            worst_idx = i;
        }
    }
    // All steps converged: report the step that got the least far.
    if (worst == 0 && !ir.inner_relres_per_step.empty()) {
        for (size_t i = 0; i < ir.inner_relres_per_step.size(); ++i)
            if (ir.inner_relres_per_step[i] > ir.inner_relres_per_step[worst_idx]) worst_idx = i;
    }
    res.ir_inner_capped = worst;
    if (worst_idx < ir.inner_relres_per_step.size())
        res.ir_inner_relres = ir.inner_relres_per_step[worst_idx];
    if (worst_idx < ir.inner_best_relres_per_step.size())
        res.ir_inner_best_relres = ir.inner_best_relres_per_step[worst_idx];
    if (worst_idx < ir.inner_best_iter_per_step.size())
        res.ir_inner_best_iter = ir.inner_best_iter_per_step[worst_idx];
}

// Estimate ||A||_2 via power iteration on A^T A. O(iters * (m+n) memory) — no materialization.
template <typename T, typename GLO>
static T estimate_op_2norm(GLO& A_op, int64_t m, int64_t n, int iters = 10) {
    T* v  = new T[n];
    T* Av = new T[m];
    {
        std::mt19937 rng(7);
        std::normal_distribution<T> N01(0, 1);
        for (int64_t i = 0; i < n; ++i) v[i] = N01(rng);
    }
    T sigma = (T)0;
    for (int it = 0; it < iters; ++it) {
        T nv = blas::nrm2(n, v, 1);
        if (nv == 0) { delete[] v; delete[] Av; return (T)0; }
        blas::scal(n, (T)1.0 / nv, v, 1);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, 1, n, (T)1.0, v, n, (T)0.0, Av, m);
        sigma = blas::nrm2(m, Av, 1);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             n, 1, m, (T)1.0, Av, m, (T)0.0, v, n);
    }
    delete[] v;
    delete[] Av;
    return sigma;
}

// orth_err: ||Q^T Q - I||_F / sqrt(n), with Q = A * R^{-1} materialized explicitly.
// O(m*n + n^2) memory. Direct path: materialize Q a column block at a time via the
// linop (so the *peak* extra memory beyond Q itself stays at O(n^2 + m*b)), then
// hand Q off to testing::orthogonality_error. This matches the OLD applications
// benchmark (pre-2026-06-01-b) and the April FEM2 plots' numbers.
//
// The earlier compute_orth_error_memlite path (X = A^T A; X ← R^{-T} X R^{-1})
// was mathematically equivalent but amplified forward error by kappa(R)^2 from
// the two TRSMs against an ill-conditioned R, producing meaningless ~1e8 values
// on FEM2 where kappa(A) ~ 1e7. The direct path here amplifies only by kappa(R).
// cond_out (optional): when non-null and n <= cond_cap, also returns
// cond(A R^{-1}) = sqrt(lambda_max/lambda_min) of Q^T Q. This is the number that says
// whether the preconditioner actually works, and it costs only one extra n x n syevd on
// top of the Gram this routine already forms -- the expensive part (materializing Q) is
// shared. Added 2026-07-27: App-1 previously logged only max|R_ii|/min|R_ii|, which says
// nothing about cond(A R^{-1}), so a "too many CG iterations" report could not be
// attributed to a weak preconditioner versus an unreachable tolerance.
template <typename T, typename GLO>
static T compute_orth_error_explicit(GLO& A_op, const T* R, int64_t m, int64_t n, int64_t block_size,
                                     T* cond_out = nullptr, int64_t cond_cap = 16384) {
    int64_t b = (block_size > 0 && block_size < n) ? block_size : n;
    T* Q       = new T[m * n]();
    T* E_block = new T[n * b]();   // identity column-block scratch

    // Materialize Q = A * R^{-1} one column block at a time:
    //   E_block = I[:, j:j+b];  Q[:, j:j+b] = A_op * E_block.
    // After all blocks, Q = A. Then a single TRSM(Side::Right) gives Q = A * R^{-1}.
    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);
        std::fill(E_block, E_block + n * b, (T)0.0);
        for (int64_t j = 0; j < bk; ++j)
            E_block[(j0 + j) + j * n] = (T)1.0;
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, bk, n, (T)1.0, E_block, n, (T)0.0, Q + j0 * m, m);
    }
    delete[] E_block;

    // Q := A * R^{-1}  (TRSM amplifies error by kappa(R) only.)
    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit, m, n, (T)1.0, R, n, Q, m);

    T orth = RandLAPACK::testing::orthogonality_error<T>(Q, m, n);

    // cond(A R^{-1}) from the eigenvalues of the Gram, reusing the Q we already built.
    // Skipped above cond_cap because syevd is O(n^3) and n reaches 33024 on FEM2-large.
    if (cond_out) {
        *cond_out = (T)-1;
        if (cond_cap <= 0 || n <= cond_cap) {
            T* G = new T[n * n]();
            blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
                       n, m, (T)1.0, Q, m, (T)0.0, G, n);
            T* evals = new T[n];
            int info = lapack::syevd(lapack::Job::NoVec, blas::Uplo::Upper, n, G, n, evals);
            if (info == 0 && evals[0] > 0)
                *cond_out = std::sqrt(evals[n - 1] / evals[0]);   // syevd returns ascending
            delete[] G;
            delete[] evals;
        }
    }

    delete[] Q;
    return orth;
}

// ============================================================================
// CSV writers — IR-LSQ (preserves the column order plot_irlsq_results.m expects)
// ============================================================================

template <typename T>
static void write_irlsq_results(
    const std::string& filename,
    const std::vector<bench_result<T>>& results,
    int64_t m, int64_t n, int64_t nnz_or_zero, const std::string& input_label,
    T noise_level, T d_factor, int64_t sketch_nnz, int64_t block_size,
    int64_t method_mask)
{
    std::ofstream out(filename);
    time_t now = time(nullptr);
    out << "# Sparse IR-LSQ Benchmark results\n"
        << "# Date: " << ctime(&now)
        << "# input=" << input_label << "\n"
        << "# M=" << m << " N=" << n << " nnz=" << nnz_or_zero << "\n"
        << "# noise_level=" << noise_level << "\n"
        << "# d_factor=" << d_factor << " sketch_nnz=" << sketch_nnz
        << " block_size=" << block_size << "\n"
        << "# method_mask=" << method_mask << "\n"
#ifdef _OPENMP
        << "# OpenMP threads: " << omp_get_max_threads() << "\n"
#else
        << "# OpenMP threads: 1\n"
#endif
        ;
    out << "algorithm,run,m,n,qr_status,qr_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,"
           "ir_total_us,ir_outer_iters,ir_inner_iters_total,"
           "ls_residual_norm,ls_solution_error,"
           "ir_inner_capped,ir_inner_relres,ir_inner_best_relres,ir_inner_best_iter,cond_precond\n";
    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << "," << r.m << "," << r.n << ","
            << r.qr_status << "," << r.qr_time_us << "," << r.peak_rss_kb << "," << r.analytical_kb << ","
            << std::scientific << std::setprecision(6) << r.orth_error << ","
            << r.ir_total_us << "," << r.ir_outer_iters << "," << r.ir_inner_iters_total << ","
            << std::scientific << std::setprecision(6) << r.ls_residual_norm << ","
            << std::scientific << std::setprecision(6) << r.ls_solution_error << ","
            << r.ir_inner_capped << ","
            << std::scientific << std::setprecision(6) << r.ir_inner_relres << ","
            << std::scientific << std::setprecision(6) << r.ir_inner_best_relres << ","
            << r.ir_inner_best_iter << ","
            << std::scientific << std::setprecision(6) << r.cond_precond
            << "\n";
    }
}

template <typename T>
static void write_irlsq_breakdown(
    const std::string& filename,
    const std::vector<bench_result<T>>& results)
{
    std::ofstream out(filename);
    out << "# Sparse IR-LSQ Benchmark runtime breakdown (microseconds)\n"
        << "# QR breakdown layout depends on algorithm (see CQRRT_linop_applications.cc).\n"
        << "# IR-LSQ breakdown (6): outer_total, inner_cg_total, trsm_total, fwd_total, adj_total, other\n"
        << "#   (x_0 = 0, so ir_total_us in the results CSV matches outer_total here up to overhead)\n"
        << "algorithm,run,phase,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10\n";
    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << ",QR";
        for (size_t i = 0; i < r.qr_breakdown.size(); ++i) out << "," << r.qr_breakdown[i];
        for (size_t i = r.qr_breakdown.size(); i < 11; ++i) out << ",0";
        out << "\n";
        out << r.alg_name << "," << r.run_idx << ",IR";
        for (size_t i = 0; i < r.ir_breakdown.size(); ++i) out << "," << r.ir_breakdown[i];
        for (size_t i = r.ir_breakdown.size(); i < 11; ++i) out << ",0";
        out << "\n";
    }
}

// ============================================================================
// CSV writer — RSPEC (reduced spectral approximation)
// ============================================================================

template <typename T>
struct rspec_result {
    int64_t m;
    int64_t n;
    int64_t run_idx;
    std::string alg_name;
    int qr_status;
    long qr_time_us;
    long peak_rss_kb;
    long analytical_kb;
    long factor_time_us;
    long rspec_total_us;
    T orth_error;                 // ||Q^T Q - I||_F / sqrt(n), Q = V_app R^{-1}
    std::vector<long> qr_breakdown;   // Q-less QR breakdown (driver times[], same layout as irlsq)
    std::vector<long> rr_breakdown;   // Rayleigh-Ritz post-processing: [orth, rr_build, syevd, resid] (us)
    std::vector<T> top_eigvals;
    std::vector<T> top_residuals;
};

template <typename T>
static void write_rspec_breakdown(
    const std::string& filename,
    const std::vector<rspec_result<T>>& results)
{
    std::ofstream out(filename);
    out << "# RSPEC runtime breakdown (microseconds)\n"
        << "# QR breakdown layout depends on algorithm (see CQRRT_linop_applications.cc).\n"
        << "# RR breakdown (4): orth_error, rayleigh_ritz_build, syevd, ritz_residuals\n"
        << "algorithm,run,phase,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10\n";
    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << ",QR";
        for (size_t i = 0; i < r.qr_breakdown.size(); ++i) out << "," << r.qr_breakdown[i];
        for (size_t i = r.qr_breakdown.size(); i < 11; ++i) out << ",0";
        out << "\n";
        out << r.alg_name << "," << r.run_idx << ",RR";
        for (size_t i = 0; i < r.rr_breakdown.size(); ++i) out << "," << r.rr_breakdown[i];
        for (size_t i = r.rr_breakdown.size(); i < 11; ++i) out << ",0";
        out << "\n";
    }
}

template <typename T>
static void write_rspec_csv(
    const std::string& filename,
    const std::vector<rspec_result<T>>& results,
    int64_t m, int64_t n, int num_runs,
    const std::string& K_file, const std::string& M_file, const std::string& V_file,
    double omega, int64_t power_j,
    int64_t sketch_nnz, int64_t block_size,
    int64_t method_mask, int top_k)
{
    std::ofstream out(filename);
    time_t now = time(nullptr);
    out << "# RSPEC (reduced spectral approximation) results\n"
        << "# Date: " << ctime(&now)
        << "# Matrix dimensions: m=" << m << " n=" << n << "\n"
        << "# Runs per algorithm: " << num_runs << "\n"
#ifdef _OPENMP
        << "# OpenMP threads: " << omp_get_max_threads() << "\n"
#else
        << "# OpenMP threads: 1\n"
#endif
        << "# K_file: " << K_file << "\n"
        << "# M_file: " << M_file << "\n"
        << "# V_file: " << V_file << "\n"
        << "# omega: " << omega << "\n"
        << "# power_j: " << power_j << "\n"
        << "# sketch_nnz: " << sketch_nnz << "\n"
        << "# block_size: " << block_size << "\n"
        << "# method_mask: " << method_mask << "\n"
        << "# top_k: " << top_k << "\n";

    out << "algorithm,run,m,n,omega,power_j,qr_status,qr_time_us,peak_rss_kb,analytical_kb,"
           "factor_time_us,rspec_total_us,orth_error";
    for (int i = 0; i < top_k; ++i) out << ",eig_" << i;
    for (int i = 0; i < top_k; ++i) out << ",resid_" << i;
    out << "\n";

    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << "," << r.m << "," << r.n << ","
            << omega << "," << power_j << ","
            << r.qr_status << "," << r.qr_time_us << ","
            << r.peak_rss_kb << "," << r.analytical_kb << ","
            << r.factor_time_us << "," << r.rspec_total_us << ","
            << std::scientific << std::setprecision(6) << r.orth_error;
        for (int i = 0; i < top_k; ++i) {
            T v = (i < (int)r.top_eigvals.size()) ? r.top_eigvals[i]
                                                  : std::numeric_limits<T>::quiet_NaN();
            out << "," << std::scientific << std::setprecision(8) << v;
        }
        for (int i = 0; i < top_k; ++i) {
            T v = (i < (int)r.top_residuals.size()) ? r.top_residuals[i]
                                                    : std::numeric_limits<T>::quiet_NaN();
            out << "," << std::scientific << std::setprecision(6) << v;
        }
        out << "\n";
    }
}

// ============================================================================
// Console summary
// ============================================================================

template <typename T>
static void print_irlsq_summary(const bench_result<T>& r) {
    std::printf("\n  [%s] Run %ld (noise=%.3f):\n",
                r.alg_name.c_str(), (long)r.run_idx, (double)r.noise_level);
    if (r.qr_status != 0) {
        std::printf("    QR returned status %d — IR-LSQ skipped.\n", r.qr_status);
        return;
    }
    std::printf("    QR: %ld us, peak_RSS=%ld KB, predicted=%ld KB\n",
                r.qr_time_us, r.peak_rss_kb, r.analytical_kb);
    if (r.orth_error >= 0) std::printf("    orth_err = %.3e\n", (double)r.orth_error);
    std::printf("    IR-LSQ (x_0=0): total=%ld us, outer=%d, inner_total=%d\n",
                r.ir_total_us, r.ir_outer_iters, r.ir_inner_iters_total);
    std::printf("    ||Ax-b||/(||A||*||x||+||b||) = %.3e\n", (double)r.ls_residual_norm);
    if (r.ls_solution_error >= 0)
        std::printf("    ||x-x_true||/||x_true|| = %.3e\n", (double)r.ls_solution_error);
    else
        std::printf("    ||x-x_true||/||x_true|| = N/A (no ground-truth x_true)\n");
}

// ============================================================================
// Core templated runner
// ============================================================================

template <typename T, typename RNG, typename OpType>
static int run_benchmark_inner(
    OpType& A_op,
    int64_t m, int64_t n, int64_t input_nnz,
    const std::string& output_dir, int64_t num_runs,
    T d_factor, int64_t sketch_nnz, int64_t block_size,
    bool compute_cond,
    int64_t method_mask, T noise_level,
    long chol_time_us,
    const std::string& op_label,
    const std::string& input_label,
    const std::vector<T>* b_ptr,        // M-vector RHS
    const std::vector<T>* x_true_ptr)   // N-vector ground truth (sparse only); nullptr otherwise
{
    // Build the ordered list of selected algorithm names from the bitmask.
    //   bit 0  ( 1): CQRRT_linop
    //   bit 1  ( 2): CholQR
    //   bit 2  ( 4): sCholQR3
    //   bit 3  ( 8): sCholQR3_basic
    //   bit 4  (16): CholQR2
    //   bit 5  (32): Blendenpik (sketch + QR + LSQR; independent LSQ solver)
    //   bit 6  (64): CQRRT_linop_bqrrp  [DROPPED in 2026-06-05 rework]
    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 16)  selected_algs.push_back("CholQR2");
    if (method_mask & 32) {
        selected_algs.push_back("Blendenpik");        // its own sketch-and-solve warm start
        selected_algs.push_back("Blendenpik_cold");   // same solver, x0 = 0
    }

    if (selected_algs.empty()) {
        std::cerr << "Error: method_mask selects no algorithms (got " << method_mask << ").\n";
        return 1;
    }

    if (compute_cond) {
        RandLAPACK::testing::print_condition_diagnostics<T>(A_op, op_label);
    }

    // Per-run RNG states
    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = main_state;
        if (r > 0) run_states[r].key.incr(r);
    }

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);

    // Warmup (CQRRT_linop), plus the solve path: the build warmup already
    // applies A_op repeatedly, but the timed IterRefineLSQ also exercises the
    // TRSM preconditioner path and LSQR's vector work, whose one-time costs
    // (thread pools, first-touch pages) otherwise land inside the FIRST
    // method's timed solve (2026-08-05 yellow-bar finding). CPU warmup only,
    // distinct from the x0 warm-start ablation.
    std::cout << "Running warmup... " << std::flush;
    {
        auto warm_state = run_states[0];
        T* R_warm = new T[n * n]();
        RandLAPACK::CQRRT_linops<T, RNG> warm_algo(false, tol, false);
        warm_algo.nnz = sketch_nnz;
        warm_algo.block_size = block_size;
        int warm_status = warm_algo.call(A_op, R_warm, n, d_factor, warm_state);
        if (b_ptr) {
            T* x_wu = new T[n]();
            int it_wu = 0; long lt_wu[4] = {0};
            RandLAPACK::lsqr<T>(A_op, m, n,
                (warm_status == 0) ? R_warm : nullptr,
                (warm_status == 0) ? n : (int64_t)0,
                b_ptr->data(), x_wu, tol, tol, 5, it_wu, lt_wu);
            delete[] x_wu;
        }
        delete[] R_warm;
    }
    std::cout << "done\n\n";

    // Precompute ||A||_2 and ||b|| for the Higham backward-error metric:
    //   ls_residual_norm = ||A x - b|| / (||A||_2 * ||x|| + ||b||)
    T A_2norm = (T)0, b_norm = (T)0;
    if (b_ptr) {
        std::cout << "Estimating ||A||_2 via power iteration (10 iters)... " << std::flush;
        A_2norm = estimate_op_2norm<T>(A_op, m, n, 10);
        b_norm  = blas::nrm2(m, b_ptr->data(), 1);
        std::cout << "||A||_2 ~ " << A_2norm << ", ||b|| = " << b_norm << "\n\n";
    }

    T x_true_norm = (T)0;
    if (x_true_ptr) x_true_norm = blas::nrm2(n, x_true_ptr->data(), 1);

    std::vector<bench_result<T>> all_results;

    // Per-iteration workspaces, hoisted once: invariant sizes across all (alg, run) iters.
    T* R    = new T[n * n]();    // QR output; zero-filled per iter to match prior behavior
    T* x_ls = new T[n];          // initial guess (x_0 = 0) + refined solution
    T* Ax   = new T[m];          // A * x_ls for residual; overwritten beta=0

    // ================================================================
    // Per-(method, run) loop
    // ================================================================
    for (const auto& alg_name : selected_algs) {
        std::cout << "\n=== Algorithm: " << alg_name << " ===\n";
        std::vector<bench_result<T>> alg_results;

        for (int64_t run_idx = 0; run_idx < num_runs; ++run_idx) {
            bench_result<T> res{};
            res.m = m; res.n = n;
            res.run_idx = run_idx;
            res.alg_name = alg_name;
            res.noise_level = noise_level;
            res.chol_time_us = chol_time_us;
            res.qr_status = 0;
            res.qr_time_us = 0;
            res.orth_error = (T)-1.0;
            res.ir_total_us = 0;
            res.ir_outer_iters = 0;
            res.ir_inner_iters_total = 0;
            res.ls_residual_norm = (T)-1.0;
            res.ls_solution_error = (T)-1.0;
            res.peak_rss_kb = 0;
            res.analytical_kb = 0;

            std::fill(R, R + n * n, (T)0);
            auto state = run_states[run_idx];
            long blendenpik_lsqr_us = 0; int blendenpik_lsqr_iters = 0;  // Blendenpik solve stats

            // ---- QR dispatch (lifted verbatim from CQRRT_linop_irlsq.cc; +Blendenpik) ----
            std::cout << "[Run " << run_idx << ", " << alg_name << "] QR ... " << std::flush;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            if (alg_name.rfind("Blendenpik", 0) == 0) {
                // Independent sketch-and-precondition LSQ solver: sketch -> QR -> LSQR.
                // Produces x_ls directly (no mu, no IR-LSQ). R_out holds the sketch R factor
                // used for the Q = A R^{-1} orthogonality check.
                RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, tol);
                bp.nnz = sketch_nnz;
                bp.warm_start = (alg_name == "Blendenpik");   // "_cold" => x0 = 0
                // Match the inner-solve budget the Q-less QR methods get from the IR
                // driver (max_inner per step x 2 steps), so the comparison is not
                // decided by an accidental cap difference.
                bp.max_iters = ((g_ir_max_inner > 0) ? g_ir_max_inner : 200) * g_ir_n_steps;
                std::fill(x_ls, x_ls + n, (T)0);
                res.qr_status = bp.call(A_op, b_ptr->data(), m, x_ls, n, d_factor, state);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = bp.times[0] + bp.times[1];       // sketch + QR (preconditioner build)
                    std::copy(bp.R_out.begin(), bp.R_out.end(), R);   // sketch R factor for orth
                    res.qr_breakdown.assign(11, 0);
                    res.analytical_kb = 0;
                    // Blendenpik has no IR loop, so reuse the inner-CG diagnosis columns:
                    // capped = 0/1 from LSQR's own convergence flag, relres = its residual.
                    res.ir_inner_capped = bp.converged ? 0 : 1;
                    res.ir_inner_relres = bp.final_relres;
                    blendenpik_lsqr_us    = bp.times[2];
                    blendenpik_lsqr_iters = bp.lsqr_iters;
                }
            } else if (alg_name == "sCholQR3") {
                RandLAPACK::sCholQR3_linops<T> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
                }
            } else if (alg_name == "sCholQR3_basic") {
                RandLAPACK::sCholQR3_linops_basic<T> qr_algo(/*time_subroutines=*/true, tol);
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m, n);
                }
            } else if (alg_name == "CholQR") {
                RandLAPACK::CholQR_linops<T> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 6);
                    res.qr_breakdown.resize(11, 0);
                    res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
                }
            } else if (alg_name == "CholQR2") {
                RandLAPACK::CholQR2_linops<T> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::cholqr2_linops_analytical_kb<T>(m, n, block_size);
                }
            } else {
                // CQRRT_linop (TRSM_IDENTITY precond). CQRRT_linop_bqrrp was
                // removed from the benchmark dispatch in the 2026-06-05 rework.
                RandLAPACK::CQRRT_linops<T, RNG> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.nnz = sketch_nnz;
                qr_algo.block_size = block_size;
                qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
                res.qr_status = qr_algo.call(A_op, R, n, d_factor, state);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
                }
            }

            if (res.qr_status != 0) {
                std::cerr << "\n  [" << alg_name << "] Run " << run_idx
                          << ": QR returned status " << res.qr_status
                          << " (likely Cholesky breakdown). Skipping post-processing.\n";
                res.qr_time_us = -1;
                res.qr_breakdown.assign(11, 0);
                res.analytical_kb = 0;
                alg_results.push_back(res);
                all_results.push_back(res);
                print_irlsq_summary(res);
                continue;
            }
            std::cout << "done (" << res.qr_time_us << " us)";

            // ---- Orth_error: ||Q^T Q - I||_F / sqrt(n), blocked compute. Runs for every method. ----
            res.orth_error = compute_orth_error_explicit(A_op, R, m, n, block_size, &res.cond_precond);

            // ---- IR-LSQ post-processing ----
            {
                const std::vector<T>& b = *b_ptr;
                if (alg_name.rfind("Blendenpik", 0) == 0) {
                    // x_ls was already computed by Blendenpik's own LSQR above; no IR-LSQ.
                    std::cout << ". LSQR ... " << std::flush;
                    res.ir_total_us         = blendenpik_lsqr_us;
                    res.ir_outer_iters       = 1;
                    res.ir_inner_iters_total = blendenpik_lsqr_iters;   // LSQR iters in the CG-iters slot
                } else {
                    std::cout << ". IR-LSQ ... " << std::flush;
                    auto ls_t0 = steady_clock::now();

                    // Initial guess x_0 = 0 (per collaborator: no sketching in the LS
                    // solve itself). The only randomness is S_1 inside Q-less QR, which
                    // yields the preconditioner R; IterRefineLSQ starts from zero and the
                    // preconditioned inner CG converges from there.
                    std::fill(x_ls, x_ls + n, (T)0.0);

                    RandLAPACK::IterRefineLSQ<T> ir(
                        /*tol=*/     (g_ir_inner_tol > 0) ? (T)g_ir_inner_tol : tol,
                        /*max_inner=*/(g_ir_max_inner > 0) ? g_ir_max_inner : 200,
                        /*n_steps=*/g_ir_n_steps,
                        /*timing=*/true,
                        /*verbose=*/false);
                    ir.round_drop = (T)g_ir_round_drop;
                    ir.outer_tol = (g_ir_outer_tol >= 0) ? (T)g_ir_outer_tol
                                 : (T)10 * std::numeric_limits<T>::epsilon();
                    int ir_status = ir.call(A_op, R, n, b.data(), m, x_ls, n);
                    auto ls_t1 = steady_clock::now();
                    if (ir_status != 0) {
                        std::cerr << "Warning: IterRefineLSQ status " << ir_status << " (CG breakdown)\n";
                    }

                    res.ir_total_us = duration_cast<microseconds>(ls_t1 - ls_t0).count();
                    res.ir_outer_iters = ir.outer_iters_done;
                    res.ir_inner_iters_total = 0;
                    for (int v : ir.inner_iters_per_step) res.ir_inner_iters_total += v;
                    record_inner_cg_diagnosis(ir, res);
                    if (!ir.times.empty()) res.ir_breakdown = ir.times;
                }

                // Higham normwise backward-error metric:
                //   ls_residual_norm = ||A x - b|| / (||A||_2 * ||x|| + ||b||)
                // Drivable to machine epsilon for a backward-stable LS solver.
                A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                     m, 1, n, (T)1.0, x_ls, n, (T)0.0, Ax, m);
                T resid_sq = 0;
                #pragma omp parallel for reduction(+:resid_sq) schedule(static)
                for (int64_t i = 0; i < m; ++i) { T d = Ax[i] - b[i]; resid_sq += d * d; }
                T resid_norm = std::sqrt(resid_sq);
                T x_norm     = blas::nrm2(n, x_ls, 1);
                T denom      = A_2norm * x_norm + b_norm;
                res.ls_residual_norm = (denom > 0) ? resid_norm / denom : (T)-1.0;

                if (x_true_ptr) {
                    T err_sq = 0;
                    for (int64_t i = 0; i < n; ++i) {
                        T d = x_ls[i] - (*x_true_ptr)[i];
                        err_sq += d * d;
                    }
                    res.ls_solution_error = (x_true_norm > 0) ? std::sqrt(err_sq) / x_true_norm : (T)-1.0;
                } else {
                    res.ls_solution_error = (T)-1.0;
                }
                std::cout << "done (" << res.ir_total_us << " us)\n";
            }

            print_irlsq_summary(res);
            alg_results.push_back(res);
            all_results.push_back(res);
        }
    }

    // ================================================================
    // CSV output
    // ================================================================
    std::string time_buf = make_run_timestamp();

    std::string results_file   = output_dir + "/" + time_buf + "_irlsq_results.csv";
    std::string breakdown_file = output_dir + "/" + time_buf + "_irlsq_breakdown.csv";
    write_irlsq_results<T>(results_file, all_results, m, n, input_nnz, input_label,
                           noise_level, d_factor, sketch_nnz, block_size, method_mask);
    std::cout << "\nIR-LSQ results written to " << results_file << "\n";
    write_irlsq_breakdown<T>(breakdown_file, all_results);
    std::cout << "IR-LSQ breakdown written to " << breakdown_file << "\n";

    delete[] R; delete[] x_ls; delete[] Ax;
    return 0;
}

// ============================================================================
// RSPEC mode runner
// ============================================================================

template <typename T, typename RNG, typename VAppOpType, typename CompCOp>
static int run_rspec_benchmark(
    VAppOpType& V_app_op,         // m_K x n_V composite: C^j * V_FEM
    CompCOp&    C_op,             // m_K x m_K composite: L^T X^{-1} L  (the operator we Rayleigh-Ritz)
    int64_t m_K, int64_t n_V,
    const std::string& output_dir, int64_t num_runs,
    T d_factor, int64_t sketch_nnz, int64_t block_size,
    int64_t method_mask,
    long factor_time_us,
    const std::string& K_file, const std::string& M_file, const std::string& V_file,
    double omega, int64_t power_j)
{
    // Build the list of selected algorithm names from the bitmask (same as run_benchmark_inner).
    // bit 6 (64) = CQRRT_linop_bqrrp removed in the 2026-06-05 rework.
    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 16)  selected_algs.push_back("CholQR2");

    if (selected_algs.empty()) {
        std::cerr << "Error: method_mask selects no algorithms (got " << method_mask << ").\n";
        return 1;
    }

    int64_t m = m_K;
    int64_t n = n_V;
    int top_k = (int)std::min<int64_t>(10, n);

    // Per-run RNG states (same scheme as run_benchmark_inner).
    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = main_state;
        if (r > 0) run_states[r].key.incr(r);
    }

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);

    // Warmup so the Cholesky-factored X^{-1} chain inside V_app_op is warm.
    std::cout << "Running rspec warmup... " << std::flush;
    {
        auto warm_state = run_states[0];
        T* R_warm = new T[n * n]();
        RandLAPACK::CQRRT_linops<T, RNG> warm_algo(false, tol, false);
        warm_algo.nnz = sketch_nnz;
        warm_algo.block_size = block_size;
        warm_algo.call(V_app_op, R_warm, n, d_factor, warm_state);
        delete[] R_warm;
    }
    std::cout << "done\n\n";

    std::vector<rspec_result<T>> all_results;

    // Per-iteration QR output; invariant size across all (alg, run) iters.
    T* R = new T[n * n]();

    for (const auto& alg_name : selected_algs) {
        std::cout << "\n=== Algorithm: " << alg_name << " (rspec) ===\n";

        for (int64_t run_idx = 0; run_idx < num_runs; ++run_idx) {
            rspec_result<T> res{};
            res.m = m;
            res.n = n;
            res.run_idx = run_idx;
            res.alg_name = alg_name;
            res.qr_status = 0;
            res.qr_time_us = 0;
            res.peak_rss_kb = 0;
            res.analytical_kb = 0;
            res.factor_time_us = factor_time_us;
            res.rspec_total_us = 0;
            res.orth_error = (T)-1.0;

            auto rspec_t0 = steady_clock::now();

            std::fill(R, R + n * n, (T)0);
            auto state = run_states[run_idx];

            std::cout << "[Run " << run_idx << ", " << alg_name << "] QR ... " << std::flush;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            if (alg_name == "sCholQR3") {
                RandLAPACK::sCholQR3_linops<T> qr_algo(true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
                }
            } else if (alg_name == "sCholQR3_basic") {
                RandLAPACK::sCholQR3_linops_basic<T> qr_algo(true, tol);
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m, n);
                }
            } else if (alg_name == "CholQR") {
                RandLAPACK::CholQR_linops<T> qr_algo(true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 6);
                    res.qr_breakdown.resize(11, 0);
                    res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
                }
            } else if (alg_name == "CholQR2") {
                RandLAPACK::CholQR2_linops<T> qr_algo(true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::cholqr2_linops_analytical_kb<T>(m, n, block_size);
                }
            } else {
                // CQRRT_linop. CQRRT_linop_bqrrp was removed in 2026-06-05.
                RandLAPACK::CQRRT_linops<T, RNG> qr_algo(true, tol);
                qr_algo.nnz = sketch_nnz;
                qr_algo.block_size = block_size;
                qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
                res.qr_status = qr_algo.call(V_app_op, R, n, d_factor, state);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.total_us();
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
                }
            }

            if (res.qr_status != 0) {
                std::cerr << "\n  [" << alg_name << "] Run " << run_idx
                          << ": QR returned status " << res.qr_status
                          << ". Skipping eigen post-processing.\n";
                res.qr_time_us = -1;
                res.qr_breakdown.assign(11, 0);
                res.rr_breakdown.assign(4, 0);
                res.orth_error = std::numeric_limits<T>::quiet_NaN();
                res.top_eigvals.assign(top_k, std::numeric_limits<T>::quiet_NaN());
                res.top_residuals.assign(top_k, std::numeric_limits<T>::quiet_NaN());
                auto rspec_t1 = steady_clock::now();
                res.rspec_total_us = duration_cast<microseconds>(rspec_t1 - rspec_t0).count();
                all_results.push_back(res);
                continue;
            }
            std::cout << "done (" << res.qr_time_us << " us)\n";

            // ---- Orthogonality loss of the Q-factor: ||Q^T Q - I||_F / sqrt(n),
            //      Q = V_app * R^{-1}, materialized explicitly (same path as irlsq).
            //      NOTE: this re-applies V_app to n columns (one extra full pass of the
            //      C^j chain); at FEM2 scale that is a meaningful cost — see dev log.
            steady_clock::time_point rr_t0, rr_t1;
            long orth_us = 0, rr_build_us = 0, syevd_us = 0, resid_us = 0;

            std::cout << "    orth loss ... " << std::flush;
            rr_t0 = steady_clock::now();
            res.orth_error = compute_orth_error_explicit(V_app_op, R, m, n, block_size);
            rr_t1 = steady_clock::now();
            orth_us = duration_cast<microseconds>(rr_t1 - rr_t0).count();
            std::cout << "done (" << std::scientific << std::setprecision(3)
                      << res.orth_error << ")\n";

            // ----------------------------------------------------------------
            // Rayleigh-Ritz: T = R^{-T} V_app^T C V_app R^{-1}
            // Materialize V_app^T C V_app column-block by column-block on identity.
            // ----------------------------------------------------------------
            std::cout << "    Building Rayleigh-Ritz matrix T (n=" << n << ") ... " << std::flush;
            rr_t0 = steady_clock::now();
            int64_t b_rr = (block_size > 0) ? std::min<int64_t>(block_size, n) : std::min<int64_t>(64, n);

            T* T_mat = new T[(size_t)n * (size_t)n]();
            T* eye_blk = new T[(size_t)n * (size_t)b_rr]();
            T* Va_blk  = new T[(size_t)m * (size_t)b_rr]();
            T* CV_blk  = new T[(size_t)m * (size_t)b_rr]();
            T* T_blk   = new T[(size_t)n * (size_t)b_rr]();

            for (int64_t j0 = 0; j0 < n; j0 += b_rr) {
                int64_t bk = std::min(b_rr, n - j0);

                std::fill_n(eye_blk, (size_t)n * (size_t)b_rr, (T)0);
                for (int64_t j = 0; j < bk; ++j)
                    eye_blk[(j0 + j) + j * n] = (T)1;

                // Va_blk = V_app * eye_blk    (m x bk)
                V_app_op(blas::Side::Left, blas::Layout::ColMajor,
                         blas::Op::NoTrans, blas::Op::NoTrans,
                         m, bk, n, (T)1.0, eye_blk, n, (T)0.0, Va_blk, m);

                // CV_blk = C * Va_blk         (m x bk)
                C_op(blas::Side::Left, blas::Layout::ColMajor,
                     blas::Op::NoTrans, blas::Op::NoTrans,
                     m, bk, m, (T)1.0, Va_blk, m, (T)0.0, CV_blk, m);

                // T_blk = V_app^T * CV_blk    (n x bk)
                V_app_op(blas::Side::Left, blas::Layout::ColMajor,
                         blas::Op::Trans, blas::Op::NoTrans,
                         n, bk, m, (T)1.0, CV_blk, m, (T)0.0, T_blk, n);

                lapack::lacpy(lapack::MatrixType::General, n, bk, T_blk, n, T_mat + j0 * n, n);
            }

            delete[] eye_blk;
            delete[] Va_blk;
            delete[] CV_blk;
            delete[] T_blk;

            // Apply R^{-T} on the left: T := R^{-T} * T
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::Trans, blas::Diag::NonUnit,
                       n, n, (T)1.0, R, n, T_mat, n);
            // Apply R^{-1} on the right: T := T * R^{-1}
            blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, n, (T)1.0, R, n, T_mat, n);

            // Symmetrize against rounding drift before the (upper-triangle) syevd.
            RandBLAS::symmetrize(blas::Layout::ColMajor, blas::Uplo::Upper, n, T_mat, n);
            rr_t1 = steady_clock::now();
            rr_build_us = duration_cast<microseconds>(rr_t1 - rr_t0).count();
            std::cout << "done\n";

            // Eigendecomposition: T = U diag(lambda) U^T, U overwrites T_mat (columns = eigvecs).
            std::cout << "    syevd ... " << std::flush;
            rr_t0 = steady_clock::now();
            T* eigvals = new T[(size_t)n]();
            int64_t syevd_info = lapack::syevd(lapack::Job::Vec, blas::Uplo::Upper,
                                                n, T_mat, n, eigvals);
            if (syevd_info != 0) {
                std::cerr << "Warning: syevd returned " << syevd_info << " for run " << run_idx << "\n";
            }
            // syevd returns eigvals in ascending order; collect top-k by absolute magnitude (largest |lambda|).
            std::cout << "done\n";

            // Sort eigenvalues by descending |lambda|; build permutation.
            int64_t* perm = new int64_t[(size_t)n];
            for (int64_t i = 0; i < n; ++i) perm[i] = i;
            std::sort(perm, perm + n, [&](int64_t a, int64_t b_){
                return std::abs(eigvals[a]) > std::abs(eigvals[b_]);
            });

            res.top_eigvals.resize(top_k);
            for (int i = 0; i < top_k; ++i) res.top_eigvals[i] = eigvals[perm[i]];
            rr_t1 = steady_clock::now();
            syevd_us = duration_cast<microseconds>(rr_t1 - rr_t0).count();

            // ----------------------------------------------------------------
            // Ritz residual norms for the top-k pairs (collaborator spec):
            //   resid_i = ||C y_i - lambda_i y_i|| / (|lambda_max| * ||y_i||)
            // where y_i = Q u_i = V_app R^{-1} u_i is the Ritz vector and u_i is an
            // eigenvector of the small RR matrix T = Q^T C Q (Q = V_app R^{-1}, with
            // Q^T Q = I via the Q-less QR). This is the ordinary (non-generalized) eigen-
            // residual of the symmetric operator C, which is exactly what Rayleigh-
            // Ritz on range(V_app) approximates. |lambda_max| is the dominant Ritz
            // value (largest |lambda|), used as the relative scale. K and M are no
            // longer needed here — only C and the Ritz vectors.
            // ----------------------------------------------------------------
            std::cout << "    Ritz residuals ... " << std::flush;
            rr_t0 = steady_clock::now();

            // u_blk = R^{-1} * U_topk  (n x top_k); columns = top-k eigenvectors of T.
            T* u_blk = new T[(size_t)n * (size_t)top_k]();
            for (int i = 0; i < top_k; ++i)
                for (int64_t r = 0; r < n; ++r)
                    u_blk[r + i * n] = T_mat[r + perm[i] * n];
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, top_k, (T)1.0, R, n, u_blk, n);

            // y_blk = V_app * (R^{-1} U_topk) = Q U_topk   (m x top_k): the Ritz vectors.
            T* y_blk = new T[(size_t)m * (size_t)top_k]();
            V_app_op(blas::Side::Left, blas::Layout::ColMajor,
                     blas::Op::NoTrans, blas::Op::NoTrans,
                     m, top_k, n, (T)1.0, u_blk, n, (T)0.0, y_blk, m);

            // Cy_blk = C * y_blk   (m x top_k).
            T* Cy_blk = new T[(size_t)m * (size_t)top_k]();
            C_op(blas::Side::Left, blas::Layout::ColMajor,
                 blas::Op::NoTrans, blas::Op::NoTrans,
                 m, top_k, m, (T)1.0, y_blk, m, (T)0.0, Cy_blk, m);

            // |lambda_max| = dominant Ritz value (top_eigvals sorted by descending |lambda|).
            T lam_max = (top_k > 0) ? std::abs(res.top_eigvals[0]) : (T)0;

            res.top_residuals.resize(top_k);
            for (int i = 0; i < top_k; ++i) {
                T lam = res.top_eigvals[i];
                T num_sq = 0, y_sq = 0;
                for (int64_t r = 0; r < m; ++r) {
                    T d = Cy_blk[r + i * m] - lam * y_blk[r + i * m];
                    num_sq += d * d;
                    y_sq   += y_blk[r + i * m] * y_blk[r + i * m];
                }
                T denom = lam_max * std::sqrt(y_sq);
                res.top_residuals[i] = (denom > 0) ? std::sqrt(num_sq) / denom
                                                   : std::numeric_limits<T>::quiet_NaN();
            }
            rr_t1 = steady_clock::now();
            resid_us = duration_cast<microseconds>(rr_t1 - rr_t0).count();

            delete[] u_blk;
            delete[] y_blk;
            delete[] Cy_blk;
            delete[] eigvals;
            delete[] perm;
            delete[] T_mat;
            std::cout << "done\n";

            auto rspec_t1 = steady_clock::now();
            res.rspec_total_us = duration_cast<microseconds>(rspec_t1 - rspec_t0).count();
            res.rr_breakdown = {orth_us, rr_build_us, syevd_us, resid_us};

            std::cout << "    Top eigvals: ";
            for (int i = 0; i < std::min(5, top_k); ++i)
                std::cout << res.top_eigvals[i] << " ";
            std::cout << "\n";
            std::cout << "    Top residuals: ";
            for (int i = 0; i < std::min(5, top_k); ++i)
                std::cout << res.top_residuals[i] << " ";
            std::cout << "\n";

            all_results.push_back(res);
        }
    }

    // CSV output
    std::string time_buf = make_run_timestamp();
    std::string results_file = output_dir + "/" + time_buf + "_rspec_results.csv";
    write_rspec_csv<T>(results_file, all_results, m, n, num_runs,
                       K_file, M_file, V_file, omega, power_j,
                       sketch_nnz, block_size, method_mask, top_k);
    std::cout << "\nRSPEC results written to " << results_file << "\n";

    std::string breakdown_file = output_dir + "/" + time_buf + "_rspec_breakdown.csv";
    write_rspec_breakdown<T>(breakdown_file, all_results);
    std::cout << "RSPEC breakdown written to " << breakdown_file << "\n";

    delete[] R;
    return 0;
}

// ============================================================================
// CSV writer — IR-LSQ regularized (irlsq_reg): base columns + regularization /
// mixed-precision metadata (kappa_target, kappa_measured, mu, precond/solve prec)
// ============================================================================

template <typename T>
static void write_irlsq_reg_results(
    const std::string& filename,
    const std::vector<bench_result<T>>& results,
    int64_t m, int64_t n, int64_t nnz_or_zero, const std::string& input_label,
    double d_factor, int64_t sketch_nnz, int64_t block_size, int64_t method_mask,
    double kappa_target, double mu,
    const std::string& precond_prec, const std::string& solve_prec)
{
    std::ofstream out(filename);
    time_t now = time(nullptr);
    out << "# Sparse IR-LSQ (regularized augmented operator) Benchmark results\n"
        << "# Date: " << ctime(&now)
        << "# input=" << input_label << "\n"
        << "# M=" << m << " N=" << n << " nnz=" << nnz_or_zero << "\n"
        << "# d_factor=" << d_factor << " sketch_nnz=" << sketch_nnz
        << " block_size=" << block_size << "\n"
        << "# method_mask=" << method_mask << "\n"
        << "# kappa_target=" << kappa_target << " mu=" << mu << "\n"
        << "# precond_prec=" << precond_prec << " solve_prec=" << solve_prec << "\n"
        << "# blendenpik=warm+cold (IR methods always cold x0; 2026-08-05 policy)\n"
        << "# A_hat = [A; mu*I];  R = chol(A^T A + mu^2 I) built in precond_prec,\n"
        << "#   used as right preconditioner for IterRefineLSQ run in solve_prec.\n"
#ifdef _OPENMP
        << "# OpenMP threads: " << omp_get_max_threads() << "\n"
#else
        << "# OpenMP threads: 1\n"
#endif
        ;
    out << "algorithm,run,m,n,qr_status,qr_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,ir_total_us,ir_outer_iters,ir_inner_iters_total,"
           "ls_residual_norm,ls_solution_error,kappa_target,kappa_measured,mu,precond_prec,solve_prec,chol_retries,"
           "ir_inner_capped,ir_inner_relres,ir_inner_best_relres,ir_inner_best_iter,cond_precond,ir_setup_us\n";
    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << "," << r.m << "," << r.n << ","
            << r.qr_status << "," << r.qr_time_us << "," << r.peak_rss_kb << "," << r.analytical_kb << ","
            << std::scientific << std::setprecision(6) << r.orth_error << ","
            << r.ir_total_us << "," << r.ir_outer_iters << "," << r.ir_inner_iters_total << ","
            << std::scientific << std::setprecision(6) << r.ls_residual_norm << ","
            << std::scientific << std::setprecision(6) << r.ls_solution_error << ","
            << std::scientific << std::setprecision(6) << kappa_target << ","
            << std::scientific << std::setprecision(6) << r.kappa_measured << ","
            << std::scientific << std::setprecision(6) << mu << ","
            << precond_prec << "," << solve_prec << "," << r.chol_retries << ","
            << r.ir_inner_capped << ","
            << std::scientific << std::setprecision(6) << r.ir_inner_relres << ","
            << std::scientific << std::setprecision(6) << r.ir_inner_best_relres << ","
            << r.ir_inner_best_iter << ","
            << std::scientific << std::setprecision(6) << r.cond_precond << ","
            << r.ir_setup_us
            << "\n";
    }
}

// kappa(A) estimate from the regularized R diagonal: max|R_ii| / min|R_ii|.
template <typename P>
static double kappa_from_R_diag(const P* R, int64_t n) {
    double mx = 0.0, mn = std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        double v = std::abs((double)R[i + i * n]);
        if (v > mx) mx = v;
        if (v > 0 && v < mn) mn = v;
    }
    return (mn > 0 && std::isfinite(mn)) ? mx / mn : -1.0;
}

// ============================================================================
// irlsq_reg runner — regularized augmented-operator preconditioner with
// independent preconditioner (P_precond) and solve (T_solve) precisions.
//
// Builds two FEM operator chains J = L^{-1} K (V D) from the same kappa-scaled
// matrices: one in P_precond (for Q-less QR of A_hat = [A; mu*I]) and one in
// T_solve (for IterRefineLSQ on the base A). For each variant: QR in P_precond
// -> R (= chol(A^T A + mu^2 I)) -> cast to T_solve -> solve. R is never stored
// for all variants at once (n^2 is huge at FEM2 scale), so QR and solve are
// interleaved and both chains coexist.
// ============================================================================

template <typename T_solve, typename P_precond, typename RNG>
static int run_irlsq_reg(
    const std::string& K_file, const std::string& M_file, const std::string& V_file,
    const std::string& output_dir, int64_t num_runs,
    double d_factor, int64_t sketch_nnz, int64_t block_size,
    int64_t method_mask, double kappa_target, double mu_factor, double noise_level,
    const std::string& precond_prec_str, const std::string& solve_prec_str)
{
    namespace rl = RandLAPACK::linops;

    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 16)  selected_algs.push_back("CholQR2");
    if (method_mask & 32) {   // independent sketch+QR+LSQR, warm + cold variants
        selected_algs.push_back("Blendenpik");
        selected_algs.push_back("Blendenpik_cold");
    }
    if (selected_algs.empty()) {
        std::cerr << "Error: method_mask selects no algorithms (got " << method_mask << ").\n";
        return 1;
    }

    // ---- Load double master CSRs ----
    int64_t m_K, n_K, nnz_K, m_M, n_M, nnz_M, m_V, n_V, nnz_V;
    auto K_master = load_csr_verbose<double>("K (stiffness)", K_file, m_K, n_K, nnz_K);
    auto M_master = load_csr_verbose<double>("M (mass)",      M_file, m_M, n_M, nnz_M);
    auto V_master = load_csr_verbose<double>("V (prolongation)", V_file, m_V, n_V, nnz_V);
    if (m_K != n_K)   { std::cerr << "Error: K must be square.\n"; return 1; }
    if (m_M != m_K || n_M != m_K) { std::cerr << "Error: M size must match K.\n"; return 1; }
    if (m_V != m_K)   { std::cerr << "Error: V rows must match K size.\n"; return 1; }
    if (m_V < n_V)    { std::cerr << "Error: need tall V (m_fine >= n_coarse).\n"; return 1; }
    int64_t m = m_V, n = n_V;

    // ---- Inject conditioning: scale V columns by the geometric diagonal ----
    auto d_scale = geometric_colscale(n_V, kappa_target);
    scale_csr_columns(V_master, d_scale);
    std::cout << "Column-scaled V to target kappa=" << kappa_target
              << " (spread " << d_scale.front() << " .. " << d_scale.back() << ")\n";

    // ---- Build SOLVE chain (precision T_solve), cast down from double master ----
    auto K_Ts = csr_cast<T_solve>(K_master);
    auto V_Ts = csr_cast<T_solve>(V_master);
    auto M_Ts = csr_cast<T_solve>(M_master);
    rl::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T_solve>> K_op_Ts(m_K, m_K, K_Ts);
    rl::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T_solve>> V_op_Ts(m_V, n_V, V_Ts);
    std::cout << "Factorizing M = L L^T (solve precision)... " << std::flush;
    RandLAPACK_extras::linops::CholSolverLinOp<T_solve> L_inv_Ts(M_Ts, /*half_solve=*/true);
    L_inv_Ts.factorize();
    std::cout << "done\n";
    rl::CompositeOperator KV_Ts(m, n, K_op_Ts, V_op_Ts); KV_Ts.block_size = block_size;
    rl::CompositeOperator J_Ts(m, n, L_inv_Ts, KV_Ts);   J_Ts.block_size = block_size;

    // Consistent RHS: x_true ~ U(-1,1)^n, b = A x_true (+ noise_level relative
    // Gaussian noise). Consistency makes the residual metric a true backward error
    // ~u (kappa-robust: the ||A|| ||x|| factor cancels), and x_true gives a ground
    // -truth forward-error metric ||x - x_true|| / ||x_true|| ~ u*kappa that exposes
    // the precision x kappa interaction. Use noise_level = 0 to see the solver's
    // u-level backward error directly. (Same construction as sparse mode.)
    std::vector<T_solve> x_true(n, (T_solve)0);
    { std::mt19937 rng_x(42); std::uniform_real_distribution<double> U(-1.0, 1.0);
      for (auto& v : x_true) v = (T_solve)U(rng_x); }
    std::vector<T_solve> b(m, (T_solve)0);
    J_Ts(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
         m, 1, n, (T_solve)1.0, x_true.data(), n, (T_solve)0.0, b.data(), m);
    if (noise_level > 0) {
        T_solve b_clean_norm = blas::nrm2(m, b.data(), 1);
        std::vector<T_solve> noise(m, (T_solve)0);
        std::mt19937 rng_n(13); std::normal_distribution<double> N01(0, 1);
        for (auto& v : noise) v = (T_solve)N01(rng_n);
        T_solve raw = blas::nrm2(m, noise.data(), 1);
        T_solve scale = (raw > 0) ? (T_solve)(noise_level) * b_clean_norm / raw : (T_solve)0;
        for (int64_t i = 0; i < m; ++i) b[i] += scale * noise[i];
    }
    const T_solve x_true_norm = blas::nrm2(n, x_true.data(), 1);
    std::cout << "Consistent RHS b = A x_true"
              << (noise_level > 0 ? " + noise" : "")
              << " (||x_true||=" << x_true_norm << ", ||b||=" << blas::nrm2(m, b.data(), 1) << ")\n";

    // ---- Build PRECOND chain (precision P_precond) + augmented operator ----
    auto K_Pp = csr_cast<P_precond>(K_master);
    auto V_Pp = csr_cast<P_precond>(V_master);
    auto M_Pp = csr_cast<P_precond>(M_master);
    rl::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<P_precond>> K_op_Pp(m_K, m_K, K_Pp);
    rl::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<P_precond>> V_op_Pp(m_V, n_V, V_Pp);
    std::cout << "Factorizing M = L L^T (precond precision)... " << std::flush;
    RandLAPACK_extras::linops::CholSolverLinOp<P_precond> L_inv_Pp(M_Pp, /*half_solve=*/true);
    L_inv_Pp.factorize();
    std::cout << "done\n";
    rl::CompositeOperator KV_Pp(m, n, K_op_Pp, V_op_Pp); KV_Pp.block_size = block_size;
    rl::CompositeOperator J_Pp(m, n, L_inv_Pp, KV_Pp);   J_Pp.block_size = block_size;

    // ||A||_2 and ||b|| for the Higham backward-error metric.
    T_solve A_2norm = estimate_op_2norm<T_solve>(J_Ts, m, n, 10);
    T_solve b_norm  = blas::nrm2(m, b.data(), 1);
    std::cout << "||A||_2 ~ " << A_2norm << ", ||b|| = " << b_norm << "\n";

    // Regularization per the collaborator's spec: mu = mu_factor * u(precond),
    // with mu_factor = 10 giving mu = 10u (u = unit roundoff of the precond
    // precision). NO ||A|| or size scaling -- the augmented operator is exactly
    // A_hat = [A; mu*I], Q-less CholeskyQR of which gives R = chol(A^T A + mu^2 I),
    // used as a right preconditioner for the LS problem in A.
    const P_precond mu_P = (P_precond)(mu_factor * (double)unit_roundoff<P_precond>());
    rl::ScaledIdentityOp<P_precond> reg_op(n, mu_P);
    rl::VStackOp<decltype(J_Pp), rl::ScaledIdentityOp<P_precond>> A_hat_Pp(J_Pp, reg_op);
    A_hat_Pp.block_size = block_size;   // caps the blocked-sketch slice width (CQRRT)
    std::cout << "Augmented operator A_hat = [J; mu*I], mu=" << (double)mu_P
              << " (= " << mu_factor << " * u(" << precond_prec_str << "))\n\n";

    const P_precond tol_P = std::pow(std::numeric_limits<P_precond>::epsilon(), (P_precond)0.85);
    const T_solve   tol_T = std::pow(std::numeric_limits<T_solve>::epsilon(), (T_solve)0.85);

    // Per-run RNG states (CQRRT only).
    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) { run_states[r] = main_state; if (r > 0) run_states[r].key.incr(r); }

    // Warmup the precond-chain CQRRT on A_hat (warms the L^{-1} K V chain, the
    // augmented Gram, and the blocked sketch overload), then the SOLVE chain:
    // the timed IR-LSQ runs LSQR on J_Ts with a TRSM preconditioner, and before
    // 2026-08-05 its thread pools / first-touch pages were paid inside the FIRST
    // method's timed solve, which at 4-7 inner iterations is the same magnitude
    // as the whole solve (the yellow-bar finding). A few untimed LSQR iterations
    // on J_Ts (with the warmup R when usable) close that gap. This is a CPU
    // warmup, distinct from Blendenpik's x0 warm start.
    std::cout << "Running warmup... " << std::flush;
    { auto ws = run_states[0]; P_precond* Rw = new P_precond[n * n]();
      RandLAPACK::CQRRT_linops<P_precond, RNG> warm(false, tol_P);
      warm.nnz = sketch_nnz; warm.block_size = block_size;
      int warm_status = warm.call(A_hat_Pp, Rw, n, (P_precond)d_factor, ws);
      T_solve* Rw_T = new T_solve[n * n];
      if (warm_status == 0)
          for (int64_t i = 0; i < n * n; ++i) Rw_T[i] = (T_solve)Rw[i];
      T_solve* x_wu = new T_solve[n]();
      int it_wu = 0; long lt_wu[4] = {0};
      RandLAPACK::lsqr<T_solve>(J_Ts, m, n,
          (warm_status == 0) ? Rw_T : nullptr, (warm_status == 0) ? n : (int64_t)0,
          b.data(), x_wu, tol_T, tol_T, 5, it_wu, lt_wu);
      delete[] Rw; delete[] Rw_T; delete[] x_wu; }
    std::cout << "done\n";

    std::vector<bench_result<T_solve>> all_results;

    P_precond* R_P = new P_precond[n * n]();
    T_solve*   R_T = new T_solve[n * n]();
    T_solve*   x_ls = new T_solve[n];

    for (const auto& alg_name : selected_algs) {
        std::cout << "\n=== Algorithm: " << alg_name << " (irlsq_reg) ===\n";
        for (int64_t run_idx = 0; run_idx < num_runs; ++run_idx) {
            bench_result<T_solve> res{};
            res.m = m; res.n = n; res.run_idx = run_idx; res.alg_name = alg_name;
            res.qr_status = 0; res.qr_time_us = 0; res.orth_error = (T_solve)-1;
            res.ls_residual_norm = (T_solve)-1; res.ls_solution_error = (T_solve)-1;
            res.kappa_measured = (T_solve)-1;

            std::fill(R_P, R_P + n * n, (P_precond)0);
            auto state = run_states[run_idx];
            const bool is_bp = (alg_name.rfind("Blendenpik", 0) == 0);
            long bp_lsqr_us = 0; int bp_lsqr_iters = 0;

            std::cout << "[Run " << run_idx << ", " << alg_name << "] QR(" << precond_prec_str
                      << ") ... " << std::flush;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            if (is_bp) {
                // Independent Blendenpik in SOLVE precision on the BASE operator J_Ts (no mu,
                // no augmented A_hat): sketch -> Householder QR -> LSQR, producing x_ls directly.
                RandLAPACK::Blendenpik_linops<T_solve, RNG> bp(true, tol_T);
                bp.nnz = sketch_nnz;
                bp.warm_start = (alg_name == "Blendenpik");   // "_cold" => x0 = 0
                // Same inner-solve budget as the IR methods (see the irlsq path).
                bp.max_iters = ((g_ir_max_inner > 0) ? g_ir_max_inner : 200) * g_ir_n_steps;
                std::fill(x_ls, x_ls + n, (T_solve)0);
                res.qr_status = bp.call(J_Ts, b.data(), m, x_ls, n, (T_solve)d_factor, state);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = bp.times[0] + bp.times[1];          // sketch + QR (preconditioner build)
                    std::copy(bp.R_out.begin(), bp.R_out.end(), R_T);    // R_T = sketch R factor (orth + kappa)
                    res.qr_breakdown.assign(11, 0); res.analytical_kb = 0;
                    res.ir_inner_capped = bp.converged ? 0 : 1;
                    res.ir_inner_relres = bp.final_relres;
                    bp_lsqr_us = bp.times[2]; bp_lsqr_iters = bp.lsqr_iters;
                }
            } else if (alg_name == "sCholQR3") {
                RandLAPACK::sCholQR3_linops<P_precond> qr(true, tol_P); qr.block_size = block_size;
                res.qr_status = qr.call(A_hat_Pp, R_P, n); res.peak_rss_kb = mem.stop(); res.chol_retries = qr.n_chol_retries;
                if (res.qr_status == 0) { res.qr_time_us = qr.total_us();
                    res.qr_breakdown.assign(qr.times.begin(), qr.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<P_precond>(m, n, block_size); }
            } else if (alg_name == "sCholQR3_basic") {
                RandLAPACK::sCholQR3_linops_basic<P_precond> qr(true, tol_P);
                res.qr_status = qr.call(A_hat_Pp, R_P, n); res.peak_rss_kb = mem.stop(); res.chol_retries = qr.n_chol_retries;
                if (res.qr_status == 0) { res.qr_time_us = qr.total_us();
                    res.qr_breakdown.assign(qr.times.begin(), qr.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<P_precond>(m, n); }
            } else if (alg_name == "CholQR") {
                RandLAPACK::CholQR_linops<P_precond> qr(true, tol_P); qr.block_size = block_size;
                res.qr_status = qr.call(A_hat_Pp, R_P, n); res.peak_rss_kb = mem.stop(); res.chol_retries = qr.n_chol_retries;
                if (res.qr_status == 0) { res.qr_time_us = qr.total_us();
                    res.qr_breakdown.assign(qr.times.begin(), qr.times.begin() + 6);
                    res.qr_breakdown.resize(11, 0);
                    res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<P_precond>(m, n, block_size); }
            } else if (alg_name == "CholQR2") {
                RandLAPACK::CholQR2_linops<P_precond> qr(true, tol_P); qr.block_size = block_size;
                res.qr_status = qr.call(A_hat_Pp, R_P, n); res.peak_rss_kb = mem.stop(); res.chol_retries = qr.n_chol_retries;
                if (res.qr_status == 0) { res.qr_time_us = qr.total_us();
                    res.qr_breakdown.assign(qr.times.begin(), qr.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::cholqr2_linops_analytical_kb<P_precond>(m, n, block_size); }
            } else {
                // CQRRT: sketch + Gram the augmented A_hat (via VStack's blocked sketch
                // overload), uniformly with the other 4 methods. R = chol(A^T A + mu^2 I).
                RandLAPACK::CQRRT_linops<P_precond, RNG> qr(true, tol_P);
                qr.nnz = sketch_nnz; qr.block_size = block_size;
                qr.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
                res.qr_status = qr.call(A_hat_Pp, R_P, n, (P_precond)d_factor, state); res.peak_rss_kb = mem.stop(); res.chol_retries = qr.n_chol_retries;
                if (res.qr_status == 0) { res.qr_time_us = qr.total_us();
                    res.qr_breakdown.assign(qr.times.begin(), qr.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<P_precond>(m, n, (P_precond)d_factor, block_size); }
            }

            if (res.qr_status != 0) {
                std::cerr << "\n  [" << alg_name << "] Run " << run_idx
                          << ": QR returned status " << res.qr_status << ". Skipping solve.\n";
                res.qr_time_us = -1; res.qr_breakdown.assign(11, 0); res.analytical_kb = 0;
                all_results.push_back(res);
                continue;
            }
            res.kappa_measured = is_bp ? (T_solve)kappa_from_R_diag<T_solve>(R_T, n)
                                       : (T_solve)kappa_from_R_diag<P_precond>(R_P, n);
            std::cout << "done (" << res.qr_time_us << " us, kappa~"
                      << std::scientific << std::setprecision(2) << (double)res.kappa_measured << ")";

            // Cast R to solve precision (Blendenpik already produced R_T directly).
            if (!is_bp) for (int64_t i = 0; i < n * n; ++i) R_T[i] = (T_solve)R_P[i];

            // Orthogonality loss of Q = A R^{-1} (base A in solve precision).
            res.orth_error = compute_orth_error_explicit<T_solve>(J_Ts, R_T, m, n, block_size, &res.cond_precond);

            // Solve in solve precision. Blendenpik already solved (its own LSQR above);
            // everyone else runs IR-LSQ with R as the right preconditioner.
            if (is_bp) {
                std::cout << ". LSQR(" << solve_prec_str << ") ... " << std::flush;
                res.ir_total_us         = bp_lsqr_us;
                res.ir_outer_iters       = 1;
                res.ir_inner_iters_total = bp_lsqr_iters;   // LSQR iters in the CG-iters slot
            } else {
                std::cout << ". IR-LSQ(" << solve_prec_str << ") ... " << std::flush;
                auto ls_t0 = steady_clock::now();
                // Always cold (2026-08-05 policy): the sketch-and-solve x0 warm-start
                // ablation that lived here is removed -- warm x0 is Blendenpik-only
                // now (see the CLI comment block). ir_setup_us stays in the CSV
                // schema and is always 0 for IR methods.
                std::fill(x_ls, x_ls + n, (T_solve)0.0);
                RandLAPACK::IterRefineLSQ<T_solve> ir(
                    (g_ir_inner_tol > 0) ? (T_solve)g_ir_inner_tol : tol_T,
                    (g_ir_max_inner > 0) ? g_ir_max_inner : 200,
                    g_ir_n_steps, true, false);
                ir.round_drop = (T_solve)g_ir_round_drop;
                ir.outer_tol = (g_ir_outer_tol >= 0) ? (T_solve)g_ir_outer_tol
                             : (T_solve)10 * std::numeric_limits<T_solve>::epsilon();
                int ir_status = ir.call(J_Ts, R_T, n, b.data(), m, x_ls, n);
                auto ls_t1 = steady_clock::now();
                if (ir_status != 0) std::cerr << "Warning: IterRefineLSQ status " << ir_status << "\n";
                res.ir_total_us = duration_cast<microseconds>(ls_t1 - ls_t0).count();
                res.ir_outer_iters = ir.outer_iters_done;
                res.ir_inner_iters_total = 0;
                for (int v : ir.inner_iters_per_step) res.ir_inner_iters_total += v;
                record_inner_cg_diagnosis(ir, res);
                if (!ir.times.empty()) res.ir_breakdown = ir.times;
            }

            // Higham normwise backward error ||Ax-b|| / (||A||_2 ||x|| + ||b||).
            std::vector<T_solve> Ax(m, (T_solve)0);
            J_Ts(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 m, 1, n, (T_solve)1.0, x_ls, n, (T_solve)0.0, Ax.data(), m);
            T_solve resid_sq = 0;
            for (int64_t i = 0; i < m; ++i) { T_solve dd = Ax[i] - b[i]; resid_sq += dd * dd; }
            T_solve x_norm = blas::nrm2(n, x_ls, 1);
            T_solve denom  = A_2norm * x_norm + b_norm;
            res.ls_residual_norm = (denom > 0) ? std::sqrt(resid_sq) / denom : (T_solve)-1;

            // Forward error vs ground truth: ||x - x_true|| / ||x_true|| ~ u*kappa.
            T_solve err_sq = 0;
            for (int64_t i = 0; i < n; ++i) { T_solve dd = x_ls[i] - x_true[i]; err_sq += dd * dd; }
            res.ls_solution_error = (x_true_norm > 0) ? std::sqrt(err_sq) / x_true_norm : (T_solve)-1;

            std::cout << "done (" << res.ir_total_us << " us, bwd_err="
                      << std::scientific << std::setprecision(3) << (double)res.ls_residual_norm
                      << ", fwd_err=" << (double)res.ls_solution_error << ")\n";

            all_results.push_back(res);
        }
    }
    delete[] R_P; delete[] R_T; delete[] x_ls;

    std::string time_buf = make_run_timestamp();
    std::string results_file   = output_dir + "/" + time_buf + "_irlsq_reg_results.csv";
    std::string breakdown_file = output_dir + "/" + time_buf + "_irlsq_reg_breakdown.csv";
    write_irlsq_reg_results<T_solve>(results_file, all_results, m, n, nnz_K,
        "L^{-1} K (V D) (M=" + M_file + ")", d_factor, sketch_nnz, block_size, method_mask,
        kappa_target, (double)mu_P, precond_prec_str, solve_prec_str);
    std::cout << "\nIR-LSQ-reg results written to " << results_file << "\n";
    write_irlsq_breakdown<T_solve>(breakdown_file, all_results);
    std::cout << "IR-LSQ-reg breakdown written to " << breakdown_file << "\n";
    return 0;
}

// Dispatch run_irlsq_reg on the runtime precond-precision string (solve precision = T).
template <typename T_solve, typename RNG>
static int dispatch_irlsq_reg(
    const std::string& precond_prec,
    const std::string& K_file, const std::string& M_file, const std::string& V_file,
    const std::string& output_dir, int64_t num_runs,
    double d_factor, int64_t sketch_nnz, int64_t block_size,
    int64_t method_mask, double kappa_target, double mu_factor, double noise_level,
    const std::string& solve_prec_str)
{
    if (precond_prec == "double") {
        return run_irlsq_reg<T_solve, double, RNG>(
            K_file, M_file, V_file, output_dir, num_runs, d_factor, sketch_nnz, block_size,
            method_mask, kappa_target, mu_factor, noise_level, "double", solve_prec_str);
    } else if (precond_prec == "single" || precond_prec == "float") {
        return run_irlsq_reg<T_solve, float, RNG>(
            K_file, M_file, V_file, output_dir, num_runs, d_factor, sketch_nnz, block_size,
            method_mask, kappa_target, mu_factor, noise_level, "single", solve_prec_str);
    }
    std::cerr << "Error: precond_prec must be 'single' or 'double'; got '" << precond_prec << "'\n";
    return 1;
}

// ============================================================================
// Main dispatcher
// ============================================================================

template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[]) {
    // <prec> <out> <runs> <mode> ...  → argc >= 5 to reach <mode>
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <mode>\n"
                  << "    sparse mode: 'sparse' <A_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [compute_cond] [method_mask] [noise_level]\n"
                  << "    FEM mode:    <K_file> <M_file> <V_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [compute_cond] [method_mask] [noise_level]"
                  << " [omega] [power_j]\n"
                  << "  mode  = irlsq | rspec | irlsq_reg   (rspec/irlsq_reg are FEM-only)\n";
        return 1;
    }

    std::string output_dir = argv[2];
    int64_t num_runs       = std::stol(argv[3]);
    std::string mode       = argv[4];
    if (mode != "irlsq" && mode != "rspec" && mode != "irlsq_reg") {
        std::cerr << "Error: <mode> must be one of {irlsq, rspec, irlsq_reg}; got '" << mode << "'\n";
        return 1;
    }

    std::string arg5 = argv[5];
    bool sparse_mode = (arg5 == "sparse");

    std::string K_file, M_file, V_file, A_file;
    T d_factor;
    int dfactor_idx;

    if (sparse_mode) {
        if (argc < 8) {
            std::cerr << "Error: sparse mode needs <A_file> <d_factor>\n";
            return 1;
        }
        A_file      = argv[6];
        d_factor    = std::stod(argv[7]);
        dfactor_idx = 7;
    } else {
        if (argc < 9) {
            std::cerr << "Error: FEM mode needs <K_file> <M_file> <V_file> <d_factor>\n";
            return 1;
        }
        K_file      = arg5;
        M_file      = argv[6];
        V_file      = argv[7];
        d_factor    = std::stod(argv[8]);
        dfactor_idx = 8;
    }

    auto opt_long = [&](int rel, int64_t def) {
        int idx = dfactor_idx + rel;
        return (argc > idx) ? std::stol(argv[idx]) : def;
    };
    auto opt_double = [&](int rel, double def) {
        int idx = dfactor_idx + rel;
        return (argc > idx) ? std::stod(argv[idx]) : def;
    };
    int64_t sketch_nnz  = opt_long(1, 4);
    int64_t block_size  = opt_long(2, 0);
    bool compute_cond   = (opt_long(3, 0) != 0);
    int64_t method_mask = opt_long(4, 31);   // bits 0..4: CQRRT_linop, CholQR, sCholQR3, sCholQR3_basic, CholQR2
    T noise_level       = (T)opt_double(5, 0.05);
    double omega        = opt_double(6, 0.0);
    int64_t power_j     = opt_long(7, 1);
    // irlsq_reg-only knobs (positions after rspec's omega/power_j):
    double kappa_target = opt_double(8, 1.0);    // V column-scaling spread (1 = native)
    double mu_factor    = opt_double(9, 10.0);   // mu = mu_factor * u(precond_prec)
    std::string precond_prec = (argc > dfactor_idx + 10) ? std::string(argv[dfactor_idx + 10]) : "single";
    // Inner-CG controls (appended 2026-07-27; both optional and backward compatible).
    // ir_max_inner <= 0 keeps the 200 default; ir_inner_tol < 0 keeps eps^0.85.
    g_ir_max_inner = (int)opt_long(11, 200);
    g_ir_inner_tol = opt_double(12, -1.0);
    // Per-round CG residual drop (Oleg's restart pacing, 2026-08-07). Slot 13
    // previously carried ir_inner_restarts, which the paced scheme obsoletes
    // (every round IS a true-residual restart). Reject values >= 1 so a stale
    // script passing the old integer restart count fails loudly rather than
    // silently running near-empty rounds. 0 restores legacy fixed-tol rounds.
    g_ir_round_drop = opt_double(13, 1e-4);
    if (g_ir_round_drop < 0.0 || g_ir_round_drop >= 1.0) {
        std::cerr << "Error: slot 13 is [ir_round_drop] since 2026-08-07 (was "
                     "[ir_inner_restarts]); it must lie in [0, 1). Regenerate "
                     "the job scripts.\n";
        return 1;
    }
    // Outer-round cap (2026-08-07, Max: 20, was 4). Under the paced scheme
    // rounds are shallow and outer_tol exits early, so well-preconditioned
    // methods use a handful of rounds while weakly-preconditioned ones get
    // room to keep descending instead of being budget-truncated.
    g_ir_n_steps = (int)opt_long(14, 20);
    if (g_ir_n_steps < 1) {
        std::cerr << "Error: ir_n_steps must be >= 1.\n";
        return 1;
    }
    // Outer early exit (2026-08-06, structure unification with the Toeplitz
    // pcg_ne): refinement stops once ||b - Jx||/||b|| meets this, capped at
    // ir_n_steps. < 0 keeps the default 10*eps of the solve precision (the
    // "refine until done" reading); 0 disables the check (always run all steps).
    g_ir_outer_tol = opt_double(15, -1.0);
    // Positions beyond 15: reject rather than ignore, so a stale job script fails
    // loudly instead of silently running a different experiment than it encodes.
    if (argc > dfactor_idx + 16) {
        std::cerr << "Error: [ir_warm_start]/[bp_warm_start] CLI knobs were removed "
                     "2026-08-05 (Blendenpik-only warm start, both variants always run). "
                     "Regenerate the job scripts.\n";
        return 1;
    }

    if (mode == "irlsq_reg" && sparse_mode) {
        std::cerr << "Error: mode 'irlsq_reg' is FEM-only; sparse input is not supported.\n";
        return 1;
    }

    if (mode == "rspec") {
        if (sparse_mode) {
            std::cerr << "Error: mode 'rspec' is FEM-only; sparse input is not supported.\n";
            return 1;
        }
        if (power_j < 1 || power_j > 3) {
            std::cerr << "Error: power_j must be in {1, 2, 3}; got " << power_j << "\n";
            return 1;
        }
    }

    std::cout << "=== CQRRT linop benchmark ===\n";
    std::cout << "  mode: " << mode << "\n";
    if (sparse_mode) {
        std::cout << "  Input mode: sparse (single-matrix SparseLinOp)\n"
                  << "  A file: " << A_file << "\n";
    } else {
        std::cout << "  Input mode: FEM composite (J = L^{-1} K V with L = chol(M))\n"
                  << "  K file: " << K_file << "\n"
                  << "  M file: " << M_file << "\n"
                  << "  V file: " << V_file << "\n";
    }
    std::cout << "  d_factor: " << d_factor << "\n"
              << "  sketch_nnz: " << sketch_nnz << "\n"
              << "  block_size: " << block_size << "\n"
              << "  compute_cond: " << (compute_cond ? "yes" : "no") << "\n"
              << "  method_mask: " << method_mask
              << " (linop=" << (method_mask&1)
              << " CholQR=" << ((method_mask>>1)&1)
              << " sCholQR3=" << ((method_mask>>2)&1)
              << " sCholQR3_basic=" << ((method_mask>>3)&1)
              << " CholQR2=" << ((method_mask>>4)&1) << ")\n"
              << "  noise_level: " << noise_level << "\n"
              << "  omega: " << omega << "\n"
              << "  power_j: " << power_j << "\n"
              << "  num_runs: " << num_runs << "\n"
#ifdef _OPENMP
              << "  OpenMP threads: " << omp_get_max_threads() << "\n\n";
#else
              << "  OpenMP threads: 1\n\n";
#endif

    // ================================================================
    // Sparse mode: SparseLinOp directly, no Cholesky.
    // ================================================================
    if (sparse_mode) {
        int64_t m, n, nnz_A;
        auto A_csr = load_csr_verbose<T>("A", A_file, m, n, nnz_A);
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

        if (m < n) {
            std::cerr << "Error: matrix must be overdetermined (m >= n), got " << m << "x" << n << "\n";
            return 1;
        }

        // Sparse irlsq b construction: x_true ~ U(-1,1)^n, b = A x_true + scaled Gaussian noise.
        std::vector<T> x_true(n, (T)0);
        {
            std::mt19937 rng(42);
            std::uniform_real_distribution<T> dist((T)-1.0, (T)1.0);
            for (auto& v : x_true) v = dist(rng);
        }
        std::vector<T> b_clean(m, (T)0), noise_vec(m, (T)0);
        A_linop(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, 1, n, (T)1.0, x_true.data(), n, (T)0.0, b_clean.data(), m);
        T b_clean_norm = blas::nrm2(m, b_clean.data(), 1);
        std::mt19937 noise_rng(13);
        std::normal_distribution<T> N01(0, 1);
        for (auto& v : noise_vec) v = N01(noise_rng);
        T raw_noise_norm = blas::nrm2(m, noise_vec.data(), 1);
        T scale = noise_level * b_clean_norm / raw_noise_norm;
        std::vector<T> b(m, (T)0);
        for (int64_t i = 0; i < m; ++i) b[i] = b_clean[i] + scale * noise_vec[i];
        std::cout << "Synthetic LS problem: ||x_true|| = " << blas::nrm2(n, x_true.data(), 1)
                  << ",  ||b|| = " << blas::nrm2(m, b.data(), 1) << "\n\n";

        return run_benchmark_inner<T, RNG>(
            A_linop, m, n, nnz_A, output_dir, num_runs,
            d_factor, sketch_nnz, block_size,
            compute_cond,
            method_mask, noise_level,
            0L /*chol_time_us*/,
            "A (" + A_file + ")", A_file,
            &b, &x_true);
    }

    // ================================================================
    // irlsq_reg mode (FEM-only): regularized augmented-operator preconditioner
    // with independent preconditioner / solve precisions. Loads + builds its own
    // (kappa-scaled, cast-down) chains, so it intercepts before the plain FEM load.
    // <precision> (argv[1]) is the SOLVE precision; precond precision is a CLI knob.
    // ================================================================
    if (mode == "irlsq_reg") {
        std::string solve_prec_str = (sizeof(T) == 8) ? "double" : "single";
        std::cout << "\n=== IR-LSQ-reg mode (regularized augmented operator) ===\n"
                  << "  kappa_target: " << kappa_target << "\n"
                  << "  mu_factor: "    << mu_factor    << "\n"
                  << "  noise_level: "  << (double)noise_level << "\n"
                  << "  precond_prec: " << precond_prec << "\n"
                  << "  solve_prec: "   << solve_prec_str << "\n\n";
        return dispatch_irlsq_reg<T, RNG>(precond_prec, K_file, M_file, V_file,
            output_dir, num_runs, (double)d_factor, sketch_nnz, block_size,
            method_mask, kappa_target, mu_factor, (double)noise_level, solve_prec_str);
    }

    // ================================================================
    // FEM mode: load K, V; Cholesky-factorize M; build J = L^{-1} K V.
    // ================================================================
    int64_t m_K, n_K, nnz_K;
    auto K_csr = load_csr_verbose<T>("K (stiffness)", K_file, m_K, n_K, nnz_K);
    if (m_K != n_K) {
        std::cerr << "Error: K must be square; got " << m_K << " x " << n_K << "\n";
        return 1;
    }
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> K_op(m_K, m_K, K_csr);

    int64_t m_V, n_V, nnz_V;
    auto V_csr = load_csr_verbose<T>("V (prolongation)", V_file, m_V, n_V, nnz_V);
    if (m_V != m_K) {
        std::cerr << "Error: V row count (" << m_V << ") must match K size (" << m_K << ")\n";
        return 1;
    }
    if (m_V < n_V) {
        std::cerr << "Error: need tall V (m_fine >= n_coarse); got " << m_V << " x " << n_V << "\n";
        return 1;
    }
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> V_op(m_V, n_V, V_csr);

    std::cout << "Factorizing M = L L^T from " << M_file << "... " << std::flush;
    RandLAPACK_extras::linops::CholSolverLinOp<T> L_inv_op(M_file, /*half_solve=*/true);
    auto chol_start = steady_clock::now();
    L_inv_op.factorize();
    auto chol_stop = steady_clock::now();
    long chol_time_us = duration_cast<microseconds>(chol_stop - chol_start).count();
    std::cout << "done (" << chol_time_us << " us)\n";
    if (L_inv_op.n_rows != m_K) {
        std::cerr << "Error: M size (" << L_inv_op.n_rows << ") must match K size ("
                  << m_K << ")\n";
        return 1;
    }

    int64_t m = m_V;
    int64_t n = n_V;

    // -------- RSPEC mode (Algorithm 4): build C = L^T X^{-1} L and V_app = C^j V_FEM. --------
    if (mode == "rspec") {
        std::cout << "\n=== RSPEC mode (Algorithm 4) ===\n"
                  << "  omega: "   << omega   << "\n"
                  << "  power_j: " << power_j << "\n";

        // Load M as a CSR so we can form X = K - omega*M via shared-pattern axpby.
        std::cout << "Loading M (mass) from " << M_file << " for X assembly... " << std::flush;
        int64_t m_M, n_M, nnz_M;
        auto M_csr = load_csr<T>(M_file, m_M, n_M, nnz_M);
        std::cout << "done (" << m_M << " x " << n_M << ", nnz=" << nnz_M << ")\n";
        if (m_M != m_K || n_M != m_K) {
            std::cerr << "Error: M size (" << m_M << " x " << n_M
                      << ") must match K size " << m_K << "\n";
            return 1;
        }

        // 1. X = K - omega * M (CSR, shares sparsity with K and M).
        std::cout << "Forming X = K - omega*M ... " << std::flush;
        auto X_csr = RandLAPACK_extras::sparse_axpby_shared_pattern<T, int64_t>(
            (T)1.0, K_csr, -(T)omega, M_csr);
        std::cout << "done (nnz=" << X_csr.nnz << ")\n";

        // 2. Factor X via sparse Cholesky.
        //
        // X = K - omega*M is SPD as long as omega < lambda_min(K, M) (the smallest
        // generalized eigenvalue of the (K, M) pencil). For omega = 0 and the
        // near-zero shifts this application uses, X stays positive definite, so a
        // sparse Cholesky factorization suffices: it confines Eigen to the
        // factorization and applies X^{-1} via RandBLAS sparse TRSM (CholSolverLinOp).
        // An interior shift (omega >= lambda_min) would make X indefinite; Cholesky
        // would then (correctly) fail and an indefinite solver would be needed.
        std::cout << "Factorizing X = L L^T (sparse Cholesky) ... " << std::flush;
        RandLAPACK_extras::linops::CholSolverLinOp<T> X_inv_op(X_csr, /*half_solve=*/false);
        auto x_fact_start = steady_clock::now();
        try {
            X_inv_op.factorize();
        } catch (RandBLAS::Error const& e) {
            auto x_fact_stop = steady_clock::now();
            long x_fact_us = duration_cast<microseconds>(x_fact_stop - x_fact_start).count();
            std::cerr << "\nCholesky factorization of X failed (X not SPD -- omega at/above "
                         "lambda_min, or near an eigenvalue): " << e.what() << "\n";

            // Write a single sentinel row to the CSV and return cleanly.
            std::string results_file = output_dir + "/" + make_run_timestamp() + "_rspec_results.csv";

            std::vector<rspec_result<T>> stub;
            rspec_result<T> r{};
            r.m = m_K; r.n = n_V;
            r.run_idx = 0;
            r.alg_name = "factorize_failed";
            r.qr_status = -99;
            r.qr_time_us = -1;
            r.factor_time_us = x_fact_us;
            r.rspec_total_us = x_fact_us;
            r.orth_error = std::numeric_limits<T>::quiet_NaN();
            int top_k = (int)std::min<int64_t>(10, n_V);
            r.top_eigvals.assign(top_k, std::numeric_limits<T>::quiet_NaN());
            r.top_residuals.assign(top_k, std::numeric_limits<T>::quiet_NaN());
            stub.push_back(r);
            write_rspec_csv<T>(results_file, stub, m_K, n_V, num_runs,
                               K_file, M_file, V_file, omega, power_j,
                               sketch_nnz, block_size, method_mask, top_k);
            std::cout << "Stub CSV written to " << results_file << " (qr_status=-99).\n";
            return 0;
        }
        auto x_fact_stop = steady_clock::now();
        long x_factor_time_us = duration_cast<microseconds>(x_fact_stop - x_fact_start).count();
        std::cout << "done (" << x_factor_time_us << " us)\n";

        // 4. L_op: wrap the L = chol(M) factor as a sparse linop (non-owning view).
        auto L_csc = L_inv_op.make_L_csc();
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T, int>> L_op(m_K, m_K, L_csc);

        // 5. Compose C = L^T * X^{-1} * L.
        RandLAPACK::linops::TransposedOp L_T_op(L_op);
        RandLAPACK::linops::CompositeOperator inner_op(m_K, m_K, X_inv_op, L_op);
        inner_op.block_size = block_size;
        RandLAPACK::linops::CompositeOperator C_op(m_K, m_K, L_T_op, inner_op);
        C_op.block_size = block_size;

        // 6. V_app = C^j * V_FEM (implicit).
        RandLAPACK::linops::PowerOp Cj_op(C_op, (int)power_j);
        RandLAPACK::linops::CompositeOperator V_app_op(m_K, n_V, Cj_op, V_op);
        V_app_op.block_size = block_size;

        std::cout << "Operator chain: V_app = C^" << power_j
                  << " * V_FEM  (" << m_K << " x " << n_V << ")\n\n";

        long total_factor_us = chol_time_us + x_factor_time_us;
        return run_rspec_benchmark<T, RNG>(
            V_app_op, C_op,
            m_K, n_V, output_dir, num_runs,
            d_factor, sketch_nnz, block_size,
            method_mask, total_factor_us,
            K_file, M_file, V_file, omega, power_j);
    }

    RandLAPACK::linops::CompositeOperator KV_op(m, n, K_op, V_op);
    KV_op.block_size = block_size;
    RandLAPACK::linops::CompositeOperator J_op(m, n, L_inv_op, KV_op);
    J_op.block_size = block_size;
    std::cout << "Composite operator J = L^{-1} K V : " << m << " x " << n << "\n\n";

    // FEM irlsq b construction: b = L^{-1} * r, r ~ N(0, 1)^{m_K}. No ground truth x_true.
    std::vector<T> r(m_K, (T)0);
    {
        std::mt19937 rng_b(13);
        std::normal_distribution<T> N01(0, 1);
        for (auto& v : r) v = N01(rng_b);
    }
    std::vector<T> b(m_K, (T)0);
    L_inv_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m_K, 1, m_K, (T)1.0, r.data(), m_K, (T)0.0, b.data(), m_K);
    std::cout << "FEM IR-LSQ b: ||b|| = " << blas::nrm2(m_K, b.data(), 1)
              << " (b = L^{-1} r, r ~ N(0,1)^M)\n\n";

    return run_benchmark_inner<T, RNG>(
        J_op, m, n, nnz_K, output_dir, num_runs,
        d_factor, sketch_nnz, block_size,
        compute_cond,
        method_mask, noise_level,
        chol_time_us,
        "L^{-1} K V (M=" + M_file + ")", K_file,
        &b, nullptr /*x_true_ptr: FEM has no ground truth*/);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <mode>\n"
                  << "    sparse mode: 'sparse' <A_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [compute_cond] [method_mask] [noise_level]\n"
                  << "    FEM mode:    <K_file> <M_file> <V_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [compute_cond] [method_mask] [noise_level]"
                  << " [omega] [power_j]\n"
                  << "  mode  = irlsq | rspec | irlsq_reg   (rspec/irlsq_reg are FEM-only)\n";
        return 1;
    }

    std::string precision = argv[1];
    if (precision == "double") {
        return run_benchmark<double>(argc, argv);
    } else if (precision == "float" || precision == "single") {
        return run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Unknown precision: " << precision << " (use 'double'/'float'/'single')\n";
        return 1;
    }
}
