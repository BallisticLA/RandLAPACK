// Q-less QR benchmark: regularized augmented-operator iterative-refinement least squares
// on a FEM composite operator.
//
// Pipeline:
//   1. Load the FEM triple: K (stiffness), M (mass), V (prolongation), as .mtx files.
//   2. Cholesky-factorize M = L L^T via CholSolverLinOp(half_solve=true).
//   3. Build J = L^{-1} K V as a doubly-nested CompositeOperator
//      J = CompositeOperator(L_inv_op, CompositeOperator(K_op, V_op)).
//   4. Run Q-less QR on the augmented operator [J; mu*I] via one of 5 variants
//      (CQRRTO_linop, CholQR, sCholQR3, sCholQR3_basic, CholQR2), selected by the mask,
//      giving R = chol(J^T J + mu^2 I).
//   5. Solve with IterRefineLSQ from x_0 = 0, preconditioned by that R. The Blendenpik
//      family rows (mask bits 32/64) instead solve on the base operator with their own
//      sketch-QR preconditioner, and bit 128 runs the refinement engine with no factor.
//
// Run with --help for the argument list. The legacy positional form is still accepted;
// `--mode` is gone because only the regularized path remains (the earlier `irlsq`,
// `sparse` and `rspec` modes were retired along with their dead code).
//
// method_mask = bitmask of methods (default 0b11111 = 31)
//                 bit 0 (  1): CQRRTO_linop (TRSM_IDENTITY)
//                 bit 1 (  2): CholQR
//                 bit 2 (  4): sCholQR3
//                 bit 3 (  8): sCholQR3_basic
//                 bit 4 ( 16): CholQR2
//                 bit 5 ( 32): Blendenpik, published (warm + cold rows; not in the
//                              default mask)
//                 bit 6 ( 64): Blendenpik refined by the shared engine (warm + cold
//                              rows; see benchmark/refined_blendenpik.hh)
//                 bit 7 (128): unpreconditioned (irlsq_reg only). The shared
//                              refinement engine on the raw operator with
//                              R = nullptr: no factor is built, so the row is the
//                              reference point without a preconditioner, the
//                              same row the Toeplitz benchmark carries.
//               rspec mode accepts bits 0-4 only and warns on 32/64/128; irlsq
//               (sparse) mode rejects 128.
//   The campaign mask 127 = bits 0-6 (all five Q-less methods + both Blendenpik
//   families); 128 is run on its own (mask 128) as an overlay row.
//
// Trailing optional args after precond_prec (irlsq / irlsq_reg):
//   [ir_max_inner] inner-CG iteration cap per outer refinement step (default 200).
//                  With 2 outer steps this is what produced the fixed 400-iteration
//                  ceiling in earlier CSVs. Pass <= 0 to keep the default.
//   [ir_inner_tol] inner-CG relative-residual tolerance (default: eps^0.85 in the
//                  working precision, ~4.9e-14 in double). Pass < 0 to keep it.
//                  In paced mode (ir_round_drop > 0) this is the ABSOLUTE floor at
//                  which a round stops at once; pass 0 to disable that floor
//                  (rounds then always run to the ir_round_drop factor, the rule
//                  the Toeplitz benchmark used before 2026-09-14). 0 is rejected
//                  in legacy mode, where the value is the per-round tolerance.
//   [ir_round_drop] per-round inner-CG residual drop (default 1e-4; restart
//                  pacing, replacing [ir_inner_restarts] in this slot).
//                  Each round's CG stops after this relative drop and the outer
//                  loop restarts against the TRUE residual; ir_inner_tol survives
//                  as the absolute floor at which rounds stop immediately. Pass 0
//                  for legacy fixed-tolerance rounds. Values >= 1 are rejected so
//                  stale scripts passing the old restart count fail loudly.
//   [ir_n_steps]   outer-round cap (default 20; previously 4). Under the
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
// Warm-start policy: the sketch-and-solve x0 warm start is Blendenpik-only.
// Method mask bit 32 runs TWO variants, "Blendenpik" (its own warm start, the
// published configuration) and "Blendenpik_cold" (x0 = 0), and IterRefineLSQ
// always starts from x0 = 0 (per collaborator request). The former
// [ir_warm_start] and [bp_warm_start] CLI knobs are removed: warm x0 is
// Blendenpik's forward-error edge, not IterRefineLSQ's.

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
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <type_traits>

// Extras utilities (Eigen-dependent)
#include "../../extras/misc/ext_util.hh"
#include "../../extras/misc/ext_sparse_axpy.hh"
#include "../../extras/linops/ext_cholsolver_linop.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrto_bench_common.hh"

// Linops algorithms
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "rl_blendenpik.hh"
#include "../refined_blendenpik.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Helper families shared with bench_toeplitz_ls (cqrrto_bench_common.hh):
// stop-reason maps, env provenance, the rounds-CSV schema/row writer, the
// chol-shift fold, and the power-2-norm and blocked-orth-error estimators.
using RandLAPACK::bench::pcg_stop_reason;
using RandLAPACK::bench::lsqr_stop_reason;
using RandLAPACK::bench::write_env_line;
using RandLAPACK::bench::kRoundsCsvHeader;
using RandLAPACK::bench::write_round_row;
using RandLAPACK::bench::fold_chol_shift;
using RandLAPACK::bench::bench_chol_max_retries;
using RandLAPACK::bench::estimate_op_2norm;
using RandLAPACK::bench::compute_orth_error_explicit;
using RandLAPACK::bench::quote_join_argv;
using RandLAPACK::bench::get_hostname;
using RandLAPACK::bench::write_host_line;

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
// Result struct (unified: sentinel values for fields irrelevant to the mode)
// ============================================================================

template <typename T>
struct bench_result {
    int64_t m, n;
    int64_t run_idx;
    std::string alg_name;
    T noise_level;

    int qr_status;         // 0 on success
    long qr_time_us;       // -1 if QR failed
    // -1 = no Cholesky ran in this row (Blendenpik family, unpreconditioned,
    // never overwrites this field) or QR failed before a retry count existed.
    // 0 = Cholesky ran unshifted. The five Q-less branches always overwrite
    // this on both success and failure, so real Cholesky rows are unaffected.
    int  chol_retries = -1;
    // The shift the retries actually applied. -1 = no Cholesky in this row
    // (Blendenpik family) or QR failed before a shift record existed; 0 =
    // every pass unshifted. abs = pass-1 absolute shift, i.e. the
    // Tikhonov-equivalent regularization baked into the returned R; rel = max
    // over passes of shift/trace(G), the scale-free severity. A bare retry
    // count cannot separate a rounding-level rescue from a spectrum-truncating
    // one; these can.
    T chol_shift_abs = -1;
    T chol_shift_rel = -1;

    // Q-factor orthogonality: ||Q^T Q - I||_F / sqrt(n), computed for all methods.
    T orth_error;

    // Solution vector, kept only until the post-pass backward-error evaluation
    // (build_kw_reference in cqrrto_bench_common.hh); cleared afterwards.
    std::vector<T> x_hat;

    // IR-LSQ-mode fields
    long ir_total_us;
    long ir_setup_us = 0;  // warm-start x0 build time, its OWN slot (NOT inside
                           // ir_total_us; 0 = cold start). Assigned for the
                           // Blendenpik rows.
    int  ir_outer_iters;
    int  ir_inner_iters_total;
    int  lsqr_iters = 0;   // LSQR iterations, where an LSQR phase ran (published
                           // Blendenpik rows only)
    int  engine_status = -1;          // restarted_pcg_ne exit status (-1 = engine not used)
    std::string stop_reason = "n/a";  // named exit condition (see the reason helpers)
    T    x0_relres = (T)-1;           // true relres of the handed-off warm x0 (refine warm rows)
    T    ls_residual_norm;
    T    ls_solution_error;   // -1 sentinel when undefined (FEM irlsq)

    // Inner-CG diagnosis. ir_inner_iters_total alone cannot say whether a
    // solve converged or merely exhausted its budget: both used to report success.
    //   ir_inner_capped   : 1 if ANY outer step hit max_inner (0 = all converged,
    //                       2 = a CG breakdown occurred). Published
    //                       Blendenpik rows carry 0/1 from LSQR's own convergence,
    //                       refine rows carry real kernel diagnoses; -1 only on
    //                       failed builds.
    //   ir_inner_relres   : worst per-step achieved ||Mz-c||/||c|| (NE space).
    //                       CAVEAT: for published Blendenpik rows this column holds
    //                       LSQR's final LS-space relres instead; do not compare the
    //                       two families through this column without saying so.
    //   ir_inner_best_relres / ir_inner_best_iter : the smallest residual seen in the
    //       worst step and the iteration it happened at. best_iter far below the
    //       iteration count means the solve STAGNATED (tolerance below the attainable
    //       floor); best_iter near the count means it was still converging when capped.
    int  ir_inner_capped      = -1;
    T    ir_inner_relres      = (T)-1;
    T    ir_inner_best_relres = (T)-1;
    int  ir_inner_best_iter   = -1;

    // cond(J R^-1): the number that actually says whether the preconditioner works.
    // -1 sentinel when not computed: the compute_cond CLI flag now gates this
    // computation directly (both irlsq and irlsq_reg), so -1 means
    // either compute_cond was off or n exceeded the internal eig cap (16384).
    T    cond_precond = (T)-1;

    // irlsq_reg only: kappa(A) estimate from the regularized R diagonal
    // (max|R_ii| / min|R_ii|; floored near sigma_max/mu when sigma_min < mu).
    // -1 sentinel for the plain irlsq path.
    T    kappa_measured = (T)-1;

    // QR timing breakdown (from algo.times[])
    std::vector<long> qr_breakdown;
    std::vector<long> ir_breakdown;

    // Per-round engine records: filled for
    // every row that ran the shared engine (IR methods and refine rows); empty for
    // published Blendenpik rows and failed builds. Written to the *_rounds.csv sidecar.
    std::vector<int> round_iters, round_status, round_best_iter;
    std::vector<T>   round_relres, round_best_relres, round_ls_relres;
    std::vector<T>   round_be;        // oracle value after each round (-1 when the oracle was off)
    long t_be_us = -1;                // wall time inside the oracle, excluded from every solve time;
                                      // -1 where no engine ran (published Blendenpik rows, failed builds)
    T    be_x0   = (T)-1;             // oracle value of a warm start (refine warm row); -1 otherwise
    T    be_final = (T)-1;            // oracle value of the RETURNED iterate (last round, or x0 on a
                                      // zero-round exit); -1 when the oracle was off or no engine ran

    // RSS WINDOW SEMANTICS, per path:
    //   irlsq / rspec: Q-less rows stop the tracker right after the QR build
    //     (build-only peak, matching the build-phase analytical models);
    //     Blendenpik-family rows stop after run_blendenpik_family returns, so
    //     their window also spans the LSQR/engine solve. analytical_kb models
    //     each window kind correspondingly (build-only vs. build-plus-LSQR).
    //   irlsq_reg: windows are UNIFIED across every algorithm, build + solve,
    //     diagnostics excluded (the orth-loss materialization is moved outside
    //     the window for exactly this reason; see the ordering note in
    //     run_irlsq_reg). NOTE: analytical_kb for the Q-less rows there is still
    //     assigned from the build-only QR model at dispatch time, before the
    //     solve runs, so peak_rss_kb (build+solve) and analytical_kb (build-only)
    //     are no longer measuring the same window for those rows (a known
    //     mismatch this comment records but does not correct).
    long peak_rss_kb;
    long analytical_kb;
};

// Fold a driver's per-pass shift record into the result row (shared fold in
// cqrrto_bench_common.hh; see record note there).
template <typename T, typename TR, size_t N>
static void record_chol_shift(bench_result<TR>& res, const T (&shifts)[N], const T (&traces)[N]) {
    fold_chol_shift(res.chol_shift_abs, res.chol_shift_rel, shifts, traces);
}

// ---- CLI-configurable inner-CG controls -------------------------------------
// File-scope rather than threaded through the runners, whose signatures already take
// 14 parameters. Both are set once in main() from argv before any runner is called and
// are read-only thereafter.
//
// Why they are configurable at all: the inner-CG budget was hard-coded at
// 200 per outer step, which with g_ir_n_steps outer steps produced a fixed iteration
// ceiling in the CSVs, and the tolerance was fixed at eps^0.85 (~4.9e-14 in double), a
// relative-residual target close enough to the floating-point stagnation floor that CG
// can be unable to reach it regardless of preconditioner quality. Exposing both lets a
// diagnostic sweep separate those two effects without a rebuild.
static int    g_ir_max_inner = 200;    // <= 0 => keep the IterRefineLSQ default
static double g_ir_inner_tol = -1.0;   // <  0 => eps^0.85 in the working precision
static double g_be_tol_mult  = 0.0;    // <= 0 => backward-error oracle off (today's behaviour);
                                       // > 0 => stop when the sketched KW estimate <= mult*sqrt(n)*u
static double g_be_tol_eff   = -1.0;   // resolved be_tol, echoed in the header; -1 = off
static double g_ir_round_drop = 1e-4;  // per-round CG drop; 0 = legacy fixed-tol rounds
// Effective inner tolerance / absolute floor: < 0 keeps `dflt` (eps^0.85 in the
// working precision), 0 disables the absolute floor (paced mode only, checked at
// parse time), > 0 is the value. One place so the three call sites agree.
template <typename T>
static T ir_inner_tol_eff(T dflt) {
    if (g_ir_inner_tol < 0.0) return dflt;
    return (T)g_ir_inner_tol;
}
static int    g_ir_n_steps = 50;       // outer-round cap (campaign-canonical 50; the
                                       // cap must not bind before tol + maxit do:
                                       // native_ill CholQR2 genuinely uses 50 rounds.)
static double g_ir_outer_tol = -1.0;   // <0 => 10*eps(solve precision); 0 disables early exit
// (g_ir_warm_start / g_bp_warm_start are gone: IR methods are always
//  cold; Blendenpik runs as two mask-32 variants, warm and cold.)

// Outer-stagnation window override: RANDLAPACK_IR_OUTER_STAG, read
// once. Default 2 (the engine default); 0 disables the LS-floor exit. The
// knob exists for the CholQR full-precision diagnostic cell, which must show
// the flat trajectory rather than exit at it. Env rather than CLI so the
// campaign arg layout stays frozen.
static int ir_outer_stag_window() {
    static const int w = []() {
        const char* s = std::getenv("RANDLAPACK_IR_OUTER_STAG");
        if (s == nullptr || *s == '\0') return 2;
        char* end = nullptr;
        long v = std::strtol(s, &end, 10);
        // atoi silently returned 0 on garbage, which silently DISABLES the
        // LS-floor exit rather than erroring (0 is also a legal value chosen
        // for that purpose, so a typo cannot be told apart from intent).
        if (end == s || *end != '\0') {
            std::cerr << "FATAL: RANDLAPACK_IR_OUTER_STAG='" << s
                      << "' is not a valid integer; refusing to silently run "
                         "with a possibly-disabled LS-floor exit. Fix or unset it.\n";
            std::exit(1);
        }
        return (int)v;
    }();
    return w;
}

// Full argv, space-joined and double-quoted, set once in run_benchmark() from
// argc/argv; echoed in every results CSV header so a CSV can be
// traced back to the exact invocation that produced it without a side log.
// quote_join_argv itself is shared (cqrrto_bench_common.hh).
static std::string g_argv_line;

// Environment provenance for every results CSV: the shared env line
// (cqrrto_bench_common.hh) plus this file's own IR-knob echo.
static void write_env_provenance(std::ofstream& out) {
    write_host_line(out);
    write_env_line(out);
    out << "# ir knobs: max_inner=" << g_ir_max_inner << " inner_tol=" << g_ir_inner_tol
        << " round_drop=" << g_ir_round_drop << " n_steps=" << g_ir_n_steps
        << " outer_tol=" << g_ir_outer_tol
        << " outer_stag_window=" << ir_outer_stag_window()
        << " be_tol_mult=" << g_be_tol_mult << " be_tol=" << g_be_tol_eff
        << " kw_sketch_nnz=" << RandLAPACK::bench::kKWSketchNNZ << "\n";   // the oracle's sketch, not --sketch-nnz
}

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
    //   Breakdown (2): solver failed outright
    //   HitCap    (1): ran out of budget while still descending
    //   Stagnated (3): reached its floor and stopped; benign, best iterate returned
    //   Converged (0): met the tolerance
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

// Same diagnosis, from a raw engine history (the refine rows bypass IterRefineLSQ).
// Identical severity ranking.
template <typename T>
static void record_inner_cg_diagnosis(const RandLAPACK::PCGRoundHistory<T>& h,
                                      bench_result<T>& res) {
    if (h.status.empty()) return;
    auto severity = [](int status) -> int {
        switch (status) {
            case 2:  return 3;   // Breakdown
            case 1:  return 2;   // HitCap
            case 3:  return 1;   // Stagnated
            default: return 0;   // Converged
        }
    };
    size_t worst_idx = 0;
    for (size_t i = 1; i < h.status.size(); ++i)
        if (severity(h.status[i]) > severity(h.status[worst_idx])) worst_idx = i;
    if (h.status[worst_idx] == 0) {
        for (size_t i = 0; i < h.relres.size(); ++i)
            if (h.relres[i] > h.relres[worst_idx]) worst_idx = i;
    }
    res.ir_inner_capped      = h.status[worst_idx];
    res.ir_inner_relres      = h.relres[worst_idx];
    res.ir_inner_best_relres = h.best_relres[worst_idx];
    res.ir_inner_best_iter   = h.best_iter[worst_idx];
}

// Round-record copiers for the *_rounds.csv sidecar.
template <typename T>
static void copy_round_records(const RandLAPACK::PCGRoundHistory<T>& h, bench_result<T>& res) {
    res.round_iters       = h.iters;
    res.round_status      = h.status;
    res.round_best_iter   = h.best_iter;
    res.round_relres      = h.relres;
    res.round_best_relres = h.best_relres;
    res.round_ls_relres   = h.ls_relres;
    res.round_be          = h.be;
}
template <typename T>
static void record_ir_outputs(const RandLAPACK::IterRefineLSQ<T>& ir, bench_result<T>& res) {
    res.engine_status = ir.engine_status;
    res.stop_reason   = pcg_stop_reason(ir.engine_status);
    res.round_iters       = ir.inner_iters_per_step;
    res.round_status      = ir.inner_status_per_step;
    res.round_best_iter   = ir.inner_best_iter_per_step;
    res.round_relres      = ir.inner_relres_per_step;
    res.round_best_relres = ir.inner_best_relres_per_step;
    res.round_ls_relres   = ir.ls_relres_per_step;
    res.round_be          = ir.be_per_step;
}

// Shared method-mask decode, to avoid per-path copies diverging (the rspec
// copy once silently ignored bits 32/64; the console echo once showed bits
// 0-4 only). with_blendenpik = false (rspec) warns on 32/64 instead.
static std::vector<std::string> decode_method_mask(int64_t method_mask, bool with_blendenpik) {
    std::vector<std::string> algs;
    if (method_mask & 1)   algs.push_back("CQRRTO_linop");
    if (method_mask & 2)   algs.push_back("CholQR");
    if (method_mask & 4)   algs.push_back("sCholQR3");
    if (method_mask & 8)   algs.push_back("sCholQR3_basic");
    if (method_mask & 16)  algs.push_back("CholQR2");
    if (with_blendenpik) {
        if (method_mask & 32) {   // published: its own sketch-and-solve warm start + cold
            algs.push_back("Blendenpik");
            algs.push_back("Blendenpik_cold");
        }
        if (method_mask & 64) {   // refined by the shared engine (refined_blendenpik.hh)
            algs.push_back("Blendenpik_refine");
            algs.push_back("Blendenpik_cold_refine");
        }
        if (method_mask & 128) algs.push_back("unpreconditioned");   // engine on the raw operator, no factor
    } else if (method_mask & (32 | 64 | 128)) {
        std::cerr << "Warning: method_mask bits 32/64/128 (Blendenpik families, "
                     "unpreconditioned) are not available in this mode and are ignored.\n";
    }
    return algs;
}

// ============================================================================
// Shared Blendenpik-family dispatch (published + refined rows), used by BOTH the
// irlsq path (run_benchmark_inner) and the irlsq_reg path. Extracted after the
// two per-path copies drifted into different wrong accountings: the warm refine
// row silently ran cold in both paths (the exact-string warm_start test was
// false for "Blendenpik_refine"), and the refinement phase was missing from the
// time columns in both, from the iteration count in one.
// ============================================================================
template <typename T, typename RNG, typename GLO>
static void run_blendenpik_family(
    GLO& A_op, const T* b, int64_t m, T* x_ls, int64_t n, T* R_T,
    T d_factor, int64_t sketch_nnz, RandBLAS::RNGState<RNG> state,
    const std::string& alg_name, T tol, T outer_tol_eff,
    const RandLAPACK::BackwardErrorOracle<T>& be_oracle, T be_tol,
    RandLAPACK::PeakRSSTracker& mem, bench_result<T>& res)
{
    const bool is_ref = (alg_name.find("_refine") != std::string::npos);
    const bool warm   = (alg_name == "Blendenpik") || (alg_name == "Blendenpik_refine");
    const int  max_inner = (g_ir_max_inner > 0) ? g_ir_max_inner : 200;
    const int  budget    = max_inner * g_ir_n_steps;   // same budget the IR methods get
    const T    inner_tol = ir_inner_tol_eff<T>(std::pow(std::numeric_limits<T>::epsilon(), (T)0.85));

    if (is_ref) {
        // Refined rows: init_only sketch-and-solve x0, ALL iterative work in the
        // shared engine, no internal LSQR (see benchmark/refined_blendenpik.hh).
        // Engine knobs mirror what IterRefineLSQ passes for the Q-less rows.
        const bool paced = (g_ir_round_drop > 0);
        const T drop      = paced ? (T)g_ir_round_drop : inner_tol;
        const T abs_guard = paced ? inner_tol : (T)0;
        std::fill(x_ls, x_ls + n, (T)0);
        auto rr = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
            A_op, b, m, x_ls, n, d_factor, sketch_nnz, state, warm,
            outer_tol_eff, budget, max_inner, drop, g_ir_n_steps - 1,
            /*stag_window=*/20, /*stag_rel_improve=*/(T)1e-3, abs_guard,
            ir_outer_stag_window(), be_oracle, be_tol);
        res.qr_status = rr.qr_status;
        res.peak_rss_kb = mem.stop();
        if (res.qr_status != 0) return;
        res.qr_time_us  = rr.qr_us;
        res.ir_setup_us = rr.setup_us;                 // x0 build (0 for the cold row)
        res.t_be_us     = rr.history.t_be_us;          // oracle time (already outside solve_us)
        res.be_x0       = rr.history.be_x0;            // oracle value of x0 (warm row; -1 cold/off)
        // Oracle on but nothing measured (no round ran and no warm x0): +inf, so the
        // value can never read as "below the target"; -1 stays "oracle off".
        res.be_final    = !rr.history.be.empty() ? rr.history.be.back()
                        : (rr.history.be_x0 >= 0) ? rr.history.be_x0
                        : (be_tol >= (T)0) ? std::numeric_limits<T>::infinity() : (T)-1;
        res.x0_relres   = rr.x0_relres;                // warm-start quality (-1 cold)
        std::copy(rr.R, rr.R + rr.R_sz, R_T);
        // QR-breakdown slots for Blendenpik-family rows (see the breakdown
        // writer header for the full per-algorithm table): t1=qr, t4=x0 setup,
        // everything else 0. Refine rows only expose the COMBINED sketch+QR
        // time (rr.qr_us is not split further), so it lands whole in t1 and t0
        // (sketch) stays 0, unlike the published branch below, which does
        // have the split.
        res.qr_breakdown.assign(5, 0L);
        res.qr_breakdown[1] = rr.qr_us;
        res.qr_breakdown[4] = rr.setup_us;
        // Refined rows use init_only, so LSQR never runs; the engine workspace
        // that follows (9n + m) is smaller than the sketch term, so the
        // Blendenpik moment is still the peak.
        res.analytical_kb = RandLAPACK::blendenpik_linops_analytical_kb<T>(
            m, n, (double)d_factor, /*warm_start=*/warm, /*with_lsqr=*/false);
        res.ir_total_us          = rr.solve_us;        // the WHOLE refinement solve (was missing)
        res.ir_outer_iters       = rr.rounds;          // rounds actually executed
        res.ir_inner_iters_total = rr.iters;           // engine inner CG only: single unit
        res.lsqr_iters           = 0;                  // no LSQR phase in the redesigned rows
        res.engine_status = rr.status;
        res.stop_reason   = pcg_stop_reason(rr.status);
        record_inner_cg_diagnosis(rr.history, res);
        copy_round_records(rr.history, res);
        // ir_breakdown in the IterRefineLSQ layout [total, inner_cg, trsm, fwd, adj, other].
        long op_outer = (rr.t_fwd_us  - rr.history.t_fwd_inner_us)
                      + (rr.t_adj_us  - rr.history.t_adj_inner_us)
                      + (rr.t_trsm_us - rr.history.t_trsm_inner_us);
        long other = rr.solve_us - rr.history.t_inner_us - op_outer;
        if (other < 0) other = 0;
        res.ir_breakdown = {rr.solve_us, rr.history.t_inner_us, rr.t_trsm_us,
                            rr.t_fwd_us, rr.t_adj_us, other};
        return;
    }

    // Published rows: sketch + QR + LSQR (warm = its own sketch-and-solve x0).
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, tol);
    bp.nnz        = sketch_nnz;
    bp.warm_start = warm;
    bp.max_iters  = budget;
    std::fill(x_ls, x_ls + n, (T)0);
    res.qr_status = bp.call(A_op, b, m, x_ls, n, d_factor, state);
    res.peak_rss_kb = mem.stop();
    if (res.qr_status != 0) return;
    res.qr_time_us  = bp.times[0] + bp.times[1];       // sketch + QR
    res.ir_setup_us = warm ? bp.times[4] : 0;          // warm x0 build, its own column
    std::copy(bp.R_out, bp.R_out + bp.R_out_sz, R_T);
    // QR-breakdown slots for Blendenpik-family rows (see the breakdown writer
    // header): t0=sketch, t1=qr, t4=x0 setup (0 for the cold row, since x0/Sb/r0
    // are only allocated when warm_start||init_only); everything else 0.
    res.qr_breakdown.assign(5, 0L);
    res.qr_breakdown[0] = bp.times[0];
    res.qr_breakdown[1] = bp.times[1];
    res.qr_breakdown[4] = bp.times[4];
    // Published rows run LSQR, so the LSQR workspace is live at the peak.
    res.analytical_kb = RandLAPACK::blendenpik_linops_analytical_kb<T>(
        m, n, (double)d_factor, /*warm_start=*/warm, /*with_lsqr=*/true);
    // No IR loop: reuse the diagnosis columns from LSQR's own convergence signals.
    res.ir_inner_capped = bp.converged ? 0 : 1;
    res.ir_inner_relres = bp.final_relres;
    res.ir_total_us          = bp.times[2];
    res.ir_outer_iters       = 1;
    res.ir_inner_iters_total = bp.lsqr_iters;   // LSQR iters in the CG-iters slot (published rows only)
    res.lsqr_iters           = bp.lsqr_iters;
    res.engine_status = -1;                     // the pcg engine did not run
    res.stop_reason   = lsqr_stop_reason(bp.converged, bp.lsqr_stop_test);
    // ir_breakdown from LSQR's op split.
    if (bp.lsqr_op_times.size() >= 3) {
        long fwd = bp.lsqr_op_times[0], adj = bp.lsqr_op_times[1], trsm = bp.lsqr_op_times[2];
        long other = bp.times[2] - (fwd + adj + trsm);
        if (other < 0) other = 0;
        res.ir_breakdown = {bp.times[2], 0, trsm, fwd, adj, other};   // no kernel split in LSQR
    }
}

// Per-round sidecar writer: one row per (algorithm, run, round) for
// every row that ran the shared engine.
template <typename T>
static void write_rounds_csv(const std::string& filename,
                             const std::vector<bench_result<T>>& results) {
    std::ofstream out(filename);
    out << "# Per-round engine records (restarted_pcg_ne / IterRefineLSQ).\n"
        << "# inner_status: 0 Converged, 1 HitCap, 2 Breakdown, 3 Stagnated.\n"
        << "# be_kw: sketched Karlson-Walden backward error after the round, relative to ||A||_F; -1 when the oracle was off.\n"
        << kRoundsCsvHeader;
    for (const auto& r : results) {
        for (size_t k = 0; k < r.round_iters.size(); ++k) {
            write_round_row(out, r.alg_name, r.run_idx, k + 1,
                r.round_iters[k], r.round_status[k], r.round_relres[k],
                r.round_best_relres[k], r.round_best_iter[k], r.round_ls_relres[k],
                r.round_be[k]);
        }
    }
}

// estimate_op_2norm and compute_orth_error_explicit are shared with
// bench_toeplitz_ls; see cqrrto_bench_common.hh (using-declared above).

// Write one breakdown row: pads/truncates the phase vector to exactly 18
// columns (t0..t17: sCholQR3's 18 slots and sCholQR3_basic's 15 are not cut to
// 11) and appends `total_val` as a dedicated final
// t_total column so phase bars can be validated against the row's own
// authoritative total (qr_time_us / ir_total_us) without inferring it from
// vector position, which used to differ silently per algorithm.
static void write_breakdown_row(std::ofstream& out, const std::string& alg,
                                int64_t run_idx, const char* phase,
                                const std::vector<long>& v, long total_val) {
    out << alg << "," << run_idx << "," << phase;
    for (int i = 0; i < 18; ++i)
        out << "," << (i < (int)v.size() ? v[i] : 0L);
    out << "," << total_val << "\n";
}

template <typename T>
static void write_irlsq_breakdown(
    const std::string& filename,
    const std::vector<bench_result<T>>& results,
    const std::string& mode_label)
{
    std::ofstream out(filename);
    out << "# " << mode_label << " Benchmark runtime breakdown (microseconds)\n"
        << "# QR breakdown layout depends on algorithm:\n"
        << "#   CQRRTO_linop   (t0-t10):  alloc,saso,qr,precond_inv,fwd,adj,gemm,chol,finalize,rest,total\n"
        << "#   CholQR        (t0-t5):   alloc,fwd,adj,chol,rest,total                      (t6-t17 = 0)\n"
        << "#   CholQR2       (t0-t10):  alloc,fwd1,adj1,chol1,upd1,fwd2,adj2,gemm2,chol2,upd2,total (t11-t17 = 0)\n"
        << "#   sCholQR3_basic(t0-t14):  alloc,fwd1,adj1,chol1,trsm1=0,fwd_q=0,syrk2,chol2,upd2,\n"
        << "#                            syrk3,chol3,upd3,q_mat,rest,total                  (t15-t17 = 0)\n"
        << "#   sCholQR3      (t0-t17):  alloc,fwd1,adj1,chol1,upd1,fwd2,adj2,gemm2,chol2,upd2,\n"
        << "#                            fwd3,adj3,gemm3,chol3,upd3,q_mat,rest,total\n"
        << "#   Blendenpik-family (t0-t4 only, t5-t17 = 0): t0=sketch, t1=qr, t4=x0 setup (0 for\n"
        << "#     cold rows). Refine rows only expose the combined sketch+QR time, so it lands\n"
        << "#     whole in t1 with t0 left 0.\n"
        << "# t_total is the row's own authoritative total (qr_time_us for the QR phase row,\n"
        << "#   ir_total_us for the IR phase row), independent of how far t0..t17 are populated.\n"
        << "# IR-LSQ breakdown (6, in t0-t5; t6-t17 = 0): outer_total, inner_cg_total, trsm_total,\n"
        << "#   fwd_total, adj_total, other\n"
        << "#   (t_total on the IR row equals ir_total_us; Blendenpik rows carry\n"
        << "#    real IR entries too: refine rows the engine split, published rows the LSQR op\n"
        << "#    split with a 0 inner_cg slot)\n"
        << "algorithm,run,phase,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t_total\n";
    for (const auto& r : results) {
        write_breakdown_row(out, r.alg_name, r.run_idx, "QR", r.qr_breakdown, r.qr_time_us);
        write_breakdown_row(out, r.alg_name, r.run_idx, "IR", r.ir_breakdown, r.ir_total_us);
    }
}


// ============================================================================
// CSV writer: IR-LSQ regularized (irlsq_reg): base columns + regularization /
// mixed-precision metadata (kappa_target, kappa_measured, mu, precond/solve prec)
// ============================================================================

template <typename T>
static void write_irlsq_reg_results(
    const std::string& filename,
    const std::vector<bench_result<T>>& results,
    int64_t m, int64_t n, int64_t nnz_or_zero, const std::string& input_label,
    double d_factor, int64_t sketch_nnz, int64_t block_size, int64_t method_mask,
    double kappa_target, double mu,
    const std::string& precond_prec, const std::string& solve_prec,
    int64_t num_runs, long chol_time_us, double noise_level)
{
    std::ofstream out(filename);
    // irlsq_reg is FEM-only (main() rejects sparse input for this mode), unlike
    // the plain irlsq writer above, which serves both; the title says so.
    out << "# FEM IR-LSQ (regularized augmented operator) Benchmark results\n"
        << "# Date: " << make_run_timestamp() << "\n"
        << "# argv=" << g_argv_line << "\n"
        << "# input=" << input_label << "\n"
        << "# M=" << m << " N=" << n << " nnz=" << nnz_or_zero << "\n"
        << "# noise_level=" << noise_level << "\n"
        << "# chol_time_us=" << chol_time_us << "\n"
        << "# d_factor=" << d_factor << " sketch_nnz=" << sketch_nnz
        << " block_size=" << block_size << "\n"
        << "# method_mask=" << method_mask << "\n"
        << "# num_runs=" << num_runs << "\n"
        << "# kappa_target=" << kappa_target << " mu=" << mu << "\n"
        << "# precond_prec=" << precond_prec << " solve_prec=" << solve_prec << "\n"
        << "# blendenpik=warm+cold (IR methods always cold x0);"
        << " refine rows = init_only x0 + shared engine\n"
        << "# A_hat = [A; mu*I];  R = chol(A^T A + mu^2 I) built in precond_prec,\n"
        << "#   used as right preconditioner for IterRefineLSQ run in solve_prec.\n"
#ifdef _OPENMP
        << "# OpenMP threads: " << omp_get_max_threads() << "\n"
#else
        << "# OpenMP threads: 1\n"
#endif
        ;
    write_env_provenance(out);
    out << "algorithm,run,m,n,qr_status,qr_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,ir_total_us,ir_outer_iters,ir_inner_iters_total,"
           "ls_residual_norm,ls_solution_error,kappa_target,kappa_measured,mu,precond_prec,solve_prec,chol_retries,"
           "ir_inner_capped,ir_inner_relres,ir_inner_best_relres,ir_inner_best_iter,cond_precond,ir_setup_us,"
           "lsqr_iters,engine_status,stop_reason,x0_relres,chol_shift_abs,chol_shift_rel,t_be_us,be_x0,be_final\n";
    // Sentinel note: chol_retries and chol_shift_abs/chol_shift_rel use -1 for
    // "no Cholesky in this row" (Blendenpik family, unpreconditioned) or "QR
    // failed before a retry/shift record existed"; 0 still means "Cholesky
    // ran unshifted" (chol_retries) or "ran, no shift applied" (the shifts).
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
            << r.ir_setup_us << ","
            << r.lsqr_iters << "," << r.engine_status << "," << r.stop_reason << ","
            << std::scientific << std::setprecision(6) << r.x0_relres << ","
            << std::scientific << std::setprecision(6) << r.chol_shift_abs << ","
            << std::scientific << std::setprecision(6) << r.chol_shift_rel << ","
            << r.t_be_us << ","
            << std::scientific << std::setprecision(6) << r.be_x0 << ","
            << std::scientific << std::setprecision(6) << r.be_final
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
// irlsq_reg runner: regularized augmented-operator preconditioner with
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
    bool compute_cond,
    int64_t method_mask, double kappa_target, double mu_factor, double noise_level,
    const std::string& precond_prec_str, const std::string& solve_prec_str)
{
    namespace rl = RandLAPACK::linops;

    // Shared decode (see decode_method_mask and the file-header mask docs).
    std::vector<std::string> selected_algs = decode_method_mask(method_mask, /*with_blendenpik=*/true);
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
    auto chol_ts_t0 = steady_clock::now();
    L_inv_Ts.factorize();
    auto chol_ts_t1 = steady_clock::now();
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
    auto chol_pp_t0 = steady_clock::now();
    L_inv_Pp.factorize();
    auto chol_pp_t1 = steady_clock::now();
    std::cout << "done\n";
    rl::CompositeOperator KV_Pp(m, n, K_op_Pp, V_op_Pp); KV_Pp.block_size = block_size;
    rl::CompositeOperator J_Pp(m, n, L_inv_Pp, KV_Pp);   J_Pp.block_size = block_size;

    // Shared per-cell cost: both M-factorizations (solve + precond
    // precision), measured once, identical for every row in this run.
    const long chol_time_us = duration_cast<microseconds>(chol_ts_t1 - chol_ts_t0).count()
                             + duration_cast<microseconds>(chol_pp_t1 - chol_pp_t0).count();

    // ||A||_2 and ||b|| for the Higham backward-error metric.
    T_solve A_2norm = estimate_op_2norm<T_solve>(J_Ts, m, n, 10);
    T_solve b_norm  = blas::nrm2(m, b.data(), 1);

    // Backward-error oracle (the paper's step-two termination test). The reference is
    // built BEFORE the timed rows when the knob is on: it is per-matrix instrumentation
    // shared by every method, sits outside every row's timer, and only raises the RSS
    // baseline the tracker subtracts. When the knob is off it is built after the rows,
    // for the sidecar, as before.
    const T_solve be_tol = RandLAPACK::bench::resolve_be_tol<T_solve>(g_be_tol_mult, n);
    g_be_tol_eff = (double)be_tol;
    if constexpr (std::is_same_v<T_solve, float>) {
        // A float gesdd of a kappa ~ 1e10 sketch returns singular values that are
        // noise below ~1e-7 sigma_max, exactly the directions the estimate weights
        // most; the number it would print is meaningless. Refuse rather than mislead.
        if (be_tol >= (T_solve)0) {
            std::cerr << "FATAL: --be-tol-mult > 0 requires a double solve precision; "
                         "the backward-error reference is not meaningful in float.\n";
            return 2;
        }
    }
    std::unique_ptr<RandLAPACK::bench::KWBackwardErrorRef<T_solve>> kw_ref_ptr;
    double kw_build_s = 0.0;
    auto build_kw = [&]() {
        if (kw_ref_ptr) return;
        std::cout << "\nBackward-error reference: sketched Karlson-Walden, d=2n=" << 2 * n
                  << ", nnz=" << RandLAPACK::bench::kKWSketchNNZ << " ... " << std::flush;
        auto kw_t0 = steady_clock::now();
        RandBLAS::RNGState<RNG> kw_state((uint32_t)20240914);
        kw_ref_ptr = std::make_unique<RandLAPACK::bench::KWBackwardErrorRef<T_solve>>(
            RandLAPACK::bench::build_kw_reference<T_solve, RNG>(J_Ts, m, n, 2 * n, RandLAPACK::bench::kKWSketchNNZ,
                                                              kw_state, block_size));
        kw_build_s = duration_cast<microseconds>(steady_clock::now() - kw_t0).count() / 1e6;
        std::cout << "done (" << std::fixed << std::setprecision(1) << kw_build_s << " s, ||A||_F="
                  << std::scientific << std::setprecision(6) << (double)kw_ref_ptr->A_fro << ")\n";
    };
    RandLAPACK::BackwardErrorOracle<T_solve> be_oracle;
    if (be_tol >= (T_solve)0) {
        // A failed reference build is fatal here, never a silent fall back to the
        // floor rule: that would produce an era labelled as oracle-terminated that was not.
        try { build_kw(); }
        catch (const std::exception& e) {
            std::cerr << "FATAL: be-tol-mult > 0 requires the backward-error reference and its build failed: "
                      << e.what() << "\n";
            return 2;
        }
        be_oracle = RandLAPACK::bench::make_kw_oracle<T_solve>(*kw_ref_ptr, m, n, b_norm);
    }
    std::cout << "||A||_2 ~ " << A_2norm << ", ||b|| = " << b_norm << "\n";

    // Regularization per the collaborator's spec: mu = mu_factor * u(precond),
    // with mu_factor = 10 giving mu = 10u (u = unit roundoff of the precond
    // precision). NO ||A|| or size scaling: the augmented operator is exactly
    // A_hat = [A; mu*I], Q-less CholeskyQR of which gives R = chol(A^T A + mu^2 I),
    // used as a right preconditioner for the LS problem in A.
    const P_precond mu_P = (P_precond)(mu_factor * (double)unit_roundoff<P_precond>());
    rl::ScaledIdentityOp<P_precond> reg_op(n, mu_P);
    rl::VStackOp<decltype(J_Pp), rl::ScaledIdentityOp<P_precond>> A_hat_Pp(J_Pp, reg_op);
    A_hat_Pp.block_size = block_size;   // caps the blocked-sketch slice width (CQRRTO)
    std::cout << "Augmented operator A_hat = [J; mu*I], mu=" << (double)mu_P
              << " (= " << mu_factor << " * u(" << precond_prec_str << "))\n\n";

    const P_precond tol_P = std::pow(std::numeric_limits<P_precond>::epsilon(), (P_precond)0.85);
    const T_solve   tol_T = std::pow(std::numeric_limits<T_solve>::epsilon(), (T_solve)0.85);

    // Per-run RNG states (CQRRTO only).
    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) { run_states[r] = main_state; if (r > 0) run_states[r].key.incr(r); }

    // Warmup the precond-chain CQRRTO on A_hat (warms the L^{-1} K V chain, the
    // augmented Gram, and the blocked sketch overload), then the SOLVE chain:
    // the timed IR-LSQ runs LSQR on J_Ts with a TRSM preconditioner, and its
    // thread pools / first-touch pages otherwise land inside the FIRST
    // method's timed solve, which at 4-7 inner iterations is the same magnitude
    // as the whole solve. A few untimed LSQR iterations
    // on J_Ts (with the warmup R when usable) close that gap. This is a CPU
    // warmup, distinct from Blendenpik's x0 warm start.
    // The factorization half of the warmup only warms the precond chain (augmented Gram,
    // blocked sketch overload), which nothing uses unless a Q-less method is selected: mask
    // bits 0-4. For a Blendenpik-only or unpreconditioned-only run it is a full extra CQRRTO
    // factorization whose result is discarded, and at the large FEM2 cell that is minutes of
    // node time. The LSQR half warms the shared solve chain and runs either way.
    const bool need_precond_warmup = (method_mask & 31) != 0;
    std::cout << "Running warmup (" << (need_precond_warmup ? "factor + solve" : "solve only")
              << ")... " << std::flush;
    { auto ws = run_states[0];
      P_precond* Rw = nullptr;
      T_solve* Rw_T = nullptr;
      int warm_status = 1;
      if (need_precond_warmup) {
          Rw = new P_precond[n * n]();
          RandLAPACK::CQRRTO_linops<P_precond, RNG> warm(false, tol_P);
          warm.nnz = sketch_nnz; warm.block_size = block_size;
          warm_status = warm.call(A_hat_Pp, Rw, n, (P_precond)d_factor, ws);
          Rw_T = new T_solve[n * n];
          if (warm_status == 0)
              for (int64_t i = 0; i < n * n; ++i) Rw_T[i] = (T_solve)Rw[i];
      }
      T_solve* x_wu = new T_solve[n]();
      int it_wu = 0; long lt_wu[4] = {0};
      RandLAPACK::lsqr<T_solve>(J_Ts, m, n,
          (warm_status == 0) ? Rw_T : nullptr, (warm_status == 0) ? n : (int64_t)0,
          b.data(), x_wu, tol_T, tol_T, 5, it_wu, lt_wu);
      delete[] Rw; delete[] Rw_T; delete[] x_wu; }
    std::cout << "done\n";

    std::vector<bench_result<T_solve>> all_results;

    // Both n^2 buffers must be pre-touched before the measured loop starts.
    // Value-init ("()") already zero-fills R_P, which pages it in
    // here; R_T gets an explicit std::fill for the same effect (kept separate
    // from allocation so the pre-touch is self-documenting rather than relying
    // on new[]() semantics matching by accident on the next edit).
    P_precond* R_P = new P_precond[n * n]();
    T_solve*   R_T = new T_solve[n * n];
    std::fill(R_T, R_T + n * n, (T_solve)0);
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
            const bool is_unprec = (alg_name == "unpreconditioned");

            // Copies one shared-helper QR harvest into this benchmark's result row and
            // folds the driver's shift records, which the two benchmarks store differently.
            auto harvest = [&](const RandLAPACK::bench::QRRun& d, const auto& qr) {
                res.qr_status = d.status;
                res.chol_retries = d.chol_retries;
                record_chol_shift(res, qr.chol_applied_shifts, qr.chol_gram_traces);
                if (d.status == 0) {
                    res.qr_time_us = d.qr_time_us;
                    res.qr_breakdown = d.breakdown;   // whole vector, not truncated to a fixed slot count
                    res.analytical_kb = d.analytical_kb;
                }
            };

            std::cout << "[Run " << run_idx << ", " << alg_name << "] QR(" << precond_prec_str
                      << ") ... " << std::flush;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            if (is_bp) {
                // Shared Blendenpik-family dispatch, in SOLVE precision on the BASE
                // operator J_Ts (no mu, no augmented A_hat); fills every accounting
                // field itself and writes the sketch R factor into R_T directly.
                T_solve outer_tol_eff = (g_ir_outer_tol >= 0) ? (T_solve)g_ir_outer_tol
                                      : (T_solve)10 * std::numeric_limits<T_solve>::epsilon();
                run_blendenpik_family<T_solve, RNG>(J_Ts, b.data(), m, x_ls, n, R_T,
                    (T_solve)d_factor, sketch_nnz, state, alg_name, tol_T, outer_tol_eff,
                    be_oracle, be_tol, mem, res);
            } else if (is_unprec) {
                // No factor: the row is the refinement engine on the raw operator
                // (R = nullptr below). Nothing is built, so the build phase, the
                // Cholesky records and the storage model keep their "no value"
                // sentinels and the solve is the whole row.
                res.qr_status = 0; res.qr_time_us = 0;
                res.qr_breakdown.clear(); res.analytical_kb = -1;
            } else if (alg_name == "sCholQR3") {
                RandLAPACK::sCholQR3_linops<P_precond> qr(true, tol_P); qr.block_size = block_size;
                harvest(RandLAPACK::bench::run_cholqr_family(qr, A_hat_Pp, R_P, n, [&]{
                    return RandLAPACK::scholqr3_linops_analytical_kb<P_precond>(A_hat_Pp.n_rows, n, block_size); }), qr);
            } else if (alg_name == "sCholQR3_basic") {
                RandLAPACK::sCholQR3_linops_basic<P_precond> qr(true, tol_P);
                harvest(RandLAPACK::bench::run_cholqr_family(qr, A_hat_Pp, R_P, n, [&]{
                    return RandLAPACK::scholqr3_linops_basic_analytical_kb<P_precond>(A_hat_Pp.n_rows, n); }), qr);
            } else if (alg_name == "CholQR") {
                RandLAPACK::CholQR_linops<P_precond> qr(true, tol_P); qr.block_size = block_size;
                harvest(RandLAPACK::bench::run_cholqr_family(qr, A_hat_Pp, R_P, n, [&]{
                    return RandLAPACK::cholqr_linops_analytical_kb<P_precond>(A_hat_Pp.n_rows, n, block_size); }), qr);
            } else if (alg_name == "CholQR2") {
                RandLAPACK::CholQR2_linops<P_precond> qr(true, tol_P); qr.block_size = block_size;
                harvest(RandLAPACK::bench::run_cholqr_family(qr, A_hat_Pp, R_P, n, [&]{
                    return RandLAPACK::cholqr2_linops_analytical_kb<P_precond>(A_hat_Pp.n_rows, n, block_size); }), qr);
            } else {
                // CQRRTO: sketch + Gram the augmented A_hat (via VStack's blocked sketch
                // overload), uniformly with the other 4 methods. R = chol(A^T A + mu^2 I).
                RandLAPACK::CQRRTO_linops<P_precond, RNG> qr(true, tol_P);
                qr.max_retries = bench_chol_max_retries();
                qr.nnz = sketch_nnz; qr.block_size = block_size;
                qr.precond_method = RandLAPACK::CQRRTOLinopPrecond::TRSM_IDENTITY;
                res.qr_status = qr.call(A_hat_Pp, R_P, n, (P_precond)d_factor, state); res.chol_retries = qr.n_chol_retries;
                record_chol_shift(res, qr.chol_applied_shifts, qr.chol_gram_traces);
                if (res.qr_status == 0) { res.qr_time_us = qr.total_us();
                    res.qr_breakdown = qr.times;   // whole vector, not truncated to a fixed slot count
                    res.analytical_kb = RandLAPACK::cqrrto_linops_analytical_kb<P_precond>(A_hat_Pp.n_rows, n, (P_precond)d_factor, block_size); }
            }

            if (res.qr_status != 0) {
                std::cerr << "\n  [" << alg_name << "] Run " << run_idx
                          << ": QR returned status " << res.qr_status << ". Skipping solve.\n";
                res.qr_time_us = -1; res.ir_total_us = -1; res.qr_breakdown.clear(); res.analytical_kb = -1;   // -1 = no value; 0 would read as a real 0 MB bar
                if (!is_bp) res.peak_rss_kb = mem.stop();   // bp rows already stopped inside the helper
                all_results.push_back(res);
                continue;
            }
            res.kappa_measured = is_unprec ? (T_solve)-1
                               : is_bp     ? (T_solve)kappa_from_R_diag<T_solve>(R_T, n)
                                           : (T_solve)kappa_from_R_diag<P_precond>(R_P, n);
            std::cout << "done (" << res.qr_time_us << " us, kappa~"
                      << std::scientific << std::setprecision(2) << (double)res.kappa_measured << ")";

            // Cast R to solve precision (Blendenpik already produced R_T directly).
            if (!is_bp && !is_unprec) for (int64_t i = 0; i < n * n; ++i) R_T[i] = (T_solve)R_P[i];

            // NOTE ordering: the orthogonality diagnostic is computed
            // AFTER the solve, not here. It materializes Q = A R^{-1} (m x n, about
            // 88 GB on the large cell), so leaving it inside the peak-RSS window
            // would swamp the measurement. Moving it below lets every row family
            // use the SAME window, build + solve with diagnostics excluded, which
            // is what the Toeplitz benchmark already did and what makes the
            // peak-vs-predicted panel comparable across the two figures.

            // Solve in solve precision. Blendenpik-family rows already solved and
            // recorded everything inside run_blendenpik_family; everyone else runs
            // IR-LSQ with R as the right preconditioner.
            if (is_bp) {
                std::cout << ". solve(" << solve_prec_str << ") recorded ... " << std::flush;
            } else {
                std::cout << ". IR-LSQ(" << solve_prec_str << ") ... " << std::flush;
                auto ls_t0 = steady_clock::now();
                // Always cold: warm x0 is Blendenpik-only (see the CLI comment
                // block). ir_setup_us stays in the CSV schema and is always 0
                // for IR methods.
                std::fill(x_ls, x_ls + n, (T_solve)0.0);
                RandLAPACK::IterRefineLSQ<T_solve> ir(
                    ir_inner_tol_eff<T_solve>(tol_T),
                    (g_ir_max_inner > 0) ? g_ir_max_inner : 200,
                    g_ir_n_steps, true, false);
                ir.round_drop = (T_solve)g_ir_round_drop;
                ir.outer_tol = (g_ir_outer_tol >= 0) ? (T_solve)g_ir_outer_tol
                             : (T_solve)10 * std::numeric_limits<T_solve>::epsilon();
                ir.outer_stag_window = ir_outer_stag_window();
                ir.be_oracle = be_oracle;
                ir.be_tol    = be_tol;
                // R = nullptr selects the engine's unpreconditioned normal
                // equations (the restarted_pcg_ne contract); every other row
                // passes its factor as the right preconditioner.
                int ir_status = ir.call(J_Ts, is_unprec ? nullptr : R_T, is_unprec ? (int64_t)0 : n,
                                        b.data(), m, x_ls, n);
                auto ls_t1 = steady_clock::now();
                if (ir_status != 0) std::cerr << "Warning: IterRefineLSQ status " << ir_status << "\n";
                // Wall clock minus the oracle: the oracle is instrumentation, not solver work.
                res.ir_total_us = duration_cast<microseconds>(ls_t1 - ls_t0).count() - ir.t_be_us;
                res.t_be_us     = ir.t_be_us;
                res.be_x0       = ir.be_x0;            // always -1: IR rows start cold
                res.be_final    = !ir.be_per_step.empty() ? ir.be_per_step.back()
                                : (be_tol >= (T_solve)0) ? std::numeric_limits<T_solve>::infinity()
                                : (T_solve)-1;          // cold rows have no x0 value
                res.ir_outer_iters = ir.outer_iters_done;
                res.ir_inner_iters_total = 0;
                for (int v : ir.inner_iters_per_step) res.ir_inner_iters_total += v;
                record_inner_cg_diagnosis(ir, res);
                record_ir_outputs(ir, res);
                if (!ir.times.empty()) res.ir_breakdown = ir.times;
                res.peak_rss_kb = mem.stop();   // build + solve, diagnostics excluded
            }

            // Orthogonality loss of Q = A R^{-1} (base A in solve precision).
            // Outside the RSS window on purpose; see the ordering note above.
            // compute_cond gates cond_precond identically in both irlsq and
            // irlsq_reg.
            if (is_unprec) {
                res.orth_error = (T_solve)-1;   // no factor, no Q; -1 = "no value" (plotters print N/A)
            } else {
                res.orth_error = compute_orth_error_explicit<T_solve>(J_Ts, R_T, m, n, block_size,
                    compute_cond ? &res.cond_precond : nullptr);
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
            res.x_hat.assign(x_ls, x_ls + n);   // for the post-pass backward error

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
    std::string rounds_file    = output_dir + "/" + time_buf + "_irlsq_reg_rounds.csv";
    write_irlsq_reg_results<T_solve>(results_file, all_results, m, n, nnz_K,
        "L^{-1} K (V D) (M=" + M_file + ")", d_factor, sketch_nnz, block_size, method_mask,
        kappa_target, (double)mu_P, precond_prec_str, solve_prec_str,
        num_runs, chol_time_us, noise_level);
    RandLAPACK::bench::check_csv_arity(results_file);
    std::cout << "\nIR-LSQ-reg results written to " << results_file << "\n";
    write_irlsq_breakdown<T_solve>(breakdown_file, all_results,
        "IR-LSQ (regularized augmented operator)");
    RandLAPACK::bench::check_csv_arity(breakdown_file);
    std::cout << "IR-LSQ-reg breakdown written to " << breakdown_file << "\n";
    write_rounds_csv<T_solve>(rounds_file, all_results);
    RandLAPACK::bench::check_csv_arity(rounds_file);
    std::cout << "IR-LSQ-reg per-round records written to " << rounds_file << "\n";

    // ---- Backward error (sketched Karlson-Walden, EMN24): post-pass ----
    // With the oracle off the reference is built here, after every timed row, so no
    // row's timing or RSS window sees it; with the oracle on it already exists.
    std::string kw_sidecar;
    try {
        build_kw();   // no-op when the oracle already built it before the rows
        const auto& kw_ref = *kw_ref_ptr;
        std::vector<T_solve> Ax(m, (T_solve)0), ATr(n, (T_solve)0);
        std::ostringstream kw_rows;
        for (auto& r : all_results) {
            if (r.qr_status != 0 || r.x_hat.empty()) continue;
            J_Ts(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 m, 1, n, (T_solve)1.0, r.x_hat.data(), n, (T_solve)0.0, Ax.data(), m);
            for (int64_t i = 0; i < m; ++i) Ax[i] = b[i] - Ax[i];   // r = b - A x
            T_solve r_norm = blas::nrm2(m, Ax.data(), 1);
            T_solve x_norm = blas::nrm2(n, r.x_hat.data(), 1);
            J_Ts(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                 n, 1, m, (T_solve)1.0, Ax.data(), m, (T_solve)0.0, ATr.data(), n);
            T_solve be_theta, be_inf, theta, res_orth;
            RandLAPACK::bench::kw_backward_error<T_solve>(kw_ref, ATr.data(), r_norm, x_norm, b_norm,
                                                          be_theta, be_inf, theta, res_orth);
            std::cout << "  [" << r.alg_name << "] run " << r.run_idx << ": BE_theta="
                      << std::scientific << std::setprecision(3) << (double)be_theta
                      << " BE_inf=" << (double)be_inf << " res_orth=" << (double)res_orth << "\n";
            kw_rows << r.alg_name << "," << r.run_idx << ","
                    << std::scientific << std::setprecision(6) << be_theta << "," << be_inf << ","
                    << theta << "," << r_norm << "," << x_norm << "," << res_orth << "\n";
            r.x_hat.clear(); r.x_hat.shrink_to_fit();
        }
        kw_sidecar = RandLAPACK::bench::kw_provenance_line<T_solve>(kw_ref, b_norm, kw_build_s)
                   + RandLAPACK::bench::kKWCsvHeader + kw_rows.str();
    } catch (const std::exception& e) {
        std::cerr << "\nWARNING: backward-error post-pass FAILED (" << e.what()
                  << "); the results CSVs above are complete, only the sidecar is missing.\n";
        kw_sidecar = std::string("# backward-error post-pass FAILED: ") + e.what() + "\n";
    }

    std::string kw_file = output_dir + "/" + time_buf + "_irlsq_reg_backward_error.csv";
    { std::ofstream kw_out(kw_file); kw_out << kw_sidecar; }
    RandLAPACK::bench::check_csv_arity(kw_file);
    std::cout << "IR-LSQ-reg backward-error sidecar written to " << kw_file << "\n";
    return 0;
}

// Dispatch run_irlsq_reg on the runtime precond-precision string (solve precision = T).
template <typename T_solve, typename RNG>
static int dispatch_irlsq_reg(
    const std::string& precond_prec,
    const std::string& K_file, const std::string& M_file, const std::string& V_file,
    const std::string& output_dir, int64_t num_runs,
    double d_factor, int64_t sketch_nnz, int64_t block_size,
    bool compute_cond,
    int64_t method_mask, double kappa_target, double mu_factor, double noise_level,
    const std::string& solve_prec_str)
{
    if (precond_prec == "double") {
        return run_irlsq_reg<T_solve, double, RNG>(
            K_file, M_file, V_file, output_dir, num_runs, d_factor, sketch_nnz, block_size,
            compute_cond, method_mask, kappa_target, mu_factor, noise_level, "double", solve_prec_str);
    } else if (precond_prec == "single" || precond_prec == "float") {
        return run_irlsq_reg<T_solve, float, RNG>(
            K_file, M_file, V_file, output_dir, num_runs, d_factor, sketch_nnz, block_size,
            compute_cond, method_mask, kappa_target, mu_factor, noise_level, "single", solve_prec_str);
    }
    std::cerr << "Error: precond_prec must be 'single' or 'double'; got '" << precond_prec << "'\n";
    return 1;
}

// ============================================================================
// Main dispatcher
// ============================================================================

static void print_app_usage(const char* exe) {
    std::cerr <<
      "Usage (named form, preferred):\n"
      "  " << exe << " --precision=double --out=DIR --runs=N \\\n"
      "      --K=K.mtx --M=M.mtx --V=V.mtx [options]\n"
      "\n"
      "Required:\n"
      "  --precision=double|single  solve precision\n"
      "  --out=DIR                  output directory for the CSVs\n"
      "  --runs=N                   repeats per method\n"
      "  --K= --M= --V=             FEM stiffness, mass and prolongation matrices\n"
      "\n"
      "Options (default in brackets):\n"
      "  --d-factor=F      [2.0]    sketch oversampling\n"
      "  --sketch-nnz=N    [4]      nonzeros per sketch column\n"
      "  --block-size=N    [256]    blocked Gram width; 0 = unblocked\n"
      "  --compute-cond    [off]    also estimate the preconditioned condition number\n"
      "  --mask=N          [31]     method bitmask (campaign uses 127)\n"
      "  --noise=F         [0.05]   relative noise added to b; pass 0 for a consistent RHS\n"
      "  --mu-factor=F     [10]     mu = mu_factor * u(precond precision)\n"
      "  --precond-prec=P  [single] preconditioner precision: double|single\n"
      "  --max-inner=N     [200]    inner CG iteration cap per round\n"
      "  --inner-tol=F     [-1]     inner absolute floor; <0 = eps^0.85, 0 = off\n"
      "  --round-drop=F    [1e-4]   per-round residual drop; 0 = legacy fixed-tol rounds\n"
      "  --steps=N         [50]     outer refinement round cap\n"
      "  --outer-tol=F     [-1]     outer early exit; <0 = 10*eps, 0 = run all steps\n"
      "  --be-tol-mult=F   [0]      stop a run once the sketched Karlson-Walden backward error\n"
      "                             is <= F*sqrt(n)*u (Epperly's termination test); 0 = off\n"
      "\n"
      "The positional form is still accepted for existing job scripts, but is deprecated:\n"
      "  <precision> <out> <runs> irlsq_reg <K> <M> <V> <d_factor> [sketch_nnz]\n"
      "  [block_size] [compute_cond] [mask] [noise] [omega] [power_j] [kappa_target]\n"
      "  [mu_factor] [precond_prec] [max_inner] [inner_tol] [round_drop] [steps] [outer_tol]\n"
      "  [be_tol_mult]\n";
}

template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[]) {
    g_argv_line = quote_join_argv(argc, argv);

    std::string output_dir, K_file, M_file, V_file, precond_prec;
    int64_t num_runs = 0, sketch_nnz = 4, block_size = 256, method_mask = 31;
    bool compute_cond = false, noise_level_explicit = false;
    T d_factor = (T)2.0, noise_level = (T)0.05;
    double mu_factor = 10.0;

    if (RandLAPACK::bench::BenchArgs::looks_named(argc, argv)) {
        try {
            RandLAPACK::bench::BenchArgs a(argc, argv);
            a.reject_unknown({"precision", "out", "runs", "K", "M", "V", "d-factor",
                              "sketch-nnz", "block-size", "compute-cond", "mask", "noise",
                              "mu-factor", "precond-prec", "max-inner", "inner-tol",
                              "round-drop", "steps", "outer-tol", "be-tol-mult", "help"});
            if (a.has("help")) { print_app_usage(argv[0]); return 0; }
            output_dir   = a.require_str("out");
            num_runs     = a.i64("runs", 1);
            K_file       = a.require_str("K");
            M_file       = a.require_str("M");
            V_file       = a.require_str("V");
            d_factor     = (T)a.dbl("d-factor", 2.0);
            sketch_nnz   = a.i64("sketch-nnz", 4);
            block_size   = a.i64("block-size", 256);
            compute_cond = a.boolean("compute-cond", false);
            method_mask  = a.i64("mask", 31);
            noise_level_explicit = a.has("noise");
            noise_level  = (T)a.dbl("noise", 0.05);
            mu_factor    = a.dbl("mu-factor", 10.0);
            precond_prec = a.str("precond-prec", "single");
            g_ir_max_inner  = (int)a.i64("max-inner", 200);
            g_ir_inner_tol  = a.dbl("inner-tol", -1.0);
            g_ir_round_drop = a.dbl("round-drop", 1e-4);
            g_ir_n_steps    = (int)a.i64("steps", 50);
            g_ir_outer_tol  = a.dbl("outer-tol", -1.0);
            g_be_tol_mult   = a.dbl("be-tol-mult", 0.0);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n\n";
            print_app_usage(argv[0]);
            return 1;
        }
    } else {
        // Deprecated positional form, kept so existing job scripts keep running unchanged.
        if (argc < 9) { print_app_usage(argv[0]); return 1; }
        std::cerr << "WARNING: positional arguments are deprecated; a value in the wrong slot "
                     "is parsed and echoed rather than rejected, which has silently invalidated "
                     "a campaign before. Re-run with --help to see the named form.\n";
        output_dir       = argv[2];
        num_runs         = std::stoll(argv[3]);
        std::string mode = argv[4];
        if (mode != "irlsq_reg") {
            std::cerr << "Error: <mode> must be 'irlsq_reg'; got '" << mode << "'\n";
            return 1;
        }
        K_file   = argv[5];
        M_file   = argv[6];
        V_file   = argv[7];
        d_factor = (T)std::stod(argv[8]);
        const int dfactor_idx = 8;
        auto opt_long = [&](int rel, int64_t def) {
            int idx = dfactor_idx + rel;
            return (argc > idx) ? std::stoll(argv[idx]) : def;
        };
        auto opt_double = [&](int rel, double def) {
            int idx = dfactor_idx + rel;
            return (argc > idx) ? std::stod(argv[idx]) : def;
        };
        sketch_nnz  = opt_long(1, 4);
        block_size  = opt_long(2, 256);
        compute_cond = (opt_long(3, 0) != 0);
        method_mask = opt_long(4, 31);
        noise_level_explicit = (argc > dfactor_idx + 5);
        noise_level = (T)opt_double(5, 0.05);
        // Slots 6, 7 and 8 are parsed and discarded: they held the retired rspec mode's omega
        // and power_j, and kappa_target, whose only supported value was 1 (the FEM2 generators
        // bake the conditioning into V and warn that a larger value re-applies the column
        // scaling). They stay in the positional layout so every later slot keeps its index.
        (void)opt_double(6, 0.0);
        (void)opt_long(7, 1);
        (void)opt_double(8, 1.0);
        mu_factor    = opt_double(9, 10.0);
        precond_prec = (argc > dfactor_idx + 10) ? std::string(argv[dfactor_idx + 10]) : "single";
        g_ir_max_inner  = (int)opt_long(11, 200);
        g_ir_inner_tol  = opt_double(12, -1.0);
        g_ir_round_drop = opt_double(13, 1e-4);
        g_ir_n_steps    = (int)opt_long(14, 50);
        g_ir_outer_tol  = opt_double(15, -1.0);
        g_be_tol_mult   = opt_double(16, 0.0);
        if (argc > dfactor_idx + 17) {
            std::cerr << "Error: no positional slot exists beyond [be_tol_mult] (slot 16). "
                         "Regenerate the job scripts or use the named form (--help).\n";
            return 1;
        }
    }

    if (g_ir_round_drop < 0.0 || g_ir_round_drop >= 1.0) {
        std::cerr << "Error: round-drop must lie in [0, 1).\n";
        return 1;
    }
    if (g_ir_inner_tol == 0.0 && g_ir_round_drop <= 0.0) {
        std::cerr << "Error: inner-tol = 0 (absolute floor off) is only meaningful in paced "
                     "mode (round-drop > 0); in legacy mode it is the per-round tolerance "
                     "and cannot be 0.\n";
        return 1;
    }
    if (g_ir_n_steps < 1) {
        std::cerr << "Error: steps must be >= 1.\n";
        return 1;
    }
    if constexpr (std::is_same_v<T, float>) {
        // Refuse at parse time, before the matrices are loaded: the backward-error
        // reference is not meaningful in float (see run_irlsq_reg for the reason).
        if (g_be_tol_mult > 0.0) {
            std::cerr << "Error: --be-tol-mult > 0 requires the double solve precision.\n";
            return 1;
        }
    }

    // Column-scaling target for V. Fixed at 1 (no rescaling): the FEM2 generators bake the
    // intended conditioning into V and warn that any larger value re-applies the geometric
    // column scaling on top of it, so 1 was the only value ever run. Kept as a named constant
    // because the CSV still reports it.
    const double kappa_target = 1.0;

    std::cout << "=== CQRRTO linop benchmark ===\n";
    std::cout << "  Input mode: FEM composite (J = L^{-1} K V with L = chol(M))\n"
              << "  K file: " << K_file << "\n"
              << "  M file: " << M_file << "\n"
              << "  V file: " << V_file << "\n";
    std::cout << "  d_factor: " << d_factor << "\n"
              << "  sketch_nnz: " << sketch_nnz << "\n"
              << "  block_size: " << block_size << "\n"
              << "  compute_cond: " << (compute_cond ? "yes" : "no") << "\n"
              << "  method_mask: " << method_mask << " (";
    // Echo the full decoded roster: limiting it to bits 0-4 would leave a mask-127 job
    // log unable to show that the Blendenpik and refine rows were selected.
    {
        auto echo_algs = decode_method_mask(method_mask, /*with_blendenpik=*/true);
        for (size_t i = 0; i < echo_algs.size(); ++i)
            std::cout << (i ? " " : "") << echo_algs[i];
    }
    std::cout << ")\n"
              << "  noise_level: " << noise_level << "\n"
              << "  num_runs: " << num_runs << "\n"
#ifdef _OPENMP
              << "  OpenMP threads: " << omp_get_max_threads() << "\n\n";
#else
              << "  OpenMP threads: 1\n\n";
#endif

    // ================================================================
    // irlsq_reg mode (FEM-only): regularized augmented-operator preconditioner
    // with independent preconditioner / solve precisions. Loads + builds its own
    // (kappa-scaled, cast-down) chains, so it intercepts before the plain FEM load.
    // <precision> (argv[1]) is the SOLVE precision; precond precision is a CLI knob.
    // ================================================================
    // The noise_level CLI slot silently defaults to 0.05. irlsq_reg's
    // consistent-RHS backward-error reading is cleanest at noise_level=0,
    // so a nonzero DEFAULTED value
    // (as opposed to one the caller explicitly asked for) is worth flagging loudly
    // rather than baking a silent, easy-to-miss assumption into the CSV.
    if (!noise_level_explicit && noise_level != (T)0) {
        std::cerr << "WARNING: irlsq_reg noise_level defaulted to " << (double)noise_level
                  << " (not explicitly set on the CLI). Pass noise_level=0 explicitly for "
                     "the pure consistent-RHS backward-error reading, or pass the intended "
                     "nonzero value explicitly to silence this warning.\n";
    }
    std::string solve_prec_str = (sizeof(T) == 8) ? "double" : "single";
    std::cout << "\n=== IR-LSQ-reg mode (regularized augmented operator) ===\n"
              << "  kappa_target: " << kappa_target << "\n"
              << "  mu_factor: "    << mu_factor    << "\n"
              << "  noise_level: "  << (double)noise_level << "\n"
              << "  precond_prec: " << precond_prec << "\n"
              << "  solve_prec: "   << solve_prec_str << "\n\n";
    return dispatch_irlsq_reg<T, RNG>(precond_prec, K_file, M_file, V_file,
        output_dir, num_runs, (double)d_factor, sketch_nnz, block_size,
        compute_cond, method_mask, kappa_target, mu_factor, (double)noise_level, solve_prec_str);
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_app_usage(argv[0]); return 1; }

    // The solve precision selects the template instantiation, so it has to be read before
    // run_benchmark parses anything else. Both argument forms are supported here.
    std::string precision;
    if (RandLAPACK::bench::BenchArgs::looks_named(argc, argv)) {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--help") { print_app_usage(argv[0]); return 0; }
            if (a.rfind("--precision=", 0) == 0) { precision = a.substr(12); break; }
        }
        if (precision.empty()) {
            std::cerr << "Error: missing required argument --precision\n\n";
            print_app_usage(argv[0]);
            return 1;
        }
    } else {
        precision = argv[1];
    }

    if (precision == "double") {
        return run_benchmark<double>(argc, argv);
    } else if (precision == "float" || precision == "single") {
        return run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Unknown precision: " << precision << " (use 'double'/'float'/'single')\n";
        return 1;
    }
}
