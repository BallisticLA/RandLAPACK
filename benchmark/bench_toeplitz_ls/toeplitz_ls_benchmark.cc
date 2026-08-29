// Toeplitz least-squares benchmark (autoregression / system-identification
// experiment) ported to RandLAPACK.
//
// Solves the regularized LS  min_x ||T x - b||^2 + lambda ||x||^2  =  min_x ||A x - rhs||,
// with A = [T; sqrt(lambda) I], rhs = [b; 0], T an m x n prolate-kernel Toeplitz matrix.
// T is matrix-free (FFT circulant embedding, ext_toeplitz_linop.hh). Q-less QR right
// preconditioners (unpreconditioned / CholQR / CholQR2 / sCholQR3 / sCholQR3_basic /
// CQRRT / Blendenpik) build R; the solve is matrix-free restarted PCG-NE (default)
// or LSQR on A with right preconditioner R.
//
// Double precision only. Records accuracy + speed (build/solve time) + storage
// (peak RSS, analytical) + Cholesky shift-retry count.
//
// Solver choice: the reference benchmark offers TWO solvers and ours now does
// too. DEFAULT = pcg_ne. This deliberately diverges from the reference's lsqr
// default; pass "lsqr" explicitly to reproduce campaigns that ran with it.
//   lsqr    = matrix-free LSQR on the right-preconditioned operator;
//   pcg_ne  = restarted PCG on the right-preconditioned normal equations
//             (rl_restarted_pcg_ne.hh; reference restarted_pcg_preconditioned_
//             normal_eq, restart after a pcg_restart_drop recursive-residual drop
//             with true LS + NE residuals recomputed at every restart).
// PUBLISHED Blendenpik rows (mask 32) solve with LSQR, the solver that is part of
// that method; the REFINED rows (mask 128) run the shared pcg_ne engine
// unconditionally, whatever the solver CLI says, and their 'solver' column
// always says pcg_ne.
//
// CLI: <prec> <outdir> <m> <n> <omega> <lambda_rel> <method_mask> <tol> <maxit>
//      <d_factor> <sketch_nnz> [seed] [num_runs] [solver] [pcg_restart_maxit]
//      [pcg_max_restarts] [pcg_restart_drop]
//   method_mask bits: 1 CQRRT, 2 CholQR, 4 sCholQR3, 8 sCholQR3_basic, 16 CholQR2,
//                     32 Blendenpik (published; warm and cold rows),
//                     64 unpreconditioned,
//                     128 Blendenpik refined by the shared engine (warm and cold
//                         rows; see benchmark/refined_blendenpik.hh: init_only
//                         sketch-and-solve x0, ALL iterative work in pcg_ne, no
//                         internal LSQR).
//   num_runs (default 1): repetitions per method, all recorded (one CSV row each,
//   'run' column). The MATLAB plotters aggregate (best run by default).
//   solver (default "pcg_ne"): "pcg_ne" (alias "restarted_pcg_ne") or "lsqr".
//   pcg_restart_maxit (default 500): inner CG cap per restart, pcg_ne only.
//   pcg_max_restarts (default 50): additional rounds after the first (< 0
//   unlimited), pcg_ne only. The round cap must not bind before tol and maxit
//   do (the FEM2 native_ill CholQR2 cell genuinely uses 50 rounds).
//   pcg_restart_drop (default 1e-4): per-round relative residual drop that ends
//   a round, pcg_ne only. CLI-exposed so this benchmark and FEM2 match on the
//   drop factor and the round/iteration caps (pcg_max_restarts,
//   pcg_restart_maxit, maxit). Two pcg_ne knobs still differ from FEM2 and are
//   NOT CLI-exposed here: the inner absolute-residual guard (FEM2 passes
//   eps^0.85; this benchmark always passes 0.0) and the outer-stagnation window
//   (FEM2 honors RANDLAPACK_IR_OUTER_STAG; this benchmark always uses
//   restarted_pcg_ne's fixed default). Round counts across the two benchmarks
//   are comparable only insofar as those two knobs do not bind.
//
// Warm start policy: the sketch-and-solve x0 warm start is Blendenpik-only.
// Bit 32 therefore runs TWO variants, "Blendenpik" (its own warm start) and
// "Blendenpik_cold" (x0 = 0), and every Q-less QR method is unconditionally
// cold. The former [qless_warm_start]/[bp_warm_start] CLI knobs are gone.

#include "RandLAPACK.hh"
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "rl_blendenpik.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "../../extras/linops/ext_toeplitz_linop.hh"
#include "../refined_blendenpik.hh"
#include "../bench_CQRRT_linops/cqrrt_bench_common.hh"

#include <RandBLAS.hh>
#include <blas.hh>
#include <lapack.hh>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <unistd.h>

using RNG = RandBLAS::DefaultRNG;
namespace rl  = RandLAPACK;
namespace rle = RandLAPACK_extras::linops;
using blas::Layout; using blas::Op; using blas::Side;
using std::chrono::steady_clock; using std::chrono::duration_cast; using std::chrono::microseconds;

static double prolate(int64_t k, double omega) {
    if (k == 0) return 2.0 * omega;
    double kk = (double)k;
    return std::sin(2.0 * M_PI * omega * kk) / (M_PI * kk);
}

// Power iteration for lambda_max(T'T) (RandLAPACK::bench::power_lambda_max)
// and the blocked orth/cond estimator below are shared with bench_CQRRT_linops;
// see cqrrt_bench_common.hh.

// Preconditioner quality of R via Q = A R^{-1}: thin wrapper over the shared
// compute_orth_error_explicit (cqrrt_bench_common.hh), which materializes Q
// one column-block at a time and reads both metrics from the same Gram.
// cond sentinel is -1 (suite convention: MATLAB readers treat NaN and -1
// differently, and -1 is what every other unavailable metric in this file uses).
template <typename AOp>
static void compute_orth_and_cond(AOp& A, const double* R, int64_t mtot, int64_t n,
                                  int64_t block_size, int64_t cond_cap,
                                  double& orth_out, double& cond_out) {
    double cond_tmp;
    orth_out = RandLAPACK::bench::compute_orth_error_explicit<double>(
        A, R, mtot, n, block_size, &cond_tmp, cond_cap, -1.0);
    cond_out = cond_tmp;
}

int main(int argc, char** argv) {
    if (argc < 12) {
        std::fprintf(stderr, "usage: %s <prec> <outdir> <m> <n> <omega> <lambda_rel> <method_mask> "
                             "<tol> <maxit> <d_factor> <sketch_nnz> [seed] [num_runs] [solver] "
                             "[pcg_restart_maxit] [pcg_max_restarts] [pcg_restart_drop]\n", argv[0]);
        return 1;
    }
    std::string prec = argv[1];          // "double" (v1)
    if (prec != "double") {
        std::fprintf(stderr, "prec must be \"double\" (this benchmark is double precision "
                             "only, v1); got \"%s\"\n", prec.c_str());
        return 1;
    }
    std::string outdir = argv[2];
    int64_t m = std::stoll(argv[3]);
    int64_t n = std::stoll(argv[4]);
    double omega = std::stod(argv[5]);
    double lambda_rel = std::stod(argv[6]);
    int64_t method_mask = std::stoll(argv[7]);
    double tol = std::stod(argv[8]);
    int maxit = std::stoi(argv[9]);
    double d_factor = std::stod(argv[10]);
    int64_t sketch_nnz = std::stoll(argv[11]);
    int64_t seed = (argc > 12) ? std::stoll(argv[12]) : 1;
    // Repetitions per method, all recorded. Timing at 4-7 LSQR iterations is at
    // the noise floor of a single run; 5 runs + best-of in the plotter is the
    // fix. Default 1 keeps old invocations valid.
    int64_t num_runs = (argc > 13) ? std::stoll(argv[13]) : 1;
    if (num_runs < 1) {
        std::fprintf(stderr, "num_runs must be >= 1 (got %lld)\n", (long long)num_runs);
        return 1;
    }
    // Solver selection; default pcg_ne (see file header for the
    // lsqr/pcg_ne split). The pcg_ne knob defaults (pcg_restart_maxit,
    // pcg_max_restarts, pcg_restart_drop) are documented at their own CLI
    // parses below, not repeated here.
    std::string solver = (argc > 14) ? argv[14] : "pcg_ne";
    if (solver == "restarted_pcg_ne") solver = "pcg_ne";
    if (solver != "lsqr" && solver != "pcg_ne") {
        std::fprintf(stderr, "solver must be \"lsqr\" or \"pcg_ne\" (got \"%s\")\n", solver.c_str());
        return 1;
    }
    const bool use_pcg = (solver == "pcg_ne");
    const int    pcg_restart_maxit = (argc > 15) ? std::stoi(argv[15]) : 500;
    // max_restarts counts ADDITIONAL rounds after the first. Default 50: the
    // round cap must not bind before tol and maxit do.
    const int    pcg_max_restarts  = (argc > 16) ? std::stoi(argv[16]) : 50;
    // Per-round pacing; CLI-exposed so this benchmark and FEM2 use the same
    // round-drop factor and stay comparable on round counts.
    const double pcg_restart_drop  = (argc > 17) ? std::stod(argv[17]) : 1e-4;
    if (!(pcg_restart_drop > 0.0 && pcg_restart_drop < 1.0)) {
        std::fprintf(stderr, "pcg_restart_drop must lie in (0,1) (got %s)\n", argv[17]);
        return 1;
    }
    int64_t block_size = 256;
    const double relnoise = 1e-11;   // data noise level (hoisted so the CSV header echoes it)
    if (m < n) { std::fprintf(stderr, "require m >= n\n"); return 1; }

    std::printf("=== Toeplitz LS benchmark: m=%lld n=%lld omega=%.4f lambda_rel=%.1e mask=%lld solver=%s ===\n",
                (long long)m, (long long)n, omega, lambda_rel, (long long)method_mask, solver.c_str());

    // 1. Prolate kernel generators c (m), r (n).
    std::vector<double> c(m), r(n);
    for (int64_t i = 0; i < m; ++i) c[i] = prolate(i, omega);
    for (int64_t j = 0; j < n; ++j) r[j] = prolate(j, omega);

    // 2. Toeplitz operator + augmented A = [T; sqrt(lambda) I].
    rle::ToeplitzLinOp<double> T(c.data(), m, r.data(), n);

    double lam_max = RandLAPACK::bench::power_lambda_max(T, m, n, 50);
    double lambda  = lambda_rel * lam_max;
    std::printf("lambda_max(T'T)~=%.6e  lambda=%.6e\n", lam_max, lambda);

    rl::linops::ScaledIdentityOp<double> regI(n, std::sqrt(lambda));
    rl::linops::VStackOp<rle::ToeplitzLinOp<double>, rl::linops::ScaledIdentityOp<double>> A_hat(T, regI);
    A_hat.block_size = block_size;
    int64_t mtot = m + n;

    // 3. Synthetic data: x_true random-smooth, b = T x_true + noise, rhs = [b; 0].
    std::mt19937 rng((unsigned)seed); std::normal_distribution<double> nd(0,1);
    std::vector<double> x_true(n);
    { std::vector<double> z(n); for (auto& v : z) v = nd(rng);
      // light smoothing (Gaussian filter, truncation half-width min(n-1,80);
      // 18.0 below is the Gaussian decay scale, not the half-width)
      int hw = std::min<int64_t>(n-1, 80);
      std::vector<double> filt(hw+1); double fs = 0;
      for (int i = 0; i <= hw; ++i) { filt[i] = std::exp(-std::pow(i/18.0,2)); fs += (i==0?filt[i]:2*filt[i]); }
      for (int i = 0; i < n; ++i) { double acc = 0;
          for (int j = -hw; j <= hw; ++j) { int idx = i+j; if (idx>=0 && idx<n) acc += z[idx]*filt[std::abs(j)]; }
          x_true[i] = acc/fs; }
      double xn = blas::nrm2(n, x_true.data(), 1); blas::scal(n, 1.0/std::max(xn,1e-300), x_true.data(), 1);
    }
    std::vector<double> b(m);
    T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x_true.data(), n, 0.0, b.data(), m);
    double b_clean_norm = blas::nrm2(m, b.data(), 1);
    { std::vector<double> e(m); for (auto& v : e) v = nd(rng);
      double en = blas::nrm2(m, e.data(), 1);
      for (int64_t i = 0; i < m; ++i) b[i] += e[i]/std::max(en,1e-300)*relnoise*b_clean_norm; }
    double b_norm = blas::nrm2(m, b.data(), 1);
    std::vector<double> rhs(mtot, 0.0); std::copy(b.begin(), b.end(), rhs.begin());
    double rhs_norm = blas::nrm2(mtot, rhs.data(), 1);
    double x_true_norm = blas::nrm2(n, x_true.data(), 1);
    // normalRhs = A^T rhs (for normalRelres = ||A^T(Ax-rhs)|| / ||A^T rhs||).
    std::vector<double> normalRhs(n);
    A_hat(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, mtot, 1.0, rhs.data(), mtot, 0.0, normalRhs.data(), n);
    double normalRhs_norm = blas::nrm2(n, normalRhs.data(), 1);
    // cond(A R^{-1}) needs an n x n eig; skip it above this size (infeasible at ISAAC-large n).
    const int64_t cond_est_max_n = 16384;   // includes small (n=8256), excludes large (n=33024)

    // 3b. CPU warmup, untimed (NOT the x0 warm-start ablation): one Q-less QR
    // build plus a few LSQR iterations so MKL thread pools, first-touch pages,
    // and the FFT apply path are all exercised before anything is measured.
    // Without this the first timed solve absorbs those one-time costs, which at
    // 4-7 LSQR iterations is the same magnitude as the whole solve.
    std::printf("Running warmup (untimed)... "); std::fflush(stdout);
    {
        std::vector<double> R_wu(n * n, 0.0), x_wu(n, 0.0);
        rl::CholQR_linops<double> qr_wu(false, tol); qr_wu.block_size = block_size;
        int wu_status = qr_wu.call(A_hat, R_wu.data(), n);
        int it_wu = 0; long lt_wu[4] = {0}; double rr_wu = -1;
        rl::lsqr<double>(A_hat, mtot, n,
                         (wu_status == 0) ? R_wu.data() : nullptr,
                         (wu_status == 0) ? n : (int64_t)0,
                         rhs.data(), x_wu.data(), tol, tol, 5, it_wu, lt_wu, &rr_wu);
        // Sketch + geqrf warmup at the campaign dimensions: CholQR exercises
        // neither the RandBLAS sparse-sketch fill/apply nor
        // MKL's first geqrf, and CQRRT is always the FIRST timed row, so at
        // num_runs=1 those one-time initializations landed inside its build bar.
        {
            auto wu_state = RandBLAS::RNGState<RNG>((uint32_t)seed);
            int64_t d_wu = std::max<int64_t>((int64_t)(d_factor * (double)n), n);
            std::vector<double> Ask_wu(d_wu * n), tau_wu(n);
            RandBLAS::SparseDist DS_wu(d_wu, mtot, sketch_nnz);
            RandBLAS::SparseSkOp<double, RNG> S_wu(DS_wu, wu_state);
            RandBLAS::fill_sparse(S_wu);
            A_hat(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  d_wu, n, mtot, 1.0, S_wu, 0.0, Ask_wu.data(), d_wu);
            lapack::geqrf(d_wu, n, Ask_wu.data(), d_wu, tau_wu.data());
        }
    }
    std::printf("done\n");

    // 4. Method list from the mask.
    std::vector<std::string> algs;
    if (method_mask & 1)  algs.push_back("CQRRT_linop");
    if (method_mask & 2)  algs.push_back("CholQR");
    if (method_mask & 4)  algs.push_back("sCholQR3");
    if (method_mask & 8)  algs.push_back("sCholQR3_basic");
    if (method_mask & 16) algs.push_back("CholQR2");
    if (method_mask & 32) {
        algs.push_back("Blendenpik");        // its own sketch-and-solve warm start
        algs.push_back("Blendenpik_cold");   // same solver, x0 = 0
    }
    if (method_mask & 64) algs.push_back("unpreconditioned");
    if (method_mask & 128) {
        // Blendenpik's preconditioner and its sketch-and-solve x0 handed to OUR
        // shared engine (see benchmark/refined_blendenpik.hh). As published, Blendenpik is
        // sketch + QR + LSQR, so the suite otherwise compares its preconditioner
        // through a different solver than every Q-less method uses, conflating
        // preconditioner quality with solver structure. These two rows keep the
        // published rows intact and add the like-for-like comparison: same R,
        // ALL iterative work in the shared pcg_ne engine, one row started from
        // the sketch-and-solve x0 (warm) and one from zero (cold). No internal
        // LSQR runs, so the iteration count has one unit.
        algs.push_back("Blendenpik_refine");
        algs.push_back("Blendenpik_cold_refine");
    }
    // Guard against the silent-no-op class of bug (the old irlsq mode bug in
    // the FEM2 benchmark): a mask with no recognized bits used to write a
    // header-only CSV and exit 0, which looks like a successful empty
    // campaign cell instead of a typo'd invocation.
    if (int64_t bad_bits = method_mask & ~((int64_t)255); bad_bits != 0) {
        std::fprintf(stderr, "method_mask 0x%llx has bits outside the valid set "
                             "(1|2|4|8|16|32|64|128 = 255); offending bits: 0x%llx\n",
                             (unsigned long long)method_mask, (unsigned long long)bad_bits);
        return 1;
    }
    if (algs.empty()) {
        std::fprintf(stderr, "method_mask 0x%llx selects no methods (valid bits: 1 CQRRT, "
                             "2 CholQR, 4 sCholQR3, 8 sCholQR3_basic, 16 CholQR2, "
                             "32 Blendenpik, 64 unpreconditioned, 128 Blendenpik_refine)\n",
                             (unsigned long long)method_mask);
        return 1;
    }

    // 5. CSV.
    std::string tstamp = make_run_timestamp();
    std::string csv = outdir + "/" + tstamp + "_toeplitz_ls_results.csv";
    std::ofstream out(csv);
    // Provenance header: full argv, every constant a
    // reproduction needs, and the env knobs that change the algorithms without
    // changing the row labels. A CSV must identify its own campaign arm.
    out << "# Toeplitz LS benchmark m=" << m << " n=" << n << " omega=" << omega
        << " lambda=" << lambda << " tol=" << tol << " d_factor=" << d_factor
        << " solver=" << solver;
    if (use_pcg || (method_mask & 128)) out << " pcg_restart_maxit=" << pcg_restart_maxit
                     << " pcg_max_restarts=" << pcg_max_restarts
                     << " pcg_restart_drop=" << pcg_restart_drop;
    out << "\n";
    // Host provenance: wall-clock timings and MKL thread behavior are
    // machine-specific, so a CSV must name the machine it ran on.
    rl::bench::write_host_line(out);
    // Tolerance-matching provenance: a CSV reader comparing this file against
    // FEM2 needs to know the two benchmarks do not target the same thing.
    out << "# tolerance: every row family in this file (published and refine/pcg_ne"
        << " alike) targets the single CLI tol=" << tol << " echoed above; rows in"
        << " this file are mutually tolerance-matched.\n";
    out << "# the FEM2 benchmark (bench_CQRRT_linops) differs internally: its published"
        << " rows target tol=eps^0.85 while its refine/IR rows drive the true LS relres"
        << " to 10*eps, so iteration/time comparisons against FEM2, or across FEM2's own"
        << " published vs refine/IR rows, are not tolerance-matched.\n";
    out << "# blendenpik=warm+cold (all Q-less methods cold);"
        << " refine rows = init_only x0 + shared engine"
        << " num_runs=" << num_runs << "\n";
    out << "# seed=" << seed << " maxit=" << maxit << " method_mask=" << method_mask
        << " sketch_nnz=" << sketch_nnz << " relnoise=" << relnoise
        << " block_size=" << block_size << "\n";
    rl::bench::write_env_line(out);
    out << "# argv:"; for (int i = 0; i < argc; ++i) out << " " << argv[i]; out << "\n";
    out << "algorithm,run,m,n,qr_status,qr_time_us,solve_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,iterations,solver_flag,solver_relres,aug_relres,normal_relres,"
           "data_relres,recovery_error,cond_estimate,chol_retries,"
           "solve_fwd_us,solve_adj_us,solve_trsm_us,setup_us,"
           "solver,pcg_rounds,"
           "lsqr_iters,stop_reason,t_inner_us,t_fwd_inner_us,t_adj_inner_us,"
           "t_trsm_inner_us,t_overhead_us,total_row_us,x0_relres,chol_shift_abs,chol_shift_rel\n";
    // Column notes: pcg_rounds (renamed from pcg_restarts) holds TOTAL
    // rounds run, which is what the engine reports. iterations = engine inner CG
    // iterations (or LSQR iterations for the published Blendenpik rows and lsqr
    // mode); lsqr_iters is nonzero only where an LSQR phase ran. stop_reason names
    // the exit (tol/budget/rounds/floor/breakdown for pcg; tol/ne_floor/budget for
    // lsqr). t_*_inner_us = inside the CG kernel; t_overhead_us = solve total minus
    // kernel wall (restart-loop residual recomputations + solver vector work).
    // total_row_us = wall clock of build + setup + solve (metrics excluded; for
    // Blendenpik_cold_refine it also contains the executed-but-unused x0 build
    // kept for phase parity with the warm row (see refined_blendenpik.hh).
    // Sentinel note: cond_estimate uses -1 for "not computed" (MATLAB readers
    // treat NaN and -1 differently, and -1 is this suite's convention).
    // chol_shift_abs/chol_shift_rel use -1 for "no Cholesky in this method"
    // (Blendenpik family, unpreconditioned) or "QR failed before a shift
    // record existed"; 0 still means "Cholesky ran unshifted". solver is
    // "none" and solver_flag is -1 on rows whose QR/build failed (no solver
    // ever ran), instead of naming the row's normal engine.

    // Per-round sidecar: one row per
    // (algorithm, run, round) for every pcg_ne solve, from PCGRoundHistory.
    std::string csv_rounds = outdir + "/" + tstamp + "_toeplitz_ls_rounds.csv";
    std::ofstream out_rounds(csv_rounds);
    out_rounds << rl::bench::kRoundsCsvHeader;

    // stop_reason mapping (shared with bench_CQRRT_linops; cqrrt_bench_common.hh):
    // names the exit condition so the CSV can distinguish "hit the LS floor
    // honestly" from "ran out of budget", which shared a flag value before.
    auto pcg_reason = [](int st) -> const char* { return rl::bench::pcg_stop_reason(st); };
    auto lsqr_reason = [](int st, int test) -> const char* {
        return rl::bench::lsqr_stop_reason(st == 0, test);
    };

    std::vector<double> R(n*n), x(n), Tx(m), Ax(mtot);

    for (const auto& alg : algs) {
    for (int64_t run_idx = 0; run_idx < num_runs; ++run_idx) {   // body indent unchanged on purpose
        std::printf("\n=== %s (run %lld/%lld) ===\n", alg.c_str(),
                    (long long)(run_idx + 1), (long long)num_runs);
        std::fill(R.begin(), R.end(), 0.0);
        std::fill(x.begin(), x.end(), 0.0);
        // Same base seed, per-run key bump: independent sketches per run,
        // matching the CQRRT_linop_applications run_states convention.
        auto state = RandBLAS::RNGState<RNG>((uint32_t)seed);
        if (run_idx > 0) state.key.incr(run_idx);
        // flag (solver_flag column) starts at -1: "no solver ran" until a solve
        // actually executes. Rows whose QR/build fails never overwrite it.
        int qr_status = 0, iters = 0, flag = -1, chol_retries = 0;
        // Applied CholQR shift (0 = every pass unshifted; see the record note in
        // the linop drivers): abs = pass-1 absolute shift baked into R, rel = max
        // shift/trace over passes. A bare retry count cannot separate a
        // rounding-level rescue from a spectrum-truncating one; these can.
        // -1 = no Cholesky in this method (Blendenpik family, unpreconditioned) or
        // QR/build failed before any shift record existed.
        double chol_shift_abs = -1.0, chol_shift_rel = -1.0;
        // analytical_kb: -1 = "no analytical model for this row" (0 would be
        // indistinguishable from zero bytes in the storage figure).
        long qr_us = 0, solve_us = 0, peak_kb = 0, analytical_kb = -1;
        auto fold_chol_shift = [&](const auto& qr, size_t npasses) {
            rl::bench::fold_chol_shift(chol_shift_abs, chol_shift_rel,
                                       qr.chol_applied_shifts, qr.chol_gram_traces, npasses);
        };
        // Solve-time DECOMPOSITION. rl_lsqr already computes these into times[0..2]
        // but the benchmark previously recorded only times[3] (the total), which
        // made the solve column impossible to interpret correctly.
        //
        // Two things must be separated before any solve-time claim is trustworthy:
        //   * operator cost (t_fwd + t_adj) vs preconditioner cost (t_trsm) vs LSQR's own
        //     vector work and one-time setup, which is total - (fwd + adj + trsm);
        //   * fixed setup vs per-iteration cost. Dividing the TOTAL by the iteration count
        //     is invalid at 4 iterations (setup dominates) yet fine at 471, which by itself
        //     manufactures a large apparent gap between preconditioned and unpreconditioned
        //     runs. Reporting the parts removes that trap.
        // -1 means "not measured for this method".
        long solve_fwd_us = -1, solve_adj_us = -1, solve_trsm_us = -1;
        long setup_us = 0;           // x0-build time: Blendenpik warm rows only (0 = no x0 built)
        int pcg_rounds = -1;         // TOTAL engine rounds run; -1 = not the pcg_ne engine
        double solver_relres = -1;   // solver's own relative residual at termination
        int lsqr_iters_col = 0;      // LSQR iterations, where an LSQR phase ran
        double x0_relres = -1;       // true relres of the handed-off warm x0 (refine warm rows)
        std::string stop_reason = "n/a";
        rl::PCGRoundHistory<double> hist;   // engine per-round records (pcg rows)
        bool have_hist = false;
        bool is_bp     = (alg.rfind("Blendenpik", 0) == 0)
                      && (alg.find("_refine") == std::string::npos);
        bool have_R = false, is_unprec = (alg == "unpreconditioned");
        // Owned R buffer for bp/bp_ref rows: those drivers already own an
        // n x n R heap allocation (bp.R_out / rres.R); this takes ownership
        // of it instead of std::copy-ing into the R vector below, removing a
        // copy that would otherwise sit inside total_row_us with no time slot
        // of its own. Freed right after its one use (compute_orth_and_cond)
        // below.
        double* R_bp_owned = nullptr;

        rl::PeakRSSTracker mem; mem.start();
        auto t0 = steady_clock::now();

        bool is_bp_ref = (alg.find("_refine") != std::string::npos);
        if (is_bp_ref) {
            // Shared refined-Blendenpik dispatch: init_only sketch-and-
            // solve x0, then ALL iterative work in the shared engine. Runs pcg_ne
            // regardless of the solver CLI (the engine IS the row's definition).
            // See benchmark/refined_blendenpik.hh for the accounting contract.
            const bool warm = (alg == "Blendenpik_refine");
            auto rres = rl::bench::run_refined_blendenpik<double, RNG>(
                A_hat, rhs.data(), mtot, x.data(), n,
                d_factor, sketch_nnz, state, warm,
                tol, maxit, pcg_restart_maxit, pcg_restart_drop, pcg_max_restarts);
            qr_status = rres.qr_status;
            if (qr_status == 0) {
                qr_us    = rres.qr_us;
                setup_us = rres.setup_us;    // x0 build (0 for the cold row)
                x0_relres = rres.x0_relres;  // warm-start quality (-1 cold)
                // Take ownership instead of copying: rres.R is heap-owned and
                // would otherwise be freed when rres goes out of scope below.
                R_bp_owned = rres.R;
                rres.R = nullptr; rres.R_sz = 0;
                have_R = true;
                // init_only: LSQR never runs.
                // warm_start=true unconditionally for BOTH refine rows: this is
                // belt-and-braces with the model's own with_lsqr=false forcing,
                // since both rows allocate the x0-build buffers regardless of
                // which one the engine actually starts from.
                analytical_kb = rl::blendenpik_linops_analytical_kb<double>(
                    mtot, n, d_factor, /*warm_start=*/true, /*with_lsqr=*/false);
                flag          = rres.status;
                iters         = rres.iters;          // engine inner CG only
                lsqr_iters_col = 0;                  // no LSQR phase in these rows
                pcg_rounds    = rres.rounds;
                solver_relres = rres.solver_relres;
                solve_us      = rres.solve_us;
                solve_fwd_us  = rres.t_fwd_us;
                solve_adj_us  = rres.t_adj_us;
                solve_trsm_us = rres.t_trsm_us;
                hist = std::move(rres.history);
                have_hist = true;
                stop_reason = pcg_reason(flag);
            }
            peak_kb = mem.stop();
        } else if (is_bp) {
            rl::Blendenpik_linops<double, RNG> bp(true, tol);
            bp.nnz = sketch_nnz;
            bp.warm_start = (alg == "Blendenpik");   // "Blendenpik_cold" => x0 = 0
            // Give Blendenpik the SAME iteration budget as every other method. It
            // previously fell back to its internal min(4n,1000) default while the others
            // received the CLI maxit (3000 in the campaigns), a silent 3x handicap.
            bp.max_iters = maxit;
            qr_status = bp.call(A_hat, rhs.data(), mtot, x.data(), n, d_factor, state);
            peak_kb = mem.stop();
            if (qr_status == 0) { qr_us = bp.times[0] + bp.times[1]; solve_us = bp.times[2];
                // setup_us = the warm x0 build (ormqr + trsv + one operator apply),
                // measured in bp.times[4], its own column.
                setup_us = bp.warm_start ? bp.times[4] : 0;
                // Published rows run LSQR, so its workspace is live at the peak.
                analytical_kb = rl::blendenpik_linops_analytical_kb<double>(
                    mtot, n, d_factor, /*warm_start=*/bp.warm_start, /*with_lsqr=*/true);
                iters = bp.lsqr_iters; lsqr_iters_col = bp.lsqr_iters;
                // Take ownership instead of copying (same reasoning as the refine
                // rows above): bp.R_out is heap-owned and would otherwise be freed
                // when bp goes out of scope at the end of this branch.
                R_bp_owned = bp.R_out;
                bp.R_out = nullptr; bp.R_out_sz = 0;
                have_R = true;
                // Report the same convergence signals as the other methods: flag 0/1 for
                // met-tolerance / hit-cap, and a real solver residual instead of -1.
                flag = bp.converged ? 0 : 1;
                solver_relres = bp.final_relres;
                stop_reason = lsqr_reason(flag, bp.lsqr_stop_test);
                // LSQR's internal op split, otherwise discarded, which would
                // force -1 sentinels into these columns for every Blendenpik row.
                if (bp.lsqr_op_times.size() >= 3) {
                    solve_fwd_us = bp.lsqr_op_times[0];
                    solve_adj_us = bp.lsqr_op_times[1];
                    solve_trsm_us = bp.lsqr_op_times[2];
                } }
        } else if (is_unprec) {
            long lt[4] = {0};
            if (use_pcg) {
                flag = rl::restarted_pcg_ne<double>(A_hat, mtot, n, nullptr, 0, rhs.data(), x.data(),
                                                    tol, maxit, iters, pcg_restart_maxit, pcg_restart_drop,
                                                    pcg_max_restarts, &pcg_rounds, lt, &solver_relres,
                                                    20, 1e-3, 0.0, &hist);
                have_hist = true;
                stop_reason = pcg_reason(flag);
            } else {
                int lsqr_test = 0;
                flag = rl::lsqr<double>(A_hat, mtot, n, nullptr, 0, rhs.data(), x.data(), tol, tol,
                                        maxit, iters, lt, &solver_relres, &lsqr_test);
                lsqr_iters_col = iters;
                stop_reason = lsqr_reason(flag, lsqr_test);
            }
            solve_us = lt[3]; peak_kb = mem.stop();
            solve_fwd_us = lt[0]; solve_adj_us = lt[1]; solve_trsm_us = lt[2];  // t_trsm is 0 here (no R)
        } else {
            // Build R via the selected Q-less QR method, then shared LSQR with right precond R.
            if (alg == "CholQR") {
                rl::CholQR_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries; fold_chol_shift(qr, 1);
                    analytical_kb = rl::cholqr_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "CholQR2") {
                rl::CholQR2_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries; fold_chol_shift(qr, 2);
                    analytical_kb = rl::cholqr2_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "sCholQR3") {
                rl::sCholQR3_linops<double> qr(true, tol); qr.block_size = block_size;
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries; fold_chol_shift(qr, 3);
                    analytical_kb = rl::scholqr3_linops_analytical_kb<double>(mtot, n, block_size); }
            } else if (alg == "sCholQR3_basic") {
                rl::sCholQR3_linops_basic<double> qr(true, tol);
                qr_status = qr.call(A_hat, R.data(), n);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries; fold_chol_shift(qr, 3);
                    analytical_kb = rl::scholqr3_linops_basic_analytical_kb<double>(mtot, n); }
            } else { // CQRRT_linop
                rl::CQRRT_linops<double, RNG> qr(true, tol);
                qr.nnz = sketch_nnz; qr.block_size = block_size;
                qr.precond_method = rl::CQRRTLinopPrecond::TRSM_IDENTITY;
                qr_status = qr.call(A_hat, R.data(), n, d_factor, state);
                if (qr_status == 0) { qr_us = qr.total_us(); chol_retries = qr.n_chol_retries; fold_chol_shift(qr, 1);
                    analytical_kb = rl::cqrrt_linops_analytical_kb<double>(mtot, n, d_factor, block_size); }
            }
            if (qr_status == 0) {
                have_R = true;
                // Cold start unconditionally: the sketch-and-solve x0 warm start
                // is Blendenpik-only.
                long lt[4] = {0};
                if (use_pcg) {
                    flag = rl::restarted_pcg_ne<double>(A_hat, mtot, n, R.data(), n, rhs.data(), x.data(),
                                                        tol, maxit, iters, pcg_restart_maxit, pcg_restart_drop,
                                                        pcg_max_restarts, &pcg_rounds, lt, &solver_relres,
                                                        20, 1e-3, 0.0, &hist);
                    have_hist = true;
                    stop_reason = pcg_reason(flag);
                } else {
                    int lsqr_test = 0;
                    flag = rl::lsqr<double>(A_hat, mtot, n, R.data(), n, rhs.data(), x.data(), tol, tol,
                                            maxit, iters, lt, &solver_relres, &lsqr_test);
                    lsqr_iters_col = iters;
                    stop_reason = lsqr_reason(flag, lsqr_test);
                }
                solve_us = lt[3];
                solve_fwd_us = lt[0]; solve_adj_us = lt[1]; solve_trsm_us = lt[2];
            }
            peak_kb = mem.stop();
        }
        // Whole-row wall clock: build + setup + solve (metrics excluded; they are
        // computed below, after this timestamp). A total that the parts must sum
        // to is what exposes dropped time slices.
        long total_row_us = duration_cast<microseconds>(steady_clock::now() - t0).count();

        // Metrics: solver/data/aug/normal relres, recovery, orth, cond.
        double orth_err = -1, aug_relres = -1, data_relres = -1, normal_relres = -1, recov = -1;
        double cond_est = -1.0;   // -1 = not computed
        if (qr_status == 0) {
            if (have_R) {
                // R_bp_owned takes priority: it is set only for bp/bp_ref rows,
                // whose driver-owned R was never copied into the R vector below.
                const double* R_metrics = R_bp_owned ? R_bp_owned : R.data();
                compute_orth_and_cond(A_hat, R_metrics, mtot, n, block_size, cond_est_max_n, orth_err, cond_est);
            }
            // augResidual = A_hat x - rhs  (length mtot).
            A_hat(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mtot, 1, n, 1.0, x.data(), n, 0.0, Ax.data(), mtot);
            double ar = 0; for (int64_t i = 0; i < mtot; ++i) { Ax[i] -= rhs[i]; ar += Ax[i]*Ax[i]; }
            aug_relres = std::sqrt(ar) / std::max(rhs_norm, 1e-300);
            // normalResidual = A_hat^T (A_hat x - rhs); normalRelres = ||.|| / ||A^T rhs||.
            std::vector<double> nres(n);
            A_hat(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, mtot, 1.0, Ax.data(), mtot, 0.0, nres.data(), n);
            normal_relres = blas::nrm2(n, nres.data(), 1) / std::max(normalRhs_norm, 1e-300);
            T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x.data(), n, 0.0, Tx.data(), m);
            double dr = 0; for (int64_t i = 0; i < m; ++i) { double d = Tx[i] - b[i]; dr += d*d; }
            data_relres = std::sqrt(dr) / std::max(b_norm, 1e-300);
            double re = 0; for (int64_t i = 0; i < n; ++i) { double d = x[i] - x_true[i]; re += d*d; }
            recov = std::sqrt(re) / std::max(x_true_norm, 1e-300);
        }
        delete[] R_bp_owned;   // no-op if never set; R_metrics above was its only use

        std::printf("  qr_status=%d iters=%d flag=%d reason=%s qr_us=%lld setup_us=%lld solve_us=%lld total_row_us=%lld peak_kb=%lld\n",
                    qr_status, iters, flag, stop_reason.c_str(), (long long)qr_us, (long long)setup_us,
                    (long long)solve_us, (long long)total_row_us, (long long)peak_kb);
        // Solve breakdown, also echoed to the job log so anomalies are visible
        // without pulling the CSV. For pcg rows `other` contains the solver's
        // vector work, allocation, and (until every op is individually timed) any
        // residue; the restart-loop operator recomputations are inside fwd/adj/trsm.
        // Per-iteration numbers divide by INNER iterations and
        // include the per-round recomputation ops, so they overstate the pure
        // iteration cost when rounds ~ iters; the inner-only columns are the
        // trustworthy per-iteration source.
        if (solve_fwd_us >= 0) {
            long other = solve_us - (solve_fwd_us + solve_adj_us + solve_trsm_us);
            std::printf("  solve breakdown: fwd=%lld adj=%lld trsm=%lld other=%lld us"
                        "  (per-iter over %d iters: fwd=%.3f adj=%.3f trsm=%.3f ms)\n",
                        (long long)solve_fwd_us, (long long)solve_adj_us,
                        (long long)solve_trsm_us, (long long)other, iters,
                        iters > 0 ? solve_fwd_us  / 1000.0 / iters : 0.0,
                        iters > 0 ? solve_adj_us  / 1000.0 / iters : 0.0,
                        iters > 0 ? solve_trsm_us / 1000.0 / iters : 0.0);
        }
        std::printf("  orth=%.3e solver_relres=%.3e aug=%.3e normal=%.3e data=%.3e recovery=%.3e cond=%.3e retries=%d\n",
                    orth_err, solver_relres, aug_relres, normal_relres, data_relres, recov, cond_est, chol_retries);

        // Inner-kernel/overhead split (pcg rows; -1 where no history exists).
        long t_inner_us = -1, t_fwd_in = -1, t_adj_in = -1, t_trsm_in = -1, t_overhead_us = -1;
        if (have_hist) {
            t_inner_us = hist.t_inner_us;
            t_fwd_in   = hist.t_fwd_inner_us;
            t_adj_in   = hist.t_adj_inner_us;
            t_trsm_in  = hist.t_trsm_inner_us;
            t_overhead_us = solve_us - hist.t_inner_us;
        }

        // solver="none" when QR/build failed: no solver ever ran, so naming its
        // normal engine would fabricate a result.
        const char* solver_col = (qr_status != 0) ? "none"
                                : (is_bp_ref ? "pcg_ne" : ((is_bp || !use_pcg) ? "lsqr" : "pcg_ne"));
        out << alg << "," << run_idx << "," << m << "," << n << "," << qr_status << "," << qr_us << "," << solve_us << ","
            << peak_kb << "," << analytical_kb << ","
            << std::scientific << orth_err << "," << iters << "," << flag << ","
            << solver_relres << "," << aug_relres << "," << normal_relres << ","
            << data_relres << "," << recov << "," << cond_est << "," << chol_retries << ","
            << solve_fwd_us << "," << solve_adj_us << "," << solve_trsm_us << "," << setup_us << ","
            << solver_col << "," << pcg_rounds << ","
            << lsqr_iters_col << "," << stop_reason << ","
            << t_inner_us << "," << t_fwd_in << "," << t_adj_in << "," << t_trsm_in << ","
            << t_overhead_us << "," << total_row_us << "," << x0_relres << ","
            << chol_shift_abs << "," << chol_shift_rel << "\n";
        out.flush();   // partial results survive a scheduler kill mid-campaign

        if (have_hist) {
            for (size_t r = 0; r < hist.iters.size(); ++r) {
                rl::bench::write_round_row(out_rounds, alg, run_idx, r + 1,
                    hist.iters[r], hist.status[r], hist.relres[r],
                    hist.best_relres[r], hist.best_iter[r], hist.ls_relres[r]);
            }
            out_rounds.flush();
        }
    }
    }
    out_rounds.close();
    out.close();
    std::printf("\nresults -> %s\nrounds  -> %s\n", csv.c_str(), csv_rounds.c_str());
    return 0;
}
