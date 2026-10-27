// Toeplitz least-squares benchmark: regularized autoregression with a prolate-kernel Toeplitz
// matrix, solved matrix-free.
//
// Solves min ||T x - b||^2 + lambda ||x||^2 = min ||A x - rhs|| with A = [T; sqrt(lambda) I] and
// rhs = [b; 0]. T = toeplitz(c, r) is m x n with prolate-kernel generators, applied by FFT
// (extras/linops/ext_toeplitz_linop.hh), and lambda = lambda_rel * lambda_max(T^T T). Each
// QR method builds R from A, and restarted_pcg_ne (default) or LSQR solves from x = 0 with R
// as the right preconditioner. Double precision only.
//
// Method mask (--mask): 1 CQRRTO_linop, 2 CholQR, 4 sCholQR3, 8 sCholQR3_basic, 16 CholQR2,
// 32 Blendenpik and Blendenpik_cold (LSQR from its sketch-and-solve x0 or from zero, whatever
// --solver says), 64 Blendenpik_refine and Blendenpik_cold_refine (Blendenpik's R, then
// restarted_pcg_ne from x0 or from zero, whatever --solver says; see
// bench_CQRRTO_linops/refined_blendenpik.hh), 128 unpreconditioned (run before the refined
// rows). The bits mean the same in CQRRTO_linop_applications.
//
// Writes, under --out with a timestamp prefix: _toeplitz_ls_results.csv (one row per method and
// run), _toeplitz_ls_rounds.csv (per-round engine records) and _toeplitz_ls_backward_error.csv
// (the sketched backward error of each solution). Run with --help for the arguments.

#include "RandLAPACK.hh"
#include "../../extras/linops/ext_toeplitz_linop.hh"
#include "../bench_CQRRTO_linops/cqrrto_bench_common.hh"
#include "../bench_CQRRTO_linops/refined_blendenpik.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

using RNG = RandBLAS::DefaultRNG;
namespace rl    = RandLAPACK;
namespace bench = RandLAPACK::bench;
namespace rle   = RandLAPACK_extras::linops;
using blas::Layout; using blas::Op; using blas::Side;
using std::chrono::steady_clock; using std::chrono::duration_cast; using std::chrono::microseconds;

// In run order; the bits mean what they mean in CQRRTO_linop_applications.
static const std::vector<bench::MethodBit> kMethods = {
    {1,   {"CQRRTO_linop"}},
    {2,   {"CholQR"}},
    {4,   {"sCholQR3"}},
    {8,   {"sCholQR3_basic"}},
    {16,  {"CholQR2"}},
    {32,  {"Blendenpik", "Blendenpik_cold"}},
    {128, {"unpreconditioned"}},
    {64,  {"Blendenpik_refine", "Blendenpik_cold_refine"}},
};

static double prolate(int64_t k, double omega) {
    if (k == 0) return 2.0 * omega;
    const double kk = (double)k;
    return std::sin(2.0 * std::numbers::pi * omega * kk) / (std::numbers::pi * kk);
}

// lambda_max(A^T A) by power iteration with Rayleigh quotients from a fixed random start.
template <typename GLO>
static double power_lambda_max(GLO& A, int64_t m, int64_t n, int iters) {
    std::vector<double> x(n), Ax(m), Gx(n);
    std::mt19937 rng(12345);
    std::normal_distribution<double> nd(0, 1);
    for (auto& v : x) v = nd(rng);
    blas::scal(n, 1.0 / blas::nrm2(n, x.data(), 1), x.data(), 1);
    double lam = 0;
    for (int it = 0; it < iters; ++it) {
        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x.data(), n, 0.0, Ax.data(), m);
        A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, m, 1.0, Ax.data(), m, 0.0, Gx.data(), n);
        lam = blas::dot(n, x.data(), 1, Gx.data(), 1);
        const double gn = blas::nrm2(n, Gx.data(), 1);
        if (gn == 0) break;
        blas::scal(n, 1.0 / gn, Gx.data(), 1);
        std::copy(Gx.begin(), Gx.end(), x.begin());
    }
    return lam;
}

struct Settings {
    std::string out_dir, solver = "pcg_ne";
    int64_t m = 0, n = 0, method_mask = 255, sketch_nnz = 4, seed = 1, num_runs = 1;
    double  omega = 0.14, lambda_rel = 1e-20, tol = 1e-12, d_factor = 2.0;
    int     maxit = 3000, pcg_restart_maxit = 500, pcg_max_restarts = 50;
    double  round_drop = 1e-4, inner_tol = -1.0, be_tol_mult = 0.0;
    int     be_poll_every = 0;
};

static void print_usage(const char* exe) {
    std::fprintf(stderr,
      "Usage: %s --out=DIR --m=ROWS --n=COLS [options]\n"
      "\n"
      "  --out=DIR              output directory\n"
      "  --m=ROWS --n=COLS      shape of T (m >= n)\n"
      "\n"
      "Options (default in brackets):\n"
      "  --precision=double     [double]  only double is supported\n"
      "  --omega=F              [0.14]    prolate kernel bandwidth, in (0, 0.5]\n"
      "  --lambda-rel=F         [1e-20]   lambda / lambda_max(T^T T), >= 0\n"
      "  --mask=N               [255]     method bitmask (see the file header)\n"
      "  --tol=F                [1e-12]   solver tolerance on ||rhs - A x|| / ||rhs||, >= 0\n"
      "  --maxit=N              [3000]    solver iteration cap, >= 1\n"
      "  --d-factor=F           [2.0]     sketch rows per column, >= 1\n"
      "  --sketch-nnz=N         [4]       nonzeros per sketch column, >= 1\n"
      "  --seed=N               [1]       seed of the data and the sketches\n"
      "  --runs=N               [1]       repeats per method, each with its own sketch\n"
      "  --solver=pcg_ne|lsqr   [pcg_ne]  solver of the QR and unpreconditioned rows\n"
      "  --pcg-restart-maxit=N  [500]     inner CG iterations per round, >= 1\n"
      "  --pcg-max-restarts=N   [50]      rounds after the first; < 0 = unlimited\n"
      "  --round-drop=F         [1e-4]    per-round residual drop, in (0,1)\n"
      "  --inner-tol=F          [-1]      absolute inner floor; < 0 = eps^0.85, 0 = off\n"
      "  --be-tol-mult=F        [0]       also stop a restarted_pcg_ne run once the sketched\n"
      "                                   backward error is <= F sqrt(n) u; 0 = off\n"
      "  --be-poll-every=N      [0]       also test that target every N inner CG iterations\n"
      "\n"
      "QR driver options come from the environment (cqrrto_bench_common.hh).\n", exe);
}

// Parses and validates the arguments; throws std::runtime_error on an invalid one.
static Settings parse_settings(const bench::BenchArgs& a) {
    a.reject_unknown({"precision", "out", "m", "n", "omega", "lambda-rel", "mask", "tol", "maxit",
                      "d-factor", "sketch-nnz", "seed", "runs", "solver", "pcg-restart-maxit",
                      "pcg-max-restarts", "round-drop", "inner-tol", "be-tol-mult", "be-poll-every", "help"});
    Settings s;
    if (a.str("precision", "double") != "double")
        throw std::runtime_error("--precision must be double (this benchmark is double precision only)");
    s.out_dir           = a.require_str("out");
    s.m                 = a.i64("m", s.m);
    s.n                 = a.i64("n", s.n);
    s.omega             = a.dbl("omega", s.omega);
    s.lambda_rel        = a.dbl("lambda-rel", s.lambda_rel);
    s.method_mask       = a.i64("mask", s.method_mask);
    s.tol               = a.dbl("tol", s.tol);
    s.maxit             = a.i32("maxit", s.maxit);
    s.d_factor          = a.dbl("d-factor", s.d_factor);
    s.sketch_nnz        = a.i64("sketch-nnz", s.sketch_nnz);
    s.seed              = a.i64("seed", s.seed);
    s.num_runs          = a.i64("runs", s.num_runs);
    s.solver            = a.str("solver", s.solver);
    s.pcg_restart_maxit = a.i32("pcg-restart-maxit", s.pcg_restart_maxit);
    s.pcg_max_restarts  = a.i32("pcg-max-restarts", s.pcg_max_restarts);
    s.round_drop        = a.dbl("round-drop", s.round_drop);
    s.inner_tol         = a.dbl("inner-tol", s.inner_tol);
    s.be_tol_mult       = a.dbl("be-tol-mult", s.be_tol_mult);
    s.be_poll_every     = a.i32("be-poll-every", s.be_poll_every);

    auto require = [](bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); };
    require(s.n >= 1 && s.m >= s.n, "--m and --n are required, with m >= n >= 1");
    require(s.seed >= 0 && s.seed <= UINT32_MAX, "--seed must lie in [0, 2^32)");
    require(s.num_runs >= 1, "--runs must be >= 1");
    require(s.omega > 0 && s.omega <= 0.5, "--omega must lie in (0, 0.5]");
    require(s.lambda_rel >= 0, "--lambda-rel must be >= 0");
    require(s.d_factor >= 1, "--d-factor must be >= 1");
    require(s.sketch_nnz >= 1, "--sketch-nnz must be >= 1");
    require(s.solver == "pcg_ne" || s.solver == "lsqr", "--solver must be pcg_ne or lsqr");
    require(s.tol >= 0, "--tol must be >= 0");
    require(s.maxit >= 1 && s.pcg_restart_maxit >= 1, "--maxit and --pcg-restart-maxit must be >= 1");
    require(s.round_drop > 0 && s.round_drop < 1, "--round-drop must lie in (0,1)");
    require(s.inner_tol < 1, "--inner-tol must be < 1");
    require(s.be_poll_every >= 0, "--be-poll-every must be >= 0");
    require(s.be_poll_every == 0 || s.be_tol_mult > 0,
            "--be-poll-every needs --be-tol-mult > 0: the poll tests the backward-error target");
    bench::decode_method_mask(s.method_mask, kMethods);   // throws on an invalid mask
    return s;
}

static int run(const Settings& s, const std::string& argv_line) {
    const int64_t m = s.m, n = s.n, mtot = m + n;
    const int64_t block_size = rl::kDefaultGramBlockSize;
    const double relnoise = 1e-11;   // data noise level relative to ||T x_true||
    const bool use_pcg = (s.solver == "pcg_ne");
    const double inner_abs_tol = (s.inner_tol < 0) ? std::pow(std::numeric_limits<double>::epsilon(), 0.85)
                                                   : s.inner_tol;
    const auto rows_to_run = bench::decode_method_mask(s.method_mask, kMethods);
    const std::string stem = s.out_dir + "/" + bench::make_run_timestamp() + "_toeplitz_ls";
    const std::string csv = stem + "_results.csv", csv_rounds = stem + "_rounds.csv",
                      csv_kw = stem + "_backward_error.csv";
    std::ofstream out(csv), out_rounds(csv_rounds);
    if (!out || !out_rounds) {
        std::fprintf(stderr, "Error: cannot write %s and %s\n", csv.c_str(), csv_rounds.c_str());
        return 1;
    }
    std::printf("=== Toeplitz LS benchmark: m=%lld n=%lld omega=%.4f lambda_rel=%.1e mask=%lld solver=%s ===\n",
                (long long)m, (long long)n, s.omega, s.lambda_rel, (long long)s.method_mask, s.solver.c_str());

    // ---- Operator A = [T; sqrt(lambda) I] ----
    std::vector<double> c(m), r(n);
    for (int64_t i = 0; i < m; ++i) c[i] = prolate(i, s.omega);
    for (int64_t j = 0; j < n; ++j) r[j] = prolate(j, s.omega);
    rle::ToeplitzLinOp<double> T(c.data(), m, r.data(), n);
    const double lam_max = power_lambda_max(T, m, n, 50);
    const double lambda  = s.lambda_rel * lam_max;
    std::printf("lambda_max(T'T) ~ %.6e, lambda = %.6e\n", lam_max, lambda);
    rl::linops::ScaledIdentityLinOp<double> sqrt_lambda_I(n, std::sqrt(lambda));
    rl::linops::VStackLinOp<rle::ToeplitzLinOp<double>, rl::linops::ScaledIdentityLinOp<double>> A(T, sqrt_lambda_I);
    A.block_size = block_size;

    // ---- Data: smooth x_true, b = T x_true + noise, rhs = [b; 0] ----
    std::mt19937 rng((unsigned)s.seed);
    std::normal_distribution<double> nd(0, 1);
    std::vector<double> x_true(n);
    {
        // Gaussian filter of width 18 truncated at half-width min(n - 1, 80), then normalized.
        std::vector<double> z(n);
        for (auto& v : z) v = nd(rng);
        const int64_t hw = std::min<int64_t>(n - 1, 80);
        std::vector<double> filt(hw + 1);
        double fs = 0;
        for (int64_t i = 0; i <= hw; ++i) {
            filt[i] = std::exp(-std::pow(i / 18.0, 2));
            fs += (i == 0) ? filt[i] : 2 * filt[i];
        }
        for (int64_t i = 0; i < n; ++i) {
            double acc = 0;
            for (int64_t j = -hw; j <= hw; ++j)
                if (i + j >= 0 && i + j < n) acc += z[i + j] * filt[std::abs(j)];
            x_true[i] = acc / fs;
        }
        const double xn = blas::nrm2(n, x_true.data(), 1);
        blas::scal(n, 1.0 / std::max(xn, 1e-300), x_true.data(), 1);
    }
    std::vector<double> b(m);
    T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x_true.data(), n, 0.0, b.data(), m);
    {
        const double b_clean = blas::nrm2(m, b.data(), 1);
        std::vector<double> e(m);
        for (auto& v : e) v = nd(rng);
        const double en = blas::nrm2(m, e.data(), 1);
        for (int64_t i = 0; i < m; ++i) b[i] += e[i] / std::max(en, 1e-300) * relnoise * b_clean;
    }
    const double b_norm = blas::nrm2(m, b.data(), 1);
    std::vector<double> rhs(mtot, 0.0);
    std::copy(b.begin(), b.end(), rhs.begin());
    const double rhs_norm = blas::nrm2(mtot, rhs.data(), 1);
    const double x_true_norm = blas::nrm2(n, x_true.data(), 1);
    std::vector<double> ATrhs(n);   // for normal_relres = ||A^T (A x - rhs)|| / ||A^T rhs||
    A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, mtot, 1.0, rhs.data(), mtot, 0.0, ATrhs.data(), n);
    const double ATrhs_norm = blas::nrm2(n, ATrhs.data(), 1);

    // ---- Untimed warmup: a CholQR build, five LSQR iterations, and one sketch and geqrf at the
    // benchmark's dimensions, so thread pools, first-touch pages and the FFT and sketch paths
    // are not charged to the first measured row ----
    std::printf("Warmup ... ");
    std::fflush(stdout);
    {
        std::vector<double> Rw(n * n), xw(n);
        rl::CholQR_linops<double> qr(false, 0.0);
        qr.block_size = block_size;
        const int st = qr.call(A, Rw.data(), n);
        int it = 0;
        rl::lsqr<double>(A, mtot, n, (st == 0) ? Rw.data() : nullptr, n, rhs.data(), xw.data(),
                         s.tol, s.tol, 5, it);
        const int64_t d = rl::cqrrt_sketch_rows(s.d_factor, n);
        std::vector<double> Ask(d * n), tau(n);
        RandBLAS::SparseDist DS(d, mtot, std::min({s.sketch_nnz, d, mtot}));
        RandBLAS::SparseSkOp<double, RNG> S(DS, RandBLAS::RNGState<RNG>((uint32_t)s.seed));
        RandBLAS::fill_sparse(S);
        A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mtot, 1.0, S, 0.0, Ask.data(), d);
        lapack::geqrf(d, n, Ask.data(), d, tau.data());
    }
    std::printf("done\n");

    bench::BackwardErrorSetup<double> be = bench::setup_backward_error<double, RNG>(
        A, mtot, n, rhs_norm, bench::resolve_be_tol<double>(s.be_tol_mult, n), block_size);
    rl::RestartedPCGNEControls<double> ctl;
    ctl.tol                 = s.tol;
    ctl.max_iters           = s.maxit;
    ctl.max_iters_per_round = s.pcg_restart_maxit;
    ctl.round_drop          = s.round_drop;
    ctl.max_restarts        = s.pcg_max_restarts;
    ctl.inner_abs_tol       = inner_abs_tol;
    ctl.oracle              = be.oracle;
    ctl.be_tol              = be.be_tol;
    ctl.be_poll_every       = s.be_poll_every;

    // ---- CSV headers ----
    // The caps in effect: a cap <= 0 disables the FFT cap, and SolveWidthScope publishes no width
    // when width matching is off or the level-2 cap is disabled.
    const int fft_cap = rle::detail::fft_thread_cap();
    const int solve_width = rl::detail::solve_width_match_enabled() ? rl::blas2_thread_cap(n) : 0;
    out << "# Toeplitz LS benchmark m=" << m << " n=" << n << " omega=" << s.omega
        << " lambda_rel=" << s.lambda_rel << " lambda_max=" << lam_max << " lambda=" << lambda
        << " tol=" << s.tol << " d_factor=" << s.d_factor << " solver=" << s.solver
        << " pcg_restart_maxit=" << s.pcg_restart_maxit << " pcg_max_restarts=" << s.pcg_max_restarts
        << " pcg_restart_drop=" << s.round_drop << " inner_abs_tol=" << inner_abs_tol
        << " be_tol_mult=" << s.be_tol_mult << " be_tol=" << be.be_tol << " be_poll_every=" << s.be_poll_every
        << " kw_sketch_nnz=" << bench::kw_sketch_nnz(mtot, n) << "\n"
        << "# every row targets tol: LSQR rows with atol = btol = tol, except the warm Blendenpik row's"
        << " correction solve, which uses btol = tol ||b|| / ||r0||; restarted_pcg_ne rows on"
        << " ||rhs - A x|| / ||rhs||. The backward-error target acts only in restarted_pcg_ne runs.\n"
        << "# Blendenpik rows run LSQR from x0 or from zero; Blendenpik_refine and Blendenpik_cold_refine"
        << " take Blendenpik's R and run restarted_pcg_ne from x0 or from zero\n"
        << "# seed=" << s.seed << " maxit=" << s.maxit << " method_mask=" << s.method_mask
        << " sketch_nnz=" << s.sketch_nnz << " relnoise=" << relnoise << " block_size=" << block_size
        << " num_runs=" << s.num_runs << "\n"
        << "# threads: OpenMP " << rl::util::get_omp_threads() << ", FFT cap "
        << (fft_cap > 0 ? std::to_string(fft_cap) : std::string("off")) << ", solve width "
        << (solve_width > 0 ? std::to_string(solve_width) : std::string("off")) << "\n";
    bench::write_env_lines(out);
    out << "# argv: " << argv_line << "\n"
        << "# -1 marks a value that is not computed or does not apply, except these zeros: qr_time_us\n"
        << "#   of the unpreconditioned row, setup_us outside the warm Blendenpik rows, lsqr_iters of\n"
        << "#   engine rows, and solve_time_us, iterations, lsqr_iters and setup_us of rows whose\n"
        << "#   build failed (which also write solver=none, solver_flag=-1 and qr_time_us=-1).\n"
        << "#   iterations: engine CG iterations, or LSQR's; pcg_rounds: engine rounds; t_*_inner_us:\n"
        << "#   inside the engine's inner CG; t_overhead_us = solve - inner CG; total_row_us: wall\n"
        << "#   clock of build, setup and solve, excluding the oracle; for Blendenpik_cold_refine it\n"
        << "#   also holds the unused x0 build.\n"
        << "algorithm,run,m,n,qr_status,qr_time_us,solve_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,iterations,solver_flag,solver_relres,aug_relres,normal_relres,"
           "data_relres,recovery_error,cond_estimate,chol_retries,"
           "solve_fwd_us,solve_adj_us,solve_trsm_us,setup_us,solver,pcg_rounds,"
           "lsqr_iters,stop_reason,t_inner_us,t_fwd_inner_us,t_adj_inner_us,"
           "t_trsm_inner_us,t_overhead_us,total_row_us,x0_relres,chol_shift_abs,chol_shift_rel,"
           "t_be_us,be_x0,be_final,t_poll_us\n";
    bench::write_rounds_header(out_rounds);

    std::vector<bench::Solution<double>> solutions;
    std::vector<double> R(n * n), x(n), Tx(m), Ax(mtot), nres(n);

    for (const auto& alg : rows_to_run) {
        for (int64_t run_idx = 0; run_idx < s.num_runs; ++run_idx) {
            std::printf("\n=== %s (run %lld/%lld) ===\n", alg.c_str(), (long long)(run_idx + 1), (long long)s.num_runs);
            std::fill(R.begin(), R.end(), 0.0);
            std::fill(x.begin(), x.end(), 0.0);
            auto state = RandBLAS::RNGState<RNG>((uint32_t)s.seed);
            if (run_idx > 0) state.key.incr(run_idx);
            const bool is_bp_ref = alg.find("_refine") != std::string::npos;
            const bool is_bp     = !is_bp_ref && alg.rfind("Blendenpik", 0) == 0;
            const bool is_unprec = (alg == "unpreconditioned");

            int qr_status = 0, iters = 0, flag = -1, chol_retries = -1, pcg_rounds = -1, lsqr_iters = 0;
            long qr_us = 0, solve_us = 0, setup_us = 0, peak_kb = 0, analytical_kb = -1;
            long solve_fwd_us = -1, solve_adj_us = -1, solve_trsm_us = -1;
            double chol_shift_abs = -1, chol_shift_rel = -1, solver_relres = -1, x0_relres = -1;
            std::string stop_reason = "n/a";
            rl::RestartedPCGNEReport<double> rep;
            bool have_rep = false;
            std::unique_ptr<double[]> R_bp;   // a Blendenpik row's factor

            auto record_engine = [&](rl::RestartedPCGNETermination term) {
                have_rep      = true;
                flag          = static_cast<int>(term);
                stop_reason   = bench::pcg_stop_reason(term);
                iters         = rep.iters_done;
                pcg_rounds    = rep.rounds_done;
                solver_relres = rep.final_relres;
                solve_us      = rep.times[3];
                solve_fwd_us  = rep.times[0];
                solve_adj_us  = rep.times[1];
                solve_trsm_us = rep.times[2];
            };
            auto engine = [&](const double* Rp, const double* x0) {
                record_engine(rl::restarted_pcg_ne<double>(A, mtot, n, Rp, n, rhs.data(), x.data(), ctl, rep, x0));
            };
            // The configured solver from x = 0 with right preconditioner Rp (nullptr: none).
            auto solve = [&](const double* Rp) {
                if (use_pcg) { engine(Rp, nullptr); return; }
                long lt[4] = {0, 0, 0, 0};
                int test = 0;
                flag = rl::lsqr<double>(A, mtot, n, Rp, n, rhs.data(), x.data(), s.tol, s.tol, s.maxit,
                                        iters, lt, &solver_relres, &test);
                lsqr_iters   = iters;
                stop_reason  = bench::lsqr_stop_reason(flag == 0, test);
                solve_us     = lt[3];
                solve_fwd_us = lt[0]; solve_adj_us = lt[1]; solve_trsm_us = lt[2];
            };

            rl::PeakRSSTracker mem;
            mem.start();
            auto t0 = steady_clock::now();
            if (is_bp_ref) {
                auto rr = bench::run_refined_blendenpik<double, RNG>(A, rhs.data(), mtot, n, s.d_factor,
                                                                     s.sketch_nnz, state, alg == "Blendenpik_refine",
                                                                     engine);
                qr_status = rr.qr_status;
                qr_us     = rr.qr_us;
                if (qr_status == 0) {
                    setup_us      = rr.setup_us;
                    x0_relres     = rr.x0_relres;
                    analytical_kb = rl::blendenpik_linops_analytical_kb<double>(mtot, n, s.d_factor, s.sketch_nnz,
                                                                                true, false);
                    R_bp          = std::move(rr.R);
                }
            } else if (is_bp) {
                rl::Blendenpik_linops<double, RNG> bp(true, s.tol);
                bp.tol        = s.tol;   // the constructor maps tol <= 0 to eps
                bp.nnz        = s.sketch_nnz;
                bp.warm_start = (alg == "Blendenpik");
                bp.max_iters  = s.maxit;
                qr_status = bp.call(A, rhs.data(), mtot, x.data(), n, s.d_factor, state);
                qr_us     = -1;
                if (qr_status == 0) {
                    qr_us         = bp.times[0] + bp.times[1];
                    solve_us      = bp.times[2];
                    setup_us      = bp.warm_start ? bp.times[4] : 0;
                    analytical_kb = rl::blendenpik_linops_analytical_kb<double>(mtot, n, s.d_factor, s.sketch_nnz,
                                                                                bp.warm_start, true);
                    iters = lsqr_iters = bp.lsqr_iters;
                    R_bp.reset(bp.R_out);
                    bp.R_out = nullptr;
                    flag          = bp.converged ? 0 : 1;
                    solver_relres = bp.final_relres;
                    stop_reason   = bench::lsqr_stop_reason(bp.converged, bp.lsqr_stop_test);
                    solve_fwd_us  = bp.lsqr_op_times[0];
                    solve_adj_us  = bp.lsqr_op_times[1];
                    solve_trsm_us = bp.lsqr_op_times[2];
                }
            } else if (is_unprec) {
                solve(nullptr);
            } else {
                bench::QRRun q = bench::run_qr_method<double, RNG>(alg, A, R.data(), n, block_size, s.sketch_nnz,
                                                                   s.d_factor, state);
                qr_status      = q.status;
                qr_us          = q.qr_time_us;
                analytical_kb  = q.analytical_kb;
                chol_retries   = q.chol_retries;
                chol_shift_abs = q.chol_shift_abs;
                chol_shift_rel = q.chol_shift_rel;
                if (qr_status == 0) solve(R.data());
            }
            long total_row_us = duration_cast<microseconds>(steady_clock::now() - t0).count();
            peak_kb = mem.stop();
            if (have_rep) total_row_us -= rep.t_be_us;
            const double* R_used = (qr_status != 0 || is_unprec) ? nullptr : (R_bp ? R_bp.get() : R.data());

            // ---- Metrics, outside the timed and RSS windows ----
            double orth_err = -1, cond_est = -1, aug_relres = -1, normal_relres = -1, data_relres = -1, recov = -1;
            if (qr_status == 0) {
                if (R_used != nullptr)
                    orth_err = bench::compute_orth_error_explicit<double>(A, R_used, block_size, &cond_est);
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mtot, 1, n, 1.0, x.data(), n, 0.0, Ax.data(), mtot);
                for (int64_t i = 0; i < mtot; ++i) Ax[i] -= rhs[i];
                aug_relres = blas::nrm2(mtot, Ax.data(), 1) / std::max(rhs_norm, 1e-300);
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, 1, mtot, 1.0, Ax.data(), mtot, 0.0, nres.data(), n);
                normal_relres = blas::nrm2(n, nres.data(), 1) / std::max(ATrhs_norm, 1e-300);
                T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, 1, n, 1.0, x.data(), n, 0.0, Tx.data(), m);
                double dr = 0, re = 0;
                for (int64_t i = 0; i < m; ++i) dr += (Tx[i] - b[i]) * (Tx[i] - b[i]);
                for (int64_t i = 0; i < n; ++i) re += (x[i] - x_true[i]) * (x[i] - x_true[i]);
                data_relres = std::sqrt(dr) / std::max(b_norm, 1e-300);
                recov       = std::sqrt(re) / std::max(x_true_norm, 1e-300);
                solutions.push_back({alg, run_idx, x});
            }

            long t_inner_us = -1, t_fwd_in = -1, t_adj_in = -1, t_trsv_in = -1, t_overhead_us = -1;
            long t_be_us = -1, t_poll_us = -1;
            double be_x0 = -1, be_final = -1;
            if (have_rep) {
                t_inner_us    = rep.t_inner_us;
                t_fwd_in      = rep.t_fwd_inner_us;
                t_adj_in      = rep.t_adj_inner_us;
                t_trsv_in     = rep.t_trsv_inner_us;
                t_overhead_us = solve_us - rep.t_inner_us;
                t_be_us       = rep.t_be_us;
                t_poll_us     = rep.t_poll_us;
                be_x0         = rep.be_x0;
                be_final      = bench::final_oracle_value(rep, be.be_tol >= 0);
            }
            std::printf("  qr_status=%d iters=%d flag=%d reason=%s qr_us=%lld setup_us=%lld solve_us=%lld"
                        " total_row_us=%lld peak_kb=%lld\n",
                        qr_status, iters, flag, stop_reason.c_str(), (long long)qr_us, (long long)setup_us,
                        (long long)solve_us, (long long)total_row_us, (long long)peak_kb);
            if (solve_fwd_us >= 0)
                std::printf("  solve: fwd=%lld adj=%lld trsv=%lld other=%lld us\n", (long long)solve_fwd_us,
                            (long long)solve_adj_us, (long long)solve_trsm_us,
                            (long long)(solve_us - (solve_fwd_us + solve_adj_us + solve_trsm_us)));
            std::printf("  orth=%.3e solver_relres=%.3e aug=%.3e normal=%.3e data=%.3e recovery=%.3e cond=%.3e"
                        " retries=%d\n", orth_err, solver_relres, aug_relres, normal_relres, data_relres, recov,
                        cond_est, chol_retries);

            const char* solver_col = (qr_status != 0) ? "none"
                                   : (is_bp_ref || (!is_bp && use_pcg)) ? "pcg_ne" : "lsqr";
            out << alg << "," << run_idx << "," << m << "," << n << "," << qr_status << "," << qr_us << ","
                << solve_us << "," << peak_kb << "," << analytical_kb << "," << std::scientific << orth_err << ","
                << iters << "," << flag << "," << solver_relres << "," << aug_relres << "," << normal_relres << ","
                << data_relres << "," << recov << "," << cond_est << "," << chol_retries << ","
                << solve_fwd_us << "," << solve_adj_us << "," << solve_trsm_us << "," << setup_us << ","
                << solver_col << "," << pcg_rounds << "," << lsqr_iters << "," << stop_reason << ","
                << t_inner_us << "," << t_fwd_in << "," << t_adj_in << "," << t_trsv_in << ","
                << t_overhead_us << "," << total_row_us << "," << x0_relres << ","
                << chol_shift_abs << "," << chol_shift_rel << "," << t_be_us << "," << be_x0 << ","
                << be_final << "," << t_poll_us << "\n";
            out.flush();   // partial results survive a job killed mid-run
            if (have_rep) {
                bench::write_rounds(out_rounds, alg, run_idx, rep);
                out_rounds.flush();
            }
        }
    }
    out.close();
    out_rounds.close();

    // Backward error of every solution against A x ~ rhs.
    bench::write_backward_error_sidecar<double, RNG>(csv_kw, be, A, mtot, n, rhs.data(), rhs_norm, block_size,
                                                     solutions);
    for (const auto& f : {csv, csv_rounds, csv_kw}) {
        bench::check_csv_arity(f);
        std::printf("wrote %s\n", f.c_str());
    }
    return 0;
}

int main(int argc, char** argv) {
    Settings s;
    try {
        bench::BenchArgs a(argc, argv);
        if (argc < 2 || a.has("help")) { print_usage(argv[0]); return argc < 2 ? 1 : 0; }
        s = parse_settings(a);
        bench::qr_driver_options();   // reject invalid environment values before any work
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n\n", e.what());
        print_usage(argv[0]);
        return 1;
    }
    try {
        return run(s, bench::quote_join_argv(argc, argv));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 2;
    }
}
