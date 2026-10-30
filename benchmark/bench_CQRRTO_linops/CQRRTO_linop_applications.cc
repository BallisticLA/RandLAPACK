// FEM least-squares benchmark: QR factors as right preconditioners for iterative refinement.
//
//   1. Load the FEM triple K (stiffness), M (mass) and V (prolongation) from .mtx files.
//   2. J = L^{-1} K V with M = L L^T (CholSolverLinOp, half solve), applied matrix-free.
//   3. Each QR method (the Q-less CQRRTO_linop, CholQR, CholQR2 and sCholQR3, and
//      sCholQR3_basic, which forms Q) factors A_hat = [J; mu I] (J itself with
//      --mu-factor=0) in the preconditioner precision, so that R^T R = J^T J + mu^2 I.
//   4. IterRefineLSQ solves min ||b - J x|| in the solve precision from x = 0, with R as the
//      right preconditioner. The unpreconditioned row runs it without R, and the refined
//      Blendenpik rows with Blendenpik's R; the published Blendenpik rows run LSQR.
//
// Method mask (--mask): 1 CQRRTO_linop, 2 CholQR, 4 sCholQR3, 8 sCholQR3_basic, 16 CholQR2,
// 32 Blendenpik (LSQR from its sketch-and-solve x0) and Blendenpik_cold (LSQR from zero),
// 64 Blendenpik_refine and Blendenpik_cold_refine (refined_blendenpik.hh), 128 unpreconditioned.
//
// Writes, under --out with a timestamp prefix: _irlsq_reg_results.csv (one row per method and
// run), _irlsq_reg_breakdown.csv (phase times), _irlsq_reg_rounds.csv (per-round engine
// records) and _irlsq_reg_backward_error.csv (the sketched backward error of each solution).
// Run with --help for the arguments.

#include "RandLAPACK.hh"
#include "../../extras/linops/ext_cholsolver_linop.hh"
#include "cqrrto_bench_common.hh"
#include "cqrrto_bench_io.hh"
#include "refined_blendenpik.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace rl = RandLAPACK;
namespace bench = RandLAPACK::bench;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

static const std::vector<bench::MethodBit> kMethods = {
    {1,   {"CQRRTO_linop"}},
    {2,   {"CholQR"}},
    {4,   {"sCholQR3"}},
    {8,   {"sCholQR3_basic"}},
    {16,  {"CholQR2"}},
    {32,  {"Blendenpik", "Blendenpik_cold"}},
    {64,  {"Blendenpik_refine", "Blendenpik_cold_refine"}},
    {128, {"unpreconditioned"}},
};
/// The bits of kMethods that select a QR method.
static constexpr int64_t kQRMethodBits = 1 | 2 | 4 | 8 | 16;

// ============================================================================
// Settings
// ============================================================================

struct Settings {
    std::string out_dir, K_file, M_file, V_file, precond_prec = "single";
    int64_t num_runs = 1, sketch_nnz = 4, block_size = rl::kDefaultGramBlockSize, method_mask = 31;
    double  d_factor = 2.0, noise_level = 0.05, mu_factor = 10.0;
    bool    compute_cond = false;
    bool    noise_given = false;
    // Refinement (see print_usage).
    int     max_inner = 200;
    double  inner_tol = -1.0;
    double  round_drop = 1e-4;
    int     n_steps = 50;
    double  outer_tol = -1.0;
    int     outer_stag_window = 2;
    double  be_tol_mult = 0.0;
};

template <typename T>
static T inner_tol_eff(const Settings& s) {
    return (s.inner_tol < 0) ? std::pow(std::numeric_limits<T>::epsilon(), (T)0.85) : (T)s.inner_tol;
}

template <typename T>
static T outer_tol_eff(const Settings& s) {
    return (s.outer_tol < 0) ? (T)10 * std::numeric_limits<T>::epsilon() : (T)s.outer_tol;
}

// The refinement of every row that is not a published Blendenpik row.
template <typename T>
static rl::IterRefineLSQ<T> make_ir(const Settings& s, const bench::BackwardErrorSetup<T>& be) {
    rl::IterRefineLSQ<T> ir(inner_tol_eff<T>(s), s.max_inner, s.n_steps, /*timing_on=*/true,
                            /*verbose_on=*/false);
    ir.round_drop        = (T)s.round_drop;
    ir.outer_tol         = outer_tol_eff<T>(s);
    ir.outer_stag_window = s.outer_stag_window;
    ir.oracle            = be.oracle;
    ir.be_tol            = be.be_tol;
    return ir;
}

// ============================================================================
// Helpers
// ============================================================================

// CSR copy with the values cast to Tdst.
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

// ||A||_2 by power iteration on A^T A from a fixed random start (util::estimate_spectral_norm
// needs a dense matrix).
template <typename T, typename GLO>
static T estimate_op_2norm(GLO& A, int64_t m, int64_t n, int iters) {
    std::vector<T> v(n), Av(m);
    std::mt19937 rng(7);
    std::normal_distribution<T> N01(0, 1);
    for (auto& e : v) e = N01(rng);
    T sigma = 0;
    for (int it = 0; it < iters; ++it) {
        T nv = blas::nrm2(n, v.data(), 1);
        if (nv == 0) return T(0);
        blas::scal(n, T(1) / nv, v.data(), 1);
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
          m, 1, n, T(1), v.data(), n, T(0), Av.data(), m);
        sigma = blas::nrm2(m, Av.data(), 1);
        A(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
          n, 1, m, T(1), Av.data(), m, T(0), v.data(), n);
    }
    return sigma;
}

// max |R_ii| / min |R_ii| over the nonzero diagonal, or -1.
template <typename P>
static double kappa_from_R_diag(const P* R, int64_t n) {
    double mx = 0.0, mn = std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        double v = std::abs((double)R[i + i * n]);
        mx = std::max(mx, v);
        if (v > 0) mn = std::min(mn, v);
    }
    return std::isfinite(mn) ? mx / mn : -1.0;
}

// ============================================================================
// Result rows
// ============================================================================

// One (method, run) row. -1 marks a value that does not apply or was not computed.
template <typename T>
struct Row {
    std::string alg;
    int64_t run = 0;
    int    qr_status = 0;
    long   qr_time_us = 0;             // 0 for the unpreconditioned row, -1 after a failed build
    long   peak_rss_kb = -1;           // build + solve, diagnostics excluded
    long   analytical_kb = -1;
    T      orth_error = -1;            // ||Q^T Q - I||_F / sqrt(n), Q = J R^{-1}
    long   ir_total_us = -1;
    int    ir_outer_iters = 0;
    int    ir_inner_iters_total = 0;   // engine CG iterations, or LSQR's (published rows)
    T      ls_residual_norm = -1;      // ||J x - b|| / (||J||_2 ||x|| + ||b||)
    T      ls_solution_error = -1;     // ||x - x_true|| / ||x_true||
    T      kappa_measured = -1;        // max |R_ii| / min |R_ii|
    int    chol_retries = -1;          // -1: no shift-retried Cholesky in this row
    double chol_shift_abs = -1, chol_shift_rel = -1;
    bench::WorstRound worst;           // ir_inner_* columns; published rows: LSQR's
    T      cond_precond = -1;          // cond(J R^{-1}) with --compute-cond
    long   ir_setup_us = 0;            // x0 build of the warm Blendenpik rows
    int    lsqr_iters = 0;
    int    engine_status = -1;         // RestartedPCGNETermination; -1 when the engine did not run
    std::string stop_reason = "n/a";
    T      x0_relres = -1;             // ||b - J x0|| / ||b|| of Blendenpik_refine's x0
    long   t_be_us = -1;               // time inside the oracle, outside every solve time
    T      be_x0 = -1, be_final = -1;  // oracle value of x0 and of the returned x
    std::vector<long> qr_breakdown, ir_breakdown;
    rl::RestartedPCGNEReport<T> report;   // per-round records; empty when the engine did not run
};

// The outcome of one IterRefineLSQ run.
template <typename T>
static void record_ir(Row<T>& r, int ir_status, rl::IterRefineLSQ<T>& ir, bool oracle_on) {
    if (ir_status != 0) std::cerr << "Warning: IterRefineLSQ inner CG broke down\n";
    r.ir_total_us          = ir.times[0];
    r.ir_outer_iters       = ir.report.rounds_done;
    r.ir_inner_iters_total = ir.report.iters_done;
    r.engine_status        = static_cast<int>(ir.termination_reason);
    r.stop_reason          = bench::pcg_stop_reason(ir.termination_reason);
    r.worst                = bench::worst_round(ir.report);
    r.t_be_us              = ir.report.t_be_us;
    r.be_x0                = ir.report.be_x0;
    r.be_final             = bench::final_oracle_value(ir.report, oracle_on);
    r.ir_breakdown         = ir.times;
    r.report               = std::move(ir.report);
}

// The Blendenpik rows: build and solve on the base operator J, R into R_out. Stops mem.
template <typename T, typename RNG, typename GLO>
static void run_blendenpik_row(GLO& J, const T* b, int64_t m, int64_t n, T* x, T* R_out,
                               const Settings& s, RandBLAS::RNGState<RNG> state,
                               const bench::BackwardErrorSetup<T>& be,
                               rl::PeakRSSTracker& mem, Row<T>& r) {
    const bool warm = (r.alg == "Blendenpik") || (r.alg == "Blendenpik_refine");
    if (r.alg.find("_refine") != std::string::npos) {
        rl::IterRefineLSQ<T> ir = make_ir<T>(s, be);
        int ir_status = 0;
        auto rr = bench::run_refined_blendenpik<T, RNG>(
            J, b, m, n, (T)s.d_factor, s.sketch_nnz, state, warm,
            [&](const T* R, const T* x0) { ir_status = ir.call(J, R, n, b, m, x, n, x0); });
        r.peak_rss_kb = mem.stop();
        r.qr_status = rr.qr_status;
        if (r.qr_status != 0) return;
        r.qr_time_us    = rr.qr_us;
        r.ir_setup_us   = rr.setup_us;
        r.x0_relres     = rr.x0_relres;
        std::copy(rr.R.get(), rr.R.get() + n * n, R_out);
        r.qr_breakdown  = {0L, rr.qr_us, 0L, 0L, rr.setup_us};
        r.analytical_kb = rl::blendenpik_linops_analytical_kb<T>(m, n, s.d_factor, s.sketch_nnz,
                                                                 /*with_x0=*/true, /*with_lsqr=*/false);
        record_ir(r, ir_status, ir, be.be_tol >= 0);
        return;
    }

    // Published rows: LSQR to eps^0.85 within the refinement rows' iteration budget.
    rl::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true,
                                     std::pow(std::numeric_limits<T>::epsilon(), (T)0.85));
    bp.nnz        = s.sketch_nnz;
    bp.warm_start = warm;
    bp.max_iters  = s.max_inner * s.n_steps;
    r.qr_status = bp.call(J, b, m, x, n, (T)s.d_factor, state);
    r.peak_rss_kb = mem.stop();
    if (r.qr_status != 0) return;
    r.qr_time_us    = bp.times[0] + bp.times[1];
    r.ir_setup_us   = warm ? bp.times[4] : 0;
    std::copy(bp.R_out, bp.R_out + n * n, R_out);
    r.qr_breakdown  = {bp.times[0], bp.times[1], 0L, 0L, bp.times[4]};
    r.analytical_kb = rl::blendenpik_linops_analytical_kb<T>(m, n, s.d_factor, s.sketch_nnz,
                                                             /*with_x0=*/warm, /*with_lsqr=*/true);
    r.worst.status         = bp.converged ? 0 : 1;
    r.worst.relres         = (double)bp.final_relres;
    r.ir_total_us          = bp.times[2];
    r.ir_outer_iters       = 1;
    r.ir_inner_iters_total = bp.lsqr_iters;
    r.lsqr_iters           = bp.lsqr_iters;
    r.stop_reason          = bench::lsqr_stop_reason(bp.converged, bp.lsqr_stop_test);
    const long fwd = bp.lsqr_op_times[0], adj = bp.lsqr_op_times[1], trsv = bp.lsqr_op_times[2];
    r.ir_breakdown = {bp.times[2], 0L, trsv, fwd, adj, std::max(0L, bp.times[2] - (fwd + adj + trsv))};
}

// ============================================================================
// CSV writers
// ============================================================================

// Per-run constants that every results row repeats.
struct RunConstants {
    int64_t m, n;
    double mu;
    std::string precond_prec, solve_prec;
};

template <typename T>
static void write_results(const std::string& filename, const std::string& header,
                          const RunConstants& c, const std::vector<Row<T>>& rows) {
    std::ofstream out(filename);
    out << header
        << "algorithm,run,m,n,qr_status,qr_time_us,peak_rss_kb,analytical_kb,"
           "orth_error,ir_total_us,ir_outer_iters,ir_inner_iters_total,"
           "ls_residual_norm,ls_solution_error,kappa_target,kappa_measured,mu,precond_prec,solve_prec,"
           "chol_retries,ir_inner_capped,ir_inner_relres,ir_inner_best_relres,ir_inner_best_iter,"
           "cond_precond,ir_setup_us,lsqr_iters,engine_status,stop_reason,x0_relres,"
           "chol_shift_abs,chol_shift_rel,t_be_us,be_x0,be_final\n";
    for (const auto& r : rows) {
        out << r.alg << "," << r.run << "," << c.m << "," << c.n << ","
            << r.qr_status << "," << r.qr_time_us << "," << r.peak_rss_kb << "," << r.analytical_kb << ","
            << std::scientific << std::setprecision(6) << r.orth_error << ","
            << r.ir_total_us << "," << r.ir_outer_iters << "," << r.ir_inner_iters_total << ","
            << r.ls_residual_norm << "," << r.ls_solution_error << ","
            << 1.0 << "," << r.kappa_measured << "," << c.mu << ","
            << c.precond_prec << "," << c.solve_prec << "," << r.chol_retries << ","
            << r.worst.status << "," << r.worst.relres << "," << r.worst.best_relres << ","
            << r.worst.best_iter << "," << r.cond_precond << "," << r.ir_setup_us << ","
            << r.lsqr_iters << "," << r.engine_status << "," << r.stop_reason << ","
            << r.x0_relres << "," << r.chol_shift_abs << "," << r.chol_shift_rel << ","
            << r.t_be_us << "," << r.be_x0 << "," << r.be_final << "\n";
    }
}

// One row per phase: the phase vector in t0..t17 (unused slots 0) and the row's own total.
template <typename T>
static void write_breakdown(const std::string& filename, const std::vector<Row<T>>& rows) {
    std::ofstream out(filename);
    out << "# Runtime breakdown (microseconds). QR phase slots by method:\n"
        << "#   CQRRTO_linop   (t0-t10): alloc,sketch,qr,tri_inv,fwd,adj,left,chol,finalize,rest,total\n"
        << "#   CholQR         (t0-t5):  alloc,fwd,adj,chol,rest,total\n"
        << "#   CholQR2        (t0-t12): alloc,fwd1,adj1,chol1,upd1,fwd2,adj2,left2,chol2,upd2,q_mat,rest,total\n"
        << "#   sCholQR3       (t0-t17): alloc,fwd1,adj1,chol1,upd1,fwd2,adj2,left2,chol2,upd2,\n"
        << "#                            fwd3,adj3,left3,chol3,upd3,q_mat,rest,total\n"
        << "#   sCholQR3_basic (t0-t14): alloc,fwd1,adj1,chol1,trsm1,fwd_q,syrk2,chol2,upd2,\n"
        << "#                            syrk3,chol3,upd3,q_mat,rest,total\n"
        << "#   Blendenpik rows (t0-t4): sketch,qr,0,0,x0 setup; the refined rows report sketch + QR in t1;\n"
        << "#     cold rows report no x0 setup (Blendenpik_cold_refine's unused x0 build is in no column)\n"
        << "# IR phase (t0-t5): total,inner_cg,trsv,fwd,adj,other (published Blendenpik rows: LSQR, inner_cg = 0)\n"
        << "# t_total: qr_time_us on the QR row, ir_total_us on the IR row.\n"
        << "algorithm,run,phase,t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t_total\n";
    auto row = [&](const Row<T>& r, const char* phase, const std::vector<long>& v, long total) {
        out << r.alg << "," << r.run << "," << phase;
        for (size_t i = 0; i < 18; ++i) out << "," << (i < v.size() ? v[i] : 0L);
        out << "," << total << "\n";
    };
    for (const auto& r : rows) {
        row(r, "QR", r.qr_breakdown, r.qr_time_us);
        row(r, "IR", r.ir_breakdown, r.ir_total_us);
    }
}

template <typename T>
static void write_rounds_csv(const std::string& filename, const std::vector<Row<T>>& rows) {
    std::ofstream out(filename);
    bench::write_rounds_header(out);
    for (const auto& r : rows) bench::write_rounds(out, r.alg, r.run, r.report);
}

// ============================================================================
// Benchmark
// ============================================================================

template <typename T_solve, typename P_precond, typename RNG = r123::Philox4x32>
static int run(const Settings& s, const std::string& argv_line) {
    namespace rll = RandLAPACK::linops;
    using T = T_solve;
    using P = P_precond;
    using CSR_T = RandBLAS::sparse_data::csr::CSRMatrix<T>;
    using CSR_P = RandBLAS::sparse_data::csr::CSRMatrix<P>;
    using CholSolver_T = RandLAPACK_extras::linops::CholSolverLinOp<T>;
    using CholSolver_P = RandLAPACK_extras::linops::CholSolverLinOp<P>;
    const std::string solve_prec   = std::is_same_v<T, double> ? "double" : "single";
    const std::string precond_prec = std::is_same_v<P, double> ? "double" : "single";
    const auto rows_to_run = bench::decode_method_mask(s.method_mask, kMethods);
    const std::string stamp = bench::make_run_timestamp();
    const std::string stem = s.out_dir + "/" + stamp + "_irlsq_reg";
    const std::string results_file = stem + "_results.csv", breakdown_file = stem + "_breakdown.csv",
                      rounds_file  = stem + "_rounds.csv",  kw_file        = stem + "_backward_error.csv";
    if (!std::ofstream(results_file)) {
        std::cerr << "Error: cannot write " << results_file << "\n";
        return 1;
    }

    // ---- Operators in both precisions; the double inputs are freed once they are copied ----
    int64_t m, n, nnz_K;
    std::unique_ptr<CSR_T> K_T, V_T;
    std::unique_ptr<CSR_P> K_P, V_P;
    std::unique_ptr<CholSolver_T> L_inv_T;
    std::unique_ptr<CholSolver_P> L_inv_P;
    {
        int64_t m_K, n_K, m_M, n_M, nnz_M, m_V, n_V, nnz_V;
        std::cout << "Loading K, M, V ... " << std::flush;
        auto K = bench::load_csr<double>(s.K_file, m_K, n_K, nnz_K);
        auto M = bench::load_csr<double>(s.M_file, m_M, n_M, nnz_M);
        auto V = bench::load_csr<double>(s.V_file, m_V, n_V, nnz_V);
        std::cout << "done (K " << m_K << " x " << n_K << ", V " << m_V << " x " << n_V << ")\n";
        if (m_K != n_K || m_M != m_K || n_M != m_K || m_V != m_K || m_V < n_V) {
            std::cerr << "Error: need square K and M of the same order and a tall V with as many rows.\n";
            return 1;
        }
        m = m_V;
        n = n_V;
        K_T = std::make_unique<CSR_T>(csr_cast<T>(K));
        V_T = std::make_unique<CSR_T>(csr_cast<T>(V));
        K_P = std::make_unique<CSR_P>(csr_cast<P>(K));
        V_P = std::make_unique<CSR_P>(csr_cast<P>(V));
        L_inv_T = std::make_unique<CholSolver_T>(M, /*half_solve=*/true);
        L_inv_P = std::make_unique<CholSolver_P>(M, /*half_solve=*/true);
    }
    std::cout << "Factorizing M = L L^T in both precisions ... " << std::flush;
    auto chol_t0 = steady_clock::now();
    L_inv_T->factorize();
    L_inv_P->factorize();
    const long chol_time_us = duration_cast<microseconds>(steady_clock::now() - chol_t0).count();
    std::cout << "done\n";
    rll::SparseLinOp<CSR_T> K_op_T(m, m, *K_T), V_op_T(m, n, *V_T);
    rll::SparseLinOp<CSR_P> K_op_P(m, m, *K_P), V_op_P(m, n, *V_P);
    rll::CompositeOperator KV_T(m, n, K_op_T, V_op_T); KV_T.block_size = s.block_size;
    rll::CompositeOperator J_T(m, n, *L_inv_T, KV_T);  J_T.block_size  = s.block_size;
    rll::CompositeOperator KV_P(m, n, K_op_P, V_op_P); KV_P.block_size = s.block_size;
    rll::CompositeOperator J_P(m, n, *L_inv_P, KV_P);  J_P.block_size  = s.block_size;

    // A_hat = [J; mu I] with mu = mu_factor * u(precond precision); mu_factor = 0 factors J.
    const bool augmented = (s.mu_factor > 0.0);
    const P mu = augmented ? (P)(s.mu_factor * 0.5 * (double)std::numeric_limits<P>::epsilon()) : P(0);
    rll::ScaledIdentityLinOp<P> mu_I(n, mu);
    rll::VStackLinOp<decltype(J_P), rll::ScaledIdentityLinOp<P>> A_hat(J_P, mu_I);
    A_hat.block_size = s.block_size;
    auto with_qr_operand = [&](auto&& body) { if (augmented) body(A_hat); else body(J_P); };

    // ---- Right-hand side: b = J x_true plus relative Gaussian noise ----
    std::vector<T> x_true(n), b(m);
    { std::mt19937 rng(42); std::uniform_real_distribution<double> U(-1.0, 1.0);
      for (auto& v : x_true) v = (T)U(rng); }
    J_T(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        m, 1, n, T(1), x_true.data(), n, T(0), b.data(), m);
    if (s.noise_level > 0) {
        const T b_clean = blas::nrm2(m, b.data(), 1);
        std::vector<T> e(m);
        std::mt19937 rng(13); std::normal_distribution<double> N01(0, 1);
        for (auto& v : e) v = (T)N01(rng);
        const T en = blas::nrm2(m, e.data(), 1);
        const T scale = (en > 0) ? (T)s.noise_level * b_clean / en : T(0);
        for (int64_t i = 0; i < m; ++i) b[i] += scale * e[i];
    }
    const T x_true_norm = blas::nrm2(n, x_true.data(), 1);
    const T b_norm      = blas::nrm2(m, b.data(), 1);
    const T J_2norm     = estimate_op_2norm<T>(J_T, m, n, 10);
    std::cout << "||x_true|| = " << x_true_norm << ", ||b|| = " << b_norm << ", ||J||_2 ~ " << J_2norm << "\n";
    if (augmented)
        std::cout << "QR methods factor [J; mu I], mu = " << (double)mu << " (" << s.mu_factor
                  << " u in " << precond_prec << ")\n";
    else
        std::cout << "QR methods factor J (mu_factor = 0)\n";

    bench::BackwardErrorSetup<T> be = bench::setup_backward_error<T, RNG>(
        J_T, m, n, b_norm, bench::resolve_be_tol<T>(s.be_tol_mult, n), s.block_size);

    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(s.num_runs, main_state);
    for (int64_t r = 1; r < s.num_runs; ++r) run_states[r].key.incr(r);

    // ---- Untimed warmup: one CQRRTO factorization of the QR operand (when a QR method
    // runs) and five LSQR iterations on J, so that thread pools and first-touch pages
    // of both operator chains are not charged to the first measured row ----
    std::cout << "Warmup ... " << std::flush;
    {
        std::vector<P> Rw((s.method_mask & kQRMethodBits) ? n * n : 0);
        std::vector<T> Rw_T(Rw.size()), xw(n);
        int status = 1;
        if (!Rw.empty()) {
            rl::CQRRTO_linops<P, RNG> warm(false, P(0));
            warm.nnz = s.sketch_nnz;
            warm.block_size = s.block_size;
            auto ws = run_states[0];
            with_qr_operand([&](auto& A_q) { status = warm.call(A_q, Rw.data(), n, (P)s.d_factor, ws); });
            std::copy(Rw.begin(), Rw.end(), Rw_T.begin());
        }
        int it = 0;
        const T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
        rl::lsqr<T>(J_T, m, n, (status == 0) ? Rw_T.data() : nullptr, (status == 0) ? n : 0,
                    b.data(), xw.data(), tol, tol, 5, it);
    }
    std::cout << "done\n";

    // ---- Rows ----
    std::vector<Row<T>> rows;
    std::vector<bench::Solution<T>> solutions;
    std::vector<P> R_P(n * n);
    std::vector<T> R_T(n * n), x(n), Jx(m);
    for (const auto& alg : rows_to_run) {
        std::cout << "\n=== " << alg << " ===\n";
        for (int64_t run_idx = 0; run_idx < s.num_runs; ++run_idx) {
            Row<T> r;
            r.alg = alg;
            r.run = run_idx;
            const bool is_bp     = (alg.rfind("Blendenpik", 0) == 0);
            const bool is_unprec = (alg == "unpreconditioned");
            std::fill(R_P.begin(), R_P.end(), P(0));
            std::fill(x.begin(), x.end(), T(0));
            std::cout << "[run " << run_idx << "] build (" << (is_bp ? solve_prec : precond_prec)
                      << ") ... " << std::flush;

            int chol_status = -1;   // cholqr_primitive's code of a failed QR method
            rl::PeakRSSTracker mem;
            mem.start();
            if (is_bp) {
                run_blendenpik_row<T, RNG>(J_T, b.data(), m, n, x.data(), R_T.data(), s,
                                           run_states[run_idx], be, mem, r);
            } else if (!is_unprec) {
                bench::QRRun q;
                with_qr_operand([&](auto& A_q) {
                    q = bench::run_qr_method<P, RNG>(alg, A_q, R_P.data(), n, s.block_size, s.sketch_nnz,
                                                     (P)s.d_factor, run_states[run_idx]);
                });
                r.qr_status      = q.status;
                r.qr_time_us     = q.qr_time_us;
                r.analytical_kb  = q.analytical_kb;
                r.chol_retries   = q.chol_retries;
                r.chol_shift_abs = q.chol_shift_abs;
                r.chol_shift_rel = q.chol_shift_rel;
                r.qr_breakdown   = q.breakdown;
                chol_status      = q.chol_status;
            }
            if (r.qr_status != 0) {
                if (!is_bp) r.peak_rss_kb = mem.stop();
                r.qr_time_us = -1;
                std::cout << "failed (status " << r.qr_status << ", Cholesky status " << chol_status
                          << "), no solve\n";
                rows.push_back(std::move(r));
                continue;
            }
            if (is_bp)
                r.kappa_measured = (T)kappa_from_R_diag(R_T.data(), n);
            else if (!is_unprec) {
                r.kappa_measured = (T)kappa_from_R_diag(R_P.data(), n);
                std::copy(R_P.begin(), R_P.end(), R_T.begin());
            }
            std::cout << "done (" << r.qr_time_us << " us). solve (" << solve_prec << ") ... " << std::flush;

            if (!is_bp) {
                rl::IterRefineLSQ<T> ir = make_ir<T>(s, be);
                int ir_status = ir.call(J_T, is_unprec ? nullptr : R_T.data(), n, b.data(), m, x.data(), n);
                r.peak_rss_kb = mem.stop();
                record_ir(r, ir_status, ir, be.be_tol >= 0);
            }

            // Diagnostics, outside the RSS window.
            if (!is_unprec)
                r.orth_error = bench::compute_orth_error_explicit<T>(J_T, R_T.data(), s.block_size,
                                                                     s.compute_cond ? &r.cond_precond : nullptr);
            J_T(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, 1, n, T(1), x.data(), n, T(0), Jx.data(), m);
            T res_sq = 0, err_sq = 0;
            for (int64_t i = 0; i < m; ++i) res_sq += (Jx[i] - b[i]) * (Jx[i] - b[i]);
            for (int64_t i = 0; i < n; ++i) err_sq += (x[i] - x_true[i]) * (x[i] - x_true[i]);
            const T denom = J_2norm * blas::nrm2(n, x.data(), 1) + b_norm;
            r.ls_residual_norm  = (denom > 0) ? std::sqrt(res_sq) / denom : T(-1);
            r.ls_solution_error = (x_true_norm > 0) ? std::sqrt(err_sq) / x_true_norm : T(-1);
            solutions.push_back({alg, run_idx, x});
            std::cout << "done (" << r.ir_total_us << " us, bwd_err=" << std::scientific << std::setprecision(3)
                      << (double)r.ls_residual_norm << ", fwd_err=" << (double)r.ls_solution_error
                      << std::defaultfloat << ")\n";
            rows.push_back(std::move(r));
        }
    }

    // ---- Output ----
    std::ostringstream h;
    h << "# FEM least-squares benchmark results\n"
      << "# Date: " << stamp << "\n"
      << "# argv=" << argv_line << "\n"
      << "# input=L^{-1} K V (K=" << s.K_file << " M=" << s.M_file << " V=" << s.V_file << ")\n"
      << "# M=" << m << " N=" << n << " nnz=" << nnz_K << "\n"
      << "# noise_level=" << s.noise_level << "\n"
      << "# chol_time_us=" << chol_time_us << "\n"
      << "# d_factor=" << s.d_factor << " sketch_nnz=" << s.sketch_nnz << " block_size=" << s.block_size << "\n"
      << "# method_mask=" << s.method_mask << "\n"
      << "# num_runs=" << s.num_runs << "\n"
      << "# kappa_target=1 (V used as given) mu=" << (double)mu << "\n"
      << "# precond_prec=" << precond_prec << " solve_prec=" << solve_prec << "\n"
      << (augmented ? "# R^T R = J^T J + mu^2 I from the QR of [J; mu I] in precond_prec;"
                    : "# R^T R = J^T J from the QR of J in precond_prec;")
      << " refinement in solve_prec. Blendenpik rows factor S J in solve_prec.\n"
      << "# tolerances: refinement rows stop at ||b - J x|| / ||b|| <= " << (double)outer_tol_eff<T>(s)
      << "; published Blendenpik rows run LSQR to " << std::pow(std::numeric_limits<T>::epsilon(), (T)0.85) << "\n"
      << "# OpenMP threads: " << rl::util::get_omp_threads() << "\n";
    bench::write_env_lines(h);
    h << "# ir settings: max_inner=" << s.max_inner << " inner_tol=" << (double)inner_tol_eff<T>(s)
      << " round_drop=" << s.round_drop << " n_steps=" << s.n_steps
      << " outer_tol=" << (double)outer_tol_eff<T>(s) << " outer_stag_window=" << s.outer_stag_window
      << " be_tol_mult=" << s.be_tol_mult << " be_tol=" << (double)be.be_tol
      << " kw_sketch_nnz=" << bench::kw_sketch_nnz(m, n) << "\n";

    write_results<T>(results_file, h.str(), {m, n, (double)mu, precond_prec, solve_prec}, rows);
    write_breakdown<T>(breakdown_file, rows);
    write_rounds_csv<T>(rounds_file, rows);
    bench::write_backward_error_sidecar<T, RNG>(kw_file, be, J_T, m, n, b.data(), b_norm, s.block_size, solutions);
    for (const auto& f : {results_file, breakdown_file, rounds_file, kw_file}) {
        bench::check_csv_arity(f);
        std::cout << "wrote " << f << "\n";
    }
    return 0;
}

// ============================================================================
// Command line
// ============================================================================

static void print_usage(const char* exe) {
    std::cerr <<
      "Usage: " << exe << " --precision=double|single --out=DIR --K=K.mtx --M=M.mtx --V=V.mtx [options]\n"
      "\n"
      "  --precision=P      solve precision: double or single (float)\n"
      "  --out=DIR          output directory\n"
      "  --K= --M= --V=     FEM stiffness, mass and prolongation matrices\n"
      "\n"
      "Options (default in brackets):\n"
      "  --runs=N           [1]       repeats per method\n"
      "  --d-factor=F       [2.0]     sketch rows per column, >= 1\n"
      "  --sketch-nnz=N     [4]       nonzeros per sketch column\n"
      "  --block-size=N     [256]     Gram column-block width; 0 = unblocked\n"
      "  --compute-cond     [off]     also report cond(J R^{-1}) (n <= 16384)\n"
      "  --mask=N           [31]      method bitmask (see the file header)\n"
      "  --noise=F          [0.05]    relative Gaussian noise added to b = J x_true\n"
      "  --mu-factor=F      [10]      mu = F u(preconditioner precision); 0 = factor J itself\n"
      "  --precond-prec=P   [single]  preconditioner precision: double or single (float)\n"
      "  --max-inner=N      [200]     inner CG iterations per round, >= 1\n"
      "  --inner-tol=F      [-1]      absolute inner floor; < 0 = eps^0.85, 0 = off\n"
      "  --round-drop=F     [1e-4]    per-round residual drop, in (0,1)\n"
      "  --steps=N          [50]      round cap, >= 1\n"
      "  --outer-tol=F      [-1]      stop at ||b - J x|| / ||b|| <= F; < 0 = 10 eps\n"
      "  --outer-stag=N     [2]       rounds without progress that end a run at the LS floor; 0 = off\n"
      "  --be-tol-mult=F    [0]       also stop once the sketched backward error is <= F sqrt(n) u\n"
      "                               (double solve precision only); 0 = off\n"
      "\n"
      "QR driver options come from the environment (cqrrto_bench_common.hh).\n";
}

// Parses and validates the arguments; throws std::runtime_error on an invalid one.
static Settings parse_settings(const bench::BenchArgs& a, bool solve_in_double) {
    a.reject_unknown({"precision", "out", "runs", "K", "M", "V", "d-factor", "sketch-nnz",
                      "block-size", "compute-cond", "mask", "noise", "mu-factor", "precond-prec",
                      "max-inner", "inner-tol", "round-drop", "steps", "outer-tol", "outer-stag",
                      "be-tol-mult", "help"});
    Settings s;
    s.out_dir           = a.require_str("out");
    s.K_file            = a.require_str("K");
    s.M_file            = a.require_str("M");
    s.V_file            = a.require_str("V");
    s.num_runs          = a.i64("runs", s.num_runs);
    s.d_factor          = a.dbl("d-factor", s.d_factor);
    s.sketch_nnz        = a.i64("sketch-nnz", s.sketch_nnz);
    s.block_size        = a.i64("block-size", s.block_size);
    s.compute_cond      = a.boolean("compute-cond", s.compute_cond);
    s.method_mask       = a.i64("mask", s.method_mask);
    s.noise_given       = a.has("noise");
    s.noise_level       = a.dbl("noise", s.noise_level);
    s.mu_factor         = a.dbl("mu-factor", s.mu_factor);
    s.precond_prec      = a.str("precond-prec", s.precond_prec);
    s.max_inner         = a.i32("max-inner", s.max_inner);
    s.inner_tol         = a.dbl("inner-tol", s.inner_tol);
    s.round_drop        = a.dbl("round-drop", s.round_drop);
    s.n_steps           = a.i32("steps", s.n_steps);
    s.outer_tol         = a.dbl("outer-tol", s.outer_tol);
    s.outer_stag_window = a.i32("outer-stag", s.outer_stag_window);
    s.be_tol_mult       = a.dbl("be-tol-mult", s.be_tol_mult);

    auto require = [](bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); };
    require(s.num_runs >= 1, "--runs must be >= 1");
    require(s.d_factor >= 1, "--d-factor must be >= 1");
    require(s.sketch_nnz >= 1, "--sketch-nnz must be >= 1");
    require(s.block_size >= 0, "--block-size must be >= 0");
    require(s.noise_level >= 0, "--noise must be >= 0");
    require(s.mu_factor >= 0, "--mu-factor must be >= 0");
    require(s.precond_prec == "double" || s.precond_prec == "single" || s.precond_prec == "float",
            "--precond-prec must be double or single");
    require(s.max_inner >= 1 && s.n_steps >= 1 && s.max_inner <= INT_MAX / s.n_steps,
            "--max-inner and --steps must be >= 1, with a product that fits in int");
    require(s.round_drop > 0 && s.round_drop < 1, "--round-drop must lie in (0,1)");
    require(s.inner_tol < 1, "--inner-tol must be < 1");
    require(solve_in_double || s.be_tol_mult <= 0,
            "--be-tol-mult needs the double solve precision: a single-precision SVD of the "
            "reference sketch cannot resolve the small singular directions the estimate weights");
    bench::decode_method_mask(s.method_mask, kMethods);   // throws on an invalid mask
    return s;
}

// The checks that depend on the precisions the values are converted to.
template <typename T, typename P>
static void check_in_precision(const Settings& s) {
    auto require = [](bool ok, const std::string& msg) { if (!ok) throw std::runtime_error(msg); };
    require((T)s.round_drop > 0 && (T)s.round_drop < 1, "--round-drop is not in (0,1) in the solve precision");
    require(std::isfinite((T)s.outer_tol) && std::isfinite((T)s.inner_tol) && std::isfinite((T)s.noise_level),
            "--outer-tol, --inner-tol and --noise must be finite in the solve precision");
    require(std::isfinite((T)s.d_factor) && std::isfinite((P)s.d_factor),
            "--d-factor must be finite in both precisions");
    require(std::isfinite((P)(s.mu_factor * 0.5 * (double)std::numeric_limits<P>::epsilon())),
            "--mu-factor gives a mu that is not finite in the preconditioner precision");
}

template <typename T_solve>
static void check_in_precisions(const Settings& s) {
    if (s.precond_prec == "double") check_in_precision<T_solve, double>(s);
    else                            check_in_precision<T_solve, float>(s);
}

template <typename T_solve>
static int dispatch(const Settings& s, const std::string& argv_line) {
    return (s.precond_prec == "double") ? run<T_solve, double>(s, argv_line) : run<T_solve, float>(s, argv_line);
}

int main(int argc, char* argv[]) {
    Settings s;
    std::string precision;
    try {
        bench::BenchArgs a(argc, argv);
        if (argc < 2 || a.has("help")) { print_usage(argv[0]); return argc < 2 ? 1 : 0; }
        precision = a.require_str("precision");
        if (precision != "double" && precision != "single" && precision != "float")
            throw std::runtime_error("--precision must be double or single");
        s = parse_settings(a, precision == "double");
        if (precision == "double") check_in_precisions<double>(s);
        else                       check_in_precisions<float>(s);
        bench::qr_driver_options();   // reject invalid environment values before any work
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
    if (!s.noise_given)
        std::cerr << "Note: --noise defaulted to " << s.noise_level << "; pass --noise=0 for a "
                     "consistent right-hand side.\n";

    std::cout << "=== FEM least-squares benchmark ===\n"
              << "  K: " << s.K_file << "\n  M: " << s.M_file << "\n  V: " << s.V_file << "\n"
              << "  solve precision: " << precision << ", preconditioner precision: " << s.precond_prec << "\n"
              << "  rows:";
    for (const auto& r : bench::decode_method_mask(s.method_mask, kMethods)) std::cout << " " << r;
    std::cout << "\n  runs: " << s.num_runs << ", OpenMP threads: " << rl::util::get_omp_threads() << "\n";

    const std::string argv_line = bench::quote_join_argv(argc, argv);
    try {
        return (precision == "double") ? dispatch<double>(s, argv_line) : dispatch<float>(s, argv_line);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
