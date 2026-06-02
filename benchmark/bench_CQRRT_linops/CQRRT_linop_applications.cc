// Unified Q-less QR benchmark — IR-LSQ application, plus rspec (Algorithm 4).
//
// Pipeline:
//   1. Load matrices (FEM mode: K, M, V .mtx files; sparse mode: a single A.mtx).
//   2. (FEM only) Cholesky-factorize M = L L^T via CholSolverLinOp(half_solve=true).
//   3. (FEM only) Build J = L^{-1} K V as a doubly-nested CompositeOperator
//      J = CompositeOperator(L_inv_op, CompositeOperator(K_op, V_op)).
//   4. Run Q-less QR via one of 5 variants (CQRRT_linop, CholQR, sCholQR3,
//      sCholQR3_basic, CQRRT_linop_bqrrp), selected by method_mask.
//   5. Post-processing dictated by <mode>:
//        irlsq — sketch-and-solve x_0 (Epperly–Meier–Nakatsukasa 2025 Alg. 1 line 3) + IterRefineLSQ
//        rspec — reduced spectral approximation (Algorithm 4): Rayleigh–Ritz on
//                range(C^j V_FEM), C = L^T (K - ω M)^{-1} L. FEM-only.
//
// Usage:
//   ./CQRRT_linop_applications <prec> <outdir> <runs> <mode>
//          sparse <A.mtx> <d_factor> [nnz] [b] [compute_cond] [method_mask] [noise_level]
//   ./CQRRT_linop_applications <prec> <outdir> <runs> <mode>
//          <K.mtx> <M.mtx> <V.mtx> <d_factor> [nnz] [b] [compute_cond] [method_mask] [noise_level] [omega] [power_j]
//
// mode        = "irlsq" | "rspec"
// method_mask = bitmask of Q-less QR variants (default 0b1001111 = 79)
//                 bit 0 ( 1): CQRRT_linop (TRSM_IDENTITY)
//                 bit 1 ( 2): CholQR
//                 bit 2 ( 4): sCholQR3
//                 bit 3 ( 8): sCholQR3_basic
//                 bit 6 (64): CQRRT_linop_bqrrp (BQRRP)

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
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
#include "../../extras/linops/ext_sparselu_linop.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrt_bench_common.hh"

// Linops algorithms
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

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

    // Q-factor orthogonality: ||Q^T Q - I||_F / sqrt(n), computed for all methods.
    T orth_error;

    // IR-LSQ-mode fields
    long ir_total_us;
    int  ir_outer_iters;
    int  ir_inner_iters_total;
    T    ls_residual_norm;
    T    ls_solution_error;   // -1 sentinel when undefined (FEM irlsq)

    // QR timing breakdown (from algo.times[])
    std::vector<long> qr_breakdown;
    std::vector<long> ir_breakdown;

    long peak_rss_kb;
    long analytical_kb;
};

// Compute A^T A via blocked linop calls. Peak memory: O(m*b + n*b).
//   AtA      n x n output (caller-allocated)
//   E_block  n x b identity-column scratch
//   A_block  m x b linop NoTrans output
//   AtA_block n x b linop Trans output, copied into AtA[:, j0:j0+bk] each iter
template <typename T, typename GLO>
static void compute_AtA_blocked(GLO& A_op, int64_t m, int64_t n, T* AtA, int64_t b) {
    std::fill(AtA, AtA + n * n, (T)0.0);
    T* E_block   = new T[n * b]();
    T* A_block   = new T[m * b];
    T* AtA_block = new T[n * b];

    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);

        std::fill(E_block, E_block + n * b, (T)0.0);
        for (int64_t j = 0; j < bk; ++j)
            E_block[(j0 + j) + j * n] = (T)1.0;

        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, bk, n, (T)1.0, E_block, n, (T)0.0, A_block, m);

        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             n, bk, m, (T)1.0, A_block, m, (T)0.0, AtA_block, n);

        lapack::lacpy(lapack::MatrixType::General, n, bk, AtA_block, n, AtA + j0 * n, n);
    }

    delete[] E_block;
    delete[] A_block;
    delete[] AtA_block;
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

// Blocked orth_err: ||R^{-T} A^T A R^{-1} - I||_F / sqrt(n).
// O(n^2 + m*b) memory (no Q materialization). Mathematically equivalent to
// ||Q^T Q - I||_F / sqrt(n) with Q = A * R^{-1}.
template <typename T, typename GLO>
static T compute_orth_error_memlite(GLO& A_op, const T* R, int64_t m, int64_t n, int64_t block_size) {
    int64_t b = (block_size > 0) ? block_size : 256;
    T* X = new T[n * n]();
    compute_AtA_blocked(A_op, m, n, X, b);
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
               blas::Op::Trans, blas::Diag::NonUnit, n, n, (T)1.0, R, n, X, n);
    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
               blas::Op::NoTrans, blas::Diag::NonUnit, n, n, (T)1.0, R, n, X, n);
    for (int64_t i = 0; i < n; ++i) X[i + i * n] -= (T)1.0;
    T s = 0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int64_t i = 0; i < n * n; ++i) s += X[i] * X[i];
    delete[] X;
    return std::sqrt(s) / std::sqrt((T)n);
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
           "ls_residual_norm,ls_solution_error\n";
    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << "," << r.m << "," << r.n << ","
            << r.qr_status << "," << r.qr_time_us << "," << r.peak_rss_kb << "," << r.analytical_kb << ","
            << std::scientific << std::setprecision(6) << r.orth_error << ","
            << r.ir_total_us << "," << r.ir_outer_iters << "," << r.ir_inner_iters_total << ","
            << std::scientific << std::setprecision(6) << r.ls_residual_norm << ","
            << std::scientific << std::setprecision(6) << r.ls_solution_error
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
        << "#   (sketch-and-solve x_0 time is folded into the difference between ir_total_us\n"
        << "#    in the results CSV and outer_total here)\n"
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
    std::vector<T> top_eigvals;
    std::vector<T> top_residuals;
};

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
           "factor_time_us,rspec_total_us";
    for (int i = 0; i < top_k; ++i) out << ",eig_" << i;
    for (int i = 0; i < top_k; ++i) out << ",resid_" << i;
    out << "\n";

    for (const auto& r : results) {
        out << r.alg_name << "," << r.run_idx << "," << r.m << "," << r.n << ","
            << omega << "," << power_j << ","
            << r.qr_status << "," << r.qr_time_us << ","
            << r.peak_rss_kb << "," << r.analytical_kb << ","
            << r.factor_time_us << "," << r.rspec_total_us;
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
    std::printf("    IR-LSQ (with x_0): total=%ld us, outer=%d, inner_total=%d\n",
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
    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 64)  selected_algs.push_back("CQRRT_linop_bqrrp");

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

    // Warmup (CQRRT_linop)
    std::cout << "Running warmup... " << std::flush;
    {
        auto warm_state = run_states[0];
        T* R_warm = new T[n * n]();
        RandLAPACK::CQRRT_linops<T, RNG> warm_algo(false, tol, false);
        warm_algo.nnz = sketch_nnz;
        warm_algo.block_size = block_size;
        warm_algo.call(A_op, R_warm, n, d_factor, warm_state);
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
    const int64_t d_init = (int64_t)(d_factor * (T)n);
    T* R    = new T[n * n]();    // QR output; zero-filled per iter to match prior behavior
    T* SA   = new T[d_init * n]; // S2 * A (sketch-and-solve LHS); overwritten beta=0
    T* Sb   = new T[d_init];     // S2 * b; overwritten beta=0
    T* x_ls = new T[n];          // initial guess + refined solution
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

            // ---- QR dispatch (5-way, lifted verbatim from CQRRT_linop_irlsq.cc) ----
            std::cout << "[Run " << run_idx << ", " << alg_name << "] QR ... " << std::flush;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            if (alg_name == "sCholQR3") {
                RandLAPACK::sCholQR3_linops<T> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[17];
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
                }
            } else if (alg_name == "sCholQR3_basic") {
                RandLAPACK::sCholQR3_linops_basic<T> qr_algo(/*time_subroutines=*/true, tol);
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[14];
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m, n);
                }
            } else if (alg_name == "CholQR") {
                RandLAPACK::CholQR_linops<T> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(A_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[5];
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 6);
                    res.qr_breakdown.resize(11, 0);
                    res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
                }
            } else {
                RandLAPACK::CQRRT_linops<T, RNG> qr_algo(/*time_subroutines=*/true, tol);
                qr_algo.nnz = sketch_nnz;
                qr_algo.block_size = block_size;
                if (alg_name == "CQRRT_linop")
                    qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
                else /* CQRRT_linop_bqrrp */
                    qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::BQRRP;
                res.qr_status = qr_algo.call(A_op, R, n, d_factor, state);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[10];
                    res.qr_breakdown.assign(qr_algo.times.begin(), qr_algo.times.begin() + 11);
                    res.analytical_kb = (alg_name == "CQRRT_linop_bqrrp")
                        ? RandLAPACK::cqrrt_linops_bqrrp_analytical_kb<T>(m, n, d_factor, block_size)
                        : RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
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
            res.orth_error = compute_orth_error_memlite(A_op, R, m, n, block_size);

            // ---- IR-LSQ post-processing ----
            {
                std::cout << ". IR-LSQ ... " << std::flush;
                const std::vector<T>& b = *b_ptr;
                auto ls_t0 = steady_clock::now();

                // Sketch-and-solve initial guess x_0 (Alg. 1 line 3 of Epperly–Meier–Nakatsukasa 2025);
                // fresh sparse sketch S_2 independent of CQRRT's S_1.
                RandBLAS::SparseDist DS_init(d_init, m, sketch_nnz);
                auto x0_state = state;
                x0_state.key.incr(0xA1B2C3D4u);
                RandBLAS::SparseSkOp<T, RNG> S2(DS_init, x0_state);
                RandBLAS::fill_sparse(S2);

                A_op(blas::Side::Right, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                     d_init, n, m, (T)1.0, S2, (T)0.0, SA, d_init);

                RandBLAS::sketch_general(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                         d_init, (int64_t)1, m, (T)1.0,
                                         S2, b.data(), m, (T)0.0, Sb, d_init);

                blas::gemv(blas::Layout::ColMajor, blas::Op::Trans, d_init, n,
                           (T)1.0, SA, d_init, Sb, 1,
                           (T)0.0, x_ls, 1);
                blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                           blas::Op::Trans, blas::Diag::NonUnit, n, 1,
                           (T)1.0, R, n, x_ls, n);
                blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                           blas::Op::NoTrans, blas::Diag::NonUnit, n, 1,
                           (T)1.0, R, n, x_ls, n);

                RandLAPACK::IterRefineLSQ<T> ir(/*tol=*/tol,
                                                /*max_inner=*/200,
                                                /*n_steps=*/2,
                                                /*timing=*/true,
                                                /*verbose=*/false);
                int ir_status = ir.call(A_op, R, n, b.data(), m, x_ls, n);
                auto ls_t1 = steady_clock::now();
                if (ir_status != 0) {
                    std::cerr << "Warning: IterRefineLSQ status " << ir_status << " (CG breakdown)\n";
                }

                res.ir_total_us = duration_cast<microseconds>(ls_t1 - ls_t0).count();
                res.ir_outer_iters = ir.outer_iters_done;
                res.ir_inner_iters_total = 0;
                for (int v : ir.inner_iters_per_step) res.ir_inner_iters_total += v;
                if (!ir.times.empty()) res.ir_breakdown = ir.times;

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

    delete[] R; delete[] SA; delete[] Sb; delete[] x_ls; delete[] Ax;
    return 0;
}

// ============================================================================
// RSPEC mode runner
// ============================================================================

template <typename T, typename RNG, typename KOpType, typename MOpType, typename VAppOpType, typename CompCOp>
static int run_rspec_benchmark(
    VAppOpType& V_app_op,         // m_K x n_V composite: C^j * V_FEM
    CompCOp&    C_op,             // m_K x m_K composite: L^T X^{-1} L
    KOpType&    K_op,             // m_K x m_K sparse: K (for residual norm)
    MOpType&    M_op,             // m_K x m_K sparse: M (for residual norm)
    int64_t m_K, int64_t n_V,
    const std::string& output_dir, int64_t num_runs,
    T d_factor, int64_t sketch_nnz, int64_t block_size,
    int64_t method_mask,
    long factor_time_us,
    const std::string& K_file, const std::string& M_file, const std::string& V_file,
    double omega, int64_t power_j)
{
    // Build the list of selected algorithm names from the bitmask (same as run_benchmark_inner).
    std::vector<std::string> selected_algs;
    if (method_mask & 1)   selected_algs.push_back("CQRRT_linop");
    if (method_mask & 2)   selected_algs.push_back("CholQR");
    if (method_mask & 4)   selected_algs.push_back("sCholQR3");
    if (method_mask & 8)   selected_algs.push_back("sCholQR3_basic");
    if (method_mask & 64)  selected_algs.push_back("CQRRT_linop_bqrrp");

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

    // Warmup (small probe of V_app_op so SparseLU is warm).
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

            auto rspec_t0 = steady_clock::now();

            std::fill(R, R + n * n, (T)0);
            auto state = run_states[run_idx];

            std::cout << "[Run " << run_idx << ", " << alg_name << "] PCholQR ... " << std::flush;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            if (alg_name == "sCholQR3") {
                RandLAPACK::sCholQR3_linops<T> qr_algo(true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[17];
                    res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
                }
            } else if (alg_name == "sCholQR3_basic") {
                RandLAPACK::sCholQR3_linops_basic<T> qr_algo(true, tol);
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[14];
                    res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m, n);
                }
            } else if (alg_name == "CholQR") {
                RandLAPACK::CholQR_linops<T> qr_algo(true, tol);
                qr_algo.block_size = block_size;
                res.qr_status = qr_algo.call(V_app_op, R, n);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[5];
                    res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
                }
            } else {
                RandLAPACK::CQRRT_linops<T, RNG> qr_algo(true, tol);
                qr_algo.nnz = sketch_nnz;
                qr_algo.block_size = block_size;
                if (alg_name == "CQRRT_linop")
                    qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY;
                else
                    qr_algo.precond_method = RandLAPACK::CQRRTLinopPrecond::BQRRP;
                res.qr_status = qr_algo.call(V_app_op, R, n, d_factor, state);
                res.peak_rss_kb = mem.stop();
                if (res.qr_status == 0) {
                    res.qr_time_us = qr_algo.times[10];
                    res.analytical_kb = (alg_name == "CQRRT_linop_bqrrp")
                        ? RandLAPACK::cqrrt_linops_bqrrp_analytical_kb<T>(m, n, d_factor, block_size)
                        : RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
                }
            }

            if (res.qr_status != 0) {
                std::cerr << "\n  [" << alg_name << "] Run " << run_idx
                          << ": QR returned status " << res.qr_status
                          << ". Skipping eigen post-processing.\n";
                res.qr_time_us = -1;
                res.top_eigvals.assign(top_k, std::numeric_limits<T>::quiet_NaN());
                res.top_residuals.assign(top_k, std::numeric_limits<T>::quiet_NaN());
                auto rspec_t1 = steady_clock::now();
                res.rspec_total_us = duration_cast<microseconds>(rspec_t1 - rspec_t0).count();
                all_results.push_back(res);
                continue;
            }
            std::cout << "done (" << res.qr_time_us << " us)\n";

            // ----------------------------------------------------------------
            // Rayleigh-Ritz: T = R^{-T} V_app^T C V_app R^{-1}
            // Materialize V_app^T C V_app column-block by column-block on identity.
            // ----------------------------------------------------------------
            std::cout << "    Building Rayleigh-Ritz matrix T (n=" << n << ") ... " << std::flush;
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

            // Symmetrize: T := (T + T^T)/2
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = j + 1; i < n; ++i) {
                    T avg = (T_mat[i + j * n] + T_mat[j + i * n]) * (T)0.5;
                    T_mat[i + j * n] = avg;
                    T_mat[j + i * n] = avg;
                }
            }
            std::cout << "done\n";

            // Eigendecomposition: T = U diag(lambda) U^T, U overwrites T_mat (columns = eigvecs).
            std::cout << "    syevd ... " << std::flush;
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

            // ----------------------------------------------------------------
            // Ritz residual norms for the top-k pairs:
            //   v = V_FEM * (R^{-1} * u_k)      -- NOTE: We use V_app * R^{-1} u_k since
            //   R is the qless-QR factor of V_app (= C^j V_FEM); thus Q = V_app * R^{-1}.
            //   residual = ||C v_q - lambda_k * M v_q|| not appropriate — see note below.
            //
            // Per spec: residual = ||C v - lambda M v|| / (||K v|| + |lambda| ||M v||)
            //   where v = V_FEM * R^{-1} * u_k. Since u_k is an eigvec of the small RR matrix
            //   T = Q^T C Q with Q = V_app * R^{-1} (i.e. Q^T Q = I via PCholQR), the Ritz vector
            //   in the C-operator coordinates is Q u_k = V_app R^{-1} u_k. The spec writes
            //   v = V_FEM * R^{-1} * u_k, which corresponds to using V_FEM (un-powered) for the
            //   physical eigenvector recovery. We follow the spec.
            // ----------------------------------------------------------------
            // Build V_FEM * R^{-1} * U_topk  (m x top_k)
            // V_FEM is wrapped inside V_app_op as the right_op... but we need direct access.
            // Instead, take the Ritz vector in the operator domain: q = V_app * R^{-1} u.
            // For residuals we follow the spec literally and substitute v -> V_app R^{-1} u.
            // The denominator uses ||K v|| + |lambda| ||M v||; we use the same v.
            std::cout << "    Ritz residuals ... " << std::flush;
            T* u_blk = new T[(size_t)n * (size_t)top_k]();  // R^{-1} * U_topk (n x top_k)
            for (int i = 0; i < top_k; ++i) {
                for (int64_t r = 0; r < n; ++r) {
                    u_blk[r + i * n] = T_mat[r + perm[i] * n];
                }
            }
            // R^{-1} * u_blk
            blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper,
                       blas::Op::NoTrans, blas::Diag::NonUnit,
                       n, top_k, (T)1.0, R, n, u_blk, n);

            // v_blk = V_app_op * (R^{-1} u_blk)   (m x top_k)
            T* v_blk = new T[(size_t)m * (size_t)top_k]();
            V_app_op(blas::Side::Left, blas::Layout::ColMajor,
                     blas::Op::NoTrans, blas::Op::NoTrans,
                     m, top_k, n, (T)1.0, u_blk, n, (T)0.0, v_blk, m);

            T* Kv_blk = new T[(size_t)m * (size_t)top_k]();
            T* Mv_blk = new T[(size_t)m * (size_t)top_k]();
            T* Cv_blk = new T[(size_t)m * (size_t)top_k]();
            K_op(blas::Side::Left, blas::Layout::ColMajor,
                 blas::Op::NoTrans, blas::Op::NoTrans,
                 m, top_k, m, (T)1.0, v_blk, m, (T)0.0, Kv_blk, m);
            M_op(blas::Side::Left, blas::Layout::ColMajor,
                 blas::Op::NoTrans, blas::Op::NoTrans,
                 m, top_k, m, (T)1.0, v_blk, m, (T)0.0, Mv_blk, m);
            C_op(blas::Side::Left, blas::Layout::ColMajor,
                 blas::Op::NoTrans, blas::Op::NoTrans,
                 m, top_k, m, (T)1.0, v_blk, m, (T)0.0, Cv_blk, m);

            res.top_residuals.resize(top_k);
            for (int i = 0; i < top_k; ++i) {
                T lam = res.top_eigvals[i];
                T num_sq = 0;
                T Kv_sq  = 0;
                T Mv_sq  = 0;
                for (int64_t r = 0; r < m; ++r) {
                    T d = Cv_blk[r + i * m] - lam * Mv_blk[r + i * m];
                    num_sq += d * d;
                    Kv_sq  += Kv_blk[r + i * m] * Kv_blk[r + i * m];
                    Mv_sq  += Mv_blk[r + i * m] * Mv_blk[r + i * m];
                }
                T num   = std::sqrt(num_sq);
                T denom = std::sqrt(Kv_sq) + std::abs(lam) * std::sqrt(Mv_sq);
                res.top_residuals[i] = (denom > 0) ? num / denom : std::numeric_limits<T>::quiet_NaN();
            }

            delete[] u_blk;
            delete[] v_blk;
            delete[] Kv_blk;
            delete[] Mv_blk;
            delete[] Cv_blk;
            delete[] eigvals;
            delete[] perm;
            delete[] T_mat;
            std::cout << "done\n";

            auto rspec_t1 = steady_clock::now();
            res.rspec_total_us = duration_cast<microseconds>(rspec_t1 - rspec_t0).count();

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
    delete[] R;
    return 0;
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
                  << "  mode  = irlsq | rspec   (rspec is FEM-only)\n";
        return 1;
    }

    std::string output_dir = argv[2];
    int64_t num_runs       = std::stol(argv[3]);
    std::string mode       = argv[4];
    if (mode != "irlsq" && mode != "rspec") {
        std::cerr << "Error: <mode> must be one of {irlsq, rspec}; got '" << mode << "'\n";
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
    int64_t method_mask = opt_long(4, 79);
    T noise_level       = (T)opt_double(5, 0.05);
    double omega        = opt_double(6, 0.0);
    int64_t power_j     = opt_long(7, 1);

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
              << " linop_bqrrp=" << ((method_mask>>6)&1) << ")\n"
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
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> M_op(m_K, m_K, M_csr);

        // 1. X = K - omega * M (CSR, shares sparsity with K and M).
        std::cout << "Forming X = K - omega*M ... " << std::flush;
        auto X_csr = RandLAPACK_extras::sparse_axpby_shared_pattern<T, int64_t>(
            (T)1.0, K_csr, -(T)omega, M_csr);
        std::cout << "done (nnz=" << X_csr.nnz << ")\n";

        // 2. Convert X to Eigen::SparseMatrix<T> (ColMajor) via triplets.
        std::cout << "Converting X to Eigen sparse... " << std::flush;
        Eigen::SparseMatrix<T> X_eigen((Eigen::Index)m_K, (Eigen::Index)m_K);
        {
            std::vector<Eigen::Triplet<T>> triplets;
            triplets.reserve((size_t)X_csr.nnz);
            for (int64_t i = 0; i < m_K; ++i) {
                for (int64_t p = X_csr.rowptr[i]; p < X_csr.rowptr[i+1]; ++p) {
                    triplets.emplace_back((int)i, (int)X_csr.colidxs[p], X_csr.vals[p]);
                }
            }
            X_eigen.setFromTriplets(triplets.begin(), triplets.end());
        }
        std::cout << "done\n";

        // 3. Factor X via SparseLU (handles indefinite case).
        std::cout << "Factorizing X = LU (SparseLU) ... " << std::flush;
        RandLAPACK_extras::linops::SparseLUSolverLinOp<T> X_inv_op(std::move(X_eigen));
        auto x_fact_start = steady_clock::now();
        try {
            X_inv_op.factorize();
        } catch (RandBLAS::Error const& e) {
            auto x_fact_stop = steady_clock::now();
            long x_fact_us = duration_cast<microseconds>(x_fact_stop - x_fact_start).count();
            std::cerr << "\nSparseLU factorization failed (omega close to an eigenvalue): "
                      << e.what() << "\n";

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
            V_app_op, C_op, K_op, M_op,
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
                  << "  mode  = irlsq | rspec   (rspec is FEM-only)\n";
        return 1;
    }

    std::string precision = argv[1];
    if (precision == "double") {
        return run_benchmark<double>(argc, argv);
    } else if (precision == "float") {
        return run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Unknown precision: " << precision << " (use 'double' or 'float')\n";
        return 1;
    }
}
