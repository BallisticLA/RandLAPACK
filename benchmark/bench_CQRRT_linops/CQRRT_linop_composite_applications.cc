// Generalized SVD / Generalized LS benchmark
//
// Pipeline:
//   1. Load K.mtx (m x m SPD) and V.mtx (m x n sparse)
//   2. Cholesky factorize K = LL^T, create L^{-1} operator (half_solve=true)
//   3. Create composite operator: CompositeOperator(L_inv_op, V_op) = L^{-1}V
//   4. Run Q-less QR on L^{-1}V via CQRRT_linops, CholQR_linops, sCholQR3_linops
//   5. Application (a): Generalized LS — solve min_x ||Vx - b||_{K^{-1}}
//   6. Application (b): Generalized singular values — SVD of R
//   7. Application (c): Generalized singular vectors — full SVD of R
//
// Usage:
//   ./GSVD_benchmark <precision> <output_dir> <num_runs> <K_file> <V_file>
//                    <d_factor> [sketch_nnz] [block_size] [skip_apps] [compute_cond]

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
#include <omp.h>
#include <algorithm>
#include <numeric>

// Extras utilities (Eigen-dependent)
#include "../../extras/misc/ext_util.hh"
#include "../../extras/linops/ext_cholsolver_linop.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"

// Linops algorithms (now in main RandLAPACK)
#include "rl_cqrrt_linops.hh"
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "RandLAPACK/testing/rl_memory_tracker.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// ============================================================================
// Result struct
// ============================================================================

template <typename T>
struct gsvd_result {
    int64_t m, n;
    int64_t run_idx;
    std::string alg_name;

    // Cholesky factorization time (shared, measured once)
    long chol_time_us;

    // Q-less QR time
    long qr_time_us;

    // Orthogonality of Q = (L^{-1}V) R^{-1}
    T orth_error;
    bool is_orthonormal;
    int64_t max_orth_cols;

    // Application (a): Generalized LS
    long app_a_time_us;       // Post-processing time only
    T ls_rel_error;           // ||x - x_true|| / ||x_true||

    // Application (b): Generalized singular values
    long app_b_time_us;       // SVD of R time

    // Application (c): Generalized singular vectors
    long app_c_time_us;       // Full SVD of R + V_R orthogonality check
    T right_svec_orth_error;  // ||V_R^T V_R - I||_F / sqrt(n)

    // Totals (QR + application post-processing)
    long total_a_time_us;
    long total_b_time_us;
    long total_c_time_us;

    // QR timing breakdown (from algo.times[])
    std::vector<long> qr_breakdown;

    // Memory tracking
    long peak_rss_kb;       // Peak RSS increase during QR call (KB)
    long analytical_kb;     // Analytical peak working memory (KB)
};

// Compute Q = A * R^{-1} uniformly for all algorithms
template <typename T, typename GLO>
static void compute_Q_from_R(
    GLO& A_op, T* R, int64_t ldr,
    T* Q_out, int64_t m, int64_t n) {
    T* Eye = new T[n * n]();
    RandLAPACK::util::eye(n, n, Eye);
    A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
         m, n, n, (T)1.0, Eye, n, (T)0.0, Q_out, m);
    delete[] Eye;
    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans,
               blas::Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_out, m);
}

// ============================================================================
// Application (a): Generalized Least Squares
// min_x ||Vx - b||_{K^{-1}} via R from QR of L^{-1}V
//
// Solution: x = R^{-1} R^{-T} V^T K^{-1} b
// Steps: c = K^{-1}b, d = V^T c, solve R^T y = d, solve Rx = y
// ============================================================================

template <typename T, typename VLinOp>
static void app_generalized_ls(
    RandLAPACK_extras::linops::CholSolverLinOp<T>& K_inv_op,
    VLinOp& V_op,
    const T* R, int64_t ldr, int64_t n,
    const T* b, int64_t m,
    T* x,
    long& app_time_us)
{
    auto start = steady_clock::now();

    // Step 1: c = K^{-1} b  (m x 1)
    std::vector<T> c(m, 0.0);
    K_inv_op(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, 1, m, (T)1.0, b, m, (T)0.0, c.data(), m);

    // Step 2: d = V^T c  (n x 1)
    std::vector<T> d(n, 0.0);
    V_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
         n, 1, m, (T)1.0, c.data(), m, (T)0.0, d.data(), n);

    // Step 3: solve R^T y = d  (n x 1)
    std::copy(d.begin(), d.end(), x);
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper, blas::Op::Trans,
               blas::Diag::NonUnit, n, 1, (T)1.0, R, ldr, x, n);

    // Step 4: solve R x = y  (n x 1)
    blas::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans,
               blas::Diag::NonUnit, n, 1, (T)1.0, R, ldr, x, n);

    auto stop = steady_clock::now();
    app_time_us = duration_cast<microseconds>(stop - start).count();
}

// ============================================================================
// Application (b): Generalized Singular Values
// SVD of R gives the generalized singular values of (V, K)
// ============================================================================

template <typename T>
static void app_generalized_svals(
    const T* R, int64_t ldr, int64_t n,
    T* sigma,
    long& app_time_us)
{
    auto start = steady_clock::now();

    // Copy R to work buffer (gesdd destroys input)
    std::vector<T> R_copy(n * n);
    lapack::lacpy(lapack::MatrixType::General, n, n, R, ldr, R_copy.data(), n);

    // SVD of n x n R: only singular values (no vectors)
    std::vector<T> dummy_U(1), dummy_Vt(1);
    lapack::gesdd(lapack::Job::NoVec, n, n, R_copy.data(), n,
                  sigma, dummy_U.data(), 1, dummy_Vt.data(), 1);

    auto stop = steady_clock::now();
    app_time_us = duration_cast<microseconds>(stop - start).count();
}

// ============================================================================
// Application (c): Generalized Singular Vectors
// Full SVD of R: R = U_R * Sigma * V_R^T
// Right generalized singular vectors = columns of V_R
// ============================================================================

template <typename T>
static void app_generalized_svecs(
    const T* R, int64_t ldr, int64_t n,
    T* sigma,
    T* V_R,  // n x n right singular vectors
    T* U_R,  // n x n left singular vectors of R
    T& right_svec_orth_error,
    long& app_time_us)
{
    auto start = steady_clock::now();

    // Copy R to work buffer
    std::vector<T> R_copy(n * n);
    lapack::lacpy(lapack::MatrixType::General, n, n, R, ldr, R_copy.data(), n);

    // Full SVD of n x n R: R = U_R * Sigma * V_R^T
    lapack::gesdd(lapack::Job::AllVec, n, n, R_copy.data(), n,
                  sigma, U_R, n, V_R, n);

    auto stop = steady_clock::now();
    app_time_us = duration_cast<microseconds>(stop - start).count();

    // Verify right singular vector orthogonality: ||V_R^T V_R - I||_F / sqrt(n)
    right_svec_orth_error = RandLAPACK::testing::orthogonality_error<T>(V_R, n, n);
}

// ============================================================================
// CSV output
// ============================================================================

template <typename T>
static void write_common_header_comments(
    std::ofstream& out, int64_t m, int64_t n, int num_runs,
    const std::string& K_file, const std::string& V_file,
    T d_factor, int64_t sketch_nnz, int64_t block_size,
    bool skip_apps, bool compute_cond)
{
    time_t now = time(nullptr);
    out << "# Date: " << ctime(&now)
        << "# Matrix dimensions: m=" << m << " n=" << n << "\n"
        << "# Runs per algorithm: " << num_runs << "\n"
        << "# OpenMP threads: " << omp_get_max_threads() << "\n"
        << "# K_file: " << K_file << "\n"
        << "# V_file: " << V_file << "\n"
        << "# d_factor: " << d_factor << "\n"
        << "# sketch_nnz: " << sketch_nnz << "\n"
        << "# block_size: " << block_size << "\n"
        << "# skip_apps: " << (skip_apps ? 1 : 0) << "\n"
        << "# compute_cond: " << (compute_cond ? 1 : 0) << "\n";
}

template <typename T>
static void write_csv_header(std::ofstream& out, int64_t m, int64_t n, int num_runs,
                             const std::string& K_file, const std::string& V_file,
                             T d_factor, int64_t sketch_nnz, int64_t block_size,
                             bool skip_apps, bool compute_cond) {
    out << "# GSVD Benchmark results\n";
    write_common_header_comments<T>(out, m, n, num_runs, K_file, V_file, d_factor, sketch_nnz, block_size, skip_apps, compute_cond);
    out << "m,n,run,algorithm,chol_time_us,qr_time_us,orth_error,max_orth_cols,"
        << "app_a_time_us,ls_rel_error,"
        << "app_b_time_us,"
        << "app_c_time_us,right_svec_orth_error,"
        << "total_a_time_us,total_b_time_us,total_c_time_us,"
        << "peak_rss_kb,analytical_kb\n";
}

template <typename T>
static void write_csv_row(std::ofstream& out, const gsvd_result<T>& r) {
    out << r.m << "," << r.n << "," << r.run_idx << "," << r.alg_name << ","
        << r.chol_time_us << ","
        << r.qr_time_us << ","
        << std::scientific << std::setprecision(6) << r.orth_error << ","
        << r.max_orth_cols << ","
        << r.app_a_time_us << ","
        << std::scientific << std::setprecision(6) << r.ls_rel_error << ","
        << r.app_b_time_us << ","
        << r.app_c_time_us << ","
        << std::scientific << std::setprecision(6) << r.right_svec_orth_error << ","
        << r.total_a_time_us << "," << r.total_b_time_us << "," << r.total_c_time_us << ","
        << r.peak_rss_kb << "," << r.analytical_kb
        << "\n";
}

// ============================================================================
// Breakdown CSV output
// ============================================================================

template <typename T>
static void write_breakdown_csv(
    const std::string& filename,
    const std::vector<gsvd_result<T>>& results,
    int64_t m, int64_t n, int num_runs,
    const std::string& K_file, const std::string& V_file,
    T d_factor, int64_t sketch_nnz, int64_t block_size,
    bool skip_apps, bool compute_cond)
{
    std::ofstream out(filename);
    out << "# GSVD Benchmark runtime breakdown\n";
    write_common_header_comments<T>(out, m, n, num_runs, K_file, V_file, d_factor, sketch_nnz, block_size, skip_apps, compute_cond);
    out << "# Times are in microseconds\n";
    out << "# CQRRT_linop breakdown (11): alloc, sketch, qr, tri_inv, fwd, adj, trmm, chol, finalize, rest, total\n";
    out << "# CholQR breakdown (6): alloc, fwd, adj, chol, rest, total\n";
    out << "# sCholQR3 breakdown (18): alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, fwd3, adj3, gemm3, chol3, upd3, q_mat, rest, total\n";
    out << "# sCholQR3_basic breakdown (15): alloc, fwd1, adj1, chol1, trsm1, fwd_q, syrk2, chol2, upd2, syrk3, chol3, upd3, q_mat, rest, total\n";

    // Column header: m, n, run, algorithm, then all breakdown times
    // Max breakdown length is 18 (sCholQR3)
    out << "m,n,run,algorithm";
    for (int i = 0; i < 18; ++i) out << ",t" << i;
    out << "\n";

    for (const auto& r : results) {
        out << r.m << "," << r.n << "," << r.run_idx << "," << r.alg_name;
        for (size_t i = 0; i < r.qr_breakdown.size(); ++i) {
            out << "," << r.qr_breakdown[i];
        }
        // Pad with zeros if fewer than 18 columns
        for (size_t i = r.qr_breakdown.size(); i < 18; ++i) {
            out << ",0";
        }
        out << "\n";
    }
}

// ============================================================================
// Console summary
// ============================================================================

template <typename T>
static void print_summary(const std::string& alg_name, const std::vector<gsvd_result<T>>& results) {
    printf("\n  %s:\n", alg_name.c_str());
    for (const auto& r : results) {
        printf("    Run %ld: orth_err=%.2e, max_orth=%ld/%ld, QR=%ld us\n",
               (long)r.run_idx, (double)r.orth_error, (long)r.max_orth_cols, (long)r.n, r.qr_time_us);
        if (r.app_a_time_us > 0 || r.ls_rel_error > 0) {
            printf("           LS_err=%.2e, App(a)=%ld us, App(b)=%ld us, App(c)=%ld us\n",
                   (double)r.ls_rel_error, r.app_a_time_us, r.app_b_time_us, r.app_c_time_us);
        }
        printf("           Memory: peak_RSS=%ld KB, predicted=%ld KB\n",
               r.peak_rss_kb, r.analytical_kb);
    }
}

// ============================================================================
// Main benchmark
// ============================================================================

template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[]) {
    // Parse arguments
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <K_file> <V_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [skip_apps] [compute_cond]\n";
        return 1;
    }

    std::string output_dir = argv[2];
    int64_t num_runs       = std::stol(argv[3]);
    std::string K_file     = argv[4];
    std::string V_file     = argv[5];
    T d_factor             = std::stod(argv[6]);
    int64_t sketch_nnz     = (argc >= 8) ? std::stol(argv[7]) : 4;
    int64_t block_size     = (argc >= 9) ? std::stol(argv[8]) : 0;
    bool skip_apps         = (argc >= 10) ? (std::stol(argv[9]) != 0) : false;
    bool compute_cond      = (argc >= 11) ? (std::stol(argv[10]) != 0) : false;

    std::cout << "=== GSVD/Generalized LS Benchmark ===\n";
    std::cout << "  K file: " << K_file << "\n";
    std::cout << "  V file: " << V_file << "\n";
    std::cout << "  d_factor: " << d_factor << "\n";
    std::cout << "  sketch_nnz: " << sketch_nnz << "\n";
    std::cout << "  block_size: " << block_size << "\n";
    std::cout << "  skip_apps: " << (skip_apps ? "yes" : "no") << "\n";
    std::cout << "  compute_cond: " << (compute_cond ? "yes" : "no") << "\n";
    std::cout << "  num_runs: " << num_runs << "\n";
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n\n";

    // ================================================================
    // Step 1: Load V from Matrix Market
    // ================================================================
    std::cout << "Loading V from " << V_file << "... " << std::flush;
    auto V_coo = RandLAPACK_extras::coo_from_matrix_market<T>(V_file);
    int64_t m = V_coo.n_rows;
    int64_t n = V_coo.n_cols;
    RandBLAS::sparse_data::csr::CSRMatrix<T> V_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(V_coo, V_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> V_linop(m, n, V_csr);
    std::cout << "done (" << m << " x " << n << ", nnz=" << V_coo.nnz << ")\n";

    // ================================================================
    // Step 2: Create L^{-1} operator (half_solve=true) and K^{-1} operator
    // ================================================================
    std::cout << "Factorizing K = LL^T from " << K_file << "... " << std::flush;

    RandLAPACK_extras::linops::CholSolverLinOp<T> L_inv_op(K_file, /*half_solve=*/true);
    auto chol_start = steady_clock::now();
    L_inv_op.factorize();
    auto chol_stop = steady_clock::now();
    long chol_time_us = duration_cast<microseconds>(chol_stop - chol_start).count();

    // Also create full K^{-1} for App (a) — only needed when running apps
    std::unique_ptr<RandLAPACK_extras::linops::CholSolverLinOp<T>> K_inv_op_ptr;
    if (!skip_apps) {
        K_inv_op_ptr = std::make_unique<RandLAPACK_extras::linops::CholSolverLinOp<T>>(K_file, /*half_solve=*/false);
        K_inv_op_ptr->factorize();
    }

    std::cout << "done (" << chol_time_us << " us)\n";

    // ================================================================
    // Step 3: Form composite operator L^{-1} * V
    // ================================================================
    RandLAPACK::linops::CompositeOperator LiV_op(m, n, L_inv_op, V_linop);
    LiV_op.block_size = block_size;
    std::cout << "Composite operator L^{-1}V: " << m << " x " << n << "\n";

    // Condition number diagnostic (materializes L^{-1}V, runs two SVDs)
    if (compute_cond) {
        RandLAPACK::testing::print_condition_diagnostics<T>(LiV_op, "L^{-1}V");
    }

    // ================================================================
    // Step 4: Generate synthetic RHS: b = V * x_true (only when running apps)
    // ================================================================
    RandBLAS::RNGState<RNG> rng_state(42);
    std::vector<T> x_true(n);
    std::vector<T> b(m, 0.0);
    T x_true_norm = 0.0;
    if (!skip_apps) {
        RandBLAS::DenseDist D(n, 1);
        auto next_state = RandBLAS::fill_dense(D, x_true.data(), rng_state);
        rng_state = next_state;

        V_linop(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, 1, n, (T)1.0, x_true.data(), n, (T)0.0, b.data(), m);

        x_true_norm = blas::nrm2(n, x_true.data(), 1);
        std::cout << "Generated b = V * x_true (||x_true|| = " << x_true_norm << ")\n";
    }
    std::cout << "\n";

    // ================================================================
    // Prepare RNG states for each run
    // ================================================================
    RandBLAS::RNGState<RNG> main_state(123);
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = main_state;
        if (r > 0) run_states[r].key.incr(r);
    }

    // Shared buffers
    std::vector<T> Q_buf(m * n);
    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);

    // Storage for all results
    std::vector<gsvd_result<T>> all_results;

    // ================================================================
    // Run warmup (unreported)
    // ================================================================
    std::cout << "Running warmup... " << std::flush;
    {
        auto warmup_state = run_states[0];
        std::vector<T> R_warmup(n * n, 0.0);
        RandLAPACK::CQRRT_linops<T, RNG> warmup_algo(false, tol, false);
        warmup_algo.nnz = sketch_nnz;
        warmup_algo.block_size = block_size;
        warmup_algo.call(LiV_op, R_warmup.data(), n, d_factor, warmup_state);
    }
    std::cout << "done\n\n";

    // ================================================================
    // Common per-algorithm loop: run QR, check orthogonality, run apps
    // ================================================================
    // call_algo(res, R, run_idx) fills res.qr_time_us, res.qr_breakdown,
    // res.peak_rss_kb, res.analytical_kb — everything else is shared.
    auto run_algo = [&](const std::string& name, auto call_algo) {
        std::cout << "\n=== " << name << " ===\n";
        std::vector<gsvd_result<T>> results(num_runs);
        for (int64_t r = 0; r < num_runs; ++r) {
            auto& res = results[r];
            res.m = m; res.n = n; res.run_idx = r;
            res.alg_name = name;
            res.chol_time_us = chol_time_us;

            std::vector<T> R(n * n, 0.0);
            call_algo(res, R, r);

            compute_Q_from_R(LiV_op, R.data(), n, Q_buf.data(), m, n);
            res.orth_error     = RandLAPACK::testing::orthogonality_error<T>(Q_buf.data(), m, n);
            res.is_orthonormal = (res.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75));
            res.max_orth_cols  = RandLAPACK::testing::max_orthonormal_cols<T>(Q_buf.data(), m, n);

            if (!skip_apps) {
                std::vector<T> x_computed(n, 0.0);
                app_generalized_ls(*K_inv_op_ptr, V_linop, R.data(), n, n,
                                   b.data(), m, x_computed.data(), res.app_a_time_us);
                blas::axpy(n, (T)-1.0, x_true.data(), 1, x_computed.data(), 1);
                res.ls_rel_error = blas::nrm2(n, x_computed.data(), 1) / x_true_norm;

                std::vector<T> sigma_b(n, 0.0);
                app_generalized_svals(R.data(), n, n, sigma_b.data(), res.app_b_time_us);

                std::vector<T> sigma_c(n, 0.0), V_R(n * n, 0.0), U_R(n * n, 0.0);
                app_generalized_svecs(R.data(), n, n, sigma_c.data(), V_R.data(), U_R.data(),
                                      res.right_svec_orth_error, res.app_c_time_us);
            }

            res.total_a_time_us = res.qr_time_us + res.app_a_time_us;
            res.total_b_time_us = res.qr_time_us + res.app_b_time_us;
            res.total_c_time_us = res.qr_time_us + res.app_c_time_us;
            all_results.push_back(res);
        }
        print_summary(name, results);
    };

    // ================================================================
    // CQRRT_linop
    // ================================================================
    run_algo("CQRRT_linop", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t r) {
        auto state = run_states[r];
        RandLAPACK::CQRRT_linops<T, RNG> algo(true, tol, false);
        algo.nnz = sketch_nnz; algo.block_size = block_size;
        RandLAPACK::PeakRSSTracker mem; mem.start();
        algo.call(LiV_op, R.data(), n, d_factor, state);
        res.peak_rss_kb = mem.stop();
        res.qr_time_us = algo.times[10];
        // breakdown: alloc, sketch, qr, tri_inv, fwd, adj, trmm, chol, finalize, rest, total
        res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 11);
        res.analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
    });

    // ================================================================
    // CholQR
    // ================================================================
    run_algo("CholQR", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t) {
        RandLAPACK::CholQR_linops<T> algo(true, tol, false);
        algo.block_size = block_size;
        RandLAPACK::PeakRSSTracker mem; mem.start();
        algo.call(LiV_op, R.data(), n);
        res.peak_rss_kb = mem.stop();
        res.qr_time_us = algo.times[5];
        // breakdown: alloc, fwd, adj, chol, rest, total
        res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 6);
        res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
    });

    // ================================================================
    // sCholQR3
    // ================================================================
    run_algo("sCholQR3", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t) {
        RandLAPACK::sCholQR3_linops<T> algo(true, tol, false);
        algo.block_size = block_size;
        RandLAPACK::PeakRSSTracker mem; mem.start();
        algo.call(LiV_op, R.data(), n);
        res.peak_rss_kb = mem.stop();
        res.qr_time_us = algo.times[17];
        // breakdown (18): alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, fwd3, adj3, gemm3, chol3, upd3, q_mat, rest, total
        res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 18);
        res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
    });

    // ================================================================
    // sCholQR3_basic (non-blocked, matches standard pseudocode)
    // ================================================================
    run_algo("sCholQR3_basic", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t) {
        RandLAPACK::sCholQR3_linops_basic<T> algo(true, tol, false);
        RandLAPACK::PeakRSSTracker mem; mem.start();
        algo.call(LiV_op, R.data(), n);
        res.peak_rss_kb = mem.stop();
        res.qr_time_us = algo.times[14];
        // breakdown (15): alloc, fwd1, adj1, chol1, trsm1, fwd_q, syrk2, chol2, upd2, syrk3, chol3, upd3, q_mat, rest, total
        res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 15);
        res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m, n);
    });

    // ================================================================
    // Write CSV output
    // ================================================================
    // Generate timestamped filenames
    char time_buf[64];
    time_t now = time(nullptr);
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&now));

    std::string results_file   = output_dir + "/" + time_buf + "_gsvd_results.csv";
    std::string breakdown_file = output_dir + "/" + time_buf + "_gsvd_breakdown.csv";

    std::ofstream out(results_file);
    write_csv_header<T>(out, m, n, num_runs, K_file, V_file, d_factor, sketch_nnz, block_size, skip_apps, compute_cond);
    for (const auto& r : all_results) {
        write_csv_row(out, r);
    }
    out.close();
    std::cout << "\n\nResults written to " << results_file << "\n";

    write_breakdown_csv(breakdown_file, all_results, m, n, num_runs, K_file, V_file, d_factor, sketch_nnz, block_size, skip_apps, compute_cond);
    std::cout << "Runtime breakdown written to " << breakdown_file << "\n";

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <K_file> <V_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [skip_apps]\n";
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
