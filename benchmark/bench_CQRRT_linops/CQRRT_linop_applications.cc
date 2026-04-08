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
//   ./CQRRT_linop_applications <prec> <outdir> <runs> <K_file|sparse> <V_file|A_file> <d_factor>
//                    [sketch_nnz] [block_size] [skip_apps] [compute_cond] [run_expl] [upcast_orth] [method_mask]
//
//   method_mask (integer bitmask, optional, default = 0b001111 or 0b011111 if run_expl=1):
//     bit 0 ( 1): CQRRT_linop
//     bit 1 ( 2): CholQR
//     bit 2 ( 4): sCholQR3
//     bit 3 ( 8): sCholQR3_basic
//     bit 4 (16): CQRRT_expl      (requires A materialization)
//     bit 5 (32): CQRRT_linop_stb (GEQP3-stabilized preconditioner)

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

// Extras utilities (Eigen-dependent)
#include "../../extras/misc/ext_util.hh"
#include "../../extras/linops/ext_cholsolver_linop.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrt_bench_common.hh"

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

    // R-factor backward error: ||A^T A - R^T R||_F / ||A||_F^2
    double r_backward_error;

    // Orthogonality computed with upcast reconstruction (float->double or double->long double)
    double orth_error_upcast;

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
    auto Eye = make_eye<T>(n);
    A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
         m, n, n, (T)1.0, Eye.data(), n, (T)0.0, Q_out, m);
    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans,
               blas::Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_out, m);
}

// Compute A^T A via blocked linop calls. Peak memory: O(m*b + n^2).
// No m x n materialization needed.
template <typename T, typename GLO>
static void compute_AtA_blocked(GLO& A_op, int64_t m, int64_t n, T* AtA, int64_t b) {
    std::fill(AtA, AtA + n * n, (T)0.0);
    std::vector<T> E_block(n * b, 0.0);   // n x b identity block
    std::vector<T> A_block(m * b, 0.0);   // m x b = A * E_block
    std::vector<T> AtA_block(n * b, 0.0); // n x b = A^T * A_block

    for (int64_t j0 = 0; j0 < n; j0 += b) {
        int64_t bk = std::min(b, n - j0);

        // E_block = I[:, j0:j0+bk]
        std::fill(E_block.begin(), E_block.end(), (T)0.0);
        for (int64_t j = 0; j < bk; ++j)
            E_block[(j0 + j) + j * n] = (T)1.0;

        // A_block = A * E_block  (m x bk)
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, bk, n, (T)1.0, E_block.data(), n, (T)0.0, A_block.data(), m);

        // AtA[:, j0:j0+bk] = A^T * A_block  (n x bk)
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             n, bk, m, (T)1.0, A_block.data(), m, (T)0.0, AtA_block.data(), n);

        // Copy into AtA columns j0..j0+bk
        lapack::lacpy(lapack::MatrixType::General, n, bk, AtA_block.data(), n, AtA + j0 * n, n);
    }
}

// Compute R-factor backward error: ||A^T A - R^T R||_F / ||A||_F^2
// Uses blocked linop calls — peak memory O(m*b + n^2), no m x n materialization.
template <typename T, typename GLO>
static double compute_r_backward_error(GLO& A_op, const T* R, int64_t m, int64_t n, int64_t block_size) {
    int64_t b = (block_size > 0) ? block_size : 256;

    // A^T A via blocked linop (n x n)
    std::vector<T> AtA(n * n, 0.0);
    compute_AtA_blocked(A_op, m, n, AtA.data(), b);

    // R^T R (n x n)
    std::vector<T> RtR(n * n, 0.0);
    blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1.0, R, n, (T)0.0, RtR.data(), n);
    fill_lower_from_upper(RtR.data(), n);

    // ||A||_F^2 = trace(A^T A)
    T norm_A_sq = 0;
    for (int64_t i = 0; i < n; ++i)
        norm_A_sq += AtA[i + i * n];

    // ||A^T A - R^T R||_F
    T diff_norm_sq = 0;
    #pragma omp parallel for reduction(+:diff_norm_sq) schedule(static)
    for (int64_t i = 0; i < n * n; ++i) {
        T d = AtA[i] - RtR[i];
        diff_norm_sq += d * d;
    }

    return (double)(std::sqrt(diff_norm_sq) / (norm_A_sq));
}

// Compute orthogonality with upcast reconstruction.
// Algorithm runs in precision T, Q = A R^{-1} computed in precision U (one level up).
// float -> double: uses BLAS++/LAPACK++ (MKL-backed DTRSM + DSYRK + DLANSY).
// double -> long double: uses Eigen (BLAS++ does not support long double).
// If A_dense is non-null, uses it directly; otherwise materializes via linop.
template <typename T, typename U, typename GLO>
static double compute_orth_upcast(GLO& A_op, const T* R, int64_t m, int64_t n,
                                   const T* A_dense = nullptr) {
    const T* A_ptr;
    std::vector<T> A_buf;
    if (A_dense) {
        A_ptr = A_dense;
    } else {
        A_buf.resize(m * n);
        auto Eye = make_eye<T>(n);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_buf.data(), m);
        A_ptr = A_buf.data();
    }

    // Upcast A and R to precision U
    std::vector<U> A_U(m * n), R_U(n * n);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m * n; ++i) A_U[i] = (U)A_ptr[i];
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n * n; ++i) R_U[i] = (U)R[i];

    if constexpr (std::is_same_v<U, double>) {
        // float->double: BLAS++ path — MKL DTRSM + DSYRK + DLANSY.
        // Solve in-place: A_U = A_U * R_U^{-1} (becomes Q in double).
        blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
                   blas::Op::NoTrans, blas::Diag::NonUnit,
                   m, n, 1.0, R_U.data(), n, A_U.data(), m);
        // GmI = Q^T Q - I (upper triangle via SYRK, then lansy for Frobenius norm)
        std::vector<U> GmI(n * n, 0.0);
        RandLAPACK::util::eye(n, n, GmI.data());
        blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
                   n, m, 1.0, A_U.data(), m, -1.0, GmI.data(), n);
        return lapack::lansy(lapack::Norm::Fro, blas::Uplo::Upper, n, GmI.data(), n) / std::sqrt((double)n);
    } else {
        // double->long double: Eigen path (BLAS++ does not support long double).
        Eigen::Map<Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            A_map(A_U.data(), m, n);
        Eigen::Map<Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>
            R_map(R_U.data(), n, n);
        // Solve R^T * Q^T = A^T for Q^T
        Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> Qt =
            R_map.transpose().template triangularView<Eigen::Lower>().solve(A_map.transpose());
        // ||Q^T Q - I||_F / sqrt(n)
        Eigen::Matrix<U, Eigen::Dynamic, Eigen::Dynamic> QtQ = Qt * Qt.transpose();
        for (int64_t i = 0; i < n; ++i) QtQ(i, i) -= (U)1.0;
        return (double)(QtQ.norm() / std::sqrt((U)n));
    }
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
#ifdef _OPENMP
        << "# OpenMP threads: " << omp_get_max_threads() << "\n"
#else
        << "# OpenMP threads: 1\n"
#endif
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
    out << "m,n,run,algorithm,chol_time_us,qr_time_us,orth_error,r_backward_error,orth_error_upcast,"
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
        << std::scientific << std::setprecision(6) << r.r_backward_error << ","
        << std::scientific << std::setprecision(6) << r.orth_error_upcast << ","
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
        printf("    Run %ld: orth_err=%.2e, r_bwd_err=%.2e, QR=%ld us\n",
               (long)r.run_idx, (double)r.orth_error, r.r_backward_error, r.qr_time_us);
        if (r.orth_error_upcast > 0)
            printf("           orth_upcast=%.2e\n", r.orth_error_upcast);
        if (r.app_a_time_us > 0 || r.ls_rel_error > 0) {
            printf("           LS_err=%.2e, App(a)=%ld us, App(b)=%ld us, App(c)=%ld us\n",
                   (double)r.ls_rel_error, r.app_a_time_us, r.app_b_time_us, r.app_c_time_us);
        }
        printf("           Memory: peak_RSS=%ld KB, predicted=%ld KB\n",
               r.peak_rss_kb, r.analytical_kb);
    }
}

// ============================================================================
// Core benchmark logic (templated over operator type)
// ============================================================================

template <typename T, typename RNG, typename OpType>
static int run_benchmark_inner(
    OpType& A_op,
    int64_t m, int64_t n,
    const std::string& output_dir,
    int64_t num_runs,
    T d_factor, int64_t sketch_nnz, int64_t block_size,
    bool skip_apps, bool compute_cond, bool upcast_orth, int64_t method_mask,
    long chol_time_us,
    const std::string& K_file, const std::string& V_file,
    const std::string& op_label,
    std::function<void(const std::vector<T>&, gsvd_result<T>&)> app_a_fn)
{
    // Condition number diagnostic (materializes A_op, runs two SVDs)
    if (compute_cond) {
        RandLAPACK::testing::print_condition_diagnostics<T>(A_op, op_label);
    }

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
        warmup_algo.call(A_op, R_warmup.data(), n, d_factor, warmup_state);
    }
    std::cout << "done\n\n";

    // ================================================================
    // Pre-materialize A for upcast orthogonality (once, shared across all algorithms)
    // ================================================================
    T* A_materialized = nullptr;
    if (upcast_orth || (method_mask & 16)) {
        std::cout << "Materializing " << op_label << " for upcast/expl (" << m << " x " << n << ", "
                  << (m * n * sizeof(T) / (1024.0 * 1024.0 * 1024.0)) << " GB)... " << std::flush;
        A_materialized = new T[m * n];
        auto Eye = make_eye<T>(n);
        A_op(blas::Side::Left, blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_materialized, m);
        std::cout << "done\n\n";
    }

    // ================================================================
    // Pre-compute A^T A and ||A||_F^2 for r_backward_error (A is constant)
    // ================================================================
    std::vector<T> AtA_precomputed;
    T norm_A_sq_precomputed = 0;
    if (A_materialized) {
        AtA_precomputed.resize(n * n, 0.0);
        blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
                   n, m, (T)1.0, A_materialized, m, (T)0.0, AtA_precomputed.data(), n);
        fill_lower_from_upper(AtA_precomputed.data(), n);
        for (int64_t i = 0; i < n; ++i)
            norm_A_sq_precomputed += AtA_precomputed[i + i * n];
    }

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

            // Compute Q = A * R^{-1}: use pre-materialized A if available, else via linop
            if (A_materialized) {
                lapack::lacpy(lapack::MatrixType::General, m, n, A_materialized, m, Q_buf.data(), m);
                blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans,
                           blas::Diag::NonUnit, m, n, (T)1.0, R.data(), n, Q_buf.data(), m);
            } else {
                compute_Q_from_R(A_op, R.data(), n, Q_buf.data(), m, n);
            }
            res.orth_error     = RandLAPACK::testing::orthogonality_error<T>(Q_buf.data(), m, n);
            res.is_orthonormal = (res.orth_error <= std::pow(std::numeric_limits<T>::epsilon(), (T)0.75));

            // R-factor backward error: ||A^T A - R^T R||_F / ||A||_F^2
            // Use precomputed A^T A (A is constant across algorithms), else blocked linop
            if (A_materialized) {
                std::vector<T> RtR(n * n, 0.0);
                blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
                           n, n, (T)1.0, R.data(), n, (T)0.0, RtR.data(), n);
                fill_lower_from_upper(RtR.data(), n);

                T diff_sq = 0;
                #pragma omp parallel for reduction(+:diff_sq) schedule(static)
                for (int64_t i = 0; i < n * n; ++i) { T d = AtA_precomputed[i] - RtR[i]; diff_sq += d * d; }
                res.r_backward_error = (double)(std::sqrt(diff_sq) / norm_A_sq_precomputed);
            } else {
                res.r_backward_error = compute_r_backward_error(A_op, R.data(), m, n, block_size);
            }

            // Upcast orthogonality: reconstruct Q in higher precision (uses pre-materialized A)
            if (upcast_orth) {
                if constexpr (std::is_same_v<T, float>) {
                    res.orth_error_upcast = compute_orth_upcast<T, double>(A_op, R.data(), m, n, A_materialized);
                } else if constexpr (std::is_same_v<T, double>) {
                    res.orth_error_upcast = compute_orth_upcast<T, long double>(A_op, R.data(), m, n, A_materialized);
                }
            } else {
                res.orth_error_upcast = 0.0;
            }

            if (!skip_apps) {
                // App (a): generalized LS — only in composite mode (app_a_fn is null in sparse mode)
                if (app_a_fn) {
                    app_a_fn(R, res);
                }

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
    // CQRRT_linop variants (standard TRSM_IDENTITY and GEQP3-stabilized)
    // ================================================================
    auto run_cqrrt_linop = [&](const std::string& name, RandLAPACK::CQRRTLinopPrecond precond) {
        run_algo(name, [&](gsvd_result<T>& res, std::vector<T>& R, int64_t r) {
            auto state = run_states[r];
            RandLAPACK::CQRRT_linops<T, RNG> algo(true, tol, false);
            algo.nnz = sketch_nnz; algo.block_size = block_size;
            algo.precond_method = precond;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            algo.call(A_op, R.data(), n, d_factor, state);
            res.peak_rss_kb = mem.stop();
            res.qr_time_us = algo.times[10];
            // breakdown: alloc, sketch, qr, tri_inv, fwd, adj, trmm, chol, finalize, rest, total
            res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 11);
            res.analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
        });
    };
    if (method_mask & 1)
        run_cqrrt_linop("CQRRT_linop", RandLAPACK::CQRRTLinopPrecond::TRSM_IDENTITY);

    // ================================================================
    // CholQR
    // ================================================================
    if (method_mask & 2) {
        run_algo("CholQR", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t) {
            RandLAPACK::CholQR_linops<T> algo(true, tol, false);
            algo.block_size = block_size;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            algo.call(A_op, R.data(), n);
            res.peak_rss_kb = mem.stop();
            res.qr_time_us = algo.times[5];
            // breakdown: alloc, fwd, adj, chol, rest, total
            res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 6);
            res.analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
        });
    }

    // ================================================================
    // sCholQR3
    // ================================================================
    if (method_mask & 4) {
        run_algo("sCholQR3", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t) {
            RandLAPACK::sCholQR3_linops<T> algo(true, tol, false);
            algo.block_size = block_size;
            RandLAPACK::PeakRSSTracker mem; mem.start();
            algo.call(A_op, R.data(), n);
            res.peak_rss_kb = mem.stop();
            res.qr_time_us = algo.times[17];
            // breakdown (18): alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, fwd3, adj3, gemm3, chol3, upd3, q_mat, rest, total
            res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 18);
            res.analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
        });
    }

    // ================================================================
    // sCholQR3_basic (non-blocked, matches standard pseudocode)
    // ================================================================
    if (method_mask & 8) {
        run_algo("sCholQR3_basic", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t) {
            RandLAPACK::sCholQR3_linops_basic<T> algo(true, tol, false);
            RandLAPACK::PeakRSSTracker mem; mem.start();
            algo.call(A_op, R.data(), n);
            res.peak_rss_kb = mem.stop();
            res.qr_time_us = algo.times[14];
            // breakdown (15): alloc, fwd1, adj1, chol1, trsm1, fwd_q, syrk2, chol2, upd2, syrk3, chol3, upd3, q_mat, rest, total
            res.qr_breakdown.assign(algo.times.begin(), algo.times.begin() + 15);
            res.analytical_kb = RandLAPACK::scholqr3_linops_basic_analytical_kb<T>(m, n);
        });
    }

    // ================================================================
    // CQRRT_expl (uses pre-materialized A, run dense CQRRT)
    // ================================================================
    if (method_mask & 16) {
        run_algo("CQRRT_expl", [&](gsvd_result<T>& res, std::vector<T>& R, int64_t r) {
            // Time the copy: CQRRT modifies A in-place, so a fresh copy is needed each run.
            // This per-run materialization cost is not captured by algo.times — store in slot [2].
            T* A_copy = new T[m * n];
            auto copy_start = std::chrono::steady_clock::now();
            lapack::lacpy(lapack::MatrixType::General, m, n, A_materialized, m, A_copy, m);
            auto copy_end = std::chrono::steady_clock::now();
            long copy_dur = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();

            auto state = run_states[r];
            RandLAPACK::CQRRT<T, RNG> algo(true, tol);
            algo.compute_Q = false;
            algo.orthogonalization = false;
            algo.nnz = sketch_nnz;

            RandLAPACK::PeakRSSTracker mem; mem.start();
            algo.call(m, n, A_copy, m, R.data(), n, d_factor, state);
            res.peak_rss_kb = mem.stop();

            // CQRRT dense times layout (10 entries, 0-indexed):
            //   [0]=saso, [1]=qr, [2]=trtri(0), [3]=precond, [4]=gram,
            //   [5]=trmm_gram(0), [6]=potrf, [7]=finalize, [8]=rest, [9]=total
            // Store copy_dur in slot [2] (repurposed trtri placeholder) and include in total.
            res.qr_breakdown.assign(algo.times.begin(), algo.times.end());
            while (res.qr_breakdown.size() < 11) res.qr_breakdown.push_back(0);
            res.qr_breakdown[2]  = copy_dur;
            res.qr_breakdown[9] += copy_dur;  // update total to include copy
            res.qr_time_us = res.qr_breakdown[9];
            res.analytical_kb = (m * n * sizeof(T)) / 1024; // just the dense matrix

            delete[] A_copy;
        });
    }

    // ================================================================
    // CQRRT_linop_stb (GEQP3-stabilized preconditioner, uses pre-materialized A for orth check)
    // ================================================================
    if (method_mask & 32)
        run_cqrrt_linop("CQRRT_linop_stb", RandLAPACK::CQRRTLinopPrecond::GEQP3);

    // Free pre-materialized A
    delete[] A_materialized;

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

// ============================================================================
// Main benchmark dispatcher
// ============================================================================

template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <K_file|sparse> <V_file|A_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [skip_apps] [compute_cond] [run_expl] [upcast_orth] [method_mask]\n"
                  << "  sparse mode: pass 'sparse' as K_file and a single A.mtx as V_file\n"
                  << "  method_mask: bitmask selecting algorithms (default = 0b001111 or 0b011111 if run_expl=1)\n"
                  << "    bit 0 ( 1): CQRRT_linop\n"
                  << "    bit 1 ( 2): CholQR\n"
                  << "    bit 2 ( 4): sCholQR3\n"
                  << "    bit 3 ( 8): sCholQR3_basic\n"
                  << "    bit 4 (16): CQRRT_expl      (requires A materialization)\n"
                  << "    bit 5 (32): CQRRT_linop_stb (GEQP3-stabilized preconditioner)\n";
        return 1;
    }

    std::string output_dir = argv[2];
    int64_t num_runs       = std::stol(argv[3]);
    std::string K_file     = argv[4];
    std::string V_file     = argv[5];
    T d_factor             = std::stod(argv[6]);
    int64_t sketch_nnz     = (argc >= 8)  ? std::stol(argv[7])  : 4;
    int64_t block_size     = (argc >= 9)  ? std::stol(argv[8])  : 0;
    bool skip_apps         = (argc >= 10) ? (std::stol(argv[9])  != 0) : false;
    bool compute_cond      = (argc >= 11) ? (std::stol(argv[10]) != 0) : false;
    bool run_expl          = (argc >= 12) ? (std::stol(argv[11]) != 0) : false;
    bool upcast_orth       = (argc >= 13) ? (std::stol(argv[12]) != 0) : false;
    // method_mask: bitmask selecting which algorithms to run.
    // Default: bits 0-3 (all linop methods), plus bit 4 if run_expl=1 (backward compat).
    // Explicit method_mask overrides run_expl for algorithm selection.
    int64_t method_mask    = (argc >= 14) ? std::stol(argv[13]) : (0b001111 | (run_expl ? 0b010000 : 0));

    bool sparse_mode = (K_file == "sparse");

    std::cout << "=== GSVD/Generalized LS Benchmark ===\n";
    if (sparse_mode) {
        std::cout << "  Mode: sparse (single-matrix SparseLinOp)\n";
        std::cout << "  A file: " << V_file << "\n";
    } else {
        std::cout << "  Mode: composite (L^{-1}V)\n";
        std::cout << "  K file: " << K_file << "\n";
        std::cout << "  V file: " << V_file << "\n";
    }
    std::cout << "  d_factor: " << d_factor << "\n";
    std::cout << "  sketch_nnz: " << sketch_nnz << "\n";
    std::cout << "  block_size: " << block_size << "\n";
    std::cout << "  skip_apps: " << (skip_apps ? "yes" : "no") << "\n";
    std::cout << "  compute_cond: " << (compute_cond ? "yes" : "no") << "\n";
    std::cout << "  run_expl: " << (run_expl ? "yes" : "no") << " (backward compat; overridden by method_mask)\n";
    std::cout << "  upcast_orth: " << (upcast_orth ? "yes" : "no") << "\n";
    std::cout << "  method_mask: " << method_mask << " (bits: linop=" << (method_mask&1)
              << " CholQR=" << ((method_mask>>1)&1) << " sCholQR3=" << ((method_mask>>2)&1)
              << " sCholQR3_basic=" << ((method_mask>>3)&1) << " expl=" << ((method_mask>>4)&1)
              << " linop_stb=" << ((method_mask>>5)&1) << ")\n";
    std::cout << "  num_runs: " << num_runs << "\n";
#ifdef _OPENMP
    std::cout << "  OpenMP threads: " << omp_get_max_threads() << "\n\n";
#else
    std::cout << "  OpenMP threads: 1\n\n";
#endif

    // ================================================================
    // Step 1: Load A (sparse mode) or V (composite mode) from Matrix Market
    // ================================================================
    std::cout << "Loading " << (sparse_mode ? "A" : "V") << " from " << V_file << "... " << std::flush;
    int64_t m, n, nnz_V;
    auto V_csr = load_csr<T>(V_file, m, n, nnz_V);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> V_linop(m, n, V_csr);
    std::cout << "done (" << m << " x " << n << ", nnz=" << nnz_V << ")\n";

    if (m < n) {
        std::cerr << "Error: matrix must be overdetermined (m >= n), got " << m << "x" << n << "\n";
        return 1;
    }

    // ================================================================
    // Sparse mode: wrap A directly as SparseLinOp, skip Cholesky
    // ================================================================
    if (sparse_mode) {
        if (!skip_apps) {
            std::cout << "  Note: in sparse mode, app (a) (Gen. LS) is unavailable (requires K).\n"
                      << "        Apps (b) and (c) will still run.\n";
        }
        std::function<void(const std::vector<T>&, gsvd_result<T>&)> app_a_fn = nullptr;
        return run_benchmark_inner<T, RNG>(
            V_linop, m, n, output_dir, num_runs,
            d_factor, sketch_nnz, block_size,
            skip_apps, compute_cond, upcast_orth, method_mask,
            0L /*chol_time_us*/, "sparse", V_file,
            "A (" + V_file + ")", app_a_fn);
    }

    // ================================================================
    // Composite mode: Steps 2-4
    // ================================================================

    // Step 2: Create L^{-1} operator (half_solve=true) and K^{-1} operator
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

    // Step 3: Form composite operator L^{-1} * V
    RandLAPACK::linops::CompositeOperator LiV_op(m, n, L_inv_op, V_linop);
    LiV_op.block_size = block_size;
    std::cout << "Composite operator L^{-1}V: " << m << " x " << n << "\n";

    // Step 4: Generate synthetic RHS: b = V * x_true (only when running apps)
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

    // App (a) callback — captures K_inv_op_ptr, V_linop, b, x_true, x_true_norm
    std::function<void(const std::vector<T>&, gsvd_result<T>&)> app_a_fn = nullptr;
    if (!skip_apps) {
        app_a_fn = [&](const std::vector<T>& R, gsvd_result<T>& res) {
            std::vector<T> x_computed(n, 0.0);
            app_generalized_ls(*K_inv_op_ptr, V_linop, R.data(), n, n,
                               b.data(), m, x_computed.data(), res.app_a_time_us);
            blas::axpy(n, (T)-1.0, x_true.data(), 1, x_computed.data(), 1);
            res.ls_rel_error = blas::nrm2(n, x_computed.data(), 1) / x_true_norm;
        };
    }

    return run_benchmark_inner<T, RNG>(
        LiV_op, m, n, output_dir, num_runs,
        d_factor, sketch_nnz, block_size,
        skip_apps, compute_cond, upcast_orth, method_mask,
        chol_time_us, K_file, V_file,
        "L^{-1}V", app_a_fn);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_runs> <K_file|sparse> <V_file|A_file> <d_factor>"
                  << " [sketch_nnz] [block_size] [skip_apps] [compute_cond] [run_expl] [upcast_orth] [method_mask]\n";
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
