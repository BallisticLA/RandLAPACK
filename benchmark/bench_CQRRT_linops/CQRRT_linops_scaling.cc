#if defined(__APPLE__)
int main() {return 0;}
#else

// CQRRT_linops scaling study - measures performance and accuracy as matrix size varies
// Tests both CQRRT (preconditioned) and CholQR (unpreconditioned) on tall sparse matrices
// Supports two modes:
//   1. Fixed aspect ratio: n = m / aspect_ratio (both dimensions scale together)
//   2. Fixed columns: n is constant while m varies

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
#include <omp.h>

// Demo utilities for Matrix Market I/O
#include "../../demos/functions/misc/dm_util.hh"

// Linops algorithms (now in main RandLAPACK)
#include "rl_cqrrt_linops.hh"
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "rl_memory_tracker.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Common quality + timing fields shared by all algorithms
template <typename T>
struct alg_quality {
    T orth_error;           // ||Q^T Q - I|| / sqrt(n)
    bool is_orthonormal;    // Is full Q block orthonormal?
    int64_t max_orth_cols;  // Maximum orthonormal prefix
    long time;              // Total computation time (microseconds)
    long peak_rss_kb;       // Peak RSS increase during algorithm call (KB)
};

template <typename T>
struct scaling_result {
    int64_t m;                // Number of rows
    int64_t n;                // Number of columns
    T cond_num;               // Target condition number
    T density;                // Sparse matrix density (computed from actual nnz)
    T aspect_ratio;           // m / n
    int64_t run_idx;          // Run index (0-based)

    alg_quality<T> cqrrt;
    alg_quality<T> cholqr;
    alg_quality<T> scholqr3;
    alg_quality<T> dense_cqrrt;

    // CQRRT subroutine times (from fastest run)
    long cqrrt_alloc_time;
    long cqrrt_saso_time;
    long cqrrt_qr_time;
    long cqrrt_trtri_time;
    long cqrrt_linop_precond_time;
    long cqrrt_linop_gram_time;
    long cqrrt_trmm_gram_time;
    long cqrrt_potrf_time;
    long cqrrt_finalize_time;
    long cqrrt_rest_time;

    // CholQR subroutine times
    long cholqr_alloc_time;
    long cholqr_materialize_time;
    long cholqr_gram_time;
    long cholqr_potrf_time;
    long cholqr_rest_time;

    // sCholQR3 subroutine times
    long scholqr3_alloc_time;
    long scholqr3_materialize_time;
    long scholqr3_gram1_time;
    long scholqr3_potrf1_time;
    long scholqr3_trsm1_time;
    long scholqr3_syrk2_time;
    long scholqr3_potrf2_time;
    long scholqr3_update2_time;
    long scholqr3_syrk3_time;
    long scholqr3_potrf3_time;
    long scholqr3_update3_time;
    long scholqr3_rest_time;

    // CQRRT_expl subroutine times
    long dense_cqrrt_materialize_time;
    long dense_cqrrt_saso_time;
    long dense_cqrrt_qr_time;
    long dense_cqrrt_precond_time;
    long dense_cqrrt_gram_time;
    long dense_cqrrt_potrf_time;
    long dense_cqrrt_finalize_time;
    long dense_cqrrt_rest_time;

    // Analytical peak working memory (KB)
    long cqrrt_analytical_kb;
    long cholqr_analytical_kb;
    long scholqr3_analytical_kb;
    long dense_cqrrt_analytical_kb;
};

// Compute orthogonality metrics for Q-factor
template <typename T>
static void measure_orthogonality(
    const T* Q, int64_t m, int64_t n,
    T& orth_error, bool& is_orthonormal, int64_t& max_orth_cols) {

    // Compute Q^T Q
    std::vector<T> QtQ(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m,
               1.0, Q, m, Q, m, 0.0, QtQ.data(), n);

    // Compute ||Q^T Q - I||_F / sqrt(n)
    std::vector<T> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    for (int64_t i = 0; i < n * n; ++i) {
        QtQ[i] -= I_ref[i];
    }
    T norm_orth = lapack::lange(Norm::Fro, n, n, QtQ.data(), n);
    orth_error = norm_orth / std::sqrt((T) n);

    // Check if full block is orthonormal
    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
    is_orthonormal = (orth_error <= tol);

    // Find maximum orthonormal prefix
    max_orth_cols = 0;
    for (int64_t k = 1; k <= n; ++k) {
        std::vector<T> QtQ_k(k * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
                   1.0, Q, m, Q, m, 0.0, QtQ_k.data(), k);

        T error_k = 0.0;
        for (int64_t j = 0; j < k; ++j) {
            for (int64_t i = 0; i < k; ++i) {
                T expected = (i == j) ? 1.0 : 0.0;
                T diff = QtQ_k[i + j * k] - expected;
                error_k += diff * diff;
            }
        }
        error_k = std::sqrt(error_k) / std::sqrt((T) k);

        if (error_k <= tol) {
            max_orth_cols = k;
        } else {
            break;
        }
    }
}

// Compute Q = A * R^{-1} uniformly for all algorithms.
// Materializes A into Q_out, then solves Q * R = A via trsm (avoids forming R^{-1}).
// This is more numerically stable than forming the explicit inverse via trtri or trsm.
// R is NOT destroyed.
template <typename T, typename GLO>
static void compute_Q_from_R(
    GLO& A_op, T* R, int64_t ldr,
    T* Q_out, int64_t m, int64_t n) {
    // Step 1: Materialize A into Q_out: Q_out = A * I
    T* Eye = new T[n * n]();
    RandLAPACK::util::eye(n, n, Eye);
    A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         m, n, n, (T)1.0, Eye, n, (T)0.0, Q_out, m);
    delete[] Eye;
    // Step 2: Solve Q * R = A for Q via trsm (backward stable, no explicit inverse)
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_out, m);
}

// Core algorithm runner: operates on a pre-constructed SparseLinOp.
// Called by both the generate-mode and file-input-mode entry points.
template <typename T, typename RNG>
static std::vector<scaling_result<T>> run_algorithms(
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>>& A_linop,
    int64_t m,
    int64_t n,
    T cond_num,
    T density,
    T d_factor,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    std::vector<RandBLAS::RNGState<RNG>>& run_states) {

    std::vector<scaling_result<T>> results(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        results[r].m = m;
        results[r].n = n;
        results[r].cond_num = cond_num;
        results[r].density = density;
        results[r].aspect_ratio = static_cast<T>(m) / static_cast<T>(n);
        results[r].run_idx = r;
    }

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);

    // Single reusable Q buffer for uniform Q = A * R^{-1} computation across all algorithms
    std::vector<T> Q_uniform(m * n);

    // ============================================================
    // Run CQRRT (preconditioned Cholesky QR) - multiple runs
    // ============================================================
    // Peak RSS measured separately with test_mode=false to exclude Q-factor allocation.
    // With column-blocking, test_mode reallocates A_pre from m*b_eff to m*n for Q.
    {
        // RSS measurement (test_mode=false)
        long cqrrt_peak_rss_kb = 0;
        {
            std::vector<T> R_rss(n * n, 0.0);
            auto state_rss = run_states[0];
            RandLAPACK::CQRRT_linops<T, RNG> CQRRT_rss(false, tol, false);
            CQRRT_rss.nnz = sketch_nnz;
            CQRRT_rss.block_size = block_size;
            RandLAPACK::PeakRSSTracker cqrrt_mem;
            cqrrt_mem.start();
            CQRRT_rss.call(A_linop, R_rss.data(), n, d_factor, state_rss);
            cqrrt_peak_rss_kb = cqrrt_mem.stop();
        }

        for (int64_t run = 0; run < num_runs; ++run) {
            std::vector<T> R_cqrrt(n * n, 0.0);
            auto state_copy = run_states[run];  // Per-run RNG state

            RandLAPACK::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, false);  // timing=true, test_mode=false
            CQRRT_QR.nnz = sketch_nnz;
            CQRRT_QR.block_size = block_size;
            CQRRT_QR.call(A_linop, R_cqrrt.data(), n, d_factor, state_copy);

            results[run].cqrrt.time = CQRRT_QR.times[10];  // total_t_dur
            results[run].cqrrt.peak_rss_kb = cqrrt_peak_rss_kb;
            results[run].cqrrt_alloc_time = CQRRT_QR.times[0];
            results[run].cqrrt_saso_time = CQRRT_QR.times[1];
            results[run].cqrrt_qr_time = CQRRT_QR.times[2];
            results[run].cqrrt_trtri_time = CQRRT_QR.times[3];
            results[run].cqrrt_linop_precond_time = CQRRT_QR.times[4];
            results[run].cqrrt_linop_gram_time = CQRRT_QR.times[5];
            results[run].cqrrt_trmm_gram_time = CQRRT_QR.times[6];
            results[run].cqrrt_potrf_time = CQRRT_QR.times[7];
            results[run].cqrrt_finalize_time = CQRRT_QR.times[8];
            results[run].cqrrt_rest_time = CQRRT_QR.times[9];

            // Uniform Q computation for every run: Q = A * R^{-1} via operator
            compute_Q_from_R(A_linop, R_cqrrt.data(), n, Q_uniform.data(), m, n);
            measure_orthogonality(Q_uniform.data(), m, n,
                                 results[run].cqrrt.orth_error,
                                 results[run].cqrrt.is_orthonormal,
                                 results[run].cqrrt.max_orth_cols);
        }
    }

    // ============================================================
    // Run CholQR (unpreconditioned Cholesky QR) - multiple runs
    // ============================================================
    // Peak RSS measured separately with test_mode=false to exclude Q-factor allocation.
    // With column-blocking, test_mode reallocates A_temp from m*b_eff to m*n for Q.
    {
        // RSS measurement (test_mode=false)
        long cholqr_peak_rss_kb = 0;
        {
            std::vector<T> R_rss(n * n, 0.0);
            RandLAPACK::CholQR_linops<T> CholQR_rss(false, tol, false);
            CholQR_rss.block_size = block_size;
            RandLAPACK::PeakRSSTracker cholqr_mem;
            cholqr_mem.start();
            CholQR_rss.call(A_linop, R_rss.data(), n);
            cholqr_peak_rss_kb = cholqr_mem.stop();
        }

        for (int64_t run = 0; run < num_runs; ++run) {
            std::vector<T> R_cholqr(n * n, 0.0);

            RandLAPACK::CholQR_linops<T> CholQR_alg(true, tol, false);  // timing=true, test_mode=false
            CholQR_alg.block_size = block_size;
            CholQR_alg.call(A_linop, R_cholqr.data(), n);

            results[run].cholqr.time = CholQR_alg.times[5];  // total
            results[run].cholqr.peak_rss_kb = cholqr_peak_rss_kb;
            results[run].cholqr_alloc_time = CholQR_alg.times[0];
            results[run].cholqr_materialize_time = CholQR_alg.times[1];
            results[run].cholqr_gram_time = CholQR_alg.times[2];
            results[run].cholqr_potrf_time = CholQR_alg.times[3];
            results[run].cholqr_rest_time = CholQR_alg.times[4];

            // Uniform Q computation for every run
            compute_Q_from_R(A_linop, R_cholqr.data(), n, Q_uniform.data(), m, n);
            measure_orthogonality(Q_uniform.data(), m, n,
                                 results[run].cholqr.orth_error,
                                 results[run].cholqr.is_orthonormal,
                                 results[run].cholqr.max_orth_cols);
        }
    }

    // ============================================================
    // Run sCholQR3 (shifted Cholesky QR with 3 iterations) - multiple runs
    // ============================================================
    {
        for (int64_t run = 0; run < num_runs; ++run) {
            std::vector<T> R_scholqr3(n * n, 0.0);

            RandLAPACK::sCholQR3_linops<T> sCholQR3_alg(true, tol, false);  // timing=true, test_mode=false
            sCholQR3_alg.block_size = block_size;

            RandLAPACK::PeakRSSTracker scholqr3_mem;
            scholqr3_mem.start();
            sCholQR3_alg.call(A_linop, R_scholqr3.data(), n);
            results[run].scholqr3.peak_rss_kb = scholqr3_mem.stop();

            results[run].scholqr3.time = sCholQR3_alg.times[12];  // total
            results[run].scholqr3_alloc_time = sCholQR3_alg.times[0];
            results[run].scholqr3_materialize_time = sCholQR3_alg.times[1];
            results[run].scholqr3_gram1_time = sCholQR3_alg.times[2];
            results[run].scholqr3_potrf1_time = sCholQR3_alg.times[3];
            results[run].scholqr3_trsm1_time = sCholQR3_alg.times[4];
            results[run].scholqr3_syrk2_time = sCholQR3_alg.times[5];
            results[run].scholqr3_potrf2_time = sCholQR3_alg.times[6];
            results[run].scholqr3_update2_time = sCholQR3_alg.times[7];
            results[run].scholqr3_syrk3_time = sCholQR3_alg.times[8];
            results[run].scholqr3_potrf3_time = sCholQR3_alg.times[9];
            results[run].scholqr3_update3_time = sCholQR3_alg.times[10];
            results[run].scholqr3_rest_time = sCholQR3_alg.times[11];

            // Uniform Q computation (same as all other algorithms)
            compute_Q_from_R(A_linop, R_scholqr3.data(), n, Q_uniform.data(), m, n);
            measure_orthogonality(Q_uniform.data(), m, n,
                                 results[run].scholqr3.orth_error,
                                 results[run].scholqr3.is_orthonormal,
                                 results[run].scholqr3.max_orth_cols);
        }
    }

    // ============================================================
    // Run CQRRT_expl (materialize operator, then call rl_cqrrt) - multiple runs
    // ============================================================
    // Peak RSS with compute_Q=true is correct: Q overwrites A_materialized in-place (no extra allocation).
    {
        for (int64_t run = 0; run < num_runs; ++run) {
            RandLAPACK::PeakRSSTracker dense_mem;
            dense_mem.start();

            // Step 1: Materialize the operator by multiplying with identity
            T* I_mat = new T[n * n]();
            RandLAPACK::util::eye(n, n, I_mat);
            T* A_materialized = new T[m * n]();

            auto materialize_start = steady_clock::now();
            A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    m, n, n, (T)1.0, I_mat, n, (T)0.0, A_materialized, m);
            auto materialize_stop = steady_clock::now();
            long materialize_time = duration_cast<microseconds>(materialize_stop - materialize_start).count();

            delete[] I_mat;

            // Step 2: Call rl_cqrrt with timing, Q-factor disabled (computed uniformly below)
            // Uses same per-run RNG state as CQRRT_linop for fair comparison
            std::vector<T> R_dense(n * n, 0.0);
            auto state_copy = run_states[run];  // Same RNG state as CQRRT_linop's run
            RandLAPACK::CQRRT<T, RNG> dense_alg(true, tol);  // timing=true
            dense_alg.compute_Q = false;
            dense_alg.orthogonalization = false;
            dense_alg.nnz = sketch_nnz;
            dense_alg.call(m, n, A_materialized, m, R_dense.data(), n, d_factor, state_copy);

            results[run].dense_cqrrt.peak_rss_kb = dense_mem.stop();

            delete[] A_materialized;  // No longer needed (Q computed via operator)

            // Total = materialization + algorithm total (Q excluded from algo total)
            results[run].dense_cqrrt.time = materialize_time + dense_alg.times[9];
            results[run].dense_cqrrt_materialize_time = materialize_time;
            results[run].dense_cqrrt_saso_time     = dense_alg.times[0];
            results[run].dense_cqrrt_qr_time       = dense_alg.times[1];
            results[run].dense_cqrrt_precond_time  = dense_alg.times[3];
            results[run].dense_cqrrt_gram_time     = dense_alg.times[4];
            results[run].dense_cqrrt_potrf_time    = dense_alg.times[6];
            results[run].dense_cqrrt_finalize_time = dense_alg.times[7];
            results[run].dense_cqrrt_rest_time     = dense_alg.times[8];

            // Uniform Q computation for every run
            compute_Q_from_R(A_linop, R_dense.data(), n, Q_uniform.data(), m, n);
            measure_orthogonality(Q_uniform.data(), m, n,
                                 results[run].dense_cqrrt.orth_error,
                                 results[run].dense_cqrrt.is_orthonormal,
                                 results[run].dense_cqrrt.max_orth_cols);
        }
    }

    // Compute analytical peak working memory for each algorithm (same for all runs)
    long cqrrt_akb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
    long cholqr_akb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
    long scholqr3_akb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
    long dense_cqrrt_akb = RandLAPACK::dense_cqrrt_analytical_kb<T>(m, n, d_factor);
    for (int64_t r = 0; r < num_runs; ++r) {
        results[r].cqrrt_analytical_kb = cqrrt_akb;
        results[r].cholqr_analytical_kb = cholqr_akb;
        results[r].scholqr3_analytical_kb = scholqr3_akb;
        results[r].dense_cqrrt_analytical_kb = dense_cqrrt_akb;
    }

    return results;
}

// Generate-mode entry point: generates a synthetic sparse matrix, then runs algorithms.
template <typename T, typename RNG>
static std::vector<scaling_result<T>> run_single_test(
    int64_t m,
    int64_t n,
    T cond_num,
    T density,
    T d_factor,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    RandBLAS::RNGState<RNG>& state) {

    // Pre-generate per-run RNG states
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = state;
        if (r > 0) run_states[r].key.incr(r);
    }

    // Generate sparse matrix A: m × n with controlled condition number.
    auto A_coo = RandLAPACK::gen::gen_sparse_cond_mat<T>(m, n, cond_num, state, density);
    T actual_density = static_cast<T>(A_coo.nnz) / (static_cast<T>(m) * n);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    return run_algorithms(A_linop, m, n, cond_num, actual_density, d_factor,
                          block_size, sketch_nnz, num_runs, run_states);
}

// Compute the 2-norm condition number of a sparse linear operator by
// materializing it and computing singular values via LAPACK gesvd.
// For tall-skinny matrices (m >> n), only n singular values are computed.
template <typename T, typename SpLinOp>
static T compute_condition_number(SpLinOp& A_linop, int64_t m, int64_t n) {
    // Materialize A into dense column-major storage
    std::vector<T> A_dense(m * n, 0.0);
    T* Eye = new T[n * n]();
    RandLAPACK::util::eye(n, n, Eye);
    A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, n, n, (T)1.0, Eye, n, (T)0.0, A_dense.data(), m);
    delete[] Eye;

    // Compute singular values only (no U or V) via divide-and-conquer
    std::vector<T> sigma(n);
    lapack::gesdd(lapack::Job::NoVec,
                  m, n, A_dense.data(), m, sigma.data(),
                  nullptr, 1, nullptr, 1);

    T cond = sigma[0] / sigma[n - 1];
    printf("  Condition number: %.6e (sigma_max=%.6e, sigma_min=%.6e)\n",
           (double)cond, (double)sigma[0], (double)sigma[n - 1]);
    return cond;
}

// File-input entry point: loads a Matrix Market file, then runs algorithms.
template <typename T, typename RNG>
static std::vector<scaling_result<T>> run_single_test_from_file(
    const std::string& filename,
    T d_factor,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    bool compute_cond,
    RandBLAS::RNGState<RNG>& state) {

    // Pre-generate per-run RNG states
    std::vector<RandBLAS::RNGState<RNG>> run_states(num_runs);
    for (int64_t r = 0; r < num_runs; ++r) {
        run_states[r] = state;
        if (r > 0) run_states[r].key.incr(r);
    }

    // Load matrix from Matrix Market file
    auto A_coo = RandLAPACK_demos::coo_from_matrix_market<T>(filename);
    int64_t m = A_coo.n_rows;
    int64_t n = A_coo.n_cols;
    T actual_density = static_cast<T>(A_coo.nnz) / (static_cast<T>(m) * n);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    T cond_num = std::numeric_limits<T>::quiet_NaN();
    if (compute_cond) {
        printf("Computing condition number via SVD (%ld x %ld)...\n", m, n);
        cond_num = compute_condition_number<T>(A_linop, m, n);
    }

    return run_algorithms(A_linop, m, n, cond_num, actual_density, d_factor,
                          block_size, sketch_nnz, num_runs, run_states);
}

// Forward declarations for shared helpers (defined below run_benchmark)
template <typename T>
static void write_results_to_csv(
    const std::vector<scaling_result<T>>& all_runs, int64_t num_runs,
    std::ofstream& out, std::ofstream& breakdown);
template <typename T>
static void print_console_summary(
    const std::vector<scaling_result<T>>& all_runs, int64_t num_runs, int64_t n);
static void write_csv_headers(
    std::ofstream& out, std::ofstream& breakdown,
    const std::string& precision, double d_factor,
    int64_t sketch_nnz, int64_t block_size, int64_t num_runs, int num_threads,
    const std::string& extra_comment);
static void prepend_runtime(const std::string& filepath, double seconds);

template <typename T>
static int run_benchmark(int argc, char *argv[]) {

    if (argc < 11 || argc > 13) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <num_sizes> <num_runs> <m_start> <m_end> <aspect_ratio> <cond_num> <density> <d_factor> [sketch_nnz] [block_size]"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  precision        : 'double' or 'float'" << std::endl;
        std::cerr << "  output_dir       : Directory to write output files" << std::endl;
        std::cerr << "  num_sizes        : Number of matrix sizes to test" << std::endl;
        std::cerr << "  num_runs         : Number of runs per matrix size (for timing)" << std::endl;
        std::cerr << "  m_start          : Starting number of rows (smallest matrix)" << std::endl;
        std::cerr << "  m_end            : Ending number of rows (largest matrix)" << std::endl;
        std::cerr << "  aspect_ratio     : Ratio m/n (e.g., 20 means n = m/20)" << std::endl;
        std::cerr << "  cond_num         : Target condition number for the sparse matrix (e.g., 1e4)" << std::endl;
        std::cerr << "  density          : Target density (e.g., 0.1); bandwidth derived as round(density*n - 1)" << std::endl;
        std::cerr << "  d_factor         : Sketching dimension factor for CQRRT_linop (e.g., 2.0)" << std::endl;
        std::cerr << "  sketch_nnz       : (Optional) Nonzeros per column in SASO sketch (default: 4)" << std::endl;
        std::cerr << "  block_size       : (Optional) Column-block size for CQRRT_linop/CholQR/sCholQR3 Gram (0 = full, default: 0)" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 30 3 1000 30000 100 1e9 0.05 2.0 4 100" << std::endl;
        std::cerr << "  (Tests 30 matrices from 1000x10 to 30000x300, aspect ratio 100:1, κ=1e9, density≈0.05, 3 runs each)" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string precision = argv[1];
    std::string output_dir = argv[2];
    int64_t num_sizes = std::stol(argv[3]);
    int64_t num_runs = std::stol(argv[4]);
    int64_t m_start = std::stol(argv[5]);
    int64_t m_end = std::stol(argv[6]);
    double aspect_ratio = std::stod(argv[7]);
    double cond_num = std::stod(argv[8]);
    double density = std::stod(argv[9]);
    double d_factor = std::stod(argv[10]);
    // Default sketch_nnz=4: the Givens-based matrix generator produces
    // high-coherence matrices (non-uniform leverage scores), so nnz >= 4
    // is needed for reliable SASO sketching (nnz=2 causes sporadic spikes).
    int64_t sketch_nnz = (argc >= 12) ? std::stol(argv[11]) : 4;
    int64_t block_size = (argc >= 13) ? std::stol(argv[12]) : 0;

    auto benchmark_start = steady_clock::now();

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    // Build list of (m, n) pairs to test
    std::vector<std::pair<int64_t, int64_t>> sizes;
    for (int64_t i = 0; i < num_sizes; ++i) {
        // Linear interpolation from m_start to m_end
        int64_t m = (num_sizes == 1) ? m_start : m_start + (m_end - m_start) * i / (num_sizes - 1);
        int64_t n = static_cast<int64_t>(m / aspect_ratio);
        if (n < 1) n = 1;  // Ensure at least 1 column
        sizes.push_back({m, n});
    }

    // Get OpenMP thread count
    int num_threads = omp_get_max_threads();

    printf("\n=== CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl Scaling Study ===\n");
    printf("Precision: %s\n", precision.c_str());
    printf("Fixed aspect ratio: %.1f:1 (m/n)\n", aspect_ratio);
    printf("Matrix sizes: %ld x %ld to %ld x %ld\n",
           sizes.front().first, sizes.front().second,
           sizes.back().first, sizes.back().second);
    printf("Number of test sizes: %zu\n", sizes.size());
    printf("Condition number: %.2e\n", cond_num);
    printf("Target density: %.3f\n", density);
    printf("d_factor (CQRRT_linop): %.2f\n", d_factor);
    printf("Sketch nnz (CQRRT_linop): %ld\n", sketch_nnz);
    printf("Block size (CQRRT_linop, CholQR, sCholQR3): %ld (0 = full)\n", block_size);
    printf("Runs per size: %ld\n", num_runs);
    printf("OpenMP threads: %d\n", num_threads);
    printf("=====================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Prepare output files with date/time prefix
    std::string output_file = output_dir + "/" + date_prefix + "scaling_results.csv";
    std::string breakdown_file = output_dir + "/" + date_prefix + "scaling_breakdown.csv";
    std::ofstream out(output_file);
    std::ofstream breakdown(breakdown_file);

    std::ostringstream extra;
    extra << "# Fixed aspect ratio: " << aspect_ratio << ":1\n"
          << "# Condition number: " << cond_num << "\n"
          << "# Target density: " << density << "\n";
    write_csv_headers(out, breakdown, precision, d_factor,
                      sketch_nnz, block_size, num_runs, num_threads, extra.str());

    // Warmup run to trigger library initialization (MKL thread pools, memory allocators, etc.)
    // This ensures first reported iteration has accurate memory measurements.
    {
        printf("Performing warmup run (not reported)...\n");
        int64_t warmup_m = sizes[0].first;
        int64_t warmup_n = sizes[0].second;
        auto warmup_state = state;  // Use copy to not affect main RNG sequence
        run_single_test<T>(warmup_m, warmup_n, (T)cond_num, (T)density, (T)d_factor, block_size, sketch_nnz, 1, warmup_state);
        printf("Warmup complete, starting measurements.\n\n");
    }

    // Run scaling study
    for (size_t i = 0; i < sizes.size(); ++i) {
        int64_t m = sizes[i].first;
        int64_t n = sizes[i].second;
        printf("Testing %ld x %ld (aspect ratio %.1f) [%zu/%zu]...\n",
               m, n, static_cast<double>(m) / n, i + 1, sizes.size());

        auto all_runs = run_single_test<T>(m, n, (T)cond_num, (T)density, (T)d_factor, block_size, sketch_nnz, num_runs, state);

        print_console_summary(all_runs, num_runs, n);
        write_results_to_csv(all_runs, num_runs, out, breakdown);
    }

    out.close();
    breakdown.close();

    auto benchmark_end = steady_clock::now();
    double total_runtime_s = duration_cast<microseconds>(benchmark_end - benchmark_start).count() / 1e6;

    prepend_runtime(output_file, total_runtime_s);
    prepend_runtime(breakdown_file, total_runtime_s);

    printf("========================================\n");
    printf("Scaling study complete! (%.1f seconds)\n", total_runtime_s);
    printf("Results saved to: %s\n", output_file.c_str());
    printf("Runtime breakdown saved to: %s\n", breakdown_file.c_str());
    printf("========================================\n");

    return 0;
}

// Write a vector of scaling_results to the two CSV files (results + breakdown).
// Shared by both generate and file-input modes.
template <typename T>
static void write_results_to_csv(
    const std::vector<scaling_result<T>>& all_runs,
    int64_t num_runs,
    std::ofstream& out,
    std::ofstream& breakdown) {

    for (int64_t run = 0; run < num_runs; ++run) {
        const auto& result = all_runs[run];

        out << std::fixed << std::setprecision(1)
            << result.m << "," << result.n << "," << run << "," << result.aspect_ratio << ","
            << std::scientific << std::setprecision(6) << result.cond_num << ","
            << std::fixed << std::setprecision(6) << result.density << ","
            << std::scientific << std::setprecision(6)
            << result.cqrrt.orth_error << ","
            << result.cqrrt.max_orth_cols << "," << (result.cqrrt.is_orthonormal ? 1 : 0) << ","
            << result.cqrrt.time << ","
            << result.cholqr.orth_error << ","
            << result.cholqr.max_orth_cols << "," << (result.cholqr.is_orthonormal ? 1 : 0) << ","
            << result.cholqr.time << ","
            << result.scholqr3.orth_error << ","
            << result.scholqr3.max_orth_cols << "," << (result.scholqr3.is_orthonormal ? 1 : 0) << ","
            << result.scholqr3.time << ","
            << result.dense_cqrrt.orth_error << ","
            << result.dense_cqrrt.max_orth_cols << "," << (result.dense_cqrrt.is_orthonormal ? 1 : 0) << ","
            << result.dense_cqrrt.time << ","
            << result.cqrrt.peak_rss_kb << "," << result.cqrrt_analytical_kb << ","
            << result.cholqr.peak_rss_kb << "," << result.cholqr_analytical_kb << ","
            << result.scholqr3.peak_rss_kb << "," << result.scholqr3_analytical_kb << ","
            << result.dense_cqrrt.peak_rss_kb << "," << result.dense_cqrrt_analytical_kb << "\n";

        breakdown << result.m << "," << result.n << "," << run << ","
                  // CQRRT (11 values)
                  << result.cqrrt_alloc_time << "," << result.cqrrt_saso_time << "," << result.cqrrt_qr_time << ","
                  << result.cqrrt_trtri_time << "," << result.cqrrt_linop_precond_time << ","
                  << result.cqrrt_linop_gram_time << "," << result.cqrrt_trmm_gram_time << ","
                  << result.cqrrt_potrf_time << "," << result.cqrrt_finalize_time << ","
                  << result.cqrrt_rest_time << "," << result.cqrrt.time << ","
                  // CholQR (6 values)
                  << result.cholqr_alloc_time << "," << result.cholqr_materialize_time << "," << result.cholqr_gram_time << ","
                  << result.cholqr_potrf_time << "," << result.cholqr_rest_time << ","
                  << result.cholqr.time << ","
                  // sCholQR3 (13 values)
                  << result.scholqr3_alloc_time << "," << result.scholqr3_materialize_time << "," << result.scholqr3_gram1_time << ","
                  << result.scholqr3_potrf1_time << "," << result.scholqr3_trsm1_time << ","
                  << result.scholqr3_syrk2_time << "," << result.scholqr3_potrf2_time << ","
                  << result.scholqr3_update2_time << "," << result.scholqr3_syrk3_time << ","
                  << result.scholqr3_potrf3_time << "," << result.scholqr3_update3_time << ","
                  << result.scholqr3_rest_time << "," << result.scholqr3.time << ","
                  // CQRRT_expl (11 values)
                  << result.dense_cqrrt_materialize_time << ","
                  << result.dense_cqrrt_saso_time << ","
                  << result.dense_cqrrt_qr_time << ","
                  << 0 << ","  // trtri (always 0 for dense)
                  << result.dense_cqrrt_precond_time << ","
                  << result.dense_cqrrt_gram_time << ","
                  << 0 << ","  // trmm_gram (always 0 for dense)
                  << result.dense_cqrrt_potrf_time << ","
                  << result.dense_cqrrt_finalize_time << ","
                  << result.dense_cqrrt_rest_time << ","
                  << result.dense_cqrrt.time << ","
                  // Memory columns (KB)
                  << result.cqrrt.peak_rss_kb << "," << result.cqrrt_analytical_kb << ","
                  << result.cholqr.peak_rss_kb << "," << result.cholqr_analytical_kb << ","
                  << result.scholqr3.peak_rss_kb << "," << result.scholqr3_analytical_kb << ","
                  << result.dense_cqrrt.peak_rss_kb << "," << result.dense_cqrrt_analytical_kb << "\n";
    }
    out.flush();
    breakdown.flush();
}

// Print console summary for a single size's results
template <typename T>
static void print_console_summary(
    const std::vector<scaling_result<T>>& all_runs,
    int64_t num_runs, int64_t n) {

    int64_t best_cqrrt = 0, best_cholqr = 0, best_scholqr3 = 0, best_dense = 0;
    for (int64_t r = 1; r < num_runs; ++r) {
        if (all_runs[r].cqrrt.time < all_runs[best_cqrrt].cqrrt.time) best_cqrrt = r;
        if (all_runs[r].cholqr.time < all_runs[best_cholqr].cholqr.time) best_cholqr = r;
        if (all_runs[r].scholqr3.time < all_runs[best_scholqr3].scholqr3.time) best_scholqr3 = r;
        if (all_runs[r].dense_cqrrt.time < all_runs[best_dense].dense_cqrrt.time) best_dense = r;
    }
    const auto& bc = all_runs[best_cqrrt];
    const auto& bq = all_runs[best_cholqr];
    const auto& bs = all_runs[best_scholqr3];
    const auto& bd = all_runs[best_dense];

    printf("  CQRRT_linop: orth_err=%.2e, max_orth=%ld/%ld, time=%ld us (run %ld)\n",
           bc.cqrrt.orth_error, bc.cqrrt.max_orth_cols, n, bc.cqrrt.time, best_cqrrt);
    printf("  CholQR:      orth_err=%.2e, max_orth=%ld/%ld, time=%ld us (run %ld)\n",
           bq.cholqr.orth_error, bq.cholqr.max_orth_cols, n, bq.cholqr.time, best_cholqr);
    printf("  sCholQR3:    orth_err=%.2e, max_orth=%ld/%ld, time=%ld us (run %ld)\n",
           bs.scholqr3.orth_error, bs.scholqr3.max_orth_cols, n, bs.scholqr3.time, best_scholqr3);
    printf("  CQRRT_expl:  orth_err=%.2e, max_orth=%ld/%ld, time=%ld us (run %ld)\n",
           bd.dense_cqrrt.orth_error, bd.dense_cqrrt.max_orth_cols, n, bd.dense_cqrrt.time, best_dense);
    printf("  Memory (peak RSS / analytical KB):\n");
    printf("    CQRRT_linop: %ld / %ld,  CholQR: %ld / %ld,  sCholQR3: %ld / %ld,  CQRRT_expl: %ld / %ld\n\n",
           bc.cqrrt.peak_rss_kb, bc.cqrrt_analytical_kb,
           bq.cholqr.peak_rss_kb, bq.cholqr_analytical_kb,
           bs.scholqr3.peak_rss_kb, bs.scholqr3_analytical_kb,
           bd.dense_cqrrt.peak_rss_kb, bd.dense_cqrrt_analytical_kb);
}

// Write CSV headers shared by both modes
static void write_csv_headers(
    std::ofstream& out, std::ofstream& breakdown,
    const std::string& precision, double d_factor,
    int64_t sketch_nnz, int64_t block_size, int64_t num_runs, int num_threads,
    const std::string& extra_comment) {

    out << "# CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl Results\n";
    out << "# Precision: " << precision << "\n";
    if (!extra_comment.empty()) out << extra_comment;
    out << "# d_factor (CQRRT_linop only): " << d_factor << "\n";
    out << "# sketch_nnz (CQRRT_linop only): " << sketch_nnz << "\n";
    out << "# block_size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# OpenMP threads: " << num_threads << "\n";
    out << "# Format: per-run per-algorithm quality metrics (orth_error, max_orth_cols, orth_flag, time), memory (KB)\n";
    out << "m,n,run,aspect_ratio,cond_num,density,"
        << "cqrrt_orth_error,cqrrt_max_orth_cols,cqrrt_is_orth,cqrrt_time_us,"
        << "cholqr_orth_error,cholqr_max_orth_cols,cholqr_is_orth,cholqr_time_us,"
        << "scholqr3_orth_error,scholqr3_max_orth_cols,scholqr3_is_orth,scholqr3_time_us,"
        << "dense_cqrrt_orth_error,dense_cqrrt_max_orth_cols,dense_cqrrt_is_orth,dense_cqrrt_time_us,"
        << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
        << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
        << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
        << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb\n";

    breakdown << "# Runtime Breakdown for All Algorithms\n";
    breakdown << "# Precision: " << precision << "\n";
    if (!extra_comment.empty()) breakdown << extra_comment;
    breakdown << "# d_factor (CQRRT_linop only): " << d_factor << "\n";
    breakdown << "# sketch_nnz (CQRRT_linop only): " << sketch_nnz << "\n";
    breakdown << "# block_size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    breakdown << "# num_runs: " << num_runs << "\n";
    breakdown << "# OpenMP threads: " << num_threads << "\n";
    breakdown << "# Times are in microseconds\n";
    breakdown << "# CQRRT_linop: alloc, saso, qr, trtri, linop_precond, linop_gram, trmm_gram, potrf, finalize, rest, total\n";
    breakdown << "# CholQR: alloc, materialize, gram, potrf, rest, total\n";
    breakdown << "# sCholQR3: alloc, materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total\n";
    breakdown << "# CQRRT_expl: materialize, saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total\n";
    breakdown << "m,n,run,"
              << "cqrrt_alloc,cqrrt_saso,cqrrt_qr,cqrrt_trtri,cqrrt_linop_precond,cqrrt_linop_gram,cqrrt_trmm_gram,cqrrt_potrf,cqrrt_finalize,cqrrt_rest,cqrrt_total,"
              << "cholqr_alloc,cholqr_materialize,cholqr_gram,cholqr_potrf,cholqr_rest,cholqr_total,"
              << "scholqr3_alloc,scholqr3_materialize,scholqr3_gram1,scholqr3_potrf1,scholqr3_trsm1,scholqr3_syrk2,scholqr3_potrf2,scholqr3_update2,scholqr3_syrk3,scholqr3_potrf3,scholqr3_update3,scholqr3_rest,scholqr3_total,"
              << "dense_materialize,dense_saso,dense_qr,dense_trtri,dense_precond,dense_gram,dense_trmm_gram,dense_potrf,dense_finalize,dense_rest,dense_total,"
              << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
              << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
              << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
              << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb\n";
}

// Prepend total runtime to a CSV file
static void prepend_runtime(const std::string& filepath, double seconds) {
    std::ifstream fin(filepath);
    std::string content;
    std::string line;
    while (std::getline(fin, line)) {
        content += line + "\n";
    }
    fin.close();
    std::ofstream fout(filepath);
    fout << std::fixed << std::setprecision(1);
    fout << "# Total benchmark runtime: " << seconds << " seconds\n";
    fout << content;
    fout.close();
}

// File-input mode: benchmark a single external Matrix Market matrix
template <typename T>
static int run_benchmark_from_file(int argc, char *argv[]) {
    // Args: <precision> <output_dir> <num_runs> <input_file> <d_factor> [sketch_nnz] [block_size] [compute_cond]
    std::string precision = argv[1];
    std::string output_dir = argv[2];
    int64_t num_runs = std::stol(argv[3]);
    std::string input_file = argv[4];
    double d_factor = std::stod(argv[5]);
    int64_t sketch_nnz = (argc >= 7) ? std::stol(argv[6]) : 4;
    int64_t block_size = (argc >= 8) ? std::stol(argv[7]) : 0;
    bool compute_cond = (argc >= 9) ? (std::stoi(argv[8]) != 0) : false;

    auto benchmark_start = steady_clock::now();

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    int num_threads = omp_get_max_threads();

    printf("\n=== CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl (File Input) ===\n");
    printf("Precision: %s\n", precision.c_str());
    printf("Input file: %s\n", input_file.c_str());
    printf("d_factor (CQRRT_linop): %.2f\n", d_factor);
    printf("Sketch nnz (CQRRT_linop): %ld\n", sketch_nnz);
    printf("Block size (CQRRT_linop, CholQR, sCholQR3): %ld (0 = full)\n", block_size);
    printf("Compute condition number: %s\n", compute_cond ? "yes" : "no");
    printf("Runs: %ld\n", num_runs);
    printf("OpenMP threads: %d\n", num_threads);
    printf("=====================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Extract base name from input file path for output naming
    std::string base_name = input_file;
    auto last_slash = base_name.find_last_of('/');
    if (last_slash != std::string::npos) base_name = base_name.substr(last_slash + 1);
    auto last_dot = base_name.find_last_of('.');
    if (last_dot != std::string::npos) base_name = base_name.substr(0, last_dot);

    std::string output_file = output_dir + "/" + date_prefix + base_name + "_results.csv";
    std::string breakdown_file = output_dir + "/" + date_prefix + base_name + "_breakdown.csv";
    std::ofstream out(output_file);
    std::ofstream breakdown(breakdown_file);

    std::string extra_comment = "# Input file: " + input_file + "\n";
    write_csv_headers(out, breakdown, precision, d_factor,
                      sketch_nnz, block_size, num_runs, num_threads, extra_comment);

    // Warmup run with a small synthetic matrix
    {
        printf("Performing warmup run (not reported)...\n");
        auto warmup_state = state;
        run_single_test<T>(1000, 50, (T)1e4, (T)0.1, (T)d_factor, block_size, sketch_nnz, 1, warmup_state);
        printf("Warmup complete, starting measurements.\n\n");
    }

    // Run benchmark on the file-loaded matrix
    printf("Loading matrix from %s...\n", input_file.c_str());
    auto all_runs = run_single_test_from_file<T>(input_file, (T)d_factor, block_size, sketch_nnz, num_runs, compute_cond, state);

    int64_t m = all_runs[0].m;
    int64_t n = all_runs[0].n;
    printf("Matrix loaded: %ld x %ld, density=%.6f\n", m, n, (double)all_runs[0].density);

    print_console_summary(all_runs, num_runs, n);
    write_results_to_csv(all_runs, num_runs, out, breakdown);

    out.close();
    breakdown.close();

    auto benchmark_end = steady_clock::now();
    double total_runtime_s = duration_cast<microseconds>(benchmark_end - benchmark_start).count() / 1e6;

    prepend_runtime(output_file, total_runtime_s);
    prepend_runtime(breakdown_file, total_runtime_s);

    printf("========================================\n");
    printf("Benchmark complete! (%.1f seconds)\n", total_runtime_s);
    printf("Results saved to: %s\n", output_file.c_str());
    printf("Runtime breakdown saved to: %s\n", breakdown_file.c_str());
    printf("========================================\n");

    return 0;
}

int main(int argc, char *argv[]) {
    // Detect mode from argument count:
    //   File-input mode: 6-9 args (precision, output_dir, num_runs, input_file, d_factor, [sketch_nnz], [block_size], [compute_cond])
    //   Generate mode:  11-13 args (precision, output_dir, num_sizes, num_runs, m_start, m_end, aspect_ratio, cond_num, density, d_factor, [sketch_nnz], [block_size])
    bool is_file_mode = (argc >= 6 && argc <= 9);
    bool is_generate_mode = (argc >= 11 && argc <= 13);

    if (!is_file_mode && !is_generate_mode) {
        std::cerr << "Usage (generate mode):" << std::endl;
        std::cerr << "  " << argv[0]
                  << " <precision> <output_dir> <num_sizes> <num_runs> <m_start> <m_end> <aspect_ratio> <cond_num> <density> <d_factor> [sketch_nnz] [block_size]"
                  << std::endl;
        std::cerr << "\nUsage (file-input mode):" << std::endl;
        std::cerr << "  " << argv[0]
                  << " <precision> <output_dir> <num_runs> <input_file.mtx> <d_factor> [sketch_nnz] [block_size] [compute_cond]"
                  << std::endl;
        std::cerr << "\n  precision    : 'double' or 'float'" << std::endl;
        std::cerr << "  compute_cond : (Optional, file mode only) 1 = compute condition number via SVD (default: 0)" << std::endl;
        std::cerr << "\nExamples:" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 30 3 1000 30000 100 1e9 0.05 2.0 4 100" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 3 ./matrix.mtx 2.0" << std::endl;
        std::cerr << "  " << argv[0] << " double ./output 3 ./matrix.mtx 2.0 4 0 1  # with condition number" << std::endl;
        return 1;
    }

    std::string precision = argv[1];
    if (precision != "double" && precision != "float") {
        std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'" << std::endl;
        return 1;
    }

    if (is_file_mode) {
        if (precision == "double") return run_benchmark_from_file<double>(argc, argv);
        else return run_benchmark_from_file<float>(argc, argv);
    } else {
        if (precision == "double") return run_benchmark<double>(argc, argv);
        else return run_benchmark<float>(argc, argv);
    }
}
#endif
