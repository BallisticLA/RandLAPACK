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
#include <iomanip>
#include <cmath>
#include <ctime>
#include <omp.h>

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
    T rel_error;            // ||A - QR|| / ||A||
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

    // Dense CQRRT subroutine times
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

// Compute ||A - QR|| / ||A|| for factorization quality measurement
template <typename T>
static T compute_factorization_error(
    const T* Q, const T* R, const T* A_ref,
    int64_t m, int64_t n, T norm_A) {
    std::vector<T> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, Q, m, R, n, 0.0, QR.data(), m);
    T norm_diff = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
        T d = A_ref[i] - QR[i];
        norm_diff += d * d;
    }
    return std::sqrt(norm_diff) / norm_A;
}

template <typename T, typename RNG>
static scaling_result<T> run_single_test(
    int64_t m,
    int64_t n,
    T cond_num,
    T density,
    T d_factor,
    bool use_dense_sketch,
    int64_t block_size,
    int64_t sketch_nnz,
    int64_t num_runs,
    RandBLAS::RNGState<RNG>& state) {

    scaling_result<T> result;
    result.m = m;
    result.n = n;
    result.cond_num = cond_num;
    result.aspect_ratio = static_cast<T>(m) / static_cast<T>(n);

    // Compute bandwidth from target density: density ≈ (bandwidth + 1) / n
    int64_t bandwidth = std::max((int64_t)1, std::min(n - 1, (int64_t)std::round(density * n - 1)));

    // Generate sparse matrix A: m × n with controlled condition number
    auto A_coo = RandLAPACK::gen::gen_sparse_cond_mat<T>(m, n, cond_num, state, bandwidth);
    result.density = static_cast<T>(A_coo.nnz) / (static_cast<T>(m) * n);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    // Compute dense representation for verification
    std::vector<T> A_dense(m * n, 0.0);
    RandLAPACK::util::sparse_to_dense(A_csr, Layout::ColMajor, A_dense.data());

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);
    T norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // ============================================================
    // Run CQRRT (preconditioned Cholesky QR) - multiple runs
    // ============================================================
    // Peak RSS measured separately with test_mode=false to exclude Q-factor allocation.
    // With column-blocking, test_mode reallocates A_pre from m*b_eff to m*n for Q.
    {
        // RSS measurement (test_mode=false)
        {
            std::vector<T> R_rss(n * n, 0.0);
            auto state_rss = state;
            RandLAPACK::CQRRT_linops<T, RNG> CQRRT_rss(false, tol, false);
            CQRRT_rss.nnz = sketch_nnz;
            CQRRT_rss.use_dense_sketch = use_dense_sketch;
            CQRRT_rss.block_size = block_size;
            RandLAPACK::PeakRSSTracker cqrrt_mem;
            cqrrt_mem.start();
            CQRRT_rss.call(A_linop, R_rss.data(), n, d_factor, state_rss);
            result.cqrrt.peak_rss_kb = cqrrt_mem.stop();
        }

        // Initialize with first run
        result.cqrrt.time = std::numeric_limits<long>::max();
        result.cqrrt_alloc_time = 0;
        result.cqrrt_saso_time = 0;
        result.cqrrt_qr_time = 0;
        result.cqrrt_trtri_time = 0;
        result.cqrrt_linop_precond_time = 0;
        result.cqrrt_linop_gram_time = 0;
        result.cqrrt_trmm_gram_time = 0;
        result.cqrrt_potrf_time = 0;
        result.cqrrt_finalize_time = 0;
        result.cqrrt_rest_time = 0;

        T best_rel_error = 0.0;
        T best_orth_error = 0.0;
        bool best_is_orthonormal = false;
        int64_t best_max_orth_cols = 0;

        for (int64_t run = 0; run < num_runs; ++run) {
            std::vector<T> R_cqrrt(n * n, 0.0);
            auto state_copy = state;

            RandLAPACK::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
            CQRRT_QR.nnz = sketch_nnz;
            CQRRT_QR.use_dense_sketch = use_dense_sketch;
            CQRRT_QR.block_size = block_size;
            CQRRT_QR.call(A_linop, R_cqrrt.data(), n, d_factor, state_copy);

            long run_time = CQRRT_QR.times[10];  // total_t_dur

            // Track fastest run and its subroutine times
            if (run_time < result.cqrrt.time) {
                result.cqrrt.time = run_time;
                result.cqrrt_alloc_time = CQRRT_QR.times[0];
                result.cqrrt_saso_time = CQRRT_QR.times[1];
                result.cqrrt_qr_time = CQRRT_QR.times[2];
                result.cqrrt_trtri_time = CQRRT_QR.times[3];
                result.cqrrt_linop_precond_time = CQRRT_QR.times[4];
                result.cqrrt_linop_gram_time = CQRRT_QR.times[5];
                result.cqrrt_trmm_gram_time = CQRRT_QR.times[6];
                result.cqrrt_potrf_time = CQRRT_QR.times[7];
                result.cqrrt_finalize_time = CQRRT_QR.times[8];
                result.cqrrt_rest_time = CQRRT_QR.times[9];

                best_rel_error = compute_factorization_error(
                    CQRRT_QR.Q, R_cqrrt.data(), A_dense.data(), m, n, norm_A);

                // Measure orthogonality
                measure_orthogonality(CQRRT_QR.Q, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }
        }

        result.cqrrt.rel_error = best_rel_error;
        result.cqrrt.orth_error = best_orth_error;
        result.cqrrt.is_orthonormal = best_is_orthonormal;
        result.cqrrt.max_orth_cols = best_max_orth_cols;
    }

    // ============================================================
    // Run CholQR (unpreconditioned Cholesky QR) - multiple runs
    // ============================================================
    // Peak RSS measured separately with test_mode=false to exclude Q-factor allocation.
    // With column-blocking, test_mode reallocates A_temp from m*b_eff to m*n for Q.
    {
        // RSS measurement (test_mode=false)
        {
            std::vector<T> R_rss(n * n, 0.0);
            RandLAPACK::CholQR_linops<T> CholQR_rss(false, tol, false);
            CholQR_rss.block_size = block_size;
            RandLAPACK::PeakRSSTracker cholqr_mem;
            cholqr_mem.start();
            CholQR_rss.call(A_linop, R_rss.data(), n);
            result.cholqr.peak_rss_kb = cholqr_mem.stop();
        }

        // Initialize with first run
        result.cholqr.time = std::numeric_limits<long>::max();
        result.cholqr_alloc_time = 0;
        result.cholqr_materialize_time = 0;
        result.cholqr_gram_time = 0;
        result.cholqr_potrf_time = 0;
        result.cholqr_rest_time = 0;

        T best_rel_error = 0.0;
        T best_orth_error = 0.0;
        bool best_is_orthonormal = false;
        int64_t best_max_orth_cols = 0;

        for (int64_t run = 0; run < num_runs; ++run) {
            std::vector<T> R_cholqr(n * n, 0.0);

            RandLAPACK::CholQR_linops<T> CholQR_alg(true, tol, true);  // timing=true, test_mode=true
            CholQR_alg.block_size = block_size;
            CholQR_alg.call(A_linop, R_cholqr.data(), n);

            long run_time = CholQR_alg.times[5];  // total

            if (run_time < result.cholqr.time) {
                result.cholqr.time = run_time;
                result.cholqr_alloc_time = CholQR_alg.times[0];
                result.cholqr_materialize_time = CholQR_alg.times[1];
                result.cholqr_gram_time = CholQR_alg.times[2];
                result.cholqr_potrf_time = CholQR_alg.times[3];
                result.cholqr_rest_time = CholQR_alg.times[4];

                best_rel_error = compute_factorization_error(
                    CholQR_alg.Q, R_cholqr.data(), A_dense.data(), m, n, norm_A);

                // Measure orthogonality
                measure_orthogonality(CholQR_alg.Q, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }
        }

        result.cholqr.rel_error = best_rel_error;
        result.cholqr.orth_error = best_orth_error;
        result.cholqr.is_orthonormal = best_is_orthonormal;
        result.cholqr.max_orth_cols = best_max_orth_cols;
    }

    // ============================================================
    // Run sCholQR3 (shifted Cholesky QR with 3 iterations) - multiple runs
    // ============================================================
    // Peak RSS with test_mode=true is correct: Q reuses Q_buf working buffer (always allocated).
    {
        result.scholqr3.time = std::numeric_limits<long>::max();
        result.scholqr3_alloc_time = 0;
        result.scholqr3_materialize_time = 0;
        result.scholqr3_gram1_time = 0;
        result.scholqr3_potrf1_time = 0;
        result.scholqr3_trsm1_time = 0;
        result.scholqr3_syrk2_time = 0;
        result.scholqr3_potrf2_time = 0;
        result.scholqr3_update2_time = 0;
        result.scholqr3_syrk3_time = 0;
        result.scholqr3_potrf3_time = 0;
        result.scholqr3_update3_time = 0;
        result.scholqr3_rest_time = 0;

        T best_rel_error = 0.0;
        T best_orth_error = 0.0;
        bool best_is_orthonormal = false;
        int64_t best_max_orth_cols = 0;

        for (int64_t run = 0; run < num_runs; ++run) {
            std::vector<T> R_scholqr3(n * n, 0.0);

            RandLAPACK::sCholQR3_linops<T> sCholQR3_alg(true, tol, true);  // timing=true, test_mode=true
            sCholQR3_alg.block_size = block_size;

            RandLAPACK::PeakRSSTracker scholqr3_mem;
            scholqr3_mem.start();
            sCholQR3_alg.call(A_linop, R_scholqr3.data(), n);
            long run_peak_rss_kb = scholqr3_mem.stop();

            long run_time = sCholQR3_alg.times[12];  // total

            if (run_time < result.scholqr3.time) {
                result.scholqr3.time = run_time;
                result.scholqr3.peak_rss_kb = run_peak_rss_kb;
                result.scholqr3_alloc_time = sCholQR3_alg.times[0];
                result.scholqr3_materialize_time = sCholQR3_alg.times[1];
                result.scholqr3_gram1_time = sCholQR3_alg.times[2];
                result.scholqr3_potrf1_time = sCholQR3_alg.times[3];
                result.scholqr3_trsm1_time = sCholQR3_alg.times[4];
                result.scholqr3_syrk2_time = sCholQR3_alg.times[5];
                result.scholqr3_potrf2_time = sCholQR3_alg.times[6];
                result.scholqr3_update2_time = sCholQR3_alg.times[7];
                result.scholqr3_syrk3_time = sCholQR3_alg.times[8];
                result.scholqr3_potrf3_time = sCholQR3_alg.times[9];
                result.scholqr3_update3_time = sCholQR3_alg.times[10];
                result.scholqr3_rest_time = sCholQR3_alg.times[11];

                best_rel_error = compute_factorization_error(
                    sCholQR3_alg.Q, R_scholqr3.data(), A_dense.data(), m, n, norm_A);

                // Measure orthogonality
                measure_orthogonality(sCholQR3_alg.Q, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }
        }

        result.scholqr3.rel_error = best_rel_error;
        result.scholqr3.orth_error = best_orth_error;
        result.scholqr3.is_orthonormal = best_is_orthonormal;
        result.scholqr3.max_orth_cols = best_max_orth_cols;
    }

    // ============================================================
    // Run Dense CQRRT (materialize operator, then call rl_cqrrt) - multiple runs
    // ============================================================
    // Peak RSS with compute_Q=true is correct: Q overwrites A_materialized in-place (no extra allocation).
    {
        result.dense_cqrrt.time = std::numeric_limits<long>::max();
        result.dense_cqrrt_materialize_time = 0;
        result.dense_cqrrt_saso_time = 0;
        result.dense_cqrrt_qr_time = 0;
        result.dense_cqrrt_precond_time = 0;
        result.dense_cqrrt_gram_time = 0;
        result.dense_cqrrt_potrf_time = 0;
        result.dense_cqrrt_finalize_time = 0;
        result.dense_cqrrt_rest_time = 0;

        T best_rel_error = 0.0;
        T best_orth_error = 0.0;
        bool best_is_orthonormal = false;
        int64_t best_max_orth_cols = 0;

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

            // Step 2: Call rl_cqrrt with timing and Q-factor enabled
            std::vector<T> R_dense(n * n, 0.0);
            auto state_copy = state;
            RandLAPACK::CQRRT<T, RNG> dense_alg(true, tol);  // timing=true
            dense_alg.compute_Q = true;
            dense_alg.orthogonalization = false;
            dense_alg.nnz = sketch_nnz;
            dense_alg.call(m, n, A_materialized, m, R_dense.data(), n, d_factor, state_copy);

            long run_peak_rss_kb = dense_mem.stop();

            // Total = materialization + algorithm total (Q excluded from algo total)
            long run_time = materialize_time + dense_alg.times[9];

            if (run_time < result.dense_cqrrt.time) {
                result.dense_cqrrt.time = run_time;
                result.dense_cqrrt.peak_rss_kb = run_peak_rss_kb;
                result.dense_cqrrt_materialize_time = materialize_time;
                result.dense_cqrrt_saso_time     = dense_alg.times[0];
                result.dense_cqrrt_qr_time       = dense_alg.times[1];
                result.dense_cqrrt_precond_time  = dense_alg.times[3];
                result.dense_cqrrt_gram_time     = dense_alg.times[4];
                result.dense_cqrrt_potrf_time    = dense_alg.times[6];
                result.dense_cqrrt_finalize_time = dense_alg.times[7];
                result.dense_cqrrt_rest_time     = dense_alg.times[8];

                // A_materialized now contains Q (overwritten by rl_cqrrt)
                best_rel_error = compute_factorization_error(
                    A_materialized, R_dense.data(), A_dense.data(), m, n, norm_A);

                // Measure orthogonality
                measure_orthogonality(A_materialized, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }

            delete[] A_materialized;
        }

        result.dense_cqrrt.rel_error = best_rel_error;
        result.dense_cqrrt.orth_error = best_orth_error;
        result.dense_cqrrt.is_orthonormal = best_is_orthonormal;
        result.dense_cqrrt.max_orth_cols = best_max_orth_cols;
    }

    // Compute analytical peak working memory for each algorithm
    result.cqrrt_analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
    result.cholqr_analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
    result.scholqr3_analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
    result.dense_cqrrt_analytical_kb = RandLAPACK::dense_cqrrt_analytical_kb<T>(m, n, d_factor);

    return result;
}

int main(int argc, char *argv[]) {

    if (argc < 10 || argc > 13) {
        std::cerr << "Usage: " << argv[0]
                  << " <m_start> <m_end> <num_sizes> <aspect_ratio> <cond_num> <density> <d_factor> <num_runs> <output_dir> [block_size] [sketch_nnz] [use_dense_sketch]"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  m_start          : Starting number of rows (smallest matrix)" << std::endl;
        std::cerr << "  m_end            : Ending number of rows (largest matrix)" << std::endl;
        std::cerr << "  num_sizes        : Number of matrix sizes to test" << std::endl;
        std::cerr << "  aspect_ratio     : Ratio m/n (e.g., 20 means n = m/20)" << std::endl;
        std::cerr << "  cond_num         : Target condition number for the sparse matrix (e.g., 1e4)" << std::endl;
        std::cerr << "  density          : Target density (e.g., 0.1); bandwidth derived as round(density*n - 1)" << std::endl;
        std::cerr << "  d_factor         : Sketching dimension factor for CQRRT_linop (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs         : Number of runs per matrix size (for timing)" << std::endl;
        std::cerr << "  output_dir       : Directory to write output files" << std::endl;
        std::cerr << "  block_size       : (Optional) Column-block size for CQRRT_linop/CholQR/sCholQR3 Gram (0 = full, default: 0)" << std::endl;
        std::cerr << "  sketch_nnz       : (Optional) Nonzeros per column in SASO sketch (default: 5)" << std::endl;
        std::cerr << "  use_dense_sketch : (Optional) 1 = dense Gaussian sketch, 0 = SASO (default: 0)" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " 500 10000 50 20 1e4 0.1 2.0 3 ./output" << std::endl;
        std::cerr << "  (Tests 50 matrices from 500x25 to 10000x500, aspect ratio 20:1, κ=1e4, density≈0.1, 3 runs each)" << std::endl;
        return 1;
    }

    // Parse arguments
    int64_t m_start = std::stol(argv[1]);
    int64_t m_end = std::stol(argv[2]);
    int64_t num_sizes = std::stol(argv[3]);
    double aspect_ratio = std::stod(argv[4]);
    double cond_num = std::stod(argv[5]);
    double density = std::stod(argv[6]);
    double d_factor = std::stod(argv[7]);
    int64_t num_runs = std::stol(argv[8]);
    std::string output_dir = argv[9];
    int64_t block_size = (argc >= 11) ? std::stol(argv[10]) : 0;
    int64_t sketch_nnz = (argc >= 12) ? std::stol(argv[11]) : 5;
    bool use_dense_sketch = (argc >= 13) ? (std::stoi(argv[12]) != 0) : false;

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
    printf("Fixed aspect ratio: %.1f:1 (m/n)\n", aspect_ratio);
    printf("Matrix sizes: %ld x %ld to %ld x %ld\n",
           sizes.front().first, sizes.front().second,
           sizes.back().first, sizes.back().second);
    printf("Number of test sizes: %zu\n", sizes.size());
    printf("Condition number: %.2e\n", cond_num);
    printf("Target density: %.3f\n", density);
    printf("d_factor (CQRRT_linop): %.2f\n", d_factor);
    printf("Sketch type (CQRRT_linop): %s\n", use_dense_sketch ? "dense Gaussian" : "SASO");
    printf("Sketch nnz (CQRRT_linop): %ld\n", sketch_nnz);
    printf("Block size (CQRRT_linop, CholQR, sCholQR3): %ld (0 = full)\n", block_size);
    printf("Runs per size: %ld\n", num_runs);
    printf("OpenMP threads: %d\n", num_threads);
    printf("=====================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Prepare output file with date/time prefix
    std::string output_file = output_dir + "/" + date_prefix + "scaling_results.csv";
    std::ofstream out(output_file);
    out << "# CQRRT_linop vs CholQR vs sCholQR3 vs CQRRT_expl Scaling Study Results\n";
    out << "# Fixed aspect ratio: " << aspect_ratio << ":1\n";
    out << "# Condition number: " << cond_num << "\n";
    out << "# Target density: " << density << "\n";
    out << "# d_factor (CQRRT_linop only): " << d_factor << "\n";
    out << "# sketch_type (CQRRT_linop only): " << (use_dense_sketch ? "dense Gaussian" : "SASO") << "\n";
    out << "# sketch_nnz (CQRRT_linop only): " << sketch_nnz << "\n";
    out << "# block_size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# OpenMP threads: " << num_threads << "\n";
    out << "# Format: per-algorithm quality metrics (rel_error, orth_error, max_orth_cols, orth_flag, time), memory (KB), and speedups\n";
    out << "m,n,aspect_ratio,cond_num,density,"
        << "cqrrt_rel_error,cqrrt_orth_error,cqrrt_max_orth_cols,cqrrt_is_orth,cqrrt_time_us,"
        << "cholqr_rel_error,cholqr_orth_error,cholqr_max_orth_cols,cholqr_is_orth,cholqr_time_us,"
        << "scholqr3_rel_error,scholqr3_orth_error,scholqr3_max_orth_cols,scholqr3_is_orth,scholqr3_time_us,"
        << "dense_cqrrt_rel_error,dense_cqrrt_orth_error,dense_cqrrt_max_orth_cols,dense_cqrrt_is_orth,dense_cqrrt_time_us,"
        << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
        << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
        << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
        << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb,"
        << "speedup_cqrrt_over_cholqr,speedup_cqrrt_over_scholqr3,speedup_cqrrt_over_dense\n";

    // Prepare runtime breakdown file with date/time prefix
    std::string breakdown_file = output_dir + "/" + date_prefix + "scaling_breakdown.csv";
    std::ofstream breakdown(breakdown_file);
    breakdown << "# Runtime Breakdown for All Algorithms (from fastest run per matrix size)\n";
    breakdown << "# Fixed aspect ratio: " << aspect_ratio << ":1\n";
    breakdown << "# Condition number: " << cond_num << "\n";
    breakdown << "# Target density: " << density << "\n";
    breakdown << "# d_factor (CQRRT_linop only): " << d_factor << "\n";
    breakdown << "# sketch_type (CQRRT_linop only): " << (use_dense_sketch ? "dense Gaussian" : "SASO") << "\n";
    breakdown << "# sketch_nnz (CQRRT_linop only): " << sketch_nnz << "\n";
    breakdown << "# block_size (CQRRT_linop, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    breakdown << "# num_runs: " << num_runs << "\n";
    breakdown << "# OpenMP threads: " << num_threads << "\n";
    breakdown << "# Times are in microseconds\n";
    breakdown << "# CQRRT_linop: alloc, saso, qr, trtri, linop_precond, linop_gram, trmm_gram, potrf, finalize, rest, total\n";
    breakdown << "# CholQR: alloc, materialize, gram, potrf, rest, total\n";
    breakdown << "# sCholQR3: alloc, materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total\n";
    breakdown << "# CQRRT_expl: materialize, saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total\n";
    breakdown << "m,n,"
              << "cqrrt_alloc,cqrrt_saso,cqrrt_qr,cqrrt_trtri,cqrrt_linop_precond,cqrrt_linop_gram,cqrrt_trmm_gram,cqrrt_potrf,cqrrt_finalize,cqrrt_rest,cqrrt_total,"
              << "cholqr_alloc,cholqr_materialize,cholqr_gram,cholqr_potrf,cholqr_rest,cholqr_total,"
              << "scholqr3_alloc,scholqr3_materialize,scholqr3_gram1,scholqr3_potrf1,scholqr3_trsm1,scholqr3_syrk2,scholqr3_potrf2,scholqr3_update2,scholqr3_syrk3,scholqr3_potrf3,scholqr3_update3,scholqr3_rest,scholqr3_total,"
              << "dense_materialize,dense_saso,dense_qr,dense_trtri,dense_precond,dense_gram,dense_trmm_gram,dense_potrf,dense_finalize,dense_rest,dense_total,"
              << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
              << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
              << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
              << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb\n";

    // Warmup run to trigger library initialization (MKL thread pools, memory allocators, etc.)
    // This ensures first reported iteration has accurate memory measurements.
    {
        printf("Performing warmup run (not reported)...\n");
        int64_t warmup_m = sizes[0].first;
        int64_t warmup_n = sizes[0].second;
        auto warmup_state = state;  // Use copy to not affect main RNG sequence
        run_single_test<double>(warmup_m, warmup_n, cond_num, density, d_factor, use_dense_sketch, block_size, sketch_nnz, 1, warmup_state);
        printf("Warmup complete, starting measurements.\n\n");
    }

    // Run scaling study
    for (size_t i = 0; i < sizes.size(); ++i) {
        int64_t m = sizes[i].first;
        int64_t n = sizes[i].second;
        printf("Testing %ld x %ld (aspect ratio %.1f) [%zu/%zu]...\n",
               m, n, static_cast<double>(m) / n, i + 1, sizes.size());

        auto result = run_single_test<double>(m, n, cond_num, density, d_factor, use_dense_sketch, block_size, sketch_nnz, num_runs, state);

        double speedup_cqrrt_over_cholqr = (result.cqrrt.time > 0) ?
            static_cast<double>(result.cholqr.time) / result.cqrrt.time : 0.0;
        double speedup_cqrrt_over_scholqr3 = (result.cqrrt.time > 0) ?
            static_cast<double>(result.scholqr3.time) / result.cqrrt.time : 0.0;
        double speedup_cqrrt_over_dense = (result.cqrrt.time > 0) ?
            static_cast<double>(result.dense_cqrrt.time) / result.cqrrt.time : 0.0;

        printf("  CQRRT_linop: orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.cqrrt.orth_error, result.cqrrt.max_orth_cols, n, result.cqrrt.time);
        printf("  CholQR:      orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.cholqr.orth_error, result.cholqr.max_orth_cols, n, result.cholqr.time);
        printf("  sCholQR3:    orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.scholqr3.orth_error, result.scholqr3.max_orth_cols, n, result.scholqr3.time);
        printf("  CQRRT_expl:  orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.dense_cqrrt.orth_error, result.dense_cqrrt.max_orth_cols, n, result.dense_cqrrt.time);
        printf("  Speedup CQRRT_linop over: CholQR=%.2fx, sCholQR3=%.2fx, CQRRT_expl=%.2fx\n",
               speedup_cqrrt_over_cholqr, speedup_cqrrt_over_scholqr3, speedup_cqrrt_over_dense);
        printf("  Memory (peak RSS / analytical KB):\n");
        printf("    CQRRT_linop: %ld / %ld,  CholQR: %ld / %ld,  sCholQR3: %ld / %ld,  CQRRT_expl: %ld / %ld\n\n",
               result.cqrrt.peak_rss_kb, result.cqrrt_analytical_kb,
               result.cholqr.peak_rss_kb, result.cholqr_analytical_kb,
               result.scholqr3.peak_rss_kb, result.scholqr3_analytical_kb,
               result.dense_cqrrt.peak_rss_kb, result.dense_cqrrt_analytical_kb);

        // Write results
        out << std::fixed << std::setprecision(1)
            << result.m << "," << result.n << "," << result.aspect_ratio << ","
            << std::scientific << std::setprecision(6) << result.cond_num << ","
            << std::fixed << std::setprecision(6) << result.density << ","
            << std::scientific << std::setprecision(6)
            << result.cqrrt.rel_error << "," << result.cqrrt.orth_error << ","
            << result.cqrrt.max_orth_cols << "," << (result.cqrrt.is_orthonormal ? 1 : 0) << ","
            << result.cqrrt.time << ","
            << result.cholqr.rel_error << "," << result.cholqr.orth_error << ","
            << result.cholqr.max_orth_cols << "," << (result.cholqr.is_orthonormal ? 1 : 0) << ","
            << result.cholqr.time << ","
            << result.scholqr3.rel_error << "," << result.scholqr3.orth_error << ","
            << result.scholqr3.max_orth_cols << "," << (result.scholqr3.is_orthonormal ? 1 : 0) << ","
            << result.scholqr3.time << ","
            << result.dense_cqrrt.rel_error << "," << result.dense_cqrrt.orth_error << ","
            << result.dense_cqrrt.max_orth_cols << "," << (result.dense_cqrrt.is_orthonormal ? 1 : 0) << ","
            << result.dense_cqrrt.time << ","
            << result.cqrrt.peak_rss_kb << "," << result.cqrrt_analytical_kb << ","
            << result.cholqr.peak_rss_kb << "," << result.cholqr_analytical_kb << ","
            << result.scholqr3.peak_rss_kb << "," << result.scholqr3_analytical_kb << ","
            << result.dense_cqrrt.peak_rss_kb << "," << result.dense_cqrrt_analytical_kb << ","
            << std::fixed << std::setprecision(3) << speedup_cqrrt_over_cholqr << "," << speedup_cqrrt_over_scholqr3 << "," << speedup_cqrrt_over_dense << "\n";
        out.flush();

        // Write runtime breakdown for all algorithms
        breakdown << result.m << "," << result.n << ","
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
                  // Dense CQRRT (11 values)
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
        breakdown.flush();
    }

    out.close();
    breakdown.close();
    printf("========================================\n");
    printf("Scaling study complete!\n");
    printf("Results saved to: %s\n", output_file.c_str());
    printf("Runtime breakdown saved to: %s\n", breakdown_file.c_str());
    printf("========================================\n");

    return 0;
}
#endif
