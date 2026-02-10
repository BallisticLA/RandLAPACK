#if defined(__APPLE__)
int main() {return 0;}
#else

// CQRRT_linops conditioning study - measures Q-factor orthogonality degradation
// as a function of SPD matrix condition number in CholSolver operator
//
// Composite operator: CholSolver * (SASO * Gaussian)
// Varies SPD matrix condition number from well-conditioned to ill-conditioned

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <functional>
#include <ctime>
#include <omp.h>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

// Linops algorithms (now in main RandLAPACK)
#include "rl_cqrrt_linops.hh"
#include "rl_cholqr_linops.hh"
#include "rl_scholqr3_linops.hh"
#include "rl_memory_tracker.hh"

// Demos utilities (Eigen-dependent, stay in demos)
#include "../../demos/functions/linops_external/dm_cholsolver_linop.hh"
#include "../../demos/functions/misc/dm_util.hh"

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
struct conditioning_result {
    T cond_num;           // Condition number of SPD matrix

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

// Compute mean/std/rate statistics for one algorithm's quality metrics across runs
template <typename T>
static void compute_quality_stats(
    const std::vector<conditioning_result<T>>& results,
    std::function<const alg_quality<T>&(const conditioning_result<T>&)> get,
    int64_t num_runs,
    double& rel_err_mean, double& rel_err_std,
    double& orth_err_mean, double& orth_err_std,
    double& max_orth_mean, double& max_orth_std,
    double& orth_rate,
    double& time_mean, double& time_std)
{
    rel_err_mean = orth_err_mean = max_orth_mean = time_mean = 0;
    int orth_count = 0;
    for (const auto& r : results) {
        const auto& q = get(r);
        rel_err_mean += q.rel_error;
        orth_err_mean += q.orth_error;
        max_orth_mean += q.max_orth_cols;
        time_mean += q.time;
        if (q.is_orthonormal) orth_count++;
    }
    rel_err_mean /= num_runs;
    orth_err_mean /= num_runs;
    max_orth_mean /= num_runs;
    time_mean /= num_runs;

    rel_err_std = orth_err_std = max_orth_std = time_std = 0;
    for (const auto& r : results) {
        const auto& q = get(r);
        rel_err_std += (q.rel_error - rel_err_mean) * (q.rel_error - rel_err_mean);
        orth_err_std += (q.orth_error - orth_err_mean) * (q.orth_error - orth_err_mean);
        max_orth_std += (q.max_orth_cols - max_orth_mean) * (q.max_orth_cols - max_orth_mean);
        time_std += (q.time - time_mean) * (q.time - time_mean);
    }
    rel_err_std = std::sqrt(rel_err_std / num_runs);
    orth_err_std = std::sqrt(orth_err_std / num_runs);
    max_orth_std = std::sqrt(max_orth_std / num_runs);
    time_std = std::sqrt(time_std / num_runs);
    orth_rate = static_cast<double>(orth_count) / num_runs;
}

template <typename T, typename RNG>
static conditioning_result<T> run_single_test(
    const std::string& spd_filename,
    T cond_num,
    int64_t m,
    int64_t k_dim,
    int64_t n,
    T d_factor,
    bool use_dense_sketch,
    int64_t block_size,
    RandBLAS::RNGState<RNG>& state) {

    conditioning_result<T> result;
    result.cond_num = cond_num;

    // Create CholSolverLinOp from SPD matrix file
    RandLAPACK_demos::CholSolverLinOp<T> A_inv_linop(spd_filename);

    // Try to factorize - if it fails due to ill-conditioning, return NaN results
    try {
        A_inv_linop.factorize();
    } catch (const std::exception& e) {
        printf("    Cholesky factorization failed for κ=%.6e - matrix too ill-conditioned\n", cond_num);
        result.cqrrt.rel_error = std::numeric_limits<T>::quiet_NaN();
        result.cqrrt.orth_error = std::numeric_limits<T>::quiet_NaN();
        result.cqrrt.is_orthonormal = false;
        result.cqrrt.max_orth_cols = -1;
        result.cqrrt.time = 0;
        result.cholqr.rel_error = std::numeric_limits<T>::quiet_NaN();
        result.cholqr.orth_error = std::numeric_limits<T>::quiet_NaN();
        result.cholqr.is_orthonormal = false;
        result.cholqr.max_orth_cols = -1;
        result.cholqr.time = 0;
        result.scholqr3.rel_error = std::numeric_limits<T>::quiet_NaN();
        result.scholqr3.orth_error = std::numeric_limits<T>::quiet_NaN();
        result.scholqr3.is_orthonormal = false;
        result.scholqr3.max_orth_cols = -1;
        result.scholqr3.time = 0;
        result.dense_cqrrt.rel_error = std::numeric_limits<T>::quiet_NaN();
        result.dense_cqrrt.orth_error = std::numeric_limits<T>::quiet_NaN();
        result.dense_cqrrt.is_orthonormal = false;
        result.dense_cqrrt.max_orth_cols = -1;
        result.dense_cqrrt.time = 0;
        result.dense_cqrrt_materialize_time = 0;
        result.cqrrt.peak_rss_kb = 0;
        result.cholqr.peak_rss_kb = 0;
        result.scholqr3.peak_rss_kb = 0;
        result.dense_cqrrt.peak_rss_kb = 0;
        result.cqrrt_analytical_kb = 0;
        result.cholqr_analytical_kb = 0;
        result.scholqr3_analytical_kb = 0;
        result.dense_cqrrt_analytical_kb = 0;
        return result;
    }

    // Generate SASO (Sparse) matrix: m × k_dim
    // COO is only needed for CSC conversion, so scope-limit it to free memory early.
    RandBLAS::sparse_data::csc::CSCMatrix<T> saso_csc(m, k_dim);
    {
        auto saso_coo = RandLAPACK::gen::gen_sparse_mat<T>(m, k_dim, 0.5, state);
        RandBLAS::sparse_data::conversions::coo_to_csc(saso_coo, saso_csc);
    }
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>> saso_linop(
        m, k_dim, saso_csc);

    // Generate Gaussian matrix: k_dim × n
    std::vector<T> gaussian_mat(k_dim * n);
    RandBLAS::DenseDist gaussian_dist(k_dim, n);
    RandBLAS::fill_dense(gaussian_dist, gaussian_mat.data(), state);
    RandLAPACK::linops::DenseLinOp<T> gaussian_linop(k_dim, n, gaussian_mat.data(), k_dim, Layout::ColMajor);

    // Create composite: CholSolver * (SASO * Gaussian)
    RandLAPACK::linops::CompositeOperator inner_composite(m, n, saso_linop, gaussian_linop);
    RandLAPACK::linops::CompositeOperator outer_composite(m, n, A_inv_linop, inner_composite);

    // Compute dense representation for verification.
    // Use SpMM directly on the CSC (avoids m×k dense copy of SASO).
    // Scope-limit intermediate to free it after A_dense is computed.
    std::vector<T> A_dense(m * n, 0.0);
    {
        std::vector<T> intermediate(m * n, 0.0);
        RandBLAS::sparse_data::left_spmm(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, n, k_dim, (T)1.0, saso_csc, 0, 0,
            gaussian_mat.data(), k_dim, (T)0.0, intermediate.data(), m);
        A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m,
                    (T)1.0, intermediate.data(), m, (T)0.0, A_dense.data(), m);
    }

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);
    T norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // ============================================================
    // Run CQRRT (preconditioned Cholesky QR)
    // ============================================================
    // Peak RSS measured with test_mode=false to exclude Q-factor allocation.
    // With column-blocking, test_mode reallocates A_pre from m*b_eff to m*n for Q.
    {
        std::vector<T> R_rss(n * n, 0.0);
        auto state_rss = state;
        RandLAPACK::CQRRT_linops<T, RNG> CQRRT_rss(false, tol, false);
        CQRRT_rss.nnz = 5;
        CQRRT_rss.use_dense_sketch = use_dense_sketch;
        CQRRT_rss.block_size = block_size;
        RandLAPACK::PeakRSSTracker cqrrt_mem;
        cqrrt_mem.start();
        CQRRT_rss.call(outer_composite, R_rss.data(), n, d_factor, state_rss);
        result.cqrrt.peak_rss_kb = cqrrt_mem.stop();
    }
    {
        std::vector<T> R_cqrrt(n * n, 0.0);

        RandLAPACK::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
        CQRRT_QR.nnz = 5;  // Optimal for sparse SPD matrices (from parameter study)
        CQRRT_QR.use_dense_sketch = use_dense_sketch;
        CQRRT_QR.block_size = block_size;
        CQRRT_QR.call(outer_composite, R_cqrrt.data(), n, d_factor, state);

        result.cqrrt.time = CQRRT_QR.times[10];  // total_t_dur
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

        result.cqrrt.rel_error = compute_factorization_error(
            CQRRT_QR.Q, R_cqrrt.data(), A_dense.data(), m, n, norm_A);

        // Measure orthogonality
        measure_orthogonality(CQRRT_QR.Q, m, n, result.cqrrt.orth_error,
                             result.cqrrt.is_orthonormal, result.cqrrt.max_orth_cols);
    }

    // ============================================================
    // Run CholQR (unpreconditioned Cholesky QR)
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
            CholQR_rss.call(outer_composite, R_rss.data(), n);
            result.cholqr.peak_rss_kb = cholqr_mem.stop();
        }

        std::vector<T> R_cholqr(n * n, 0.0);

        RandLAPACK::CholQR_linops<T> CholQR_alg(true, tol, true);  // timing=true, test_mode=true
        CholQR_alg.block_size = block_size;
        CholQR_alg.call(outer_composite, R_cholqr.data(), n);

        result.cholqr.time = CholQR_alg.times[5];  // total
        result.cholqr_alloc_time = CholQR_alg.times[0];
        result.cholqr_materialize_time = CholQR_alg.times[1];
        result.cholqr_gram_time = CholQR_alg.times[2];
        result.cholqr_potrf_time = CholQR_alg.times[3];
        result.cholqr_rest_time = CholQR_alg.times[4];

        result.cholqr.rel_error = compute_factorization_error(
            CholQR_alg.Q, R_cholqr.data(), A_dense.data(), m, n, norm_A);

        // Measure orthogonality
        measure_orthogonality(CholQR_alg.Q, m, n, result.cholqr.orth_error,
                             result.cholqr.is_orthonormal, result.cholqr.max_orth_cols);
    }

    // ============================================================
    // Run sCholQR3 (shifted Cholesky QR with 3 iterations)
    // ============================================================
    // Peak RSS with test_mode=true is correct: Q reuses Q_buf working buffer (always allocated).
    {
        std::vector<T> R_scholqr3(n * n, 0.0);

        RandLAPACK::sCholQR3_linops<T> sCholQR3_alg(true, tol, true);  // timing=true, test_mode=true
        sCholQR3_alg.block_size = block_size;

        RandLAPACK::PeakRSSTracker scholqr3_mem;
        scholqr3_mem.start();
        sCholQR3_alg.call(outer_composite, R_scholqr3.data(), n);
        result.scholqr3.peak_rss_kb = scholqr3_mem.stop();

        result.scholqr3.time = sCholQR3_alg.times[12];  // total
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

        result.scholqr3.rel_error = compute_factorization_error(
            sCholQR3_alg.Q, R_scholqr3.data(), A_dense.data(), m, n, norm_A);

        // Measure orthogonality
        measure_orthogonality(sCholQR3_alg.Q, m, n, result.scholqr3.orth_error,
                             result.scholqr3.is_orthonormal, result.scholqr3.max_orth_cols);
    }

    // ============================================================
    // Run Dense CQRRT (materialize operator, then call rl_cqrrt)
    // ============================================================
    // Peak RSS with compute_Q=true is correct: Q overwrites A_materialized in-place (no extra allocation).
    {
        RandLAPACK::PeakRSSTracker dense_mem;
        dense_mem.start();

        // Step 1: Materialize the operator by multiplying with identity
        T* I_mat = new T[n * n]();
        RandLAPACK::util::eye(n, n, I_mat);
        T* A_materialized = new T[m * n]();

        auto materialize_start = steady_clock::now();
        outer_composite(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                        m, n, n, (T)1.0, I_mat, n, (T)0.0, A_materialized, m);
        auto materialize_stop = steady_clock::now();
        result.dense_cqrrt_materialize_time = duration_cast<microseconds>(materialize_stop - materialize_start).count();

        delete[] I_mat;

        // Step 2: Call rl_cqrrt with timing and Q-factor enabled
        std::vector<T> R_dense(n * n, 0.0);
        RandLAPACK::CQRRT<T, RNG> dense_alg(true, tol);  // timing=true
        dense_alg.compute_Q = true;
        dense_alg.orthogonalization = false;
        dense_alg.nnz = 2;
        auto state_copy = state;
        dense_alg.call(m, n, A_materialized, m, R_dense.data(), n, d_factor, state_copy);

        result.dense_cqrrt.peak_rss_kb = dense_mem.stop();

        // Extract subroutine times (10-element vector matching CQRRT_linops indices)
        result.dense_cqrrt_saso_time     = dense_alg.times[0];
        result.dense_cqrrt_qr_time       = dense_alg.times[1];
        // times[2] = trtri = 0, times[5] = trmm_gram = 0 (omitted from struct)
        result.dense_cqrrt_precond_time  = dense_alg.times[3];
        result.dense_cqrrt_gram_time     = dense_alg.times[4];
        result.dense_cqrrt_potrf_time    = dense_alg.times[6];
        result.dense_cqrrt_finalize_time = dense_alg.times[7];
        result.dense_cqrrt_rest_time     = dense_alg.times[8];
        // Total = materialization + algorithm total (Q excluded from algo total)
        result.dense_cqrrt.time = result.dense_cqrrt_materialize_time + dense_alg.times[9];

        // A_materialized now contains Q (overwritten by rl_cqrrt when compute_Q=true)
        result.dense_cqrrt.rel_error = compute_factorization_error(
            A_materialized, R_dense.data(), A_dense.data(), m, n, norm_A);

        // Measure orthogonality
        measure_orthogonality(A_materialized, m, n, result.dense_cqrrt.orth_error,
                             result.dense_cqrrt.is_orthonormal, result.dense_cqrrt.max_orth_cols);

        delete[] A_materialized;
    }

    // Compute analytical peak working memory for each algorithm
    result.cqrrt_analytical_kb = RandLAPACK::cqrrt_linops_analytical_kb<T>(m, n, d_factor, block_size);
    result.cholqr_analytical_kb = RandLAPACK::cholqr_linops_analytical_kb<T>(m, n, block_size);
    result.scholqr3_analytical_kb = RandLAPACK::scholqr3_linops_analytical_kb<T>(m, n, block_size);
    result.dense_cqrrt_analytical_kb = RandLAPACK::dense_cqrrt_analytical_kb<T>(m, n, d_factor);

    return result;
}

int main(int argc, char *argv[]) {

    if (argc < 7 || argc > 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <spd_matrix_dir> <k_dim> <n_cols> <d_factor> <num_runs> <output_file> [use_dense_sketch] [block_size]"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  spd_matrix_dir   : Directory containing SPD matrices (with metadata.txt)" << std::endl;
        std::cerr << "  k_dim            : Intermediate dimension (SASO cols / Gaussian rows)" << std::endl;
        std::cerr << "  n_cols           : Final number of columns (Gaussian cols)" << std::endl;
        std::cerr << "  d_factor         : Sketching dimension factor (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs         : Number of runs per condition number (for averaging)" << std::endl;
        std::cerr << "  output_file      : Output CSV file for results" << std::endl;
        std::cerr << "  use_dense_sketch : (Optional) 1 = dense Gaussian sketch, 0 = sparse SASO (default: 0)" << std::endl;
        std::cerr << "  block_size       : (Optional) Column-block size for CQRRT/CholQR/sCholQR3 Gram (0 = full, default: 0)" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " ./spd_matrices 1138 100 2.0 3 conditioning_results.csv" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string spd_dir = argv[1];
    int64_t k_dim = std::stol(argv[2]);
    int64_t n = std::stol(argv[3]);
    double d_factor = std::stod(argv[4]);
    int64_t num_runs = std::stol(argv[5]);
    std::string output_file_arg = argv[6];
    bool use_dense_sketch = (argc >= 8) ? (std::stoi(argv[7]) != 0) : false;
    int64_t block_size = (argc >= 9) ? std::stol(argv[8]) : 0;

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    // Extract directory and filename, prepend date prefix to filename
    std::filesystem::path output_path(output_file_arg);
    std::string output_dir = output_path.parent_path().string();
    std::string output_filename = output_path.filename().string();
    std::string output_file = (output_dir.empty() ? "" : output_dir + "/") + date_prefix + output_filename;

    // Read metadata
    std::string metadata_file = spd_dir + "/metadata.txt";
    std::ifstream meta(metadata_file);
    if (!meta) {
        std::cerr << "Error: Cannot open metadata file: " << metadata_file << std::endl;
        std::cerr << "Please generate SPD matrices first using generate_spd_matrices" << std::endl;
        return 1;
    }

    // Parse metadata
    int64_t m = 0;
    int64_t num_matrices = 0;
    std::string line;
    while (std::getline(meta, line)) {
        if (line[0] == '#') continue;
        if (line.find("matrix_size:") != std::string::npos) {
            sscanf(line.c_str(), "matrix_size: %ld", &m);
        }
        if (line.find("num_matrices:") != std::string::npos) {
            sscanf(line.c_str(), "num_matrices: %ld", &num_matrices);
        }
    }
    meta.close();

    if (m == 0 || num_matrices == 0) {
        std::cerr << "Error: Failed to parse metadata" << std::endl;
        return 1;
    }

    // Get OpenMP thread count
    int num_threads = omp_get_max_threads();

    printf("\n=== CQRRT vs CholQR vs sCholQR3 vs Dense CQRRT Conditioning Study ===\n");
    printf("SPD matrix directory: %s\n", spd_dir.c_str());
    printf("Number of condition numbers: %ld\n", num_matrices);
    printf("Matrix dimensions: %ld x %ld x %ld\n", m, k_dim, n);
    printf("Sketching factor (CQRRT only): %.2f\n", d_factor);
    printf("Sketch type (CQRRT only): %s\n", use_dense_sketch ? "dense Gaussian" : "sparse SASO");
    printf("Block size (CQRRT, CholQR, sCholQR3): %ld (0 = full)\n", block_size);
    printf("Runs per condition: %ld\n", num_runs);
    printf("OpenMP threads: %d\n", num_threads);
    printf("Output file: %s\n", output_file.c_str());
    printf("==========================================\n\n");

    // Re-read metadata to get file list
    meta.open(metadata_file);
    std::vector<std::pair<double, std::string>> matrix_files;
    while (std::getline(meta, line)) {
        if (line[0] == '#') continue;
        if (line.find(',') == std::string::npos) continue;

        int64_t idx;
        double cond;
        char filename[256];
        if (sscanf(line.c_str(), "%ld, %lf, %s", &idx, &cond, filename) == 3) {
            matrix_files.push_back({cond, spd_dir + "/" + std::string(filename)});
        }
    }
    meta.close();

    if (matrix_files.empty()) {
        std::cerr << "Error: No matrix files found in metadata" << std::endl;
        return 1;
    }

    printf("Loaded %zu matrix files\n\n", matrix_files.size());

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Open output file
    std::ofstream out(output_file);
    out << "# CQRRT vs CholQR vs sCholQR3 vs Dense CQRRT Conditioning Study Results\n";
    out << "# Composite operator: CholSolver(κ) * (SASO * Gaussian)\n";
    out << "# Matrix dimensions: " << m << " x " << k_dim << " x " << n << "\n";
    out << "# d_factor (CQRRT only): " << d_factor << "\n";
    out << "# sketch_type (CQRRT only): " << (use_dense_sketch ? "dense Gaussian" : "sparse SASO") << "\n";
    out << "# block_size (CQRRT, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# OpenMP threads: " << num_threads << "\n";
    out << "# Format: cond_num, then per-algorithm: rel_error, orth_error, max_orth_cols, orth_rate, time (mean/std), memory (KB)\n";
    out << "cond_num,"
        << "cqrrt_rel_error_mean,cqrrt_rel_error_std,"
        << "cqrrt_orth_error_mean,cqrrt_orth_error_std,"
        << "cqrrt_max_orth_cols_mean,cqrrt_max_orth_cols_std,"
        << "cqrrt_orth_rate,"
        << "cqrrt_time_mean,cqrrt_time_std,"
        << "cholqr_rel_error_mean,cholqr_rel_error_std,"
        << "cholqr_orth_error_mean,cholqr_orth_error_std,"
        << "cholqr_max_orth_cols_mean,cholqr_max_orth_cols_std,"
        << "cholqr_orth_rate,"
        << "cholqr_time_mean,cholqr_time_std,"
        << "scholqr3_rel_error_mean,scholqr3_rel_error_std,"
        << "scholqr3_orth_error_mean,scholqr3_orth_error_std,"
        << "scholqr3_max_orth_cols_mean,scholqr3_max_orth_cols_std,"
        << "scholqr3_orth_rate,"
        << "scholqr3_time_mean,scholqr3_time_std,"
        << "dense_cqrrt_rel_error_mean,dense_cqrrt_rel_error_std,"
        << "dense_cqrrt_orth_error_mean,dense_cqrrt_orth_error_std,"
        << "dense_cqrrt_max_orth_cols_mean,dense_cqrrt_max_orth_cols_std,"
        << "dense_cqrrt_orth_rate,"
        << "dense_cqrrt_time_mean,dense_cqrrt_time_std,"
        << "cqrrt_peak_rss_kb,cqrrt_analytical_kb,"
        << "cholqr_peak_rss_kb,cholqr_analytical_kb,"
        << "scholqr3_peak_rss_kb,scholqr3_analytical_kb,"
        << "dense_cqrrt_peak_rss_kb,dense_cqrrt_analytical_kb\n";

    // Open runtime breakdown file
    std::string breakdown_file = output_file.substr(0, output_file.find_last_of('.')) + "_breakdown.csv";
    std::ofstream breakdown(breakdown_file);
    breakdown << "# Runtime Breakdown for All Algorithms (from fastest run per condition number)\n";
    breakdown << "# Composite operator: CholSolver(κ) * (SASO * Gaussian)\n";
    breakdown << "# Matrix dimensions: " << m << " x " << k_dim << " x " << n << "\n";
    breakdown << "# d_factor (CQRRT only): " << d_factor << "\n";
    breakdown << "# sketch_type (CQRRT only): " << (use_dense_sketch ? "dense Gaussian" : "sparse SASO") << "\n";
    breakdown << "# block_size (CQRRT, CholQR, sCholQR3): " << block_size << " (0 = full)\n";
    breakdown << "# num_runs: " << num_runs << "\n";
    breakdown << "# OpenMP threads: " << num_threads << "\n";
    breakdown << "# Times are in microseconds\n";
    breakdown << "# CQRRT: alloc, saso, qr, trtri, linop_precond, linop_gram, trmm_gram, potrf, finalize, rest, total\n";
    breakdown << "# CholQR: alloc, materialize, gram, potrf, rest, total\n";
    breakdown << "# sCholQR3: alloc, materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total\n";
    breakdown << "# Dense CQRRT: materialize, saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total\n";
    breakdown << "cond_num,"
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
        double warmup_cond = matrix_files[0].first;
        std::string warmup_file = matrix_files[0].second;
        auto warmup_state = state;  // Use copy to not affect main RNG sequence
        run_single_test<double>(warmup_file, warmup_cond, m, k_dim, n, d_factor, use_dense_sketch, block_size, warmup_state);
        printf("Warmup complete, starting measurements.\n\n");
    }

    // Run conditioning study
    for (size_t i = 0; i < matrix_files.size(); ++i) {
        double cond_num = matrix_files[i].first;
        std::string filepath = matrix_files[i].second;

        printf("Testing condition number %.6e (%zu/%zu)...\n", cond_num, i + 1, matrix_files.size());

        // Run multiple times for statistics
        std::vector<conditioning_result<double>> results;
        int64_t fastest_cqrrt_idx = 0;
        int64_t fastest_cholqr_idx = 0;
        int64_t fastest_scholqr3_idx = 0;
        int64_t fastest_dense_cqrrt_idx = 0;
        long fastest_cqrrt_time = std::numeric_limits<long>::max();
        long fastest_cholqr_time = std::numeric_limits<long>::max();
        long fastest_scholqr3_time = std::numeric_limits<long>::max();
        long fastest_dense_cqrrt_time = std::numeric_limits<long>::max();

        for (int64_t run = 0; run < num_runs; ++run) {
            auto result = run_single_test<double>(filepath, cond_num, m, k_dim, n, d_factor, use_dense_sketch, block_size, state);
            results.push_back(result);

            // Track fastest runs for each algorithm
            if (result.cqrrt.time < fastest_cqrrt_time) {
                fastest_cqrrt_time = result.cqrrt.time;
                fastest_cqrrt_idx = run;
            }
            if (result.cholqr.time < fastest_cholqr_time) {
                fastest_cholqr_time = result.cholqr.time;
                fastest_cholqr_idx = run;
            }
            if (result.scholqr3.time < fastest_scholqr3_time) {
                fastest_scholqr3_time = result.scholqr3.time;
                fastest_scholqr3_idx = run;
            }
            if (result.dense_cqrrt.time < fastest_dense_cqrrt_time) {
                fastest_dense_cqrrt_time = result.dense_cqrrt.time;
                fastest_dense_cqrrt_idx = run;
            }

            printf("  Run %ld/%ld:\n", run + 1, num_runs);
            printf("    CQRRT:   orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.cqrrt.orth_error, result.cqrrt.max_orth_cols, n, result.cqrrt.time);
            printf("    CholQR:  orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.cholqr.orth_error, result.cholqr.max_orth_cols, n, result.cholqr.time);
            printf("    sCholQR3: orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.scholqr3.orth_error, result.scholqr3.max_orth_cols, n, result.scholqr3.time);
            printf("    Dense CQRRT: orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.dense_cqrrt.orth_error, result.dense_cqrrt.max_orth_cols, n, result.dense_cqrrt.time);
        }

        // Get subroutine times from fastest runs
        const auto& fastest_cqrrt = results[fastest_cqrrt_idx];
        const auto& fastest_cholqr = results[fastest_cholqr_idx];
        const auto& fastest_scholqr3 = results[fastest_scholqr3_idx];
        const auto& fastest_dense_cqrrt = results[fastest_dense_cqrrt_idx];

        // Compute statistics for all algorithms
        double cqrrt_rel_err_mean, cqrrt_rel_err_std, cqrrt_orth_err_mean, cqrrt_orth_err_std;
        double cqrrt_max_orth_mean, cqrrt_max_orth_std, cqrrt_orth_rate;
        double cqrrt_time_mean, cqrrt_time_std;
        compute_quality_stats<double>(results,
            [](const auto& r) -> const auto& { return r.cqrrt; }, num_runs,
            cqrrt_rel_err_mean, cqrrt_rel_err_std,
            cqrrt_orth_err_mean, cqrrt_orth_err_std,
            cqrrt_max_orth_mean, cqrrt_max_orth_std,
            cqrrt_orth_rate, cqrrt_time_mean, cqrrt_time_std);

        double cholqr_rel_err_mean, cholqr_rel_err_std, cholqr_orth_err_mean, cholqr_orth_err_std;
        double cholqr_max_orth_mean, cholqr_max_orth_std, cholqr_orth_rate;
        double cholqr_time_mean, cholqr_time_std;
        compute_quality_stats<double>(results,
            [](const auto& r) -> const auto& { return r.cholqr; }, num_runs,
            cholqr_rel_err_mean, cholqr_rel_err_std,
            cholqr_orth_err_mean, cholqr_orth_err_std,
            cholqr_max_orth_mean, cholqr_max_orth_std,
            cholqr_orth_rate, cholqr_time_mean, cholqr_time_std);

        double scholqr3_rel_err_mean, scholqr3_rel_err_std, scholqr3_orth_err_mean, scholqr3_orth_err_std;
        double scholqr3_max_orth_mean, scholqr3_max_orth_std, scholqr3_orth_rate;
        double scholqr3_time_mean, scholqr3_time_std;
        compute_quality_stats<double>(results,
            [](const auto& r) -> const auto& { return r.scholqr3; }, num_runs,
            scholqr3_rel_err_mean, scholqr3_rel_err_std,
            scholqr3_orth_err_mean, scholqr3_orth_err_std,
            scholqr3_max_orth_mean, scholqr3_max_orth_std,
            scholqr3_orth_rate, scholqr3_time_mean, scholqr3_time_std);

        double dense_cqrrt_rel_err_mean, dense_cqrrt_rel_err_std, dense_cqrrt_orth_err_mean, dense_cqrrt_orth_err_std;
        double dense_cqrrt_max_orth_mean, dense_cqrrt_max_orth_std, dense_cqrrt_orth_rate;
        double dense_cqrrt_time_mean, dense_cqrrt_time_std;
        compute_quality_stats<double>(results,
            [](const auto& r) -> const auto& { return r.dense_cqrrt; }, num_runs,
            dense_cqrrt_rel_err_mean, dense_cqrrt_rel_err_std,
            dense_cqrrt_orth_err_mean, dense_cqrrt_orth_err_std,
            dense_cqrrt_max_orth_mean, dense_cqrrt_max_orth_std,
            dense_cqrrt_orth_rate, dense_cqrrt_time_mean, dense_cqrrt_time_std);

        // Write results
        out << std::scientific << std::setprecision(6)
            << cond_num << ","
            << cqrrt_rel_err_mean << "," << cqrrt_rel_err_std << ","
            << cqrrt_orth_err_mean << "," << cqrrt_orth_err_std << ","
            << cqrrt_max_orth_mean << "," << cqrrt_max_orth_std << ","
            << cqrrt_orth_rate << ","
            << cqrrt_time_mean << "," << cqrrt_time_std << ","
            << cholqr_rel_err_mean << "," << cholqr_rel_err_std << ","
            << cholqr_orth_err_mean << "," << cholqr_orth_err_std << ","
            << cholqr_max_orth_mean << "," << cholqr_max_orth_std << ","
            << cholqr_orth_rate << ","
            << cholqr_time_mean << "," << cholqr_time_std << ","
            << scholqr3_rel_err_mean << "," << scholqr3_rel_err_std << ","
            << scholqr3_orth_err_mean << "," << scholqr3_orth_err_std << ","
            << scholqr3_max_orth_mean << "," << scholqr3_max_orth_std << ","
            << scholqr3_orth_rate << ","
            << scholqr3_time_mean << "," << scholqr3_time_std << ","
            << dense_cqrrt_rel_err_mean << "," << dense_cqrrt_rel_err_std << ","
            << dense_cqrrt_orth_err_mean << "," << dense_cqrrt_orth_err_std << ","
            << dense_cqrrt_max_orth_mean << "," << dense_cqrrt_max_orth_std << ","
            << dense_cqrrt_orth_rate << ","
            << dense_cqrrt_time_mean << "," << dense_cqrrt_time_std << ","
            << fastest_cqrrt.cqrrt.peak_rss_kb << "," << fastest_cqrrt.cqrrt_analytical_kb << ","
            << fastest_cholqr.cholqr.peak_rss_kb << "," << fastest_cholqr.cholqr_analytical_kb << ","
            << fastest_scholqr3.scholqr3.peak_rss_kb << "," << fastest_scholqr3.scholqr3_analytical_kb << ","
            << fastest_dense_cqrrt.dense_cqrrt.peak_rss_kb << "," << fastest_dense_cqrrt.dense_cqrrt_analytical_kb << "\n";
        out.flush();

        printf("  Summary:\n");
        printf("    CQRRT:   orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               cqrrt_orth_err_mean, cqrrt_orth_err_std, cqrrt_max_orth_mean, cqrrt_max_orth_std, cqrrt_orth_rate);
        printf("    CholQR:  orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               cholqr_orth_err_mean, cholqr_orth_err_std, cholqr_max_orth_mean, cholqr_max_orth_std, cholqr_orth_rate);
        printf("    sCholQR3: orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               scholqr3_orth_err_mean, scholqr3_orth_err_std, scholqr3_max_orth_mean, scholqr3_max_orth_std, scholqr3_orth_rate);
        printf("    Dense CQRRT: orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               dense_cqrrt_orth_err_mean, dense_cqrrt_orth_err_std, dense_cqrrt_max_orth_mean, dense_cqrrt_max_orth_std, dense_cqrrt_orth_rate);
        printf("                 time=%.1f±%.1f μs\n",
               dense_cqrrt_time_mean, dense_cqrrt_time_std);
        printf("  Memory (peak RSS / analytical KB):\n");
        printf("    CQRRT: %ld / %ld,  CholQR: %ld / %ld,  sCholQR3: %ld / %ld,  Dense: %ld / %ld\n\n",
               fastest_cqrrt.cqrrt.peak_rss_kb, fastest_cqrrt.cqrrt_analytical_kb,
               fastest_cholqr.cholqr.peak_rss_kb, fastest_cholqr.cholqr_analytical_kb,
               fastest_scholqr3.scholqr3.peak_rss_kb, fastest_scholqr3.scholqr3_analytical_kb,
               fastest_dense_cqrrt.dense_cqrrt.peak_rss_kb, fastest_dense_cqrrt.dense_cqrrt_analytical_kb);

        // Write runtime breakdown from fastest runs for all algorithms
        breakdown << std::scientific << std::setprecision(6)
                  << cond_num << ","
                  // CQRRT (11 values)
                  << fastest_cqrrt.cqrrt_alloc_time << ","
                  << fastest_cqrrt.cqrrt_saso_time << ","
                  << fastest_cqrrt.cqrrt_qr_time << ","
                  << fastest_cqrrt.cqrrt_trtri_time << ","
                  << fastest_cqrrt.cqrrt_linop_precond_time << ","
                  << fastest_cqrrt.cqrrt_linop_gram_time << ","
                  << fastest_cqrrt.cqrrt_trmm_gram_time << ","
                  << fastest_cqrrt.cqrrt_potrf_time << ","
                  << fastest_cqrrt.cqrrt_finalize_time << ","
                  << fastest_cqrrt.cqrrt_rest_time << ","
                  << fastest_cqrrt.cqrrt.time << ","
                  // CholQR (6 values)
                  << fastest_cholqr.cholqr_alloc_time << ","
                  << fastest_cholqr.cholqr_materialize_time << ","
                  << fastest_cholqr.cholqr_gram_time << ","
                  << fastest_cholqr.cholqr_potrf_time << ","
                  << fastest_cholqr.cholqr_rest_time << ","
                  << fastest_cholqr.cholqr.time << ","
                  // sCholQR3 (13 values)
                  << fastest_scholqr3.scholqr3_alloc_time << ","
                  << fastest_scholqr3.scholqr3_materialize_time << ","
                  << fastest_scholqr3.scholqr3_gram1_time << ","
                  << fastest_scholqr3.scholqr3_potrf1_time << ","
                  << fastest_scholqr3.scholqr3_trsm1_time << ","
                  << fastest_scholqr3.scholqr3_syrk2_time << ","
                  << fastest_scholqr3.scholqr3_potrf2_time << ","
                  << fastest_scholqr3.scholqr3_update2_time << ","
                  << fastest_scholqr3.scholqr3_syrk3_time << ","
                  << fastest_scholqr3.scholqr3_potrf3_time << ","
                  << fastest_scholqr3.scholqr3_update3_time << ","
                  << fastest_scholqr3.scholqr3_rest_time << ","
                  << fastest_scholqr3.scholqr3.time << ","
                  // Dense CQRRT (11 values: materialize, saso, qr, trtri=0, precond, gram, trmm_gram=0, potrf, finalize, rest, total)
                  << fastest_dense_cqrrt.dense_cqrrt_materialize_time << ","
                  << fastest_dense_cqrrt.dense_cqrrt_saso_time << ","
                  << fastest_dense_cqrrt.dense_cqrrt_qr_time << ","
                  << 0 << ","  // trtri (always 0 for dense)
                  << fastest_dense_cqrrt.dense_cqrrt_precond_time << ","
                  << fastest_dense_cqrrt.dense_cqrrt_gram_time << ","
                  << 0 << ","  // trmm_gram (always 0 for dense)
                  << fastest_dense_cqrrt.dense_cqrrt_potrf_time << ","
                  << fastest_dense_cqrrt.dense_cqrrt_finalize_time << ","
                  << fastest_dense_cqrrt.dense_cqrrt_rest_time << ","
                  << fastest_dense_cqrrt.dense_cqrrt.time << ","
                  // Memory columns (KB)
                  << fastest_cqrrt.cqrrt.peak_rss_kb << "," << fastest_cqrrt.cqrrt_analytical_kb << ","
                  << fastest_cholqr.cholqr.peak_rss_kb << "," << fastest_cholqr.cholqr_analytical_kb << ","
                  << fastest_scholqr3.scholqr3.peak_rss_kb << "," << fastest_scholqr3.scholqr3_analytical_kb << ","
                  << fastest_dense_cqrrt.dense_cqrrt.peak_rss_kb << "," << fastest_dense_cqrrt.dense_cqrrt_analytical_kb << "\n";
        breakdown.flush();
    }

    out.close();
    breakdown.close();
    printf("========================================\n");
    printf("Conditioning study complete!\n");
    printf("Results saved to: %s\n", output_file.c_str());
    printf("Runtime breakdown saved to: %s\n", breakdown_file.c_str());
    printf("========================================\n");

    return 0;
}
#endif
