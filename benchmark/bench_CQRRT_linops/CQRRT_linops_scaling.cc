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

// Need to include demos utilities
#include "../../demos/functions/drivers/dm_cqrrt_linops.hh"
#include "../../demos/functions/drivers/dm_cholqr_linops.hh"
#include "../../demos/functions/drivers/dm_scholqr3_linops.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

template <typename T>
struct scaling_result {
    int64_t m;                // Number of rows
    int64_t n;                // Number of columns
    T density;                // Sparse matrix density
    T aspect_ratio;           // m / n

    // CQRRT results
    T cqrrt_rel_error;        // ||A - QR|| / ||A||
    T cqrrt_orth_error;       // ||Q^T Q - I|| / sqrt(n)
    bool cqrrt_is_orthonormal;
    int64_t cqrrt_max_orth_cols;
    long cqrrt_time;          // Total time (microseconds)

    // CQRRT subroutine times (from fastest run)
    long cqrrt_saso_time;
    long cqrrt_qr_time;
    long cqrrt_trtri_time;
    long cqrrt_linop_precond_time;
    long cqrrt_linop_gram_time;
    long cqrrt_trmm_gram_time;
    long cqrrt_potrf_time;
    long cqrrt_finalize_time;
    long cqrrt_rest_time;

    // CholQR results
    T cholqr_rel_error;
    T cholqr_orth_error;
    bool cholqr_is_orthonormal;
    int64_t cholqr_max_orth_cols;
    long cholqr_time;

    // CholQR subroutine times (5 entries: materialize, gram, potrf, rest, total)
    long cholqr_materialize_time;
    long cholqr_gram_time;
    long cholqr_potrf_time;
    long cholqr_rest_time;

    // sCholQR3 results
    T scholqr3_rel_error;
    T scholqr3_orth_error;
    bool scholqr3_is_orthonormal;
    int64_t scholqr3_max_orth_cols;
    long scholqr3_time;

    // sCholQR3 subroutine times (12 entries)
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

template <typename T, typename RNG>
static scaling_result<T> run_single_test(
    int64_t m,
    int64_t n,
    T density,
    T d_factor,
    int64_t num_runs,
    RandBLAS::RNGState<RNG>& state) {

    scaling_result<T> result;
    result.m = m;
    result.n = n;
    result.density = density;
    result.aspect_ratio = static_cast<T>(m) / static_cast<T>(n);

    // Generate sparse matrix A: m × n
    auto A_coo = RandLAPACK::gen::gen_sparse_mat<T>(m, n, density, state);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    // Compute dense representation for verification
    std::vector<T> A_dense(m * n, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(A_csr, Layout::ColMajor, A_dense.data());

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);
    T norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // ============================================================
    // Run CQRRT (preconditioned Cholesky QR) - multiple runs
    // ============================================================
    {
        // Initialize with first run
        result.cqrrt_time = std::numeric_limits<long>::max();
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

            RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
            CQRRT_QR.nnz = 5;
            CQRRT_QR.call(A_linop, R_cqrrt.data(), n, d_factor, state_copy);

            long run_time = CQRRT_QR.times[9];  // total_t_dur

            // Track fastest run and its subroutine times
            if (run_time < result.cqrrt_time) {
                result.cqrrt_time = run_time;
                result.cqrrt_saso_time = CQRRT_QR.times[0];
                result.cqrrt_qr_time = CQRRT_QR.times[1];
                result.cqrrt_trtri_time = CQRRT_QR.times[2];
                result.cqrrt_linop_precond_time = CQRRT_QR.times[3];
                result.cqrrt_linop_gram_time = CQRRT_QR.times[4];
                result.cqrrt_trmm_gram_time = CQRRT_QR.times[5];
                result.cqrrt_potrf_time = CQRRT_QR.times[6];
                result.cqrrt_finalize_time = CQRRT_QR.times[7];
                result.cqrrt_rest_time = CQRRT_QR.times[8];

                // Compute factorization error for fastest run
                std::vector<T> QR(m * n, 0.0);
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
                           1.0, CQRRT_QR.Q, m, R_cqrrt.data(), n, 0.0, QR.data(), m);

                T norm_diff = 0.0;
                for (int64_t i = 0; i < m * n; ++i) {
                    T diff = A_dense[i] - QR[i];
                    norm_diff += diff * diff;
                }
                norm_diff = std::sqrt(norm_diff);
                best_rel_error = norm_diff / norm_A;

                // Measure orthogonality
                measure_orthogonality(CQRRT_QR.Q, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }
        }

        result.cqrrt_rel_error = best_rel_error;
        result.cqrrt_orth_error = best_orth_error;
        result.cqrrt_is_orthonormal = best_is_orthonormal;
        result.cqrrt_max_orth_cols = best_max_orth_cols;
    }

    // ============================================================
    // Run CholQR (unpreconditioned Cholesky QR) - multiple runs
    // ============================================================
    {
        result.cholqr_time = std::numeric_limits<long>::max();
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

            RandLAPACK_demos::CholQR_linops<T> CholQR_alg(true, tol, true);  // timing=true, test_mode=true
            CholQR_alg.call(A_linop, R_cholqr.data(), n);

            long run_time = CholQR_alg.times[4];  // total

            if (run_time < result.cholqr_time) {
                result.cholqr_time = run_time;
                result.cholqr_materialize_time = CholQR_alg.times[0];
                result.cholqr_gram_time = CholQR_alg.times[1];
                result.cholqr_potrf_time = CholQR_alg.times[2];
                result.cholqr_rest_time = CholQR_alg.times[3];

                // Compute factorization error
                std::vector<T> QR(m * n, 0.0);
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
                           1.0, CholQR_alg.Q, m, R_cholqr.data(), n, 0.0, QR.data(), m);

                T norm_diff = 0.0;
                for (int64_t i = 0; i < m * n; ++i) {
                    T diff = A_dense[i] - QR[i];
                    norm_diff += diff * diff;
                }
                norm_diff = std::sqrt(norm_diff);
                best_rel_error = norm_diff / norm_A;

                // Measure orthogonality
                measure_orthogonality(CholQR_alg.Q, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }
        }

        result.cholqr_rel_error = best_rel_error;
        result.cholqr_orth_error = best_orth_error;
        result.cholqr_is_orthonormal = best_is_orthonormal;
        result.cholqr_max_orth_cols = best_max_orth_cols;
    }

    // ============================================================
    // Run sCholQR3 (shifted Cholesky QR with 3 iterations) - multiple runs
    // ============================================================
    {
        result.scholqr3_time = std::numeric_limits<long>::max();
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

            RandLAPACK_demos::sCholQR3_linops<T> sCholQR3_alg(true, tol, true);  // timing=true, test_mode=true
            sCholQR3_alg.call(A_linop, R_scholqr3.data(), n);

            long run_time = sCholQR3_alg.times[11];  // total

            if (run_time < result.scholqr3_time) {
                result.scholqr3_time = run_time;
                result.scholqr3_materialize_time = sCholQR3_alg.times[0];
                result.scholqr3_gram1_time = sCholQR3_alg.times[1];
                result.scholqr3_potrf1_time = sCholQR3_alg.times[2];
                result.scholqr3_trsm1_time = sCholQR3_alg.times[3];
                result.scholqr3_syrk2_time = sCholQR3_alg.times[4];
                result.scholqr3_potrf2_time = sCholQR3_alg.times[5];
                result.scholqr3_update2_time = sCholQR3_alg.times[6];
                result.scholqr3_syrk3_time = sCholQR3_alg.times[7];
                result.scholqr3_potrf3_time = sCholQR3_alg.times[8];
                result.scholqr3_update3_time = sCholQR3_alg.times[9];
                result.scholqr3_rest_time = sCholQR3_alg.times[10];

                // Compute factorization error
                std::vector<T> QR(m * n, 0.0);
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
                           1.0, sCholQR3_alg.Q, m, R_scholqr3.data(), n, 0.0, QR.data(), m);

                T norm_diff = 0.0;
                for (int64_t i = 0; i < m * n; ++i) {
                    T diff = A_dense[i] - QR[i];
                    norm_diff += diff * diff;
                }
                norm_diff = std::sqrt(norm_diff);
                best_rel_error = norm_diff / norm_A;

                // Measure orthogonality
                measure_orthogonality(sCholQR3_alg.Q, m, n, best_orth_error,
                                     best_is_orthonormal, best_max_orth_cols);
            }
        }

        result.scholqr3_rel_error = best_rel_error;
        result.scholqr3_orth_error = best_orth_error;
        result.scholqr3_is_orthonormal = best_is_orthonormal;
        result.scholqr3_max_orth_cols = best_max_orth_cols;
    }

    return result;
}

int main(int argc, char *argv[]) {

    if (argc != 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <aspect_ratio> <m_start> <m_end> <num_sizes> <density> <d_factor> <num_runs> <output_dir>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  aspect_ratio : Ratio m/n (e.g., 20 means n = m/20)" << std::endl;
        std::cerr << "  m_start      : Starting number of rows (smallest matrix)" << std::endl;
        std::cerr << "  m_end        : Ending number of rows (largest matrix)" << std::endl;
        std::cerr << "  num_sizes    : Number of matrix sizes to test" << std::endl;
        std::cerr << "  density      : Sparse matrix density (e.g., 0.1)" << std::endl;
        std::cerr << "  d_factor     : Sketching dimension factor for CQRRT (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs     : Number of runs per matrix size (for timing)" << std::endl;
        std::cerr << "  output_dir   : Directory to write output files" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " 20 500 10000 50 0.1 2.0 3 ./output" << std::endl;
        std::cerr << "  (Tests 50 matrices from 500x25 to 10000x500, all with aspect ratio 20:1, 3 runs each)" << std::endl;
        return 1;
    }

    // Parse arguments
    double aspect_ratio = std::stod(argv[1]);
    int64_t m_start = std::stol(argv[2]);
    int64_t m_end = std::stol(argv[3]);
    int64_t num_sizes = std::stol(argv[4]);
    double density = std::stod(argv[5]);
    double d_factor = std::stod(argv[6]);
    int64_t num_runs = std::stol(argv[7]);
    std::string output_dir = argv[8];

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    // Build list of (m, n) pairs to test
    std::vector<std::pair<int64_t, int64_t>> sizes;
    for (int64_t i = 0; i < num_sizes; ++i) {
        // Linear interpolation from m_start to m_end
        int64_t m = m_start + (m_end - m_start) * i / (num_sizes - 1);
        int64_t n = static_cast<int64_t>(m / aspect_ratio);
        if (n < 1) n = 1;  // Ensure at least 1 column
        sizes.push_back({m, n});
    }

    // Get OpenMP thread count
    int num_threads = omp_get_max_threads();

    printf("\n=== CQRRT vs CholQR vs sCholQR3 Scaling Study ===\n");
    printf("Fixed aspect ratio: %.1f:1 (m/n)\n", aspect_ratio);
    printf("Matrix sizes: %ld x %ld to %ld x %ld\n",
           sizes.front().first, sizes.front().second,
           sizes.back().first, sizes.back().second);
    printf("Number of test sizes: %zu\n", sizes.size());
    printf("Density: %.3f\n", density);
    printf("d_factor (CQRRT): %.2f\n", d_factor);
    printf("Runs per size: %ld\n", num_runs);
    printf("OpenMP threads: %d\n", num_threads);
    printf("=====================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Prepare output file with date/time prefix
    std::string output_file = output_dir + "/" + date_prefix + "scaling_results.csv";
    std::ofstream out(output_file);
    out << "# CQRRT vs CholQR vs sCholQR3 Scaling Study Results\n";
    out << "# Fixed aspect ratio: " << aspect_ratio << ":1\n";
    out << "# Density: " << density << "\n";
    out << "# d_factor (CQRRT only): " << d_factor << "\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# OpenMP threads: " << num_threads << "\n";
    out << "m,n,aspect_ratio,density,"
        << "cqrrt_rel_error,cqrrt_orth_error,cqrrt_max_orth_cols,cqrrt_is_orth,cqrrt_time_us,"
        << "cholqr_rel_error,cholqr_orth_error,cholqr_max_orth_cols,cholqr_is_orth,cholqr_time_us,"
        << "scholqr3_rel_error,scholqr3_orth_error,scholqr3_max_orth_cols,scholqr3_is_orth,scholqr3_time_us,"
        << "speedup_cqrrt_over_cholqr,speedup_scholqr3_over_cholqr\n";

    // Prepare runtime breakdown file with date/time prefix
    std::string breakdown_file = output_dir + "/" + date_prefix + "scaling_breakdown.csv";
    std::ofstream breakdown(breakdown_file);
    breakdown << "# Runtime Breakdown for All Algorithms (from fastest run per matrix size)\n";
    breakdown << "# Fixed aspect ratio: " << aspect_ratio << ":1\n";
    breakdown << "# Density: " << density << "\n";
    breakdown << "# d_factor (CQRRT only): " << d_factor << "\n";
    breakdown << "# num_runs: " << num_runs << "\n";
    breakdown << "# OpenMP threads: " << num_threads << "\n";
    breakdown << "# Times are in microseconds\n";
    breakdown << "# CQRRT: saso, qr, trtri, linop_precond, linop_gram, trmm_gram, potrf, finalize, rest, total\n";
    breakdown << "# CholQR: materialize, gram, potrf, rest, total\n";
    breakdown << "# sCholQR3: materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total\n";
    breakdown << "m,n,"
              << "cqrrt_saso,cqrrt_qr,cqrrt_trtri,cqrrt_linop_precond,cqrrt_linop_gram,cqrrt_trmm_gram,cqrrt_potrf,cqrrt_finalize,cqrrt_rest,cqrrt_total,"
              << "cholqr_materialize,cholqr_gram,cholqr_potrf,cholqr_rest,cholqr_total,"
              << "scholqr3_materialize,scholqr3_gram1,scholqr3_potrf1,scholqr3_trsm1,scholqr3_syrk2,scholqr3_potrf2,scholqr3_update2,scholqr3_syrk3,scholqr3_potrf3,scholqr3_update3,scholqr3_rest,scholqr3_total\n";

    // Run scaling study
    for (size_t i = 0; i < sizes.size(); ++i) {
        int64_t m = sizes[i].first;
        int64_t n = sizes[i].second;
        printf("Testing %ld x %ld (aspect ratio %.1f) [%zu/%zu]...\n",
               m, n, static_cast<double>(m) / n, i + 1, sizes.size());

        auto result = run_single_test<double>(m, n, density, d_factor, num_runs, state);

        double speedup_cqrrt = (result.cqrrt_time > 0) ?
            static_cast<double>(result.cholqr_time) / result.cqrrt_time : 0.0;
        double speedup_scholqr3 = (result.scholqr3_time > 0) ?
            static_cast<double>(result.cholqr_time) / result.scholqr3_time : 0.0;

        printf("  CQRRT:   orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.cqrrt_orth_error, result.cqrrt_max_orth_cols, n, result.cqrrt_time);
        printf("  CholQR:  orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.cholqr_orth_error, result.cholqr_max_orth_cols, n, result.cholqr_time);
        printf("  sCholQR3: orth_err=%.2e, max_orth=%ld/%ld, time=%ld us\n",
               result.scholqr3_orth_error, result.scholqr3_max_orth_cols, n, result.scholqr3_time);
        printf("  Speedup (CholQR/CQRRT): %.2fx, (CholQR/sCholQR3): %.2fx\n\n", speedup_cqrrt, speedup_scholqr3);

        // Write results
        out << std::fixed << std::setprecision(1)
            << result.m << "," << result.n << "," << result.aspect_ratio << ","
            << std::setprecision(3) << result.density << ","
            << std::scientific << std::setprecision(6)
            << result.cqrrt_rel_error << "," << result.cqrrt_orth_error << ","
            << result.cqrrt_max_orth_cols << "," << (result.cqrrt_is_orthonormal ? 1 : 0) << ","
            << result.cqrrt_time << ","
            << result.cholqr_rel_error << "," << result.cholqr_orth_error << ","
            << result.cholqr_max_orth_cols << "," << (result.cholqr_is_orthonormal ? 1 : 0) << ","
            << result.cholqr_time << ","
            << result.scholqr3_rel_error << "," << result.scholqr3_orth_error << ","
            << result.scholqr3_max_orth_cols << "," << (result.scholqr3_is_orthonormal ? 1 : 0) << ","
            << result.scholqr3_time << ","
            << std::fixed << std::setprecision(3) << speedup_cqrrt << "," << speedup_scholqr3 << "\n";
        out.flush();

        // Write runtime breakdown for all algorithms
        breakdown << result.m << "," << result.n << ","
                  // CQRRT (10 values)
                  << result.cqrrt_saso_time << "," << result.cqrrt_qr_time << ","
                  << result.cqrrt_trtri_time << "," << result.cqrrt_linop_precond_time << ","
                  << result.cqrrt_linop_gram_time << "," << result.cqrrt_trmm_gram_time << ","
                  << result.cqrrt_potrf_time << "," << result.cqrrt_finalize_time << ","
                  << result.cqrrt_rest_time << "," << result.cqrrt_time << ","
                  // CholQR (5 values)
                  << result.cholqr_materialize_time << "," << result.cholqr_gram_time << ","
                  << result.cholqr_potrf_time << "," << result.cholqr_rest_time << ","
                  << result.cholqr_time << ","
                  // sCholQR3 (12 values)
                  << result.scholqr3_materialize_time << "," << result.scholqr3_gram1_time << ","
                  << result.scholqr3_potrf1_time << "," << result.scholqr3_trsm1_time << ","
                  << result.scholqr3_syrk2_time << "," << result.scholqr3_potrf2_time << ","
                  << result.scholqr3_update2_time << "," << result.scholqr3_syrk3_time << ","
                  << result.scholqr3_potrf3_time << "," << result.scholqr3_update3_time << ","
                  << result.scholqr3_rest_time << "," << result.scholqr3_time << "\n";
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
