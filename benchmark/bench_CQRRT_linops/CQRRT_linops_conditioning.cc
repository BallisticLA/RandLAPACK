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
#include <ctime>
#include <omp.h>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

// Need to include demos utilities
#include "../../demos/functions/linops_external/dm_cholsolver_linop.hh"
#include "../../demos/functions/drivers/dm_cqrrt_linops.hh"
#include "../../demos/functions/drivers/dm_cholqr_linops.hh"
#include "../../demos/functions/drivers/dm_scholqr3_linops.hh"
#include "../../demos/functions/misc/dm_util.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

template <typename T>
struct conditioning_result {
    T cond_num;           // Condition number of SPD matrix

    // CQRRT results
    T cqrrt_rel_error;          // ||A - QR|| / ||A||
    T cqrrt_orth_error;         // ||Q^T Q - I|| / sqrt(n)
    bool cqrrt_is_orthonormal;  // Is full Q block orthonormal?
    int64_t cqrrt_max_orth_cols;  // Maximum orthonormal prefix
    long cqrrt_time;      // Total computation time (microseconds)

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
    T cholqr_rel_error;          // ||A - QR|| / ||A||
    T cholqr_orth_error;         // ||Q^T Q - I|| / sqrt(n)
    bool cholqr_is_orthonormal;  // Is full Q block orthonormal?
    int64_t cholqr_max_orth_cols;  // Maximum orthonormal prefix
    long cholqr_time;      // Total computation time (microseconds)

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
static conditioning_result<T> run_single_test(
    const std::string& spd_filename,
    T cond_num,
    int64_t m,
    int64_t k_dim,
    int64_t n,
    T d_factor,
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
        result.cqrrt_rel_error = std::numeric_limits<T>::quiet_NaN();
        result.cqrrt_orth_error = std::numeric_limits<T>::quiet_NaN();
        result.cqrrt_is_orthonormal = false;
        result.cqrrt_max_orth_cols = -1;
        result.cqrrt_time = 0;
        result.cholqr_rel_error = std::numeric_limits<T>::quiet_NaN();
        result.cholqr_orth_error = std::numeric_limits<T>::quiet_NaN();
        result.cholqr_is_orthonormal = false;
        result.cholqr_max_orth_cols = -1;
        result.cholqr_time = 0;
        return result;
    }

    // Generate SASO (Sparse) matrix: m × k_dim
    auto saso_coo = RandLAPACK::gen::gen_sparse_mat<T>(m, k_dim, 0.5, state);
    RandBLAS::sparse_data::csc::CSCMatrix<T> saso_csc(m, k_dim);
    RandBLAS::sparse_data::conversions::coo_to_csc(saso_coo, saso_csc);
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

    // Compute dense representation for verification
    std::vector<T> saso_dense(m * k_dim, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(saso_csc, Layout::ColMajor, saso_dense.data());

    std::vector<T> intermediate(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_dim, 1.0,
              saso_dense.data(), m, gaussian_mat.data(), k_dim, 0.0, intermediate.data(), m);

    std::vector<T> A_dense(m * n, 0.0);
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m,
                1.0, intermediate.data(), m, 0.0, A_dense.data(), m);

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);
    T norm_A = lapack::lange(Norm::Fro, m, n, A_dense.data(), m);

    // ============================================================
    // Run CQRRT (preconditioned Cholesky QR)
    // ============================================================
    {
        std::vector<T> R_cqrrt(n * n, 0.0);

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
        CQRRT_QR.nnz = 5;  // Optimal for sparse SPD matrices (from parameter study)
        CQRRT_QR.call(outer_composite, R_cqrrt.data(), n, d_factor, state);

        result.cqrrt_time = CQRRT_QR.times[9];  // total_t_dur
        result.cqrrt_saso_time = CQRRT_QR.times[0];
        result.cqrrt_qr_time = CQRRT_QR.times[1];
        result.cqrrt_trtri_time = CQRRT_QR.times[2];
        result.cqrrt_linop_precond_time = CQRRT_QR.times[3];
        result.cqrrt_linop_gram_time = CQRRT_QR.times[4];
        result.cqrrt_trmm_gram_time = CQRRT_QR.times[5];
        result.cqrrt_potrf_time = CQRRT_QR.times[6];
        result.cqrrt_finalize_time = CQRRT_QR.times[7];
        result.cqrrt_rest_time = CQRRT_QR.times[8];

        // Compute factorization error
        std::vector<T> QR(m * n, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
                   1.0, CQRRT_QR.Q, m, R_cqrrt.data(), n, 0.0, QR.data(), m);

        T norm_diff = 0.0;
        for (int64_t i = 0; i < m * n; ++i) {
            T diff = A_dense[i] - QR[i];
            norm_diff += diff * diff;
        }
        norm_diff = std::sqrt(norm_diff);
        result.cqrrt_rel_error = norm_diff / norm_A;

        // Measure orthogonality
        measure_orthogonality(CQRRT_QR.Q, m, n, result.cqrrt_orth_error,
                             result.cqrrt_is_orthonormal, result.cqrrt_max_orth_cols);
    }

    // ============================================================
    // Run CholQR (unpreconditioned Cholesky QR)
    // ============================================================
    {
        std::vector<T> R_cholqr(n * n, 0.0);

        RandLAPACK_demos::CholQR_linops<T> CholQR_alg(true, tol, true);  // timing=true, test_mode=true
        CholQR_alg.call(outer_composite, R_cholqr.data(), n);

        result.cholqr_time = CholQR_alg.times[4];  // total
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
        result.cholqr_rel_error = norm_diff / norm_A;

        // Measure orthogonality
        measure_orthogonality(CholQR_alg.Q, m, n, result.cholqr_orth_error,
                             result.cholqr_is_orthonormal, result.cholqr_max_orth_cols);
    }

    // ============================================================
    // Run sCholQR3 (shifted Cholesky QR with 3 iterations)
    // ============================================================
    {
        std::vector<T> R_scholqr3(n * n, 0.0);

        RandLAPACK_demos::sCholQR3_linops<T> sCholQR3_alg(true, tol, true);  // timing=true, test_mode=true
        sCholQR3_alg.call(outer_composite, R_scholqr3.data(), n);

        result.scholqr3_time = sCholQR3_alg.times[11];  // total
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
        result.scholqr3_rel_error = norm_diff / norm_A;

        // Measure orthogonality
        measure_orthogonality(sCholQR3_alg.Q, m, n, result.scholqr3_orth_error,
                             result.scholqr3_is_orthonormal, result.scholqr3_max_orth_cols);
    }

    return result;
}

int main(int argc, char *argv[]) {

    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <spd_matrix_dir> <k_dim> <n_cols> <d_factor> <num_runs> <output_file>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  spd_matrix_dir : Directory containing SPD matrices (with metadata.txt)" << std::endl;
        std::cerr << "  k_dim          : Intermediate dimension (SASO cols / Gaussian rows)" << std::endl;
        std::cerr << "  n_cols         : Final number of columns (Gaussian cols)" << std::endl;
        std::cerr << "  d_factor       : Sketching dimension factor (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs       : Number of runs per condition number (for averaging)" << std::endl;
        std::cerr << "  output_file    : Output CSV file for results" << std::endl;
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

    printf("\n=== CQRRT vs CholQR vs sCholQR3 Conditioning Study ===\n");
    printf("SPD matrix directory: %s\n", spd_dir.c_str());
    printf("Number of condition numbers: %ld\n", num_matrices);
    printf("Matrix dimensions: %ld x %ld x %ld\n", m, k_dim, n);
    printf("Sketching factor (CQRRT only): %.2f\n", d_factor);
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
    out << "# CQRRT vs CholQR vs sCholQR3 Conditioning Study Results\n";
    out << "# Composite operator: CholSolver(κ) * (SASO * Gaussian)\n";
    out << "# Matrix dimensions: " << m << " x " << k_dim << " x " << n << "\n";
    out << "# d_factor (CQRRT only): " << d_factor << "\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# OpenMP threads: " << num_threads << "\n";
    out << "# Format: cond_num, cqrrt_*, cholqr_*, scholqr3_* (rel_error, orth_error, max_orth_cols, orth_rate, time)\n";
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
        << "scholqr3_time_mean,scholqr3_time_std\n";

    // Open runtime breakdown file
    std::string breakdown_file = output_file.substr(0, output_file.find_last_of('.')) + "_breakdown.csv";
    std::ofstream breakdown(breakdown_file);
    breakdown << "# Runtime Breakdown for All Algorithms (from fastest run per condition number)\n";
    breakdown << "# Composite operator: CholSolver(κ) * (SASO * Gaussian)\n";
    breakdown << "# Matrix dimensions: " << m << " x " << k_dim << " x " << n << "\n";
    breakdown << "# d_factor (CQRRT only): " << d_factor << "\n";
    breakdown << "# num_runs: " << num_runs << "\n";
    breakdown << "# OpenMP threads: " << num_threads << "\n";
    breakdown << "# Times are in microseconds\n";
    breakdown << "# CQRRT: saso, qr, trtri, linop_precond, linop_gram, trmm_gram, potrf, finalize, rest, total\n";
    breakdown << "# CholQR: materialize, gram, potrf, rest, total\n";
    breakdown << "# sCholQR3: materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total\n";
    breakdown << "cond_num,"
              << "cqrrt_saso,cqrrt_qr,cqrrt_trtri,cqrrt_linop_precond,cqrrt_linop_gram,cqrrt_trmm_gram,cqrrt_potrf,cqrrt_finalize,cqrrt_rest,cqrrt_total,"
              << "cholqr_materialize,cholqr_gram,cholqr_potrf,cholqr_rest,cholqr_total,"
              << "scholqr3_materialize,scholqr3_gram1,scholqr3_potrf1,scholqr3_trsm1,scholqr3_syrk2,scholqr3_potrf2,scholqr3_update2,scholqr3_syrk3,scholqr3_potrf3,scholqr3_update3,scholqr3_rest,scholqr3_total\n";

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
        long fastest_cqrrt_time = std::numeric_limits<long>::max();
        long fastest_cholqr_time = std::numeric_limits<long>::max();
        long fastest_scholqr3_time = std::numeric_limits<long>::max();

        for (int64_t run = 0; run < num_runs; ++run) {
            auto result = run_single_test<double>(filepath, cond_num, m, k_dim, n, d_factor, state);
            results.push_back(result);

            // Track fastest runs for each algorithm
            if (result.cqrrt_time < fastest_cqrrt_time) {
                fastest_cqrrt_time = result.cqrrt_time;
                fastest_cqrrt_idx = run;
            }
            if (result.cholqr_time < fastest_cholqr_time) {
                fastest_cholqr_time = result.cholqr_time;
                fastest_cholqr_idx = run;
            }
            if (result.scholqr3_time < fastest_scholqr3_time) {
                fastest_scholqr3_time = result.scholqr3_time;
                fastest_scholqr3_idx = run;
            }

            printf("  Run %ld/%ld:\n", run + 1, num_runs);
            printf("    CQRRT:   orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.cqrrt_orth_error, result.cqrrt_max_orth_cols, n, result.cqrrt_time);
            printf("    CholQR:  orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.cholqr_orth_error, result.cholqr_max_orth_cols, n, result.cholqr_time);
            printf("    sCholQR3: orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.scholqr3_orth_error, result.scholqr3_max_orth_cols, n, result.scholqr3_time);
        }

        // Get subroutine times from fastest runs
        const auto& fastest_cqrrt = results[fastest_cqrrt_idx];
        const auto& fastest_cholqr = results[fastest_cholqr_idx];
        const auto& fastest_scholqr3 = results[fastest_scholqr3_idx];

        // Compute statistics for CQRRT
        double cqrrt_rel_err_mean = 0, cqrrt_orth_err_mean = 0, cqrrt_time_mean = 0;
        double cqrrt_max_orth_mean = 0;
        int cqrrt_orth_count = 0;

        for (const auto& r : results) {
            cqrrt_rel_err_mean += r.cqrrt_rel_error;
            cqrrt_orth_err_mean += r.cqrrt_orth_error;
            cqrrt_max_orth_mean += r.cqrrt_max_orth_cols;
            cqrrt_time_mean += r.cqrrt_time;
            if (r.cqrrt_is_orthonormal) cqrrt_orth_count++;
        }
        cqrrt_rel_err_mean /= num_runs;
        cqrrt_orth_err_mean /= num_runs;
        cqrrt_max_orth_mean /= num_runs;
        cqrrt_time_mean /= num_runs;

        double cqrrt_rel_err_std = 0, cqrrt_orth_err_std = 0, cqrrt_time_std = 0;
        double cqrrt_max_orth_std = 0;
        for (const auto& r : results) {
            cqrrt_rel_err_std += (r.cqrrt_rel_error - cqrrt_rel_err_mean) * (r.cqrrt_rel_error - cqrrt_rel_err_mean);
            cqrrt_orth_err_std += (r.cqrrt_orth_error - cqrrt_orth_err_mean) * (r.cqrrt_orth_error - cqrrt_orth_err_mean);
            cqrrt_max_orth_std += (r.cqrrt_max_orth_cols - cqrrt_max_orth_mean) * (r.cqrrt_max_orth_cols - cqrrt_max_orth_mean);
            cqrrt_time_std += (r.cqrrt_time - cqrrt_time_mean) * (r.cqrrt_time - cqrrt_time_mean);
        }
        cqrrt_rel_err_std = std::sqrt(cqrrt_rel_err_std / num_runs);
        cqrrt_orth_err_std = std::sqrt(cqrrt_orth_err_std / num_runs);
        cqrrt_max_orth_std = std::sqrt(cqrrt_max_orth_std / num_runs);
        cqrrt_time_std = std::sqrt(cqrrt_time_std / num_runs);

        double cqrrt_orth_rate = static_cast<double>(cqrrt_orth_count) / num_runs;

        // Compute statistics for CholQR
        double cholqr_rel_err_mean = 0, cholqr_orth_err_mean = 0, cholqr_time_mean = 0;
        double cholqr_max_orth_mean = 0;
        int cholqr_orth_count = 0;

        for (const auto& r : results) {
            cholqr_rel_err_mean += r.cholqr_rel_error;
            cholqr_orth_err_mean += r.cholqr_orth_error;
            cholqr_max_orth_mean += r.cholqr_max_orth_cols;
            cholqr_time_mean += r.cholqr_time;
            if (r.cholqr_is_orthonormal) cholqr_orth_count++;
        }
        cholqr_rel_err_mean /= num_runs;
        cholqr_orth_err_mean /= num_runs;
        cholqr_max_orth_mean /= num_runs;
        cholqr_time_mean /= num_runs;

        double cholqr_rel_err_std = 0, cholqr_orth_err_std = 0, cholqr_time_std = 0;
        double cholqr_max_orth_std = 0;
        for (const auto& r : results) {
            cholqr_rel_err_std += (r.cholqr_rel_error - cholqr_rel_err_mean) * (r.cholqr_rel_error - cholqr_rel_err_mean);
            cholqr_orth_err_std += (r.cholqr_orth_error - cholqr_orth_err_mean) * (r.cholqr_orth_error - cholqr_orth_err_mean);
            cholqr_max_orth_std += (r.cholqr_max_orth_cols - cholqr_max_orth_mean) * (r.cholqr_max_orth_cols - cholqr_max_orth_mean);
            cholqr_time_std += (r.cholqr_time - cholqr_time_mean) * (r.cholqr_time - cholqr_time_mean);
        }
        cholqr_rel_err_std = std::sqrt(cholqr_rel_err_std / num_runs);
        cholqr_orth_err_std = std::sqrt(cholqr_orth_err_std / num_runs);
        cholqr_max_orth_std = std::sqrt(cholqr_max_orth_std / num_runs);
        cholqr_time_std = std::sqrt(cholqr_time_std / num_runs);

        double cholqr_orth_rate = static_cast<double>(cholqr_orth_count) / num_runs;

        // Compute statistics for sCholQR3
        double scholqr3_rel_err_mean = 0, scholqr3_orth_err_mean = 0, scholqr3_time_mean = 0;
        double scholqr3_max_orth_mean = 0;
        int scholqr3_orth_count = 0;

        for (const auto& r : results) {
            scholqr3_rel_err_mean += r.scholqr3_rel_error;
            scholqr3_orth_err_mean += r.scholqr3_orth_error;
            scholqr3_max_orth_mean += r.scholqr3_max_orth_cols;
            scholqr3_time_mean += r.scholqr3_time;
            if (r.scholqr3_is_orthonormal) scholqr3_orth_count++;
        }
        scholqr3_rel_err_mean /= num_runs;
        scholqr3_orth_err_mean /= num_runs;
        scholqr3_max_orth_mean /= num_runs;
        scholqr3_time_mean /= num_runs;

        double scholqr3_rel_err_std = 0, scholqr3_orth_err_std = 0, scholqr3_time_std = 0;
        double scholqr3_max_orth_std = 0;
        for (const auto& r : results) {
            scholqr3_rel_err_std += (r.scholqr3_rel_error - scholqr3_rel_err_mean) * (r.scholqr3_rel_error - scholqr3_rel_err_mean);
            scholqr3_orth_err_std += (r.scholqr3_orth_error - scholqr3_orth_err_mean) * (r.scholqr3_orth_error - scholqr3_orth_err_mean);
            scholqr3_max_orth_std += (r.scholqr3_max_orth_cols - scholqr3_max_orth_mean) * (r.scholqr3_max_orth_cols - scholqr3_max_orth_mean);
            scholqr3_time_std += (r.scholqr3_time - scholqr3_time_mean) * (r.scholqr3_time - scholqr3_time_mean);
        }
        scholqr3_rel_err_std = std::sqrt(scholqr3_rel_err_std / num_runs);
        scholqr3_orth_err_std = std::sqrt(scholqr3_orth_err_std / num_runs);
        scholqr3_max_orth_std = std::sqrt(scholqr3_max_orth_std / num_runs);
        scholqr3_time_std = std::sqrt(scholqr3_time_std / num_runs);

        double scholqr3_orth_rate = static_cast<double>(scholqr3_orth_count) / num_runs;

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
            << scholqr3_time_mean << "," << scholqr3_time_std << "\n";
        out.flush();

        printf("  Summary:\n");
        printf("    CQRRT:   orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               cqrrt_orth_err_mean, cqrrt_orth_err_std, cqrrt_max_orth_mean, cqrrt_max_orth_std, cqrrt_orth_rate);
        printf("    CholQR:  orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               cholqr_orth_err_mean, cholqr_orth_err_std, cholqr_max_orth_mean, cholqr_max_orth_std, cholqr_orth_rate);
        printf("    sCholQR3: orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n\n",
               scholqr3_orth_err_mean, scholqr3_orth_err_std, scholqr3_max_orth_mean, scholqr3_max_orth_std, scholqr3_orth_rate);

        // Write runtime breakdown from fastest runs for all algorithms
        breakdown << std::scientific << std::setprecision(6)
                  << cond_num << ","
                  // CQRRT (10 values)
                  << fastest_cqrrt.cqrrt_saso_time << ","
                  << fastest_cqrrt.cqrrt_qr_time << ","
                  << fastest_cqrrt.cqrrt_trtri_time << ","
                  << fastest_cqrrt.cqrrt_linop_precond_time << ","
                  << fastest_cqrrt.cqrrt_linop_gram_time << ","
                  << fastest_cqrrt.cqrrt_trmm_gram_time << ","
                  << fastest_cqrrt.cqrrt_potrf_time << ","
                  << fastest_cqrrt.cqrrt_finalize_time << ","
                  << fastest_cqrrt.cqrrt_rest_time << ","
                  << fastest_cqrrt.cqrrt_time << ","
                  // CholQR (5 values)
                  << fastest_cholqr.cholqr_materialize_time << ","
                  << fastest_cholqr.cholqr_gram_time << ","
                  << fastest_cholqr.cholqr_potrf_time << ","
                  << fastest_cholqr.cholqr_rest_time << ","
                  << fastest_cholqr.cholqr_time << ","
                  // sCholQR3 (12 values)
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
                  << fastest_scholqr3.scholqr3_time << "\n";
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
