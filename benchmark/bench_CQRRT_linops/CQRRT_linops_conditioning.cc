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
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

// Need to include demos utilities
#include "../../demos/functions/linops_external/dm_cholsolver_linop.hh"
#include "../../demos/functions/drivers/dm_cqrrt_linops.hh"
#include "../../demos/functions/drivers/dm_cholqr_linops.hh"
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

    // CholQR results
    T cholqr_rel_error;          // ||A - QR|| / ||A||
    T cholqr_orth_error;         // ||Q^T Q - I|| / sqrt(n)
    bool cholqr_is_orthonormal;  // Is full Q block orthonormal?
    int64_t cholqr_max_orth_cols;  // Maximum orthonormal prefix
    long cholqr_time;      // Total computation time (microseconds)
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
        auto t_start = steady_clock::now();

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_QR(false, tol, true);  // timing=false, test_mode=true
        CQRRT_QR.nnz = 5;  // Optimal for sparse SPD matrices (from parameter study)
        CQRRT_QR.call(outer_composite, R_cqrrt.data(), n, d_factor, state);

        auto t_stop = steady_clock::now();
        result.cqrrt_time = duration_cast<microseconds>(t_stop - t_start).count();

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
        auto t_start = steady_clock::now();

        RandLAPACK_demos::CholQR_linops<T> CholQR_alg(false, tol, true);  // timing=false, test_mode=true
        CholQR_alg.call(outer_composite, R_cholqr.data(), n);

        auto t_stop = steady_clock::now();
        result.cholqr_time = duration_cast<microseconds>(t_stop - t_start).count();

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
    std::string output_file = argv[6];

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

    printf("\n=== CQRRT vs CholQR Conditioning Study ===\n");
    printf("SPD matrix directory: %s\n", spd_dir.c_str());
    printf("Number of condition numbers: %ld\n", num_matrices);
    printf("Matrix dimensions: %ld x %ld x %ld\n", m, k_dim, n);
    printf("Sketching factor (CQRRT only): %.2f\n", d_factor);
    printf("Runs per condition: %ld\n", num_runs);
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
    out << "# CQRRT vs CholQR Conditioning Study Results\n";
    out << "# Composite operator: CholSolver(κ) * (SASO * Gaussian)\n";
    out << "# Matrix dimensions: " << m << " x " << k_dim << " x " << n << "\n";
    out << "# d_factor (CQRRT only): " << d_factor << "\n";
    out << "# num_runs: " << num_runs << "\n";
    out << "# Format: cond_num, cqrrt_*, cholqr_* (rel_error, orth_error, max_orth_cols, orth_rate, time)\n";
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
        << "cholqr_time_mean,cholqr_time_std\n";

    // Run conditioning study
    for (size_t i = 0; i < matrix_files.size(); ++i) {
        double cond_num = matrix_files[i].first;
        std::string filepath = matrix_files[i].second;

        printf("Testing condition number %.6e (%zu/%zu)...\n", cond_num, i + 1, matrix_files.size());

        // Run multiple times for statistics
        std::vector<conditioning_result<double>> results;
        for (int64_t run = 0; run < num_runs; ++run) {
            auto result = run_single_test<double>(filepath, cond_num, m, k_dim, n, d_factor, state);
            results.push_back(result);

            printf("  Run %ld/%ld:\n", run + 1, num_runs);
            printf("    CQRRT: orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.cqrrt_orth_error, result.cqrrt_max_orth_cols, n, result.cqrrt_time);
            printf("    CholQR: orth_error=%.6e, max_orth_cols=%ld/%ld, time=%ld μs\n",
                   result.cholqr_orth_error, result.cholqr_max_orth_cols, n, result.cholqr_time);
        }

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
            << cholqr_time_mean << "," << cholqr_time_std << "\n";
        out.flush();

        printf("  Summary:\n");
        printf("    CQRRT:  orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n",
               cqrrt_orth_err_mean, cqrrt_orth_err_std, cqrrt_max_orth_mean, cqrrt_max_orth_std, cqrrt_orth_rate);
        printf("    CholQR: orth_error=%.6e±%.6e, max_orth=%.1f±%.1f, orth_rate=%.2f\n\n",
               cholqr_orth_err_mean, cholqr_orth_err_std, cholqr_max_orth_mean, cholqr_max_orth_std, cholqr_orth_rate);
    }

    out.close();
    printf("========================================\n");
    printf("Conditioning study complete!\n");
    printf("Results saved to: %s\n", output_file.c_str());
    printf("========================================\n");

    return 0;
}
#endif
