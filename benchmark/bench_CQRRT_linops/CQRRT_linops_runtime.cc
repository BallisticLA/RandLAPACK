#if defined(__APPLE__)
int main() {return 0;}
#else

// CQRRT_linops runtime benchmark - measures time for CQRRT with nested composite linear operators
// Nested composite: CholSolver * (Sparse * Gaussian) representing A^{-1} * (S * G)
// SPD matrix from Matrix Market file, Sparse (SASO) and Gaussian matrices generated randomly

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

// Need to include demos utilities for CholSolverLinOp and dm_cqrrt_linops
#include "../../demos/functions/linops_external/dm_cholsolver_linop.hh"
#include "../../demos/functions/drivers/dm_cqrrt_linops.hh"
#include "../../demos/functions/misc/dm_util.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

template <typename T>
struct CQRRT_linops_benchmark_data {
    int64_t m;  // Rows (from SPD matrix dimension)
    int64_t n;  // Columns (from sparse matrix columns)
    T d_factor;
    std::vector<T> R;
    std::vector<T> A_dense;  // Dense representation for verification

    CQRRT_linops_benchmark_data(int64_t rows, int64_t cols, T sampling_factor) :
        m(rows),
        n(cols),
        d_factor(sampling_factor),
        R(cols * cols, 0.0),
        A_dense(rows * cols, 0.0)
    {}
};

// Error checking utility - verifies A = QR factorization and Q orthogonality
template <typename T>
static void verify_factorization(
    const T* Q, int64_t Q_rows, int64_t Q_cols,
    const T* R, int64_t ldr,
    const T* A_expected, int64_t m, int64_t n,
    T& rel_error, T& orth_error, int64_t& num_orthonormal_cols) {

    // Compute Q * R
    std::vector<T> QR(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n,
               1.0, Q, m, R, ldr, 0.0, QR.data(), m);

    // Compute ||A - QR||_F
    T norm_diff = 0.0;
    for (int64_t i = 0; i < m * n; ++i) {
        T diff = A_expected[i] - QR[i];
        norm_diff += diff * diff;
    }
    norm_diff = std::sqrt(norm_diff);

    // Compute ||A||_F
    T norm_A = lapack::lange(Norm::Fro, m, n, A_expected, m);

    // Relative error
    rel_error = norm_diff / norm_A;

    // Check orthogonality: ||Q^T Q - I||_F
    std::vector<T> QtQ(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m,
               1.0, Q, m, Q, m, 0.0, QtQ.data(), n);

    // Compute overall orthogonality error
    std::vector<T> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    for (int64_t i = 0; i < n * n; ++i) {
        QtQ[i] -= I_ref[i];
    }
    T norm_orth = lapack::lange(Norm::Fro, n, n, QtQ.data(), n);
    orth_error = norm_orth / std::sqrt((T) n);

    // Count how many columns are orthonormal
    // A column i is considered orthonormal if:
    // 1. ||q_i||_2 ≈ 1 (diagonal element Q^T Q [i,i] ≈ 1)
    // 2. q_i ⊥ q_j for all j ≠ i (off-diagonal elements Q^T Q [i,j] ≈ 0)
    T tol = 100 * std::numeric_limits<T>::epsilon() * m;  // Tolerance scaled by problem size
    num_orthonormal_cols = 0;

    // Recompute Q^T Q with original values
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m,
               1.0, Q, m, Q, m, 0.0, QtQ.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        bool is_orthonormal = true;

        // Check diagonal: should be ≈ 1
        T diag_error = std::abs(QtQ[i + i * n] - 1.0);
        if (diag_error > tol) {
            is_orthonormal = false;
        }

        // Check off-diagonals in column i: should be ≈ 0
        if (is_orthonormal) {
            for (int64_t j = 0; j < n; ++j) {
                if (j != i) {
                    T offdiag_error = std::abs(QtQ[j + i * n]);
                    if (offdiag_error > tol) {
                        is_orthonormal = false;
                        break;
                    }
                }
            }
        }

        if (is_orthonormal) {
            num_orthonormal_cols++;
        }
    }
}

template <typename T, typename RNG>
static void run_benchmark(
    const std::string& spd_filename,
    int64_t k_dim,
    int64_t n_cols,
    T saso_density,
    int64_t numruns,
    T d_factor,
    CQRRT_linops_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    const std::string& output_filename) {

    printf("Loading SPD matrix from: %s\n", spd_filename.c_str());
    printf("Generating SASO (Sparse) matrix: %ld x %ld with density %.3f\n", data.m, k_dim, saso_density);
    printf("Generating Gaussian matrix: %ld x %ld\n", k_dim, n_cols);

    // Create CholSolverLinOp from SPD matrix file
    RandLAPACK_demos::CholSolverLinOp<T> A_inv_linop(spd_filename);

    // Verify SPD matrix dimension matches expected m
    if (A_inv_linop.n_rows != data.m) {
        throw std::runtime_error("SPD matrix dimension doesn't match expected m");
    }

    // Generate SASO (Sparse) matrix: m × k_dim
    auto saso_coo = RandLAPACK::gen::gen_sparse_mat<T>(data.m, k_dim, saso_density, state);
    RandBLAS::sparse_data::csc::CSCMatrix<T> saso_csc(data.m, k_dim);
    RandBLAS::sparse_data::conversions::coo_to_csc(saso_coo, saso_csc);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>> saso_linop(
        data.m, k_dim, saso_csc);

    // Generate Gaussian matrix: k_dim × n_cols
    std::vector<T> gaussian_mat(k_dim * n_cols);
    RandBLAS::DenseDist gaussian_dist(k_dim, n_cols);
    RandBLAS::fill_dense(gaussian_dist, gaussian_mat.data(), state);
    RandLAPACK::linops::DenseLinOp<T> gaussian_linop(k_dim, n_cols, gaussian_mat.data(), k_dim, Layout::ColMajor);

    // Create inner composite: Sparse * Gaussian
    RandLAPACK::linops::CompositeOperator inner_composite(data.m, n_cols, saso_linop, gaussian_linop);

    // Create outer composite: CholSolver * (Sparse * Gaussian)
    RandLAPACK::linops::CompositeOperator outer_composite(data.m, data.n, A_inv_linop, inner_composite);

    printf("SPD matrix dimension: %ld x %ld\n", A_inv_linop.n_rows, A_inv_linop.n_cols);
    printf("SASO matrix dimension: %ld x %ld (nnz: %ld)\n", saso_csc.n_rows, saso_csc.n_cols, saso_csc.nnz);
    printf("Gaussian matrix dimension: %ld x %ld\n", k_dim, n_cols);
    printf("Nested composite dimension: %ld x %ld\n", data.m, data.n);

    // Compute dense representation for verification
    printf("Computing dense representation of nested composite operator...\n");

    // Step 1: Densify SASO and compute SASO * Gaussian -> intermediate
    std::vector<T> saso_dense(data.m * k_dim, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(saso_csc, Layout::ColMajor, saso_dense.data());

    std::vector<T> intermediate(data.m * n_cols, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, data.m, n_cols, k_dim, 1.0,
              saso_dense.data(), data.m, gaussian_mat.data(), k_dim, 0.0, intermediate.data(), data.m);

    // Step 2: Compute CholSolver * intermediate -> final dense representation
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, data.m, data.n, data.m,
                1.0, intermediate.data(), data.m, 0.0, data.A_dense.data(), data.m);
    printf("Dense representation computed.\n");

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);

    // Timing results storage
    std::vector<long> timing_results_R_only;
    std::vector<long> timing_results_QR;
    T rel_error, orth_error;
    int64_t num_orthonormal_cols;

    printf("\nRunning %ld iterations of CQRRT_linops...\n", numruns);

    for (int64_t i = 0; i < numruns; ++i) {
        printf("\n=== Iteration %ld/%ld ===\n", i + 1, numruns);

        // ===== MODE 1: R-only (test_mode=false) =====
        printf("Running R-only mode...\n");
        std::fill(data.R.begin(), data.R.end(), 0.0);
        auto state_run = state;

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_R_only(true, tol, false);  // timing=true, test_mode=false
        CQRRT_R_only.nnz = 2;
        CQRRT_R_only.call(outer_composite, data.R.data(), data.n, d_factor, state_run);

        timing_results_R_only = CQRRT_R_only.times;
        printf("  R-only total time: %ld microseconds\n", timing_results_R_only.back());

        // ===== MODE 2: Q+R (test_mode=true) =====
        printf("Running Q+R mode...\n");
        std::fill(data.R.begin(), data.R.end(), 0.0);
        state_run = state;  // Reset to same RNG state

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
        CQRRT_QR.nnz = 2;
        CQRRT_QR.call(outer_composite, data.R.data(), data.n, d_factor, state_run);

        timing_results_QR = CQRRT_QR.times;
        printf("  Q+R total time: %ld microseconds\n", timing_results_QR.back());

        // ===== VERIFICATION =====
        printf("Verifying factorization...\n");
        verify_factorization(CQRRT_QR.Q, CQRRT_QR.Q_rows, CQRRT_QR.Q_cols,
                            data.R.data(), data.n,
                            data.A_dense.data(), data.m, data.n,
                            rel_error, orth_error, num_orthonormal_cols);

        printf("  ||A - QR|| / ||A||: %.6e\n", rel_error);
        printf("  ||Q'Q - I|| / sqrt(n): %.6e\n", orth_error);
        printf("  Orthonormal columns: %ld / %ld\n", num_orthonormal_cols, data.n);

        // Check if errors are acceptable
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        if (rel_error > atol || orth_error > atol) {
            printf("  WARNING: Errors exceed tolerance (%.6e)!\n", atol);
        } else {
            printf("  Verification passed!\n");
        }

        // Write results to file
        // Format: R-only timings (6 values), Q+R timings (6 values), rel_error, orth_error, num_orthonormal_cols
        std::ofstream file(output_filename, std::ios::out | std::ios::app);

        // R-only timings
        for (size_t j = 0; j < timing_results_R_only.size(); ++j) {
            file << timing_results_R_only[j] << ", ";
        }

        // Q+R timings
        for (size_t j = 0; j < timing_results_QR.size(); ++j) {
            file << timing_results_QR[j] << ", ";
        }

        // Error metrics and orthonormal column count
        file << std::scientific << std::setprecision(6)
             << rel_error << ", " << orth_error << ", "
             << num_orthonormal_cols << "\n";
        file.flush();
    }

    printf("\nBenchmark complete. Results written to: %s\n", output_filename.c_str());
}

int main(int argc, char *argv[]) {

    if (argc != 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_dir> <spd_matrix.mtx> <k_dim> <n_cols> <saso_density> <d_factor> <num_runs>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  output_dir     : Directory for output file (use '.' for current dir)" << std::endl;
        std::cerr << "  spd_matrix.mtx : Path to SPD matrix in Matrix Market format (determines m)" << std::endl;
        std::cerr << "  k_dim          : Intermediate dimension (SASO cols / Gaussian rows)" << std::endl;
        std::cerr << "  n_cols         : Final number of columns (Gaussian cols)" << std::endl;
        std::cerr << "  saso_density   : Density for sparse SASO matrix (e.g., 0.1)" << std::endl;
        std::cerr << "  d_factor       : Sketching dimension factor (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs       : Number of benchmark iterations" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string output_dir = argv[1];
    std::string spd_filename = argv[2];
    int64_t k_dim = std::stol(argv[3]);
    int64_t n_cols = std::stol(argv[4]);
    double saso_density = std::stod(argv[5]);
    double d_factor = std::stod(argv[6]);
    int64_t numruns = std::stol(argv[7]);

    // Helper lambda to read SPD matrix dimension (square matrix)
    auto read_spd_dimension = [](const std::string& filename) -> int64_t {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        // Skip comment lines
        std::string line;
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        // Parse dimensions
        std::istringstream iss(line);
        int64_t rows, cols, nnz;
        iss >> rows >> cols >> nnz;
        file.close();

        if (rows != cols) {
            throw std::runtime_error("SPD matrix must be square");
        }

        return rows;
    };

    // Read SPD matrix dimension
    int64_t m = read_spd_dimension(spd_filename);
    int64_t n = n_cols;

    // Validate dimensions
    if (k_dim <= 0 || n_cols <= 0) {
        std::cerr << "Error: k_dim and n_cols must be positive" << std::endl;
        return 1;
    }

    if (saso_density <= 0.0 || saso_density > 1.0) {
        std::cerr << "Error: saso_density must be in (0, 1]" << std::endl;
        return 1;
    }

    printf("\n=== CQRRT_linops Benchmark (Nested Composite) ===\n");
    printf("Nested composite: A^{-1} * (S * G)\n");
    printf("  A (SPD): %ld x %ld (from file)\n", m, m);
    printf("  S (SASO): %ld x %ld (generated, density: %.3f)\n", m, k_dim, saso_density);
    printf("  G (Gaussian): %ld x %ld (generated)\n", k_dim, n);
    printf("  Final composite: %ld x %ld\n", m, n);
    printf("Sketching factor: %.2f\n", d_factor);
    printf("Number of runs: %ld\n", numruns);
    printf("=================================================\n\n");

    // Allocate benchmark data
    CQRRT_linops_benchmark_data<double> bench_data(m, n, d_factor);

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Prepare output file
    std::string output_filename = "_CQRRT_linops_runtime_num_info_lines_7.txt";
    std::string output_path;
    if (output_dir != ".") {
        output_path = output_dir + "/" + output_filename;
    } else {
        output_path = output_filename;
    }

    std::ofstream file(output_path, std::ios::out | std::ios::trunc);  // Clear file first

    // Write header information
    file << "Description: CQRRT_linops runtime benchmark with nested composite operator (CholSolver * (SASO * Gaussian))\n"
         << "File format: 14 columns - R-only mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
         << "Q+R mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
         << "Errors (2): rel_error, orth_error (times in microseconds, errors unitless)\n"
         << "Each row represents one iteration with both R-only and Q+R runs plus verification\n"
         << "Num OMP threads: " << RandLAPACK::util::get_omp_threads() << "\n"
         << "SPD matrix file: " << spd_filename << "\n"
         << "SPD matrix dimension (m): " << m << "\n"
         << "Intermediate dimension (k): " << k_dim << "\n"
         << "Final columns (n): " << n << "\n"
         << "SASO density: " << saso_density << "\n"
         << "d_factor: " << d_factor << "\n"
         << "num_runs: " << numruns << "\n";
    file.flush();
    file.close();

    // Run benchmark
    auto start_time = steady_clock::now();
    run_benchmark(spd_filename, k_dim, n_cols, saso_density, numruns, d_factor, bench_data, state, output_path);
    auto stop_time = steady_clock::now();
    long total_time = duration_cast<microseconds>(stop_time - start_time).count();

    // Append total benchmark time
    file.open(output_path, std::ios::out | std::ios::app);
    file << "Total benchmark execution time: " << total_time << " microseconds\n";
    file.flush();

    return 0;
}
#endif
