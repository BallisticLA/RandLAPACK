#if defined(__APPLE__)
int main() {return 0;}
#else

// CQRRT_linops runtime benchmark - measures time for CQRRT with composite linear operators
// Composite operator: CholSolver (left) * Sparse (right) representing A^{-1} * B
// Matrices sourced from user-provided Matrix Market files

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
    T& rel_error, T& orth_error) {

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
    std::vector<T> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m,
               1.0, Q, m, -1.0, I_ref.data(), n);
    T norm_orth = lapack::lansy(Norm::Fro, Uplo::Upper, n, I_ref.data(), n);

    // Normalized orthogonality error
    orth_error = norm_orth / std::sqrt((T) n);
}

template <typename T, typename RNG>
static void run_benchmark(
    const std::string& spd_filename,
    const std::string& sparse_filename,
    int64_t numruns,
    T d_factor,
    CQRRT_linops_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    const std::string& output_filename) {

    printf("Loading SPD matrix from: %s\n", spd_filename.c_str());
    printf("Loading sparse matrix from: %s\n", sparse_filename.c_str());

    // Create CholSolverLinOp from SPD matrix file
    RandLAPACK_demos::CholSolverLinOp<T> A_inv_linop(spd_filename);

    // Load sparse matrix from Matrix Market file using Eigen
    Eigen::SparseMatrix<T> B_eigen;
    if (!Eigen::loadMarket(B_eigen, sparse_filename)) {
        throw std::runtime_error("Failed to load sparse matrix from: " + sparse_filename);
    }

    // Convert Eigen sparse matrix to RandBLAS CSC format
    B_eigen.makeCompressed();  // Ensure CSC format
    int64_t B_rows = B_eigen.rows();
    int64_t B_cols = B_eigen.cols();
    int64_t B_nnz = B_eigen.nonZeros();

    // Allocate RandBLAS CSC matrix using reserve_csc
    RandBLAS::sparse_data::csc::CSCMatrix<T> B_csc(B_rows, B_cols);
    RandBLAS::sparse_data::csc::reserve_csc(B_nnz, B_csc);

    // Copy data from Eigen to RandBLAS format (raw pointers)
    std::copy(B_eigen.valuePtr(), B_eigen.valuePtr() + B_nnz, B_csc.vals);
    std::copy(B_eigen.innerIndexPtr(), B_eigen.innerIndexPtr() + B_nnz, B_csc.rowidxs);
    std::copy(B_eigen.outerIndexPtr(), B_eigen.outerIndexPtr() + B_cols + 1, B_csc.colptr);

    printf("SPD matrix dimension: %ld x %ld\n", A_inv_linop.n_rows, A_inv_linop.n_cols);
    printf("Sparse matrix dimension: %ld x %ld\n", B_csc.n_rows, B_csc.n_cols);
    printf("Composite operator dimension: %ld x %ld\n", data.m, data.n);

    // Verify dimensions match
    if (A_inv_linop.n_rows != data.m || B_csc.n_cols != data.n) {
        throw std::runtime_error("Matrix dimensions don't match benchmark data");
    }
    if (A_inv_linop.n_cols != B_csc.n_rows) {
        throw std::runtime_error("Inner dimensions of composite don't match");
    }

    // Create SparseLinOp
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>> B_sp_linop(
        B_csc.n_rows, B_csc.n_cols, B_csc);

    // Create CompositeOperator
    RandLAPACK::linops::CompositeOperator A_composite(data.m, data.n, A_inv_linop, B_sp_linop);

    // Convert sparse matrix to dense for composite computation
    std::vector<T> B_dense(B_csc.n_rows * B_csc.n_cols, 0.0);
    RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense.data());

    // Compute dense representation for verification
    printf("Computing dense representation of composite operator...\n");
    A_inv_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans, data.m, data.n, B_csc.n_rows,
                1.0, B_dense.data(), B_csc.n_rows, 0.0, data.A_dense.data(), data.m);
    printf("Dense representation computed.\n");

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);

    // Timing results storage
    std::vector<long> timing_results_R_only;
    std::vector<long> timing_results_QR;
    T rel_error, orth_error;

    printf("\nRunning %ld iterations of CQRRT_linops...\n", numruns);

    for (int64_t i = 0; i < numruns; ++i) {
        printf("\n=== Iteration %ld/%ld ===\n", i + 1, numruns);

        // ===== MODE 1: R-only (test_mode=false) =====
        printf("Running R-only mode...\n");
        std::fill(data.R.begin(), data.R.end(), 0.0);
        auto state_run = state;

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_R_only(true, tol, false);  // timing=true, test_mode=false
        CQRRT_R_only.nnz = 2;
        CQRRT_R_only.call(A_composite, data.R.data(), data.n, d_factor, state_run);

        timing_results_R_only = CQRRT_R_only.times;
        printf("  R-only total time: %ld microseconds\n", timing_results_R_only.back());

        // ===== MODE 2: Q+R (test_mode=true) =====
        printf("Running Q+R mode...\n");
        std::fill(data.R.begin(), data.R.end(), 0.0);
        state_run = state;  // Reset to same RNG state

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
        CQRRT_QR.nnz = 2;
        CQRRT_QR.call(A_composite, data.R.data(), data.n, d_factor, state_run);

        timing_results_QR = CQRRT_QR.times;
        printf("  Q+R total time: %ld microseconds\n", timing_results_QR.back());

        // ===== VERIFICATION =====
        printf("Verifying factorization...\n");
        verify_factorization(CQRRT_QR.Q, CQRRT_QR.Q_rows, CQRRT_QR.Q_cols,
                            data.R.data(), data.n,
                            data.A_dense.data(), data.m, data.n,
                            rel_error, orth_error);

        printf("  ||A - QR|| / ||A||: %.6e\n", rel_error);
        printf("  ||Q'Q - I|| / sqrt(n): %.6e\n", orth_error);

        // Check if errors are acceptable
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        if (rel_error > atol || orth_error > atol) {
            printf("  WARNING: Errors exceed tolerance (%.6e)!\n", atol);
        } else {
            printf("  Verification passed!\n");
        }

        // Write results to file
        // Format: R-only timings (6 values), Q+R timings (6 values), rel_error, orth_error
        std::ofstream file(output_filename, std::ios::out | std::ios::app);

        // R-only timings
        for (size_t j = 0; j < timing_results_R_only.size(); ++j) {
            file << timing_results_R_only[j] << ", ";
        }

        // Q+R timings
        for (size_t j = 0; j < timing_results_QR.size(); ++j) {
            file << timing_results_QR[j] << ", ";
        }

        // Error metrics
        file << std::scientific << std::setprecision(6)
             << rel_error << ", " << orth_error << "\n";
        file.flush();
    }

    printf("\nBenchmark complete. Results written to: %s\n", output_filename.c_str());
}

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_dir> <spd_matrix.mtx> <sparse_matrix.mtx> <d_factor> <num_runs>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  output_dir       : Directory for output file (use '.' for current dir)" << std::endl;
        std::cerr << "  spd_matrix.mtx   : Path to SPD matrix in Matrix Market format" << std::endl;
        std::cerr << "  sparse_matrix.mtx: Path to sparse matrix in Matrix Market format" << std::endl;
        std::cerr << "  d_factor         : Sketching dimension factor (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs         : Number of benchmark iterations" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string output_dir = argv[1];
    std::string spd_filename = argv[2];
    std::string sparse_filename = argv[3];
    double d_factor = std::stod(argv[4]);
    int64_t numruns = std::stol(argv[5]);

    // Helper lambda to read Matrix Market dimensions
    auto read_mm_dimensions = [](const std::string& filename) -> std::pair<int64_t, int64_t> {
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

        return {rows, cols};
    };

    // Read matrix dimensions
    auto [spd_rows, spd_cols] = read_mm_dimensions(spd_filename);
    auto [sparse_rows, sparse_cols] = read_mm_dimensions(sparse_filename);

    if (spd_rows != spd_cols) {
        std::cerr << "Error: SPD matrix must be square" << std::endl;
        return 1;
    }

    if (spd_cols != sparse_rows) {
        std::cerr << "Error: SPD matrix columns (" << spd_cols
                  << ") must match sparse matrix rows (" << sparse_rows << ")" << std::endl;
        return 1;
    }

    int64_t m = spd_rows;
    int64_t n = sparse_cols;

    printf("\n=== CQRRT_linops Benchmark ===\n");
    printf("Composite operator: A^{-1} * B\n");
    printf("  A (SPD): %ld x %ld\n", spd_rows, spd_cols);
    printf("  B (sparse): %ld x %ld\n", sparse_rows, sparse_cols);
    printf("  Composite: %ld x %ld\n", m, n);
    printf("Sketching factor: %.2f\n", d_factor);
    printf("Number of runs: %ld\n", numruns);
    printf("==============================\n\n");

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
    file << "Description: CQRRT_linops runtime benchmark with composite operator (CholSolver * Sparse)\n"
         << "File format: 14 columns - R-only mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
         << "Q+R mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
         << "Errors (2): rel_error, orth_error (times in microseconds, errors unitless)\n"
         << "Each row represents one iteration with both R-only and Q+R runs plus verification\n"
         << "Num OMP threads: " << RandLAPACK::util::get_omp_threads() << "\n"
         << "SPD matrix file: " << spd_filename << "\n"
         << "Sparse matrix file: " << sparse_filename << "\n"
         << "Composite dimensions: " << m << " x " << n << "\n"
         << "d_factor: " << d_factor << "\n"
         << "num_runs: " << numruns << "\n";
    file.flush();
    file.close();

    // Run benchmark
    auto start_time = steady_clock::now();
    run_benchmark(spd_filename, sparse_filename, numruns, d_factor, bench_data, state, output_path);
    auto stop_time = steady_clock::now();
    long total_time = duration_cast<microseconds>(stop_time - start_time).count();

    // Append total benchmark time
    file.open(output_path, std::ios::out | std::ios::app);
    file << "Total benchmark execution time: " << total_time << " microseconds\n";
    file.flush();

    return 0;
}
#endif
