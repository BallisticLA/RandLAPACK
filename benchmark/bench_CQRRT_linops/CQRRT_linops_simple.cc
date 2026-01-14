#if defined(__APPLE__)
int main() {return 0;}
#else

// Simple CQRRT_linops benchmark - uses dense matrices with controlled condition numbers
// Tests CQRRT with well-conditioned (κ=10) and ill-conditioned (κ=10^6) dense matrices

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>

// Need to include demos utilities for dm_cqrrt_linops
#include "../../demos/functions/drivers/dm_cqrrt_linops.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

template <typename T>
struct CQRRT_simple_benchmark_data {
    int64_t m;  // Rows
    int64_t n;  // Columns
    T d_factor;
    std::vector<T> R;
    std::vector<T> A;  // Dense matrix

    CQRRT_simple_benchmark_data(int64_t rows, int64_t cols, T sampling_factor) :
        m(rows),
        n(cols),
        d_factor(sampling_factor),
        R(cols * cols, 0.0),
        A(rows * cols, 0.0)
    {}
};

// Error checking utility - verifies A = QR factorization and Q orthogonality
template <typename T>
static void verify_factorization(
    const T* Q, int64_t m, int64_t n,
    const T* R, int64_t ldr,
    const T* A_expected,
    T& rel_error, T& orth_error, bool& is_orthonormal, int64_t& max_orth_cols) {

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

    // Compute overall orthogonality error for full block
    std::vector<T> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    for (int64_t i = 0; i < n * n; ++i) {
        QtQ[i] -= I_ref[i];
    }
    T norm_orth = lapack::lange(Norm::Fro, n, n, QtQ.data(), n);
    orth_error = norm_orth / std::sqrt((T) n);

    // Tolerance for orthonormality check
    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
    is_orthonormal = (orth_error <= tol);

    // Find the largest k such that Q[:, 0:k] is orthonormal
    // Binary search for the maximum orthonormal prefix
    max_orth_cols = 0;

    for (int64_t k = 1; k <= n; ++k) {
        // Compute Q[:, 0:k]^T * Q[:, 0:k] for the first k columns
        std::vector<T> QtQ_k(k * k, 0.0);
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
                   1.0, Q, m, Q, m, 0.0, QtQ_k.data(), k);

        // Compute ||Q_k^T Q_k - I_k||_F / sqrt(k)
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
            // Once we fail, no point checking larger blocks
            break;
        }
    }
}

template <typename T, typename RNG>
static void run_benchmark(
    int64_t m,
    int64_t n,
    T cond_num,
    int64_t numruns,
    T d_factor,
    CQRRT_simple_benchmark_data<T>& data,
    RandBLAS::RNGState<RNG>& state,
    const std::string& output_filename,
    const std::string& matrix_label) {

    printf("\n=== Benchmark: %s (κ=%.2e) ===\n", matrix_label.c_str(), cond_num);
    printf("Generating dense matrix: %ld x %ld\n", m, n);

    // Generate dense matrix with polynomial decay using gen_poly_mat
    // gen_poly_mat creates matrix with SVD: A = U * Σ * V^T
    // where singular values decay polynomially from 1 to 1/cond_num
    T frac_spectrum_one = 0.1;  // First 10% of singular values = 1
    T exponent = 2.0;           // Polynomial decay exponent
    bool diagon = false;        // Generate full matrix (not diagonal)

    RandLAPACK::gen::gen_poly_mat(m, n, data.A.data(), n, frac_spectrum_one,
                                   cond_num, exponent, diagon, state);

    printf("Matrix generated: %ld x %ld with κ=%.2e\n", m, n, cond_num);

    // Create DenseLinOp wrapper
    RandLAPACK::linops::DenseLinOp<T> A_linop(m, n, data.A.data(), m, Layout::ColMajor);

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);

    // Timing results storage
    std::vector<long> timing_results_R_only;
    std::vector<long> timing_results_QR;
    T rel_error, orth_error;
    bool is_orthonormal;
    int64_t max_orth_cols;

    printf("Running %ld iterations of CQRRT_linops...\n", numruns);

    for (int64_t i = 0; i < numruns; ++i) {
        printf("\n=== Iteration %ld/%ld ===\n", i + 1, numruns);

        // ===== MODE 1: R-only (test_mode=false) =====
        printf("Running R-only mode...\n");
        std::fill(data.R.begin(), data.R.end(), 0.0);
        auto state_run = state;

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_R_only(true, tol, false);  // timing=true, test_mode=false
        CQRRT_R_only.nnz = 2;
        CQRRT_R_only.call(A_linop, data.R.data(), data.n, d_factor, state_run);

        timing_results_R_only = CQRRT_R_only.times;
        printf("  R-only total time: %ld microseconds\n", timing_results_R_only.back());

        // ===== MODE 2: Q+R (test_mode=true) =====
        printf("Running Q+R mode...\n");
        std::fill(data.R.begin(), data.R.end(), 0.0);
        state_run = state;  // Reset to same RNG state

        RandLAPACK_demos::CQRRT_linops<T, RNG> CQRRT_QR(true, tol, true);  // timing=true, test_mode=true
        CQRRT_QR.nnz = 2;
        CQRRT_QR.call(A_linop, data.R.data(), data.n, d_factor, state_run);

        timing_results_QR = CQRRT_QR.times;
        printf("  Q+R total time: %ld microseconds\n", timing_results_QR.back());

        // ===== VERIFICATION =====
        printf("Verifying factorization...\n");
        verify_factorization(CQRRT_QR.Q, data.m, data.n,
                            data.R.data(), data.n,
                            data.A.data(),
                            rel_error, orth_error, is_orthonormal, max_orth_cols);

        printf("  ||A - QR|| / ||A||: %.6e\n", rel_error);
        printf("  ||Q'Q - I|| / sqrt(n): %.6e\n", orth_error);
        printf("  Q block orthonormal: %s\n", is_orthonormal ? "YES" : "NO");
        printf("  Max orthonormal columns: %ld / %ld\n", max_orth_cols, data.n);

        // Check if errors are acceptable
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        if (rel_error > atol || orth_error > atol) {
            printf("  WARNING: Errors exceed tolerance (%.6e)!\n", atol);
        } else {
            printf("  Verification passed!\n");
        }

        // Write results to file
        // Format: R-only timings (6 values), Q+R timings (6 values), rel_error, orth_error, is_orthonormal (0/1), max_orth_cols
        std::ofstream file(output_filename, std::ios::out | std::ios::app);

        // R-only timings
        for (size_t j = 0; j < timing_results_R_only.size(); ++j) {
            file << timing_results_R_only[j] << ", ";
        }

        // Q+R timings
        for (size_t j = 0; j < timing_results_QR.size(); ++j) {
            file << timing_results_QR[j] << ", ";
        }

        // Error metrics, orthonormality flag, and max orthonormal columns
        file << std::scientific << std::setprecision(6)
             << rel_error << ", " << orth_error << ", "
             << (is_orthonormal ? 1 : 0) << ", " << max_orth_cols << "\n";
        file.flush();
    }

    printf("\nBenchmark complete for %s.\n", matrix_label.c_str());
}

int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <m> <n> <d_factor> <num_runs>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  m         : Number of rows" << std::endl;
        std::cerr << "  n         : Number of columns" << std::endl;
        std::cerr << "  d_factor  : Sketching dimension factor (e.g., 2.0)" << std::endl;
        std::cerr << "  num_runs  : Number of benchmark iterations per condition number" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " 1000 100 2.0 5" << std::endl;
        std::cerr << "\nThis will benchmark CQRRT on two matrices:" << std::endl;
        std::cerr << "  1. Well-conditioned: κ = 10" << std::endl;
        std::cerr << "  2. Ill-conditioned:  κ = 10^6" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    int64_t m = std::stol(argv[1]);
    int64_t n = std::stol(argv[2]);
    double d_factor = std::stod(argv[3]);
    int64_t numruns = std::stol(argv[4]);

    // Validate dimensions
    if (m <= 0 || n <= 0) {
        std::cerr << "Error: m and n must be positive" << std::endl;
        return 1;
    }

    if (n > m) {
        std::cerr << "Error: n must be <= m (overdetermined system)" << std::endl;
        return 1;
    }

    if (d_factor < 1.0) {
        std::cerr << "Error: d_factor must be >= 1.0" << std::endl;
        return 1;
    }

    printf("\n=== CQRRT_linops Simple Benchmark (Dense Matrices) ===\n");
    printf("Matrix dimensions: %ld x %ld\n", m, n);
    printf("Sketching factor: %.2f\n", d_factor);
    printf("Number of runs per condition: %ld\n", numruns);
    printf("Num OMP threads: %d\n", RandLAPACK::util::get_omp_threads());
    printf("=======================================================\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Prepare output files
    std::string output_well = "CQRRT_simple_well_conditioned.txt";
    std::string output_ill = "CQRRT_simple_ill_conditioned.txt";

    // Well-conditioned matrix (κ = 10)
    {
        std::ofstream file(output_well, std::ios::out | std::ios::trunc);
        file << "Description: CQRRT_linops simple benchmark - Well-conditioned dense matrix (κ=10)\n"
             << "File format: 16 columns - R-only mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
             << "Q+R mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
             << "Errors (4): rel_error, orth_error, is_orthonormal (0/1), max_orth_cols (times in microseconds)\n"
             << "Each row represents one iteration with both R-only and Q+R runs plus verification\n"
             << "Num OMP threads: " << RandLAPACK::util::get_omp_threads() << "\n"
             << "Matrix dimensions: " << m << " x " << n << "\n"
             << "Condition number: 10\n"
             << "d_factor: " << d_factor << "\n"
             << "num_runs: " << numruns << "\n";
        file.flush();
        file.close();

        CQRRT_simple_benchmark_data<double> bench_data_well(m, n, d_factor);
        auto start_time = steady_clock::now();
        run_benchmark(m, n, 10.0, numruns, d_factor, bench_data_well, state, output_well, "Well-conditioned");
        auto stop_time = steady_clock::now();
        long total_time = duration_cast<microseconds>(stop_time - start_time).count();

        file.open(output_well, std::ios::out | std::ios::app);
        file << "Total benchmark execution time: " << total_time << " microseconds\n";
        file.flush();
        file.close();
    }

    // Ill-conditioned matrix (κ = 10^6)
    {
        state = RandBLAS::RNGState<r123::Philox4x32>();  // Reset RNG for fair comparison

        std::ofstream file(output_ill, std::ios::out | std::ios::trunc);
        file << "Description: CQRRT_linops simple benchmark - Ill-conditioned dense matrix (κ=10^6)\n"
             << "File format: 16 columns - R-only mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
             << "Q+R mode (6): saso_t, qr_t, cholqr_t, a_mod_trsm_t, t_rest, total_t; "
             << "Errors (4): rel_error, orth_error, is_orthonormal (0/1), max_orth_cols (times in microseconds)\n"
             << "Each row represents one iteration with both R-only and Q+R runs plus verification\n"
             << "Num OMP threads: " << RandLAPACK::util::get_omp_threads() << "\n"
             << "Matrix dimensions: " << m << " x " << n << "\n"
             << "Condition number: 1e6\n"
             << "d_factor: " << d_factor << "\n"
             << "num_runs: " << numruns << "\n";
        file.flush();
        file.close();

        CQRRT_simple_benchmark_data<double> bench_data_ill(m, n, d_factor);
        auto start_time = steady_clock::now();
        run_benchmark(m, n, 1e6, numruns, d_factor, bench_data_ill, state, output_ill, "Ill-conditioned");
        auto stop_time = steady_clock::now();
        long total_time = duration_cast<microseconds>(stop_time - start_time).count();

        file.open(output_ill, std::ios::out | std::ios::app);
        file << "Total benchmark execution time: " << total_time << " microseconds\n";
        file.flush();
        file.close();
    }

    printf("\n=======================================================\n");
    printf("All benchmarks complete!\n");
    printf("Results written to:\n");
    printf("  - %s\n", output_well.c_str());
    printf("  - %s\n", output_ill.c_str());
    printf("=======================================================\n");

    return 0;
}
#endif
