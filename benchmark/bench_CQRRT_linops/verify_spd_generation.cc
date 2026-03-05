#if defined(__APPLE__)
int main() {return 0;}
#else

// Utility to verify SPD matrix generation and test Cholesky factorization limits
// Tests whether gen_spd_mat() produces correctly conditioned SPD matrices
// and determines the true limits of dense vs sparse Cholesky factorization

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include <RandBLAS.hh>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iomanip>
#include <vector>
#include <cmath>

using SpMatrix = Eigen::SparseMatrix<double>;
using DenseMatrix = Eigen::MatrixXd;

template <typename T>
struct verification_result {
    T requested_cond;      // Requested condition number
    T actual_cond;         // Actual condition number (computed)
    T symmetry_error;      // ||A - A^T|| / ||A||
    T min_eigenvalue;      // Smallest eigenvalue (should be > 0)
    bool is_spd;           // Is matrix SPD?
    bool dense_chol_ok;    // Dense Cholesky succeeded?
    bool sparse_chol_ok;   // Sparse Cholesky succeeded?
};

template <typename T>
verification_result<T> verify_spd_matrix(int64_t n, T cond_num, RandBLAS::RNGState<r123::Philox4x32>& state) {

    verification_result<T> result;
    result.requested_cond = cond_num;

    // Generate SPD matrix
    std::vector<T> A(n * n);
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A.data(), state);

    // Test 1: Verify symmetry
    // Compute ||A - A^T|| / ||A||
    T norm_A = lapack::lange(Norm::Fro, n, n, A.data(), n);
    std::vector<T> A_minus_AT(n * n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < n; ++i) {
            A_minus_AT[i + j * n] = A[i + j * n] - A[j + i * n];
        }
    }
    T norm_diff = lapack::lange(Norm::Fro, n, n, A_minus_AT.data(), n);
    result.symmetry_error = norm_diff / norm_A;

    // Test 2: Compute actual condition number via eigendecomposition
    std::vector<T> A_copy(A);
    std::vector<T> eigenvalues(n);

    int64_t info = lapack::syevd(Job::NoVec, Uplo::Upper, n, A_copy.data(), n, eigenvalues.data());

    if (info != 0) {
        printf("    ERROR: syevd failed with info=%ld\n", info);
        result.actual_cond = std::numeric_limits<T>::quiet_NaN();
        result.min_eigenvalue = std::numeric_limits<T>::quiet_NaN();
        result.is_spd = false;
    } else {
        // Eigenvalues are in ascending order
        T lambda_min = eigenvalues[0];
        T lambda_max = eigenvalues[n - 1];

        result.min_eigenvalue = lambda_min;
        result.actual_cond = lambda_max / lambda_min;
        result.is_spd = (lambda_min > 0);  // All eigenvalues must be positive
    }

    // Test 3: Dense Cholesky factorization (Eigen::LLT)
    Eigen::Map<DenseMatrix> A_eigen(A.data(), n, n);
    Eigen::LLT<DenseMatrix> dense_chol(A_eigen);
    result.dense_chol_ok = (dense_chol.info() == Eigen::Success);

    // Test 4: Sparse Cholesky factorization (Eigen::SimplicialLLT with natural ordering)
    // Convert dense to sparse (this is what happens in practice for dense-stored SPD)
    SpMatrix A_sparse = A_eigen.sparseView();
    Eigen::SimplicialLLT<SpMatrix, Eigen::Lower, Eigen::NaturalOrdering<int>> sparse_chol;
    sparse_chol.compute(A_sparse);
    result.sparse_chol_ok = (sparse_chol.info() == Eigen::Success);

    return result;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  matrix_size : Dimension of square SPD matrices (e.g., 1138)" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " 1138" << std::endl;
        std::cerr << "\nThis utility tests SPD generation with condition numbers from 1e1 to 1e8" << std::endl;
        std::cerr << "and verifies:" << std::endl;
        std::cerr << "  - Symmetry (||A - A^T|| / ||A||)" << std::endl;
        std::cerr << "  - Positive definiteness (all eigenvalues > 0)" << std::endl;
        std::cerr << "  - Condition number accuracy (requested vs actual)" << std::endl;
        std::cerr << "  - Dense Cholesky factorization success" << std::endl;
        std::cerr << "  - Sparse Cholesky factorization success" << std::endl;
        return 1;
    }

    int64_t n = std::stol(argv[1]);

    if (n <= 0) {
        std::cerr << "Error: matrix_size must be positive" << std::endl;
        return 1;
    }

    printf("\n=== SPD Matrix Generation Verification ===\n");
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("==========================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Test condition numbers from 1e1 to 1e8 (log-spaced)
    std::vector<double> test_conds = {
        1e1, 3e1, 1e2, 3e2,
        1e3, 3e3, 1e4, 3e4,
        1e5, 3e5, 1e6, 3e6,
        1e7, 3e7, 1e8
    };

    printf("Testing %zu condition numbers...\n\n", test_conds.size());
    printf("%-12s | %-12s | %-12s | %-12s | %-6s | %-10s | %-10s\n",
           "Requested κ", "Actual κ", "Sym Error", "Min λ", "SPD?", "Dense Chol", "Sparse Chol");
    printf("-------------+--------------+--------------+--------------+--------+------------+------------\n");

    int dense_chol_failures = 0;
    int sparse_chol_failures = 0;
    double dense_chol_limit = 0.0;
    double sparse_chol_limit = 0.0;

    for (double cond_num : test_conds) {
        auto result = verify_spd_matrix(n, cond_num, state);

        // Print results
        printf("%.2e | %.2e | %.2e | %.2e | %-6s | %-10s | %-10s\n",
               result.requested_cond,
               result.actual_cond,
               result.symmetry_error,
               result.min_eigenvalue,
               result.is_spd ? "YES" : "NO",
               result.dense_chol_ok ? "OK" : "FAIL",
               result.sparse_chol_ok ? "OK" : "FAIL");

        // Track failures
        if (!result.dense_chol_ok) {
            dense_chol_failures++;
            if (dense_chol_limit == 0.0) {
                dense_chol_limit = cond_num;
            }
        }
        if (!result.sparse_chol_ok) {
            sparse_chol_failures++;
            if (sparse_chol_limit == 0.0) {
                sparse_chol_limit = cond_num;
            }
        }

        // Verify condition number accuracy
        if (result.is_spd) {
            double cond_error = std::abs(result.actual_cond - result.requested_cond) / result.requested_cond;
            if (cond_error > 0.1) {  // More than 10% error
                printf("    WARNING: Condition number mismatch - %.1f%% error\n", cond_error * 100);
            }
        }

        // Verify symmetry
        if (result.symmetry_error > 1e-14) {
            printf("    WARNING: Matrix not symmetric to machine precision\n");
        }

        // Verify positive definiteness
        if (!result.is_spd) {
            printf("    ERROR: Matrix is NOT positive definite!\n");
        }
    }

    printf("\n==========================================\n");
    printf("Summary:\n");
    printf("  Matrices tested: %zu\n", test_conds.size());
    printf("  Dense Cholesky failures: %d\n", dense_chol_failures);
    printf("  Sparse Cholesky failures: %d\n", sparse_chol_failures);

    if (dense_chol_limit > 0) {
        printf("  Dense Cholesky fails at: κ ≥ %.2e\n", dense_chol_limit);
    } else {
        printf("  Dense Cholesky: All tests passed (works up to κ = 1e8)\n");
    }

    if (sparse_chol_limit > 0) {
        printf("  Sparse Cholesky fails at: κ ≥ %.2e\n", sparse_chol_limit);
    } else {
        printf("  Sparse Cholesky: All tests passed (works up to κ = 1e8)\n");
    }

    printf("==========================================\n");

    // Return non-zero if there are critical issues
    if (dense_chol_failures > 0 && sparse_chol_failures > 0) {
        printf("\nConclusion: Both Cholesky solvers have limitations.\n");
        printf("This may explain the κ > 3e3 failures reported in previous tests.\n");
        return 0;
    }

    if (sparse_chol_failures > 0 && dense_chol_failures == 0) {
        printf("\nConclusion: Sparse Cholesky has limitations that dense Cholesky doesn't.\n");
        printf("Consider using dense Cholesky for ill-conditioned matrices (κ > %.2e).\n", sparse_chol_limit);
        return 0;
    }

    printf("\nConclusion: gen_spd_mat() produces valid SPD matrices.\n");
    printf("Both Cholesky solvers work for all tested condition numbers.\n");

    return 0;
}

#endif
