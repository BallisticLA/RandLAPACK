// Shared utilities for RandLAPACK extras tests
// Common helper functions used across multiple extras test files

#ifndef RANDLAPACK_EXTRAS_TEST_UTILS_HH
#define RANDLAPACK_EXTRAS_TEST_UTILS_HH

#include <RandBLAS.hh>
#include <Eigen/Sparse>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace RandLAPACK_extras {
namespace test_utils {

/// Generate a sparse SPD matrix with controlled condition number
/// Uses banded structure with diagonal dominance to ensure SPD property
///
/// This function creates a sparse symmetric positive definite matrix by:
/// 1. Generating eigenvalues with polynomial decay based on condition number
/// 2. Creating a banded matrix structure (bandwidth = 5) for sparsity
/// 3. Setting diagonal entries to ensure diagonal dominance
/// 4. Writing the result in Matrix Market symmetric format
///
/// @param n Matrix dimension (n × n)
/// @param cond_num Target condition number (ratio of max to min eigenvalues)
/// @param filename Output file path (Matrix Market format)
/// @return Absolute path to the generated file
/// @throws std::runtime_error if file cannot be opened or path resolution fails
///
/// Example usage:
/// @code
/// auto filepath = generate_sparse_spd_matrix(100, 10.0, "/tmp/matrix.mtx");
/// // Use filepath with CholSolverLinOp or other solvers
/// @endcode
inline std::string generate_sparse_spd_matrix(int64_t n, double cond_num, const std::string& filename) {
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Generate eigenvalues with polynomial decay to control condition number
    double lambda_max = 1.0;
    double lambda_min = 1.0 / cond_num;

    std::vector<double> eigenvalues(n);
    for (int64_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n - 1);
        eigenvalues[i] = lambda_min + (lambda_max - lambda_min) * std::pow(1.0 - t, 2.0);
    }

    // Create sparse SPD matrix using banded structure
    // Banded matrices are naturally sparse and easy to make SPD
    std::vector<Eigen::Triplet<double>> triplets;

    // Set diagonal to ensure positive definiteness and diagonal dominance
    double diag_sum = 0.0;
    for (auto ev : eigenvalues) diag_sum += ev;
    double diag_value = diag_sum / n + 1.0;  // +1 ensures strict diagonal dominance

    int64_t bandwidth = 5;  // Keep it sparse with limited bandwidth
    for (int64_t i = 0; i < n; ++i) {
        // Diagonal entry (largest in each row/column)
        triplets.emplace_back(i, i, diag_value);

        // Off-diagonal bands (symmetric) - decay with distance from diagonal
        for (int64_t b = 1; b <= bandwidth && i + b < n; ++b) {
            double off_diag = 0.1 * diag_value / b;  // Decay with distance
            triplets.emplace_back(i, i + b, off_diag);
            triplets.emplace_back(i + b, i, off_diag);
        }
    }

    // Construct sparse matrix from triplets
    Eigen::SparseMatrix<double> A_sparse(n, n);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());

    // Write to Matrix Market file in symmetric format
    // This format stores only the lower triangle to save space
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "%%MatrixMarket matrix coordinate real symmetric\n";

    // Count lower triangular entries (including diagonal)
    int64_t nnz_lower = 0;
    for (int k = 0; k < A_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, k); it; ++it) {
            if (it.row() >= it.col()) {
                ++nnz_lower;
            }
        }
    }

    // Write dimensions and number of nonzeros
    file << n << " " << n << " " << nnz_lower << "\n";
    file << std::scientific << std::setprecision(16);

    // Write lower triangular entries (1-indexed, as per Matrix Market format)
    for (int k = 0; k < A_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_sparse, k); it; ++it) {
            if (it.row() >= it.col()) {
                file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
            }
        }
    }

    file.close();

    // Return absolute path for portability
    char abs_path[PATH_MAX];
    if (realpath(filename.c_str(), abs_path) == nullptr) {
        throw std::runtime_error("Failed to get absolute path for: " + filename);
    }
    return std::string(abs_path);
}

} // namespace test_utils
} // namespace RandLAPACK_extras

#endif // RANDLAPACK_DEMOS_TEST_UTILS_HH
