#if defined(__APPLE__)
int main() {return 0;}
#else

// Generate a sparse SPD matrix using banded structure with prescribed condition number
// Strategy: Create a banded symmetric matrix, then scale to achieve target condition number
//
// Unlike the thresholding approach in generate_sparse_spd.cc, this maintains sparsity
// by construction and preserves the eigenvalue structure.

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <fstream>
#include <vector>
#include <filesystem>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "../../demos/functions/misc/dm_util.hh"

/// Generate banded sparse SPD matrix with prescribed condition number
/// Uses a tridiagonal or pentadiagonal structure for guaranteed sparsity
Eigen::SparseMatrix<double> generate_banded_spd(
    int64_t n,
    double target_cond,
    int64_t bandwidth,
    RandBLAS::RNGState<r123::Philox4x32>& state
) {
    // Generate a banded symmetric positive definite matrix
    // Strategy: Create B = D + alpha*I where D is banded random symmetric

    std::vector<Eigen::Triplet<double>> triplets;

    // Generate random banded structure
    RandBLAS::DenseDist dist(1, 1);

    // Diagonal entries (make positive and large enough)
    for (int64_t i = 0; i < n; ++i) {
        double val = 2.0 * (bandwidth + 1);  // Ensure diagonal dominance
        triplets.emplace_back(i, i, val);
    }

    // Off-diagonal bands (symmetric)
    for (int64_t band = 1; band <= bandwidth; ++band) {
        for (int64_t i = 0; i < n - band; ++i) {
            double val;
            RandBLAS::fill_dense(dist, &val, state);
            val = std::abs(val) * 0.5;  // Small positive off-diagonals

            triplets.emplace_back(i, i + band, val);
            triplets.emplace_back(i + band, i, val);  // Symmetric
        }
    }

    Eigen::SparseMatrix<double> A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // Compute actual eigenvalues to determine scaling
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A_dense);
    auto eigenvalues = solver.eigenvalues();

    double lambda_min_orig = eigenvalues.minCoeff();
    double lambda_max_orig = eigenvalues.maxCoeff();
    double cond_orig = lambda_max_orig / lambda_min_orig;

    // Scale matrix to achieve target condition number using diagonal regularization
    // We want: (lambda_max + alpha) / (lambda_min + alpha) = target_cond
    // Solving: alpha = (lambda_max - target_cond * lambda_min) / (target_cond - 1)

    double alpha = (lambda_max_orig - target_cond * lambda_min_orig) / (target_cond - 1.0);

    // Add alpha*I to diagonal
    for (int64_t i = 0; i < n; ++i) {
        A.coeffRef(i, i) += alpha;
    }

    return A;
}

/// Compute condition number of sparse matrix
double compute_condition_number(const Eigen::SparseMatrix<double>& A) {
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A_dense);

    auto eigenvalues = solver.eigenvalues();
    double lambda_max = eigenvalues.maxCoeff();
    double lambda_min = eigenvalues.minCoeff();

    return lambda_max / lambda_min;
}

/// Write sparse matrix in symmetric Matrix Market format
void write_sparse_matrix_symmetric(
    const std::string& filename,
    const Eigen::SparseMatrix<double>& A
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "%%MatrixMarket matrix coordinate real symmetric\n";

    int64_t n = A.rows();

    // Count only lower triangular entries (including diagonal)
    int64_t nnz_lower = 0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col()) {
                ++nnz_lower;
            }
        }
    }

    file << n << " " << n << " " << nnz_lower << "\n";

    // Write lower triangular entries (1-indexed) with high precision
    file << std::scientific << std::setprecision(17);
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col()) {
                file << (it.row() + 1) << " " << (it.col() + 1) << " " << it.value() << "\n";
            }
        }
    }

    file.close();
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_file.mtx> <matrix_size> <condition_number>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  output_file.mtx  : Output Matrix Market file" << std::endl;
        std::cerr << "  matrix_size      : Dimension of square matrix (e.g., 1138)" << std::endl;
        std::cerr << "  condition_number : Target condition number (e.g., 1e16)" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " sparse_1138_cond_1e16.mtx 1138 1e16" << std::endl;
        std::cerr << "\nThis generates a banded sparse SPD matrix with prescribed condition number." << std::endl;
        std::cerr << "The matrix will have a pentadiagonal structure (bandwidth=2) for guaranteed sparsity." << std::endl;
        return 1;
    }

    // Parse arguments
    std::string output_file = argv[1];
    int64_t n = std::stol(argv[2]);
    double target_cond = std::stod(argv[3]);

    // Validate arguments
    if (n <= 0) {
        std::cerr << "Error: matrix_size must be positive" << std::endl;
        return 1;
    }

    if (target_cond <= 1.0) {
        std::cerr << "Error: condition_number must be > 1" << std::endl;
        return 1;
    }

    printf("\n=== Banded Sparse SPD Matrix Generation ===\n");
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Target condition number: %.6e\n", target_cond);
    printf("Structure: Pentadiagonal (bandwidth = 2)\n");
    printf("===========================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);

    // Generate banded SPD matrix
    printf("Generating banded SPD matrix...\n");
    int64_t bandwidth = 2;  // Pentadiagonal: diagonal + 2 bands above/below
    Eigen::SparseMatrix<double> A = generate_banded_spd(n, target_cond, bandwidth, state);

    printf("Matrix generated. Computing properties...\n");

    // Verify properties
    int64_t nnz = A.nonZeros();
    double density = (double)nnz / (n * n);
    double actual_cond = compute_condition_number(A);

    printf("\nMatrix Properties:\n");
    printf("  Dimension: %ld x %ld\n", n, n);
    printf("  Non-zeros: %ld\n", nnz);
    printf("  Density: %.4f%%\n", density * 100.0);
    printf("  Target condition number: %.6e\n", target_cond);
    printf("  Actual condition number: %.6e\n", actual_cond);
    printf("  Relative error: %.2f%%\n", 100.0 * std::abs(actual_cond - target_cond) / target_cond);

    // Check if we're within reasonable tolerance
    double rel_error = std::abs(actual_cond - target_cond) / target_cond;
    if (rel_error > 0.1) {  // More than 10% error
        printf("\n  WARNING: Condition number differs significantly from target!\n");
        printf("  This may happen for very high condition numbers near machine precision limits.\n");
    }

    // Save to file
    printf("\nSaving to: %s\n", output_file.c_str());
    write_sparse_matrix_symmetric(output_file, A);

    printf("\n===========================================\n");
    printf("Generation complete!\n");
    printf("===========================================\n");

    return 0;
}
#endif
