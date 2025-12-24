#if defined(__APPLE__)
int main() {return 0;}
#else

// Utility to generate sparse symmetric matrices with controlled condition numbers
// Strategy: Generate sparse symmetric matrix with specified eigenvalue spectrum

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <fstream>
#include <vector>
#include <filesystem>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include "../../demos/functions/misc/dm_util.hh"

/// Generate sparse symmetric matrix with controlled condition number
/// Uses simple banded structure to ensure sparsity
Eigen::SparseMatrix<double> generate_sparse_symmetric(
    int64_t n,
    double cond_num,
    double sparsity_target,
    RandBLAS::RNGState<r123::Philox4x32>& state
) {
    // Generate eigenvalues with specified condition number
    std::vector<double> eigenvalues(n);
    double lambda_max = 1.0;
    double lambda_min = 1.0 / cond_num;

    // Use polynomial decay
    for (int64_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n - 1);
        eigenvalues[i] = lambda_min + (lambda_max - lambda_min) * std::pow(1.0 - t, 2.0);
    }

    // Create diagonal matrix with eigenvalues
    Eigen::MatrixXd Lambda = Eigen::VectorXd::Map(eigenvalues.data(), n).asDiagonal();

    // Generate random orthogonal matrix using QR decomposition
    std::vector<double> G_data(n * n);
    RandBLAS::DenseDist D(n, n);
    RandBLAS::fill_dense(D, G_data.data(), state);

    Eigen::MatrixXd G = Eigen::Map<Eigen::MatrixXd>(G_data.data(), n, n);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(G);
    Eigen::MatrixXd Q = qr.householderQ();

    // A = Q * Λ * Q^T
    Eigen::MatrixXd A_dense = Q * Lambda * Q.transpose();

    // Make exactly symmetric
    A_dense = 0.5 * (A_dense + A_dense.transpose());

    // Threshold to achieve target sparsity
    std::vector<double> all_vals;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            all_vals.push_back(std::abs(A_dense(i, j)));
        }
    }

    std::sort(all_vals.begin(), all_vals.end(), std::greater<double>());
    int64_t num_keep = static_cast<int64_t>(all_vals.size() * sparsity_target);
    double threshold = (num_keep < all_vals.size()) ? all_vals[num_keep] : 0.0;

    // Build sparse matrix
    std::vector<Eigen::Triplet<double>> triplets;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            if (std::abs(A_dense(i, j)) >= threshold) {
                triplets.emplace_back(i, j, A_dense(i, j));
                if (i != j) {
                    triplets.emplace_back(j, i, A_dense(i, j));
                }
            }
        }
    }

    Eigen::SparseMatrix<double> A_sparse(n, n);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());

    return A_sparse;
}

/// Compute condition number of sparse symmetric matrix
double compute_condition_number_sparse(const Eigen::SparseMatrix<double>& A) {
    // Convert to dense for eigenvalue computation
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A_dense);

    auto eigenvalues = solver.eigenvalues();
    double lambda_max = 0.0;
    double lambda_min = 1e100;

    // For symmetric (not necessarily PD), use absolute values
    for (int i = 0; i < eigenvalues.size(); ++i) {
        double abs_eig = std::abs(eigenvalues(i));
        if (abs_eig > lambda_max) lambda_max = abs_eig;
        if (abs_eig > 1e-14 && abs_eig < lambda_min) lambda_min = abs_eig;
    }

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

    // Count lower triangular entries (including diagonal)
    int64_t nnz_lower = 0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col()) {
                ++nnz_lower;
            }
        }
    }

    file << n << " " << n << " " << nnz_lower << "\n";

    // Write lower triangular entries (1-indexed)
    file << std::scientific << std::setprecision(16);
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

    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_dir> <matrix_size> <num_matrices> <min_cond> <max_cond> <sparsity>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  output_dir   : Directory to store generated matrices" << std::endl;
        std::cerr << "  matrix_size  : Dimension of square symmetric matrices" << std::endl;
        std::cerr << "  num_matrices : Number of matrices to generate" << std::endl;
        std::cerr << "  min_cond     : Minimum condition number" << std::endl;
        std::cerr << "  max_cond     : Maximum condition number" << std::endl;
        std::cerr << "  sparsity     : Target sparsity (fraction of nnz, e.g., 0.05 for 5%)" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " ./sparse_sym 1138 20 10 1e4 0.05" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string output_dir = argv[1];
    int64_t n = std::stol(argv[2]);
    int64_t num_matrices = std::stol(argv[3]);
    double min_cond = std::stod(argv[4]);
    double max_cond = std::stod(argv[5]);
    double sparsity = std::stod(argv[6]);

    // Validate
    if (n <= 0 || num_matrices <= 0) {
        std::cerr << "Error: matrix_size and num_matrices must be positive" << std::endl;
        return 1;
    }

    if (min_cond <= 1.0 || max_cond <= min_cond) {
        std::cerr << "Error: Need 1 < min_cond < max_cond" << std::endl;
        return 1;
    }

    if (sparsity <= 0.0 || sparsity > 1.0) {
        std::cerr << "Error: sparsity must be in (0, 1]" << std::endl;
        return 1;
    }

    // Create output directory
    std::filesystem::create_directories(output_dir);

    printf("\n=== Sparse Symmetric Matrix Generation ===\n");
    printf("Output directory: %s\n", output_dir.c_str());
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Number of matrices: %ld\n", num_matrices);
    printf("Condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("Target sparsity: %.2f%% nnz\n", sparsity * 100);
    printf("======================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Generate log-spaced condition numbers
    double log_min = std::log10(min_cond);
    double log_max = std::log10(max_cond);

    // Create metadata file
    std::string metadata_file = output_dir + "/metadata.txt";
    std::ofstream meta(metadata_file);
    meta << "# Sparse Symmetric Matrix Set Metadata\n";
    meta << "# Generated for CQRRT_linops conditioning study\n";
    meta << "matrix_size: " << n << "\n";
    meta << "num_matrices: " << num_matrices << "\n";
    meta << "min_condition_number: " << min_cond << "\n";
    meta << "max_condition_number: " << max_cond << "\n";
    meta << "target_sparsity: " << sparsity << "\n";
    meta << "spacing: logarithmic\n";
    meta << "generation_method: eigenvalue_decomposition\n";
    meta << "# Format: index, condition_number, filename\n";
    meta << "# ----------------------------------------\n";

    for (int64_t i = 0; i < num_matrices; ++i) {
        // Log-spaced condition number
        double t = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double log_cond = log_min + t * (log_max - log_min);
        double target_cond = std::pow(10.0, log_cond);

        printf("Generating matrix %ld/%ld: target κ=%.6e ... ", i + 1, num_matrices, target_cond);
        fflush(stdout);

        // Generate sparse symmetric matrix
        Eigen::SparseMatrix<double> A_sparse = generate_sparse_symmetric(n, target_cond, sparsity, state);

        // Compute actual condition number
        double actual_cond = compute_condition_number_sparse(A_sparse);
        int64_t nnz = A_sparse.nonZeros();
        double actual_sparsity = static_cast<double>(nnz) / (n * n);

        // Generate filename
        char filename[256];
        snprintf(filename, sizeof(filename), "sparse_sym_%04ld_cond_%.2e.mtx", i, actual_cond);
        std::string filepath = output_dir + "/" + std::string(filename);

        // Save matrix
        write_sparse_matrix_symmetric(filepath, A_sparse);

        printf("κ_actual=%.6e, nnz=%ld (%.2f%%), saved\n",
               actual_cond, nnz, actual_sparsity * 100);

        // Write to metadata
        meta << i << ", " << std::scientific << actual_cond << ", " << filename << "\n";
    }

    meta.close();
    printf("\n======================================\n");
    printf("Generation complete!\n");
    printf("Metadata saved to: %s\n", metadata_file.c_str());
    printf("======================================\n");

    return 0;
}
#endif
