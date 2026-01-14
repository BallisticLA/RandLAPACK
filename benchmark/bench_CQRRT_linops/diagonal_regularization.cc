#if defined(__APPLE__)
int main() {return 0;}
#else

// Utility to create perturbed versions of 1138_bus.mtx with varying condition numbers
// Uses diagonal regularization: A_new = A_original + alpha*I
// This preserves sparsity while controlling the condition number

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <fstream>
#include <vector>
#include <filesystem>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include "../../demos/functions/misc/dm_util.hh"

/// Compute condition number of sparse SPD matrix using Eigen
template <typename T>
T compute_condition_number(const Eigen::SparseMatrix<T>& A) {
    // For SPD matrices, we can use Lanczos method for extremal eigenvalues
    // For simplicity, convert to dense (only for computing condition number)
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A_dense);

    auto eigenvalues = solver.eigenvalues();
    T lambda_max = eigenvalues.maxCoeff();
    T lambda_min = eigenvalues.minCoeff();

    return lambda_max / lambda_min;
}

/// Add diagonal regularization to SPD matrix: A_new = A + alpha*I
template <typename T>
void add_diagonal_regularization(
    Eigen::SparseMatrix<T>& A,
    T alpha
) {
    int64_t n = A.rows();

    // Add alpha to diagonal entries
    for (int64_t i = 0; i < n; ++i) {
        A.coeffRef(i, i) += alpha;
    }
}

/// Write Eigen sparse matrix to Matrix Market file
template <typename T>
void write_sparse_matrix_to_mtx(
    const std::string& filename,
    const Eigen::SparseMatrix<T>& A
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    file << "%%MatrixMarket matrix coordinate real symmetric\n";

    int64_t n = A.rows();
    int64_t nnz = A.nonZeros();

    // Count only lower triangular entries for symmetric storage
    int64_t nnz_lower = 0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col()) {  // Lower triangular including diagonal
                ++nnz_lower;
            }
        }
    }

    file << n << " " << n << " " << nnz_lower << "\n";

    // Write entries (Matrix Market uses 1-based indexing, symmetric format)
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
            if (it.row() >= it.col()) {  // Only store lower triangular
                file << (it.row() + 1) << " " << (it.col() + 1) << " "
                     << std::scientific << std::setprecision(16) << it.value() << "\n";
            }
        }
    }

    file.close();
}

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_mtx> <output_dir> <num_matrices> <min_cond> <max_cond>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  input_mtx    : Path to input 1138_bus.mtx file" << std::endl;
        std::cerr << "  output_dir   : Directory to store perturbed matrices" << std::endl;
        std::cerr << "  num_matrices : Number of perturbed versions to generate" << std::endl;
        std::cerr << "  min_cond     : Minimum target condition number" << std::endl;
        std::cerr << "  max_cond     : Maximum target condition number" << std::endl;
        std::cerr << "\nExample: " << argv[0]
                  << " /home/mymel/data/CQRRT_linop_test_matrices/left_op/1138_bus.mtx"
                  << " ./perturbed_1138 100 10 1e6" << std::endl;
        std::cerr << "\nThis creates perturbed versions using diagonal regularization:" << std::endl;
        std::cerr << "  A_new = A_original + alpha*I" << std::endl;
        std::cerr << "Alpha values are chosen to achieve target condition numbers (log-spaced)" << std::endl;
        return 1;
    }

    // Parse arguments
    std::string input_file = argv[1];
    std::string output_dir = argv[2];
    int64_t num_matrices = std::stol(argv[3]);
    double min_cond = std::stod(argv[4]);
    double max_cond = std::stod(argv[5]);

    // Validate arguments
    if (num_matrices <= 0) {
        std::cerr << "Error: num_matrices must be positive" << std::endl;
        return 1;
    }

    if (min_cond <= 1.0 || max_cond <= min_cond) {
        std::cerr << "Error: Need 1 < min_cond < max_cond" << std::endl;
        return 1;
    }

    // Create output directory
    std::filesystem::create_directories(output_dir);

    printf("\n=== 1138_bus Perturbation for CQRRT Conditioning Study ===\n");
    printf("Input file: %s\n", input_file.c_str());
    printf("Output directory: %s\n", output_dir.c_str());
    printf("Number of perturbed matrices: %ld\n", num_matrices);
    printf("Target condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("==========================================================\n\n");

    // Load original 1138_bus matrix
    printf("Loading original matrix...\n");
    Eigen::SparseMatrix<double> A_original;
    RandLAPACK_demos::eigen_sparse_from_matrix_market(input_file, A_original);

    int64_t n = A_original.rows();
    int64_t nnz_original = A_original.nonZeros();

    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Original nnz: %ld\n", nnz_original);

    // Compute original condition number
    printf("Computing original condition number (this may take a moment)...\n");
    double cond_original = compute_condition_number(A_original);
    printf("Original condition number: %.6e\n\n", cond_original);

    // Compute original eigenvalue range
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A_original);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A_dense);
    auto eigenvalues = solver.eigenvalues();
    double lambda_max = eigenvalues.maxCoeff();
    double lambda_min = eigenvalues.minCoeff();

    printf("Original eigenvalue range: [%.6e, %.6e]\n", lambda_min, lambda_max);
    printf("==========================================================\n\n");

    // Generate log-spaced condition numbers
    double log_min = std::log10(min_cond);
    double log_max = std::log10(max_cond);

    // Create metadata file
    std::string metadata_file = output_dir + "/metadata.txt";
    std::ofstream meta(metadata_file);
    meta << "# Perturbed 1138_bus Matrix Set Metadata\n";
    meta << "# Generated for CQRRT_linops conditioning study\n";
    meta << "original_file: " << input_file << "\n";
    meta << "matrix_size: " << n << "\n";
    meta << "original_nnz: " << nnz_original << "\n";
    meta << "original_condition_number: " << std::scientific << cond_original << "\n";
    meta << "original_lambda_min: " << lambda_min << "\n";
    meta << "original_lambda_max: " << lambda_max << "\n";
    meta << "num_matrices: " << num_matrices << "\n";
    meta << "min_target_condition: " << min_cond << "\n";
    meta << "max_target_condition: " << max_cond << "\n";
    meta << "perturbation_method: diagonal_regularization (A + alpha*I)\n";
    meta << "spacing: logarithmic\n";
    meta << "# Format: index, condition_number, filename\n";
    meta << "# ----------------------------------------\n";

    for (int64_t i = 0; i < num_matrices; ++i) {
        // Log-spaced target condition number
        double t = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double log_cond = log_min + t * (log_max - log_min);
        double target_cond = std::pow(10.0, log_cond);

        // Compute alpha to achieve target condition number
        // For A_new = A + alpha*I:
        // lambda_min_new = lambda_min + alpha
        // lambda_max_new = lambda_max + alpha
        // cond_new = (lambda_max + alpha) / (lambda_min + alpha)
        // Solving: alpha = (lambda_max - target_cond * lambda_min) / (target_cond - 1)

        double alpha = (lambda_max - target_cond * lambda_min) / (target_cond - 1.0);

        // Skip if alpha is negative (target condition is worse than original)
        if (alpha < 0) {
            printf("Skipping matrix %ld/%ld: target κ=%.6e requires negative alpha (worse than original)\n",
                   i + 1, num_matrices, target_cond);
            continue;
        }

        // Create perturbed matrix
        printf("Generating matrix %ld/%ld: target κ=%.6e, alpha=%.6e ... ",
               i + 1, num_matrices, target_cond, alpha);
        fflush(stdout);

        Eigen::SparseMatrix<double> A_perturbed = A_original;
        add_diagonal_regularization(A_perturbed, alpha);

        // Verify actual condition number
        double actual_cond = compute_condition_number(A_perturbed);

        // Generate filename
        char filename[256];
        snprintf(filename, sizeof(filename), "1138_bus_perturbed_%04ld_cond_%.2e.mtx", i, actual_cond);
        std::string filepath = output_dir + "/" + std::string(filename);

        // Save matrix
        write_sparse_matrix_to_mtx(filepath, A_perturbed);

        int64_t nnz_perturbed = A_perturbed.nonZeros();

        printf("κ_actual=%.6e, nnz=%ld, saved to %s\n", actual_cond, nnz_perturbed, filename);

        // Write to metadata (format: index, condition_number, filename)
        meta << i << ", " << std::scientific << actual_cond << ", " << filename << "\n";
    }

    meta.close();
    printf("\n==========================================================\n");
    printf("Perturbation complete!\n");
    printf("Metadata saved to: %s\n", metadata_file.c_str());
    printf("==========================================================\n");

    return 0;
}
#endif
