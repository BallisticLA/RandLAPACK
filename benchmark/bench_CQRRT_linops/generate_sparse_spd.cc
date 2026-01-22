#if defined(__APPLE__)
int main() {return 0;}
#else

// Generate sparse SPD matrices with varying condition numbers for CQRRT conditioning study
//
// This tool combines banded SPD matrix generation with diagonal regularization:
// 1. Creates a pentadiagonal SPD matrix with high condition number (seed matrix)
// 2. Applies diagonal regularization (A + alpha*I) to create matrices spanning [min_cond, max_cond]
//
// The approach preserves sparsity by construction and maintains exact condition number control.

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <fstream>
#include <vector>
#include <filesystem>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "../../demos/functions/misc/dm_util.hh"

/// Generate banded sparse SPD matrix with high condition number
/// Uses a pentadiagonal structure for guaranteed sparsity
Eigen::SparseMatrix<double> generate_banded_spd_seed(
    int64_t n,
    int64_t bandwidth,
    RandBLAS::RNGState<r123::Philox4x32>& state
) {
    std::vector<Eigen::Triplet<double>> triplets;

    // Generate random banded structure
    RandBLAS::DenseDist dist(1, 1);

    // Diagonal entries (make positive and large enough for diagonal dominance)
    for (int64_t i = 0; i < n; ++i) {
        double val = 2.0 * (bandwidth + 1);
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

    return A;
}

/// Compute eigenvalue range of sparse SPD matrix
void compute_eigenvalue_range(
    const Eigen::SparseMatrix<double>& A,
    double& lambda_min,
    double& lambda_max
) {
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A_dense);
    auto eigenvalues = solver.eigenvalues();
    lambda_min = eigenvalues.minCoeff();
    lambda_max = eigenvalues.maxCoeff();
}

/// Compute condition number of sparse matrix
double compute_condition_number(const Eigen::SparseMatrix<double>& A) {
    double lambda_min, lambda_max;
    compute_eigenvalue_range(A, lambda_min, lambda_max);
    return lambda_max / lambda_min;
}

/// Add diagonal regularization to SPD matrix: A_new = A + alpha*I
void add_diagonal_regularization(
    Eigen::SparseMatrix<double>& A,
    double alpha
) {
    int64_t n = A.rows();
    for (int64_t i = 0; i < n; ++i) {
        A.coeffRef(i, i) += alpha;
    }
}

/// Write sparse matrix in symmetric Matrix Market format with full precision
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

    // Write lower triangular entries (1-indexed) with full double precision
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

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_dir> <matrix_size> <num_matrices> <min_cond> <max_cond>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  output_dir   : Directory to store generated matrices" << std::endl;
        std::cerr << "  matrix_size  : Dimension of square SPD matrices (e.g., 1138)" << std::endl;
        std::cerr << "  num_matrices : Number of matrices to generate (e.g., 50)" << std::endl;
        std::cerr << "  min_cond     : Minimum condition number (e.g., 10)" << std::endl;
        std::cerr << "  max_cond     : Maximum condition number (e.g., 1e16)" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " ./sparse_spd 1138 50 10 1e16" << std::endl;
        std::cerr << "\nThis generates sparse SPD matrices with condition numbers log-spaced" << std::endl;
        std::cerr << "from min_cond to max_cond using a pentadiagonal structure." << std::endl;
        std::cerr << "\nThe tool creates a high-kappa seed matrix internally and uses" << std::endl;
        std::cerr << "diagonal regularization (A + alpha*I) to achieve target condition numbers." << std::endl;
        return 1;
    }

    // Parse arguments
    std::string output_dir = argv[1];
    int64_t n = std::stol(argv[2]);
    int64_t num_matrices = std::stol(argv[3]);
    double min_cond = std::stod(argv[4]);
    double max_cond = std::stod(argv[5]);

    // Validate arguments
    if (n <= 0 || num_matrices <= 0) {
        std::cerr << "Error: matrix_size and num_matrices must be positive" << std::endl;
        return 1;
    }

    if (min_cond <= 1.0 || max_cond <= min_cond) {
        std::cerr << "Error: Need 1 < min_cond < max_cond" << std::endl;
        return 1;
    }

    // Create output directory
    std::filesystem::create_directories(output_dir);

    printf("\n=== Sparse SPD Matrix Generation for CQRRT Conditioning Study ===\n");
    printf("Output directory: %s\n", output_dir.c_str());
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Number of matrices: %ld\n", num_matrices);
    printf("Condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("Structure: Pentadiagonal (bandwidth = 2)\n");
    printf("================================================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);

    // Step 1: Generate seed banded SPD matrix (without condition number control)
    printf("Step 1: Generating banded SPD seed matrix...\n");
    int64_t bandwidth = 2;  // Pentadiagonal: diagonal + 2 bands above/below
    Eigen::SparseMatrix<double> A_seed = generate_banded_spd_seed(n, bandwidth, state);

    // Compute seed matrix properties
    double lambda_min_seed, lambda_max_seed;
    compute_eigenvalue_range(A_seed, lambda_min_seed, lambda_max_seed);
    double cond_seed = lambda_max_seed / lambda_min_seed;

    int64_t nnz = A_seed.nonZeros();
    double density = (double)nnz / (n * n);

    printf("Seed matrix properties:\n");
    printf("  Non-zeros: %ld (%.4f%% density)\n", nnz, density * 100.0);
    printf("  Eigenvalue range: [%.6e, %.6e]\n", lambda_min_seed, lambda_max_seed);
    printf("  Condition number: %.6e\n\n", cond_seed);

    // Step 2: Generate log-spaced condition numbers and create matrices
    printf("Step 2: Generating %ld matrices with diagonal regularization...\n\n", num_matrices);

    double log_min = std::log10(min_cond);
    double log_max = std::log10(max_cond);

    // Create metadata file
    std::string metadata_file = output_dir + "/metadata.txt";
    std::ofstream meta(metadata_file);
    meta << "# Sparse SPD Matrix Set Metadata\n";
    meta << "# Generated for CQRRT_linops conditioning study\n";
    meta << "# Method: Banded structure + diagonal regularization\n";
    meta << "matrix_size: " << n << "\n";
    meta << "num_matrices: " << num_matrices << "\n";
    meta << "min_condition_number: " << min_cond << "\n";
    meta << "max_condition_number: " << max_cond << "\n";
    meta << "structure: pentadiagonal (bandwidth=2)\n";
    meta << "nnz_per_matrix: " << nnz << "\n";
    meta << "seed_lambda_min: " << std::scientific << lambda_min_seed << "\n";
    meta << "seed_lambda_max: " << std::scientific << lambda_max_seed << "\n";
    meta << "seed_condition_number: " << std::scientific << cond_seed << "\n";
    meta << "spacing: logarithmic\n";
    meta << "# Format: index, condition_number, filename\n";
    meta << "# ----------------------------------------\n";

    int64_t matrices_generated = 0;

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
        double alpha = (lambda_max_seed - target_cond * lambda_min_seed) / (target_cond - 1.0);

        // Skip if alpha is too negative (would make matrix non-SPD)
        // We need lambda_min + alpha > 0
        if (alpha <= -lambda_min_seed) {
            printf("Skipping matrix %ld/%ld: target kappa=%.6e requires alpha=%.6e (violates SPD)\n",
                   i + 1, num_matrices, target_cond, alpha);
            continue;
        }

        // Create regularized matrix
        Eigen::SparseMatrix<double> A_reg = A_seed;
        add_diagonal_regularization(A_reg, alpha);

        // Verify actual condition number
        double actual_cond = compute_condition_number(A_reg);

        // Generate filename
        char filename[256];
        snprintf(filename, sizeof(filename), "sparse_spd_%04ld_cond_%.2e.mtx", matrices_generated, actual_cond);
        std::string filepath = output_dir + "/" + std::string(filename);

        // Save matrix
        write_sparse_matrix_symmetric(filepath, A_reg);

        printf("Generated matrix %ld/%ld: target kappa=%.2e, actual kappa=%.2e, alpha=%.2e -> %s\n",
               i + 1, num_matrices, target_cond, actual_cond, alpha, filename);

        // Write to metadata (format: index, condition_number, filename)
        meta << matrices_generated << ", " << std::scientific << actual_cond << ", " << filename << "\n";

        matrices_generated++;
    }

    // Update metadata with actual count
    meta.close();

    // Rewrite metadata with correct num_matrices
    std::ofstream meta_final(metadata_file);
    meta_final << "# Sparse SPD Matrix Set Metadata\n";
    meta_final << "# Generated for CQRRT_linops conditioning study\n";
    meta_final << "# Method: Banded structure + diagonal regularization\n";
    meta_final << "matrix_size: " << n << "\n";
    meta_final << "num_matrices: " << matrices_generated << "\n";
    meta_final << "min_condition_number: " << min_cond << "\n";
    meta_final << "max_condition_number: " << max_cond << "\n";
    meta_final << "structure: pentadiagonal (bandwidth=2)\n";
    meta_final << "nnz_per_matrix: " << nnz << "\n";
    meta_final << "seed_lambda_min: " << std::scientific << lambda_min_seed << "\n";
    meta_final << "seed_lambda_max: " << std::scientific << lambda_max_seed << "\n";
    meta_final << "seed_condition_number: " << std::scientific << cond_seed << "\n";
    meta_final << "spacing: logarithmic\n";
    meta_final << "# Format: index, condition_number, filename\n";
    meta_final << "# ----------------------------------------\n";

    // Re-read entries from original meta and copy
    std::ifstream meta_read(output_dir + "/metadata.txt.tmp");
    // Just regenerate the entries
    int64_t entry_idx = 0;
    for (int64_t i = 0; i < num_matrices && entry_idx < matrices_generated; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double log_cond = log_min + t * (log_max - log_min);
        double target_cond = std::pow(10.0, log_cond);

        double alpha = (lambda_max_seed - target_cond * lambda_min_seed) / (target_cond - 1.0);

        if (alpha <= -lambda_min_seed) {
            continue;  // Skip same entries as before
        }

        // Recompute actual condition number
        Eigen::SparseMatrix<double> A_reg = A_seed;
        add_diagonal_regularization(A_reg, alpha);
        double actual_cond = compute_condition_number(A_reg);

        char filename[256];
        snprintf(filename, sizeof(filename), "sparse_spd_%04ld_cond_%.2e.mtx", entry_idx, actual_cond);

        meta_final << entry_idx << ", " << std::scientific << actual_cond << ", " << filename << "\n";
        entry_idx++;
    }
    meta_final.close();

    printf("\n================================================================\n");
    printf("Generation complete!\n");
    printf("Generated %ld/%ld matrices (some targets may be skipped if they violate SPD)\n",
           matrices_generated, num_matrices);
    printf("Metadata saved to: %s\n", metadata_file.c_str());
    printf("================================================================\n");

    return 0;
}
#endif
