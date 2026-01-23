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
//
// NOTE: Uses BLAS++/LAPACK++ and RandBLAS sparse types (no Eigen dependency).

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <lapack.hh>
#include <fstream>
#include <vector>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>
#include <iomanip>
#include "../../demos/functions/misc/dm_util.hh"

/// Sparse matrix storage using separate arrays (COO-like structure for generation)
/// Stored in lower-triangular symmetric format for file output
class SparseSPDMatrix {
public:
    int64_t dim;
    int64_t num_entries;  // Only lower triangular entries (including diagonal)
    std::vector<int64_t> row_indices;
    std::vector<int64_t> col_indices;
    std::vector<double> values;

    SparseSPDMatrix(int64_t n) : dim(n), num_entries(0) {}

    void reserve_space(int64_t count) {
        row_indices.reserve(count);
        col_indices.reserve(count);
        values.reserve(count);
    }

    void add_entry(int64_t i, int64_t j, double val) {
        // Store only lower triangular (i >= j)
        if (i >= j) {
            row_indices.push_back(i);
            col_indices.push_back(j);
            values.push_back(val);
            num_entries++;
        }
    }

    // Create a copy with modified diagonal
    SparseSPDMatrix with_diagonal_shift(double alpha) const {
        SparseSPDMatrix result(dim);
        result.row_indices = row_indices;
        result.col_indices = col_indices;
        result.values = values;
        result.num_entries = num_entries;

        // Add alpha to diagonal entries
        for (int64_t k = 0; k < result.num_entries; ++k) {
            if (result.row_indices[k] == result.col_indices[k]) {
                result.values[k] += alpha;
            }
        }
        return result;
    }
};

/// Generate banded sparse SPD matrix with high condition number
/// Uses a pentadiagonal structure for guaranteed sparsity
SparseSPDMatrix generate_banded_spd_seed(
    int64_t n,
    int64_t bandwidth,
    RandBLAS::RNGState<r123::Philox4x32>& state
) {
    SparseSPDMatrix A(n);

    // Estimate num_entries for lower triangular: n diagonal + (bandwidth * n) off-diagonal
    A.reserve_space(n + bandwidth * n);

    // Generate random values for off-diagonals
    RandBLAS::DenseDist dist(1, 1);

    // Diagonal entries (make positive and large enough for diagonal dominance)
    for (int64_t i = 0; i < n; ++i) {
        double val = 2.0 * (bandwidth + 1);
        A.add_entry(i, i, val);
    }

    // Off-diagonal bands (only lower triangular: i > j)
    for (int64_t band = 1; band <= bandwidth; ++band) {
        for (int64_t i = band; i < n; ++i) {
            double val;
            RandBLAS::fill_dense(dist, &val, state);
            val = std::abs(val) * 0.5;  // Small positive off-diagonals
            A.add_entry(i, i - band, val);  // Lower triangular entry
        }
    }

    return A;
}

/// Convert sparse matrix to dense (column-major) for eigensolve
/// The sparse matrix is stored as lower triangular, but we need full symmetric
void sparse_to_dense_symmetric(const SparseSPDMatrix& A, std::vector<double>& dense) {
    int64_t n = A.dim;
    dense.assign(n * n, 0.0);

    // Fill from lower triangular storage
    for (int64_t k = 0; k < A.num_entries; ++k) {
        int64_t i = A.row_indices[k];
        int64_t j = A.col_indices[k];
        double val = A.values[k];

        // Column-major indexing: A[i,j] = dense[i + j*n]
        dense[i + j * n] = val;
        if (i != j) {
            dense[j + i * n] = val;  // Symmetric: A[j,i] = A[i,j]
        }
    }
}

/// Compute eigenvalue range using LAPACK++ syev
void compute_eigenvalue_range(
    const SparseSPDMatrix& A,
    double& lambda_min,
    double& lambda_max
) {
    int64_t n = A.dim;

    // Convert to dense
    std::vector<double> dense;
    sparse_to_dense_symmetric(A, dense);

    // Eigenvalues output array
    std::vector<double> eigenvalues(n);

    // Call LAPACK syev (eigenvalues only, no eigenvectors)
    // Job='N' means eigenvalues only, uplo='L' means lower triangular input
    int64_t info = lapack::syev(
        lapack::Job::NoVec,
        lapack::Uplo::Lower,
        n,
        dense.data(),
        n,
        eigenvalues.data()
    );

    if (info != 0) {
        throw std::runtime_error("LAPACK syev failed with info = " + std::to_string(info));
    }

    // Eigenvalues are returned in ascending order
    lambda_min = eigenvalues[0];
    lambda_max = eigenvalues[n - 1];
}

/// Write sparse matrix in symmetric Matrix Market format with full precision
void write_sparse_matrix_symmetric(
    const std::string& filename,
    const SparseSPDMatrix& A
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "%%MatrixMarket matrix coordinate real symmetric\n";
    file << A.dim << " " << A.dim << " " << A.num_entries << "\n";

    // Write lower triangular entries (1-indexed) with full double precision
    file << std::scientific << std::setprecision(17);
    for (int64_t k = 0; k < A.num_entries; ++k) {
        file << (A.row_indices[k] + 1) << " " << (A.col_indices[k] + 1) << " " << A.values[k] << "\n";
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
        std::cerr << "\nNOTE: Uses BLAS++/LAPACK++ (no Eigen dependency)." << std::endl;
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

    // Create output directory (ignore error if already exists)
    mkdir(output_dir.c_str(), 0755);

    printf("\n=== Sparse SPD Matrix Generation for CQRRT Conditioning Study ===\n");
    printf("Output directory: %s\n", output_dir.c_str());
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Number of matrices: %ld\n", num_matrices);
    printf("Condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("Structure: Pentadiagonal (bandwidth = 2)\n");
    printf("Implementation: BLAS++/LAPACK++ (no Eigen)\n");
    printf("================================================================\n\n");

    // Initialize RNG
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);

    // Step 1: Generate seed banded SPD matrix (without condition number control)
    printf("Step 1: Generating banded SPD seed matrix...\n");
    int64_t bandwidth = 2;  // Pentadiagonal: diagonal + 2 bands above/below
    SparseSPDMatrix A_seed = generate_banded_spd_seed(n, bandwidth, state);

    // Compute seed matrix properties using LAPACK++ eigensolve
    double lambda_min_seed, lambda_max_seed;
    compute_eigenvalue_range(A_seed, lambda_min_seed, lambda_max_seed);
    double cond_seed = lambda_max_seed / lambda_min_seed;

    // Compute full nnz (for symmetric matrix: lower + upper - diagonal = 2*num_entries - n)
    int64_t nnz_full = 2 * A_seed.num_entries - n;
    double density = (double)nnz_full / (n * n);

    printf("Seed matrix properties:\n");
    printf("  Non-zeros (full): %ld (%.4f%% density)\n", nnz_full, density * 100.0);
    printf("  Non-zeros (lower tri): %ld\n", A_seed.num_entries);
    printf("  Eigenvalue range: [%.6e, %.6e]\n", lambda_min_seed, lambda_max_seed);
    printf("  Condition number: %.6e\n\n", cond_seed);

    // Step 2: Generate log-spaced condition numbers and create matrices
    int num_threads = omp_get_max_threads();
    printf("Step 2: Generating %ld matrices with diagonal regularization (using %d threads)...\n\n",
           num_matrices, num_threads);

    double log_min = std::log10(min_cond);
    double log_max = std::log10(max_cond);

    // Pre-compute which matrices are valid (alpha > -lambda_min_seed)
    // and store their parameters for parallel generation
    struct MatrixGenParams {
        int64_t output_idx;
        double alpha;
        double actual_cond;  // Computed analytically
        std::string filename;
    };
    std::vector<MatrixGenParams> valid_matrices;

    int64_t output_idx = 0;
    for (int64_t i = 0; i < num_matrices; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double log_cond = log_min + t * (log_max - log_min);
        double target_cond = std::pow(10.0, log_cond);

        // Compute alpha to achieve target condition number
        // For A_new = A + alpha*I:
        // cond_new = (lambda_max + alpha) / (lambda_min + alpha)
        // Solving: alpha = (lambda_max - target_cond * lambda_min) / (target_cond - 1)
        double alpha = (lambda_max_seed - target_cond * lambda_min_seed) / (target_cond - 1.0);

        // Skip if alpha is too negative (would make matrix non-SPD)
        if (alpha <= -lambda_min_seed) {
            printf("Skipping matrix %ld/%ld: target kappa=%.6e requires alpha=%.6e (violates SPD)\n",
                   i + 1, num_matrices, target_cond, alpha);
            continue;
        }

        // Compute actual condition number analytically (no eigensolve needed!)
        // actual_cond = (lambda_max_seed + alpha) / (lambda_min_seed + alpha)
        double actual_cond = (lambda_max_seed + alpha) / (lambda_min_seed + alpha);

        // Generate filename
        char fname_buf[256];
        snprintf(fname_buf, sizeof(fname_buf), "sparse_spd_%04ld_cond_%.2e.mtx", output_idx, actual_cond);

        valid_matrices.push_back({output_idx, alpha, actual_cond, std::string(fname_buf)});
        output_idx++;
    }

    int64_t matrices_to_generate = static_cast<int64_t>(valid_matrices.size());
    printf("Will generate %ld matrices (skipped %ld due to SPD constraints)\n\n",
           matrices_to_generate, num_matrices - matrices_to_generate);

    // Parallel matrix generation
    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < matrices_to_generate; ++idx) {
        const MatrixGenParams& params = valid_matrices[idx];

        // Create regularized matrix (each thread gets its own copy)
        SparseSPDMatrix A_reg = A_seed.with_diagonal_shift(params.alpha);

        // Save matrix
        std::string filepath = output_dir + "/" + params.filename;
        write_sparse_matrix_symmetric(filepath, A_reg);

        #pragma omp critical
        {
            printf("Generated matrix %ld/%ld: kappa=%.2e, alpha=%.2e -> %s\n",
                   params.output_idx + 1, matrices_to_generate, params.actual_cond,
                   params.alpha, params.filename.c_str());
        }
    }

    // Write metadata file (sequential, after all matrices generated)
    std::string metadata_file = output_dir + "/metadata.txt";
    std::ofstream meta(metadata_file);
    meta << "# Sparse SPD Matrix Set Metadata\n";
    meta << "# Generated for CQRRT_linops conditioning study\n";
    meta << "# Method: Banded structure + diagonal regularization\n";
    meta << "# Implementation: BLAS++/LAPACK++ (no Eigen)\n";
    meta << "# OpenMP threads used: " << num_threads << "\n";
    meta << "matrix_size: " << n << "\n";
    meta << "num_matrices: " << matrices_to_generate << "\n";
    meta << "min_condition_number: " << min_cond << "\n";
    meta << "max_condition_number: " << max_cond << "\n";
    meta << "structure: pentadiagonal (bandwidth=2)\n";
    meta << "nnz_per_matrix_lower: " << A_seed.num_entries << "\n";
    meta << "nnz_per_matrix_full: " << nnz_full << "\n";
    meta << "seed_lambda_min: " << std::scientific << lambda_min_seed << "\n";
    meta << "seed_lambda_max: " << std::scientific << lambda_max_seed << "\n";
    meta << "seed_condition_number: " << std::scientific << cond_seed << "\n";
    meta << "spacing: logarithmic\n";
    meta << "# Format: index, condition_number, filename\n";
    meta << "# ----------------------------------------\n";

    for (const MatrixGenParams& params : valid_matrices) {
        meta << params.output_idx << ", " << std::scientific << params.actual_cond
             << ", " << params.filename << "\n";
    }
    meta.close();

    printf("\n================================================================\n");
    printf("Generation complete!\n");
    printf("Generated %ld/%ld matrices (some targets may be skipped if they violate SPD)\n",
           matrices_to_generate, num_matrices);
    printf("Metadata saved to: %s\n", metadata_file.c_str());
    printf("================================================================\n");

    return 0;
}
#endif
