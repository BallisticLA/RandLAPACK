#if defined(__APPLE__)
int main() {return 0;}
#else

// Generate dense SPD matrices with varying condition numbers for CQRRT conditioning study
// Uses eigenvalue decomposition: A = Q * Lambda * Q^T with controlled eigenvalue distribution

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <fstream>
#include <vector>
#include <filesystem>
#include <omp.h>
#include "../../demos/functions/misc/dm_util.hh"

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_dir> <matrix_size> <num_matrices> <min_cond> <max_cond>"
                  << std::endl;
        std::cerr << "\nArguments:" << std::endl;
        std::cerr << "  output_dir   : Directory to store generated matrices" << std::endl;
        std::cerr << "  matrix_size  : Dimension of square SPD matrices (e.g., 1138)" << std::endl;
        std::cerr << "  num_matrices : Number of matrices to generate (e.g., 100)" << std::endl;
        std::cerr << "  min_cond     : Minimum condition number (e.g., 10)" << std::endl;
        std::cerr << "  max_cond     : Maximum condition number (e.g., 1e16)" << std::endl;
        std::cerr << "\nExample: " << argv[0] << " ./dense_spd 1138 100 10 1e16" << std::endl;
        std::cerr << "\nThis generates DENSE matrices with condition numbers log-spaced" << std::endl;
        std::cerr << "from min_cond to max_cond using eigenvalue decomposition." << std::endl;
        std::cerr << "\nFor SPARSE SPD matrices, use generate_sparse_spd instead." << std::endl;
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

    int num_threads = omp_get_max_threads();
    printf("\n=== Dense SPD Matrix Generation for CQRRT Conditioning Study ===\n");
    printf("Output directory: %s\n", output_dir.c_str());
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Number of matrices: %ld\n", num_matrices);
    printf("Condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("Method: Eigenvalue decomposition (A = Q * Lambda * Q^T)\n");
    printf("OpenMP threads: %d\n", num_threads);
    printf("================================================================\n\n");

    // Generate log-spaced condition numbers
    double log_min = std::log10(min_cond);
    double log_max = std::log10(max_cond);

    // Pre-compute condition numbers and filenames
    struct MatrixParams {
        int64_t idx;
        double cond_num;
        std::string filename;
        std::string filepath;
    };
    std::vector<MatrixParams> matrices(num_matrices);

    for (int64_t i = 0; i < num_matrices; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double log_cond = log_min + t * (log_max - log_min);
        double cond_num = std::pow(10.0, log_cond);

        char filename[256];
        snprintf(filename, sizeof(filename), "dense_spd_%04ld_cond_%.2e.mtx", i, cond_num);

        matrices[i] = {i, cond_num, std::string(filename), output_dir + "/" + std::string(filename)};
    }

    // Parallel matrix generation
    // Each thread gets an independent RNG state based on matrix index
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < num_matrices; ++i) {
        const auto& params = matrices[i];

        // Create independent RNG state for this matrix using index as key offset
        // This ensures reproducibility: same index -> same matrix
        auto state = RandBLAS::RNGState<r123::Philox4x32>(static_cast<uint32_t>(i));

        RandLAPACK_demos::generate_spd_matrix_file<double>(params.filepath, n, params.cond_num, state);

        #pragma omp critical
        {
            printf("Generated matrix %ld/%ld: kappa = %.6e -> %s\n",
                   params.idx + 1, num_matrices, params.cond_num, params.filename.c_str());
        }
    }

    // Write metadata file (sequential, after all matrices generated)
    std::string metadata_file = output_dir + "/metadata.txt";
    std::ofstream meta(metadata_file);
    meta << "# Dense SPD Matrix Set Metadata\n";
    meta << "# Generated for CQRRT_linops conditioning study\n";
    meta << "# Method: Eigenvalue decomposition (A = Q * Lambda * Q^T)\n";
    meta << "# OpenMP threads used: " << num_threads << "\n";
    meta << "matrix_size: " << n << "\n";
    meta << "num_matrices: " << num_matrices << "\n";
    meta << "min_condition_number: " << min_cond << "\n";
    meta << "max_condition_number: " << max_cond << "\n";
    meta << "spacing: logarithmic\n";
    meta << "matrix_type: dense\n";
    meta << "# Format: index, condition_number, filename\n";
    meta << "# ----------------------------------------\n";

    for (const auto& params : matrices) {
        meta << params.idx << ", " << std::scientific << params.cond_num << ", " << params.filename << "\n";
    }
    meta.close();
    printf("\n================================================================\n");
    printf("Generation complete!\n");
    printf("Metadata saved to: %s\n", metadata_file.c_str());
    printf("================================================================\n");

    return 0;
}
#endif
