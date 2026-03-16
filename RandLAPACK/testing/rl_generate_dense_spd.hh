#pragma once

// Batch CLI driver for generating dense SPD matrices with varying condition numbers.
// The actual matrix generation is delegated to RandLAPACK::testing::generate_spd_matrix_file.
//
// Usage:
//   <exe> <output_dir> <matrix_size> <num_matrices> <min_cond> <max_cond>

#include "RandLAPACK/testing/rl_test_utils.hh"
#include <RandBLAS.hh>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <omp.h>

namespace RandLAPACK::testing {

inline int run_generate_dense_spd(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <output_dir> <matrix_size> <num_matrices> <min_cond> <max_cond>\n";
        return 1;
    }

    std::string output_dir  = argv[1];
    int64_t n               = std::stol(argv[2]);
    int64_t num_matrices    = std::stol(argv[3]);
    double  min_cond        = std::stod(argv[4]);
    double  max_cond        = std::stod(argv[5]);

    if (n <= 0 || num_matrices <= 0) {
        std::cerr << "Error: matrix_size and num_matrices must be positive\n";
        return 1;
    }
    if (min_cond <= 1.0 || max_cond <= min_cond) {
        std::cerr << "Error: Need 1 < min_cond < max_cond\n";
        return 1;
    }

    std::filesystem::create_directories(output_dir);

    int num_threads = omp_get_max_threads();
    printf("\n=== Dense SPD Matrix Generation ===\n");
    printf("Output directory: %s\n",  output_dir.c_str());
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Number of matrices: %ld\n", num_matrices);
    printf("Condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("OpenMP threads: %d\n", num_threads);
    printf("===================================\n\n");

    // Pre-compute log-spaced condition numbers and filenames
    struct MatrixParams { int64_t idx; double cond_num; std::string filepath; std::string filename; };
    std::vector<MatrixParams> matrices(num_matrices);
    double log_min = std::log10(min_cond), log_max = std::log10(max_cond);
    for (int64_t i = 0; i < num_matrices; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double cond_num = std::pow(10.0, log_min + t * (log_max - log_min));
        char buf[256];
        snprintf(buf, sizeof(buf), "dense_spd_%04ld_cond_%.2e.mtx", i, cond_num);
        matrices[i] = {i, cond_num, output_dir + "/" + buf, buf};
    }

    // Parallel generation: each matrix gets an independent RNG state from its index
    #pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < num_matrices; ++i) {
        auto state = RandBLAS::RNGState<r123::Philox4x32>(static_cast<uint32_t>(i));
        RandLAPACK::testing::generate_spd_matrix_file<double>(matrices[i].filepath, n, matrices[i].cond_num, state);
        #pragma omp critical
        printf("Generated %ld/%ld: kappa=%.6e -> %s\n",
               matrices[i].idx + 1, num_matrices, matrices[i].cond_num, matrices[i].filename.c_str());
    }

    // Metadata
    std::ofstream meta(output_dir + "/metadata.txt");
    meta << "# Dense SPD matrix set\n";
    meta << "# Method: eigenvalue decomposition (A = Q * Lambda * Q^T)\n";
    meta << "matrix_size: "          << n            << "\n";
    meta << "num_matrices: "         << num_matrices  << "\n";
    meta << "min_condition_number: " << min_cond      << "\n";
    meta << "max_condition_number: " << max_cond      << "\n";
    meta << "spacing: logarithmic\n";
    meta << "# index, condition_number, filename\n";
    for (const auto& p : matrices)
        meta << p.idx << ", " << std::scientific << p.cond_num << ", " << p.filename << "\n";

    printf("\nDone. Metadata: %s/metadata.txt\n", output_dir.c_str());
    return 0;
}

} // namespace RandLAPACK::testing
