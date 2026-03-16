#pragma once

// Batch CLI driver for generating sparse SPD matrices with varying condition numbers.
// Strategy: build one pentadiagonal seed matrix, then apply diagonal shifts A + alpha*I
// to achieve log-spaced target condition numbers analytically (no per-matrix eigensolve).
//
// Generation primitives (SparseSPDMatrix, gen_banded_spd_seed, compute_eigenvalue_range,
// write_shifted_spd_matrix) live in RandLAPACK::gen (rl_gen.hh).
//
// Usage:
//   <exe> <output_dir> <matrix_size> <num_matrices> <min_cond> <max_cond>

#include <RandLAPACK.hh>
#include "rl_gen.hh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <omp.h>

namespace RandLAPACK::testing {

inline int run_generate_sparse_spd(int argc, char* argv[]) {
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

    mkdir(output_dir.c_str(), 0755);

    printf("\n=== Sparse SPD Matrix Generation ===\n");
    printf("Output directory: %s\n", output_dir.c_str());
    printf("Matrix size: %ld x %ld\n", n, n);
    printf("Number of matrices: %ld\n", num_matrices);
    printf("Condition number range: %.2e to %.2e (log-spaced)\n", min_cond, max_cond);
    printf("Structure: Pentadiagonal (bandwidth = 2)\n");
    printf("====================================\n\n");

    // Step 1: build seed matrix
    printf("Step 1: Generating banded SPD seed matrix...\n");
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);
    int64_t bandwidth = 2;
    auto A_seed = RandLAPACK::gen::gen_banded_spd_seed(n, bandwidth, state);

    double lambda_min_seed, lambda_max_seed;
    RandLAPACK::gen::compute_eigenvalue_range(A_seed, bandwidth, lambda_min_seed, lambda_max_seed);
    double cond_seed   = lambda_max_seed / lambda_min_seed;
    int64_t nnz_full   = 2 * A_seed.num_entries - n;
    double density     = static_cast<double>(nnz_full) / (n * n);

    printf("Seed properties:\n");
    printf("  nnz (full): %ld (%.4f%% density)\n", nnz_full, density * 100.0);
    printf("  Eigenvalue range: [%.6e, %.6e]\n", lambda_min_seed, lambda_max_seed);
    printf("  Condition number: %.6e\n\n", cond_seed);

    // Step 2: compute diagonal shifts and collect valid matrices
    int num_threads = omp_get_max_threads();
    printf("Step 2: Generating %ld matrices with diagonal regularization (%d threads)...\n\n",
           num_matrices, num_threads);

    struct MatrixGenParams { int64_t output_idx; double alpha; double actual_cond; std::string filename; };
    std::vector<MatrixGenParams> valid_matrices;
    double log_min = std::log10(min_cond), log_max = std::log10(max_cond);

    for (int64_t i = 0; i < num_matrices; ++i) {
        double t          = static_cast<double>(i) / static_cast<double>(num_matrices - 1);
        double target_cond = std::pow(10.0, log_min + t * (log_max - log_min));
        // alpha such that (lambda_max + alpha) / (lambda_min + alpha) = target_cond
        double alpha      = (lambda_max_seed - target_cond * lambda_min_seed) / (target_cond - 1.0);
        if (alpha <= -lambda_min_seed) {
            printf("Skipping %ld/%ld: kappa=%.6e requires alpha=%.6e (violates SPD)\n",
                   i + 1, num_matrices, target_cond, alpha);
            continue;
        }
        double actual_cond = (lambda_max_seed + alpha) / (lambda_min_seed + alpha);
        char buf[256];
        int64_t idx = static_cast<int64_t>(valid_matrices.size());
        snprintf(buf, sizeof(buf), "sparse_spd_%04ld_cond_%.2e.mtx", idx, actual_cond);
        valid_matrices.push_back({idx, alpha, actual_cond, buf});
    }

    int64_t to_generate = static_cast<int64_t>(valid_matrices.size());
    printf("Will generate %ld matrices (skipped %ld due to SPD constraints)\n\n",
           to_generate, num_matrices - to_generate);

    // Parallel write: A_seed is read-only, each thread writes a different file
    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < to_generate; ++idx) {
        const auto& p = valid_matrices[idx];
        RandLAPACK::gen::write_shifted_spd_matrix(output_dir + "/" + p.filename, A_seed, p.alpha);
        #pragma omp critical
        printf("Generated %ld/%ld: kappa=%.2e, alpha=%.2e -> %s\n",
               p.output_idx + 1, to_generate, p.actual_cond, p.alpha, p.filename.c_str());
    }

    // Metadata
    std::ofstream meta(output_dir + "/metadata.txt");
    meta << "# Sparse SPD matrix set\n";
    meta << "# Method: banded structure + diagonal regularization\n";
    meta << "matrix_size: "          << n             << "\n";
    meta << "num_matrices: "         << to_generate    << "\n";
    meta << "min_condition_number: " << min_cond       << "\n";
    meta << "max_condition_number: " << max_cond       << "\n";
    meta << "structure: pentadiagonal (bandwidth=2)\n";
    meta << "nnz_per_matrix_lower: " << A_seed.num_entries << "\n";
    meta << "nnz_per_matrix_full: "  << nnz_full          << "\n";
    meta << "seed_lambda_min: "      << std::scientific << lambda_min_seed << "\n";
    meta << "seed_lambda_max: "      << std::scientific << lambda_max_seed << "\n";
    meta << "seed_condition_number: "<< std::scientific << cond_seed       << "\n";
    meta << "spacing: logarithmic\n";
    meta << "# index, condition_number, filename\n";
    for (const auto& p : valid_matrices)
        meta << p.output_idx << ", " << std::scientific << p.actual_cond << ", " << p.filename << "\n";

    printf("\nDone. Generated %ld/%ld matrices. Metadata: %s/metadata.txt\n",
           to_generate, num_matrices, output_dir.c_str());
    return 0;
}

} // namespace RandLAPACK::testing
