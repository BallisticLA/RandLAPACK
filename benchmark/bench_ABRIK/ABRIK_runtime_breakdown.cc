/*
Unified ABRIK runtime breakdown benchmark — assesses the time taken by each
subcomponent of ABRIK on dense or sparse input matrices (Matrix Market format).

Precision (float or double) is specified as the first CLI argument.
Records all runs (not just the best).

Output: CSV file with '#'-prefixed metadata header, column names, then data rows.
ABRIK allocates U/V/Sigma with new[] internally -> cleanup with delete[].

Subroutine timings (microseconds):
  1. Allocation/free  2. SVD factors  3. UNGQR  4. Reorthogonalization
  5. QR  6. GEMM A  7. Main loop  8. Sketching
  9. R_ii copy  10. S_ii copy  11. Norm R  12. Rest  13. Total

Usage:
  ABRIK_runtime_breakdown <precision> <output_dir> <input.mtx>
                          <num_runs> <num_block_sizes> <num_matmul_sizes>
                          <block_sizes...> <matmul_sizes...>
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "ext_matrix_io.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <ctime>
#include <string>
#include <vector>

using Subroutines = RandLAPACK::ABRIKSubroutines;

// Core benchmark loop — templated on LinOp type.
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp>
static void run_all_configs(
    LinOp& A_op,
    int64_t num_runs,
    bool use_cqrrt,
    std::vector<int64_t>& block_sizes,
    std::vector<int64_t>& matmul_counts,
    RandBLAS::RNGState<RNG>& state,
    std::ofstream& outfile)
{
    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    RandLAPACK::ABRIK<T, RNG> ABRIK(false, true, tol);  // timing enabled
    if (use_cqrrt)
        ABRIK.qr_exp = Subroutines::QR_explicit::cqrrt;

    T* U = nullptr;
    T* V = nullptr;
    T* Sigma = nullptr;

    for (auto b_sz : block_sizes) {
        for (auto num_matmuls : matmul_counts) {
            ABRIK.max_krylov_iters = (int) num_matmuls;

            for (int run = 0; run < num_runs; ++run) {
                printf("\nBlock size %ld, num matmuls %ld. Run %d.\n", b_sz, num_matmuls, run);

                auto state_alg = state;
                ABRIK.call(A_op, b_sz, U, V, Sigma, state_alg);

                // Write CSV row: b_sz, num_matmuls, then 13 timing columns
                outfile << b_sz << ", " << num_matmuls;
                for (const auto &t : ABRIK.times)
                    outfile << ", " << t;
                outfile << "\n";
                outfile.flush();

                delete[] U;     U     = nullptr;
                delete[] V;     V     = nullptr;
                delete[] Sigma; Sigma = nullptr;
            }
        }
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input.mtx>"
                  << " <num_runs> <num_block_sizes> <num_matmul_sizes>"
                  << " <block_sizes...> <matmul_sizes...>"
                  << std::endl;
        return;
    }

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int num_runs           = std::stoi(argv[4]);
    int num_b_sz           = std::stoi(argv[5]);
    int num_mm             = std::stoi(argv[6]);

    std::vector<int64_t> block_sizes, matmul_counts;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stol(argv[7 + i]));
    for (int i = 0; i < num_mm; ++i)
        matmul_counts.push_back(std::stol(argv[7 + num_b_sz + i]));

    // Optional trailing args: [sub_ratio] [use_cqrrt]
    int args_consumed = 7 + num_b_sz + num_mm;
    double sub_ratio = (argc > args_consumed) ? std::stod(argv[args_consumed]) : 1.0;
    bool cli_cqrrt = (argc > args_consumed + 1) ? (std::stoi(argv[args_consumed + 1]) != 0) : false;

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state = RandBLAS::RNGState();

    // --- Load matrix (auto-detects .mtx vs .txt) ---
    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;

    // --- Open output CSV ---
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string out_filename = std::string(date_prefix) + "ABRIK_runtime_breakdown.csv";
    std::string out_path = (output_dir != ".") ? output_dir + "/" + out_filename : out_filename;
    std::ofstream outfile(out_path);

    std::ostringstream oss_b, oss_m;
    for (auto v : block_sizes) oss_b << v << ", ";
    for (auto v : matmul_counts) oss_m << v << ", ";

    outfile << "# ABRIK Runtime Breakdown Benchmark (unified dense/sparse)\n"
            << "# Precision: " << argv[1] << "\n"
            << "# Input matrix: " << input_path << "\n"
            << "# Input size: " << m << " x " << n << "\n"
            << "# Format: " << (mat.is_sparse ? "sparse" : "dense") << "\n"
            << "# Block sizes: " << oss_b.str() << "\n"
            << "# Matmul counts: " << oss_m.str() << "\n"
            << "# Runs per configuration: " << num_runs << "\n"
            << "# Tolerance: " << tol << "\n"
            << "# Timings in microseconds\n";
    outfile << "b_sz, num_matmuls, "
            << "allocation_t, get_factors_t, ungqr_t, reorth_t, qr_t, gemm_A_t, "
            << "main_loop_t, sketching_t, r_cpy_t, s_cpy_t, norm_t, t_rest, total_t\n";
    outfile.flush();

    auto t_total = steady_clock::now();

    if (mat.is_sparse) {
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>>
            A_op(m, n, *mat.csc);
        run_all_configs<T>(A_op, num_runs, cli_cqrrt, block_sizes, matmul_counts, state, outfile);
    } else {
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, mat.data(), m, Layout::ColMajor);
        run_all_configs<T>(A_op, num_runs, cli_cqrrt, block_sizes, matmul_counts, state, outfile);
    }

    long total_us = duration_cast<microseconds>(steady_clock::now() - t_total).count();
    printf("\nTOTAL BENCHMARK TIME: %.2f seconds\n", total_us / 1e6);
    printf("Results: %s\n", out_path.c_str());
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision: double|float> <output_dir> <input.mtx>"
                  << " <num_runs> <num_block_sizes> <num_matmul_sizes>"
                  << " <block_sizes...> <matmul_sizes...>"
                  << std::endl;
        return 1;
    }
    std::string precision = argv[1];
    if (precision == "double")     run_benchmark<double>(argc, argv);
    else if (precision == "float") run_benchmark<float>(argc, argv);
    else {
        std::cerr << "Error: precision must be 'double' or 'float'" << std::endl;
        return 1;
    }
    return 0;
}
