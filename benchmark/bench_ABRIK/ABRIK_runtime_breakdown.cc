/*
ABRIK runtime breakdown: the time each subcomponent of ABRIK takes, on dense or sparse
input, over a grid of block sizes and Krylov iteration budgets. Every run is recorded,
and every run repeats the same sketch (one RNG seed), so the spread across runs is
machine noise rather than a different draw.

Output CSV: '#' metadata lines, the column header, then one row per
(b_sz, num_matmuls, run):
  b_sz, num_matmuls, allocation_t, get_factors_t, ungqr_t, reorth_t, qr_t, gemm_A_t,
  main_loop_t, sketching_t, r_cpy_t, s_cpy_t, norm_t, t_rest, total_t
in microseconds, the thirteen timing slots of ABRIK::times. num_matmuls is the budget
handed to ABRIK (max_krylov_iters); BK applies the operator once more than that for the
initial block, and can stop earlier on its own criteria.

Usage:
  ABRIK_runtime_breakdown <precision> <output_dir> <input_file> <num_runs>
                          <num_block_sizes> <num_matmul_sizes> <block_sizes...>
                          <matmul_sizes...> [sub_ratio] [use_cqrrt]

  precision  = double | float
  input_file = .mtx (array or coordinate), .bin, or whitespace-delimited text
  sub_ratio  = keep the top-left fraction of rows and columns (default 1.0)
  use_cqrrt  = 1 to use CQRRT instead of Householder QR inside ABRIK (default 0)
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_linops.hh"
#include "ext_matrix_io.hh"
#include "abrik_bench_common.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char* kUsage =
    "<precision> <output_dir> <input_file> <num_runs> <num_block_sizes> <num_matmul_sizes>"
    " <block_sizes...> <matmul_sizes...> [sub_ratio] [use_cqrrt]";

template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp>
static void run_all_configs(
    LinOp& A_op, T tol, int num_runs, bool use_cqrrt,
    const std::vector<int64_t>& block_sizes, const std::vector<int64_t>& matmul_counts,
    const RandBLAS::RNGState<RNG>& state, std::ofstream& outfile)
{
    RandLAPACK::ABRIK<T, RNG> ABRIK(false, true, tol);   // timing on
    if (use_cqrrt)
        ABRIK.qr_exp = RandLAPACK::ABRIKSubroutines::QR_explicit::cqrrt;

    T* U = nullptr;
    T* V = nullptr;
    T* Sigma = nullptr;

    for (auto b_sz : block_sizes) {
        for (auto num_matmuls : matmul_counts) {
            ABRIK.max_krylov_iters = (int) num_matmuls;
            for (int run = 0; run < num_runs; ++run) {
                printf("\nBlock size %ld, num matmuls %ld. Run %d.\n", (long) b_sz, (long) num_matmuls, run);
                auto state_alg = state;
                int status = ABRIK.call(A_op, b_sz, U, V, Sigma, state_alg);
                if (status == 0) {
                    outfile << b_sz << ", " << num_matmuls;
                    for (const auto& t : ABRIK.times)
                        outfile << ", " << t;
                    outfile << "\n";
                    outfile.flush();
                } else {
                    fprintf(stderr, "ABRIK failed with status %d (b_sz %ld, num_matmuls %ld, run %d); no row written\n",
                            status, (long) b_sz, (long) num_matmuls, run);
                }
                delete[] U;     U     = nullptr;
                delete[] V;     V     = nullptr;
                delete[] Sigma; Sigma = nullptr;
            }
        }
    }
}

template <typename T>
static int run_benchmark(int argc, char* argv[]) {
    if (argc < 7) return abrik_usage(argv[0], kUsage);

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int num_runs           = std::stoi(argv[4]);
    int num_b_sz           = std::stoi(argv[5]);
    int num_mm             = std::stoi(argv[6]);
    if (num_runs < 1 || num_b_sz < 1 || num_mm < 1 || argc < 7 + num_b_sz + num_mm) {
        std::cerr << "Error: num_runs, num_block_sizes and num_matmul_sizes must be >= 1, with every size given\n";
        return abrik_usage(argv[0], kUsage);
    }
    std::vector<int64_t> block_sizes, matmul_counts;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stol(argv[7 + i]));
    for (int i = 0; i < num_mm; ++i)
        matmul_counts.push_back(std::stol(argv[7 + num_b_sz + i]));
    int args_consumed = 7 + num_b_sz + num_mm;
    double sub_ratio = (argc > args_consumed)     ? std::stod(argv[args_consumed])     : 1.0;
    bool use_cqrrt   = (argc > args_consumed + 1) ? (std::stoi(argv[args_consumed + 1]) != 0) : false;

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.85);
    auto state = RandBLAS::RNGState();

    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;

    std::ofstream outfile;
    std::string out_path = abrik_open_csv(output_dir, "ABRIK_runtime_breakdown", outfile);
    if (!outfile) return 1;

    std::ostringstream oss_b, oss_m;
    for (auto v : block_sizes) oss_b << v << ", ";   // comma list, as the readers split it
    for (auto v : matmul_counts) oss_m << v << ", ";

    outfile << "# ABRIK runtime breakdown\n"
            << "# RANDLAPACK_GIT_COMMIT=" << abrik_build_commit() << "\n"
            << "# Precision: " << argv[1] << "\n"
            << "# Input matrix: " << input_path << "\n"
            << "# Input size: " << m << " x " << n << "\n"
            << "# Format: " << (mat.is_sparse ? "sparse" : "dense") << "\n"
            << "# Block sizes: " << oss_b.str() << "\n"
            << "# Matmul counts: " << oss_m.str() << "\n"
            << "# num_matmuls is max_krylov_iters; the initial block is one more application\n"
            << "# Runs per configuration: " << num_runs << " (same sketch every run)\n"
            << "# ABRIK QR: " << (use_cqrrt ? "CQRRT" : "Householder") << "\n"
            << "# Tolerance: " << tol << "\n"
            << "# Timings in microseconds\n"
            << "b_sz, num_matmuls, "
            << "allocation_t, get_factors_t, ungqr_t, reorth_t, qr_t, gemm_A_t, "
            << "main_loop_t, sketching_t, r_cpy_t, s_cpy_t, norm_t, t_rest, total_t\n";
    outfile.flush();

    auto t_total = steady_clock::now();
    if (mat.is_sparse) {
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>> A_op(m, n, *mat.csc);
        run_all_configs<T>(A_op, tol, num_runs, use_cqrrt, block_sizes, matmul_counts, state, outfile);
    } else {
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, mat.data(), m, Layout::ColMajor);
        run_all_configs<T>(A_op, tol, num_runs, use_cqrrt, block_sizes, matmul_counts, state, outfile);
    }
    long total_us = duration_cast<microseconds>(steady_clock::now() - t_total).count();
    printf("\nTOTAL BENCHMARK TIME: %.2f seconds\n", total_us / 1e6);
    printf("Results: %s\n", out_path.c_str());
    return 0;
}

int main(int argc, char* argv[]) {
    return abrik_bench_main(argc, argv, kUsage, run_benchmark<double>, run_benchmark<float>);
}
