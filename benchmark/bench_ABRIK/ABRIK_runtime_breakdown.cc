/*
ABRIK runtime breakdown benchmark - assesses the time taken by each subcomponent of ABRIK.
Precision (float or double) is specified as the first CLI argument.
Records all data, not just the best.

Output: CSV file with '#'-prefixed metadata header, column names, then data rows.
ABRIK allocates U/V/Sigma with new[] internally -> cleanup with delete[].

Subroutine timings (microseconds):
  1. Allocation/free  2. SVD factors  3. UNGQR  4. Reorthogonalization
  5. QR  6. GEMM A  7. Main loop  8. Sketching
  9. R_ii copy  10. S_ii copy  11. Norm R  12. Rest  13. Total
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

using Subroutines = RandLAPACK::ABRIKSubroutines;

template <typename T>
struct ABRIK_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    T* A;
    T* U;
    T* V;
    T* Sigma;

    ABRIK_benchmark_data(int64_t m, int64_t n, T tol)
    {
        row       = m;
        col       = n;
        tolerance = tol;
        A         = new T[m * n]();
        U         = nullptr;
        V         = nullptr;
        Sigma     = nullptr;
    }

    ~ABRIK_benchmark_data() {
        delete[] A;
    }
};

template <typename T, typename RNG>
static void call_all_algs(
    int64_t num_runs,
    int64_t k,
    int64_t num_matmuls,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;
    bool time_subroutines = true;

    // Additional params setup.
    RandLAPACK::ABRIK<T, r123::Philox4x32> ABRIK(false, time_subroutines, tol);
    ABRIK.max_krylov_iters = num_matmuls;

    auto state_alg = state;
    std::vector<long> inner_timing;

    for (int i = 0; i < num_runs; ++i) {
        printf("\nBlock size %ld, num matmuls %ld. Iteration %d start.\n", k, num_matmuls, i);
        ABRIK.call(m, n, all_data.A, m, k, all_data.U, all_data.V, all_data.Sigma, state_alg);

        // Write CSV data row: b_sz, num_matmuls, then timing columns
        outfile << k << ", " << num_matmuls;
        for (const auto &t : ABRIK.times)
            outfile << ", " << t;
        outfile << "\n";
        outfile.flush();

        // Cleanup ABRIK outputs (new[])
        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;

        state_alg = state;
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {

    if (argc < 12) {
        std::cerr << "Usage: " << argv[0] << " <precision> <output_directory_path> <input_matrix_path> <num_runs> <num_rows> <num_cols> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return;
    }

    int num_runs        = std::stol(argv[4]);
    int64_t m_expected  = std::stol(argv[5]);
    int64_t n_expected  = std::stol(argv[6]);
    int64_t custom_rank = std::stol(argv[7]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[8]); ++i)
        b_sz.push_back(std::stoi(argv[i + 10]));
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[9]); ++i)
        matmuls.push_back(std::stoi(argv[i + 10 + std::stol(argv[8])]));
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    T tol               = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    int64_t m = 0, n = 0;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[3];
    m_info.workspace_query_mod = 1;
    RandLAPACK::gen::mat_gen<T>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected input size (" << m_expected << ", " << n_expected << ") did not match actual input size (" << m << ", " << n << "). Aborting." << std::endl;
        return;
    }

    // Allocate basic workspace.
    ABRIK_benchmark_data<T> all_data(m, n, tol);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    printf("Finished data preparation\n");

    // Build output file path
    std::string output_filename = "_ABRIK_runtime_breakdown.csv";
    std::string path;
    if (std::string(argv[2]) != ".") {
        path = argv[2] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Write metadata header (prefixed with # for easy parsing)
    file << "# ABRIK Runtime Breakdown Benchmark\n"
         << "# Precision: " << argv[1] << "\n"
         << "# Input matrix: " << argv[3] << "\n"
         << "# Input size: " << m << " x " << n << "\n"
         << "# Target rank: " << custom_rank << "\n"
         << "# Krylov block sizes: " << b_sz_string << "\n"
         << "# Matmul counts: " << matmuls_string << "\n"
         << "# Runs per configuration: " << num_runs << "\n"
         << "# Tolerance: " << tol << "\n"
         << "# Timings in microseconds\n";
    // Write CSV column header
    file << "b_sz, num_matmuls, "
         << "allocation_t, get_factors_t, ungqr_t, reorth_t, qr_t, gemm_A_t, "
         << "main_loop_t, sketching_t, r_cpy_t, s_cpy_t, norm_t, t_rest, total_t\n";
    file.flush();

    size_t i = 0, j = 0;
    for (; i < b_sz.size(); ++i) {
        for (; j < matmuls.size(); ++j) {
            call_all_algs(num_runs, b_sz[i], matmuls[j], all_data, state_constant, file);
        }
        j = 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <precision: double|float> <output_directory_path> <input_matrix_path> <num_runs> <num_rows> <num_cols> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    std::string precision = argv[1];
    if (precision == "double") {
        run_benchmark<double>(argc, argv);
    } else if (precision == "float") {
        run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'" << std::endl;
        return 1;
    }
    return 0;
}
