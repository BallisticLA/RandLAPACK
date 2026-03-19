/*
ABRIK runtime breakdown benchmark (sparse) - assesses the time taken by each subcomponent of ABRIK
on sparse input matrices read from Matrix Market format.
Precision (float or double) is specified as the first CLI argument.

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
#include <ctime>

#include <fast_matrix_market/fast_matrix_market.hpp>

using Subroutines = RandLAPACK::ABRIKSubroutines;

template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
struct ABRIK_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    SpMat* A_input;
    T* U;
    T* V;
    T* Sigma;

    ABRIK_benchmark_data(int64_t m, int64_t n, T tol)
    {
        row       = m;
        col       = n;
        tolerance = tol;
        U         = nullptr;
        V         = nullptr;
        Sigma     = nullptr;
    }

    ~ABRIK_benchmark_data() {}
};

template <typename T>
RandBLAS::sparse_data::coo::COOMatrix<T> from_matrix_market(std::string fn) {
    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows, cols, vals
    );

    RandBLAS::sparse_data::coo::COOMatrix<T> out(n_rows, n_cols);
    reserve_coo(vals.size(), out);
    for (int i = 0; i < out.nnz; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }

    return out;
}

template <typename T>
RandBLAS::CSCMatrix<T> format_conversion(int64_t m, int64_t n, RandBLAS::COOMatrix<T>& input_mat_coo)
{
    // Grab the leading principal submatrix
    RandBLAS::COOMatrix<T> input_mat_transformed(m, n);

    // Count nonzeros in the submatrix
    int64_t nnz_sub = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n)
            ++nnz_sub;
    }

    reserve_coo(nnz_sub, input_mat_transformed);

    int64_t ell = 0;
    for (int64_t i = 0; i < input_mat_coo.nnz; ++i) {
        if (input_mat_coo.rows[i] < m && input_mat_coo.cols[i] < n) {
            input_mat_transformed.rows[ell] = input_mat_coo.rows[i];
            input_mat_transformed.cols[ell] = input_mat_coo.cols[i];
            input_mat_transformed.vals[ell] = input_mat_coo.vals[i];
            ++ell;
        }
    }

    RandBLAS::CSCMatrix<T> input_mat_csc(m, n);
    RandBLAS::conversions::coo_to_csc(input_mat_transformed, input_mat_csc);

    return input_mat_csc;
}

template <typename T, typename RNG, RandBLAS::sparse_data::SparseMatrix SpMat>
static void call_all_algs(
    int64_t num_runs,
    int64_t k,
    int64_t num_matmuls,
    ABRIK_benchmark_data<T, SpMat> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile,
    int use_cqrrt) {

    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;
    bool time_subroutines = true;

    // Additional params setup.
    RandLAPACK::ABRIK<T, r123::Philox4x32> ABRIK(false, time_subroutines, tol);
    ABRIK.max_krylov_iters = num_matmuls;
    if (use_cqrrt)
        ABRIK.qr_exp = Subroutines::QR_explicit::cqrrt;

    auto state_alg = state;

    for (int i = 0; i < num_runs; ++i) {
        printf("\nBlock size %ld, num matmuls %ld. Iteration %d start.\n", k, num_matmuls, i);
        ABRIK.call(m, n, *all_data.A_input, m, k, all_data.U, all_data.V, all_data.Sigma, state_alg);

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

    if (argc < 8) {
        std::cerr << "Usage: " << argv[0] << " <precision> <output_directory_path> <input_matrix_path> <num_runs> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return;
    }

    T submatrix_dim_ratio = (T)0.5;

    int num_runs        = std::stol(argv[4]);
    int64_t custom_rank = std::stol(argv[5]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[6]); ++i)
        b_sz.push_back(std::stoi(argv[i + 8]));
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[7]); ++i)
        matmuls.push_back(std::stoi(argv[i + 8 + std::stol(argv[6])]));
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();

    T tol               = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;

    // Read the input matrix market data
    auto input_mat_coo = from_matrix_market<T>(std::string(argv[3]));
    auto m = (int64_t)(input_mat_coo.n_rows * submatrix_dim_ratio);
    auto n = (int64_t)(input_mat_coo.n_cols * submatrix_dim_ratio);

    // Convert COO to CSC (grabs leading principal submatrix)
    auto input_mat_csc = format_conversion<T>(m, n, input_mat_coo);

    // Allocate basic workspace.
    ABRIK_benchmark_data<T, RandBLAS::sparse_data::CSCMatrix<T>> all_data(m, n, tol);
    all_data.A_input = &input_mat_csc;

    printf("Finished data preparation\n");

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    // Build output file path
    std::string output_filename = std::string(date_prefix) + "ABRIK_runtime_breakdown_sparse.csv";
    std::string path;
    if (std::string(argv[2]) != ".") {
        path = std::string(argv[2]) + "/" + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Write metadata header (prefixed with # for easy parsing)
    file << "# ABRIK Runtime Breakdown Benchmark (Sparse)\n"
         << "# Precision: " << argv[1] << "\n"
         << "# Input matrix: " << argv[3] << "\n"
         << "# Input size: " << m << " x " << n << " (submatrix ratio: " << submatrix_dim_ratio << ")\n"
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

    auto start_total = steady_clock::now();

    int use_cqrrt = 1;
    size_t i = 0, j = 0;
    for (; i < b_sz.size(); ++i) {
        for (; j < matmuls.size(); ++j) {
            call_all_algs(num_runs, b_sz[i], matmuls[j], all_data, state_constant, file, use_cqrrt);
        }
        j = 0;
    }

    auto stop_total = steady_clock::now();
    long total_us = duration_cast<microseconds>(stop_total - start_total).count();
    printf("\nTOTAL BENCHMARK TIME: %.2f seconds\n", total_us / 1e6);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <precision: double|float> <output_directory_path> <input_matrix_path> <num_runs> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
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
