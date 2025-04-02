/*
RBKI speed benchmark - technically only runs RBKI, but has an option to run SVD (gesdd()) to be compared against RBKI (direct SVD is WAY slower than RBKI). 
The user is required to provide a matrix file to be read, set min and max numbers of large gemms (Krylov iterations) that the algorithm is allowed to perform min and max block sizes that RBKI is to use; 
furthermore, the user is to provide a 'custom rank' parameter (number of singular vectors to approximate by RBKI). 
The benchmark outputs the basic data of a given run, as well as the RBKI runtime and singular vector residual error, 
which is computed as "sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F / sqrt(custom_rank)" (for "custom rank" singular vectors and values).
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>

template <typename T>
struct RBKI_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    std::vector<T> A;
    std::vector<T> U;
    std::vector<T> VT; // RBKI returns V'
    std::vector<T> Sigma;
    std::vector<T> U_cpy;
    std::vector<T> VT_cpy;

    RBKI_benchmark_data(int64_t m, int64_t n, T tol) :
    A(m * n, 0.0),
    U(m * n, 0.0),
    VT(n * n, 0.0),
    Sigma(n, 0.0)
    {
        row = m;
        col = n;
        tolerance = tol;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        RBKI_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int overwrite_A) {

    if (overwrite_A)
        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.U.begin(), all_data.U.end(), 0.0);
    std::fill(all_data.VT.begin(), all_data.VT.end(), 0.0);
    std::fill(all_data.Sigma.begin(), all_data.Sigma.end(), 0.0);
}

// This routine computes the residual norm error, consisting of two parts (one of which) vanishes
// in exact precision. Target_rank defines size of U, V as returned by RBKI; custom_rank <= target_rank.
template <typename T>
static T
residual_error_comp(RBKI_benchmark_data<T> &all_data, int64_t custom_rank) {
    auto m = all_data.row;
    auto n = all_data.col;

    T* U_cpy_dat = RandLAPACK::util::upsize(m * n, all_data.U_cpy);
    T* VT_cpy_dat = RandLAPACK::util::upsize(n * n, all_data.VT_cpy);

    lapack::lacpy(MatrixType::General, m, n, all_data.U.data(), m, U_cpy_dat, m);
    lapack::lacpy(MatrixType::General, n, n, all_data.VT.data(), n, VT_cpy_dat, n);

    // AV - US
    // Scale columns of U by S
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(m, all_data.Sigma[i], &U_cpy_dat[m * i], 1);


    // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.A.data(), m, all_data.VT.data(), n, -1.0, U_cpy_dat, m);


    // A'U - VS
    // Scale columns of V by S
    // Since we have VT, we will be scaling its rows
    // The data is, however, stored in a column-major format, so it is a bit weird.
    //for (int i = 0; i < n; ++i)
    //    blas::scal(custom_rank, all_data.Sigma[i], &VT_cpy_dat[i], n);
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(n, all_data.Sigma[i], &VT_cpy_dat[i], n);
    // Compute A'U(:, 1:custom_rank) - VS(1:custom_rank).
    // We will actually have to perform U' * A - Sigma * VT.

    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, custom_rank, n, m, 1.0, all_data.U.data(), m, all_data.A.data(), m, -1.0, VT_cpy_dat, n);

    T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, U_cpy_dat, m);
    T nrm2 = lapack::lange(Norm::Fro, custom_rank, n, VT_cpy_dat, n);

    return std::hypot(nrm1, nrm2);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    int64_t b_sz,
    int64_t num_matmuls,
    int64_t custom_rank,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {
    printf("\nBlock size %ld, num matmuls %ld\n", b_sz, num_matmuls);

    int i;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;
    bool time_subroutines = false;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, time_subroutines, tol);
    RBKI.num_threads_min = 4;
    RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();
    // Matrices R or S that give us the singular value spectrum returned by RBKI will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    // RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);
    // 
    // Instead of the above approach, we now pre-specify the maximum number of Krylov iters that we allow for in num_matmuls.
    RBKI.max_krylov_iters = (int) num_matmuls;
    int64_t target_rank = b_sz * num_matmuls / 2;

    // timing vars
    long dur_rbki  = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (i = 0; i < num_runs; ++i) {
        printf("Iteration %d start.\n", i);
        
        // Testing RBKI
        auto start_rbki = steady_clock::now();
        RBKI.call(m, n, all_data.A.data(), m, b_sz, all_data.U.data(), all_data.VT.data(), all_data.Sigma.data(), state_alg);
        auto stop_rbki = steady_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();

        T residual_err_custom = residual_error_comp<T>(all_data, custom_rank);
        T residual_err_target = residual_error_comp<T>(all_data, target_rank);

        // Print accuracy info
        printf("sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom);
        printf("sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(traget_rank): %.16e\n", residual_err_target);
        
        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << RBKI.max_krylov_iters <<  ",  " << target_rank << ",  " << custom_rank << ",  " << residual_err_target << ",  " << residual_err_custom <<  ",  " << dur_rbki  << ",\n";
        state_gen = state;
        state_alg = state;
        data_regen(m_info, all_data, state_gen, 0);
    }
}

int main(int argc, char *argv[]) {

    if (argc < 11) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <output_directory_path> <input_matrix_path> <num_runs> <num_rows> <num_cols> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    int num_runs        = std::stol(argv[3]);
    int64_t m_expected        = std::stol(argv[5]);
    int64_t n_expected        = std::stol(argv[6]);
    int64_t custom_rank = std::stol(argv[6]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[7]); ++i)
        b_sz.push_back(std::stoi(argv[i + 9]));
    // Save elements in string for logging purposes
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[8]); ++i)
        matmuls.push_back(std::stoi(argv[i + 9 + std::stol(argv[7])]));
    // Save elements in string for logging purposes
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    int64_t m = 0, n = 0;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[2];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected input size did not matrch actual input size. Aborting." << std::endl;
        return 1;
    }


    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, tol);
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    printf("Finished data preparation\n");
    // Declare a data file
    std::string output_filename = "_ABRIK_speed_num_info_lines_" + std::to_string(6) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".") {
        path = argv[1] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the ABRIK speed benchmark, recording the time it takes to perform ABRIK."
              "\nFile format: 6 columns, showing krylov block size, nummber of matmuls permitted, and num svals and svecs to approximate, followed by the residual error, standard lowrank error and execution time for RBKI."
              "\n Rows correspond to algorithm runs with Krylov block sizes varying as specified, and numbers of matmuls varying as specified per eah block size, with num_runs repititions of each number of matmuls."
              "\nInput type:"       + std::string(argv[2]) +
              "\nInput size:"       + std::to_string(m) + " by "             + std::to_string(n) +
              "\nAdditional parameters: Krylov block sizes "                 + b_sz_string +
                                        " matmuls: "                         + matmuls_string +
                                        " num runs per size "                + std::to_string(num_runs) +
                                        " num singular values and vectors approximated " + std::to_string(custom_rank) +
              "\n";
    file.flush();

    size_t i, j = 0;
    for (;i < b_sz.size(); ++i) {
        for (;j < matmuls.size(); ++j) {
            call_all_algs(m_info, num_runs, b_sz[i], matmuls[j], custom_rank, all_data, state_constant, path);
        }
        j = 0;
    }
}