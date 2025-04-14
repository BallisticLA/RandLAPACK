/*
Additional RBKI speed comparison benchmark - runs RBKI, RSVD and SVDS from Spectra library.
The user is required to provide a matrix file to be read, set min and max numbers of large gemms (Krylov iterations) that the algorithm is allowed to perform min and max block sizes that RBKI is to use; 
furthermore, the user is to provide a 'custom rank' parameter (number of singular vectors to approximate by RBKI). 
The benchmark outputs the basic data of a given run, as well as the RBKI runtime and singular vector residual error, 
which is computed as "sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F / sqrt(custom_rank)" (for "custom rank" singular vectors and values).
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>

// External libs includes
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Spectra/contrib/PartialSVDSolver.h>
//#include <Spectra/SparseSVD.h>
//using SpMatrix = Eigen::SparseMatrix;
//using SpVector = Eigen::VectorXd;

template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
struct RBKI_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    SpMat* A_input;
    RandLAPACK::linops::SpLinOp<T, SpMat> A_linop;
    T* U;
    T* VT; // RBKI returns V'
    T* V;  // RSVD returns V
    T* Sigma;
    T* U_RSVD;
    T* V_RSVD;
    T* Sigma_RSVD;
    T* Buffer;
    T* Sigma_cpy;
    T* U_cpy;
    T* VT_cpy;

    RBKI_benchmark_data(SpMat& input_mat_coo, int64_t m, int64_t n, T tol) :
    A_linop(m, n, input_mat_coo, Layout::ColMajor)
    {
        U          = new T[m * n]();
        VT         = new T[n * n]();
        V          = new T[n * n]();
        Sigma      = new T[m]();
        Buffer     = new T[m * n]();
        Sigma_cpy  = new T[n * n]();
        U_cpy      = new T[m * n]();
        VT_cpy     = new T[n * n]();

        row                 = m;
        col                 = n;
        tolerance           = tol;
    }

    ~RBKI_benchmark_data() {
        delete[] U;
        delete[] VT;
        delete[] V;
        delete[] Sigma;
        delete[] Buffer;
        delete[] Sigma_cpy;
        delete[] U_cpy;
        delete[] VT_cpy;
        free(A_input);
    }
};

template <typename T, typename RNG>
struct RBKI_algorithm_objects {
    RandLAPACK::RBKI<T, RNG> RBKI;

    RBKI_algorithm_objects(
        bool verbosity, 
        bool time_subroutines, 
        T tol
    ) :
        RBKI(verbosity, time_subroutines, tol)
        {}
};

template <typename T>
RandBLAS::sparse_data::coo::COOMatrix<T> from_matrix_market(std::string fn) {

    int64_t n_rows, n_cols = 0;
    std::vector<int64_t> rows{};
    std::vector<int64_t> cols{};
    std::vector<T> vals{};

    std::ifstream file_stream(fn);
    fast_matrix_market::read_matrix_market_triplet(
        file_stream, n_rows, n_cols, rows,  cols, vals
    );

    RandBLAS::sparse_data::coo::COOMatrix<T> out(n_rows, n_cols);
    reserve_coo(vals.size(),out);
    for (int i = 0; i < out.nnz; ++i) {
        out.rows[i] = rows[i];
        out.cols[i] = cols[i];
        out.vals[i] = vals[i];
    }

    return out;
}

// Re-generate and clear data
template <typename T, typename RNG, RandBLAS::sparse_data::SparseMatrix SpMat>
static void data_regen(RBKI_benchmark_data<T, SpMat> &all_data, 
                        RandBLAS::RNGState<RNG> &state) {

    auto m = all_data.row;
    auto n = all_data.col;

    std::fill(all_data.U,         &all_data.U[m * n],         0.0);
    std::fill(all_data.VT,        &all_data.VT[n * n],        0.0);
    std::fill(all_data.V,         &all_data.V[n * n],         0.0);
    std::fill(all_data.Sigma,     &all_data.Sigma[n],         0.0);

    std::fill(all_data.Buffer,    &all_data.Buffer[m * n],    0.0);
    std::fill(all_data.Sigma_cpy, &all_data.Sigma_cpy[n * n], 0.0);
    std::fill(all_data.U_cpy,     &all_data.U_cpy[m * n],     0.0);
    std::fill(all_data.VT_cpy,    &all_data.VT_cpy[n * n],    0.0);
}

// This routine computes the residual norm error, consisting of two parts (one of which) vanishes
// in exact precision. Target_rank defines size of U, V as returned by RBKI; custom_rank <= target_rank.
template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
static T
residual_error_comp(SpMat &A, RBKI_benchmark_data<T, SpMat> &all_data, int64_t custom_rank) {
    auto m = all_data.row;
    auto n = all_data.col;

    lapack::lacpy(MatrixType::General, m, n, all_data.U, m, all_data.U_cpy, m);
    lapack::lacpy(MatrixType::General, n, n, all_data.VT, n, all_data.VT_cpy, n);
    
    // AV - US
    // Scale columns of U by S
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(m, all_data.Sigma[i], &all_data.U_cpy[m * i], 1);

    // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
    //blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.A, m, all_data.VT, n, -1.0, all_data.U_cpy, m);
    all_data.A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.VT, n, -1.0, all_data.U_cpy, m);

    // A'U - VS
    // Scale columns of V by S
    // Since we have VT, we will be scaling its rows
    // The data is, however, stored in a column-major format, so it is a bit weird.
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(n, all_data.Sigma[i], &all_data.VT_cpy[i], n);
    // Compute A'U(:, 1:custom_rank) - VS(1:custom_rank).
    // We will actually have to perform U' * A - Sigma * VT.
    //blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, custom_rank, n, m, 1.0, all_data.U, m, all_data.A, m, -1.0, all_data.VT_cpy, n);
    all_data.A_linop(Side::Right, Layout::ColMajor, Op::NoTrans, Op::Trans, custom_rank, n, m, 1.0, all_data.U, m, -1.0, all_data.VT_cpy, n);

    T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, all_data.U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, custom_rank, n, all_data.VT_cpy, n);

    return std::hypot(nrm1, nrm2);
}

template <typename T, typename RNG, RandBLAS::sparse_data::SparseMatrix SpMat>
static void call_all_algs(
    int64_t num_runs,
    int64_t b_sz,
    int64_t num_matmuls,
    int64_t custom_rank,
    RBKI_algorithm_objects<T, RNG> &all_algs,
    RBKI_benchmark_data<T, SpMat> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    std::string input_path) {
    printf("\nBlock size %ld, num matmuls %ld\n", b_sz, num_matmuls);

    int i;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;

    // Additional params setup.
    // Matrices R or S that give us the singular value spectrum returned by RBKI will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    // RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);
    // 
    // Instead of the above approach, we now pre-specify the maximum number of Krylov iters that we allow for in num_matmuls.
    all_algs.RBKI.max_krylov_iters = (int) num_matmuls;
    all_algs.RBKI.num_threads_min = 4;
    all_algs.RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();
    
    // timing vars
    long dur_rbki = 0;
    long dur_svds = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    T residual_err_custom_SVDS = 0;
    T residual_err_custom_RBKI = 0;

    for (i = 0; i < num_runs; ++i) {
        printf("Iteration %d start.\n", i);
    
        // Running RBKI
        auto start_rbki = steady_clock::now();
        all_algs.RBKI.call(m, n, *all_data.A_input, m, b_sz, all_data.U, all_data.VT, all_data.Sigma, state_alg);
        auto stop_rbki = steady_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();
        printf("TOTAL TIME FOR RBKI %ld\n", dur_rbki);

        residual_err_custom_RBKI = residual_error_comp<T>(A, all_data, custom_rank);
        printf("RBKI sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom_RBKI);

        state_alg = state;
        state_gen = state;
        data_regen(all_data, state_gen);

        // There is no reason to run SVDS many times, as it always outputs the same result.
        /*
        if ((num_matmuls == 2) && ((i == 0) || (i == 1))) {
            // Running SVDS
            auto start_svds = steady_clock::now();
            Spectra::SparseSVD<Eigen::SparseMatrix<T>> svds(all_data.A_spectra, std::min(custom_rank, n-2), std::min(2 * custom_rank, n-1));
            svds.compute();
            auto stop_svds = steady_clock::now();
            dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();
            printf("TOTAL TIME FOR SVDS %ld\n", dur_svds);

            // Copy data from Spectra (Eigen) format to the nomal C++.
            Eigen::MatrixXd U_spectra = svds.matrixU().leftCols(custom_rank);
            Eigen::MatrixXd V_spectra = svds.matrixV().leftCols(custom_rank);
            Eigen::VectorXd S_spectra = svds.singularValues().head(custom_rank);

            Eigen::Map<Eigen::MatrixXd>(all_data.U, m, custom_rank)  = U_spectra;
            Eigen::Map<Eigen::MatrixXd>(all_data.V, n, custom_rank)  = V_spectra;
            Eigen::Map<Eigen::VectorXd>(all_data.Sigma, custom_rank) = S_spectra;

            RandLAPACK::util::transposition(n, n, all_data.V, n, all_data.VT, n, 0);

            residual_err_custom_SVDS = residual_error_comp<T>(all_data, custom_rank);
            printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom_SVDS);
            
            state_alg = state;
            state_gen = state;
            data_regen(all_data, state_gen);
        }
        */

        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << all_algs.RBKI.max_krylov_iters  <<  ",  " << custom_rank << ",  " 
        << residual_err_custom_RBKI << ",  " <<  ",  " << dur_rbki    << ",  " 
        << residual_err_custom_SVDS << ",  " <<  ",  " << dur_svds    << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if (argc < 9) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <output_directory_path> <input_matrix_path> <num_runs> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    int num_runs              = std::stol(argv[3]);
    int64_t custom_rank       = std::stol(argv[4]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[5]); ++i)
        b_sz.push_back(std::stoi(argv[i + 7]));
    // Save elements in string for logging purposes
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[6]); ++i)
        matmuls.push_back(std::stoi(argv[i + 7 + std::stol(argv[5])]));
    // Save elements in string for logging purposes
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    double tol                 = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                 = RandBLAS::RNGState();
    auto state_constant        = state;
    double norm_A_lowrank      = 0;

    // Read the input fast matrix market data
    auto input_mat_coo = from_matrix_market<double>(std::string(argv[2]));
    auto m = input_mat_coo.n_rows;
    auto n = input_mat_coo.n_cols;
    //linops::SpLinOp<T, SpMat> A_linop(m, n, input_mat_coo, Layout::ColMajor);

    // Allocate basic workspace.
    RBKI_benchmark_data<double, RandBLAS::sparse_data::COOMatrix<double>> all_data(input_mat_coo, m, n, tol);
    all_data.A_input = &input_mat_coo;

    // Declare RBKI object
    RBKI_algorithm_objects<double, r123::Philox4x32> all_algs(false, false, tol);

    // Copying input data into a Spectra (Eigen) matrix object
    //Eigen::Map<Eigen::MatrixXd>(all_data.A_spectra.data(), all_data.A_spectra.rows(), all_data.A_spectra.cols()) = Eigen::Map<const Eigen::MatrixXd>(all_data.A, m, n);

    printf("Finished data preparation\n");
    // Declare a data file
    std::string output_filename = "_ABRIK_speed_comparisons_sparse_num_info_lines_" + std::to_string(6) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".") {
        path = argv[1] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the ABRIK speed comparison benchmark, recording the time it takes to perform RBKI and alternative methods for low-rank SVD, specifically on sparse matrices."
              "\nFile format: 15 columns, showing krylov block size, nummber of matmuls permitted, and num svals and svecs to approximate, followed by the residual error, standard lowrank error and execution time for all algorithms (ABRIK, SVDS)"
              "\n Rows correspond to algorithm runs with Krylov block sizes varying as specified, and numbers of matmuls varying as specified per eah block size, with num_runs repititions of each number of matmuls."
              "\nInput type:"       + std::string(argv[2]) +
              "\nInput size:"       + std::to_string(m) + " by "             + std::to_string(n) +
              "\nAdditional parameters: Krylov block sizes "                 + b_sz_string +
                                        " matmuls: "                         + matmuls_string +
                                        " num runs per size "                + std::to_string(num_runs) +
                                        " num singular values and vectors approximated " + std::to_string(custom_rank) +
              "\n";
    file.flush();

    size_t i = 0, j = 0;
    for (;i < b_sz.size(); ++i) {
        for (;j < matmuls.size(); ++j) {
            call_all_algs(num_runs, b_sz[i], matmuls[j], custom_rank, all_algs, all_data, state_constant, path, std::string(argv[2]));
        }
        j = 0;
    }
}
