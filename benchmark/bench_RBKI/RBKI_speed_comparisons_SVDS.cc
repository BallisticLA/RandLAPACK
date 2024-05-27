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

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>

// External libs includes
#include <Eigen/Dense>
#include <Spectra/contrib/PartialSVDSolver.h>
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

template <typename T>
struct RBKI_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    T* A;
    T* U;
    T* VT; // RBKI returns V'
    T* Sigma;
    T* U_cpy;
    T* VT_cpy;
    Matrix A_spectra;

    RBKI_benchmark_data(int64_t m, int64_t n, T tol) :
    A_spectra(m, n)
    {
        A     = ( T * ) calloc(m * n, sizeof( T ) );
        U     = ( T * ) calloc(m * n, sizeof( T ) );
        VT    = ( T * ) calloc(n * n, sizeof( T ) );
        Sigma = ( T * ) calloc(m,     sizeof( T ) );
        row = m;
        col = n;
        tolerance = tol;
    }
};

template <typename T, typename RNG>
struct RBKI_algorithm_objects {
    RandLAPACK::PLUL<T> Stab;
    RandLAPACK::RS<T, RNG> RS;
    RandLAPACK::CholQRQ<T> Orth_RF;
    RandLAPACK::RF<T, RNG> RF;
    RandLAPACK::CholQRQ<T> Orth_QB;
    RandLAPACK::QB<T, RNG> QB;
    RandLAPACK::RSVD<T, RNG> RSVD;
    RandLAPACK::RBKI<T, RNG> RBKI;

    RBKI_algorithm_objects(
        bool verbosity, 
        bool cond_check, 
        bool orth_check, 
        bool time_subroutines, 
        int64_t p, 
        int64_t passes_per_iteration, 
        int64_t block_sz,
        T tol
    ) :
        Stab(cond_check, verbosity),
        RS(Stab, p, passes_per_iteration, verbosity, cond_check),
        Orth_RF(cond_check, verbosity),
        RF(RS, Orth_RF, verbosity, cond_check),
        Orth_QB(cond_check, verbosity),
        QB(RF, Orth_QB, verbosity, orth_check),
        RSVD(QB, verbosity, block_sz),
        RBKI(verbosity, time_subroutines, tol)
        {}
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        RBKI_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int overwrite_A) {

    auto m = all_data.row;
    auto n = all_data.col;

    if (overwrite_A) {
        RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
        Eigen::Map<Eigen::MatrixXd>(all_data.A_spectra.data(), all_data.A_spectra.rows(), all_data.A_spectra.cols()) = Eigen::Map<const Eigen::MatrixXd>(all_data.A, m, n);
    }
    std::fill(all_data.U,     &all_data.U[m * n],  0.0);
    std::fill(all_data.VT,    &all_data.VT[n * n], 0.0);
    std::fill(all_data.Sigma, &all_data.Sigma[n],  0.0);
}

// This routine computes the residual norm error, consisting of two parts (one of which) vanishes
// in exact precision. Target_rank defines size of U, V as returned by RBKI; custom_rank <= target_rank.
template <typename T>
static T
residual_error_comp(RBKI_benchmark_data<T> &all_data, int64_t custom_rank) {
    auto m = all_data.row;
    auto n = all_data.col;

    T* U_cpy_dat  = ( T * ) calloc(m * n, sizeof( T ) ); 
    T* VT_cpy_dat = ( T * ) calloc(n * n, sizeof( T ) );

    lapack::lacpy(MatrixType::General, m, n, all_data.U, m, U_cpy_dat, m);
    lapack::lacpy(MatrixType::General, n, n, all_data.VT, n, VT_cpy_dat, n);

    // AV - US
    // Scale columns of U by S
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(m, all_data.Sigma[i], &U_cpy_dat[m * i], 1);


    // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.A, m, all_data.VT, n, -1.0, U_cpy_dat, m);


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

    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, custom_rank, n, m, 1.0, all_data.U, m, all_data.A, m, -1.0, VT_cpy_dat, n);

    T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, U_cpy_dat, m);
    T nrm2 = lapack::lange(Norm::Fro, custom_rank, n, VT_cpy_dat, n);

    return std::hypot(nrm1, nrm2);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    int64_t num_matmuls,
    int64_t custom_rank,
    RBKI_algorithm_objects<T, RNG> &all_algs,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    long dur_svd) {
    printf("\nBlock size %ld, num matmuls %ld\n", b_sz, num_matmuls);

    int i;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;
    bool time_subroutines = false;

    // Additional params setup.
    all_algs.RSVD.block_sz = b_sz;
    all_algs.RBKI.num_threads_some = 4;
    all_algs.RBKI.num_threads_rest = 48;
    // Matrices R or S that give us the singular value spectrum returned by RBKI will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    // RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);
    // 
    // Instead of the above approach, we now pre-specify the maximum number of Krylov iters that we allow for in num_matmuls.
    all_algs.RBKI.max_krylov_iters = (int) num_matmuls;
    int64_t target_rank = b_sz * num_matmuls / 2;

    // timing vars
    long dur_rbki = 0;
    long dur_rsvd = 0;
    long dur_svds = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    T residual_err_custom = 0;
    T residual_err_target = 0;

    for (i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        
        // Running RBKI
        auto start_rbki = high_resolution_clock::now();
        all_algs.RBKI.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.VT, all_data.Sigma, state_alg);
        auto stop_rbki = high_resolution_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();

        residual_err_custom = residual_error_comp<T>(all_data, custom_rank);
        residual_err_target = residual_error_comp<T>(all_data, target_rank);

        // Print accuracy info
        printf("RBKI sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom);
        printf("RBKI sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(traget_rank): %.16e\n", residual_err_target);
        
        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);

        // Running RSVD
        auto start_rsvd = high_resolution_clock::now();
        all_algs.RSVD.call(m, n, all_data.A, n, tol, all_data.U, all_data.Sigma, all_data.VT, state_alg);
        auto stop_rsvd = high_resolution_clock::now();
        dur_rsvd = duration_cast<microseconds>(stop_rsvd - start_rsvd).count();

        residual_err_custom = residual_error_comp<T>(all_data, custom_rank);
        residual_err_target = residual_error_comp<T>(all_data, target_rank);

        // Print accuracy info
        printf("RSVD sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom);
        printf("RSVD sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(traget_rank): %.16e\n", residual_err_target);
        
        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);

        // Running SVDS
        auto start_svds = high_resolution_clock::now();
        Spectra::PartialSVDSolver<Eigen::MatrixXd> svds(all_data.A_spectra, custom_rank, 2 * custom_rank);
        svds.compute();
        auto stop_svds = high_resolution_clock::now();
        dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();

        // Copy data from Spectra (Eigen) format to the nomal C++.
        Matrix U_spectra = svds.matrix_U(custom_rank);
        Matrix V_spectra = svds.matrix_V(custom_rank);
        Vector S_spectra = svds.singular_values();


        printf("%ld\n", U_spectra.rows());
        printf("%ld\n", U_spectra.cols());

        Eigen::Map<Eigen::MatrixXd>(all_data.U, m, custom_rank)  = U_spectra;
        Eigen::Map<Eigen::MatrixXd>(all_data.VT, n, custom_rank) = V_spectra.transpose();
        Eigen::Map<Eigen::VectorXd>(all_data.Sigma, custom_rank) = S_spectra;

        residual_err_custom = residual_error_comp<T>(all_data, custom_rank);
        residual_err_target = residual_error_comp<T>(all_data, target_rank);

        // Print accuracy info
        printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom);
        printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(traget_rank): %.16e\n", residual_err_target);
        
        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);


        //std::ofstream file(output_filename, std::ios::app);
        //file << b_sz << ",  " << RBKI.max_krylov_iters <<  ",  " << target_rank << ",  " << custom_rank << ",  " << residual_err_target << ",  " << residual_err_custom <<  ",  " << dur_rbki  << ",  " << dur_svd << ",\n";
    }
}

int main(int argc, char *argv[]) {

    printf("Function begin\n");

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }

    int64_t m                      = 0;
    int64_t n                      = 0;
    int64_t b_sz_start             = 0;
    int64_t b_sz_stop              = 0;
    int64_t num_matmuls_start      = 2;
    int64_t num_matmuls_curr       = num_matmuls_start;
    int64_t num_matmuls_stop       = 2;
    int64_t custom_rank            = 10;
    double tol                     = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                     = RandBLAS::RNGState();
    auto state_constant            = state;
    int numruns                    = 1;
    long dur_svd = 0;
    std::vector<long> res;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[1];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double>(m_info, NULL, state);
    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    b_sz_start = 10;//std::max((int64_t) 1, n / 10);
    b_sz_stop  = 10;//std::max((int64_t) 1, n / 10);
    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, tol);
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    printf("Finished data preparation\n");

    // Declare objects for RSVD and RBKI
    int64_t p = 10;
    int64_t passes_per_iteration = 1;
    // Block size will need to be altered.
    int64_t block_sz = 0;
    RBKI_algorithm_objects<double, r123::Philox4x32> all_algs(false, false, false, false, p, passes_per_iteration, block_sz, tol);

    // Copying input data into a Spectra (Eigen) matrix object
    Eigen::Map<Eigen::MatrixXd>(all_data.A_spectra.data(), all_data.A_spectra.rows(), all_data.A_spectra.cols()) = Eigen::Map<const Eigen::MatrixXd>(all_data.A, m, n);

    // Declare a data file
    std::string output_filename = "RBKI_speed_comp_SVDS_m_"        + std::to_string(m)
                                      + "_n_"                      + std::to_string(n)
                                      + "_b_sz_start_"             + std::to_string(b_sz_start)
                                      + "_b_sz_stop_"              + std::to_string(b_sz_stop)
                                      + "_num_matmuls_start_" + std::to_string(num_matmuls_start)
                                      + "_num_matmuls_stop_"  + std::to_string(num_matmuls_stop)
                                      + ".dat"; 

    for (;b_sz_start <= b_sz_stop; b_sz_start *=2) {
        for (;num_matmuls_curr <= num_matmuls_stop; ++num_matmuls_curr) {
            call_all_algs(m_info, numruns, b_sz_start, num_matmuls_curr, custom_rank, all_algs, all_data, state_constant, output_filename, dur_svd);
        }
        num_matmuls_curr = num_matmuls_start;
    }
}
