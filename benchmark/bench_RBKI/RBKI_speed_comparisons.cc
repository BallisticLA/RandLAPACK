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
    T* V;  // RSVD returns V
    T* Sigma;
    T* U_RSVD;
    T* V_RSVD;
    T* Sigma_RSVD;
    T* A_lowrank_svd;
    T* A_lowrank_svd_const;
    T* Buffer;
    T* Sigma_cpy;
    T* U_cpy;
    T* VT_cpy;
    Matrix A_spectra;

    RBKI_benchmark_data(int64_t m, int64_t n, T tol) :
    A_spectra(m, n)
    {
        A          = new T[m * n]();
        U          = new T[m * n]();
        VT         = new T[n * n]();
        V          = new T[n * n]();
        Sigma      = new T[m]();
        U_RSVD     = new T[m * n]();
        V_RSVD     = new T[n * n]();
        Sigma_RSVD = new T[n]();
        Buffer     = new T[m * n]();
        Sigma_cpy  = new T[n * n]();
        U_cpy      = new T[m * n]();
        VT_cpy     = new T[n * n]();

        A_lowrank_svd       = nullptr;
        A_lowrank_svd_const = nullptr;
        row                 = m;
        col                 = n;
        tolerance           = tol;
    }

    ~RBKI_benchmark_data() {
        delete[] A;
        delete[] U;
        delete[] VT;
        delete[] V;
        delete[] Sigma;
        delete[] U_RSVD;
        delete[] V_RSVD;
        delete[] Sigma_RSVD;
        delete[] Buffer;
        delete[] Sigma_cpy;
        delete[] U_cpy;
        delete[] VT_cpy;
        delete[] A_lowrank_svd;
        delete[] A_lowrank_svd_const;
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
        RSVD(QB, block_sz),
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
        if (all_data.A_lowrank_svd != nullptr)
            lapack::lacpy(MatrixType::General, m, n, all_data.A_lowrank_svd_const, m, all_data.A_lowrank_svd, m);
    }

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
template <typename T>
static T
residual_error_comp(RBKI_benchmark_data<T> &all_data, int64_t custom_rank) {
    auto m = all_data.row;
    auto n = all_data.col;

    lapack::lacpy(MatrixType::General, m, n, all_data.U, m, all_data.U_cpy, m);
    lapack::lacpy(MatrixType::General, n, n, all_data.VT, n, all_data.VT_cpy, n);
    
    // AV - US
    // Scale columns of U by S
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(m, all_data.Sigma[i], &all_data.U_cpy[m * i], 1);

    // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.A, m, all_data.VT, n, -1.0, all_data.U_cpy, m);

    // A'U - VS
    // Scale columns of V by S
    // Since we have VT, we will be scaling its rows
    // The data is, however, stored in a column-major format, so it is a bit weird.
    //for (int i = 0; i < n; ++i)
    //    blas::scal(custom_rank, all_data.Sigma[i], &all_data.VT_cpy[i], n);
    for (int i = 0; i < custom_rank; ++i)
        blas::scal(n, all_data.Sigma[i], &all_data.VT_cpy[i], n);
    // Compute A'U(:, 1:custom_rank) - VS(1:custom_rank).
    // We will actually have to perform U' * A - Sigma * VT.

    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, custom_rank, n, m, 1.0, all_data.U, m, all_data.A, m, -1.0, all_data.VT_cpy, n);

    T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, all_data.U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, custom_rank, n, all_data.VT_cpy, n);

    return std::hypot(nrm1, nrm2);
}

template <typename T>
static T
approx_error_comp(RBKI_benchmark_data<T> &all_data, int64_t custom_rank, T norm_A_lowrank) {
    
    auto m = all_data.row;
    auto n = all_data.col;

    RandLAPACK::util::diag(n, n, all_data.Sigma, custom_rank, all_data.Sigma_cpy);
    lapack::lacpy(MatrixType::General, m, n, all_data.U, m, all_data.U_cpy, m);
    lapack::lacpy(MatrixType::General, n, n, all_data.VT, n, all_data.VT_cpy, n);

    // U * S = Buffer
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, custom_rank, 1.0, all_data.U_cpy, m, all_data.Sigma_cpy, n, 0.0, all_data.Buffer, m);
    // Buffer * VT1 - A_cpy ~= 0?
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, custom_rank, 1.0, all_data.Buffer, m, all_data.VT_cpy, n, -1.0, all_data.A_lowrank_svd, m);

    T nrm = lapack::lange(Norm::Fro, m, n, all_data.A_lowrank_svd, m);
    printf("||A_hat_cursom_rank - A_svd_custom_rank||_F / ||A_svd_custom_rank||_F: %e\n", nrm / norm_A_lowrank);

    return nrm / norm_A_lowrank;
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
    T norm_A_lowrank) {
    printf("\nBlock size %ld, num matmuls %ld\n", b_sz, num_matmuls);

    int i;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;

    // Additional params setup.
    all_algs.RSVD.block_sz = b_sz;
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
    long dur_rsvd = 0;
    long dur_svds = 0;
    long dur_svd  = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    T residual_err_custom_SVD  = 0;
    T residual_err_custom_RBKI = 0;
    T residual_err_custom_RSVD = 0;
    T residual_err_custom_SVDS = 0;

    T lowrank_err_SVD  = 0;
    T lowrank_err_RBKI = 0;
    T lowrank_err_RSVD = 0;
    T lowrank_err_SVDS = 0;

    for (i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        
        // There is no reason to run SVD many times, as it always outputs the same result.
        if ((b_sz == 16) && (num_matmuls == 2) && ((i == 0) || (i == 1))) {
            // Running SVD
            auto start_svd = steady_clock::now();
            lapack::gesdd(Job::SomeVec, m, n, all_data.A, m, all_data.Sigma, all_data.U, m, all_data.VT, n);
            auto stop_svd = steady_clock::now();
            dur_svd = duration_cast<microseconds>(stop_svd - start_svd).count();
            printf("TOTAL TIME FOR SVD %ld\n", dur_svd);

            // Standard SVD destorys matrix A, need to re-read it before running accuracy tests.
            state_gen = state;
            RandLAPACK::gen::mat_gen(m_info, all_data.A, state_gen);

            residual_err_custom_SVD = residual_error_comp<T>(all_data, custom_rank);
            printf("\nSVD sqrt(||AV - US||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom_SVD);

            if (all_data.A_lowrank_svd != nullptr)
                lowrank_err_SVD = approx_error_comp(all_data, custom_rank, norm_A_lowrank);

            state_alg = state;
            state_gen = state;
            data_regen(m_info, all_data, state_gen, 1);
        }
        
        // Running RBKI
        auto start_rbki = steady_clock::now();
        all_algs.RBKI.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.VT, all_data.Sigma, state_alg);
        auto stop_rbki = steady_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();
        printf("TOTAL TIME FOR RBKI %ld\n", dur_rbki);

        residual_err_custom_RBKI = residual_error_comp<T>(all_data, custom_rank);
        printf("RBKI sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom_RBKI);

        if (all_data.A_lowrank_svd != nullptr)
            lowrank_err_RBKI = approx_error_comp(all_data, custom_rank, norm_A_lowrank);

        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);
        
        // Running RSVD
        auto start_rsvd = steady_clock::now();
        int64_t threshold_RSVD = (int64_t ) (b_sz * num_matmuls / 2);
        all_algs.RSVD.call(m, n, all_data.A, threshold_RSVD, tol, all_data.U_RSVD, all_data.Sigma_RSVD, all_data.V_RSVD, state_alg);
        auto stop_rsvd = steady_clock::now();
        dur_rsvd = duration_cast<microseconds>(stop_rsvd - start_rsvd).count();
        printf("TOTAL TIME FOR RSVD %ld\n", dur_rsvd);

        lapack::lacpy(MatrixType::General, m, threshold_RSVD, all_data.U_RSVD, m, all_data.U, m);
        lapack::lacpy(MatrixType::General, n, threshold_RSVD, all_data.V_RSVD, n, all_data.V, n);
        blas::copy(threshold_RSVD, all_data.Sigma_RSVD, 1, all_data.Sigma, 1);

        RandLAPACK::util::transposition(n, n, all_data.V, n, all_data.VT, n, 0);
        
        residual_err_custom_RSVD = residual_error_comp<T>(all_data, custom_rank);
        printf("RSVD sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom_RSVD);

        if (all_data.A_lowrank_svd != nullptr)
            lowrank_err_RSVD = approx_error_comp(all_data, custom_rank, norm_A_lowrank);
        
        state_alg = state;
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);

        // There is no reason to run SVDS many times, as it always outputs the same result.
        if ((num_matmuls == 2) && ((i == 0) || (i == 1))) {
            // Running SVDS
            auto start_svds = steady_clock::now();
            Spectra::PartialSVDSolver<Eigen::MatrixXd> svds(all_data.A_spectra, std::min(custom_rank, n-2), std::min(2 * custom_rank, n-1));
            svds.compute();
            auto stop_svds = steady_clock::now();
            dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();
            printf("TOTAL TIME FOR SVDS %ld\n", dur_svds);

            // Copy data from Spectra (Eigen) format to the nomal C++.
            Matrix U_spectra = svds.matrix_U(custom_rank);
            Matrix V_spectra = svds.matrix_V(custom_rank);
            Vector S_spectra = svds.singular_values();

            Eigen::Map<Eigen::MatrixXd>(all_data.U, m, custom_rank)  = U_spectra;
            Eigen::Map<Eigen::MatrixXd>(all_data.V, n, custom_rank)  = V_spectra;
            Eigen::Map<Eigen::VectorXd>(all_data.Sigma, custom_rank) = S_spectra;

            RandLAPACK::util::transposition(n, n, all_data.V, n, all_data.VT, n, 0);

            residual_err_custom_SVDS = residual_error_comp<T>(all_data, custom_rank);
            printf("SVDS sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom_SVDS);

            if (all_data.A_lowrank_svd != nullptr)
                lowrank_err_SVDS = approx_error_comp(all_data, custom_rank, norm_A_lowrank);
            
            state_alg = state;
            state_gen = state;
            data_regen(m_info, all_data, state_gen, 1);
        }

        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << all_algs.RBKI.max_krylov_iters  <<  ",  " << custom_rank << ",  " 
        << residual_err_custom_RBKI << ",  " << lowrank_err_RBKI <<  ",  " << dur_rbki    << ",  " 
        << residual_err_custom_RSVD << ",  " << lowrank_err_RSVD <<  ",  " << dur_rsvd    << ",  "
        << residual_err_custom_SVDS << ",  " << lowrank_err_SVDS <<  ",  " << dur_svds    << ",  " 
        << residual_err_custom_SVD  << ",  " << lowrank_err_SVD  <<  ",  " << dur_svd     << ",\n";
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
    int64_t num_matmuls_stop       = 30;
    int64_t custom_rank            = 10;
    double tol                     = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                     = RandBLAS::RNGState();
    auto state_constant            = state;
    int numruns                    = 2;
    double norm_A_lowrank          = 0;
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
    b_sz_start = 16;//std::max((int64_t) 1, n / 10);
    b_sz_stop  = 128;//std::max((int64_t) 1, n / 10);
    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, tol);
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Declare objects for RSVD and RBKI
    int64_t p = 5;
    int64_t passes_per_iteration = 1;
    // Block size will need to be altered.
    int64_t block_sz = 0;
    RBKI_algorithm_objects<double, r123::Philox4x32> all_algs(false, false, false, false, p, passes_per_iteration, block_sz, tol);

    // Copying input data into a Spectra (Eigen) matrix object
    Eigen::Map<Eigen::MatrixXd>(all_data.A_spectra.data(), all_data.A_spectra.rows(), all_data.A_spectra.cols()) = Eigen::Map<const Eigen::MatrixXd>(all_data.A, m, n);

    // Optional pass of lowrank SVD matrix into the benchmark
    if (argc > 2) {
        printf("Name passed\n");
        RandLAPACK::gen::mat_gen_info<double> m_info_A_svd(m, n, RandLAPACK::gen::custom_input);
        m_info_A_svd.filename            = argv[2];
        m_info_A_svd.workspace_query_mod = 0;
        all_data.A_lowrank_svd       = new double[m * n]();
        all_data.A_lowrank_svd_const = new double[m * n]();
        RandLAPACK::gen::mat_gen<double>(m_info_A_svd, all_data.A_lowrank_svd_const, state);
        lapack::lacpy(MatrixType::General, m, n, all_data.A_lowrank_svd_const, m, all_data.A_lowrank_svd, m);
    
        // Pre-compute norm(A lowrank) for future benchmarking
        norm_A_lowrank = lapack::lange(Norm::Fro, m, n, all_data.A_lowrank_svd, m);
    }

    printf("Finished data preparation\n");
    // Declare a data file
    std::string output_filename = RandLAPACK::util::getCurrentDateTime<double>() + "RBKI_speed_comparisons" 
                                                                + "_num_info_lines_" + std::to_string(6) +
                                                                ".txt";
    std::ofstream file(output_filename, std::ios::out | std::ios::trunc);
    std::string input_name_and_path(argv[1]);

    // Writing important data into file
    file << "Description: Results from the RBKI speed comparison benchmark, recording the time it takes to perform RBKI and alternative methods for low-rank SVD."
              "\nFile format: 15 columns, showing krylov block size, nummber of matmuls permitted, and num svals and svecs to approximate, followed by the residual error, standard lowrank error and execution time for all algorithms (RBKI, RSVD, SVDS, SVD)"
              "               rows correspond to algorithm runs with Krylov block sizes varying in powers of 2, and numbers of matmuls varying in powers of two per eah block size, with numruns repititions of each number of matmuls."
              "\nInput type:"       + input_name_and_path +
              "\nInput size:"       + std::to_string(m) + " by "             + std::to_string(n) +
              "\nAdditional parameters: Krylov block size start: "           + std::to_string(b_sz_start) + 
                                        " Krylov block size end: "           + std::to_string(b_sz_stop) + 
                                        " num matmuls start: "               + std::to_string(num_matmuls_start) + 
                                        " num matmuls end: "                 + std::to_string(num_matmuls_stop) + 
                                        " num runs per size "                + std::to_string(numruns) +
                                        " num svals and svecs approximated " + std::to_string(custom_rank) +
              "\n";
    file.flush();

    for (;b_sz_start <= b_sz_stop; b_sz_start *=2) {
        for (;num_matmuls_curr <= num_matmuls_stop; num_matmuls_curr*=2) {
            call_all_algs(m_info, numruns, b_sz_start, num_matmuls_curr, custom_rank, all_algs, all_data, state_constant, output_filename, norm_A_lowrank);
        }
        num_matmuls_curr = num_matmuls_start;
    }
}
