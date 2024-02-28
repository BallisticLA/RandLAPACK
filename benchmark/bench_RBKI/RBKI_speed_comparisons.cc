#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

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
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    std::fill(all_data.U.begin(), all_data.U.end(), 0.0);
    std::fill(all_data.VT.begin(), all_data.VT.end(), 0.0);
    std::fill(all_data.Sigma.begin(), all_data.Sigma.end(), 0.0);
}

template <typename T>
static void update_best_time(int iter, long &t_best, long &t_curr, T* S1, T* S2, int64_t target_rank)
{
    if (iter == 0 || t_curr < t_best) {
        t_best = t_curr;
        blas::copy(target_rank, S1, 1, S2, 1);
    }
}

    // This routine computes the residual norm error, consisting of two parts (one of which) vanishes
    // in exact precision. Target_rank defines size of U, V as returned by RBKI; custom_rank <= target_rank.
    template <typename T>
    static T
    residual_error_comp(RBKI_benchmark_data<T> &all_data, int64_t target_rank, int64_t custom_rank) {

        auto m = all_data.row;
        auto n = all_data.col;

        T* U_cpy_dat = RandLAPACK::util::upsize(m * target_rank, all_data.U_cpy);
        T* VT_cpy_dat = RandLAPACK::util::upsize(n * target_rank, all_data.VT_cpy);

        lapack::lacpy(MatrixType::General, m, target_rank, all_data.U.data(), m, U_cpy_dat, m);
        lapack::lacpy(MatrixType::General, n, target_rank, all_data.VT.data(), n, VT_cpy_dat, n);

        // AV - US
        // Scale columns of U by S
        for (int i = 0; i < target_rank; ++i)
            blas::scal(n, all_data.Sigma[i], &U_cpy_dat[m * i], 1);
        // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.A.data(), m, all_data.VT.data(), n, -1.0, U_cpy_dat, m);

        // A'U - VS
        // Scale columns of V by S
        // Since we have VT, we will be scaling its rows
        for (int i = 0; i < n; ++i)
            blas::scal(custom_rank, all_data.Sigma[i], &VT_cpy_dat[i], n);
        // Compute A'U(:, 1:custom_rank) - VS(1:custom_rank).
        // We will actually have to perform U' * A - Sigma * VT.
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, target_rank, custom_rank, m, 1.0, all_data.U.data(), m, all_data.A.data(), m, -1.0, VT_cpy_dat, n);

        T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, U_cpy_dat, m) / std::sqrt(custom_rank);
        T nrm2 = lapack::lange(Norm::Fro, target_rank, custom_rank, VT_cpy_dat, n) / std::sqrt(custom_rank);

        return std::sqrt( std::pow(nrm2, 2) );
    }

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    int64_t target_rank,
    int64_t custom_rank,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    long dur_svd) {
    printf("\nBlock size %ld, target rank %ld\n", b_sz, target_rank);

    int i, j;
    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;
    T norm_svd_k;
    T err_rbki;
    bool time_subroutines = false;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, time_subroutines, tol);
    // Matrices R or S that give us the singular value spectrum returned by RBKI will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);
    printf("Max Krylov iters %d\n", RBKI.max_krylov_iters);

    // timing vars
    long dur_rbki    = 0;
    long t_rbki_best = 0;

    // Making sure the states are unchanged
    auto state_gen = state;

    for (i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        
        // Testing RBKI
        auto start_rbki = high_resolution_clock::now();
        RBKI.call(m, n, all_data.A.data(), m, b_sz, all_data.U.data(), all_data.VT.data(), all_data.Sigma.data(), state);
        auto stop_rbki = high_resolution_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();
    

        T residual_err_custom = residual_error_comp<T>(all_data, target_rank, custom_rank);
        T residual_err_target = residual_error_comp<T>(all_data, target_rank, target_rank);

        // Print accuracy info
        printf("sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(custom_rank): %.16e\n", residual_err_custom);
        printf("sqrt(||AV - SU||^2_F + ||A'U - VS||^2_F) / sqrt(traget_rank): %.16e\n", residual_err_target);
        
        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << RBKI.max_krylov_iters <<  ",  " << target_rank << ",  " << custom_rank << ",  " << residual_err_target <<  ",  " << residual_err_custom <<  ",  " << dur_rbki  << ",  " << dur_svd << ",\n";
    
        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen, 0);
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
    int64_t target_rank_start      = 512;
    int64_t target_rank_curr       = target_rank_start;
    int64_t target_rank_stop       = 4096;
    int64_t custom_rank            = 10;
    double tol                     = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                     = RandBLAS::RNGState();
    auto state_constant            = state;
    int numruns                    = 3;
    long dur_svd = 0;
    std::vector<long> res;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[1];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    b_sz_start = 8;//std::max((int64_t) 1, n / 10);
    b_sz_stop  = 128;//std::max((int64_t) 1, n / 10);

    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, tol);
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    printf("Finished data preparation\n");

    // Declare a data file
    std::string output_filename = "RBKI_speed_comp_m_"             + std::to_string(m)
                                      + "_n_"                      + std::to_string(n)
                                      + "_b_sz_start_"             + std::to_string(b_sz_start)
                                      + "_b_sz_stop_"              + std::to_string(b_sz_stop)
                                      + "_num_krylov_iters_start_" + std::to_string(target_rank_start)
                                      + "_num_krylov_iters_stop_"  + std::to_string(target_rank_stop)
                                      + ".dat"; 

    for (;b_sz_start <= b_sz_stop; b_sz_start *=2) {
        for (;target_rank_curr <= target_rank_stop; target_rank_curr *=2) {
            call_all_algs<double, r123::Philox4x32>(m_info, numruns, b_sz_start, target_rank_curr, custom_rank, all_data, state_constant, output_filename, dur_svd);
        }
        target_rank_curr = target_rank_start;
    }
}