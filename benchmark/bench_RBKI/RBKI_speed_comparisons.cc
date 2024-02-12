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
    std::vector<T> V;
    std::vector<T> Sigma;
    std::vector<T> Sigma_cpy_RBKI;
    std::vector<T> Sigma_SVD;

    RBKI_benchmark_data(int64_t m, int64_t n, T tol) :
    A(m * n, 0.0),
    U(m * n, 0.0),
    V(n * n, 0.0),
    Sigma(n, 0.0),
    Sigma_cpy_RBKI(n, 0.0),
    Sigma_SVD(n, 0.0)
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
    std::fill(all_data.V.begin(), all_data.V.end(), 0.0);
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

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    int64_t target_rank,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    long dur_svd) {
    printf("\nBlock size %ld, target rank %ld\n", b_sz, target_rank);

    int i, j;
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    T norm_svd_k;
    T err_rbki;
    bool time_subroutines = false;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, time_subroutines, tol);
    // Matrices R or S that give us the singular value spectrum returned by RBKI will be of size b_sz * num_krylov_iters / 2.
    // These matrices will be full-rank.
    // Hence, target_rank = b_sz * num_krylov_iters / 2 
    RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);

    // timing vars
    long dur_rbki    = 0;
    long t_rbki_best = 0;

    // Making sure the states are unchanged
    auto state_gen = state;

    // Pre-compute the 2-norm of the Sigma vector from Direct SVD
    norm_svd_k = blas::nrm2(target_rank, all_data.Sigma_SVD.data(),  1);

    for (i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        
        // Testing RBKI
        auto start_rbki = high_resolution_clock::now();
        RBKI.call(m, n, all_data.A.data(), m, b_sz, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);
        auto stop_rbki = high_resolution_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();
    
        for(j = 0; j < target_rank; ++j)
            all_data.Sigma[j] -= all_data.Sigma_SVD[j];

        err_rbki   = blas::nrm2(target_rank, all_data.Sigma.data(), 1) / norm_svd_k;

        // Print accuracy info
        printf("||Sigma_ksvd - Sigma_rbki||_F / ||Sigma_ksvd||_F: %.16e\n", err_rbki);
        printf("RBKI is %f times faster that SVD.\n", (T) dur_svd / t_rbki_best);

        std::ofstream file(output_filename, std::ios::app);
        file << b_sz << ",  " << RBKI.max_krylov_iters <<  ",  " << target_rank << ",  " << err_rbki <<  ",  " << dur_rbki  << ",  " << dur_svd << ",\n";
    
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
    double tol                     = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                     = RandBLAS::RNGState();
    auto state_constant            = state;
    int numruns                    = 5;
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
    // Read the singular vectors from argv2
    int64_t buf1 = 1;
    int buf2     = 0;
    RandLAPACK::gen::process_input_mat(m, buf1, all_data.Sigma_SVD.data(), argv[2], buf2);

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
            call_all_algs<double, r123::Philox4x32>(m_info, numruns, b_sz_start, target_rank_curr, all_data, state_constant, output_filename, dur_svd);
        }
        target_rank_curr = target_rank_start;
    }
}