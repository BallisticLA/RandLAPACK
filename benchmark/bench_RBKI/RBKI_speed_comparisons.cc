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
    int64_t rank; // has to be modifiable
    T tolerance;
    std::vector<T> A;
    std::vector<T> U;
    std::vector<T> V;
    std::vector<T> Sigma;
    std::vector<T> Sigma_cpy_RBKI;
    std::vector<T> Sigma_cpy_SVD;
    std::vector<T> Sigma_cpy_Other;

    RBKI_benchmark_data(int64_t m, int64_t n, int64_t k, T tol) :
    A(m * n, 0.0),
    U(m * n, 0.0),
    V(n * n, 0.0),
    Sigma(n, 0.0),
    Sigma_cpy_RBKI(n, 0.0),
    Sigma_cpy_SVD(n, 0.0),
    Sigma_cpy_Other(n, 0.0)
    {
        row = m;
        col = n;
        rank = k;
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
static void update_best_time(int iter, long &t_best, long &t_curr, T* S1, T* S2, int64_t k, long* break_in, long* break_out, int timing)
{
    if (iter == 0 || t_curr < t_best) {
        t_best = t_curr;
        blas::copy(k, S1, 1, S2, 1);
    }
    if (timing)
        blas::copy(13, break_out, 1, break_in, 1);
}

template <typename T, typename RNG>
static long run_svd(    
    RandLAPACK::gen::mat_gen_info<T> m_info,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state)
{

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;

    // Testing Other - SVD
    auto start_svd = high_resolution_clock::now();
    lapack::gesdd(Job::NoVec, m, n, all_data.A.data(), m, all_data.Sigma.data(), all_data.U.data(), m, all_data.V.data(), n);
    auto stop_svd = high_resolution_clock::now();
    long dur_svd = duration_cast<microseconds>(stop_svd - start_svd).count();

    blas::copy(n, all_data.Sigma.data(), 1, all_data.Sigma_cpy_SVD.data(), 1);

    auto state_gen = state;
    data_regen<T, RNG>(m_info, all_data, state_gen, 1);
  
    return dur_svd;
}


template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t k,
    int64_t num_krylov_iters,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename, 
    long dur_svd) {

    int i, j;
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    T norm_svd_k;
    T norm_svd_lanc;
    T err_rbki;
    T err_lan;
    int64_t k_lanc = std::min((int64_t) (num_krylov_iters / (T) 2), k);
    bool time_subroutines = false;

    // Set the threshold for Lanchosz 
    // Setting up Lanchosz - RBKI with k = 1.
    RandLAPACK::RBKI<double, r123::Philox4x32> Lanchosz(false, false, tol);
    Lanchosz.max_krylov_iters = num_krylov_iters;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, time_subroutines, tol);
    RBKI.max_krylov_iters = num_krylov_iters;

    // timing vars
    long dur_rbki        = 0;
    long dur_lanchosz    = 0;
    long t_rbki_best     = 0;
    long t_lanchosz_best = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    //auto state_alg = state;

    // Timing breakdown vectors;
    std::vector<long> Lanc_timing_breakdown (13, 0.0);
    std::vector<long> RBKI_timing_breakdown (13, 0.0);

    for (i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);

        // Testing Lanchosz
        auto start_lanchosz = high_resolution_clock::now();
        //Lanchosz.call(m, n, all_data.A.data(), m, 1, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);
        auto stop_lanchosz = high_resolution_clock::now();
        dur_lanchosz = duration_cast<microseconds>(stop_lanchosz - start_lanchosz).count();

        // Update best timing and save the singular values.
        //update_best_time<T>(i, t_lanchosz_best, dur_lanchosz, all_data.Sigma.data(), all_data.Sigma_cpy_Other.data(), k_lanc, Lanc_timing_breakdown.data(), Lanchosz.times.data(), false);

        //state_gen = state;
        //data_regen<T, RNG>(m_info, all_data, state_gen, 0);
        
        // Testing RBKI
        auto start_rbki = high_resolution_clock::now();
        RBKI.call(m, n, all_data.A.data(), m, k, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);

        auto stop_rbki = high_resolution_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();

        // Update best timing and save the singular values.
        update_best_time<T>(i, t_rbki_best, dur_rbki, all_data.Sigma.data(), all_data.Sigma_cpy_RBKI.data(), k, RBKI_timing_breakdown.data(), RBKI.times.data(), time_subroutines);

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen, 0);
    }
    
    for(j = 0; j < k; ++j)
        all_data.Sigma_cpy_RBKI[j] -= all_data.Sigma_cpy_SVD[j];

    for(j = 0; j < k_lanc; ++j) 
        all_data.Sigma_cpy_Other[j] -= all_data.Sigma_cpy_SVD[j];

    norm_svd_k    = blas::nrm2(k,      all_data.Sigma_cpy_SVD.data(), 1);
    norm_svd_lanc = blas::nrm2(k_lanc, all_data.Sigma_cpy_SVD.data(), 1);
    
    err_rbki = blas::nrm2(k,      all_data.Sigma_cpy_RBKI.data(), 1) / norm_svd_k;
    err_lan  = blas::nrm2(k_lanc, all_data.Sigma_cpy_Other.data(), 1) / norm_svd_lanc;

    if (time_subroutines) {
        printf("\n\n/------------RBKI TIMING RESULTS BEGIN------------/\n");
        printf("Basic info: b_sz=%ld krylov_iters=%ld\n",      k, num_krylov_iters);

        printf("Allocate and free time:          %25ld μs,\n", RBKI_timing_breakdown[0]);
        printf("Time to acquire the SVD factors: %25ld μs,\n", RBKI_timing_breakdown[1]);
        printf("UNGQR time:                      %25ld μs,\n", RBKI_timing_breakdown[2]);
        printf("Reorthogonalization time:        %25ld μs,\n", RBKI_timing_breakdown[3]);
        printf("QR time:                         %25ld μs,\n", RBKI_timing_breakdown[4]);
        printf("GEMM A time:                     %25ld μs,\n", RBKI_timing_breakdown[5]);
        printf("Sketching time:                  %25ld μs,\n", RBKI_timing_breakdown[7]);
        printf("R_ii cpy time:                   %25ld μs,\n", RBKI_timing_breakdown[8]);
        printf("S_ii cpy time:                   %25ld μs,\n", RBKI_timing_breakdown[9]);
        printf("Norm time:                       %25ld μs,\n", RBKI_timing_breakdown[10]);

        printf("\nAllocation takes %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[0] / (T) RBKI_timing_breakdown[12]));
        printf("Factors takes    %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[1] / (T) RBKI_timing_breakdown[12]));
        printf("Ungqr takes      %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[2] / (T) RBKI_timing_breakdown[12]));
        printf("Reorth takes     %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[3] / (T) RBKI_timing_breakdown[12]));
        printf("QR takes         %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[4] / (T) RBKI_timing_breakdown[12]));
        printf("GEMM A takes     %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[5] / (T) RBKI_timing_breakdown[12]));
        printf("Sketching takes  %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[7] / (T) RBKI_timing_breakdown[12]));
        printf("R_ii cpy takes   %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[8] / (T) RBKI_timing_breakdown[12]));
        printf("S_ii cpy takes   %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[9] / (T) RBKI_timing_breakdown[12]));
        printf("Norm R takes     %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[10] / (T) RBKI_timing_breakdown[12]));
        printf("Rest takes       %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[11] / (T) RBKI_timing_breakdown[12]));

        printf("\nMain loop takes  %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[6] / (T) RBKI_timing_breakdown[12]));
        printf("/-------------RBKI TIMING RESULTS END-------------/\n\n");
    }

    // Print accuracy info
    printf("||Sigma_ksvd - Sigma_rbki||_F / ||Sigma_ksvd||_F: %.16e\n", err_rbki);
    printf("||Sigma_ksvd - Sigma_lanc||_F / ||Sigma_lanc||_F: %.16e\n", err_lan);

    printf("RBKI     is %f times faster that SVD.\n", (T) dur_svd / t_rbki_best);
    printf("Lanchosz is %f times faster that SVD.\n", (T) dur_svd / t_lanchosz_best);

    std::ofstream file(output_filename, std::ios::app);
    file << k << ",  " << num_krylov_iters << ",  " << err_rbki << ",  " << err_lan <<  ",  " << t_rbki_best  << ",  " << dur_svd << ",  " << t_lanchosz_best  << ",\n";
}

int main(int argc, char *argv[]) {

    printf("Function begin\n");

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }

    int64_t m                      = 0;
    int64_t n                      = 0;
    int64_t k_start                = 0;
    int64_t k_stop                 = 0;
    int64_t num_krylov_iters_start = 2;
    int64_t num_krylov_iters_curr  = num_krylov_iters_start;
    int64_t num_krylov_iters_stop  = 64;
    double tol                     = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                     = RandBLAS::RNGState();
    auto state_constant            = state;
    int numruns                    = 10;
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
    k_start = 2;//std::max((int64_t) 1, n / 10);
    k_stop  = 256;//std::max((int64_t) 1, n / 10);

    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    printf("Finished data preparation\n");

    // Declare a data file
    std::string output_filename = "RBKI_speed_comp_m_"                         + std::to_string(m)
                                      + "_n_"                      + std::to_string(n)
                                      + "_k_start_"                + std::to_string(k_start)
                                      + "_k_stop_"                 + std::to_string(k_stop)
                                      + "_num_krylov_iters_start_" + std::to_string(num_krylov_iters_start)
                                      + "_num_krylov_iters_stop_"  + std::to_string(num_krylov_iters_stop)
                                      + ".dat"; 

    // SVD run takes very long & is only needed once for all sizes
    long dur_svd = run_svd<double, r123::Philox4x32>(m_info, all_data, state);

    for (;k_start <= k_stop; k_start *=2) {
        for (;num_krylov_iters_curr <= num_krylov_iters_stop; num_krylov_iters_curr *=2) {
            call_all_algs<double, r123::Philox4x32>(m_info, numruns, k_start, num_krylov_iters_curr, all_data, state_constant, output_filename, dur_svd);
        }
        num_krylov_iters_curr = num_krylov_iters_start;
    }
}