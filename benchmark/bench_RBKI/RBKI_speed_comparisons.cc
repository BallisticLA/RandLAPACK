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
    std::vector<T> Sigma_cpy_1;
    std::vector<T> Sigma_cpy_2;
    std::vector<T> Sigma_cpy_3;

    RBKI_benchmark_data(int64_t m, int64_t n, int64_t k, T tol) :
    A(m * n, 0.0),
    U(m * n, 0.0),
    V(n * n, 0.0),
    Sigma(n, 0.0),
    Sigma_cpy_1(n, 0.0),
    Sigma_cpy_2(n, 0.0),
    Sigma_cpy_3(n, 0.0)
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
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    std::fill(all_data.U.begin(), all_data.U.end(), 0.0);
    std::fill(all_data.V.begin(), all_data.V.end(), 0.0);
    std::fill(all_data.Sigma.begin(), all_data.Sigma.end(), 0.0);
}

template <typename T>
static void update_best_time(int iter, long &t_best, long &t_curr, T* S1, T* S2, int64_t k, long* break_in, long* break_out, int timing)
{
    // Can also do this is one line 
    // i == 0 ? (void) (t_rbki_best = dur_rbki, accuracy_check ? blas::copy(n, all_data.Sigma.data(), 1, all_data.Sigma_cpy_1.data(), 1): (void) NULL) : (dur_rbki < t_rbki_best) ? ((void) (t_rbki_best = dur_rbki), accuracy_check ? blas::copy(n, all_data.Sigma.data(), 1, all_data.Sigma_cpy_1.data(), 1): (void) NULL) : (void) NULL;
    if (iter == 0 || t_curr < t_best) {
        t_best = t_curr;
        blas::copy(k, S1, 1, S2, 1);
    }
    if (timing)
        blas::copy(8, break_out, 1, break_in, 1);
}
/*
template <typename T>
static void svd_error(T* U1, T* S1, T* VT1, T* U2, T* S2, T* VT2)
{
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);
  
}
*/

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t k,
    int64_t num_krylov_iters,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    int i, j;
    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    T norm_svd_k;
    T norm_svd_lanc;
    T err_rbki;
    T err_lan;
    int64_t k_lanc = std::min((int64_t) (num_krylov_iters / (T) 2), k);
    bool time_subroutines = true;

    // Set the threshold for Lanchosz 
    // Setting up Lanchosz - RBKI with k = 1.
    RandLAPACK::RBKI<double, r123::Philox4x32> Lanchosz(false, false, tol);
    Lanchosz.max_krylov_iters = num_krylov_iters;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, time_subroutines, tol);
    RBKI.max_krylov_iters = num_krylov_iters;

    // timing vars
    long dur_rbki        = 0;
    long dur_other       = 0;
    long dur_lanchosz    = 0;
    long t_rbki_best     = 0;
    long t_other_best    = 0;
    long t_lanchosz_best = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    //auto state_alg = state;

    // Timing breakdown vectors;
    std::vector<long> Lanc_timing_breakdown (8, 0.0);
    std::vector<long> RBKI_timing_breakdown (8, 0.0);

    for (i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);

        // Testing Lanchosz
        auto start_lanchosz = high_resolution_clock::now();
        //Lanchosz.call(m, n, all_data.A.data(), m, 1, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);
        auto stop_lanchosz = high_resolution_clock::now();
        dur_lanchosz = duration_cast<microseconds>(stop_lanchosz - start_lanchosz).count();
        
        // Update best timing and save the singular values.
        update_best_time<T>(i, t_lanchosz_best, dur_lanchosz, all_data.Sigma.data(), all_data.Sigma_cpy_3.data(), k_lanc, Lanc_timing_breakdown.data(), Lanchosz.times.data(), false);

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);
        
        // Testing RBKI
        auto start_rbki = high_resolution_clock::now();
        RBKI.call(m, n, all_data.A.data(), m, k, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);
        auto stop_rbki = high_resolution_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();

        // Update best timing and save the singular values.
        update_best_time<T>(i, t_rbki_best, dur_rbki, all_data.Sigma.data(), all_data.Sigma_cpy_1.data(), k, RBKI_timing_breakdown.data(), RBKI.times.data(), time_subroutines);

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);

        // Testing Other - SVD
        auto start_other = high_resolution_clock::now();
        lapack::gesdd(Job::NoVec, m, n, all_data.A.data(), m, all_data.Sigma.data(), all_data.U.data(), m, all_data.V.data(), n);
        auto stop_other = high_resolution_clock::now();
        dur_other = duration_cast<microseconds>(stop_other - start_other).count();

        blas::copy(n, all_data.Sigma.data(), 1, all_data.Sigma_cpy_2.data(), 1);

        // Update best timing and save the singular values.
        update_best_time<T>(i, t_other_best, dur_other, all_data.Sigma.data(), all_data.Sigma_cpy_2.data(), k, NULL, NULL, 0);

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);
    }
    
    for(j = 0; j < k; ++j)
        all_data.Sigma_cpy_1[j] -= all_data.Sigma_cpy_2[j];

    for(j = 0; j < k_lanc; ++j) 
        all_data.Sigma_cpy_3[j] -= all_data.Sigma_cpy_2[j];

    norm_svd_k    = blas::nrm2(k,      all_data.Sigma_cpy_2.data(), 1);
    norm_svd_lanc = blas::nrm2(k_lanc, all_data.Sigma_cpy_2.data(), 1);
    
    err_rbki = blas::nrm2(k,      all_data.Sigma_cpy_1.data(), 1) / norm_svd_k;
    err_lan  = blas::nrm2(k_lanc, all_data.Sigma_cpy_3.data(), 1) / norm_svd_lanc;

    if (time_subroutines) {
        printf("\n\n/------------RBKI TIMING RESULTS BEGIN------------/\n");
        printf("Basic info: b_sz=%ld krylov_iters=%ld\n",      k, num_krylov_iters);

        printf("Allocate and free time:          %25ld μs,\n", RBKI_timing_breakdown[0]);
        printf("Time to acquire the SVD factors: %25ld μs,\n", RBKI_timing_breakdown[1]);
        printf("UNGQR time:                      %25ld μs,\n", RBKI_timing_breakdown[2]);
        printf("Reorthogonalization time:        %25ld μs,\n", RBKI_timing_breakdown[3]);
        printf("QR time:                         %25ld μs,\n", RBKI_timing_breakdown[4]);
        printf("GEMM A time:                     %25ld μs,\n", RBKI_timing_breakdown[5]);

        printf("\nAllocation takes %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[0] / (T) RBKI_timing_breakdown[7]));
        printf("Factors takes    %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[1] / (T) RBKI_timing_breakdown[7]));
        printf("Ungqr takes      %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[2] / (T) RBKI_timing_breakdown[7]));
        printf("Reorth takes     %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[3] / (T) RBKI_timing_breakdown[7]));
        printf("QR takes         %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[4] / (T) RBKI_timing_breakdown[7]));
        printf("GEMM A takes     %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[5] / (T) RBKI_timing_breakdown[7]));
        printf("Rest takes       %22.2f%% of runtime.\n", 100 * ((T) RBKI_timing_breakdown[6] / (T) RBKI_timing_breakdown[7]));
        printf("/-------------RBKI TIMING RESULTS END-------------/\n\n");
    }

    // Print accuracy info
    printf("||Sigma_ksvd - Sigma_rbki||_F / ||Sigma_ksvd||_F: %.16e\n", err_rbki);
    printf("||Sigma_ksvd - Sigma_lanc||_F / ||Sigma_lanc||_F: %.16e\n", err_lan);

    printf("RBKI     is %f times faster that SVD.\n", (T) t_other_best / t_rbki_best);
    printf("Lanchosz is %f times faster that SVD.\n", (T) t_other_best / t_lanchosz_best);

    std::ofstream file(output_filename, std::ios::app);
    file << k << ",  " << num_krylov_iters << ",  " << err_rbki << ",  " << err_lan <<  ",  " << t_rbki_best  << ",  " << t_other_best << ",  " << t_lanchosz_best  << ",\n";
}

int main(int argc, char *argv[]) {

    if(argc <= 1)
        // No input
        return 0;

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
    k_stop  = 128;//std::max((int64_t) 1, n / 10);

    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "RBKI_speed_comp_m_"                         + std::to_string(m)
                                      + "_n_"                      + std::to_string(n)
                                      + "_k_start_"                + std::to_string(k_start)
                                      + "_k_stop_"                 + std::to_string(k_stop)
                                      + "_num_krylov_iters_start_" + std::to_string(num_krylov_iters_start)
                                      + "_num_krylov_iters_stop_"  + std::to_string(num_krylov_iters_stop)
                                      + ".dat"; 

    for (;k_start <= k_stop; k_start *=2) {
        for (;num_krylov_iters_curr <= num_krylov_iters_stop; num_krylov_iters_curr *=2) {
            call_all_algs<double, r123::Philox4x32>(m_info, numruns, k_start, num_krylov_iters_curr, all_data, state_constant, output_filename);
        }
        num_krylov_iters_curr = num_krylov_iters_start;
    }
}

/*
int main() {
    // Declare parameters
    int64_t m           = std::pow(10, 3);
    int64_t n           = std::pow(10, 3);
    int64_t k_start     = 100;
    int64_t k_stop      = 100;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);

    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    // Declare a data file
    std::fstream file("RBKI_speed_comp_m_"          + std::to_string(m)
                                      + "_n_"       + std::to_string(n)
                                      + "_k_start_" + std::to_string(k_start)
                                      + "_k_stop_"  + std::to_string(k_stop)
                                      + ".dat", std::fstream::app);

    for (;k_start <= k_stop; k_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, k_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",\n";
    }
}
*/