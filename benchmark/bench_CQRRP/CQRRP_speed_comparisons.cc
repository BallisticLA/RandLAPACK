/*
ICQRRP speed comparison benchmark - runs:
    1. ICQRRP
    2. GEQRF
    3. GEQP3 - takes too long!
    5. HQRRP + CholQR
    6. HQRRP + GEQRF
for a matrix with fixed number of rows and columns and a varying ICQRRP block size.
Records the best timing, saves that into a file.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct QR_speed_benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_speed_benchmark_data(int64_t m, int64_t n, T tol, T d_factor) :
    A(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0)
    {
        row             = m;
        col             = n;
        tolerance       = tol;
        sampling_factor = d_factor;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_speed_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int apply_itoa) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (apply_itoa) {
        std::iota(all_data.J.begin(), all_data.J.end(), 1);
    } else {
        std::fill(all_data.J.begin(), all_data.J.end(), 0);
    }
}

template <typename T, typename RNG>
static std::vector<long> call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 8;
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vars
    long dur_cqrrp        = 0;
    long dur_hqrrp_geqrf  = 0;
    long dur_hqrrp_cholqr = 0;
    long dur_geqrf        = 0;
    long t_cqrrp_best        = 0;
    long t_hqrrp_geqrf_best  = 0;
    long t_hqrrp_cholqr_best = 0;
    long t_geqrf_best        = 0;
    
    // Making sure the states are unchanged
    auto state_gen_0 = state;
    auto state_alg_0 = state;

    auto state_buf = state; 

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION\n");
        // Testing GEQRF
        auto start_geqp3 = high_resolution_clock::now();
        //lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
        auto stop_geqp3 = high_resolution_clock::now();
        auto dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
        printf("TOTAL TIME FOR GEQP3 %ld\n", dur_geqp3);

        data_regen(m_info, all_data, state_buf, 0);

        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        printf("TOTAL TIME FOR GEQRF %ld\n", dur_geqrf);
        // Update best timing
        i == 0 ? t_geqrf_best = dur_geqrf : (dur_geqrf < t_geqrf_best) ? t_geqrf_best = dur_geqrf : NULL;

        // Making sure the states are unchanged
        auto state_gen_1 = state_gen_0;
        auto state_alg_1 = state_alg_0;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen_0, 0);

        // Testing CQRRP - best setup
        auto start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg_0);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        printf("TOTAL TIME FOR CQRRP %ld\n", dur_cqrrp);
        // Update best timing
        i == 0 ? t_cqrrp_best = dur_cqrrp : (dur_cqrrp < t_cqrrp_best) ? t_cqrrp_best = dur_cqrrp : NULL;

        auto state_gen_3 = state_gen_1;
        auto state_alg_3 = state_alg_1;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen_1, 1);

        // Testing HQRRP with GEQRF
        auto start_hqrrp_geqrf = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 0, state_alg_1, (T*) nullptr);
        auto stop_hqrrp_geqrf = high_resolution_clock::now();
        dur_hqrrp_geqrf = duration_cast<microseconds>(stop_hqrrp_geqrf - start_hqrrp_geqrf).count();
        printf("TOTAL TIME FOR HQRRP WITH GEQRF %ld\n", dur_hqrrp_geqrf);
        // Update best timing
        i == 0 ? t_hqrrp_geqrf_best = dur_hqrrp_geqrf : (dur_hqrrp_geqrf < t_hqrrp_geqrf_best) ? t_hqrrp_geqrf_best = dur_hqrrp_geqrf : NULL;

        // Making sure the states are unchanged
        auto state_gen_4 = state_gen_3;
        auto state_alg_4 = state_alg_3;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen_3, 1);

        // Testing HQRRP with Cholqr
        auto start_hqrrp_cholqr = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 1, state_alg_3, (T*) nullptr);
        auto stop_hqrrp_cholqr = high_resolution_clock::now();
        dur_hqrrp_cholqr = duration_cast<microseconds>(stop_hqrrp_cholqr - start_hqrrp_cholqr).count();
        printf("TOTAL TIME FOR HQRRP WITH CHOLQRQ %ld\n", dur_hqrrp_cholqr);
        // Update best timing
        i == 0 ? t_hqrrp_cholqr_best = dur_hqrrp_cholqr : (dur_hqrrp_cholqr < t_hqrrp_cholqr_best) ? t_hqrrp_cholqr_best = dur_hqrrp_cholqr : NULL;

        // Making sure the states are unchanged
        state_gen_0 = state_gen_4;
        state_alg_0 = state_alg_4;
        state_buf = state_gen_4;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen_4, 0);
    }

    printf("CQRRP takes %ld μs\n", t_cqrrp_best);
    printf("HQRRP with GEQRF takes %ld μs\n", t_hqrrp_geqrf_best);
    printf("HQRRP with CHOLQR takes %ld μs\n", t_hqrrp_cholqr_best);
    printf("GEQRF takes %ld μs\n\n", t_geqrf_best);
    std::vector<long> res{t_cqrrp_best, t_hqrrp_geqrf_best, t_hqrrp_cholqr_best, t_geqrf_best};

    return res;
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 16);
    int64_t n          = std::pow(2, 16);
    double d_factor   = 1.25;
    int64_t b_sz_start = 256;
    int64_t b_sz_end   = 2048;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 2;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::fstream file("ICQRRP_QP3_QR_time_raw_rows_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);
#if !defined(__APPLE__)
    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        res = call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",  " << res[2] << ",  " << res[3] << ",\n";
    }
#endif
}


