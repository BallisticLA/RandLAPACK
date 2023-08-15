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
    int64_t sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_speed_benchmark_data(int64_t m, int64_t n, T tol, int64_t d_factor) :
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

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);
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
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vars
    long dur_cqrrp_geqp3  = 0;
    long dur_cqrrp_hqrrp  = 0;
    long dur_cqrrp_cqrrp  = 0;
    long dur_hqrrp_geqrf  = 0;
    long dur_hqrrp_cholqr = 0;
    long dur_geqrf        = 0;
    long t_cqrrp_geqp3_best  = 0;
    long t_cqrrp_hqrrp_best  = 0;
    long t_cqrrp_cqrrp_best  = 0;
    long t_hqrrp_geqrf_best  = 0;
    long t_hqrrp_cholqr_best = 0;
    long t_geqrf_best        = 0;
    
    // Making sure the states are unchanged
    auto state_gen_0 = state;
    auto state_alg_0 = state;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION\n");
        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        // Update best timing
        i == 0 ? t_geqrf_best = dur_geqrf : (dur_geqrf < t_geqrf_best) ? t_geqrf_best = dur_geqrf : NULL;

        // Making sure the states are unchanged
        auto state_gen_1 = state_gen_0;
        auto state_alg_1 = state_alg_0;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen_0, 0);

        // Testing CQRRP with GEQP3
        CQRRP_blocked.qrcp = 0;
        auto start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), d_factor, all_data.tau.data(), all_data.J.data(), state_alg_0);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp_geqp3 = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        printf("TOTAL TIME FOR CQRRP with GEQP3 %ld\n", dur_cqrrp_geqp3);
        // Update best timing
        i == 0 ? t_cqrrp_geqp3_best = dur_cqrrp_geqp3 : (dur_cqrrp_geqp3 < t_cqrrp_geqp3_best) ? t_cqrrp_geqp3_best = dur_cqrrp_geqp3 : NULL;

        auto state_gen_4 = state_gen_1;
        auto state_alg_4 = state_alg_1;

/*
        // Making sure the states are unchanged
        auto state_gen_2 = state_gen_1;
        auto state_alg_2 = state_alg_1;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen_1, 0);

        // Testing CQRRP with HQRRP
        CQRRP_blocked.qrcp = 1;
        start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), d_factor, all_data.tau.data(), all_data.J.data(), state_alg_1);
        stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp_hqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        printf("TOTAL TIME FOR CQRRP with HQRRP %ld\n", dur_cqrrp_hqrrp);
        // Update best timing
        i == 0 ? t_cqrrp_hqrrp_best = dur_cqrrp_hqrrp : (dur_cqrrp_hqrrp < t_cqrrp_hqrrp_best) ? t_cqrrp_hqrrp_best = dur_cqrrp_hqrrp : NULL;

        // Making sure the states are unchanged
        auto state_gen_3 = state_gen_2;
        auto state_alg_3 = state_alg_2;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen_2, 0);

        // Testing CQRRP with CQRRP
        CQRRP_blocked.qrcp = 2;
        start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), d_factor, all_data.tau.data(), all_data.J.data(), state_alg_2);
        stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp_cqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        printf("TOTAL TIME FOR CQRRP with CQRRP %ld\n", dur_cqrrp_cqrrp);
        // Update best timing
        i == 0 ? t_cqrrp_cqrrp_best = dur_cqrrp_cqrrp : (dur_cqrrp_cqrrp < t_cqrrp_cqrrp_best) ? t_cqrrp_cqrrp_best = dur_cqrrp_cqrrp : NULL;

        // Making sure the states are unchanged
        auto state_gen_4 = state_gen_3;
        auto state_alg_4 = state_alg_3;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen_3, 1);

        // Testing HQRRP with GEQRF
        auto start_hqrrp_geqrf = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 0, state_alg_3, (T*) nullptr);
        auto stop_hqrrp_geqrf = high_resolution_clock::now();
        dur_hqrrp_geqrf = duration_cast<microseconds>(stop_hqrrp_geqrf - start_hqrrp_geqrf).count();
        printf("TOTAL TIME FOR HQRRP WITH GEQRF %ld\n", dur_hqrrp_geqrf);
        // Update best timing
        i == 0 ? t_hqrrp_geqrf_best = dur_hqrrp_geqrf : (dur_hqrrp_geqrf < t_hqrrp_geqrf_best) ? t_hqrrp_geqrf_best = dur_hqrrp_geqrf : NULL;
*/
        // Making sure the states are unchanged
        auto state_gen_5 = state_gen_4;
        auto state_alg_5 = state_alg_4;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen_4, 1);

        // Testing HQRRP with Cholqr
        auto start_hqrrp_cholqr = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 1, state_alg_4, (T*) nullptr);
        auto stop_hqrrp_cholqr = high_resolution_clock::now();
        dur_hqrrp_cholqr = duration_cast<microseconds>(stop_hqrrp_cholqr - start_hqrrp_cholqr).count();
        printf("TOTAL TIME FOR HQRRP WITH CHOLQRQ %ld\n", dur_hqrrp_cholqr);
        // Update best timing
        i == 0 ? t_hqrrp_cholqr_best = dur_hqrrp_cholqr : (dur_hqrrp_cholqr < t_hqrrp_cholqr_best) ? t_hqrrp_cholqr_best = dur_hqrrp_cholqr : NULL;

        // Making sure the states are unchanged
        state_gen_0 = state_gen_5;
        state_alg_0 = state_alg_5;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen_5, 0);
    }

    printf("CQRRP with GEQP3 takes %ld μs\n", t_cqrrp_geqp3_best);
    printf("CQRRP with HQRRP takes %ld μs\n", t_cqrrp_hqrrp_best);
    printf("CQRRP with CQRRP takes %ld μs\n", t_cqrrp_cqrrp_best);
    printf("HQRRP with GEQRF takes %ld μs\n", t_hqrrp_geqrf_best);
    printf("HQRRP with CHOLQR takes %ld μs\n", t_hqrrp_cholqr_best);
    printf("GEQRF takes %ld μs\n\n", t_geqrf_best);
    std::vector<long> res{t_cqrrp_geqp3_best, t_cqrrp_hqrrp_best, t_cqrrp_cqrrp_best, t_hqrrp_geqrf_best, t_hqrrp_cholqr_best, t_geqrf_best};

    return res;
}

int main() {
    // Declare parameters
    int64_t m          = 1.125 * 1024;//std::pow(2, 14);
    int64_t n          = std::pow(2, 14);
    int64_t d_factor   = 1.125;
    int64_t b_sz_start = 256;
    int64_t b_sz_end   = 1024;//2048;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 10;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    // Declare a data file
    std::fstream file("QR_time_raw_rows_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

    //double norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    //printf("Matrix A norm %e\n", norm_A);

    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, b_sz_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",  " << res[2] << ",  " << res[3] << ", " << res[4] << ", " << res[5] << ",\n";
    }
}


