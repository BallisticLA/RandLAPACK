/*
ICQRRP runtime breakdown benchmark - assesses the time taken by each subcomponent of ICQRRP.
There are 9 things that we time:
                1. SASO generation and application time
                2. QRCP time.
                3. Preconditioning time.
                4. Time to perform Cholesky QR.
                5. Time to restore Householder vectors.
                6. Time to compute A_new, R12.
                7. Time to update factors Q, R.
                8. Time to update the sketch.
                9. Time to pivot trailing columns of R-factor.
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
    T       sampling_factor;
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
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
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
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(true, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 8;
    
    // Making sure the states are unchanged
    auto state_gen_0 = state;
    auto state_alg_0 = state;

    // Timing vars
    long dur_cqrrp    = 0;
    long t_cqrrp_best = 0;
    std::vector<long> inner_timing_best;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION\n");

        // Testing CQRRP - best setuo
        auto start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_gen_0);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        // Update best timing
        if (!i || dur_cqrrp < t_cqrrp_best) {t_cqrrp_best = dur_cqrrp; inner_timing_best = CQRRP_blocked.times;}

        // Making sure the states are unchanged
        state_gen_0 = state;
        state_alg_0 = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen_0);
    }

    return inner_timing_best;
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 14);
    int64_t n          = std::pow(2, 14);
    double  d_factor   = 1.125;
    int64_t b_sz_start = 256;
    int64_t b_sz_end   = 2048;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::fstream file("CQRRP_inner_speed_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

#if !defined(__APPLE__)
    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        res = call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",  " << res[2] << ",  " << res[3] << ",  " << res[4] << ",  " << res[5] << ",  " << res[6] << ",  " << res[7] << ",  " << res[8] << ",  " << res[9] << ",  " << res[10] << ",  " << res[11] << ",\n";
    }
#endif
}
