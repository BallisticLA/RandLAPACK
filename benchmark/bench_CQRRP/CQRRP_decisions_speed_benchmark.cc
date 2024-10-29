#if defined(__APPLE__)
int main() {return 0;}
#else
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
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<T, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vars
    long dur_cqrrp_cholqr = 0;
    long dur_cqrrp_qrf    = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, B_SZ %ld\n", i, b_sz);
        
        // Testing CQRRP - QRF
        CQRRP_blocked.use_qrf = true;
        CQRRP_blocked.use_gemqrt = false;
        auto start_cqrrp_qrf = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
        auto stop_cqrrp_qrf = high_resolution_clock::now();
        dur_cqrrp_qrf = duration_cast<microseconds>(stop_cqrrp_qrf - start_cqrrp_qrf).count();
        printf("TOTAL TIME FOR CQRRP_QRF %ld\n", dur_cqrrp_qrf);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);

        // Testing CQRRP - CholQR
        CQRRP_blocked.use_qrf = false;
        CQRRP_blocked.use_gemqrt = true;
        auto start_cqrrp_cholqr = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
        auto stop_cqrrp_cholqr = high_resolution_clock::now();
        dur_cqrrp_cholqr = duration_cast<microseconds>(stop_cqrrp_cholqr - start_cqrrp_cholqr).count();
        printf("TOTAL TIME FOR CQRRP_CHOLQR %ld\n", dur_cqrrp_cholqr);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        std::ofstream file(output_filename, std::ios::app);
        file << ",  " << dur_cqrrp_cholqr << ",  " << dur_cqrrp_qrf << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }

    auto size = argv[1];

    // Declare parameters
    int64_t m          = std::stol(size);
    int64_t n          = std::stol(size);
    double d_factor    = 1.0;
    int64_t b_sz_start = 256;
    int64_t b_sz_end   = 2048;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 3;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "ICQRRP_time_raw_rows_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat";
    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant, output_filename);
    }
}
#endif
