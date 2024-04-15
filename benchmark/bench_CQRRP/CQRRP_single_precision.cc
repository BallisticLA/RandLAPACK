/*
This benchmarks compares single-precision ICQRRP with double-precision GETRF and GEQRF.
We anticipate that single-precision ICQRRP can be used as part of the linear system solving process.
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
template <typename T>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_speed_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<> &state, int apply_itoa) {

    RandLAPACK::gen::mat_gen<T, r123::Philox4x32>(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (apply_itoa) {
        std::iota(all_data.J.begin(), all_data.J.end(), 1);
    } else {
        std::fill(all_data.J.begin(), all_data.J.end(), 0);
    }
}

template <typename T_rest, typename T_cqrrp>
static std::vector<long> call_all_algs(
    RandLAPACK::gen::mat_gen_info<T_cqrrp> m_info_cqrrp,
    RandLAPACK::gen::mat_gen_info<T_rest> m_info_rest,
    int64_t numruns,
    int64_t b_sz,
    QR_speed_benchmark_data<T_cqrrp> &all_data_cqrrp,
    QR_speed_benchmark_data<T_rest> &all_data_rest,
    RandBLAS::RNGState<> &state) {

    auto m        = all_data_cqrrp.row;
    auto n        = all_data_cqrrp.col;
    auto tol      = all_data_cqrrp.tolerance;
    auto d_factor = all_data_cqrrp.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<float, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 48;
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vars
    long dur_cqrrp    = 0;
    long dur_geqrf    = 0;
    long dur_getrf    = 0;
    long t_cqrrp_best = 0;
    long t_geqrf_best = 0;
    long t_getrf_best = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION\n");
        // Testing GEQRF
        auto start_getrf = high_resolution_clock::now();
        lapack::getrf(m, n, all_data_rest.A.data(), m, all_data_rest.J.data());
        auto stop_getrf = high_resolution_clock::now();
        auto dur_getrf = duration_cast<microseconds>(stop_getrf - start_getrf).count();
        printf("TOTAL TIME FOR GETRF %ld\n", dur_getrf);
        // Update best timing
        i == 0 ? t_getrf_best = dur_getrf : (dur_getrf < t_getrf_best) ? t_getrf_best = dur_getrf : NULL;

        data_regen<T_rest>(m_info_rest, all_data_rest, state_gen, 0);
        state_gen = state;

        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data_rest.A.data(), m, all_data_rest.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        printf("TOTAL TIME FOR GEQRF %ld\n", dur_geqrf);
        // Update best timing
        i == 0 ? t_geqrf_best = dur_geqrf : (dur_geqrf < t_geqrf_best) ? t_geqrf_best = dur_geqrf : NULL;

        // Clear and re-generate data
        data_regen<T_rest>(m_info_rest, all_data_rest, state_gen, 0);
        state_gen = state;

        // Testing CQRRP - best setup
        auto start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data_cqrrp.A.data(), m, d_factor, all_data_cqrrp.tau.data(), all_data_cqrrp.J.data(), state_alg);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        printf("TOTAL TIME FOR CQRRP %ld\n", dur_cqrrp);
        // Update best timing
        i == 0 ? t_cqrrp_best = dur_cqrrp : (dur_cqrrp < t_cqrrp_best) ? t_cqrrp_best = dur_cqrrp : NULL;

        // Clear and re-generate data
        data_regen<T_cqrrp>(m_info_cqrrp, all_data_cqrrp, state_gen, 1);
        state_gen = state;
        state_alg = state;
    }

    std::vector<long> res{t_cqrrp_best, t_geqrf_best, t_getrf_best};

    return res;
}

int main() {
    // Declare parameters
    int64_t m           = std::pow(2, 14);
    int64_t n           = std::pow(2, 14);
    double d_factor     = 1.25;
    int64_t b_sz_start  = 256;
    int64_t b_sz_end    = 2048;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_cpy      = state;
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace - double
    QR_speed_benchmark_data<double> all_data_d(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info_d(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info_d, all_data_d.A.data(), state);

    // Allocate basic workspace - float
    QR_speed_benchmark_data<float> all_data_f(m, n, (float) tol, (float) d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<float> m_info_f(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<float, r123::Philox4x32>(m_info_f, all_data_f.A.data(), state_cpy);

    // Declare a data file
    std::fstream file("Apple_QR_time_raw_rows_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);
#if !defined(__APPLE__)
    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        res = call_all_algs<double, float>(m_info_f, m_info_d, numruns, b_sz_start, all_data_f, all_data_d, state_constant);
        file << res[0]  << ",  " << res[1]  << ",  " << res[2] << ",\n";
    }
#endif
}
