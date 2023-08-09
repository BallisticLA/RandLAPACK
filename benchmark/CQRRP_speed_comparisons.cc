#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>




template <typename T>
struct QR_speed_benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    int64_t sampling_factor;
    std::vector<T> A1;
    std::vector<T> A2;
    std::vector<T> A3;
    std::vector<T> tau1;
    std::vector<T> tau2;
    std::vector<T> tau3;
    std::vector<int64_t> J1;
    std::vector<int64_t> J2;
    std::vector<int64_t> J3;

    QR_speed_benchmark_data(int64_t m, int64_t n, T tol, int64_t d_factor) :
    A1(m * n, 0.0),
    A2(m * n, 0.0),
    A3(m * n, 0.0),
    tau1(n, 0.0),
    tau2(n, 0.0),
    tau3(n, 0.0),
    J1(n, 0),
    J2(n, 0),
    J3(n, 0) 
    {
        row             = m;
        col             = n;
        tolerance       = tol;
        sampling_factor = d_factor;
    }
};

template <typename T, typename RNG>
static void copy_computational_helper(CQRRPTestData<T> &all_data) {
    auto m = all_data.row;
    auto n = all_data.col;

    lapack::lacpy(MatrixType::General, m, n, all_data.A1.data(), m, all_data.A2.data(), m);
    lapack::lacpy(MatrixType::General, m, n, all_data.A1.data(), m, all_data.A3.data(), m);
}

template <typename T, typename RNG>
static std::vector<long> call_all_algs(
    int64_t b_sz,
    CQRRPTestData<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Testing CQRRP
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    auto start_cqrrp = high_resolution_clock::now();
    CQRRP.call(m, n, all_data.A1, d_factor, all_data.tau1, all_data.J1, state);
    auto stop_cqrrp = high_resolution_clock::now();
    long dur_cqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();

    // Testing HQRRP
    // HQRRP oversampling factor is hardcoded per Riley's suggestion.
    T d_factor_hqrrp = 0.125;
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;
    auto start_hqrrp = high_resolution_clock::now();
    hqrrp(m, n, all_data.A2.data(), m, all_data.J2.data(), all_data.tau2.data(), b_sz,  d_factor_hqrrp * block_size, panel_pivoting, state);
    auto stop_hqrrp = high_resolution_clock::now();
    long dur_hqrrp = duration_cast<microseconds>(stop_hqrrp - start_hqrrp).count();

    // Testing GEQRF
    auto start_geqrf = high_resolution_clock::now();
    lapack::geqrf(m, n, all_data.A3.data(), m, all_data.tau3.data());
    auto stop_geqrf = high_resolution_clock::now();
    long dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

    std::vector<long> res{dur_cqrrp, dur_hqrrp, dur_geqrf};

    return res;
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 17);
    int64_t n          = std::pow(2, 17);
    int64_t k          = std::pow(2, 17);
    int64_t d_factor   = 2.0;
    int64_t b_sz_start = 512;
    int64_t b_sz_end   = 4096;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState();
    // Timing results
    std::vector<double> res;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    // Create copies of the input matrix
    copy_computational_helper<double, r123::Philox4x32>(all_data);

    // Declare a data file
    std::fstream file("QR_time_raw" + "_rows_"         + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz)
                                    + "_b_sz_end_"     + std::to_string(b_sz)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

    for (b_sz_start < b_sz_end; b_sz_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(b_sz_start, all_data, state);
        file << res[0]  << "  " << res[1]  << "  " << res[2];
    }
}


