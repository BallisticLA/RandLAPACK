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
                                        RandBLAS::RNGState<RNG> &state, int apply_itoa) {

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (apply_itoa) {
        std::iota(all_data.J.begin(), all_data.J.end(), 1);
    } else {
        std::fill(all_data.J.begin(), all_data.J.end(), 0);
    }
}

template <typename T>
static void select_best(std::vector<T> best, std::vector<T> curr) {
    for(int i = 0; i < (int) best.size(); ++i) {
        if (curr[i] < best[i]) { best[i] = curr[i]; }
    }
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    RandBLAS::RNGState<RNG> &state_constant) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;
    CQRRP_blocked.timing_advanced = 1;
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vectors
    std::vector<T> best_time_cqrrpt(std::ceil(n / b_sz), 0.0);
    std::vector<T> best_time_hqrrp_geqrf(std::ceil(n / b_sz), 0.0);
    std::vector<T> best_time_hqrrp_cholqr(std::ceil(n / b_sz), 0.0);

    std::vector<T> time_hqrrp_geqrf(std::ceil(n / b_sz), 0.0);
    std::vector<T> time_hqrrp_cholqr(std::ceil(n / b_sz), 0.0);


    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION\n");
        // Testing CQRRP
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state);
        // Update best timing
        if (i == 0) { best_time_cqrrpt = CQRRP_blocked.block_per_time; } else { select_best<T>(best_time_cqrrpt, CQRRP_blocked.block_per_time); }

        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_constant, 1);

        // Testing HQRRP with GEQRF
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 0, state, time_hqrrp_geqrf.data());
        // Update best timing

        if(i == 0) { best_time_hqrrp_geqrf = time_hqrrp_geqrf; } else { select_best<T>(best_time_hqrrp_geqrf, time_hqrrp_geqrf); }

        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_constant, 1);

        // Testing HQRRP with Cholqr
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 1, state, time_hqrrp_cholqr.data());
        // Update best timing
        if(i == 0) { best_time_hqrrp_cholqr = time_hqrrp_cholqr; } else { select_best<T>(best_time_hqrrp_cholqr, time_hqrrp_cholqr);}

        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_constant, 0);
    }

    // The actual output may be interpreted as 
    std::fstream file("QR_block_per_time_raw_rows_"    + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_"         + std::to_string(b_sz)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);
    
    for (int i = 0; i < std::ceil(n / b_sz); ++i) {
        file << best_time_cqrrpt[i]  << "  " << best_time_hqrrp_geqrf[i]  << "  " << best_time_hqrrp_cholqr[i] << "\n";
    }
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 14);
    int64_t n          = std::pow(2, 14);
    double d_factor   = 1.125;
    int64_t b_sz_start = 1024;
    int64_t b_sz_end   = 1024;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        call_all_algs<double, r123::Philox4x32>(m_info, numruns, b_sz_start, all_data, state, state_constant);
    }
}
