/*
CQRRPT runtime breakdown benchmark - assesses the time taken by each subcomponent of CQRRPT.
There are 6 things that we time:
                1. SASO generation and application time
                2. QRCP time.
                3. Time it takes to compute numerical rank k.
                4. piv(A).
                5. TRSM(A).
                6. Time to perform Cholesky QR.
*/
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct QR_benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    T       sampling_factor;
    std::vector<T> A;
    std::vector<T> R;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_benchmark_data(int64_t m, int64_t n, T tol, T d_factor) :
    A(m * n, 0.0),
    R(n * n, 0.0),
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
                                        QR_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static std::vector<long> call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(true, true, tol);
    CQRRPT.nnz = 4;
    CQRRPT.num_threads = 8;
    
    // Making sure the states are unchanged
    auto state_alg = state;
    auto state_gen = state;

    // Timing vars
    long dur_cqrrpt    = 0;
    long t_cqrrpt_best = 0;
    std::vector<long> inner_timing_best;

    for (int i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        auto start_cqrrpt = high_resolution_clock::now();
        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state_alg);
        auto stop_cqrrpt = high_resolution_clock::now();
        dur_cqrrpt = duration_cast<microseconds>(stop_cqrrpt - start_cqrrpt).count();
        // Update best timing
        if (!i || dur_cqrrpt < t_cqrrpt_best) {t_cqrrpt_best = dur_cqrrpt; inner_timing_best = CQRRPT.times;}
        // Making sure the states are unchanged
        state_alg = state;
        state_gen = state;
        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state_gen);
    }

    return inner_timing_best;
}

int main() {
    // Declare parameters
    int64_t m           = std::pow(2, 12);
    int64_t n_start     = std::pow(2, 5);
    int64_t n_stop      = std::pow(2, 5);
    double  d_factor    = 1.25;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 25;

    // Allocate basic workspace at its max size.
    QR_benchmark_data<double> all_data(m, n_stop, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    // Declare a data file
    std::fstream file("CQRRPT_inner_speed_"              + std::to_string(m)
                                      + "_col_start_"    + std::to_string(n_start)
                                      + "_col_stop_"     + std::to_string(n_stop)
                                      + "_d_factor_"     + std::to_string(d_factor)
                                      + ".dat", std::fstream::app);

    for (;n_start <= n_stop; n_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, n_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",  " << res[2] << ",  " << res[3] << ",  " << res[4] << ",  " << res[5] << ",  " << res[6] << ",  " << res[7] << ",\n";
    }
}