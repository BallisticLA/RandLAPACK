/*
HQRRP runtime breakdown benchmark - assesses the time taken by each subcomponent of HQRRP.
There are 7 things that we time:
                1. Preallocation time.
                2. Sketch generation and application time.
                3. Downdating time.
                4. QRCP time.
                5. QR time.
                6. Updating A time.
                7. Updating Sketch time.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_bench_utils.hh"

#include <RandBLAS.hh>
#include <fstream>

using RandLAPACK::bench::QR_benchmark_data;
using RandLAPACK::bench::data_regen;

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto d_factor = all_data.sampling_factor;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;
    int panel_pivoting = 0;

    // Timing vars
    T* times = new T[27]();
    T** times_ptr = &times;

    for (int i = 0; i < numruns; ++i) {
        std::cout << "ITERATION " << i << ", B_SZ " << b_sz << "\n";

        // Testing HQRRP
        // No CholQR
        RandLAPACK::hqrrp(m, n, all_data.A, m, all_data.J, all_data.tau, b_sz, (d_factor - 1) * b_sz, panel_pivoting, 0, state_alg, times_ptr);

        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        RandLAPACK::bench::write_breakdown_row(file, times, times + 27);

        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        state_gen = state;
        state_alg = state;
    }

    delete[] times;
}

int main(int argc, char *argv[]) {

    if (argc < 6) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <num_runs> <num_rows> <num_cols> <block_sizes>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m          = std::stol(argv[3]);
    int64_t n          = std::stol(argv[4]);
    double  d_factor   = 1.0;
    std::vector<int64_t> b_sz(argc - 5);
    RandLAPACK::bench::parse_sizes(argc, argv, 5, b_sz.data());
    // Save elements in string for logging purposes
    std::string b_sz_string = RandLAPACK::bench::sizes_string(b_sz.data(), b_sz.size());

    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs.
    int64_t numruns = std::stol(argv[2]);

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n, {.d_factor = d_factor});
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Declare a data file
    std::string output_filename = "_HQRRP_runtime_breakdown_num_info_lines_" + std::to_string(7) + ".txt";

    std::string path = RandLAPACK::bench::output_path(argv[1], output_filename);

    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the HQRRP runtime breakdown benchmark, recording the time it takes to perform every subroutine in HQRRP."
              "\nFile format: 26 data columns, each corresponding to a given HQRRP subroutine (please see /RandLAPACK/drivers/rl_hqrrp.hh for details)"
              "\nrows correspond to HQRRP runs with block sizes varying as specified, with numruns repititions of each block size"
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: HQRRP block sizes: " + b_sz_string + "num runs per size " + std::to_string(numruns) + " HQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();
    
    auto start_time_all = steady_clock::now();
    size_t i = 0;
    for (;i < b_sz.size(); ++i) {
        call_all_algs(m_info, numruns, b_sz[i], all_data, state_constant, path);
    }
    RandLAPACK::bench::write_total_time(file, start_time_all);
}
