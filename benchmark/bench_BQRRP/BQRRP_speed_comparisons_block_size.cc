/*
BQRRP speed comparison benchmark - runs:
    1. BQRRP_CQR and BQRRP_HQR
    2. GEQRF
    3. GEQP3 - takes too long!
    5. HQRRP + CholQR
    6. HQRRP + GEQRF
for a matrix with fixed number of rows and columns and a varying BQRRP block size.
Records the best timing, saves that into a file.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_bench_utils.hh"

#include <RandBLAS.hh>
#include <fstream>

using Subroutines = RandLAPACK::BQRRPSubroutines;
using RandLAPACK::bench::QR_benchmark_data;
using RandLAPACK::bench::data_regen;

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    QR_benchmark_data<T> &all_data,
    std::string operation_mode,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m          = all_data.row;
    auto n          = all_data.col;
    auto d_factor   = all_data.sampling_factor;
    auto hqrrp_b_sz = b_sz;
    if (operation_mode == "hqrrp_const" || operation_mode == "default_hqrrp_const") {
        hqrrp_b_sz = 128;
    }

    // Additional params setup.
    RandLAPACK::BQRRP<T, r123::Philox4x32> BQRRP(false, b_sz);
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vars
    long dur_bqrrp_cholqr = 0;
    long dur_bqrrp_qrf    = 0;
    long dur_hqrrp        = 0;
    long dur_hqrrp_geqrf  = 0;
    long dur_hqrrp_cholqr = 0;
    long dur_geqrf        = 0;
    long dur_geqp3        = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        std::cout << "ITERATION " << i << ", B_SZ " << b_sz << "\n";

        // Testing GEQRF
        auto start_geqrf = steady_clock::now();
        lapack::geqrf(m, n, all_data.A, m, all_data.tau);
        auto stop_geqrf = steady_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        std::cout << "TOTAL TIME FOR GEQRF " << dur_geqrf << "\n";

        // Making sure the states are unchanged
        state_gen = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        if (operation_mode != "hqrrp" && operation_mode != "hqrrp_cost") {
            // Testing BQRRP - QRF
            BQRRP.qr_tall = Subroutines::QRTall::geqrf;
            BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
            auto start_bqrrp_qrf = steady_clock::now();
            BQRRP.call(m, n, all_data.A, m, d_factor, all_data.tau, all_data.J, state_alg);
            auto stop_bqrrp_qrf = steady_clock::now();
            dur_bqrrp_qrf = duration_cast<microseconds>(stop_bqrrp_qrf - start_bqrrp_qrf).count();
            std::cout << "TOTAL TIME FOR BQRRP_QRF " << dur_bqrrp_qrf << "\n";

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);

            // Testing BQRRP - CholQR
            BQRRP.qr_tall = Subroutines::QRTall::cholqr;
            BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
            auto start_bqrrp_cholqr = steady_clock::now();
            BQRRP.call(m, n, all_data.A, m, d_factor, all_data.tau, all_data.J, state_alg);
            auto stop_bqrrp_cholqr = steady_clock::now();
            dur_bqrrp_cholqr = duration_cast<microseconds>(stop_bqrrp_cholqr - start_bqrrp_cholqr).count();
            std::cout << "TOTAL TIME FOR BQRRP_CHOLQR " << dur_bqrrp_cholqr << "\n";

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);
        }
        
        // Testing HQRRP DEFAULT
        auto start_hqrrp = steady_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A, m, all_data.J, all_data.tau, hqrrp_b_sz, (d_factor - 1) * hqrrp_b_sz, panel_pivoting, 0, state_alg, (T**) nullptr);
        auto stop_hqrrp = steady_clock::now();
        dur_hqrrp = duration_cast<microseconds>(stop_hqrrp - start_hqrrp).count();
        std::cout << "TOTAL TIME FOR HQRRP " << dur_hqrrp << "\n";

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        if (operation_mode == "full") {
            // Testing HQRRP with GEQRF
            auto start_hqrrp_geqrf = steady_clock::now();
            RandLAPACK::hqrrp(m, n, all_data.A, m, all_data.J, all_data.tau, hqrrp_b_sz,  (d_factor - 1) * hqrrp_b_sz, panel_pivoting, 1, state_alg, (T**) nullptr);
            auto stop_hqrrp_geqrf = steady_clock::now();
            dur_hqrrp_geqrf = duration_cast<microseconds>(stop_hqrrp_geqrf - start_hqrrp_geqrf).count();
            std::cout << "TOTAL TIME FOR HQRRP WITH GEQRF " << dur_hqrrp_geqrf << "\n";

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);

            // Testing HQRRP with CholQR
            auto start_hqrrp_cholqr = steady_clock::now();
            RandLAPACK::hqrrp(m, n, all_data.A, m, all_data.J, all_data.tau, hqrrp_b_sz,  (d_factor - 1) * hqrrp_b_sz, panel_pivoting, 2, state_alg, (T**) nullptr);
            auto stop_hqrrp_cholqr = steady_clock::now();
            dur_hqrrp_cholqr = duration_cast<microseconds>(stop_hqrrp_cholqr - start_hqrrp_cholqr).count();
            std::cout << "TOTAL TIME FOR HQRRP WITH CHOLQRQ " << dur_hqrrp_cholqr << "\n";

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);
        }

        if ((i <= 2) && (b_sz == 256)) {
            // Testing GEQP3
            auto start_geqp3 = steady_clock::now();
            lapack::geqp3(m, n, all_data.A, m, all_data.J, all_data.tau);
            auto stop_geqp3 = steady_clock::now();
            dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
            std::cout << "TOTAL TIME FOR GEQP3 " << dur_geqp3 << "\n";

            state_gen = state;
            data_regen(m_info, all_data, state_gen);
        }
        
        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        RandLAPACK::bench::write_row(file, dur_bqrrp_cholqr, dur_bqrrp_qrf, dur_hqrrp, dur_hqrrp_geqrf, dur_hqrrp_cholqr, dur_geqrf, dur_geqp3);
    }
}

int main(int argc, char *argv[]) {

    if (argc < 7) {
        // Expected input into this benchmark.
        // "Operation mode" options: default, default_hqrrp_const, hqrrp, hqrrp_const, full
        std::cerr << "Usage: " << argv[0] << " <directory_path> <operation_mode> <num_runs> <num_rows> <num_cols> <block_sizes>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m          = std::stol(argv[4]);
    int64_t n          = std::stol(argv[5]);
    double d_factor    = 1.0;
    std::vector<int64_t> b_sz(argc - 6);
    RandLAPACK::bench::parse_sizes(argc, argv, 6, b_sz.data());
    // Save elements in string for logging purposes
    std::string b_sz_string = RandLAPACK::bench::sizes_string(b_sz.data(), b_sz.size());

    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = std::stol(argv[3]);

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n, {.d_factor = d_factor});
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Declare a data file
    std::string output_filename = "_BQRRP_speed_comparisons_block_size_num_info_lines_" + std::to_string(7) + ".txt";

    std::string path = RandLAPACK::bench::output_path(argv[1], output_filename);
                                                               
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the BQRRP speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
              "\nFile format: 7 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, HQRRP, HQRRP+QRF, HQRRP+CholQR, QRF, QP3;"
              "               rows correspond to BQRRP runs with block sizes varying as specified or multiples of 10, with numruns repititions of each block size"
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: BQRRP block sizes: " + b_sz_string + "num runs per size " + std::to_string(numruns) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();

    auto start_time_all = steady_clock::now();
    size_t i = 0;
    for (;i < b_sz.size(); ++i) {
        call_all_algs(m_info, numruns, b_sz[i], all_data, argv[2], state_constant, path);
    }
    RandLAPACK::bench::write_total_time(file, start_time_all);
}
