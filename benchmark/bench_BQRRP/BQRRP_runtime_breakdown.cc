#if defined(__APPLE__)
int main() {return 0;}
#else
/*
IBQRRP runtime breakdown benchmark - assesses the time taken by each subcomponent of IBQRRP.
There are 10 things that we time:
                1. Preallocation time
                2. SKOP time
                3. QRCP_wide time
                4. Panel preprocessing time
                5. QR_tall time
                6. Householder reconstruction time
                7. Apply QT time
                8. Sample updating time
                9. Other routines time
                10. Total time
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

using Subroutines = RandLAPACK::BQRRPSubroutines;

template <typename T>
struct QR_speed_benchmark_data {
    int64_t row;
    int64_t col;
    T       sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_speed_benchmark_data(int64_t m, int64_t n, T d_factor) :
    A(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0)
    {
        row             = m;
        col             = n;
        sampling_factor = d_factor;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_speed_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state) {

    std::fill(all_data.A.begin(), all_data.A.end(), 0.0);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t b_sz,
    std::string qr_tall,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::BQRRP<T, r123::Philox4x32> BQRRP(true, b_sz);
    if(qr_tall == "geqrt") {
        BQRRP.qr_tall       = Subroutines::QRTall::geqrt;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::gemqrt;
    } else if (qr_tall == "cholqr") {
        BQRRP.qr_tall       = Subroutines::QRTall::cholqr;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
    } else {
        BQRRP.qr_tall       = Subroutines::QRTall::geqrf;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
    }

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    // Timing vars
    std::vector<long> inner_timing;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, b_sz %ld\n", i, b_sz);
        BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);

        // Update timing vector
        inner_timing = BQRRP.times;
        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        std::copy(inner_timing.begin(), inner_timing.end(), std::ostream_iterator<long>(file, ", "));
        file << "\n";

        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        state_gen = state;
        state_alg = state;
    }
}

int main(int argc, char *argv[]) {

    if (argc < 7) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <qr_fname> <num_runs> <num_rows> <num_cols> <block_sizes>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m          = std::stol(argv[4]);
    int64_t n          = std::stol(argv[5]);
    double  d_factor   = 1.0;
    // Fill the block size vector
    std::vector<int64_t> b_sz;
    for (int i = 0; i < argc-6; ++i)
        b_sz.push_back(std::stoi(argv[i + 6]));
    // Save elements in string for logging purposes
    std::ostringstream oss;
    for (const auto &val : b_sz)
        oss << val << ", ";
    std::string b_sz_string = oss.str();

    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns     = std::stol(argv[3]);
    std::string qr_tall = argv[2];

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "_BQRRP_runtime_breakdown_num_info_lines_" + std::to_string(7) + ".txt";

    std::string path;
    if (std::string(argv[1]) != ".") {
        path = std::string(argv[1]) + output_filename;
    } else {
        path = output_filename;
    }

    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the BQRRP runtime breakdown benchmark, recording the time it takes to perform every subroutine in BQRRP."
              "\nFile format: 10 data columns, each corresponding to a given BQRRP subroutine: skop_t_dur, preallocation_t_dur, qrcp_wide_t_dur, panel_preprocessing_t_dur, qr_tall_t_dur, q_reconstruction_t_dur, apply_transq_t_dur, sample_update_t_dur, t_other, total_t_dur"
              "               rows correspond to BQRRP runs with block sizes varying as specified, with numruns repititions of each block size"
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: Tall QR subroutine " + argv[1] + " BQRRP block sizes: " + b_sz_string + "num runs per size " + std::to_string(numruns) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();

    auto start_time_all = steady_clock::now();
    size_t i = 0;
    for (;i < b_sz.size(); ++i) {
        call_all_algs(m_info, numruns, b_sz[i], qr_tall, all_data, state_constant, path);
    }
    auto stop_time_all = steady_clock::now();
    long dur_time_all = duration_cast<microseconds>(stop_time_all - start_time_all).count();
    file << "Total benchmark execution time:" +  std::to_string(dur_time_all) + "\n";
    file.flush();   
}
#endif
