#if defined(__APPLE__)
int main() {return 0;}
#else
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

#include <RandBLAS.hh>
#include <fstream>

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
    auto d_factor = all_data.sampling_factor;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;
    int panel_pivoting = 0;

    // Timing vars
    T* times  = ( T * ) calloc( 27, sizeof( T ) );
    T** times_ptr = &times;

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, B_SZ %ld\n", i, b_sz);

        // Testing HQRRP
        // No CholQR
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz, (d_factor - 1) * b_sz, panel_pivoting, 0, state_alg, times_ptr);

        std::ofstream file(output_filename, std::ios::app);
        std::copy(times, times + 27, std::ostream_iterator<T>(file, ", "));
        file << "\n";

        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        state_gen = state;
        state_alg = state;
    }

    free(times);
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <num_runs> <num_rows> <num_cols> <block_sizes>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m          = std::stol(argv[3]);
    int64_t n          = std::stol(argv[4]);
    double  d_factor   = 1.0;
    // Fill the block size vector
    std::vector<int64_t> b_sz;
    for (int i = 0; i < argc-5; ++i)
        b_sz.push_back(std::stoi(argv[i + 5]));
    // Save elements in string for logging purposes
    std::ostringstream oss;
    for (const auto &val : b_sz)
        oss << val << ", ";
    std::string b_sz_string = oss.str();

    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs.
    int64_t numruns = std::stol(argv[2]);

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "_HQRRP_runtime_breakdown_num_info_lines_" + std::to_string(7) + ".txt";

    std::string path;
    if (std::string(argv[1]) != ".")
        path = std::string(argv[1]) + output_filename;

    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the HQRRP runtime breakdown benchmark, recording the time it takes to perform every subroutine in HQRRP."
              "\nFile format: 26 data columns, each corresponding to a given HQRRP subroutine (please see /RandLAPACK/drivers/rl_hqrrp.hh for details)"
              "\nrows correspond to HQRRP runs with block sizes varying in powers of 2, with numruns repititions of each block size"
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: HQRRP block sizes: " + b_sz_string + "num runs per size " + std::to_string(numruns) + " HQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();
    
    auto start_time_all = steady_clock::now();
    size_t i = 0;
    for (;i < b_sz.size(); ++i) {
        call_all_algs(m_info, numruns, b_sz[i], all_data, state_constant, output_filename);
    }
    auto stop_time_all = steady_clock::now();
    long dur_time_all = duration_cast<microseconds>(stop_time_all - start_time_all).count();
    file << "Total benchmark execution time:" +  std::to_string(dur_time_all) + "\n";
    file.flush();   
}
#endif
