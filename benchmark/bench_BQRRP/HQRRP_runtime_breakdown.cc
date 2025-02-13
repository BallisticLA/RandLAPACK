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

    for (int i = 0; i < numruns; ++i) {
        printf("ITERATION %d, NUMCOLS %ld\n", i, n);

        // Testing HQRRP
        // No CholQR
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz, (d_factor - 1) * b_sz, panel_pivoting, 0, state_alg, times);

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

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }
    auto size = argv[1];

    // Declare parameters
    int64_t m          = std::stol(size);
    int64_t n          = std::stol(size);
    double  d_factor   = 1.0;
    int64_t b_sz_start = 32;
    int64_t b_sz_end   = 2048;
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs.
    int64_t numruns = 3;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = RandLAPACK::util::getCurrentDate<double>() + "HQRRP_runtime_breakdown" 
                                                                 + "_num_info_lines_" + std::to_string(6) +
                                                                   ".txt";

    std::ofstream file(output_filename, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the HQRRP runtime breakdown benchmark, recording the time it takes to perform every subroutine in HQRRP."
              "\nFile format: 27 data columns, each corresponding to a given HQRRP subroutine (please see /RandLAPACK/drivers/rl_hqrrp.hh for details) "
              "               rows correspond to HQRRP runs with block sizes varying in powers of 2, with numruns repititions of each block size"
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: HQRRP block size start: " + std::to_string(b_sz_start) + " HQRRP block size end: " + std::to_string(b_sz_end) + " num runs per size " + std::to_string(numruns) + " HQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();
    

    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant, output_filename);
    }
}
#endif
