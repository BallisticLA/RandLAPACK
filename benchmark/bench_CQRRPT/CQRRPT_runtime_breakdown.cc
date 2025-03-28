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

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m        = all_data.row;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT(true, tol);
    CQRRPT.nnz = 4;
    
    // Making sure the states are unchanged
    auto state_alg = state;
    auto state_gen = state;

    for (int i = 0; i < numruns; ++i) {
        printf("\nITERATION %d, N_SZ %ld\n", i, n);
        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state_alg);
        
        std::ofstream file(output_filename, std::ios::app);
        file << CQRRPT.times[0] << ",  " << CQRRPT.times[1] << ",  " << CQRRPT.times[2] << ",  " 
             << CQRRPT.times[3] << ",  " << CQRRPT.times[4] << ",  " << CQRRPT.times[5] << ",  " 
             << CQRRPT.times[6] << ",  " << CQRRPT.times[7] << ",\n";

        // Making sure the states are unchanged
        state_alg = state;
        state_gen = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
    }
}


int main(int argc, char *argv[]) {

    if (argc < 3) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <num_runs> <num_rows> <column_sizes>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m = std::stol(argv[3]);
    // Fill the block size vector
    std::vector<int64_t> n_sz;
    for (int i = 0; i < argc-3; ++i)
    n_sz.push_back(std::stoi(argv[i + 3]));
    // Save elements in string for logging purposes
    std::ostringstream oss;
    for (const auto &val : n_sz)
        oss << val << ", ";
    std::string n_sz_string = oss.str();

    double  d_factor    = 1.25;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = std::stol(argv[1]);

    // Allocate basic workspace at its max size.
    int64_t n_max = *std::max_element(n_sz.begin(), n_sz.end());
    QR_benchmark_data<double> all_data(m, n_max, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_max, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = RandLAPACK::util::get_current_date_time<double>() + "_CQRRPT_runtime_breakdown" 
                                                                 + "_num_info_lines_" + std::to_string(7) +
                                                                   ".txt";

    std::ofstream file(output_filename, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the CQRRPT runtime breakdown benchmark, recording the time it takes to perform every subroutine in CQRRPT."
              "\nFile format: 8 data columns, each corresponding to a given CQRRPT subroutine: saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqr_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, t_rest, total_t_dur"
              "               rows correspond to CQRRPT runs with block sizes varying in powers of 2, with numruns repititions of each block size"
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + n_sz_string +
              "\nAdditional parameters: num runs per size " + std::to_string(numruns) + " CQRRPT d factor: " + std::to_string(d_factor) +
              "\n";
    file.flush();

    auto start_time_all = steady_clock::now();
    size_t i = 0;
    for (;i < n_sz.size(); ++i) {
        call_all_algs(m_info, numruns, n_sz[i], all_data, state_constant, output_filename);
    }
    auto stop_time_all = steady_clock::now();
    long dur_time_all = duration_cast<microseconds>(stop_time_all - start_time_all).count();
    file << "Total benchmark execution time:" +  std::to_string(dur_time_all) + "\n";
    file.flush();   
}
