

#if defined(__APPLE__)
int main() {return 0;}
#else

// Making sure that HQRRP's performance instability is specific to HQRRP and not related to the flaws in behcnmarking logic
// by comparing HQRRP and GEMM side-by-side.
// This benchmark is to be ran on multiple systems.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

using Subroutines = RandLAPACK::BQRRPSubroutines;

template <typename T>
struct QR_benchmark_data {
    int64_t row;
    int64_t col;
    std::vector<T> A;
    std::vector<T> B;
    std::vector<T> C;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_benchmark_data(int64_t m, int64_t n) :
    A(m * n, 0.0),
    B(n * m, 0.0),
    C(m * m, 0.0),
    tau(n, 0.0),
    J(n, 0)
    {
        row = m;
        col = n;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state) {
    auto state_buf = state;

    std::fill(all_data.A.begin(), all_data.A.end(), 0.0);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state_buf);
    RandLAPACK::gen::mat_gen(m_info, all_data.B.data(), state_buf);
    std::fill(all_data.C.begin(), all_data.C.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t rows,
    int64_t cols,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m          = rows;
    auto n          = cols;
    auto hqrrp_b_sz = 128;

    int panel_pivoting = 0;

    // timing vars
    long dur_hqrrp = 0;
    long dur_gemm  = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        std::cout << "ITERATION " << i << ", ROWS " << m << "\n";

        // Testing HQRRP DEFAULT
        auto start_hqrrp = steady_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), hqrrp_b_sz,  1.0 * hqrrp_b_sz, panel_pivoting, 0, state_alg, (T**) nullptr);
        auto stop_hqrrp = steady_clock::now();
        dur_hqrrp = duration_cast<microseconds>(stop_hqrrp - start_hqrrp).count();
        std::cout << "TOTAL TIME FOR HQRRP " << dur_hqrrp << "\n";

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);

        // Testing GEMM
        auto start_gemm = steady_clock::now();
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, m, n, 1.0, all_data.A.data(), m, all_data.B.data(), n, 0.0, all_data.C.data(), m);
        auto stop_gemm = steady_clock::now();
        dur_gemm = duration_cast<microseconds>(stop_gemm - start_gemm).count();
        std::cout << "TOTAL TIME FOR GEMM " << dur_gemm << "\n";

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        file << dur_hqrrp << ",  " << dur_gemm << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if (argc < 4) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <num_runs> <square_matrix_dim (multiple)>..." << std::endl;
        return 1;
    }

    // Declare parameters
    // Fill the block size vector
    std::vector<int64_t> m_sz;
    for (int i = 0; i < argc-3; ++i)
        m_sz.push_back(std::stoi(argv[i + 3]));
    // Save elements in string for logging purposes
    std::ostringstream oss;
    for (const auto &val : m_sz)
        oss << val << ", ";
    std::string m_sz_string = oss.str();

    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = std::stol(argv[2]);

    // Allocate basic workspace
    int64_t m_max = *std::max_element(m_sz.begin(), m_sz.end());
    QR_benchmark_data<double> all_data(m_max, m_max);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m_max, m_max, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "_HQRRP_GEMM_speed_comparisons_mat_size_num_info_lines_" + std::to_string(7) + ".txt";

    std::string path;
    if (std::string(argv[1]) != ".") {
        path = std::string(argv[1]) + output_filename;
    } else {
        path = output_filename;
    }

    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the sanity check for the odd performance of HQRRP."
              "\nFile format: 2 columns, containing time for each algorithm: HQRRP, GEMM;"
              "\nrows correspond to BQRRP runs with varying mat sizes, with numruns repititions of each mat size."
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput row sizes:"  + m_sz_string +
              "\nAdditional parameters: HQRRP columns/block size: 128 num runs per size " + std::to_string(numruns) + " HQRRP d factor: 1.0"  
              "\n";
    file.flush();

    auto start_time_all = steady_clock::now();
    size_t i = 0;
    for (;i < m_sz.size(); ++i) {
        call_all_algs(m_info, numruns, m_sz[i], m_sz[i], all_data, state_constant, path);
    }
    auto stop_time_all = steady_clock::now();
    long dur_time_all = duration_cast<microseconds>(stop_time_all - start_time_all).count();
    file << "Total benchmark execution time:" +  std::to_string(dur_time_all) + "\n";
    file.flush();   
}
#endif