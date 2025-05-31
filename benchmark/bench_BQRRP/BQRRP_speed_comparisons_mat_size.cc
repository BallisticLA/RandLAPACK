#if defined(__APPLE__)
int main() {return 0;}
#else
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

#include <RandBLAS.hh>
#include <fstream>

using Subroutines = RandLAPACK::BQRRPSubroutines;

template <typename T>
struct QR_speed_benchmark_data {
    int64_t row;
    int64_t col;
    int64_t b_sz;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_speed_benchmark_data(int64_t m, int64_t n, int64_t block_size, T d_factor) :
    A(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0)
    {
        row             = m;
        col             = n;
        b_sz            = block_size;
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
    int64_t rows,
    int64_t cols,
    int64_t block_sz,
    QR_speed_benchmark_data<T> &all_data,
    std::string operation_mode,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m          = rows;
    auto n          = cols;
    auto b_sz       = block_sz;
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
        printf("ITERATION %d, ROWS %ld\n", i, m);
        
        // Testing GEQRF
        auto start_geqrf = steady_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = steady_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        printf("TOTAL TIME FOR GEQRF %ld\n", dur_geqrf);

        // Making sure the states are unchanged
        state_gen = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        if (operation_mode != "hqrrp" && operation_mode != "hqrrp_cost") {
            // Testing BQRRP - QRF
            BQRRP.qr_tall = Subroutines::QRTall::geqrf;
            BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
            auto start_bqrrp_qrf = steady_clock::now();
            BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
            auto stop_bqrrp_qrf = steady_clock::now();
            dur_bqrrp_qrf = duration_cast<microseconds>(stop_bqrrp_qrf - start_bqrrp_qrf).count();
            printf("TOTAL TIME FOR BQRRP_QRF %ld\n", dur_bqrrp_qrf);

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);

            // Testing BQRRP - CholQR
            BQRRP.qr_tall = Subroutines::QRTall::cholqr;
            BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
            auto start_bqrrp_cholqr = steady_clock::now();
            BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
            auto stop_bqrrp_cholqr = steady_clock::now();
            dur_bqrrp_cholqr = duration_cast<microseconds>(stop_bqrrp_cholqr - start_bqrrp_cholqr).count();
            printf("TOTAL TIME FOR BQRRP_CHOLQR %ld\n", dur_bqrrp_cholqr);

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);
        }
        
        // Testing HQRRP DEFAULT
        auto start_hqrrp = steady_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), hqrrp_b_sz,  (d_factor - 1) * hqrrp_b_sz, panel_pivoting, 0, state_alg, (T**) nullptr);
        auto stop_hqrrp = steady_clock::now();
        dur_hqrrp = duration_cast<microseconds>(stop_hqrrp - start_hqrrp).count();
        printf("TOTAL TIME FOR HQRRP %ld\n", dur_hqrrp);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        if (operation_mode == "full") {
            // Testing HQRRP with GEQRF
            auto start_hqrrp_geqrf = steady_clock::now();
            RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), hqrrp_b_sz,  (d_factor - 1) * hqrrp_b_sz, panel_pivoting, 1, state_alg, (T**) nullptr);
            auto stop_hqrrp_geqrf = steady_clock::now();
            dur_hqrrp_geqrf = duration_cast<microseconds>(stop_hqrrp_geqrf - start_hqrrp_geqrf).count();
            printf("TOTAL TIME FOR HQRRP WITH GEQRF %ld\n", dur_hqrrp_geqrf);

            // Making sure the states are unchanged
            state_gen = state;
            state_alg = state;
            // Clear and re-generate data
            data_regen(m_info, all_data, state_gen);

            // Testing HQRRP with CholQR
            auto start_hqrrp_cholqr = steady_clock::now();
            RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), hqrrp_b_sz,  (d_factor - 1) * hqrrp_b_sz, panel_pivoting, 2, state_alg, (T**) nullptr);
            auto stop_hqrrp_cholqr = steady_clock::now();
            dur_hqrrp_cholqr = duration_cast<microseconds>(stop_hqrrp_cholqr - start_hqrrp_cholqr).count();
            printf("TOTAL TIME FOR HQRRP WITH CHOLQRQ %ld\n", dur_hqrrp_cholqr);
        }

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);

        if (i <= 2) {
            // Testing GEQP3
            auto start_geqp3 = steady_clock::now();
            lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
            auto stop_geqp3 = steady_clock::now();
            dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
            printf("TOTAL TIME FOR GEQP3 %ld\n", dur_geqp3);

            state_gen = state;
            data_regen(m_info, all_data, state_gen);
        }
        
        std::ofstream file(output_filename, std::ios::out | std::ios::app);
        file << dur_bqrrp_cholqr << ",  " << dur_bqrrp_qrf << ",  " << dur_hqrrp << ",  " << dur_hqrrp_geqrf << ",  " << dur_hqrrp_cholqr << ",  " << dur_geqrf << ",  " << dur_geqp3 << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if (argc < 7) {
        // Expected input into this benchmark.
        // "Operation mode" options: default, default_hqrrp_const, hqrrp, hqrrp_const, full
        std::cerr << "Usage: " << argv[0] << " <directory_path> <operation_mode> <num_runs> <column_size_ratio> <block_size_ratio> <square_matrix_dim (multiple)>..." << std::endl;
        return 1;
    }

    // Declare parameters
    // Fill the block size vector
    std::vector<int64_t> m_sz;
    for (int i = 0; i < argc-6; ++i)
        m_sz.push_back(std::stoi(argv[i + 6]));
    // Save elements in string for logging purposes
    std::ostringstream oss;
    for (const auto &val : m_sz)
        oss << val << ", ";
    std::string m_sz_string = oss.str();

    double d_factor     = 1.0;
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = std::stol(argv[3]);

    // Allocate basic workspace
    double column_size_ratio = std::stof(argv[4]);
    int64_t m_max = *std::max_element(m_sz.begin(), m_sz.end());
    if (column_size_ratio < 1) 
        m_max = m_max / column_size_ratio;
    QR_speed_benchmark_data<double> all_data(m_max, m_max, m_max / std::stol(argv[5]), d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m_max, m_max, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "_BQRRP_speed_comparisons_mat_size_num_info_lines_" + std::to_string(7) + ".txt";

    std::string path;
    if (std::string(argv[1]) != ".") {
        path = std::string(argv[1]) + output_filename;
    } else {
        path = output_filename;
    }

    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the BQRRP speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
              "\nFile format: 7 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, HQRRP, HQRRP+QRF, HQRRP+CholQR, QRF, QP3;"
              "               rows correspond to BQRRP runs with varying mat sizes, with numruns repititions of each mat size."
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput row sizes:"  + m_sz_string + ", input row/column ratio: " + argv[3] +
              "\nAdditional parameters: BQRRP columns/block size ratio: " + argv[4] + " num runs per size " + std::to_string(numruns) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();

    auto start_time_all = steady_clock::now();
    size_t i = 0;
    int64_t columns;
    int64_t b_sz;
    for (;i < m_sz.size(); ++i) {
        columns = m_sz[i] / column_size_ratio;
        b_sz = columns / std::stof(argv[5]);
        call_all_algs(m_info, numruns, m_sz[i], columns, b_sz, all_data, argv[2], state_constant, path);
    }
    auto stop_time_all = steady_clock::now();
    long dur_time_all = duration_cast<microseconds>(stop_time_all - start_time_all).count();
    file << "Total benchmark execution time:" +  std::to_string(dur_time_all) + "\n";
    file.flush();   
}
#endif