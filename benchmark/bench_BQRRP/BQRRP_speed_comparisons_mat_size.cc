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

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t dim,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m        = dim;
    auto n        = dim;
    auto b_sz     = all_data.b_sz;
    auto d_factor = all_data.sampling_factor;

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
        printf("ITERATION %d, DIM %ld\n", i, dim);
        
        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        printf("TOTAL TIME FOR GEQRF %ld\n", dur_geqrf);

        // Making sure the states are unchanged
        state_gen = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        // Testing BQRRP - QRF
        BQRRP.qr_tall = Subroutines::QRTall::geqrf;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
        auto start_bqrrp_qrf = high_resolution_clock::now();
        BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
        auto stop_bqrrp_qrf = high_resolution_clock::now();
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
        auto start_bqrrp_cholqr = high_resolution_clock::now();
        BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
        auto stop_bqrrp_cholqr = high_resolution_clock::now();
        dur_bqrrp_cholqr = duration_cast<microseconds>(stop_bqrrp_cholqr - start_bqrrp_cholqr).count();
        printf("TOTAL TIME FOR BQRRP_CHOLQR %ld\n", dur_bqrrp_cholqr);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        // Testing HQRRP DEFAULT
        auto start_hqrrp = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 0, state_alg, (T*) nullptr);
        auto stop_hqrrp = high_resolution_clock::now();
        dur_hqrrp = duration_cast<microseconds>(stop_hqrrp - start_hqrrp).count();
        printf("TOTAL TIME FOR HQRRP %ld\n", dur_hqrrp);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);
        
        // Testing HQRRP with GEQRF
        auto start_hqrrp_geqrf = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 1, state_alg, (T*) nullptr);
        auto stop_hqrrp_geqrf = high_resolution_clock::now();
        dur_hqrrp_geqrf = duration_cast<microseconds>(stop_hqrrp_geqrf - start_hqrrp_geqrf).count();
        printf("TOTAL TIME FOR HQRRP WITH GEQRF %ld\n", dur_hqrrp_geqrf);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);

        // Testing HQRRP with CholQR
        auto start_hqrrp_cholqr = high_resolution_clock::now();
        RandLAPACK::hqrrp(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data(), b_sz,  (d_factor - 1) * b_sz, panel_pivoting, 2, state_alg, (T*) nullptr);
        auto stop_hqrrp_cholqr = high_resolution_clock::now();
        dur_hqrrp_cholqr = duration_cast<microseconds>(stop_hqrrp_cholqr - start_hqrrp_cholqr).count();
        printf("TOTAL TIME FOR HQRRP WITH CHOLQRQ %ld\n", dur_hqrrp_cholqr);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen);

        if (i <= 2) {
            // Testing GEQP3
            auto start_geqp3 = high_resolution_clock::now();
            lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
            auto stop_geqp3 = high_resolution_clock::now();
            dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
            printf("TOTAL TIME FOR GEQP3 %ld\n", dur_geqp3);

            state_gen = state;
            data_regen(m_info, all_data, state_gen);
        }
        
        std::ofstream file(output_filename, std::ios::app);
        file << dur_bqrrp_cholqr << ",  " << dur_bqrrp_qrf << ",  " << dur_hqrrp << ",  " << dur_hqrrp_geqrf << ",  " << dur_hqrrp_cholqr << ",  " << dur_geqrf << ",  " << dur_geqp3 << ",\n";
    }
}

int main(int argc, char *argv[]) {

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }

    auto block_size = argv[1];

    // Declare parameters
    int64_t m_start     = std::pow(2, 10);
    int64_t m_end       = std::pow(2, 16);
    double d_factor     = 1.0;
    int64_t b_sz        = std::stol(block_size);
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 3;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m_end, m_end, b_sz, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m_end, m_end, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = RandLAPACK::util::getCurrentDate<double>() + "BQRRP_speed_comparisons_mat_size" 
                                                                 + "_num_info_lines_" + std::to_string(6) +
                                                                   ".txt";

    std::ofstream file(output_filename, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the BQRRP speed comparison benchmark, recording the time it takes to perform BQRRP and alternative QR and QRCP factorizations."
              "\nFile format: 7 columns, containing time for each algorithm: BQRRP+CholQR, BQRRP+QRF, HQRRP, HQRRP+QRF, HQRRP+CholQR, QRF, QP3;"
              "               rows correspond to BQRRP runs with mat sizes varying with powers of 2, with numruns repititions of each mat size."
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + " dim start: " + std::to_string(m_start) + " dim stop: "  + std::to_string(m_end) +
              "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) + " num runs per size " + std::to_string(numruns) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();

    for (;m_start <= m_end; m_start *= 2) {
        call_all_algs(m_info, numruns, m_start, all_data, state_constant, output_filename);
    }
}
#endif