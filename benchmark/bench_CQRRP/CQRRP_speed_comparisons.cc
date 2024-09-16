#if defined(__APPLE__)
int main() {return 0;}
#else
/*
ICQRRP speed comparison benchmark - runs:
    1. ICQRRP
    2. GEQRF
    3. GEQP3 - takes too long!
    5. HQRRP + CholQR
    6. HQRRP + GEQRF
for a matrix with fixed number of rows and columns and a varying ICQRRP block size.
Records the best timing, saves that into a file.
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
    T       tolerance;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_speed_benchmark_data(int64_t m, int64_t n, T tol, T d_factor) :
    A(m * n, 0.0),
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
                                        QR_speed_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int apply_itoa) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (apply_itoa) {
        std::iota(all_data.J.begin(), all_data.J.end(), 1);
    } else {
        std::fill(all_data.J.begin(), all_data.J.end(), 0);
    }
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
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<T, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    // We are nbot using panel pivoting in performance testing.
    int panel_pivoting = 0;

    // timing vars
    long dur_cqrrp        = 0;
    long dur_cqrrp_qp3    = 0;
    long dur_hqrrp        = 0;
    long dur_hqrrp_geqrf  = 0;
    long dur_hqrrp_cholqr = 0;
    long dur_geqrf        = 0;
    long dur_geqp3        = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        printf("\nITERATION %d\n", i);
        
        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        printf("TOTAL TIME FOR GEQRF %ld\n", dur_geqrf);

        // Making sure the states are unchanged
        state_gen = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen, 0);
        
        // Testing CQRRP - best setup
        CQRRP_blocked.use_qp3 = false;
        auto start_cqrrp = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrp = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();
        printf("TOTAL TIME FOR CQRRP %ld\n", dur_cqrrp);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen, 0);

        // Testing CQRRP - using QP3
        CQRRP_blocked.use_qp3 = true;
        auto start_cqrrp_qp3 = high_resolution_clock::now();
        CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
        auto stop_cqrrp_qp3 = high_resolution_clock::now();
        CQRRP_blocked.use_qp3 = false;
        dur_cqrrp_qp3 = duration_cast<microseconds>(stop_cqrrp_qp3 - start_cqrrp_qp3).count();
        printf("TOTAL TIME FOR CQRRP WITH GEQP3 %ld\n", dur_cqrrp_qp3);

        // Making sure the states are unchanged
        state_gen = state;
        state_alg = state;
        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen, 1);
        
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
        data_regen(m_info, all_data, state_gen, 1);
        
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
        data_regen(m_info, all_data, state_gen, 1);

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
        data_regen(m_info, all_data, state_gen, 0);

        if ((i <= 2) && (b_sz == 256)) {
            // Testing GEQP3
            auto start_geqp3 = high_resolution_clock::now();
            lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
            auto stop_geqp3 = high_resolution_clock::now();
            dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
            printf("TOTAL TIME FOR GEQP3 %ld\n", dur_geqp3);

            state_gen = state;
            data_regen(m_info, all_data, state_gen, 0);
        }
        
        std::ofstream file(output_filename, std::ios::app);
        file << dur_cqrrp << ",  " << dur_cqrrp_qp3 << ",  " << dur_hqrrp << ",  " << dur_hqrrp_geqrf << ",  " << dur_hqrrp_cholqr << ",  " << dur_geqrf << ",  " << dur_geqp3 << ",\n";
    }
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
    double d_factor    = 1.0;
    int64_t b_sz_start = 256;
    int64_t b_sz_end   = 2048;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "ICQRRP_time_raw_rows_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz_start)
                                    + "_b_sz_end_"     + std::to_string(b_sz_end)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat";
    for (;b_sz_start <= b_sz_end; b_sz_start *= 2) {
        call_all_algs(m_info, numruns, b_sz_start, all_data, state_constant, output_filename);
    }
}
#endif
