/*
CQRRPT speed comparison benchmark - runs:
    1. CQRRPT
    2. GEQR
    3. GEQRF
    4. GEQP3
    5. GEQPT
    6. SCHOLQR
for a matrix with fixed number of rows and a varying number of columns.
Records the best timing, saves that into a file.
*/
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_bench_utils.hh"

#include <RandBLAS.hh>
#include <fstream>

using Subroutines = RandLAPACK::CQRRPTSubroutines;
using RandLAPACK::bench::QR_benchmark_data;
using RandLAPACK::bench::data_regen;

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
    RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT_default(true, tol);
    CQRRPT_default.nnz = 4;

    RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT_hqrrp(true, tol);
    CQRRPT_hqrrp.nnz = 4;
    CQRRPT_hqrrp.qrcp = Subroutines::QRCP::hqrrp;

    RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT_bqrrp(true, tol);
    CQRRPT_bqrrp.nnz = 4;
    CQRRPT_bqrrp.qrcp = Subroutines::QRCP::bqrrp;

    // timing vars
    long dur_cqrrpt_default = 0;
    long dur_cqrrpt_hqrrp   = 0;
    long dur_cqrrpt_bqrrp   = 0;
    long dur_geqp3          = 0;
    long dur_geqr           = 0;
    long dur_geqpt          = 0;
    long dur_geqrf          = 0;
    long dur_scholqr        = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        std::cout << "\nITERATION " << i << ", N_SZ " << n << "\n";
        // Testing GEQP3
        auto start_geqp3 = steady_clock::now();
        lapack::geqp3(m, n, all_data.A, m, all_data.J, all_data.tau);
        auto stop_geqp3 = steady_clock::now();
        dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
        std::cout << "TOTAL TIME FOR GEQP3 " << dur_geqp3 << "\n";

        state_gen = state;
        data_regen(m_info, all_data, state_gen, false);

        // Testing GEQRF
        auto start_geqrf = steady_clock::now();
        lapack::geqrf(m, n, all_data.A, m, all_data.tau);
        auto stop_geqrf = steady_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        std::cout << "TOTAL TIME FOR GEQRF " << dur_geqrf << "\n";

        state_gen = state;
        data_regen(m_info, all_data, state_gen, false);

        // Testing CQRRPT default
        auto start_cqrrp_default = steady_clock::now();
        CQRRPT_default.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state_alg);
        auto stop_cqrrp_default = steady_clock::now();
        dur_cqrrpt_default = duration_cast<microseconds>(stop_cqrrp_default - start_cqrrp_default).count();
        std::cout << "TOTAL TIME FOR CQRRPT default " << dur_cqrrpt_default << "\n";

        state_gen = state;
        state_alg = state;
        data_regen(m_info, all_data, state_gen, false);

        // Testing CQRRPT hqrrp
        auto start_cqrrp_hqrrp = steady_clock::now();
        CQRRPT_hqrrp.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state_alg);
        auto stop_cqrrp_hqrrp = steady_clock::now();
        dur_cqrrpt_hqrrp = duration_cast<microseconds>(stop_cqrrp_hqrrp - start_cqrrp_hqrrp).count();
        std::cout << "TOTAL TIME FOR CQRRPT hqrrp " << dur_cqrrpt_hqrrp << "\n";

        state_gen = state;
        state_alg = state;
        data_regen(m_info, all_data, state_gen, false);

        // Testing CQRRPT bqrrp
        auto start_cqrrp_bqrrp = steady_clock::now();
        CQRRPT_bqrrp.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state_alg);
        auto stop_cqrrp_bqrrp = steady_clock::now();
        dur_cqrrpt_bqrrp = duration_cast<microseconds>(stop_cqrrp_bqrrp - start_cqrrp_bqrrp).count();
        std::cout << "TOTAL TIME FOR CQRRPT bqrrp " << dur_cqrrpt_bqrrp << "\n";

        state_gen = state;
        state_alg = state;
        data_regen(m_info, all_data, state_gen, false);

        // Testing SCHOLQR3
        auto start_scholqr = steady_clock::now();
        //--------------------------------------------------------------------------------------------------------------------------//
        T norm_A = lapack::lange(Norm::Fro, m, n, all_data.A, m);
        T shift = 11 * std::numeric_limits<T>::epsilon() * n * std::pow(norm_A, 2);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A, m, 0.0, all_data.R, n);
        for (int i = 0; i < n; ++i)
            all_data.R[i * (n + 1)] += shift;
        lapack::potrf(Uplo::Upper, n, all_data.R, n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R, n, all_data.A, m);
        // CholeskyQR2
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A, m, 0.0, all_data.R, n);
        lapack::potrf(Uplo::Upper, n, all_data.R, n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R, n, all_data.A, m);
        // CholeskyQR3
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A, m, 0.0, all_data.R, n);
        lapack::potrf(Uplo::Upper, n, all_data.R, n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R, n, all_data.A, m);
        //--------------------------------------------------------------------------------------------------------------------------//
        auto stop_scholqr = steady_clock::now();
        dur_scholqr = duration_cast<microseconds>(stop_scholqr - start_scholqr).count();
        std::cout << "TOTAL TIME FOR SCHOLQR3 " << dur_scholqr << "\n";

        auto state_gen = state;
        data_regen(m_info, all_data, state_gen, false);

        // Testing GEQR + GEQPT
        auto start_geqpt = steady_clock::now();
        auto start_geqr  = steady_clock::now();
        // GEQR(A) part
        lapack::geqr(m, n, all_data.A, m,  all_data.tau, -1);
        int64_t tsize = (int64_t) all_data.tau[0]; 
        RandLAPACK::bench::grow(all_data.tau, all_data.tau_sz, tsize);
        lapack::geqr(m, n, all_data.A, m, all_data.tau, tsize);

        auto stop_geqr = steady_clock::now();
        dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();
        std::cout << "TOTAL TIME FOR GEQR " << dur_geqr << "\n";

        // GEQP3(R) part
        lapack::lacpy(MatrixType::Upper, n, n, all_data.A, m, all_data.R, n);
        lapack::geqp3(n, n, all_data.R, n, all_data.J, all_data.tau);
        auto stop_geqpt = steady_clock::now();
        dur_geqpt = duration_cast<microseconds>(stop_geqpt - start_geqpt).count();
        std::cout << "TOTAL TIME FOR GEQPT " << dur_geqpt << "\n";

        state_gen = state;
        data_regen(m_info, all_data, state_gen, false);

        std::ofstream file(output_filename, std::ios::app);
        RandLAPACK::bench::write_row(file, dur_cqrrpt_default, dur_cqrrpt_hqrrp, dur_cqrrpt_bqrrp, dur_geqpt, dur_geqrf, dur_geqr, dur_geqp3, dur_scholqr);
    }
}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <num_runs> <num_rows> <column_sizes>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m = std::stol(argv[2]);
    std::vector<int64_t> n_sz(argc - 3);
    RandLAPACK::bench::parse_sizes(argc, argv, 3, n_sz.data());
    // Save elements in string for logging purposes
    std::string n_sz_string = RandLAPACK::bench::sizes_string(n_sz.data(), n_sz.size());

    double d_factor     = 1.25;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = std::stol(argv[1]);

    // Allocate basic workspace
    int64_t n_max = *std::max_element(n_sz.begin(), n_sz.end());
    QR_benchmark_data<double> all_data(m, n_max, {.d_factor = d_factor, .tol = tol, .R = true});
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_max, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Declare a data file
    std::string output_filename = RandLAPACK::util::get_current_date_time<double>() + "_CQRRPT_speed_comparisons_" 
                                                                 + "_num_info_lines_" + std::to_string(7) +
                                                                   ".txt";

    std::ofstream file(output_filename, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the CQRRPT speed comparison benchmark, recording the time it takes to perform CQRRPT and alternative QR and QRCP factorizations."
              "\nFile format: 6 columns, containing time for each algorithm: CQRRPT, GEQPT, GEQRF, GEQR, GEQP3, SCHOLQR3;"
              "               rows correspond to CQRRPT runs with column sizes varying as specified, with numruns repititions of each column size"
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
    RandLAPACK::bench::write_total_time(file, start_time_all);
}
