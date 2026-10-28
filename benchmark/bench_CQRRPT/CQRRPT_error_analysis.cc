/*
Performs computations in order to assess the pivot quality of BQRRP.
The setup is described in detail in Section 4 of The arXiv version 2 CQRRPT (https://arxiv.org/pdf/2311.08316.pdf) paper.
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
static void CQRRPT_benchmark_run(
    RandLAPACK::gen::mat_gen_info<T> mat_info,
    T atol,
    int64_t col_sz,
    std::string alg_to_run,
    int num_runs,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m = all_data.row;
    auto n = col_sz;

    auto state_alg = state;
    auto state_gen = state;

    // Output error vector
    std::vector<T> error_output(4, 0.0);

    // Additional params setup.
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, atol);
    CQRRPT.nnz = 4;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;
    double d_factor = all_data.sampling_factor;

    // Parse through all the needed matrix types
    for (int i = 0; i < num_runs; ++i) {
        // Clear and re-generate data
        state_gen = state;
        data_regen(mat_info, all_data, state_gen);

        if(alg_to_run == "cqrrpt") {
            std::cout << "CQRRPT run " << i << " with columns_size " << n << "\n";
            // State_alg changes at every iteration, consequently, we have different sketches
            CQRRPT.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state_alg);
            all_data.rank = CQRRPT.rank;
        } else {
            std::cout << "GEQP3 run with columns_size " << n << "\n";
            lapack::geqp3(m, n, all_data.A, m, all_data.J, all_data.tau);
            // Copying the matrix R into the buffer
            lapack::lacpy(MatrixType::Upper, n, n, all_data.A, m, all_data.R, n);
            // Extracting an explicit Q-factor 
            lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.A, m, all_data.tau);
            all_data.rank = n;
        }

        // Permuting the columns of the copies of the original matrix A
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1, m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2, m, all_data.J);
    
        auto [reconstruction_error, orth_loss] = RandLAPACK::bench::qr_error_check(all_data, col_sz, error_output.data());
        std::cout << "REL NORM OF AP - QR:    " << std::scientific << std::setw(14) << reconstruction_error << "\n";
        std::cout << "FRO NORM OF (Q'Q - I):  " << std::scientific << std::setw(14) << orth_loss << "\n\n";
    }

    std::ofstream file(output_filename, std::ios::out | std::ios::app);
    RandLAPACK::bench::write_error_row(file, error_output.data(), num_runs);
}

int main(int argc, char *argv[]) {
    if (argc < 6) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <alg_to_run> <num_runs> <num_rows> <column_sizes>" << std::endl;
        return 1;
    }

    // Declare parameters
    std::string path       = argv[1];
    std::string alg_to_run = argv[2];
    int num_runs           = std::stol(argv[3]);
    int64_t m              = std::stol(argv[4]);
    double d_factor        = 1.25;
    std::vector<int64_t> col_sz(argc - 5);
    RandLAPACK::bench::parse_sizes(argc, argv, 5, col_sz.data());
    // Maximum value from the vector will serve as n
    int64_t n = *std::max_element(col_sz.begin(), col_sz.end());;
    // Save elements in string for logging purposes
    std::string col_sz_string = RandLAPACK::bench::sizes_string(col_sz.data(), col_sz.size());
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    double atol         = std::pow(std::numeric_limits<double>::epsilon(), 0.75);

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, m, {.d_factor = d_factor, .R = true, .copies = true});
    
    // Set the input matrices: polynomial, staircase, spiked and Kahan
    auto tests_info = RandLAPACK::bench::qr_test_matrices<double>(m, n);

    // Declare a data file
    std::string output_filename = "_CQRRPT_error_analysis_num_info_lines_" + std::to_string(4) + ".txt";
    path = RandLAPACK::bench::output_path(argv[1], output_filename);
    std::ofstream file(path, std::ios::out | std::ios::app);
    // Writing important data into file
    file << "Description: Results from the " + alg_to_run +" error analysis; putput rows capture results per given matrix type, columns capture results per error type."
            "\nAt the moment, i test polynomial, staircase and spiked matrices with reconstructiuon error, max column norm error and orthogonality loss."
            "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
            "\nInput size:"       + std::to_string(m) + " by " + col_sz_string +
            "\n";
    file.flush();

    // Call the benchmark
    int i = 0, j = 0; 
    for (; i < col_sz.size(); ++i) {
        // Go through all matrix types
        for (; j < tests_info.size(); ++j) {
            std::cout << "/--------------------------------------RUNS ON MAT TYPE " << j << "--------------------------------------/\n";
            CQRRPT_benchmark_run(tests_info[j], atol, col_sz[i], alg_to_run, num_runs, all_data, state_constant, path);
        }
        j = 0;
    }
}
