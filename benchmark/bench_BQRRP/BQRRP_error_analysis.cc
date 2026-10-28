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

using RandLAPACK::bench::QR_benchmark_data;
using RandLAPACK::bench::data_regen;

template <typename T, typename RNG>
static void BQRRP_benchmark_run(
    RandLAPACK::gen::mat_gen_info<T> mat_info,
    int64_t b_sz,
    std::string alg_to_run,
    int num_runs,
    std::string qr_tall,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m = all_data.row;
    auto n = all_data.col;

    auto state_alg = state;
    auto state_gen = state;

    // Output error vector
    std::vector<T> error_output(4, 0.0);

    // Additional params setup.
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(false, b_sz);
    RandLAPACK::bench::set_bqrrp_qr_tall(BQRRP, qr_tall);

    // Parse through all the needed matrix types
    for (int i = 0; i < num_runs; ++i) {
        // Clear and re-generate data
        state_gen = state;
        data_regen(mat_info, all_data, state_gen);

        if(alg_to_run == "bqrrp") {
            std::cout << "BQRRP run " << i << " with columns_size " << n << "\n";
            // State_alg changes at every iteration, consequently, we have different sketches
            BQRRP.call(m, n, all_data.A, m, 1.0, all_data.tau, all_data.J, state_alg);
            all_data.rank = BQRRP.rank;
        } else {
            std::cout << "GEQP3\n";
            lapack::geqp3(m, n, all_data.A, m, all_data.J, all_data.tau);
        }

        // Copying the matrix R into the buffer
        lapack::lacpy(MatrixType::Upper, n, n, all_data.A, m, all_data.R, n);
        // Extracting an explicit Q-factor 
        lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.A, m, all_data.tau);
        // Permuting the columns of the copies of the original matrix A
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1, m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2, m, all_data.J);
        
        auto [reconstruction_error, orth_loss] = RandLAPACK::bench::qr_error_check(all_data, n, error_output.data());
        std::cout << "REL NORM OF AP - QR:    " << std::scientific << std::setprecision(14) << reconstruction_error << "\n";
        std::cout << "FRO NORM OF (Q'Q - I):  " << std::scientific << std::setprecision(14) << orth_loss << "\n\n";
    }

    std::ofstream file(output_filename, std::ios::out | std::ios::app);
    RandLAPACK::bench::write_error_row(file, error_output.data(), num_runs);
}

int main(int argc, char *argv[]) {
    if (argc < 7) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <alg_to_run> <num_runs> <num_rows> <num_cols> <block_size>" << std::endl;
        return 1;
    }

    // Declare parameters
    std::string path       = argv[1];
    std::string alg_to_run = argv[2];
    int num_runs           = std::stol(argv[3]);
    int64_t m              = std::stol(argv[4]);
    int64_t n              = std::stol(argv[5]);
    std::vector<int64_t> b_sz(argc - 6);
    RandLAPACK::bench::parse_sizes(argc, argv, 6, b_sz.data());
    // Save elements in string for logging purposes
    std::string b_sz_string = RandLAPACK::bench::sizes_string(b_sz.data(), b_sz.size());
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n, {.R = true, .copies = true});
    
    // Set the input matrices: polynomial, staircase, spiked and Kahan
    auto tests_info = RandLAPACK::bench::qr_test_matrices<double>(m, n);

    // Declare a data file
    std::string output_filename = "_BQRRP_error_analysis_num_info_lines_" + std::to_string(5) + ".txt";
    path = RandLAPACK::bench::output_path(argv[1], output_filename);
    std::ofstream file(path, std::ios::out | std::ios::app);
    // Writing important data into file
    file << "Description: Results from the " + alg_to_run + " error analysis; putput rows capture results per given matrix type, columns capture results per error type."
            "\nAt the moment, i test polynomial, staircase and spiked matrices with reconstructiuon error, max column norm error and orthogonality loss."
            "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
            "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
            "\nAdditional parameters: BQRRP block sizes: " + b_sz_string +
            "\n";
    file.flush();

    // Call the benchmark
    int i = 0, j = 0; 
    for (; i < b_sz.size(); ++i) {
        // Go through all matrix types
        for (; j < tests_info.size(); ++j) {
            std::cout << "/--------------------------------------RUNS ON MAT TYPE " << j << "--------------------------------------/\n";
            BQRRP_benchmark_run(tests_info[j], b_sz[i], alg_to_run, num_runs, "cholqr", all_data, state_constant, path);
        }
        j = 0;
    }
}
