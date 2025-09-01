#if defined(__APPLE__)
int main() {return 0;}
#else
/*
Performs computations in order to assess the pivot quality of BQRRP.
The setup is described in detail in Section 4 of The arXiv version 2 CQRRPT (https://arxiv.org/pdf/2311.08316.pdf) paper.
*/
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
    std::vector<T> R;
    std::vector<T> tau;
    std::vector<int64_t> J;
    std::vector<T> A_cpy1;
    std::vector<T> A_cpy2;
    std::vector<T> I_ref;

    QR_benchmark_data(int64_t m, int64_t n) :
    A(m * n, 0.0),
    R(n * n, 0.0),
    tau(n, 0.0),
    J(n, 0),
    A_cpy1(m * n, 0.0),
    A_cpy2(m * n, 0.0),
    I_ref(n * n, 0.0) 
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
    
    std::fill(all_data.A.begin(), all_data.A.end(), 0.0);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    lapack::lacpy(MatrixType::General, all_data.row, all_data.col, all_data.A.data(), all_data.row, all_data.A_cpy1.data(), all_data.row);
    lapack::lacpy(MatrixType::General, all_data.row, all_data.col, all_data.A.data(), all_data.row, all_data.A_cpy2.data(), all_data.row);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
    std::fill(all_data.I_ref.begin(), all_data.I_ref.end(), 0.0);
}


template <typename T>
static void
error_check(QR_benchmark_data<T> &all_data, 
            T atol, 
            std::vector<T> &error_output) {

    auto m = all_data.row;
    auto n = all_data.col;
    auto k = n;

    RandLAPACK::util::upsize(k * k, all_data.I_ref);
    RandLAPACK::util::eye(k, k, all_data.I_ref);

    T* A_dat           = all_data.A_cpy1.data();
    T const* A_cpy_dat = all_data.A_cpy2.data();
    T const* Q_dat     = all_data.A.data();
    T const* R_dat     = all_data.R.data();
    T* I_ref_dat       = all_data.I_ref.data();

    // Get the norm of the input matrix
    T norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, I_ref_dat, k);
    T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

    // A - QR
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, k, -1.0, A_dat, m);
    
    // Implementing max col norm metric
    T max_col_norm = 0.0;
    T col_norm = 0.0;
    int max_idx = 0;
    for(int i = 0; i < n; ++i) {
        col_norm = blas::nrm2(m, &A_dat[m * i], 1);
        if(max_col_norm < col_norm) {
            max_col_norm = col_norm;
            max_idx = i;
        }
    }
    T col_norm_A = blas::nrm2(n, &A_cpy_dat[m * max_idx], 1);
    T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);

    T reconstruction_error = norm_AQR / norm_A;
    T max_col_norm_error   = max_col_norm / col_norm_A;
    T orth_loss            = norm_0 / std::sqrt((T) n);
    
    printf("REL NORM OF AP - QR:    %14e\n",   reconstruction_error);
    printf("MAX COL NORM METRIC:    %14e\n",   max_col_norm_error);
    printf("FRO NORM OF (Q'Q - I):  %14e\n\n", orth_loss);

    // For computing average reconstruction error across all runs
    error_output[0] += reconstruction_error;
    // For capturing maximal reconstruction error across all runs
    if (reconstruction_error > error_output[1]) { error_output[1] = reconstruction_error;}

    // For computing average orth error across all runs
    error_output[2] += orth_loss;
    // For capturing maximal orth error across all runs
    if (orth_loss > error_output[3]) { error_output[3] = orth_loss;}
}

template <typename T, typename RNG>
static void BQRRP_benchmark_run(
    RandLAPACK::gen::mat_gen_info<T> mat_info,
    T atol,
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

    T reconstruction_error = 0;
    T orthogonality_loss   = 0;

    // Output error vector
    std::vector<T> error_output(4, 0.0);

    // Additional params setup.
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(false, b_sz);
    if(qr_tall == "geqrt") {
        BQRRP.qr_tall       = Subroutines::QRTall::geqrt;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::gemqrt;
    } else if (qr_tall == "cholqr") {
        BQRRP.qr_tall       = Subroutines::QRTall::cholqr;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
    } else {
        BQRRP.qr_tall       = Subroutines::QRTall::geqrf;
        BQRRP.apply_trans_q = Subroutines::ApplyTransQ::ormqr;
    }

    // Parse through all the needed matrix types
    for (int i = 0; i < num_runs; ++i) {
        // Clear and re-generate data
        state_gen = state;
        data_regen(mat_info, all_data, state_gen);

        if(alg_to_run == "bqrrp") {
            printf("BQRRP run %d with columns_size %ld\n", i, n);
            state_alg = state;
            BQRRP.call(m, n, all_data.A.data(), m, 1.0, all_data.tau.data(), all_data.J.data(), state_alg);
        } else {
            printf("GEQP3\n");
            lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
        }

        // Copying the matrix R into the buffer
        lapack::lacpy(MatrixType::Upper, n, n, all_data.A.data(), m, all_data.R.data(), n);
        // Extracting an explicit Q-factor 
        lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.A.data(), m, all_data.tau.data());
        // Permuting the columns of the copies of the original matrix A
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);
        
        error_check<T>(all_data, atol, error_output);
    }

    std::ofstream file(output_filename, std::ios::out | std::ios::app);
    T avg_reconstruction_error = error_output[0] / num_runs;
    T avg_orthogonality_loss   = error_output[0] / num_runs;
    file << avg_reconstruction_error << ",  " << error_output[1] - avg_reconstruction_error << ",  " << avg_reconstruction_error << ",  " << error_output[3] - avg_orthogonality_loss <<  ",\n";
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
    std::vector<int64_t> b_sz;
    for (int i = 0; i < argc-6; ++i)
        b_sz.push_back(std::stoi(argv[i + 6]));
    // Save elements in string for logging purposes
    std::ostringstream oss;
    for (const auto &val : b_sz)
        oss << val << ", ";
    std::string b_sz_string = oss.str();
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    double atol         = std::pow(std::numeric_limits<double>::epsilon(), 0.75);

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n);
    
    // Set the input matrices
    // Polynomial matrix
    RandLAPACK::gen::mat_gen_info<double> m_info_poly(m, n, RandLAPACK::gen::spiked);
    m_info_poly.cond_num = std::pow(10, 10);
    m_info_poly.exponent = 2.0;
    // Matrix with staircase spectrum
    RandLAPACK::gen::mat_gen_info<double> m_info_stair(m, n, RandLAPACK::gen::spiked);
    m_info_stair.cond_num = std::pow(10, 10);
    // Matrix with spiked spectrum
    RandLAPACK::gen::mat_gen_info<double> m_info_spiked(m, n, RandLAPACK::gen::spiked);
    m_info_spiked.scaling = std::pow(10, 10);

    // Put all matrices info into an array
    std::vector<RandLAPACK::gen::mat_gen_info<double>> tests_info = {m_info_poly, m_info_stair, m_info_spiked};

    // Declare a data file
    std::string output_filename = "_BQRRP_error_analysis_num_info_lines_" + std::to_string(5) + ".txt";
    if (std::string(argv[1]) != ".") {
        path = std::string(argv[1]) + output_filename;
    } else {
        path = output_filename;
    }
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
    int max_iters = 0;
    alg_to_run == "bqrrp" ? max_iters = b_sz.size() : max_iters = 1;
    int i = 0, j = 0; 
    for (; i < b_sz.size(); ++i) {
        // Go through all matrix types
        for (; j < tests_info.size(); ++j) {
            printf("BQRRP ON MAT TYPE %ld\n", j);
            BQRRP_benchmark_run(tests_info[j], atol, b_sz[i], alg_to_run, num_runs, "cholqr", all_data, state_constant, output_filename);
        }
        j = 0;
    }
}
#endif