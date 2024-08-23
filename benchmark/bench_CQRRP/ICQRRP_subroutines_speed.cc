/*
QR speed comparison benchmark - runs:
    1. GEQRF
    2. GEQR
    3. GEQR+UNGQR
    4. CholQR
for a matrix with fixed number of rows and a varying number of columns.
*/
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> A1;
    std::vector<T> A2;
    std::vector<T> B;
    std::vector<T> B1;
    std::vector<T> B2;
    std::vector<T> C;
    std::vector<T> R;
    std::vector<T> Q;
    std::vector<T> A_trans;
    std::vector<T> R_trans;
    std::vector<T> T_mat;
    std::vector<T> tau;
    std::vector<T> D;
    std::vector<int64_t> J;
    std::vector<int64_t> J_lu;

    benchmark_data(int64_t m, int64_t n) :
    A(m * n, 0.0),
    A1(m * n, 0.0),
    A2(m * m, 0.0),
    A_gemqrt(m * n, 0.0),
    B(m * n, 0.0),
    B1(m * n, 0.0),
    B2(m * n, 0.0),
    C(m * n, 0.0),
    R(n * n, 0.0),
    Q(m * n, 0.0),
    A_trans(m * n, 0.0),
    R_trans(n * n, 0.0),
    T_mat(n * n, 0.0),
    T_gemqrt(n * n, 0.0),
    tau(n, 0.0),
    D(n, 0.0),
    J(m, 0.0),
    J_lu(m, 0.0)
    {
        row = m;
        col = n;
        tolerance = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state_const,
                                        RandBLAS::RNGState<RNG> &state_const_B,
                                        int bench_type) {

    auto state   = state_const;
    auto state_B = state_const_B;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    
    if(bench_type == 1) {
        std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
        std::fill(all_data.J.begin(), all_data.J.end(), 0.0);
        std::fill(all_data.J_lu.begin(), all_data.J_lu.end(), 0.0);
        std::fill(all_data.A_trans.begin(), all_data.A_trans.end(), 0.0);
    } else if(bench_type == 2) {
        std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
        std::fill(all_data.T_mat.begin(), all_data.T_mat.end(), 0.0);
        std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
        std::fill(all_data.D.begin(), all_data.D.end(), 0.0);
    } else if(bench_type == 3) {
        RandLAPACK::gen::mat_gen(m_info, all_data.B.data(), state_B);
        std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
        std::fill(all_data.T_mat.begin(), all_data.T_mat.end(), 0.0);
        std::fill(all_data.D.begin(), all_data.D.end(), 0.0);
        std::fill(all_data.A1.begin(), all_data.A1.end(), 0.0);
        std::fill(all_data.A2.begin(), all_data.A2.end(), 0.0);
        std::fill(all_data.B1.begin(), all_data.B1.end(), 0.0);
        std::fill(all_data.B2.begin(), all_data.B2.end(), 0.0);
        std::fill(all_data.C.begin(), all_data.C.end(), 0.0);
    }
}

template <typename T, typename RNG>
static void call_wide_qrcp(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    RandBLAS::RNGState<RNG> &state_B,
    std::string output_filename) {

    auto m = all_data.row;  
    auto tol = all_data.tolerance;

    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, tol);
    CQRRPT.nnz = 4;
    CQRRPT.num_threads = 4;

    // timing vars
    long dur_geqp3  = 0;
    long dur_luqr   = 0;
    long dur_cqrrpt = 0;
 
    // Making sure the states are unchanged
    auto state_alg = state;

    int i, j = 0;
    for (i = 0; i < numruns; ++i) {
        printf("Wide QRCP iteration %d; m==%d start.\n", i, n);
        // Testing GEQP3
        auto start_geqp3 = high_resolution_clock::now();
        lapack::geqp3(n, m, all_data.A.data(), n, all_data.J.data(), all_data.tau.data());
        auto stop_geqp3 = high_resolution_clock::now();
        dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
        data_regen(m_info, all_data, state, state_B, 1);


        // Testing CQRRPT
        auto start_cqrrpt = high_resolution_clock::now();
        RandLAPACK::util::transposition(n, m, all_data.A.data(), n, all_data.A_trans.data(), m, 0);

        CQRRPT.call(m, n, all_data.A_trans.data(), m, all_data.R_trans.data(), n, all_data.J.data(), 1.25, state_alg);
        RandLAPACK::util::transposition(n, n, all_data.R_trans.data(), n, all_data.R.data(), n, 0);
        auto stop_cqrrpt = high_resolution_clock::now();
        dur_cqrrpt = duration_cast<microseconds>(stop_cqrrpt - start_cqrrpt).count();
        state_alg = state;
        data_regen(m_info, all_data, state, state_B, 1);

        // Testing LUQR
        auto start_luqr = high_resolution_clock::now();
            // Perform pivoted LU on A_sk', follow it up by unpivoted QR on a permuted A_sk.
            // Get a transpose of A_sk 
            RandLAPACK::util::transposition(n, m, all_data.A.data(), n, all_data.A_trans.data(), m, 0);
            // Perform a row-pivoted LU on a transpose of A_sk
            lapack::getrf(m, n, all_data.A_trans.data(), m, all_data.J_lu.data());
            // Fill the pivot vector, apply swaps found via lu on A_sk'.
            std::iota(&(all_data.J)[0], &(all_data.J)[m], 1);
            for (j = 0; j < n; ++j) {
                int tmp = all_data.J[all_data.J_lu[j] - 1];
                all_data.J[all_data.J_lu[j] - 1] = all_data.J[j];
                all_data.J[j] = tmp;
            }
            // Apply pivots to A_sk
            RandLAPACK::util::col_swap(n, m, m, all_data.A.data(), n, all_data.J);
            // Perform an unpivoted QR on A_sk
            lapack::geqrf(n, m, all_data.A.data(), n, all_data.tau.data());
        auto stop_luqr = high_resolution_clock::now();
        dur_luqr = duration_cast<microseconds>(stop_luqr - start_luqr).count();
        data_regen(m_info, all_data, state, state_B, 1);
    
        std::ofstream file(output_filename, std::ios::app);
        file << n << ",  " << dur_geqp3 << ",  " << dur_luqr << ",  " << dur_cqrrpt << ",\n";
    }
}


template <typename T, typename RNG>
static void call_tsqr(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m   = all_data.row;
    auto tol = all_data.tolerance;
    int64_t tsize = 0;

    // timing vars
    long dur_geqrf       = 0;
    long dur_geqr        = 0;
    long dur_cholqr      = 0;
    long dur_cholqr_orhr = 0;

    for (int i = 0; i < numruns; ++i) {
        printf("TSQR iteration %d; n==%d start.\n", i, n);
        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        data_regen(m_info, all_data, state, state, 2);

#if !defined(__APPLE__)
        // Testing GEQR
        auto start_geqr = high_resolution_clock::now();
        lapack::geqr(m, n, all_data.A.data(), m,  all_data.tau.data(), -1);
        tsize = (int64_t) all_data.tau[0]; 
        all_data.tau.resize(tsize);
        lapack::geqr(m, n, all_data.A.data(), m, all_data.tau.data(), tsize);
        auto stop_geqr = high_resolution_clock::now();
        dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();
        data_regen(m_info, all_data, state, state, 2);
#endif
        // Testing CholQR
        auto start_cholqr = high_resolution_clock::now();
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, all_data.A.data(), m, (T) 0.0, all_data.R.data(), n);
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.R.data(), n, all_data.A.data(), m);
        auto stop_cholqr = high_resolution_clock::now();
        dur_cholqr = duration_cast<microseconds>(stop_cholqr - start_cholqr).count();
        lapack::orhr_col(m, n, n, all_data.A.data(), m, all_data.T_mat.data(), n, all_data.D.data());
        auto stop_cholqr_orhr = high_resolution_clock::now();
        dur_cholqr_orhr = duration_cast<microseconds>(stop_cholqr_orhr - start_cholqr).count();
        data_regen(m_info, all_data, state, state, 2);
    
        std::ofstream file(output_filename, std::ios::app);
        file << n << ",  " << dur_geqrf << ",  " << dur_geqr << ",  " << dur_cholqr <<  ",  " << dur_cholqr_orhr << ",\n";
    }
}

template <typename T, typename RNG>
static void call_apply_q(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    int64_t gemqrt_nb,
    benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m   = all_data.row;
    auto tol = all_data.tolerance;
    int64_t gemqrt_nb = n;

    int64_t tsize = 0;

    // timing vars
    long dur_ormqr  = 0;
    long dur_gemqrt = 0;
    long dur_gemm   = 0;

    int i, j = 0;
    for (i = 0; i < numruns; ++i) {
        printf("Apply Q iteration %d; n==%d start.\n", i, n);
        // Performing CholQR
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, all_data.A.data(), m, (T) 0.0, all_data.R.data(), n);
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.R.data(), n, all_data.A.data(), m);
        
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_gemqrt.data(), m);
        lapack::orhr_col(m, n, n, all_data.A.data(), m, all_data.T_mat.data(), n, all_data.D.data());
        lapack::orhr_col(m, n, gemqrt_nb, all_data.A_gemqrt.data(), m, all_data.T_gemqrt.data(), n, all_data.D.data());
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A2.data(), m);

        for(j = 0; j < n; ++j)
            all_data.tau[j] = all_data.T_mat[(n + 1) * j];

        auto start_ormqr = high_resolution_clock::now();
        lapack::ormqr(Side::Left, Op::NoTrans, m, n, n, all_data.A.data(), m, all_data.tau.data(), all_data.B.data(), m);
        auto stop_ormqr = high_resolution_clock::now();
        dur_ormqr = duration_cast<microseconds>(stop_ormqr - start_ormqr).count();

        auto start_gemqrt = high_resolution_clock::now();
        lapack::gemqrt(Side::Left, Op::NoTrans, m, n, n, gemqrt_nb, all_data.A_gemqrt.data(), m, all_data.T_gemqrt.data(), n, all_data.B1.data(), m);
        auto stop_gemqrt = high_resolution_clock::now();
        dur_gemqrt = duration_cast<microseconds>(stop_gemqrt - start_gemqrt).count();

        auto start_gemm = high_resolution_clock::now();
        lapack::ungqr(m, m, n, all_data.A2.data(), m, all_data.tau.data());
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, m, 1.0, all_data.A2.data(), m, all_data.B2.data(), m, 0.0, all_data.C.data(), m);
        auto stop_gemm = high_resolution_clock::now();
        dur_gemm = duration_cast<microseconds>(stop_gemm - start_gemm).count();

        data_regen(m_info, all_data, state, state, 3);
    
        std::ofstream file(output_filename, std::ios::app);
        file << dur_ormqr << ",  " << dur_gemqrt << ",  " << dur_gemm << ",\n";
    }
}

int main() {
    int64_t i = 0;
    // Declare parameters
    int64_t m             = std::pow(2, 16);
    int64_t n_start       = std::pow(2, 8);
    int64_t n_stop        = std::pow(2, 11);
    auto state            = RandBLAS::RNGState();
    auto state_B          = RandBLAS::RNGState();
    auto state_constant   = state;
    auto state_constant_B = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    benchmark_data<double> all_data(m, n_stop);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    RandLAPACK::gen::mat_gen(m_info, all_data.B.data(), state_B);

    // Declare a data file
    std::string output_filename = "ICQRRP_subroutines_speed_comp_"              + std::to_string(m)
                                      + "_col_start_"    + std::to_string(n_start)
                                      + "_col_stop_"     + std::to_string(n_stop)
                                      + ".dat"; 
    std::ofstream file(output_filename, std::ios::app);

    //file << "GEQP3  LUQR  CQRRPT\n";
    for (i = n_start; i <= n_stop; i *= 2)
        call_wide_qrcp(m_info, numruns, i, all_data, state, state_B, output_filename);

    //file << "GEQRF  GEQR  CHOLQR  CHOLQR_ORHR\n";
    for (i = n_start; i <= n_stop; i *= 2)
        call_tsqr(m_info, numruns, i, all_data, state, output_filename);

    //file << "ORMQR  GEMQRT  GEMM\n";
    for (i = n_start; i <= n_stop; i *= 2)
        call_apply_q(m_info, numruns, i, all_data, state, output_filename);
}
