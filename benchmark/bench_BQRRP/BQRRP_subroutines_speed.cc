#if defined(__APPLE__)
int main() {return 0;}
#else
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
    std::vector<T> A_gemqrt;
    std::vector<T> B;
    std::vector<T> B1;
    std::vector<T> C;
    std::vector<T> R;
    std::vector<T> Q;
    std::vector<T> A_trans;
    std::vector<T> R_trans;
    std::vector<T> T_mat;
    std::vector<T> T_gemqrt;
    std::vector<T> tau;
    std::vector<T> D;
    std::vector<int64_t> J;
    std::vector<int64_t> J_lu;

    benchmark_data(int64_t m, int64_t n) :
    A(m * n, 0.0),
    A1(m * m, 0.0),
    A_gemqrt(m * n, 0.0),
    B(m * m, 0.0),
    B1(m * m, 0.0),
    C(m * m, 0.0),
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


// Ideally, we would wnat to use the function below to determine the suitable internal gemqrt block size.
/*
template <typename T>
void _LAPACK_ilaenv(
	int   ISPEC,
    char* NAME,
    char* OPTS,
    int   N1,
    int   N2,
    int   N3,
    int   N4 
){
    lapack_int ISPEC_ = (lapack_int) ISPEC;
    lapack_int N1_ = (lapack_int) N1;
    lapack_int N2_ = (lapack_int) N2;
    lapack_int N3_ = (lapack_int) N3;
    lapack_int N4_ = (lapack_int) N4;

    LAPACK_ilaenv( & ISPEC_, & NAME, & OPTS, 
        N1_, N2_, N3_, N4_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        //, 1
        #endif
        );
    return;
}
*/

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
        std::fill(all_data.B1.begin(), all_data.B1.end(), 0.0);
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
    std::string output_filename) {

    auto m = all_data.row;  
    auto tol = all_data.tolerance;

    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 4;

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
        auto start_geqp3 = steady_clock::now();
        lapack::geqp3(n, m, all_data.A.data(), n, all_data.J.data(), all_data.tau.data());
        auto stop_geqp3 = steady_clock::now();
        dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();
        data_regen(m_info, all_data, state, state, 1);

        // Testing LUQR
        auto start_luqr = steady_clock::now();
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
        auto stop_luqr = steady_clock::now();
        dur_luqr = duration_cast<microseconds>(stop_luqr - start_luqr).count();
        data_regen(m_info, all_data, state, state, 1);
    
        std::ofstream file(output_filename, std::ios::app);
        file << dur_geqp3 << ",  " << dur_luqr << ",\n";
    }
}


template <typename T, typename RNG>
static void call_tsqr(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    int64_t geqrt_nb_start,
    benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m   = all_data.row;
    auto tol = all_data.tolerance;
    int64_t tsize = 0;

    // timing vars
    long dur_geqrf             = 0;
    long dur_geqr              = 0;
    long dur_geqrt             = 0;
    long dur_cholqr            = 0;
    long dur_cholqr_precond    = 0;
    long dur_cholqr_house_rest = 0;
    long dur_cholqr_r_restore  = 0;

    // Imitating the QRCP on a sketch stage of BQRRP - needed to get a preconditioner
    T* S       = new T[n * m]();
    T* A_sk    = new T[n * n]();
    int64_t* J = new int64_t[n]();
    T* tau     = new T[n]();
    
    RandBLAS::DenseDist D(n, m);
    auto state_const = state;
    RandBLAS::fill_dense(D, S, state_const);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, m, 1.0, S, n, all_data.A.data(), m, 0.0, A_sk, n);
    lapack::geqp3(n, n, A_sk, n, J, tau);

    std::ofstream file(output_filename, std::ios::app);

    int64_t nb = 0;
    int i = 0;
    for (i = 0; i < numruns; ++i) {
        for(nb = geqrt_nb_start; nb <= n; nb *=2) {
            printf("TSQR iteration %d; n==%ld start.\n", i, n);

            auto start_geqrt = steady_clock::now();
            lapack::geqrt( m, n, nb, all_data.A.data(), m, all_data.T_mat.data(), n );
            auto stop_geqrt = steady_clock::now();
            dur_geqrt = duration_cast<microseconds>(stop_geqrt - start_geqrt).count();

            if(nb == geqrt_nb_start) {
                // Testing GEQRF
                auto start_geqrf = steady_clock::now();
                lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
                auto stop_geqrf = steady_clock::now();
                dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
                data_regen(m_info, all_data, state, state, 2);

                // Testing GEQR
                auto start_geqr = steady_clock::now();
                lapack::geqr(m, n, all_data.A.data(), m,  all_data.tau.data(), -1);
                tsize = (int64_t) all_data.tau[0]; 
                all_data.tau.resize(tsize);
                lapack::geqr(m, n, all_data.A.data(), m, all_data.tau.data(), tsize);
                auto stop_geqr = steady_clock::now();
                dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();
                data_regen(m_info, all_data, state, state, 2);

                // Testing CholQR
                auto start_precond = steady_clock::now();
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, A_sk, n, all_data.A.data(), m);
                auto stop_precond = steady_clock::now();
                dur_cholqr_precond = duration_cast<microseconds>(stop_precond - start_precond).count();
                auto start_cholqr = steady_clock::now();
                blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, all_data.A.data(), m, (T) 0.0, all_data.R.data(), n);
                lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.R.data(), n, all_data.A.data(), m);
                auto stop_cholqr = steady_clock::now();
                dur_cholqr = duration_cast<microseconds>(stop_cholqr - start_cholqr).count();
                auto start_orhr_col = steady_clock::now();
                lapack::orhr_col(m, n, n, all_data.A.data(), m, all_data.T_mat.data(), n, all_data.D.data());
                auto stop_cholqr_orhr = steady_clock::now();
                dur_cholqr_house_rest = duration_cast<microseconds>(stop_cholqr_orhr - start_orhr_col).count();
                auto start_r_restore = steady_clock::now();
                // Construct the proper R-factor
                for(int i = 0; i < n; ++i) {
                    for(int j = 0; j < (i + 1); ++j) {
                        all_data.R[(n * i) + j] *=  all_data.D[j];
                    }
                }
                blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, (T) 1.0, A_sk, n, all_data.R.data(), n);
                lapack::lacpy(MatrixType::Upper, n, n, all_data.R.data(), n, all_data.A.data(), m);
                auto stop_r_restore = steady_clock::now();
                dur_cholqr_r_restore = duration_cast<microseconds>(stop_r_restore - start_r_restore).count();
                data_regen(m_info, all_data, state, state, 2);
            
                file << dur_geqrf << ",  " << dur_geqr << ",  " << dur_cholqr <<  ",  " << dur_cholqr_precond << ",  " << dur_cholqr_house_rest << ",  " << dur_cholqr_r_restore << ",  ";
            }
            file << dur_geqrt << ",  ";
            data_regen(m_info, all_data, state, state, 2);
        }
        nb = geqrt_nb_start;
        file << "\n";
    }

    delete[] A_sk;
    delete[] S;
    delete[] J;
    delete[] tau;
}

template <typename T, typename RNG>
static void call_apply_q(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    int64_t gemqrt_nb_start,
    benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    RandBLAS::RNGState<RNG> &state_B,
    std::string output_filename) {

    auto m   = all_data.row;

    // timing vars
    long dur_ormqr  = 0;
    long dur_gemqrt = 0;
    long dur_gemm   = 0;

    std::ofstream file(output_filename, std::ios::app);

    int i, j   = 0;
    int64_t nb = 0;
    for (i = 0; i < numruns; ++i) {
        for(nb = gemqrt_nb_start; nb <= n; nb *=2) {
            printf("Apply Q iteration %d; n==%d start.\n", i, n);
            // Performing CholQR
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, all_data.A.data(), m, (T) 0.0, all_data.R.data(), n);
            lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.R.data(), n, all_data.A.data(), m);
            
            lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_gemqrt.data(), m);
            lapack::orhr_col(m, n, nb, all_data.A_gemqrt.data(), m, all_data.T_gemqrt.data(), n, all_data.D.data());
            
            auto start_gemqrt = high_resolution_clock::now();
            lapack::gemqrt(Side::Left, Op::Trans, m, m - n, n, nb, all_data.A_gemqrt.data(), m, all_data.T_gemqrt.data(), n, all_data.B1.data(), m);
            auto stop_gemqrt = high_resolution_clock::now();
            dur_gemqrt = duration_cast<microseconds>(stop_gemqrt - start_gemqrt).count();

            // We do not re-run ormqr and gemm for different nbs
            if(nb == gemqrt_nb_start) {
                lapack::orhr_col(m, n, n, all_data.A.data(), m, all_data.T_mat.data(), n, all_data.D.data());
                lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A1.data(), m);

                for(j = 0; j < n; ++j)
                    all_data.tau[j] = all_data.T_mat[(n + 1) * j];

                auto start_ormqr = high_resolution_clock::now();
                lapack::ormqr(Side::Left, Op::Trans, m, m - n, n, all_data.A.data(), m, all_data.tau.data(), all_data.B.data(), m);
                auto stop_ormqr = high_resolution_clock::now();
                dur_ormqr = duration_cast<microseconds>(stop_ormqr - start_ormqr).count();
            
                file << dur_ormqr << ",  ";                
            } 
            file << dur_gemqrt << ",  ";
            data_regen(m_info, all_data, state, state_B, 3);
        }
        nb = gemqrt_nb_start;
        file << "\n";
    }
}

int main(int argc, char *argv[]) {

    auto size = argv[1];

    int64_t i = 0;
    // Declare parameters
    int64_t m             = std::stol(size);
    int64_t n_start       = 256;
    int64_t n_stop        = 2048;
    int64_t nb_start      = 256;
    auto state            = RandBLAS::RNGState();
    auto state_B          = RandBLAS::RNGState();
    auto state_constant   = state;
    auto state_constant_B = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 3;

    // Allocate basic workspace
    benchmark_data<double> all_data(m, n_stop);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    RandLAPACK::gen::mat_gen(m_info, all_data.B.data(), state_B);

    // Declare a data file
    std::string output_filename = RandLAPACK::util::getCurrentDate<double>() + "BQRRP_subroutines_speed" 
                                                                 + "_num_info_lines_" + std::to_string(9) +
                                                                   ".txt";
    std::ofstream file(output_filename, std::ios::out | std::ios::trunc);

    // Writing important data into file
    file << "Description: Results from the BQRRP subroutines benchmark, recording time for the alternative options of the three main BQRRP subroutines: wide_qrcp, tall qr and application of transpose orthonormal matrix."
              "\nFile format: the format varies for each subroutine"
              "               \n qrcp_wide: the first two columns show ORMQR and GEMM time, the third and any subsequent columns show time for GEMQRT with a given block size (from nb_start to n in powers of 2). Rows vary from n_start to n_stop in powers of two (with numruns runs per size)."
              "               \n qr_tall:   six columns with timing for different tall QR candidates and their related parts: GEQRF, GEQR, CHOLQR, CHOLQR_PREPROCESSING, CHOLQR_HOUSEHOLDER_RESTORATION, CHOLQR_UNTO_PRECONDITIONING."
              "               \n apply_Q:   three columns with tall QRCP candidates: GEQP3, LUQR"
              "               \n In all cases, rows vary from n_start to n_stop in powers of two (with numruns runs per size)."
              "\nInput type:" + std::to_string(m_info.m_type) +
              "\nInput size:" + std::to_string(m) + " by "  + std::to_string(n_start) + " to " + std::to_string(n_stop) +
              "\nAdditional parameters num runs per size " + std::to_string(numruns) + " nb_start "   + std::to_string(nb_start) +
              "\n";
    file.flush();

    for (i = n_start; i <= n_stop; i *= 2)
        call_wide_qrcp(m_info, numruns, i, all_data, state, output_filename);

    for (i = n_start; i <= n_stop; i *= 2)
        call_tsqr(m_info, numruns, i, nb_start, all_data, state, output_filename);

    for (i = n_start; i <= n_stop; i *= 2)
        call_apply_q(m_info, numruns, i, nb_start, all_data, state, state_B, output_filename);
}
#endif