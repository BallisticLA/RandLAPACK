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

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct QR_benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> R;
    std::vector<T> tau;
    std::vector<int64_t> J;

    QR_benchmark_data(int64_t m, int64_t n, T tol, T d_factor) :
    A(m * n, 0.0),
    R(n * n, 0.0),
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
                                        QR_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

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
    RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT(true, tol);
    CQRRPT.nnz = 4;
    CQRRPT.num_threads = 48;

    // timing vars
    long dur_cqrrpt     = 0;
    long dur_geqp3      = 0;
    long dur_geqr       = 0;
    long dur_geqpt      = 0;
    long dur_geqrf      = 0;
    long dur_scholqr    = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        // Testing GEQP3
        auto start_geqp3 = high_resolution_clock::now();
        lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
        auto stop_geqp3 = high_resolution_clock::now();
        dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen);

        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen);

        // Testing CQRRPT
        auto start_cqrrp = high_resolution_clock::now();
        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state_alg);
        printf("CQRRPT RANK: %ld\n", CQRRPT.rank);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrpt = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();

        state_gen = state;
        state_alg = state;
        data_regen(m_info, all_data, state_gen);

        // Testing SCHOLQR3
        auto start_scholqr = high_resolution_clock::now();
        //--------------------------------------------------------------------------------------------------------------------------//
        T norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        T shift = 11 * std::numeric_limits<T>::epsilon() * n * std::pow(norm_A, 2);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A.data(), m, 0.0, all_data.R.data(), n);
        for (int i = 0; i < n; ++i)
            all_data.R[i * (n + 1)] += shift;
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R.data(), n, all_data.A.data(), m);
        // CholeskyQR2
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A.data(), m, 0.0, all_data.R.data(), n);
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R.data(), n, all_data.A.data(), m);
        // CholeskyQR3
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A.data(), m, 0.0, all_data.R.data(), n);
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R.data(), n, all_data.A.data(), m);
        //--------------------------------------------------------------------------------------------------------------------------//
        auto stop_scholqr = high_resolution_clock::now();
        dur_scholqr = duration_cast<microseconds>(stop_scholqr - start_scholqr).count();

        auto state_gen = state;
        data_regen(m_info, all_data, state_gen);

        // Testing GEQR + GEQPT
#if !defined(__APPLE__)
        auto start_geqpt = high_resolution_clock::now();
        auto start_geqr  = high_resolution_clock::now();
        // GEQR(A) part
        lapack::geqr(m, n, all_data.A.data(), m,  all_data.tau.data(), -1);
        int64_t tsize = (int64_t) all_data.tau[0]; 
        all_data.tau.resize(tsize);
        lapack::geqr(m, n, all_data.A.data(), m, all_data.tau.data(), tsize);

        auto stop_geqr = high_resolution_clock::now();
        dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();

        // GEQP3(R) part
        lapack::lacpy(MatrixType::Upper, n, n, all_data.A.data(), m, all_data.R.data(), n);
        lapack::geqp3(n, n, all_data.R.data(), n, all_data.J.data(), all_data.tau.data());
        auto stop_geqpt = high_resolution_clock::now();
        dur_geqpt = duration_cast<microseconds>(stop_geqpt - start_geqpt).count();
        state_gen = state;
        data_regen(m_info, all_data, state_gen);
#endif

        std::ofstream file(output_filename, std::ios::app);
        file << dur_cqrrpt << ",  " << dur_geqpt <<  ",  " << dur_geqrf   << ",  " 
             << dur_geqr   << ",  " << dur_geqp3 <<  ",  " << dur_scholqr << ",\n";
    }
}

int main() {
    // Declare parameters
    int64_t m           = 10000;//std::pow(2, 17);
    int64_t n_start     = std::pow(2, 5);
    int64_t n_stop      = std::pow(2, 11);
    double  d_factor    = 1.25;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 25;

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n_stop, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "CQRRPT_speed_comp_"   + std::to_string(m)
                                      + "_col_start_"    + std::to_string(n_start)
                                      + "_col_stop_"     + std::to_string(n_stop)
                                      + "_d_factor_"     + std::to_string(d_factor)
                                      + ".dat";

    for (;n_start <= n_stop; n_start *= 2) {
        call_all_algs(m_info, numruns, n_start, all_data, state_constant, output_filename);
    }
}
