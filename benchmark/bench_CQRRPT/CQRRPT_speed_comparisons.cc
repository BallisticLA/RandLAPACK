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

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

template <typename T, typename RNG>
static std::vector<long> call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(true, true, tol);
    CQRRPT.nnz = 4;
    CQRRPT.num_threads = 48;

    // timing vars
    long dur_cqrrpt     = 0;
    long dur_geqp3      = 0;
    long dur_geqr       = 0;
    long dur_geqpt      = 0;
    long dur_geqrf      = 0;
    long dur_scholqr    = 0;
    long t_cqrrpt_best  = 0;
    long t_geqp3_best   = 0;
    long t_geqr_best    = 0;
    long t_geqpt_best   = 0;
    long t_geqrf_best   = 0;
    long t_scholqr_best = 0;
    
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
        data_regen<T, RNG>(m_info, all_data, state_gen);

        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);

        // Testing CQRRPT
        auto start_cqrrp = high_resolution_clock::now();
        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state_alg);
        auto stop_cqrrp = high_resolution_clock::now();
        dur_cqrrpt = duration_cast<microseconds>(stop_cqrrp - start_cqrrp).count();

        state_gen = state;
        state_alg = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);

        // Testing SCHOLQR3
        auto start_scholqr = high_resolution_clock::now();
        //--------------------------------------------------------------------------------------------------------------------------//
        T norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
        T shift = 11 * std::numeric_limits<double>::epsilon() * n * std::pow(norm_A, 2);
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
        data_regen<T, RNG>(m_info, all_data, state_gen);

        // Testing GEQR + GEQPT
        auto start_geqpt = high_resolution_clock::now();
        auto start_geqr  = high_resolution_clock::now();
#if !defined(__APPLE__)
        // GEQR(A) part
        lapack::geqr(m, n, all_data.A.data(), m,  all_data.tau.data(), -1);
        int64_t tsize = (int64_t) all_data.tau[0]; 
        all_data.tau.resize(tsize);
        lapack::geqr(m, n, all_data.A.data(), m, all_data.tau.data(), tsize);
#endif
        auto stop_geqr = high_resolution_clock::now();
        dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();
#if !defined(__APPLE__)
        // GEQP3(R) part
        lapack::lacpy(MatrixType::Upper, n, n, all_data.A.data(), m, all_data.R.data(), n);
        lapack::geqp3(n, n, all_data.R.data(), n, all_data.J.data(), all_data.tau.data());
#endif
        auto stop_geqpt = high_resolution_clock::now();
        dur_geqpt = duration_cast<microseconds>(stop_geqpt - start_geqpt).count();

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);
    
        i == 0 ? t_cqrrpt_best  = dur_cqrrpt  : (dur_cqrrpt < t_cqrrpt_best)   ? t_cqrrpt_best = dur_cqrrpt   : NULL;
        i == 0 ? t_geqpt_best   = dur_geqpt   : (dur_geqpt < t_geqpt_best)     ? t_geqpt_best = dur_geqpt     : NULL;
        i == 0 ? t_geqrf_best   = dur_geqrf   : (dur_geqrf < t_geqrf_best)     ? t_geqrf_best = dur_geqrf     : NULL;
        i == 0 ? t_geqr_best    = dur_geqr    : (dur_geqr < t_geqr_best)       ? t_geqr_best = dur_geqr       : NULL;
        i == 0 ? t_geqp3_best   = dur_geqp3   : (dur_geqp3 < t_geqp3_best)     ? t_geqp3_best = dur_geqp3     : NULL;
        i == 0 ? t_scholqr_best = dur_scholqr : (dur_scholqr < t_scholqr_best) ? t_scholqr_best = dur_scholqr : NULL;
    }

    std::vector<long> res{t_cqrrpt_best, t_geqpt_best, t_geqrf_best, t_geqr_best, t_geqp3_best, t_scholqr_best};

    return res;
}

int main() {
    // Declare parameters
    int64_t m           = std::pow(2, 17);
    int64_t n_start     = std::pow(2, 9);
    int64_t n_stop      = std::pow(2, 13);
    double  d_factor    = 1.25;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 1;

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n_stop, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    // Declare a data file
    std::fstream file("CQRRPT_speed_comp_"              + std::to_string(m)
                                      + "_col_start_"    + std::to_string(n_start)
                                      + "_col_stop_"     + std::to_string(n_stop)
                                      + "_d_factor_"     + std::to_string(d_factor)
                                      + ".dat", std::fstream::app);

    for (;n_start <= n_stop; n_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, n_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",  " << res[2] << ",  " << res[3] << ",  " << res[4] << ",  " << res[5] << ",\n";
    }
}