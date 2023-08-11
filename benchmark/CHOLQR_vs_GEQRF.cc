#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct CHOLQR_vs_GEQRF_speed_benchmark_data {
    int64_t row;
    int64_t col;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<T> R;
    std::vector<T> T_mat;
    std::vector<T> D;

    CHOLQR_vs_GEQRF_speed_benchmark_data(int64_t m, int64_t n) :
    A(m * n, 0.0),
    tau(n, 0.0),
    R(n * n, 0.0),
    T_mat(n * n, 0.0),
    D(n, 0.0)
    {
        row = m;
        col = n;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        CHOLQR_vs_GEQRF_speed_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int is_cholqr) {

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (is_cholqr) {
        std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
        std::fill(all_data.T_mat.begin(), all_data.T_mat.end(), 0.0);
        std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
        std::fill(all_data.D.begin(), all_data.D.end(), 0.0);
    }
}

template <typename T, typename RNG>
static std::vector<long> call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    CHOLQR_vs_GEQRF_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;

    // timing vars
    long dur_cholqr       = 0;
    long dur_geqrf        = 0;
    long t_cholqr_best       = 0;
    long t_geqrf_best        = 0;

    T* R_dat = all_data.R.data();
    T* D_dat = all_data.D.data();
    T* T_dat = all_data.T_mat.data();
    T* tau_dat = all_data.tau.data();

    for (int k = 0; k < numruns; ++k) {
        // Testing cholqr
        auto start_cholqr = high_resolution_clock::now();
        //----------------------------------------------------------------------------------------------------------------------------------------/
        // Find R = A^TA.
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, all_data.A.data(), m, 0.0, all_data.R.data(), n);
        // Perform Cholesky factorization on A.
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        // Find Q = A * inv(R)
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, all_data.R.data(), n, all_data.A.data(), m);
        // Perform Householder reconstruction
        lapack::orhr_col(m, n, n, all_data.A.data(), m, all_data.T_mat.data(), n, all_data.D.data());
        // Update the signs in the R-factor
        int i, j;
        for(i = 0; i < n; ++i)
            for(j = 0; j < (i + 1); ++j)
                R_dat[(n * i) + j] *= D_dat[j];

        // Copy the R-factor into the upper-trianular portion of A
        lapack::lacpy(MatrixType::Upper, n, n, all_data.R.data(), n, all_data.A.data(), m);
        // Entries of tau will be placed on the main diagonal of matrix T from orhr_col().
        for(i = 0; i < n; ++i)
            tau_dat[i] = T_dat[(n + 1) * i];
        //----------------------------------------------------------------------------------------------------------------------------------------/
        auto stop_cholqr = high_resolution_clock::now();
        dur_cholqr = duration_cast<microseconds>(stop_cholqr - start_cholqr).count();
        // Update best timing
        k == 0 ? t_cholqr_best = dur_cholqr : (dur_cholqr < t_cholqr_best) ? t_cholqr_best = dur_cholqr : NULL;

        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state, 0);

        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();
        // Update best timing
        k == 0 ? t_geqrf_best = dur_geqrf : (dur_geqrf < t_geqrf_best) ? t_geqrf_best = dur_geqrf : NULL;

        // Clear and re-generate data
        data_regen<T, RNG>(m_info, all_data, state, 1);
    }

    printf("CHOLQR takes %ld μs\n", t_cholqr_best);
    printf("GEQRF takes %ld μs\n\n", t_geqrf_best);
    std::vector<long> res{t_cholqr_best, t_geqrf_best};

    return res;
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 14);
    int64_t n          = 256;
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 15;

    // Allocate basic workspace
    CHOLQR_vs_GEQRF_speed_benchmark_data<double> all_data(m, n);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    // Declare a data file
    std::fstream file("CHOLQR_vs_GEQRF_time_raw_rows_"              + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + ".dat", std::fstream::app);

        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, all_data, state_constant);
}