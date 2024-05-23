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
struct QR_benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> R;
    std::vector<T> Q;
    std::vector<T> tau;

    QR_benchmark_data(int64_t m, int64_t n) :
    A(m * n, 0.0),
    R(n * n, 0.0),
    Q(m * n, 0.0),
    tau(n, 0.0)
    {
        row = m;
        col = n;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        QR_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state,
                                        int zero_Q) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    if (zero_Q) {
        std::fill(all_data.Q.begin(), all_data.Q.end(), 0.0);
    }
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

    int64_t tsize = 0;

    // timing vars
    long dur_geqrf      = 0;
    long dur_geqr       = 0;
    long dur_geqr_ungqr = 0;
    long dur_cholqr     = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;

    for (int i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        // Testing GEQRF
        auto start_geqrf = high_resolution_clock::now();
        lapack::geqrf(m, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqrf = high_resolution_clock::now();
        dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen, 0);

        // Testing GEQR
        auto start_geqr = high_resolution_clock::now();
        lapack::geqr(m, n, all_data.A.data(), m,  all_data.tau.data(), -1);
        tsize = (int64_t) all_data.tau[0]; 
        all_data.tau.resize(tsize);
        lapack::geqr(m, n, all_data.A.data(), m, all_data.tau.data(), tsize);
        auto stop_geqr = high_resolution_clock::now();
        dur_geqr = duration_cast<microseconds>(stop_geqr - start_geqr).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen, 0);

        // Testing GEQR + UNGQR
        auto start_geqr_ungqr = high_resolution_clock::now();
        lapack::geqr(m, n, all_data.A.data(), m,  all_data.tau.data(), -1);
        tsize = (int64_t) all_data.tau[0]; 
        all_data.tau.resize(tsize);
        lapack::geqr(m, n, all_data.A.data(), m, all_data.tau.data(), tsize);
        lapack::ungqr(m, n, n, all_data.A.data(), m, all_data.tau.data());
        auto stop_geqr_ungqr = high_resolution_clock::now();
        dur_geqr_ungqr = duration_cast<microseconds>(stop_geqr_ungqr - start_geqr_ungqr).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen, 0);

        // Testing CholQR
        auto start_cholqr = high_resolution_clock::now();
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, all_data.A.data(), m, (T) 0.0, all_data.R.data(), n);
        lapack::potrf(Uplo::Upper, n, all_data.R.data(), n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, all_data.R.data(), n, all_data.A.data(), m);
        auto stop_cholqr = high_resolution_clock::now();
        dur_cholqr = duration_cast<microseconds>(stop_cholqr - start_cholqr).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen, 1);
    
        std::ofstream file(output_filename, std::ios::app);
        file << n << ",  " << dur_geqrf << ",  " << dur_geqr << ",  " << dur_geqr_ungqr << ",  " << dur_cholqr << ",\n";
    }
}

int main() {
    // Declare parameters
    int64_t m           = std::pow(2, 17);
    int64_t n_start     = std::pow(2, 9);
    int64_t n_stop      = std::pow(2, 13);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n_stop);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n_stop, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    // Declare a data file
    std::string output_filename = "QR_speed_comp_"              + std::to_string(m)
                                      + "_col_start_"    + std::to_string(n_start)
                                      + "_col_stop_"     + std::to_string(n_stop)
                                      + ".dat"; 

    for (;n_start <= n_stop; n_start *= 2) {
        call_all_algs(m_info, numruns, n_start, all_data, state_constant, output_filename);
    }
}
