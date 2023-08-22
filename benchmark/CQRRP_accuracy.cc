#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct QR_speed_benchmark_data {
    int64_t row;
    int64_t col;
    T       tolerance;
    int64_t sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;
    std::vector<T> S;

    QR_speed_benchmark_data(int64_t m, int64_t n, T tol, int64_t d_factor) :
    A(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0),
    S(n, 0.0)
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
                                        QR_speed_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

// Re-generate and clear data
template <typename T, typename RNG>
static std::vector<T> get_norms( QR_speed_benchmark_data<T> &all_data) {

    int64_t m = all_data.row;
    int64_t n = all_data.col;

    std::vector<T> R_norms (n, 0.0);
    for (int i = 0; i < n; ++i) {
        R_norms[i] = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, n - i, n - i, &all_data.A.data()[(m + 1) * i], m);
    }
    return R_norms;
}

template <typename T, typename RNG>
static void R_norm_ratio(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t b_sz,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 8;
    CQRRP_blocked.qrcp = 1;

    // Running GEQP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
    std::vector<T> R_norms_GEQP3 = get_norms<T, RNG>(all_data);

    // Clear and re-generate data
    data_regen<T, RNG>(m_info, all_data, state);

    // Running CQRRP
    CQRRP_blocked.call(m, n, all_data.A.data(), d_factor, all_data.tau.data(), all_data.J.data(), state);
    std::vector<T> R_norms_CQRRP = get_norms<T, RNG>(all_data);

    // Declare a data file
    std::fstream file1("QR_R_norm_ratios_rows_"        + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_"         + std::to_string(b_sz)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

    // Write the 1st metric info into a file.
    for (int i = 0; i < n; ++i)
        file1 << R_norms_GEQP3[i] / R_norms_CQRRP[i] << ",  ";
}

template <typename T, typename RNG>
static void sv_ratio(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t b_sz,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;
    std::vector<T> geqp3 (n, 0.0);
    std::vector<T> sv_ratios_cqrrp (n, 0.0);

    auto state1 = state;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 8;
    CQRRP_blocked.qrcp = 1;

    std::fstream file2("QR_sv_ratios_rows_"            + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_start_"   + std::to_string(b_sz)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

    T* R_dat = all_data.A.data();
    T* S_dat = all_data.S.data();

    // Running SVD
    lapack::gesdd(Job::NoVec, m, n, all_data.A.data(), m, all_data.S.data(), (T*) nullptr, m, (T*) nullptr, n);

    // Clear and re-generate data
    data_regen<T, RNG>(m_info, all_data, state);

    // Running GEQP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());

    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i)
        file2 << R_dat[(m + 1) * i] / S_dat[i] << ",  ";
    
    file2  << ",\n";

    // Clear and re-generate data
    data_regen<T, RNG>(m_info, all_data, state1);

    // Running CQRRP
    CQRRP_blocked.call(m, n, all_data.A.data(), d_factor, all_data.tau.data(), all_data.J.data(), state);

    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i)
        file2 << R_dat[(m + 1) * i] / S_dat[i] << ",  ";
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 14);
    int64_t n          = std::pow(2, 14);
    int64_t d_factor   = 1.125;
    int64_t b_sz_start = 128;
    int64_t b_sz_end   = 288;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant1 = state;
    auto state_constant2 = state;
    // results
    std::vector<double> res1;
    std::vector<double> res2;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    for (;b_sz_start <= b_sz_end; b_sz_start += 32) {
        R_norm_ratio<double, r123::Philox4x32>(m_info, b_sz_start, all_data, state_constant1);
        printf("R done\n");
        sv_ratio<double, r123::Philox4x32>(m_info, b_sz_start, all_data, state_constant2);
        printf("SV done\n\n");
    }
}