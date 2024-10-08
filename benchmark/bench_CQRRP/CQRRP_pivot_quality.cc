#if defined(__APPLE__)
int main() {return 0;}
#else
/*
Performs computations in order to assess the pivot quality of ICQRRP.
The setup is described in detail in Section 4 of The arXiv version 2 CQRRPT (https://arxiv.org/pdf/2311.08316.pdf) paper.
*/
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
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;
    std::vector<T> S;

    QR_speed_benchmark_data(int64_t m, int64_t n, T tol, T d_factor) :
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

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

// Re-generate and clear data
template <typename T>
static std::vector<T> get_norms( QR_speed_benchmark_data<T> &all_data) {

    int64_t m = all_data.row;
    int64_t n = all_data.col;

    std::vector<T> R_norms (n, 0.0);
    for (int i = 0; i < n; ++i) {
        R_norms[i] = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, n - i, n - i, &all_data.A.data()[(m + 1) * i], m);
        if (i < 10)
            printf("%e\n", R_norms[i]);
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

    auto state_alg = state;
    auto state_gen = state;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    //CQRRP_blocked.use_qp3 = 1;

    // Running QP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
    std::vector<T> R_norms_HQRRP = get_norms(all_data);
    printf("\nDone with QP3\n");

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    printf("\nStarting CQRRP\n");
    // Running CQRRP
    state_alg = state;
    CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
    std::vector<T> R_norms_CQRRP = get_norms(all_data);

    // Declare a data file
    std::fstream file1("QR_R_norm_ratios_rows_"        + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_"         + std::to_string(b_sz)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

    // Write the 1st metric info into a file.
    for (int i = 0; i < n; ++i)
        file1 << R_norms_HQRRP[i] / R_norms_CQRRP[i] << ",  ";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);
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

    auto state_alg = state;
    auto state_gen = state;

    // Additional params setup.
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, tol, b_sz);
    //CQRRP_blocked.use_qp3 = 1;

    std::fstream file2("QR_sv_ratios_rows_"            + std::to_string(m)
                                    + "_cols_"         + std::to_string(n)
                                    + "_b_sz_"   + std::to_string(b_sz)
                                    + "_d_factor_"     + std::to_string(d_factor)
                                    + ".dat", std::fstream::app);

    T* R_dat = all_data.A.data();
    T* S_dat = all_data.S.data();

    // Running SVD
    lapack::gesdd(Job::NoVec, m, n, all_data.A.data(), m, all_data.S.data(), (T*) nullptr, m, (T*) nullptr, n);

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    // Running GEQP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());

    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i){
        printf("%e\n", S_dat[i]);
        file2 << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ",  ";
    }
    file2  << ",\n";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    // Running CQRRP
    state_alg = state;
    CQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);

    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i)
        file2 << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ",  ";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);
}

int main() {
    // Declare parameters
    int64_t m          = std::pow(2, 10);
    int64_t n          = std::pow(2, 10);
    double d_factor    = 1.25;
    int64_t b_sz       = 256;
    double tol         = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant1 = state;
    auto state_constant2 = state;
    // results
    std::vector<double> res1;
    std::vector<double> res2;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::spiked);
    m_info.cond_num = std::pow(10, 10);
    m_info.rank = n;
    m_info.exponent = 2.0;
    m_info.scaling = std::pow(10, 10);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    R_norm_ratio(m_info, b_sz, all_data, state_constant1);
    printf("R done\n");
    sv_ratio(m_info, b_sz, all_data, state_constant2);
    printf("SV done\n\n");
}
#endif