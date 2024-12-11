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
    RandLAPACK::BQRRP_blocked<double, r123::Philox4x32> BQRRP_blocked(false, tol, b_sz);
    BQRRP_blocked.qr_tall = "cholqr";
    //BQRRP_blocked.qrcp_wide = "qp3";

    // Running QP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
    std::vector<T> R_norms_HQRRP = get_norms(all_data);
    printf("\nDone with QP3\n");

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    printf("\nStarting BQRRP\n");
    // Running BQRRP
    state_alg = state;
    BQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);
    printf("%ld\n", BQRRP_blocked.rank);
    std::vector<T> R_norms_BQRRP = get_norms(all_data);

    // Declare a data file
    std::ofstream file1(RandLAPACK::util::getCurrentDate<T>() + "BQRRP_pivot_quality_metric_1"
                                                          + "_num_info_lines_" + std::to_string(5) +
                                                            ".txt", std::ios::out | std::ios::trunc);

    // Writing important data into file
    file1 << "Description: Results of the BQRRP pivot quality benchmark for the metric of ratios of the norms of R factors output by QP3 and BQRRP."
              "\nFile format: File output is one-line."
              "\nInput type:"        + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file1.flush();

    // Write the 1st metric info into a file.
    for (int i = 0; i < n; ++i)
        file1 << R_norms_HQRRP[i] / R_norms_BQRRP[i] << ",  ";

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
    std::vector<T> sv_ratios_bqrrp (n, 0.0);

    auto state_alg = state;
    auto state_gen = state;

    // Additional params setup.
    RandLAPACK::BQRRP_blocked<double, r123::Philox4x32> BQRRP_blocked(false, tol, b_sz);
    BQRRP_blocked.qr_tall = "cholqr";
    //BQRRP_blocked.qrcp_wide = "qp3";

    std::ofstream file2(RandLAPACK::util::getCurrentDate<T>() + "BQRRP_pivot_quality_metric_2"
                                                          + "_num_info_lines_" + std::to_string(5) +
                                                            ".txt", std::ios::out | std::ios::trunc);
    // Writing important data into file
    file2 << "Description: Results of the BQRRP pivot quality benchmark for the metric of ratios of the diagonal R entries to true singular values."
              "\nFile format: Line one contains BQRRP retults, line 2 contains GEQP3 retults."
              "\nInput type:"        + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file2.flush();

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
        file2 << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ",  ";
    }
    file2  << ",\n";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    // Running BQRRP
    state_alg = state;
    BQRRP_blocked.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);

    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i)
        file2 << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ",  ";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);
}

int main(int argc, char *argv[]) {

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }
    auto size = argv[1];

    // Declare parameters
    int64_t m          = std::stol(size);
    int64_t n          = std::stol(size);
    double d_factor    = 1.0;
    int64_t b_sz       = 4096;
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
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::kahan);
    m_info.theta   = 1.2;
    m_info.perturb = 1e3;
    //m_info.cond_num = std::pow(10, 10);
    //m_info.rank = n;
    //m_info.exponent = 2.0;
    //m_info.scaling = std::pow(10, 10);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    std::ofstream file(RandLAPACK::util::getCurrentDate<double>() + "Pivot_quality_benchmark_generated_matrix" 
                                              + "_mat_type_"     + std::to_string(m_info.m_type)
                                              + "_numrows_"      + std::to_string(m)
                                              + "_numcols_"      + std::to_string(n)
                                              , std::ios::out | std::ios::trunc);

    R_norm_ratio(m_info, b_sz, all_data, state_constant1);
    printf("Pivot quality metric 1 done\n");
    sv_ratio(m_info, b_sz, all_data, state_constant2);
    printf("SPivot quality metric 2 done\n\n");
}
#endif