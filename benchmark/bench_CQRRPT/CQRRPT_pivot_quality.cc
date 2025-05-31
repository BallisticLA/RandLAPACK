/*
Performs computations in order to assess the pivot quality of CQRRPT.
The setup is described in detail in Section 4 of The arXiv version 2 CQRRPT (https://arxiv.org/pdf/2311.08316.pdf) paper.
*/
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct QR_benchmark_data {
    int64_t              row;
    int64_t              col;
    T                    tolerance;
    T                    sampling_factor;
    std::vector<T>       A;
    std::vector<T>       R;
    std::vector<T>       tau;
    std::vector<int64_t> J;
    std::vector<T>       S;

    QR_benchmark_data(int64_t m, int64_t n, T tol, T d_factor) :
    A(m * n, 0.0),
    R(n * n, 0.0),
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
                                        QR_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.R.begin(), all_data.R.end(), 0.0);
    std::fill(all_data.tau.begin(), all_data.tau.end(), 0.0);
    std::fill(all_data.J.begin(), all_data.J.end(), 0);
}

// Re-generate and clear data
template <typename T>
static std::vector<T> get_norms(int64_t n, std::vector<T> Mat, int64_t lda) {

    std::vector<T> R_norms (n, 0.0);
    for (int i = 0; i < n; ++i) {
        R_norms[i] = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, n - i, n - i, &Mat[(lda + 1) * i], lda);
        if (i < 10)
            printf("%e\n", R_norms[i]);
    }
    return R_norms;
}

template <typename T, typename RNG>
static void R_norm_ratio(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    QR_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;
    auto d_factor = all_data.sampling_factor;

    auto state_alg = state;
    auto state_gen = state;

    // Additional params setup.
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(true, tol);
    CQRRPT.nnz = 4;

    // Running GEQP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());
    std::vector<T> R_norms_GEQP3 = get_norms(n, all_data.A, m);
    printf("\nDone with QP3\n");

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    printf("\nStarting CQRRPT\n");
    // Running CQRRP
    state_alg = state;
    CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state_alg);
    std::vector<T> R_norms_CQRRPT = get_norms(n, all_data.R, n);

    // Declare a data file
    std::ofstream file1(RandLAPACK::util::get_current_date_time<T>() + "_CQRRPT_pivot_quality_metric_1"
                                                          + "_num_info_lines_" + std::to_string(5) +
                                                            ".txt", std::ios::out | std::ios::trunc);

    // Writing important data into file
    file1 << "Description: Results of the CQRRPT pivot quality benchmark for the metric of ratios of the norms of R factors output by QP3 and CQRRPT."
              "\nFile format: File output is one-line."
              "\nInput type:"       + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: CQRRPT d factor: "        + std::to_string(d_factor) +
              "\n";
    file1.flush();

    // Write the 1st metric info into a file.
    for (int i = 0; i < n; ++i)
        file1 << R_norms_GEQP3[i] / R_norms_CQRRPT[i] << ", ";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);
}

template <typename T, typename RNG>
static void sv_ratio(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    QR_benchmark_data<T> &all_data,
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
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(true, tol);
    CQRRPT.nnz = 4;

    std::ofstream file2(RandLAPACK::util::get_current_date_time<T>() + "_CQRRPT_pivot_quality_metric_2"
                                                          + "_num_info_lines_" + std::to_string(6) +
                                                            ".txt", std::ios::out | std::ios::trunc);
    // Writing important data into file
    file2 << "Description: Results of the CQRRPT pivot quality benchmark for the metric of ratios of the diagonal R entries to true singular values."
              "\nFile format: Line one contains CQRRPT retults, line 2 contains GEQP3 retults."
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"        + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: CQRRPT d factor: "   + std::to_string(d_factor) +
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
    for (int i = 0; i < n; ++i)
        file2 << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ", ";
    file2  << ",\n";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    // Running CQRRP
    state_alg = state;
    CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state_alg);

    R_dat = all_data.R.data();
    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i)
        file2 << std::abs(R_dat[(n + 1) * i] / S_dat[i]) << ",  ";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <num_rows> <num_columns>..." << std::endl;
        return 1;
    }

    // Declare parameters
    int64_t m           = std::stol(argv[1]);
    int64_t n           = std::stol(argv[2]);
    double  d_factor    = 1.25;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant1 = state;
    auto state_constant2 = state;
    // results
    std::vector<double> res1;
    std::vector<double> res2;

    // Allocate basic workspace
    QR_benchmark_data<double> all_data(m, n, tol, d_factor);
    // Generate the input matrix: 
    // polynomial & step for low coherence; 
    // spiked for high coherence.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::spiked);
    m_info.cond_num = std::pow(10, 10);
    m_info.rank = n;
    m_info.exponent = 2.0;
    m_info.scaling = std::pow(10, 10);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    R_norm_ratio<double>(m_info, all_data, state_constant1);
    printf("R done\n");
    sv_ratio<double>(m_info, all_data, state_constant2);
    printf("SV done\n\n");
}