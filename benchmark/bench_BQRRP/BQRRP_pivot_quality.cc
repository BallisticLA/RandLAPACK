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

using Subroutines = RandLAPACK::BQRRPSubroutines;

template <typename T>
struct QR_speed_benchmark_data {
    int64_t row;
    int64_t col;
    T sampling_factor;
    std::vector<T> A;
    std::vector<T> tau;
    std::vector<int64_t> J;
    std::vector<T> S;

    QR_speed_benchmark_data(int64_t m, int64_t n, T d_factor) :
    A(m * n, 0.0),
    tau(n, 0.0),
    J(n, 0),
    S(n, 0.0)
    {
        row             = m;
        col             = n;
        sampling_factor = d_factor;
    }
};

template <typename T>
void _LAPACK_gejsv(
    char joba, char jobu, char jobv, char jobr,
    char jobt, char jobp,
    int64_t m, int64_t n,
    T *A, int64_t lda,
    T *S,
    T *U, int64_t ldu,
    T *V, int64_t ldv,
    T* work, int64_t* lwork,
    int64_t* iwork,
    int64_t* info
){

    char joba_ = joba; //lapack::to_char( joba );
    char jobu_ = jobu; //lapack::to_char( jobu );
    char jobv_ = jobv; //lapack::to_char( jobv );
    char jobr_ = jobr; //lapack::to_char( jobr );
    char jobt_ = jobt; //lapack::to_char( jobt );;
    char jobp_ = jobp; //lapack::to_char( jobp );

    lapack_int m_   = (lapack_int) m;
    lapack_int n_   = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldv_ = (lapack_int) ldv;
    
    lapack_int *lwork_ = (lapack_int *) lwork;
    lapack_int *iwork_ = (lapack_int *) iwork;
    lapack_int *info_  = (lapack_int *) info;

    LAPACK_dgejsv( & joba_, & jobu_, & jobv_, & jobr_,
        & jobt_, & jobp_,
        & m_, & n_,
        A, & lda_,
        S,
        U, & ldu_,
        V, & ldv_,
        work, lwork_,
        iwork_,
        info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        //, 1, 1, 1, 1, 1, 1
        #endif
        );

    return;
}

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
    RandBLAS::RNGState<RNG> &state,
    std::string path) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto d_factor = all_data.sampling_factor;

    auto state_alg = state;
    auto state_gen = state;

    // Additional params setup.
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(false, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

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
    BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);

    std::vector<T> R_norms_BQRRP = get_norms(all_data);

    // Declare a data file
    std::string output_filename = "_BQRRP_pivot_quality_metric_1_num_info_lines_" + std::to_string(6) + ".txt";

    if (path != ".")
        path += output_filename;
    else
        path = output_filename;
                                                                
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results of the BQRRP pivot quality benchmark for the metric of ratios of the norms of R factors output by QP3 and BQRRP."
              "\nFile format: File output is one-line."
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"        + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();

    // Write the 1st metric info into a file.
    for (int i = 0; i < n; ++i) {
        file << R_norms_HQRRP[i] / R_norms_BQRRP[i] << ",  ";
    }
    file  << "\n";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);
}

template <typename T, typename RNG>
static void sv_ratio(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t b_sz,
    QR_speed_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string path) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto d_factor = all_data.sampling_factor;
    std::vector<T> geqp3 (n, 0.0);
    std::vector<T> sv_ratios_bqrrp (n, 0.0);

    auto state_alg = state;
    auto state_gen = state;

    // Additional params setup.
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(false, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    // Declare a data file
    std::string output_filename = "_BQRRP_pivot_quality_metric_2_num_info_lines_" + std::to_string(6) + ".txt";

    if (path != ".")
        path += output_filename;
    else
        path = output_filename;
                                                                
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results of the BQRRP pivot quality benchmark for the metric of ratios of the diagonal R entries to true singular values."
              "\nFile format: Line one contains BQRRP retults, line 2 contains GEQP3 retults."
              "\nNum OMP threads:"  + std::to_string(RandLAPACK::util::get_omp_threads()) +
              "\nInput type:"        + std::to_string(m_info.m_type) +
              "\nInput size:"       + std::to_string(m) + " by "  + std::to_string(n) +
              "\nAdditional parameters: BQRRP block size: " + std::to_string(b_sz) + " BQRRP d factor: "   + std::to_string(d_factor) +
              "\n";
    file.flush();

    T* S_dat = all_data.S.data();
    T* R_dat = nullptr;

    // Running SVD
    lapack::gesdd(Job::NoVec, m, n, all_data.A.data(), m, all_data.S.data(), (T*) nullptr, m, (T*) nullptr, n);

    char joba = 'C'; 
    char jobu = 'N';
    char jobv = 'N';
    char jobr = 'N';
    char jobt = 'N';
    char jobp = 'N';
    
    double* buff_workspace  = new double[8 * m * n]();
    int64_t lwork[1]; 
    lwork[0] = 8 * m * n;
    int64_t iwork[8 * std::min(m,n)];
    int64_t info[1];

    _LAPACK_gejsv(
        joba, jobu, jobv, jobr,
        jobt, jobp,
        m, n,
        all_data.A.data(), m,
        all_data.S.data(),
        (T*) nullptr, m,
        (T*) nullptr, n,
        buff_workspace, lwork,
        iwork,
        info
    );

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    // Running GEQP3
    lapack::geqp3(m, n, all_data.A.data(), m, all_data.J.data(), all_data.tau.data());

    R_dat = all_data.A.data();
    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i){
        file << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ",  ";
    }
    file  << "\n";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    // Running BQRRP
    state_alg = state;
    BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state_alg);

    R_dat = all_data.A.data();
    // Write the 2nd metric info into a file.
    for (int i = 0; i < n; ++i) {
        file << std::abs(R_dat[(m + 1) * i] / S_dat[i]) << ",  ";
    }
    file  << "\n";

    // Clear and re-generate data
    state_gen = state;
    data_regen(m_info, all_data, state_gen);

    delete[] buff_workspace;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <directory_path> <num_rows> <num_cols> <block_size>" << std::endl;
        return 1;
    }

    // Declare parameters
    std::string path   = argv[1];
    int64_t m          = std::stol(argv[2]);
    int64_t n          = std::stol(argv[3]);
    double d_factor    = 1.0;
    int64_t b_sz       = std::stol(argv[4]);;
    auto state         = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant1 = state;
    auto state_constant2 = state;
    // results
    std::vector<double> res1;
    std::vector<double> res2;

    // Allocate basic workspace
    QR_speed_benchmark_data<double> all_data(m, n, d_factor);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::kahan);
    m_info.theta   = 1.2;
    m_info.perturb = 1e3;
    //m_info.cond_num = std::pow(10, 10);
    //m_info.rank = n;
    //m_info.exponent = 2.0;
    //m_info.scaling = std::pow(10, 10);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    R_norm_ratio(m_info, b_sz, all_data, state_constant1, path);
    printf("Pivot quality metric 1 done\n");
    sv_ratio(m_info, b_sz, all_data, state_constant2, path);
    printf("Pivot quality metric 2 done\n\n");
}
#endif