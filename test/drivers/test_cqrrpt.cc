#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestCQRRPT : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRPTTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> R;
        std::vector<int64_t> J;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        CQRRPTTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0), 
        J(n, 0),  
        A_cpy1(m * n, 0.0),
        A_cpy2(m * n, 0.0),
        I_ref(k * k, 0.0) 
        {
            row = m;
            col = n;
            rank = k;
        }
    };

    template <typename T, typename RNG>
    static void norm_and_copy_computational_helper(T &norm_A, CQRRPTTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, CQRRPTTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat         = all_data.A_cpy1.data();
        T const* A_cpy_dat = all_data.A_cpy2.data();
        T const* Q_dat   = all_data.A.data();
        T const* R_dat   = all_data.R.data();
        T* I_ref_dat     = all_data.I_ref.data();

        // Check orthogonality of Q
        // Q' * Q  - I = 0
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, -1.0, I_ref_dat, k);
        T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

        // A - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, k, -1.0, A_dat, m);
        
        // Implementing max col norm metric
        T max_col_norm = 0.0;
        T col_norm = 0.0;
        int max_idx = 0;
        for(int i = 0; i < n; ++i) {
            col_norm = blas::nrm2(m, &A_dat[m * i], 1);
            if(max_col_norm < col_norm) {
                max_col_norm = col_norm;
                max_idx = i;
            }
        }
        T col_norm_A = blas::nrm2(n, &A_cpy_dat[m * max_idx], 1);
        T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);
        
        printf("REL NORM OF AP - QR:    %15e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %15e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I)/sqrt(n): %2e\n\n", norm_0 / std::sqrt((T) n));

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_NEAR(norm_AQR / norm_A,         0.0, atol);
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
        ASSERT_NEAR(norm_0 / std::sqrt((T) n), 0.0, atol);
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_general(
        int64_t d, 
        T norm_A,
        CQRRPTTestData<T> &all_data,
        alg_type &CQRRPT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT.call(m, n, all_data.A, d, all_data.R, all_data.J, state);

        all_data.rank = CQRRPT.rank;
        printf("RANK AS RETURNED BY CQRRPT %ld\n", all_data.rank);

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), all_data.J);

        error_check(norm_A, all_data); 

    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_full_rank_no_hqrrp) {
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 200;
    int64_t d = 400;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.num_threads = 4;
    CQRRPT.no_hqrrp = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT<double, r123::Philox4x32>>(d, norm_A, all_data, CQRRPT, state);
}

TEST_F(TestCQRRPT, CQRRPT_low_rank_with_hqrrp) {
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    int64_t d = 400;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.num_threads = 4;
    CQRRPT.no_hqrrp = 0;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT<double, r123::Philox4x32>>(d, norm_A, all_data, CQRRPT, state);
}

// Using L2 norm rank estimation here is similar to using raive estimation. 
// Fro norm underestimates rank even worse. 
TEST_F(TestCQRRPT, CQRRPT_bad_orth) {
    int64_t m = 10e4;
    int64_t n = 300;
    int64_t k = 0;
    int64_t d = 300;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.num_threads = 4;
    CQRRPT.no_hqrrp = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::adverserial);
    m_info.scaling = 1e7;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT<double, r123::Philox4x32>>(d, norm_A, all_data, CQRRPT, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRP_blocked_full_rank_no_hqrrp) {
    int64_t m = 10;
    int64_t n = 6;
    int64_t k = 6;
    int64_t d = 6;
    int64_t b_sz = 3;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, false, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRP_blocked<double, r123::Philox4x32>>(d, norm_A, all_data, CQRRP_blocked, state);
}

TEST_F(TestCQRRPT, something) {
    int64_t m = 10;
    int64_t n = 6;
    int64_t k = 6;
    auto state = RandBLAS::RNGState();

    // Allocate m by m space
    std::vector<double> A     (m * n, 0.0);
    std::vector<double> A_cpy (m * n, 0.0);
    std::vector<double> Q (m * m, 0.0);
    std::vector<double> R_buf (n * n, 0.0);
    std::vector<double> R     (m * n, 0.0);
    std::vector<double> T     (n * n, 0.0);
    std::vector<double> D     (n, 0.0);
    std::vector<double> Ident (m * m, 0.0);
    std::vector<double> tau     (n, 0.0);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A, state);

    RandLAPACK::util::eye(m, m, Q.data());
    RandLAPACK::util::eye(m, m, Ident.data());

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A.data(), m, 0.0, R_buf.data(), n);
    lapack::potrf(Uplo::Upper, n, R_buf.data(), n);
    // At this point, an m by n Q is stored in A
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_buf.data(), n, A.data(), m);

    char name [] = "A";
    
    lapack::orhr_col(m, n, n, A.data(), m, T.data(), n, D.data());

    // Apply the Q factor to some matrix on the right
    lapack::gemqrt(Side::Left, Op::NoTrans, m, m, n, n, A.data(), m, T.data(), n, Q.data(), m);

    char nameQ [] = "Q";
    RandBLAS::util::print_colmaj(m, m, Q.data(), nameQ);

    //blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, m, m, 1.0, Q.data(), m, -1.0, Ident.data(), m);

    //double norm_test_1 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, m, Ident.data(), m);
    //printf("FRO NORM OF Q'Q - I:   %e\n", norm_test_1);


/*
    std::vector<double> B (m * n, 0.0);
    std::vector<double> C (n * n, 0.0);
    RandBLAS::DenseDist Dist{.n_rows = m, .n_cols = n};
    state = RandBLAS::fill_dense(Dist, B.data(), state);

    char nameQ [] = "Q";
    RandBLAS::util::print_colmaj(m, n, Q.data(), nameQ);
    char nameB [] = "B";
    RandBLAS::util::print_colmaj(m, n, B.data(), nameB);

    // Apply the decompressed Q to some matrix
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, Q.data(), m, B.data(), m, 0.0, C.data(), n);
    char nameC [] = "C";
    RandBLAS::util::print_colmaj(n, n, C.data(), nameC);

    lapack::gemqrt(Side::Left, Op::Trans, m, n, n, n, A.data(), m, T.data(), n, B.data(), m);

    RandBLAS::util::print_colmaj(m, n, B.data(), name);
*/
}


