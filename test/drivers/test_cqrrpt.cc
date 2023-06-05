#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

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
        J(n, 0.0),  
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
    static void norm_and_copy_computational_helper(T& norm_A, CQRRPTTestData<T>& all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T& norm_A, CQRRPTTestData<T>& all_data) {

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
        
        printf("REL NORM OF AP - QR: %15e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC: %15e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF Q' * Q - I: %2e\n\n", norm_0);

        ASSERT_NEAR(norm_AQR / norm_A,         0.0, 10 * std::pow(10, -13));
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, 10 * std::pow(10, -13));
        ASSERT_NEAR(norm_0,                    0.0, 10 * std::pow(10, -13));
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG>
    static void test_CQRRPT_general(
        int64_t d, 
        T norm_A,
        CQRRPTTestData<T>& all_data,
        RandLAPACK::CQRRPT<T, RNG>& CQRRPT) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT.call(m, n, all_data.A, d, all_data.R, all_data.J);

        all_data.rank = CQRRPT.rank;

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2, all_data.J);

        error_check(norm_A, all_data); 
    }
};

// Note: If Subprocess killed exception -> reload vscode
/*
TEST_F(TestCQRRPT, CQRRPT_full_rank_no_hqrrp)
{
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 200;
    int64_t d = 400;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, state, tol);
    CQRRPT.nnz = 2;
    CQRRPT.num_threads = 4;
    CQRRPT.no_hqrrp = 1;

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32>(d, norm_A, all_data, CQRRPT);
}
*/
/*
TEST_F(TestCQRRPT, CQRRPT_low_rank_with_hqrrp)
{
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    int64_t d = 400;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, state, tol);
    CQRRPT.nnz = 2;
    CQRRPT.num_threads = 4;
    CQRRPT.no_hqrrp = 0;

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32>(d, norm_A, all_data, CQRRPT);
}
*/

TEST_F(TestCQRRPT, CQRRPT_bad_orth)
{
    int64_t m = 10e4;
    int64_t n = 300;
    int64_t k = 0;
    int64_t d = 300;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, false, state, tol);
    CQRRPT.nnz = 2;
    CQRRPT.num_threads = 4;
    CQRRPT.no_hqrrp = 1;

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(9, 1e7, false));
    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32>(d, norm_A, all_data, CQRRPT);
}

/*
//shows that row_resize works well
TEST_F(TestCQRRPT, sanity_check)
{
    int64_t m = 5;
    int64_t n = 5;
    int64_t k = 5;
    auto state = RandBLAS::base::RNGState();
    std::vector<double> A (m * n, 0.0);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, A, k, state, std::make_tuple(0, 2, false));
    char name [] = "A";
    RandBLAS::util::print_colmaj(m, n, A.data(), name);
    RandLAPACK::util::row_resize(m, n, A, 3);
    RandBLAS::util::print_colmaj(3, n, A.data(), name);
    
}
*/