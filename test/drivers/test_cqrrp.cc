#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestCQRRP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRPTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> Q;
        std::vector<T> R;
        std::vector<int64_t> J;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        CQRRPTestData(int64_t m, int64_t n, int64_t k) :
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
    static void norm_and_copy_computational_helper(T &norm_A, CQRRPTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, CQRRPTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat         = all_data.A_cpy1.data();
        T const* A_cpy_dat = all_data.A_cpy2.data();
        T const* Q_dat   = all_data.Q.data();
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
        
        printf("REL NORM OF AP - QR:    %19e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %19e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I)/sqrt(n): %2e\n\n", norm_0 / std::sqrt((T) n));

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_NEAR(norm_AQR / norm_A,         0.0, atol);
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
        ASSERT_NEAR(norm_0 / std::sqrt((T) n), 0.0, atol);
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRP_general(
        int64_t d_factor, 
        T norm_A,
        CQRRPTestData<T> &all_data,
        alg_type &CQRRP,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRP.call(m, n, all_data.A, d_factor, all_data.Q, all_data.R, all_data.J, state);

        all_data.rank = CQRRP.rank;
        printf("RANK AS RETURNED BY CQRRP %9ld\n", all_data.rank);

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), all_data.J);

        error_check(norm_A, all_data); 

    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, CQRRP_blocked_full_rank_no_hqrrp) {
    int64_t m = 8;
    int64_t n = 6;
    int64_t k = 6;
    int64_t d_factor = 4.0;
    int64_t b_sz = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRP_blocked<double, r123::Philox4x32> CQRRP_blocked(false, true, tol, b_sz);
    CQRRP_blocked.nnz = 2;
    CQRRP_blocked.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRP_general<double, r123::Philox4x32, RandLAPACK::CQRRP_blocked<double, r123::Philox4x32>>(d_factor, norm_A, all_data, CQRRP_blocked, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRP, Something2) {
    int64_t m = 4;
    int64_t n = 4;

    std::vector<double> A = {0, 0, 0, 1,
                             0, 0, 1, 0,
                             0, 0, 1, 0,
                             0, 0, 1, 0};
    std::vector<double> B = {1, 1,
                             1, 1};
    std::vector<double> C (2 * 2, 0.0);

    double* A_dat = A.data();
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 2, 2, 1.0, &A_dat[10], 2, B.data(), 2, 0.0, C.data(), 2);

        char name1 [] = "A";
    RandBLAS::util::print_colmaj(4, 4, A.data(), name1);

        char name2 [] = "B";
    RandBLAS::util::print_colmaj(2, 2, B.data(), name2);

    char name [] = "C";
    RandBLAS::util::print_colmaj(2, 2, C.data(), name);
}
