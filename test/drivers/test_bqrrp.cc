#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestBQRRP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct BQRRPTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> Q;
        std::vector<T> R;
        std::vector<T> tau;
        std::vector<int64_t> J;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        BQRRPTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0),
        Q(m * n, 0.0),
        tau(n, 0.0),
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

    template <typename T>
    static void norm_and_copy_computational_helper(T &norm_A, BQRRPTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, BQRRPTestData<T> &all_data, T atol) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = all_data.rank;

        RandLAPACK::util::upsize(k * k, all_data.I_ref);
        RandLAPACK::util::eye(k, k, all_data.I_ref);

        T* A_dat           = all_data.A_cpy1.data();
        T const* A_cpy_dat = all_data.A_cpy2.data();
        T const* Q_dat     = all_data.Q.data();
        T const* R_dat     = all_data.R.data();
        T* I_ref_dat       = all_data.I_ref.data();

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
        
        printf("REL NORM OF AP - QR:    %14e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %14e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I):  %14e\n\n", norm_0 / std::sqrt((T) n));

        ASSERT_LE(norm_AQR, atol * norm_A);
        ASSERT_LE(max_col_norm, atol * col_norm_A);
        ASSERT_LE(norm_0, atol * std::sqrt((T) n));
    }

    /// General test for BQRRP:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_BQRRP_general(
        T d_factor, 
        T norm_A,
        BQRRPTestData<T> &all_data,
        alg_type &BQRRP,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);

        BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state);

        if(BQRRP.rank == 0) {
            for(int i = 0; i < m * n; ++i) {
                ASSERT_NEAR(all_data.A[i], 0.0, atol);
            }
        } else {
            all_data.rank = BQRRP.rank;
            printf("RANK AS RETURNED BY BQRRP %4ld\n", all_data.rank);

            RandLAPACK::util::upsize(all_data.rank * n, all_data.R);

            lapack::lacpy(MatrixType::Upper, all_data.rank, n, all_data.A.data(), m, all_data.R.data(), all_data.rank);

            lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.A.data(), m, all_data.tau.data());
            
            lapack::lacpy(MatrixType::General, m, all_data.rank, all_data.A.data(), m, all_data.Q.data(), m);

            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

            error_check(norm_A, all_data, atol);
        }
    }
};

#if !defined(__APPLE__)
// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_full_rank_basic) {
    int64_t m = 5000;//5000;
    int64_t n = 2000;//2000;
    int64_t k = 2000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 500;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_full_rank_block_change) {
    int64_t m = 5000;//5000;
    int64_t n = 2000;//2000;
    int64_t k = 2000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 700;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_low_rank) {
    int64_t m = 5000;
    int64_t n = 2000;
    int64_t k = 100;
    double d_factor = 2.0;
    int64_t b_sz = 200;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_pivot_qual) {
    int64_t m = std::pow(2, 10);
    int64_t n = std::pow(2, 10);
    int64_t k = std::pow(2, 10);
    double d_factor = 1.25;
    int64_t b_sz = 256;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall    = RandLAPACK::cholqr;
    BQRRP.qrcp_wide  = RandLAPACK::geqp3;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::step);
    m_info.cond_num = std::pow(10, 10);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_gemqrt) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall       = RandLAPACK::cholqr;
    BQRRP.apply_trans_q = RandLAPACK::gemqrt;
    BQRRP.internal_nb = 10;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_near_zero_input_qp3) {
    int64_t m = 1000;//5000;
    int64_t n = 1000;//2000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall   = RandLAPACK::cholqr;
    BQRRP.qrcp_wide = RandLAPACK::geqp3;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);
    all_data.A[1000*200 + 10] = 1;

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_near_zero_luqr) {
    int64_t m = 1000;
    int64_t n = 1000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);
    //all_data.A[1000*200 + 10] = 1;
    all_data.A[10*5 + 1] = 1;

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_half_zero_luqr) {
    int64_t m = 5000;//5000;
    int64_t n = 2000;//2000;
    int64_t k = 2000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 500;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(&(all_data.A.data())[m * n / 2], &(all_data.A.data())[m * n], 0.0);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_zero_mat) {
    int64_t m = 1000;//5000;
    int64_t n = 1000;//2000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

TEST_F(TestBQRRP, BQRRP_qrf) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::geqrf;
    BQRRP.internal_nb = 10;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

TEST_F(TestBQRRP, BQRRP_qrt) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::geqrt;
    BQRRP.internal_nb = 10;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

TEST_F(TestBQRRP, BQRRP_cholqr_nb) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = RandLAPACK::cholqr;
    BQRRP.internal_nb = 7;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}
#endif
