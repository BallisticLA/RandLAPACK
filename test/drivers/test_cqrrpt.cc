#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

using Subroutines = RandLAPACK::CQRRPTSubroutines;

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
        R(n * n, 0.0),
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
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, n, -1.0, A_dat, m);
        
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
        
        std::cout << "REL NORM OF AP - QR:    " << std::scientific << std::setw(15) << norm_AQR / norm_A << "\n";
        std::cout << "MAX COL NORM METRIC:    " << std::scientific << std::setw(15) << max_col_norm / col_norm_A << "\n";
        std::cout << "FRO NORM OF (Q'Q - I)/sqrt(n): " << std::scientific << std::setw(2) << norm_0 / std::sqrt((T) n) << "\n\n";

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_LE(norm_AQR, atol * norm_A);
        ASSERT_LE(max_col_norm, atol * col_norm_A);
        ASSERT_LE(norm_0, atol * std::sqrt((T) n));
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_general(
        T d_factor,
        T norm_A,
        CQRRPTTestData<T> &all_data,
        alg_type &CQRRPT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state);

        all_data.rank = CQRRPT.rank;
        std::cout << "RANK AS RETURNED BY CQRRPT " << all_data.rank << "\n";

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

        error_check(norm_A, all_data);
    }

    /// Test for CQRRPT in orthogonalization mode:
    /// Verifies that when input is low-rank and orthogonalization mode is enabled,
    /// CQRRPT completes the orthonormal basis by filling remaining columns.
    /// Checks that all n columns form an orthonormal set (Q'Q = I).
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_orthogonalization(
        T d_factor,
        T norm_A,
        CQRRPTTestData<T> &all_data,
        alg_type &CQRRPT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k_expected = all_data.rank;  // Expected rank from matrix generation

        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state);

        int64_t detected_rank = CQRRPT.rank;
        std::cout << "DETECTED RANK: " << detected_rank << " (expected ~" << k_expected << ")\n";
        std::cout << "COLUMNS COMPLETED: " << n - detected_rank << "\n";

        // Verify that all n columns of A form an orthonormal set
        // Compute Q'Q where Q is all n columns of A
        std::vector<T> QtQ(n * n, 0.0);
        std::vector<T> I_ref(n * n, 0.0);
        RandLAPACK::util::eye(n, n, I_ref);

        // QtQ = A' * A
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                   n, n, m,
                   1.0, all_data.A.data(), m,
                   all_data.A.data(), m,
                   0.0, QtQ.data(), n);

        // QtQ = QtQ - I
        blas::axpy(n * n, -1.0, I_ref.data(), 1, QtQ.data(), 1);

        // Check || Q'Q - I ||_F
        T orth_error = lapack::lange(Norm::Fro, n, n, QtQ.data(), n);
        std::cout << "ORTHOGONALITY ERROR ||Q'Q - I||_F: " << std::scientific << orth_error << "\n";
        std::cout << "NORMALIZED ORTH ERROR: " << std::scientific << orth_error / std::sqrt((T) n) << "\n\n";

        // Test should pass if orthogonality is maintained
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_LE(orth_error, atol * std::sqrt((T) n));

        // Verify rank detection was reasonable (within some tolerance)
        ASSERT_GE(detected_rank, k_expected - 5);  // Allow some slack in rank detection
        ASSERT_LE(detected_rank, k_expected + 5);
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_full_rank_no_hqrrp) {
    int64_t m = 10;
    int64_t n = 5;
    int64_t k = 5;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}

TEST_F(TestCQRRPT, CQRRPT_low_rank_with_hqrrp) {
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::hqrrp;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}
#if !defined(__APPLE__)
TEST_F(TestCQRRPT, CQRRPT_low_rank_with_bqrrp) {
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::bqrrp;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}
#endif
// Using L2 norm rank estimation here is similar to using raive estimation.
// Fro norm underestimates rank even worse.
TEST_F(TestCQRRPT, CQRRPT_bad_orth) {
    int64_t m = 10e4;
    int64_t n = 300;
    int64_t k = 0;
    double d_factor = 1;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::adverserial);
    m_info.scaling = 1e7;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}

TEST_F(TestCQRRPT, CQRRPT_orthogonalization_mode_low_rank) {
    int64_t m = 1000;
    int64_t n = 100;
    int64_t k = 60;  // True rank < n
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;
    CQRRPT.orthogonalization = true;  // Enable orthogonalization mode

    // Generate a low-rank matrix (rank k < n)
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 100;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_orthogonalization(d_factor, norm_A, all_data, CQRRPT, state);
}
