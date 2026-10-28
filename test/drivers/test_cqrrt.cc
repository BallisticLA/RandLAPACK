#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <gtest/gtest.h>

using RandLAPACK::testing::QR_test_data;
using RandLAPACK::testing::norm_and_copy_computational_helper;

class TestCQRRT : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// Prints and checks the error metrics of A = QR, over all n columns of Q.
    template <typename T>
    static void
    error_check(T &norm_A, QR_test_data<T> &all_data) {

        auto n = all_data.col;
        auto [norm_AQR, max_col_norm, col_norm_A, norm_0] = RandLAPACK::testing::qr_test_errors(all_data, n, all_data.A, n);

        std::cout << "REL NORM OF A - QR:    " << std::scientific << std::setw(15) << norm_AQR / norm_A << "\n";
        std::cout << "MAX COL NORM METRIC:    " << std::scientific << std::setw(15) << max_col_norm / col_norm_A << "\n";
        std::cout << "FRO NORM OF (Q'Q - I)/sqrt(n): " << std::scientific << std::setw(2) << norm_0 / std::sqrt((T) n) << "\n\n";

        // eps^0.65 rather than eps^0.7: the orthogonality metric lands within
        // 4x of the tighter bound depending on the BLAS build (first seen with
        // vcpkg oneMKL sequential on Windows: 4.4e-11 vs eps^0.7 = 1.1e-11).
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.65);
        ASSERT_LE(norm_AQR / norm_A, atol);
        ASSERT_LE(max_col_norm / col_norm_A, atol);
        ASSERT_LE(norm_0 / std::sqrt((T) n), atol);
    }

    /// General test for CQRRT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRT_general(
        T d_factor, 
        T norm_A,
        QR_test_data<T> &all_data,
        alg_type &CQRRT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRT.call(m, n, all_data.A, m, all_data.R, n, d_factor, state);

        error_check(norm_A, all_data); 
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRT, CQRRT_full_rank_no_hqrrp) {
    int64_t m = 10;
    int64_t n = 5;
    int64_t k = 5;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRT<double, r123::Philox4x32> CQRRT(false, tol);
    CQRRT.nnz = 2;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);

    test_CQRRT_general(d_factor, norm_A, all_data, CQRRT, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRT, CQRRT_large_full_rank) {
    int64_t m = 5000;
    int64_t n = 5000;
    int64_t k = 5000;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRT<double, r123::Philox4x32> CQRRT(false, tol);
    CQRRT.nnz = 2;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);

    test_CQRRT_general(d_factor, norm_A, all_data, CQRRT, state);
}

// A single column at d_factor = 1.25 gives a one-row sketch, fewer rows than the default two
// nonzeros per column of the sparse sketching operator.
TEST_F(TestCQRRT, CQRRT_single_column) {
    int64_t m = 100;
    int64_t n = 1;
    double d_factor = 1.25;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<double> all_data(m, n, n);
    RandLAPACK::CQRRT<double, r123::Philox4x32> CQRRT(false, tol);
    CQRRT.nnz = 2;

    RandBLAS::DenseDist D(m, n);
    state = RandBLAS::fill_dense(D, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);

    test_CQRRT_general(d_factor, norm_A, all_data, CQRRT, state);
}
