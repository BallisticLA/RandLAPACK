#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <gtest/gtest.h>

using RandLAPACK::testing::QR_householder_test_data;

class TestHQRRP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Also zeroes J.
    template <typename T>
    static void norm_and_copy_computational_helper(T &norm_A, QR_householder_test_data<T> &all_data) {
        RandLAPACK::testing::norm_and_copy_computational_helper(norm_A, all_data);
        std::fill(all_data.J, all_data.J + all_data.J_sz, 0);
    }

    /// Prints and checks the error metrics of A[:, J] = QR.
    template <typename T>
    static void
    error_check(T &norm_A, QR_householder_test_data<T> &all_data) {

        auto n = all_data.col;
        auto k = all_data.rank;
        auto [norm_AQR, max_col_norm, col_norm_A, norm_0] = RandLAPACK::testing::qr_test_errors(all_data, k, all_data.Q, k);

        std::cout << "REL NORM OF AP - QR:    " << std::scientific << std::setw(14) << norm_AQR / norm_A << "\n";
        std::cout << "MAX COL NORM METRIC:    " << std::scientific << std::setw(14) << max_col_norm / col_norm_A << "\n";
        std::cout << "FRO NORM OF (Q'Q - I):  " << std::scientific << std::setw(14) << norm_0 / std::sqrt((T) n) << "\n\n";

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_LE(norm_AQR, atol * norm_A);
        ASSERT_LE(max_col_norm, atol * col_norm_A);
        ASSERT_LE(norm_0, atol * std::sqrt((T) n));
    }

    /// General test for HQRRP:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG>
    static void test_HQRRP_general(
        T d_factor, 
        int64_t b_sz,
        int64_t use_cholqr,
        int panel_pivoting,
        T norm_A,
        QR_householder_test_data<T> &all_data,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        RandLAPACK::hqrrp(m, n, all_data.A, m, all_data.J, all_data.tau, b_sz, (int64_t) (d_factor * b_sz), panel_pivoting, use_cholqr, state, (T**) nullptr);

        RandLAPACK::testing::grow(all_data.R, all_data.R_sz, all_data.rank * n);
        lapack::lacpy(MatrixType::Upper, all_data.rank, n, all_data.A, m, all_data.R, all_data.rank);

        lapack::ungqr(m, n, n, all_data.A, m, all_data.tau);
        lapack::lacpy(MatrixType::General, m, all_data.rank, all_data.A, m, all_data.Q, m);

        // I don't think hqrrp actually returns anything.
        //std::cout << "RANK AS RETURNED BY HQRRP " << std::setw(4) << all_data.rank << "\n";

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1, m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2, m, all_data.J);

        error_check(norm_A, all_data);
             
    }
};

// This test uses orhr_col
// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestHQRRP, HQRRP_full_rank_cholqr) {
    int64_t m = 500;
    int64_t n = 200;
    int64_t k = 200;
    double d_factor = 1.0;
    int64_t b_sz = 50;
    int64_t use_cholqr = 1;
    int panel_pivoting = 0;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    QR_householder_test_data<double> all_data(m, n, k);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_HQRRP_general(d_factor, b_sz, use_cholqr, panel_pivoting, norm_A, all_data, state);
}
