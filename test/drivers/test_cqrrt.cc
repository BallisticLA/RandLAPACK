#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>

#include <RandBLAS.hh>
#include <algorithm>
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

// The finalize trmm reads R in full, so a garbage strict lower triangle in the caller's R buffer
// must not reach the result: a prefilled run reproduces a zero-filled one to rounding (the two
// runs use separate buffers), and R's strict lower triangle comes back exactly zero.
TEST_F(TestCQRRT, CQRRT_garbage_R_lower_unaffected) {
    int64_t m = 80, n = 25;
    double d_factor = 2.0;
    std::vector<double> A_clean(m * n);
    RandBLAS::RNGState<> state(7);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A_clean.data(), state);
    std::vector<double> A_garbage = A_clean;

    std::vector<double> R_clean(n * n, 0.0), R_garbage(n * n, 0.0);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            R_garbage[i + j * n] = 1.0e300;

    RandLAPACK::CQRRT<double> alg(false, 0.0);
    RandBLAS::RNGState<> sk_clean(9), sk_garbage(9);
    ASSERT_EQ(alg.call(m, n, A_clean.data(), m, R_clean.data(), n, d_factor, sk_clean), 0);
    ASSERT_EQ(alg.call(m, n, A_garbage.data(), m, R_garbage.data(), n, d_factor, sk_garbage), 0);

    using RandLAPACK::testing::rel_fro_error;
    EXPECT_LE(rel_fro_error(R_garbage.data(), R_clean.data(), n * n), 1e-12);
    EXPECT_LE(rel_fro_error(A_garbage.data(), A_clean.data(), m * n), 1e-12);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            ASSERT_EQ(R_garbage[i + j * n], 0.0) << "R(" << i << "," << j << ") nonzero";
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            ASSERT_EQ(R_clean[i + j * n], 0.0) << "R(" << i << "," << j << ") nonzero";
}

// times holds 10 entries after a timed success and is cleared by a call that throws on
// validation and by a failing one: a zero column sketches to a zero column, so the sketch's R
// is singular and call() returns 1.
TEST_F(TestCQRRT, CQRRT_times_cleared_on_failure) {
    int64_t m = 60, n = 12;
    std::vector<double> A(m * n), R(n * n, 0.0);
    RandBLAS::RNGState<> state(3);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);
    std::vector<double> A_singular = A;
    std::fill(A_singular.begin() + 4 * m, A_singular.begin() + 5 * m, 0.0);

    RandLAPACK::CQRRT<double> alg(/*timing=*/true, 0.0);
    std::vector<double> A_copy = A;
    ASSERT_EQ(alg.call(m, n, A.data(), m, R.data(), n, 2.0, state), 0);
    ASSERT_EQ(alg.times.size(), 10u);
    for (int i = 0; i < 10; ++i) EXPECT_GE(alg.times[i], 0) << "slot " << i;
    EXPECT_EQ(alg.total_us(), alg.times[9]);
    EXPECT_THROW(alg.call(m, n, A_copy.data(), m, R.data(), n, 0.5, state), RandLAPACK::Error);
    EXPECT_TRUE(alg.times.empty());

    ASSERT_EQ(alg.call(m, n, A_copy.data(), m, R.data(), n, 2.0, state), 0);
    ASSERT_EQ(alg.call(m, n, A_singular.data(), m, R.data(), n, 2.0, state), 1);
    EXPECT_TRUE(alg.times.empty());
    EXPECT_EQ(alg.total_us(), -1L);
}

TEST_F(TestCQRRT, CQRRT_invalid_input_throws) {
    int64_t m = 10, n = 20;
    std::vector<double> A(m * n, 1.0), R(n * n, 0.0);
    RandBLAS::RNGState<> state(0);
    RandLAPACK::CQRRT<double> alg(false, 0.0);
    EXPECT_THROW(alg.call(m, n, A.data(), m, R.data(), n, 2.0, state), RandLAPACK::Error);   // m < n
    EXPECT_THROW(alg.call(n, m, A.data(), n, R.data(), m, 0.5, state), RandLAPACK::Error);   // d_factor < 1
    EXPECT_THROW(alg.call(n, m, A.data(), n, R.data(), m, std::numeric_limits<double>::infinity(), state),
                 RandLAPACK::Error);                                                          // d_factor = inf
    EXPECT_THROW(alg.call(n, 0, A.data(), n, R.data(), 1, 2.0, state), RandLAPACK::Error);   // n = 0
}

// cqrrt_sketch_rows rounds d_factor * n in T and truncates (1.4f * 5 is 7 in float, 6.99999988 in
// double), raises it to n when float rounds n itself down (16777217 is 16777216 in float), and
// throws when the product is negative, infinite or reaches 2^63.
TEST_F(TestCQRRT, CQRRT_sketch_rows) {
    EXPECT_EQ(RandLAPACK::cqrrt_sketch_rows(1.4f, 5), 7);
    EXPECT_EQ(RandLAPACK::cqrrt_sketch_rows((double)1.4f, 5), 6);
    EXPECT_EQ(RandLAPACK::cqrrt_sketch_rows(1.0f, 16777217), 16777217);
    EXPECT_EQ(RandLAPACK::cqrrt_sketch_rows(0x1.fffffffffffffp62, 1), (int64_t)0x1.fffffffffffffp62);
    EXPECT_EQ(RandLAPACK::cqrrt_sketch_rows(0x1.fffffep62f, 1), (int64_t)0x1.fffffep62f);
    EXPECT_THROW(RandLAPACK::cqrrt_sketch_rows(0x1p63, 1), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::cqrrt_sketch_rows(0x1p63f, 1), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::cqrrt_sketch_rows(0x1p62, 2), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::cqrrt_sketch_rows(std::numeric_limits<double>::infinity(), 1), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::cqrrt_sketch_rows(std::numeric_limits<double>::quiet_NaN(), 1), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::cqrrt_sketch_rows(-1.0, 1), RandLAPACK::Error);
}

// A finite d_factor whose d_factor * n does not fit in int64_t throws before anything is
// allocated or the sketch is drawn.
TEST_F(TestCQRRT, CQRRT_huge_d_factor_throws_before_the_sketch) {
    int64_t m = 4, n = 1;
    std::vector<double> A(m, 1.0), R(1, -3.0);
    RandLAPACK::CQRRT<double> alg(false, 0.0);
    RandBLAS::RNGState<> state(47);
    auto before = state;
    EXPECT_THROW(alg.call(m, n, A.data(), m, R.data(), n, 0x1p63, state), RandLAPACK::Error);
    EXPECT_TRUE(state == before);
    EXPECT_EQ(R[0], -3.0);
    for (double v : A) EXPECT_EQ(v, 1.0);
}

// d_factor * n rounds in T, as in CQRRTO_linops: 1.4f * 5 is 7 in float. With m = n < d the
// sketch's RNG use counts its d rows, so the state after a call shows the driver's d, which
// must be the memory model's.
TEST_F(TestCQRRT, CQRRT_sketch_rows_match_the_model) {
    for (auto [d_factor, n] : {std::pair<float, int64_t>{1.4f, 5}, {1.3f, 10}, {1.1f, 10}, {2.0f, 7}}) {
        int64_t m = n, d = RandLAPACK::cqrrt_sketch_rows(d_factor, n);
        std::vector<float> A(m * n), R(n * n, 0.0f);
        RandBLAS::RNGState<> state(41);
        RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);
        RandLAPACK::CQRRT<float> alg(false, 0.0f);
        RandBLAS::SparseSkOp<float> S(RandBLAS::SparseDist(d, m, std::min({alg.nnz, d, m})), state);
        alg.call(m, n, A.data(), m, R.data(), n, d_factor, state);
        EXPECT_TRUE(state == S.next_state) << "d_factor " << d_factor << ", n " << n << ", d " << d;
    }
}

// A NaN entry of A reaches the Gram's Cholesky factor: call() returns 1 on every LAPACK, not only
// on those whose potrf reports a NaN pivot.
TEST_F(TestCQRRT, CQRRT_nan_input_fails) {
    int64_t m = 60, n = 12;
    std::vector<double> A(m * n), R(n * n, 0.0);
    RandBLAS::RNGState<> state(5);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);
    A[7 + 3 * m] = std::numeric_limits<double>::quiet_NaN();
    RandLAPACK::CQRRT<double> alg(false, 0.0);
    EXPECT_EQ(alg.call(m, n, A.data(), m, R.data(), n, 2.0, state), 1);
}
