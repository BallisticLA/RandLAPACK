#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "cqrrpt_test_inputs.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

using Subroutines = RandLAPACK::CQRRPTSubroutines;
using RandLAPACK::testing::QR_test_data;
using RandLAPACK::testing::norm_and_copy_computational_helper;

namespace {

constexpr int64_t bad_cholqr_rows = 64;
constexpr int64_t bad_cholqr_cols = 16;
constexpr int64_t bad_cholqr_rank = 4;

void make_bad_cholqr_matrix_with_duplicate_columns(std::vector<double>& A) {
    int64_t m = bad_cholqr_rows, n = bad_cholqr_cols, k = bad_cholqr_rank;
    double cliff = std::sqrt(std::numeric_limits<double>::epsilon());
    std::vector<double> diagonal(m * n, 0.0);
    auto state = RandBLAS::RNGState();
    RandLAPACK::gen::gen_bad_cholqr_mat(
        m, n, diagonal.data(), k, 0.5, 4.0 / cliff, true, state);
    const double expected[] = {1.0, 1.0, cliff, cliff / 4.0};
    for (int64_t ell = 0; ell < k; ++ell)
        ASSERT_EQ(diagonal[ell + ell * m], expected[ell]);

    // H64[:, 0:4]/8 and H4/2 preserve the spectrum {1, 1, 2^-26, 2^-28}.
    // Mixing this small core gives exact binary64 entries despite the sqrt(eps)
    // cliff. Copying each dense column doubles its multiplicity: rank and condition
    // number stay fixed, and every nonzero singular value grows by sqrt(2).
    static constexpr int H4[4][4] = {
        {1,  1,  1,  1},
        {1, -1,  1, -1},
        {1,  1, -1, -1},
        {1, -1, -1,  1}
    };
    A.assign(m * n, 0.0);
    for (int64_t j = 0; j < k; ++j) {
        int64_t active_col = 4 * j + 3;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t ell = 0; ell < k; ++ell)
                A[i + active_col * m] += H4[i % 4][ell] * diagonal[ell + ell * m]
                    * H4[j][ell] / 16.0;
            A[i + (active_col - 1) * m] = A[i + active_col * m];
        }
    }
}

void check_bad_cholqr_pivots(const std::vector<int64_t>& J) {
    std::vector<bool> seen(bad_cholqr_cols, false);
    bool seen_group[bad_cholqr_rank] = {};
    for (int64_t j = 0; j < bad_cholqr_cols; ++j) {
        ASSERT_GE(J[j], 1);
        ASSERT_LE(J[j], bad_cholqr_cols);
        int64_t col = J[j] - 1;
        ASSERT_FALSE(seen[col]);
        seen[col] = true;
        if (j < bad_cholqr_rank) {
            // Either copy is valid, but all four independent directions are needed.
            ASSERT_GE(col % 4, 2);
            ASSERT_FALSE(seen_group[col / 4]);
            seen_group[col / 4] = true;
        }
    }
}

void check_bad_cholqr_reconstruction(
    const std::vector<double>& original, const std::vector<double>& Q,
    const std::vector<double>& R, const std::vector<int64_t>& J, double tol
) {
    int64_t m = bad_cholqr_rows, n = bad_cholqr_cols, k = bad_cholqr_rank;
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < k; ++i)
            ASSERT_TRUE(std::isfinite(R[i + j * n]));
    }
    // Gather AP directly; using the production column-permutation helper here
    // could hide a permutation error shared by the algorithm and its oracle.
    std::vector<double> residual(m * n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i)
            residual[i + j * m] = original[i + (J[j] - 1) * m];
    }
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
        -1.0, Q.data(), m, R.data(), n, 1.0, residual.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, original.data(), m);
    double relative_residual = lapack::lange(Norm::Fro, m, n, residual.data(), m) / norm_A;
    ASSERT_LE(relative_residual, tol);
}

void check_bad_cholqr_orthogonality(const std::vector<double>& Q, double tol) {
    int64_t m = bad_cholqr_rows, k = bad_cholqr_rank;
    for (int64_t j = 0; j < k; ++j) {
        for (int64_t i = 0; i < m; ++i)
            ASSERT_TRUE(std::isfinite(Q[i + j * m]));
    }
    std::vector<double> gram(k * k, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
        1.0, Q.data(), m, Q.data(), m, 0.0, gram.data(), k);
    for (int64_t i = 0; i < k; ++i)
        gram[i + i * k] -= 1.0;
    double orthogonality_error = lapack::lange(Norm::Fro, k, k, gram.data(), k) / std::sqrt(double(k));
    ASSERT_LE(orthogonality_error, tol);
}

} // namespace

class TestCQRRPT : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// Prints and checks the error metrics of A[:, J] = QR.
    template <typename T>
    static void
    error_check(T &norm_A, QR_test_data<T> &all_data) {

        auto n = all_data.col;
        auto [norm_AQR, max_col_norm, col_norm_A, norm_0] = RandLAPACK::testing::qr_test_errors(all_data, all_data.rank, all_data.A, n);

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
        QR_test_data<T> &all_data,
        alg_type &CQRRPT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state);

        all_data.rank = CQRRPT.rank;
        std::cout << "RANK AS RETURNED BY CQRRPT " << all_data.rank << "\n";

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1, m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2, m, all_data.J);

        error_check(norm_A, all_data);
    }

    /// Test for CQRRPT in orthogonalization mode:
    /// Verifies that when input is low-rank and orthogonalization mode is enabled,
    /// CQRRPT completes the orthonormal basis by filling remaining columns.
    /// Checks that all n columns form an orthonormal set (Q'Q = I).
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_orthogonalization(
        T d_factor,
        QR_test_data<T> &all_data,
        alg_type &CQRRPT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k_expected = all_data.rank;  // Expected rank from matrix generation

        CQRRPT.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state);

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
                   1.0, all_data.A, m,
                   all_data.A, m,
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

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}

TEST_F(TestCQRRPT, CQRRPT_bad_cholqr_exact_rank_deficient) {
    int64_t m = bad_cholqr_rows, n = bad_cholqr_cols, k = bad_cholqr_rank;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    std::vector<double> original;
    ASSERT_NO_FATAL_FAILURE(make_bad_cholqr_matrix_with_duplicate_columns(original));

    for (uint32_t seed = 0; seed < 8; ++seed) {
        SCOPED_TRACE(seed);
        auto Q = original;
        std::vector<double> R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<> state(seed);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 4;
        CQRRPT.qrcp = Subroutines::QRCP::geqp3;
        ASSERT_EQ(CQRRPT.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        ASSERT_EQ(CQRRPT.rank, k);

        ASSERT_NO_FATAL_FAILURE(check_bad_cholqr_pivots(J));
        ASSERT_NO_FATAL_FAILURE(check_bad_cholqr_reconstruction(original, Q, R, J, tol));
        ASSERT_NO_FATAL_FAILURE(check_bad_cholqr_orthogonality(Q, tol));
    }
}

TEST_F(TestCQRRPT, CQRRPT_low_rank_with_hqrrp) {
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::hqrrp;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}
TEST_F(TestCQRRPT, CQRRPT_low_rank_with_bqrrp) {
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    double d_factor = 2;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::bqrrp;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}

// geqp3 reads jpvt on entry (nonzero marks a fixed column), so the prior
// contents of the caller's J buffer must not influence pivoting. Run the same
// rank-deficient factorization with a clean J and with two dirtied J buffers,
// at the same RNG state each time; rank and factorization quality must match.
TEST_F(TestCQRRPT, CQRRPT_dirty_J_rank_deficient) {
    int64_t m = 2000;
    int64_t n = 50;
    int64_t k = 40;
    double d_factor = 2;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate the rank-deficient input once.
    auto gen_state = RandBLAS::RNGState();
    std::vector<double> A_orig(m * n, 0.0);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, A_orig.data(), gen_state);

    int64_t rank_ref = -1;
    for (int trial = 0; trial < 3; ++trial) {
        QR_test_data<double> all_data(m, n, k);
        lapack::lacpy(MatrixType::General, m, n, A_orig.data(), m, all_data.A, m);

        // Trial 0 keeps the zero-initialized J; trials 1 and 2 dirty it with
        // different nonzero garbage.
        if (trial > 0) {
            for (int64_t i = 0; i < n; ++i)
                all_data.J[i] = 1 + ((7919 * trial + 31 * i) % n);
        }

        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        CQRRPT.qrcp = Subroutines::QRCP::geqp3;

        double norm_A = 0;
        norm_and_copy_computational_helper(norm_A, all_data);
        // Same RNG state in every trial, so any difference is due to J alone.
        auto state = RandBLAS::RNGState();
        test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);

        if (trial == 0)
            rank_ref = all_data.rank;
        ASSERT_EQ(all_data.rank, rank_ref);
    }
}

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

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::adverserial);
    m_info.scaling = 1e7;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

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

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;
    CQRRPT.orthogonalization = true;  // Enable orthogonalization mode

    // Generate a low-rank matrix (rank k < n)
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 100;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_orthogonalization(d_factor, all_data, CQRRPT, state);
}

// After a failed potrf, CQRRPT cuts the rank, and the leading rank rows of R must be built
// from computed data. With Q (nearly) orthonormal, the truncated factorization then cannot
// leave a residual larger than A itself. The input defeats the sketch, so CholQR fails.
TEST_F(TestCQRRPT, CQRRPT_cholesky_failure_R_consistent) {
    int64_t m = 600;
    int64_t n = 60;
    uint32_t seed = 7;
    std::vector<double> A = cqrrpt_test::sketch_defeating_input(m, n, 2, 2.0, seed, 1.0);
    double norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);

    for (double tol : {std::pow(std::numeric_limits<double>::epsilon(), 0.85), 1e-6}) {
        SCOPED_TRACE(tol);
        std::vector<double> Q(A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        ASSERT_EQ(CQRRPT.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        int64_t r = CQRRPT.rank;
        ASSERT_LT(r, n) << "input no longer makes CholQR fail; the test does not exercise the failure path";

        // ||A[:, J] - Q_r R_r||_F / ||A||_F
        std::vector<double> AP(A);
        RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
        double err_qr = lapack::lange(Norm::Fro, m, n, AP.data(), m) / norm_A;
        EXPECT_LE(err_qr, 1.0) << "rank " << r;
    }
}

// util::cholqr_failure_rank, the rule of both drivers, on synthetic diagonals with potrf failed at
// column info: the rank must not depend on the failed pivot or anything past it, and must be the same
// for a contiguous diagonal (as CQRRPT_GPU passes it) and the diagonal of a padded matrix (CQRRPT).
TEST_F(TestCQRRPT, CQRRPT_cholqr_failure_rank_rule) {
    auto check = [](auto zero, auto tol) {
        using T = decltype(zero);
        const T thr = std::sqrt(tol / std::numeric_limits<T>::epsilon());
        const T inf = std::numeric_limits<T>::infinity();
        const T nan = std::numeric_limits<T>::quiet_NaN();
        const T huge = std::numeric_limits<T>::max();
        // Completed entries, with max/min 1.5, far below the ratio threshold thr.
        const T base[] = {1.0, 0.9, 1.1, 0.8, 1.2, 0.95, 1.05, 0.85};
        // Diagonal base[0:k], with base[pos] replaced by value if pos >= 0.
        struct Case { const char* name; int64_t k, info, pos; T value; int64_t rank; };
        const Case cases[] = {
            {"no failure",                          8, 0, -1, 0,             8},
            {"info = 1",                            8, 1, -1, 0,             0},
            {"info = 2",                            8, 2, -1, 0,             1},
            {"info = 3",                            8, 3, -1, 0,             1},
            {"info = 5",                            8, 5, -1, 0,             3},
            {"info = k",                            8, 8, -1, 0,             6},
            {"k = 2, info = 2",                     2, 2, -1, 0,             1},
            {"k = 1, info = 1",                     1, 1, -1, 0,             0},
            {"small entry cuts before failure",     8, 7,  3, 1 / (4 * thr), 2},
            {"large entry cuts before failure",     8, 7,  4, 4 * thr,       3},
            {"ratio below threshold, no cut",       8, 7,  3, 4 / thr,       5},
            {"ratio trips at i = 1, cut at i = 2",  8, 6,  1, 1 / (4 * thr), 1},
        };
        for (const auto& c : cases) {
            for (T failed : {-huge, (T) -1, (T) 0, nan}) {
                for (T past : {nan, inf, -inf, (T) 0, (T) 1, huge, 1 / (4 * thr)}) {
                    SCOPED_TRACE(::testing::Message() << (sizeof(T) == 8 ? "double, " : "float, ") << c.name
                                                      << ", failed pivot " << failed << ", entries past it " << past);
                    std::vector<T> diag(base, base + c.k);
                    if (c.pos >= 0)
                        diag[c.pos] = c.value;
                    if (c.info > 0) {
                        diag[c.info - 1] = failed;
                        for (int64_t i = c.info; i < c.k; ++i)
                            diag[i] = past;
                    }
                    int64_t ldr = c.k + 3;
                    std::vector<T> R(ldr * c.k, nan);
                    for (int64_t i = 0; i < c.k; ++i)
                        R[i + i * ldr] = diag[i];
                    EXPECT_EQ(RandLAPACK::util::cholqr_failure_rank(c.k, diag.data(), (int64_t) 1, c.info, (T) tol), c.rank);
                    EXPECT_EQ(RandLAPACK::util::cholqr_failure_rank(c.k, R.data(), ldr + 1, c.info, (T) tol), c.rank);
                }
            }
        }
    };
    check(0.0, 1e-6);
    check(0.0f, 1e-5f);
}

// A full-rank m-by-2 A on which potrf fails at column 2, exactly and on any BLAS (built by
// cholesky_failure_k2_input from the count sketch, nnz = 1). With k = 2 the ratio scan, which
// starts at i = 2, cannot cut the rank: CQRRPT must still drop the failed column.
TEST_F(TestCQRRPT, CQRRPT_cholesky_failure_k2_drops_failed_column) {
    int64_t m = 32;
    int64_t n = 2;
    int64_t d = 4;
    uint32_t seed = 3;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A = cqrrpt_test::cholesky_failure_k2_input(m, d, seed);
    ASSERT_EQ((int64_t) A.size(), m * n) << "the sketch from this seed cannot carry the input";

    std::vector<double> Q(A), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state(seed);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 1;
    CQRRPT.qrcp = Subroutines::QRCP::geqp3;
    ASSERT_EQ(CQRRPT.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    ASSERT_EQ(J, (std::vector<int64_t>{1, 2})) << "the sketch's QRCP must not pivot";
    EXPECT_EQ(CQRRPT.rank, 1);

    // Q(:, 0:rank) orthonormal, and A[:, J] = Q R up to the dropped direction, of relative size 2e-9.
    int64_t r = CQRRPT.rank;
    bool finite = true;
    for (int64_t j = 0; j < r; ++j)
        for (int64_t i = 0; i < m; ++i)
            finite = finite && std::isfinite(Q[i + j * m]);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < r; ++i)
            finite = finite && std::isfinite(R[i + j * n]);
    ASSERT_TRUE(finite) << "rank " << r << ": Q or R contains NaN or Inf";
    std::vector<double> G(r * r, 0.0);
    RandLAPACK::util::eye(r, r, G.data());
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, 1.0, Q.data(), m, -1.0, G.data(), r);
    double err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, G.data(), r);
    EXPECT_LE(err_orth, 10 * std::numeric_limits<double>::epsilon()) << "rank " << r;
    std::vector<double> AP(A);
    RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
    double err_qr = lapack::lange(Norm::Fro, m, n, AP.data(), m) / lapack::lange(Norm::Fro, m, n, A.data(), m);
    EXPECT_LE(err_qr, 1e-8) << "rank " << r;
}

// R's contents on entry must not matter: with R filled with NaN, CQRRPT must return exactly
// the R it returns for a zeroed R, and that R is zero below the diagonal and past the rank.
TEST_F(TestCQRRPT, CQRRPT_R_contents_on_entry_ignored) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    struct Input { const char* name; int64_t m, n; std::vector<double> A; uint32_t seed; };
    std::vector<Input> inputs;
    for (int64_t rank_A : {(int64_t) 50, (int64_t) 40}) {
        int64_t m = 2000, n = 50;
        std::vector<double> A(m * n, 0.0);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
        m_info.cond_num = 2;
        m_info.rank = rank_A;
        m_info.exponent = 2.0;
        auto gen_state = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), gen_state);
        inputs.push_back({rank_A == n ? "full rank" : "rank 40", m, n, A, 0});
    }
    inputs.push_back({"failed potrf", 600, 60, cqrrpt_test::sketch_defeating_input(600, 60, 2, 2.0, 7, 1.0), 7});

    for (auto& in : inputs) {
        SCOPED_TRACE(in.name);
        int64_t m = in.m, n = in.n;
        std::vector<double> Q0(in.A), R0(n * n, 0.0), Q1(in.A), R1(n * n, std::numeric_limits<double>::quiet_NaN());
        std::vector<int64_t> J0(n, 0), J1(n, 0);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        RandBLAS::RNGState<r123::Philox4x32> state0(in.seed), state1(in.seed);
        ASSERT_EQ(CQRRPT.call(m, n, Q0.data(), m, R0.data(), n, J0.data(), 2.0, state0), 0);
        int64_t rank = CQRRPT.rank;
        ASSERT_EQ(CQRRPT.call(m, n, Q1.data(), m, R1.data(), n, J1.data(), 2.0, state1), 0);
        ASSERT_EQ(CQRRPT.rank, rank);

        int64_t mismatched = 0, nonzero = 0;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i) {
                mismatched += !(R1[i + j * n] == R0[i + j * n]);
                nonzero += (i > j || i >= rank) && R1[i + j * n] != 0.0;
            }
        EXPECT_EQ(mismatched, 0) << "entries of R that depend on its contents on entry";
        EXPECT_EQ(nonzero, 0) << "nonzero entries below the diagonal or past rank " << rank;
        EXPECT_EQ(Q1, Q0);
    }
}

// On an all-zero input CQRRPT must report rank 0, whatever rank held before the call, and
// return R as documented (zero) with A left zero.
TEST_F(TestCQRRPT, CQRRPT_zero_input) {
    int64_t m = 100;
    int64_t n = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(m * n, 0.0), R(n * n, std::numeric_limits<double>::quiet_NaN());
    std::vector<int64_t> J(n, 0);

    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    EXPECT_EQ(CQRRPT.rank, 0);
    CQRRPT.nnz = 2;
    CQRRPT.rank = -7;
    auto state = RandBLAS::RNGState();
    ASSERT_EQ(CQRRPT.call(m, n, A.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    EXPECT_EQ(CQRRPT.rank, 0);
    EXPECT_EQ(R, std::vector<double>(n * n, 0.0));
    EXPECT_EQ(A, std::vector<double>(m * n, 0.0));
}
