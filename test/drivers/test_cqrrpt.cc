#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

using Subroutines = RandLAPACK::CQRRPTSubroutines;

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
        T col_norm_A = blas::nrm2(m, &A_cpy_dat[m * max_idx], 1);
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

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J.data());
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J.data());

        error_check(norm_A, all_data);
    }

    /// Test for CQRRPT in orthogonalization mode:
    /// Verifies that when input is low-rank and orthogonalization mode is enabled,
    /// CQRRPT completes the orthonormal basis by filling remaining columns.
    /// Checks that all n columns form an orthonormal set (Q'Q = I).
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_orthogonalization(
        T d_factor,
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
        CQRRPTTestData<double> all_data(m, n, k);
        lapack::lacpy(MatrixType::General, m, n, A_orig.data(), m, all_data.A.data(), m);

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
    test_CQRRPT_orthogonalization(d_factor, all_data, CQRRPT, state);
}
