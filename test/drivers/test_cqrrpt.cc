#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

using Subroutines = RandLAPACK::CQRRPTSubroutines;

// ============================================================================
// Unified Testing Framework for CQRRPT
// ============================================================================

/// Matrix distribution types for test matrix generation
enum class CQRRPTMatrixDistribution {
    Polynomial,
    Adversarial
};

/// Configuration structure for CQRRPT tests
/// Encapsulates all parameters needed to run a CQRRPT test case
template <typename T>
struct CQRRPTTestConfig {
    // Matrix dimensions
    int64_t m;           ///< Number of rows
    int64_t n;           ///< Number of columns
    int64_t k;           ///< Rank (for low-rank matrices)

    // Algorithm parameters
    T d_factor;          ///< Damping factor for pivoting
    T tol;               ///< Tolerance for rank determination
    int64_t nnz = 2;     ///< Number of nonzeros in sketch (default: 2)

    // CQRRPT configuration options
    Subroutines::QRCP qrcp = Subroutines::QRCP::geqp3;
    bool orthogonalization = false;  ///< Enable orthogonalization mode

    // Matrix generation
    CQRRPTMatrixDistribution dist_type = CQRRPTMatrixDistribution::Polynomial;
    T cond_num = 2.0;    ///< Condition number (for polynomial)
    T exponent = 2.0;    ///< Decay exponent (for polynomial)
    T scaling = 1e7;     ///< Scaling factor (for adversarial)

    // Test metadata
    const char* description = "";
    bool use_orthogonalization_test = false;  ///< Use orthogonalization validation instead of standard
};

/// Helper function to create mat_gen_info from CQRRPT config
template <typename T>
RandLAPACK::gen::mat_gen_info<T> create_cqrrpt_matrix_info(const CQRRPTTestConfig<T>& config) {
    int64_t m = config.m;
    int64_t n = config.n;

    switch (config.dist_type) {
        case CQRRPTMatrixDistribution::Polynomial: {
            RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::polynomial);
            m_info.cond_num = config.cond_num;
            m_info.rank = config.k;
            m_info.exponent = config.exponent;
            return m_info;
        }

        case CQRRPTMatrixDistribution::Adversarial: {
            RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::adverserial);
            m_info.scaling = config.scaling;
            return m_info;
        }
    }

    // Should never reach here
    return RandLAPACK::gen::mat_gen_info<T>(m, n, RandLAPACK::gen::polynomial);
}

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
        
        printf("REL NORM OF AP - QR:    %15e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %15e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I)/sqrt(n): %2e\n\n", norm_0 / std::sqrt((T) n));

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
        printf("RANK AS RETURNED BY CQRRPT %ld\n", all_data.rank);

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
        printf("DETECTED RANK: %ld (expected ~%ld)\n", detected_rank, k_expected);
        printf("COLUMNS COMPLETED: %ld\n", n - detected_rank);

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
        printf("ORTHOGONALITY ERROR ||Q'Q - I||_F: %e\n", orth_error);
        printf("NORMALIZED ORTH ERROR: %e\n\n", orth_error / std::sqrt((T) n));

        // Test should pass if orthogonality is maintained
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);
        ASSERT_LE(orth_error, atol * std::sqrt((T) n));

        // Verify rank detection was reasonable (within some tolerance)
        ASSERT_GE(detected_rank, k_expected - 5);  // Allow some slack in rank detection
        ASSERT_LE(detected_rank, k_expected + 5);
    }

    /// Unified test function for CQRRPT using configuration-based approach
    /// This function encapsulates the common pattern of all CQRRPT tests
    template <typename T, typename RNG = r123::Philox4x32>
    static void test_cqrrpt_unified(const CQRRPTTestConfig<T>& config) {
        auto state = RandBLAS::RNGState<RNG>();
        T norm_A = 0;

        // Create test data container
        CQRRPTTestData<T> all_data(config.m, config.n, config.k);

        // Create and configure CQRRPT instance
        RandLAPACK::CQRRPT<T, RNG> CQRRPT(false, config.tol);
        CQRRPT.nnz = config.nnz;
        CQRRPT.qrcp = config.qrcp;
        CQRRPT.orthogonalization = config.orthogonalization;

        // Generate matrix based on distribution type
        RandLAPACK::gen::mat_gen_info<T> m_info = create_cqrrpt_matrix_info(config);
        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

        // Run test validation (standard or orthogonalization mode)
        norm_and_copy_computational_helper(norm_A, all_data);
        if (config.use_orthogonalization_test) {
            test_CQRRPT_orthogonalization(config.d_factor, norm_A, all_data, CQRRPT, state);
        } else {
            test_CQRRPT_general(config.d_factor, norm_A, all_data, CQRRPT, state);
        }
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_full_rank_no_hqrrp) {
    test_cqrrpt_unified<double>({
        .m = 10,
        .n = 5,
        .k = 5,
        .d_factor = 2.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85),
        .nnz = 2,
        .qrcp = Subroutines::QRCP::geqp3,
        .dist_type = CQRRPTMatrixDistribution::Polynomial,
        .cond_num = 2.0,
        .exponent = 2.0,
        .description = "Full rank with geqp3"
    });
}

TEST_F(TestCQRRPT, CQRRPT_low_rank_with_hqrrp) {
    test_cqrrpt_unified<double>({
        .m = 10000,
        .n = 200,
        .k = 100,
        .d_factor = 2.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85),
        .nnz = 2,
        .qrcp = Subroutines::QRCP::hqrrp,
        .dist_type = CQRRPTMatrixDistribution::Polynomial,
        .cond_num = 2.0,
        .exponent = 2.0,
        .description = "Low rank with HQRRP"
    });
}
#if !defined(__APPLE__)
TEST_F(TestCQRRPT, CQRRPT_low_rank_with_bqrrp) {
    test_cqrrpt_unified<double>({
        .m = 10000,
        .n = 200,
        .k = 100,
        .d_factor = 2.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85),
        .nnz = 2,
        .qrcp = Subroutines::QRCP::bqrrp,
        .dist_type = CQRRPTMatrixDistribution::Polynomial,
        .cond_num = 2.0,
        .exponent = 2.0,
        .description = "Low rank with BQRRP"
    });
}
#endif
// Using L2 norm rank estimation here is similar to using raive estimation.
// Fro norm underestimates rank even worse.
TEST_F(TestCQRRPT, CQRRPT_bad_orth) {
    test_cqrrpt_unified<double>({
        .m = static_cast<int64_t>(10e4),
        .n = 300,
        .k = 0,
        .d_factor = 1.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75),
        .nnz = 2,
        .qrcp = Subroutines::QRCP::geqp3,
        .dist_type = CQRRPTMatrixDistribution::Adversarial,
        .scaling = 1e7,
        .description = "Adversarial matrix (bad orthogonalization case)"
    });
}

TEST_F(TestCQRRPT, CQRRPT_orthogonalization_mode_low_rank) {
    test_cqrrpt_unified<double>({
        .m = 1000,
        .n = 100,
        .k = 60,  // True rank < n
        .d_factor = 2.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85),
        .nnz = 2,
        .qrcp = Subroutines::QRCP::geqp3,
        .orthogonalization = true,
        .dist_type = CQRRPTMatrixDistribution::Polynomial,
        .cond_num = 100.0,
        .exponent = 2.0,
        .description = "Orthogonalization mode test (completes basis)",
        .use_orthogonalization_test = true
    });
}
