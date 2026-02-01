#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

// ============================================================================
// Unified Testing Framework for CQRRT
// ============================================================================

/// Configuration structure for CQRRT tests
/// Encapsulates all parameters needed to run a CQRRT test case
template <typename T>
struct CQRRTTestConfig {
    // Matrix dimensions
    int64_t m;           ///< Number of rows
    int64_t n;           ///< Number of columns
    int64_t k;           ///< Rank (for matrix generation)

    // Algorithm parameters
    T d_factor;          ///< Damping factor
    T tol;               ///< Tolerance for algorithm
    int64_t nnz = 2;     ///< Number of nonzeros in sketch (default: 2)

    // Matrix generation (polynomial only for CQRRT)
    T cond_num = 2.0;    ///< Condition number
    T exponent = 2.0;    ///< Decay exponent

    // Test metadata
    const char* description = "";
};

/// Helper function to create mat_gen_info from CQRRT config
template <typename T>
RandLAPACK::gen::mat_gen_info<T> create_cqrrt_matrix_info(const CQRRTTestConfig<T>& config) {
    int64_t m = config.m;
    int64_t n = config.n;

    RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = config.cond_num;
    m_info.rank = config.k;
    m_info.exponent = config.exponent;
    return m_info;
}

class TestCQRRT : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRTTestData {
        int64_t row;
        int64_t col;
        int64_t rank; // has to be modifiable
        std::vector<T> A;
        std::vector<T> R;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        CQRRTTestData(int64_t m, int64_t n, int64_t k) :
        A(m * n, 0.0), 
        R(n * n, 0.0),
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
    static void norm_and_copy_computational_helper(T &norm_A, CQRRTTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, CQRRTTestData<T> &all_data) {

        auto m = all_data.row;
        auto n = all_data.col;
        auto k = n;

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
        
        printf("REL NORM OF A - QR:    %15e\n", norm_AQR / norm_A);
        printf("MAX COL NORM METRIC:    %15e\n", max_col_norm / col_norm_A);
        printf("FRO NORM OF (Q'Q - I)/sqrt(n): %2e\n\n", norm_0 / std::sqrt((T) n));

        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.7);
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
        CQRRTTestData<T> &all_data,
        alg_type &CQRRT,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, d_factor, state);

        error_check(norm_A, all_data);
    }

    /// Unified test function for CQRRT using configuration-based approach
    /// This function encapsulates the common pattern of all CQRRT tests
    template <typename T, typename RNG = r123::Philox4x32>
    static void test_cqrrt_unified(const CQRRTTestConfig<T>& config) {
        auto state = RandBLAS::RNGState<RNG>();
        T norm_A = 0;

        // Create test data container
        CQRRTTestData<T> all_data(config.m, config.n, config.k);

        // Create and configure CQRRT instance
        RandLAPACK::CQRRT<T, RNG> CQRRT(false, config.tol);
        CQRRT.nnz = config.nnz;

        // Generate matrix
        RandLAPACK::gen::mat_gen_info<T> m_info = create_cqrrt_matrix_info(config);
        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

        // Run test validation
        norm_and_copy_computational_helper(norm_A, all_data);
        test_CQRRT_general(config.d_factor, norm_A, all_data, CQRRT, state);
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRT, CQRRT_full_rank_no_hqrrp) {
    test_cqrrt_unified<double>({
        .m = 10,
        .n = 5,
        .k = 5,
        .d_factor = 2.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85),
        .nnz = 2,
        .cond_num = 2.0,
        .exponent = 2.0,
        .description = "Small full rank test"
    });
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRT, CQRRT_large_full_rank) {
    test_cqrrt_unified<double>({
        .m = 5000,
        .n = 5000,
        .k = 5000,
        .d_factor = 2.0,
        .tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85),
        .nnz = 2,
        .cond_num = 2.0,
        .exponent = 2.0,
        .description = "Large full rank test"
    });
}