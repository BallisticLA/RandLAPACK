#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


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
        std::vector<T> A_row;
        std::vector<T> R;
        std::vector<int64_t> J;
        std::vector<T> A_cpy1;
        std::vector<T> A_cpy2;
        std::vector<T> I_ref;

        Layout layout;

        CQRRPTTestData(int64_t m, int64_t n, int64_t k, Layout layout=Layout::ColMajor):
        row(m),
        col(n),
        rank(k),
        A(m * n, 0.0),
        A_row(m * n, 0.0),
        R(n * n, 0.0),
        J(n, 0),
        A_cpy1(m * n, 0.0),
        A_cpy2(m * n, 0.0),
        A_cpy3(m * n, 0.0),
        I_ref(k * k, 0.0),
        layout(layout)
        {}

    };

    template<typename T>
    void ColMajor2RowMajor(
        CQRRPTTestData<T> &all_data
    ) {
        // Iterate over each row for the output format
        for (int64_t i = 0; i < all_data.row; ++i) {
            // Copy a single row from A_col to A_row
            blas::copy(all_data.col, all_data.A.data() + i, all_data.row, all_data.A_row.data() + i * all_data.col, 1);
        }
    }

    template <typename T>
    static void norm_and_copy_computational_helper(T &norm_A, CQRRPTTestData<T> &all_data) {
        //NOTE: This function is safe for use with RowMajor inputs as well
        auto m = all_data.row;
        auto n = all_data.col;
        auto lda = all_data.layout == Layout::ColMajor ? m : n;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), lda);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), lda);

        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), lda);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, CQRRPTTestData<T> &all_data) {
        //NOTE: This function should be safe for both RowMajor and ColMajor matrices

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
        auto ldq = all_data.layout == Layout::ColMajor ? m : n;
        blas::syrk(all_data.layout, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, ldq, -1.0, I_ref_dat, k);
        T norm_0 = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, k, I_ref_dat, k);

        // A - QR
        blas::gemm(all_data.layout, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, ldq, R_dat, n, -1.0, A_dat, ldq);

        // Implementing max col norm metric
        T max_col_norm = 0.0;
        T col_norm = 0.0;
        int max_idx = 0;
        for(int i = 0; i < n; ++i) {
            col_norm = blas::nrm2(m, &A_dat[ldq * i], 1);
            if(max_col_norm < col_norm) {
                max_col_norm = col_norm;
                max_idx = i;
            }
        }
        T col_norm_A = blas::nrm2(n, &A_cpy_dat[ldq * max_idx], 1);
        T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, ldq);

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

        CQRRPT.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state, all_data.layout);

        all_data.rank = CQRRPT.rank;
        printf("RANK AS RETURNED BY CQRRPT %ld\n", all_data.rank);

        if(all_data.layout == Layout::ColMajor) {
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);
        }
        else {
            RandLAPACK::util::col_swap_row_major(m, n, n, all_data.A_cpy1.data(), n, all_data.J);
            RandLAPACK::util::col_swap_row_major(m, n, n, all_data.A_cpy2.data(), n, all_data.J);
        }

        error_check(norm_A, all_data);
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
    CQRRPT.no_hqrrp = 1;

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
    CQRRPT.no_hqrrp = 0;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
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
    CQRRPT.no_hqrrp = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::adverserial);
    m_info.scaling = 1e7;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general(d_factor, norm_A, all_data, CQRRPT, state);
}
