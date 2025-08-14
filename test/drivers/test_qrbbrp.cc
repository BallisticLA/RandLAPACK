#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>


class TestQRBBRP : public ::testing::Test {
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct QRBBRPTestData {
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

        QRBBRPTestData(int64_t m, int64_t n, int64_t k) :
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
    static void norm_and_copy_computational_helper(T &norm_A, QRBBRPTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }

    template <typename T>
    static void
    error_check(T &norm_A, QRBBRPTestData<T> &all_data, T atol) {

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

    /// General test for QRBBRP:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_QRBBRP_general(
        T d_factor, 
        T norm_A,
        QRBBRPTestData<T> &all_data,
        alg_type &QRBBRP,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);

        QRBBRP.call(m, n, all_data.A.data(), m, all_data.tau.data(), all_data.J.data(), state);

        int64_t small_dim = std::min(m, n);
        all_data.rank = small_dim;

        RandLAPACK::util::upsize(small_dim * n, all_data.R);

        lapack::lacpy(MatrixType::Upper, small_dim, n, all_data.A.data(), m, all_data.R.data(), small_dim);
        lapack::ungqr(m, small_dim, small_dim, all_data.A.data(), m, all_data.tau.data());
        lapack::lacpy(MatrixType::General, m, small_dim, all_data.A.data(), m, all_data.Q.data(), m);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J);

        error_check(norm_A, all_data, atol);
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestQRBBRP, QRBBRP_full_rank_basic) {
    int64_t m = 500;
    int64_t n = 200;
    int64_t k = 200;
    double d_factor = 1;
    int64_t b_sz = 20;
    double norm_A = 0;
    RandBLAS::RNGState state(0);

    QRBBRPTestData<double> all_data(m, n, k);

    struct qrcp_wide_qp3 {
        std::vector<double> tau_vec;
        void reserve(int64_t _m, int64_t _n) { tau_vec.resize(_n, 0.0); }
        void free() { return; }
        void operator()(int64_t _m, int64_t _n, double* A, int64_t _lda, int64_t* J) {
            randblas_require( static_cast<int64_t>(tau_vec.size()) >= _n);
            double* tau = tau_vec.data();
            lapack::geqp3(_m, _n, A, _lda, J, tau);
        }
    };

    qrcp_wide_qp3 qp3{};
    RandLAPACK::QRBBRP<double, qrcp_wide_qp3> QRBBRP(qp3, true, b_sz, d_factor);

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_QRBBRP_general(d_factor, norm_A, all_data, QRBBRP, state);
}

