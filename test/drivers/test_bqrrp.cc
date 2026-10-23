#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <tuple>
#include <gtest/gtest.h>

using Subroutines = RandLAPACK::BQRRPSubroutines;

class TestBQRRP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct BQRRPTestData {
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

        BQRRPTestData(int64_t m, int64_t n, int64_t k) :
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
    static void norm_and_copy_computational_helper(T &norm_A, BQRRPTestData<T> &all_data) {
        auto m = all_data.row;
        auto n = all_data.col;

        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy1.data(), m);
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy2.data(), m);
        norm_A = lapack::lange(Norm::Fro, m, n, all_data.A.data(), m);
    }


    /// This routine also appears in benchmarks, but idk if it should be put into utils
    template <typename T>
    static void
    error_check(T &norm_A, BQRRPTestData<T> &all_data, T atol) {

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
        T col_norm_A = blas::nrm2(m, &A_cpy_dat[m * max_idx], 1);
        T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);

        std::cout << "REL NORM OF AP - QR:    " << std::scientific << std::setw(14) << norm_AQR / norm_A << "\n";
        std::cout << "MAX COL NORM METRIC:    " << std::scientific << std::setw(14) << max_col_norm / col_norm_A << "\n";
        std::cout << "FRO NORM OF (Q'Q - I):  " << std::scientific << std::setw(14) << norm_0 / std::sqrt((T) n) << "\n\n";

        ASSERT_LE(norm_AQR, atol * norm_A);
        ASSERT_LE(max_col_norm, atol * col_norm_A);
        ASSERT_LE(norm_0, atol * std::sqrt((T) n));
    }

    /// General test for BQRRP:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_BQRRP_general(
        T d_factor, 
        T norm_A,
        BQRRPTestData<T> &all_data,
        alg_type &BQRRP,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);

        BQRRP.call(m, n, all_data.A.data(), m, d_factor, all_data.tau.data(), all_data.J.data(), state);

        if(BQRRP.rank == 0) {
            for(int i = 0; i < m * n; ++i) {
                ASSERT_NEAR(all_data.A[i], 0.0, atol);
            }
        } else {
            all_data.rank = BQRRP.rank;
            std::cout << "RANK AS RETURNED BY BQRRP " << std::setw(4) << all_data.rank << "\n";

            RandLAPACK::util::upsize(all_data.rank * n, all_data.R);

            lapack::lacpy(MatrixType::Upper, all_data.rank, n, all_data.A.data(), m, all_data.R.data(), all_data.rank);

            lapack::ungqr(m, std::min(m, n), std::min(m, n), all_data.A.data(), m, all_data.tau.data());
            
            lapack::lacpy(MatrixType::General, m, all_data.rank, all_data.A.data(), m, all_data.Q.data(), m);

            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J.data());
            RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J.data());

            error_check(norm_A, all_data, atol);
        }
    }

    /// Rank-deficient test for BQRRP:
    /// A = G * H has exact rank r, with r not a multiple of b_sz, so the block holding
    /// column r is detected as rank-deficient and BQRRP stops there. For every qr_tall and
    /// apply_trans_q pair, checks the reported rank and the residual of A[:, J] - Q1 * R(1:r, :),
    /// over the columns up to the end of that block and, separately, over the trailing columns,
    /// whose R entries come from that block's apply_trans_q call.
    /// A is scaled by 1e4: correct relative residuals do not change, but errors that grow
    /// with ||A|| (such as R11 built from stale workspace) rise far above rounding.
    template <typename T, typename RNG>
    static void test_BQRRP_rank_deficient(
        int64_t m,
        int64_t n,
        int64_t r,
        int64_t b_sz,
        RandBLAS::RNGState<RNG> &state) {

        // The block holding column r must be rank-deficient, so r must fall inside a block.
        ASSERT_NE(r % b_sz, 0) << "r=" << r << " must not be a multiple of b_sz=" << b_sz;
        ASSERT_LT(r, std::min(m, n)) << "r=" << r << " must be < min(m, n)=" << std::min(m, n);

        T d_factor = 1.25;
        T atol = std::pow(std::numeric_limits<T>::epsilon(), 0.75);

        std::vector<T> G(m * r, 0.0);
        std::vector<T> H(r * n, 0.0);
        std::vector<T> A(m * n, 0.0);
        RandBLAS::DenseDist DG(m, r);
        RandBLAS::DenseDist DH(r, n);
        state = RandBLAS::fill_dense(DG, G.data(), state);
        state = RandBLAS::fill_dense(DH, H.data(), state);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, (T) 1e4, G.data(), m, H.data(), r, (T) 0.0, A.data(), m);
        T norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);

        // The rank-deficient block ends at column split; R(:, split:n) is its R12.
        int64_t split = std::min(std::min(m, n), (r / b_sz + 1) * b_sz);

        // geqrf provides no T factor, so BQRRP always applies its Q' through ormqr.
        const std::vector<std::tuple<const char*, Subroutines::QRTall, Subroutines::ApplyTransQ>> variants = {
            {"cholqr + gemqrt", Subroutines::QRTall::cholqr, Subroutines::ApplyTransQ::gemqrt},
            {"cholqr + ormqr",  Subroutines::QRTall::cholqr, Subroutines::ApplyTransQ::ormqr},
            {"geqrt + gemqrt",  Subroutines::QRTall::geqrt,  Subroutines::ApplyTransQ::gemqrt},
            {"geqrt + ormqr",   Subroutines::QRTall::geqrt,  Subroutines::ApplyTransQ::ormqr},
            {"geqrf + ormqr",   Subroutines::QRTall::geqrf,  Subroutines::ApplyTransQ::ormqr}
        };

        for (const auto& [name, qr_tall, apply_trans_q] : variants) {
            SCOPED_TRACE(name);
            std::vector<T> A_fact(A);
            std::vector<T> A_piv(A);
            std::vector<T> R(r * n, 0.0);
            std::vector<T> tau(n, 0.0);
            std::vector<int64_t> J(n, 0);
            auto state_alg = state;

            RandLAPACK::BQRRP<T, RNG> BQRRP(false, b_sz);
            BQRRP.tol           = 1e-10;
            BQRRP.internal_nb   = 10;
            BQRRP.qr_tall       = qr_tall;
            BQRRP.apply_trans_q = apply_trans_q;
            BQRRP.call(m, n, A_fact.data(), m, d_factor, tau.data(), J.data(), state_alg);

            // Build R(1:r, :) and Q1 = Q(:, 1:r) from the known rank r, so that the residual
            // checks below do not depend on the reported rank.
            lapack::lacpy(MatrixType::Upper, r, n, A_fact.data(), m, R.data(), r);
            lapack::ungqr(m, r, r, A_fact.data(), m, tau.data());

            // A[:, J] - Q1 * R(1:r, :)
            RandLAPACK::util::col_swap(m, n, n, A_piv.data(), m, J.data());
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, (T) 1.0, A_fact.data(), m, R.data(), r, (T) -1.0, A_piv.data(), m);
            T res_lead  = lapack::lange(Norm::Fro, m, split, A_piv.data(), m) / norm_A;
            T res_trail = lapack::lange(Norm::Fro, m, n - split, &A_piv[m * split], m) / norm_A;

            std::cout << std::setw(16) << name << ":  RANK " << std::setw(5) << BQRRP.rank << " (expected " << r << "),"
                      << "  REL RESIDUAL, COLS [0, " << split << "): " << std::scientific << res_lead
                      << ",  COLS [" << split << ", " << n << "): " << res_trail << "\n";

            EXPECT_EQ(BQRRP.rank, r);
            EXPECT_LE(res_lead, atol);
            EXPECT_LE(res_trail, atol);
        }
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_full_rank_basic) {
    int64_t m = 5000;//5000;
    int64_t n = 2000;//2000;
    int64_t k = 2000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 500;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_full_rank_block_change) {
    int64_t m = 5000;//5000;
    int64_t n = 2000;//2000;
    int64_t k = 2000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 700;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_low_rank) {
    int64_t m = 5000;
    int64_t n = 2000;
    int64_t k = 100;
    double d_factor = 2.0;
    int64_t b_sz = 200;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_pivot_qual) {
    int64_t m = std::pow(2, 10);
    int64_t n = std::pow(2, 10);
    int64_t k = std::pow(2, 10);
    double d_factor = 1.25;
    int64_t b_sz = 256;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall    = Subroutines::QRTall::cholqr;
    BQRRP.qrcp_wide  = Subroutines::QRCPWide::geqp3;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::step);
    m_info.cond_num = std::pow(10, 10);
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_gemqrt) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall       = Subroutines::QRTall::cholqr;
    BQRRP.apply_trans_q = Subroutines::ApplyTransQ::gemqrt;
    BQRRP.internal_nb = 10;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_near_zero_input_qp3) {
    int64_t m = 1000;//5000;
    int64_t n = 1000;//2000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall   = Subroutines::QRTall::cholqr;
    BQRRP.qrcp_wide = Subroutines::QRCPWide::geqp3;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);
    all_data.A[1000*200 + 10] = 1;

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_near_zero_luqr) {
    int64_t m = 1000;
    int64_t n = 1000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);
    //all_data.A[1000*200 + 10] = 1;
    all_data.A[10*5 + 1] = 1;

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_half_zero_luqr) {
    int64_t m = 5000;//5000;
    int64_t n = 2000;//2000;
    int64_t k = 2000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 500;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(&(all_data.A.data())[m * n / 2], &(all_data.A.data())[m * n], 0.0);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestBQRRP, BQRRP_zero_mat) {
    int64_t m = 1000;//5000;
    int64_t n = 1000;//2000;
    int64_t k = 1000;
    double d_factor = 1;//1.0;
    int64_t b_sz = 100;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;

    std::fill(&(all_data.A.data())[0], &(all_data.A.data())[m * n], 0.0);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

TEST_F(TestBQRRP, BQRRP_qrf) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::geqrf;
    BQRRP.internal_nb = 10;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

TEST_F(TestBQRRP, BQRRP_qrt) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::geqrt;
    BQRRP.internal_nb = 10;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

TEST_F(TestBQRRP, BQRRP_cholqr_nb) {
    int64_t m = 5000;//5000;
    int64_t n = 2800;//2000;
    int64_t k = 2800;
    double d_factor = 1;//1.0;
    int64_t b_sz = 900;//500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::cholqr;
    BQRRP.internal_nb = 7;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Wide aspect ratio (m < n).  Regression test for the wide-A correctness
// fix: prior to the fix, the termination condition `curr_sz >= n` never
// fired for wide A (curr_sz tops out at min(m, n) = m < n), so the loop
// fell through without setting `this->rank`. On the CPU the factorization
// itself still ended up in caller's A (because column permutation is
// in-place), but `rank` was left at its default (0). This test guards
// against re-introducing the bug.
TEST_F(TestBQRRP, BQRRP_wide_aspect) {
    int64_t m = 2000;
    int64_t n = 5000;   // n > m -- wide
    int64_t k = m;      // expected rank
    double d_factor = 1;
    int64_t b_sz = 500;
    double norm_A = 0;
    auto state = RandBLAS::RNGState();

    BQRRPTestData<double> all_data(m, n, k);
    RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(true, b_sz);
    BQRRP.qr_tall = Subroutines::QRTall::geqrf;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_BQRRP_general(d_factor, norm_A, all_data, BQRRP, state);
}

// Scale invariance: the same A, scaled by 1e-20, 1 and 1e20, must give the same rank and a
// rounding-level residual ||A[:, J] - Q(:, 1:rank) R(1:rank, :)|| / ||A||, for every qr_tall
// and apply_trans_q pair. Cases: a full-rank Gaussian with the default tol, and an exact
// rank-400 matrix with b_sz = 200 and tol = 1e-10, whose rank drop at a block boundary only
// the zero-block check can detect.
TEST_F(TestBQRRP, BQRRP_scale_invariance) {
    int64_t m = 1000;
    int64_t n = 600;
    int64_t r = 400;
    int64_t b_sz = 200;
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::RNGState();

    const std::vector<std::tuple<const char*, Subroutines::QRTall, Subroutines::ApplyTransQ>> variants = {
        {"cholqr + gemqrt", Subroutines::QRTall::cholqr, Subroutines::ApplyTransQ::gemqrt},
        {"cholqr + ormqr",  Subroutines::QRTall::cholqr, Subroutines::ApplyTransQ::ormqr},
        {"geqrt + gemqrt",  Subroutines::QRTall::geqrt,  Subroutines::ApplyTransQ::gemqrt},
        {"geqrt + ormqr",   Subroutines::QRTall::geqrt,  Subroutines::ApplyTransQ::ormqr},
        {"geqrf + ormqr",   Subroutines::QRTall::geqrf,  Subroutines::ApplyTransQ::ormqr}
    };

    auto check_scales = [&](const std::vector<double>& A, double tol, int64_t expected_rank) {
        for (double scale : {1e-20, 1.0, 1e20}) {
            std::vector<double> A_s(A);
            blas::scal(m * n, scale, A_s.data(), 1);
            double norm_A = lapack::lange(Norm::Fro, m, n, A_s.data(), m);
            for (const auto& [name, qr_tall, apply_trans_q] : variants) {
                SCOPED_TRACE(::testing::Message() << name << ", scale " << scale);
                std::vector<double> A_fact(A_s);
                std::vector<double> A_piv(A_s);
                std::vector<double> tau(n, 0.0);
                std::vector<int64_t> J(n, 0);
                auto state_alg = state;

                RandLAPACK::BQRRP<double, r123::Philox4x32> BQRRP(false, b_sz);
                BQRRP.tol           = tol;
                BQRRP.internal_nb   = 10;
                BQRRP.qr_tall       = qr_tall;
                BQRRP.apply_trans_q = apply_trans_q;
                BQRRP.call(m, n, A_fact.data(), m, 1.25, tau.data(), J.data(), state_alg);
                int64_t k = BQRRP.rank;

                // A[:, J] - Q(:, 1:k) * R(1:k, :)
                RandLAPACK::util::col_swap(m, n, n, A_piv.data(), m, J.data());
                if (k > 0) {
                    std::vector<double> R(k * n, 0.0);
                    lapack::lacpy(MatrixType::Upper, k, n, A_fact.data(), m, R.data(), k);
                    lapack::ungqr(m, k, k, A_fact.data(), m, tau.data());
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_fact.data(), m, R.data(), k, -1.0, A_piv.data(), m);
                }
                double res = lapack::lange(Norm::Fro, m, n, A_piv.data(), m) / norm_A;

                std::cout << std::setw(16) << name << ",  SCALE " << std::scientific << std::setprecision(0) << scale
                          << ":  RANK " << std::setw(5) << k << " (expected " << expected_rank << "),  REL RESIDUAL "
                          << std::setprecision(6) << res << "\n";

                EXPECT_EQ(k, expected_rank);
                EXPECT_LE(res, atol);
            }
        }
    };

    std::vector<double> A_full(m * n, 0.0);
    RandBLAS::DenseDist D(m, n);
    state = RandBLAS::fill_dense(D, A_full.data(), state);
    check_scales(A_full, std::numeric_limits<double>::epsilon(), n);

    std::vector<double> G(m * r, 0.0);
    std::vector<double> H(r * n, 0.0);
    std::vector<double> A_low(m * n, 0.0);
    RandBLAS::DenseDist DG(m, r);
    RandBLAS::DenseDist DH(r, n);
    state = RandBLAS::fill_dense(DG, G.data(), state);
    state = RandBLAS::fill_dense(DH, H.data(), state);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, 1.0, G.data(), m, H.data(), r, 0.0, A_low.data(), m);
    check_scales(A_low, 1e-10, r);
}

// Exact rank r < b_sz: the first block is rank-deficient. Its R12 needs Q' applied to all
// rows, not only the top block_rank, and the reported rank must be r, not b_sz.
TEST_F(TestBQRRP, BQRRP_rank_deficient_first_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_rank_deficient<double>(1000, 600, 150, 200, state);
}

// Rank drop inside the second block: the reported rank must count the full first block,
// and with cholqr, R11 of the rank-deficient block must not reuse the first block's R11.
TEST_F(TestBQRRP, BQRRP_rank_deficient_later_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_rank_deficient<double>(1000, 600, 350, 200, state);
}

// Wide A whose clipped last block (b_sz = 100 of 250) is rank-deficient.
TEST_F(TestBQRRP, BQRRP_rank_deficient_wide_clipped_block) {
    auto state = RandBLAS::RNGState();
    test_BQRRP_rank_deficient<double>(600, 1000, 550, 250, state);
}
