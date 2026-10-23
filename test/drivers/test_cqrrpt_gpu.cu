#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "cqrrpt_test_inputs.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>
#include <cusolverDn.h>

// The GPU drivers and kernels come in with RandLAPACK.hh, under its __CUDACC__ guard.

using cqrrpt_test::sketch_defeating_input;

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

    template <typename T, typename RNG>
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
        ASSERT_NEAR(norm_AQR / norm_A,         0.0, atol);
        ASSERT_NEAR(max_col_norm / col_norm_A, 0.0, atol);
        ASSERT_NEAR(norm_0 / std::sqrt((T) n), 0.0, atol);
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG, typename alg_type>
    static void test_CQRRPT_general(
        T d_factor, 
        T norm_A,
        CQRRPTTestData<T> &all_data,
        alg_type &CQRRPT_GPU,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT_GPU.call(m, n, all_data.A.data(), m, all_data.R.data(), n, all_data.J.data(), d_factor, state);

        all_data.rank = CQRRPT_GPU.rank;
        std::cout << "RANK AS RETURNED BY CQRRPT " << all_data.rank << "\n";

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1.data(), m, all_data.J.data());
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2.data(), m, all_data.J.data());

        error_check(norm_A, all_data);
    }

    struct FactorChecks {
        bool finite;      // Q[:, 0:r] and R[0:r, :] contain no NaN or Inf
        double err_orth;  // ||Q_r' Q_r - I||_F / sqrt(r)
        double err_proj;  // ||R_r - Q_r' A[:, J]||_F / ||A||_F
    };

    /// Checks the first r columns of Q (m-by-n, ld m) and rows of R (ld ldr) against A[:, J].
    static FactorChecks factor_checks(int64_t m, int64_t n, int64_t r, const std::vector<double>& A, const std::vector<double>& Q, const std::vector<double>& R, int64_t ldr, std::vector<int64_t>& J) {
        FactorChecks c{true, 0.0, 0.0};
        for (int64_t j = 0; j < r; ++j)
            for (int64_t i = 0; i < m; ++i)
                c.finite = c.finite && std::isfinite(Q[i + j * m]);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < r; ++i)
                c.finite = c.finite && std::isfinite(R[i + j * ldr]);
        if (!c.finite || r == 0)
            return c;
        std::vector<double> G(r * r, 0.0), AP(A), P(r * n, 0.0);
        RandLAPACK::util::eye(r, r, G.data());
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, 1.0, Q.data(), m, -1.0, G.data(), r);
        c.err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, G.data(), r) / std::sqrt((double) r);
        RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
        lapack::lacpy(MatrixType::General, r, n, R.data(), ldr, P.data(), r);
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, r, n, m, -1.0, Q.data(), m, AP.data(), m, 1.0, P.data(), r);
        c.err_proj = lapack::lange(Norm::Fro, r, n, P.data(), r) / lapack::lange(Norm::Fro, m, n, A.data(), m);
        return c;
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_GPU_full_rank_no_hqrrp) {
    int64_t m = 1000;
    int64_t n = 200;
    int64_t k = 200;
    double d_factor = 2.0;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    CQRRPTTestData<double> all_data(m, n, k);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.num_threads = 4;
    CQRRPT_GPU.no_hqrrp = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32>>(d_factor, norm_A, all_data, CQRRPT_GPU, state);
}

// GPU counterpart of CQRRPT_dirty_J_rank_deficient in test_cqrrpt.cc (the QRCP
// itself runs on the host): dirty J buffers must not influence pivoting.
TEST_F(TestCQRRPT, CQRRPT_GPU_dirty_J_rank_deficient) {
    int64_t m = 2000;
    int64_t n = 50;
    int64_t k = 40;
    double d_factor = 2.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate the rank-deficient input once.
    auto gen_state = RandBLAS::RNGState();
    std::vector<double> A_orig(m * n, 0.0);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A_orig.data(), gen_state);

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

        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.num_threads = 4;
        CQRRPT_GPU.no_hqrrp = 1;

        double norm_A = 0;
        norm_and_copy_computational_helper<double, r123::Philox4x32>(norm_A, all_data);
        // Same RNG state in every trial, so any difference is due to J alone.
        auto state = RandBLAS::RNGState();
        test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32>>(d_factor, norm_A, all_data, CQRRPT_GPU, state);

        if (trial == 0)
            rank_ref = all_data.rank;
        ASSERT_EQ(all_data.rank, rank_ref);
    }
}

// Repeats CQRRPT_GPU on one input. The device CholQR mixes cuBLAS and cuSOLVER
// calls, so a missing ordering between them shows up as an occasional NaN or
// inaccurate factor rather than a deterministic failure. Larger inputs widen
// the window in which an unordered potrf can overtake syrk.
TEST_F(TestCQRRPT, CQRRPT_GPU_repeated_calls_stable) {
    int64_t m = 3000;
    int64_t n = 300;
    int64_t k = 300;
    int num_trials = 50;
    double d_factor = 2.0;
    double tol  = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);

    auto gen_state = RandBLAS::RNGState();
    std::vector<double> A_orig(m * n, 0.0);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A_orig.data(), gen_state);
    double norm_A = lapack::lange(Norm::Fro, m, n, A_orig.data(), m);

    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.num_threads = 4;
    CQRRPT_GPU.no_hqrrp = 1;

    std::vector<double> A(m * n), A_piv(m * n), R(n * n), QtQ(n * n);
    std::vector<int64_t> J(n);
    int num_bad = 0;
    for (int trial = 0; trial < num_trials; ++trial) {
        std::copy(A_orig.begin(), A_orig.end(), A.begin());
        std::copy(A_orig.begin(), A_orig.end(), A_piv.begin());
        std::fill(R.begin(), R.end(), 0.0);
        auto state = RandBLAS::RNGState();
        CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, J.data(), d_factor, state);
        int64_t r = CQRRPT_GPU.rank;

        // ||A[:, J] - QR||_F / ||A||_F
        RandLAPACK::util::col_swap(m, n, n, A_piv.data(), m, J.data());
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, 1.0, A.data(), m, R.data(), n, -1.0, A_piv.data(), m);
        double err_qr = lapack::lange(Norm::Fro, m, n, A_piv.data(), m) / norm_A;
        // ||Q'Q - I||_F / sqrt(n)
        RandLAPACK::util::eye(r, r, QtQ.data());
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, 1.0, A.data(), m, -1.0, QtQ.data(), r);
        double err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, QtQ.data(), r) / std::sqrt((double) n);

        bool ok = (r == k) && std::isfinite(err_qr) && std::isfinite(err_orth) && err_qr <= atol && err_orth <= atol;
        EXPECT_TRUE(ok) << "trial " << trial << ": rank " << r << ", ||AP - QR||/||A|| = " << err_qr << ", ||Q'Q - I||/sqrt(n) = " << err_orth;
        num_bad += !ok;
    }
    ASSERT_EQ(num_bad, 0) << num_bad << " of " << num_trials << " calls gave a non-finite or inaccurate factorization";
}

// An input that defeats the sketch makes potrf fail inside CholQR. CQRRPT_GPU must notice
// the failure and cut the rank, as CQRRPT does, instead of carrying NaN into Q and R.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_failure_handled) {
    int64_t m = 600;
    int64_t n = 60;
    uint32_t seed = 7;
    double tol = 1e-6;
    std::vector<double> A = sketch_defeating_input(m, n, 2, 2.0, seed, 1.0);

    // The CPU driver confirms that this input makes CholQR fail and cut the rank.
    std::vector<double> Q_cpu(A), R_cpu(n * n, 0.0);
    std::vector<int64_t> J_cpu(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state_cpu(seed);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
    ASSERT_LT(CQRRPT.rank, n) << "input no longer makes CholQR fail; the test does not exercise the failure path";

    std::vector<double> Q(A), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state(seed);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.no_hqrrp = 1;
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    auto c = factor_checks(m, n, CQRRPT_GPU.rank, A, Q, R, n, J);
    EXPECT_TRUE(c.finite) << "rank " << CQRRPT_GPU.rank << ": Q or R contains NaN or Inf";
}

// When potrf succeeds, CQRRPT keeps the full rank. CQRRPT_GPU must not cut the rank from the
// Cholesky diagonal alone: a cut it then ignores leaves Q's trailing columns unorthogonalized.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_success_keeps_rank) {
    int64_t m = 200;
    int64_t n = 20;
    uint32_t seed = 7;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A = sketch_defeating_input(m, n, 2, 2.0, seed, 0.6);

    std::vector<double> Q_cpu(A), R_cpu(n * n, 0.0);
    std::vector<int64_t> J_cpu(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state_cpu(seed);
    RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
    CQRRPT.nnz = 2;
    ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
    ASSERT_EQ(CQRRPT.rank, n);
    auto c_cpu = factor_checks(m, n, CQRRPT.rank, A, Q_cpu, R_cpu, n, J_cpu);

    std::vector<double> Q(A), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state(seed);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.no_hqrrp = 1;
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    EXPECT_EQ(CQRRPT_GPU.rank, n);
    auto c = factor_checks(m, n, CQRRPT_GPU.rank, A, Q, R, n, J);
    ASSERT_TRUE(c.finite);
    // CholQR loses orthogonality like u * cond^2 here; stay within 100x of the CPU driver.
    EXPECT_LE(c.err_orth, 100 * c_cpu.err_orth + 1e-12) << "CPU ||Q'Q - I|| = " << c_cpu.err_orth;
}

// After a failed potrf, CQRRPT cuts the rank a posteriori. CQRRPT_GPU must report the same
// rank, and its first rank columns of Q and rows of R must be at least as accurate as
// CQRRPT's: Q orthonormal and R = Q' A[:, J], up to the loss CholQR incurs on this input.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_failure_rank_matches_cpu) {
    int64_t m = 600;
    int64_t n = 60;
    uint32_t seed = 7;
    std::vector<double> A = sketch_defeating_input(m, n, 2, 2.0, seed, 1.0);

    for (double tol : {std::pow(std::numeric_limits<double>::epsilon(), 0.85), 1e-6}) {
        SCOPED_TRACE(tol);
        std::vector<double> Q_cpu(A), R_cpu(n * n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state_cpu(seed);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
        ASSERT_LT(CQRRPT.rank, n) << "input no longer makes CholQR fail; the test does not exercise the failure path";

        std::vector<double> Q(A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        EXPECT_EQ(CQRRPT_GPU.rank, CQRRPT.rank);
        auto c = factor_checks(m, n, CQRRPT_GPU.rank, A, Q, R, n, J);
        auto c_cpu = factor_checks(m, n, CQRRPT.rank, A, Q_cpu, R_cpu, n, J_cpu);
        ASSERT_TRUE(c.finite);
        EXPECT_LE(c.err_orth, 100 * c_cpu.err_orth) << "CPU ||Q'Q - I|| = " << c_cpu.err_orth;
        EXPECT_LE(c.err_proj, 100 * c_cpu.err_proj) << "CPU ||R - Q'AP|| = " << c_cpu.err_proj;
    }
}

// Padded leading dimensions (lda > m, ldr > n) must give the same factorization as compact
// storage and leave the padding rows untouched. Full rank sends every column of A through the
// device; rank deficiency (sketch rank k < n) also fills R's trailing columns on the host.
TEST_F(TestCQRRPT, CQRRPT_GPU_leading_dimensions) {
    int64_t m = 2000;
    int64_t n = 50;
    int64_t lda = m + 7;
    int64_t ldr = n + 3;
    double pad = -777.0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    for (int64_t rank_A : {n, (int64_t) 40}) {
        SCOPED_TRACE(rank_A);
        auto gen_state = RandBLAS::RNGState();
        std::vector<double> A(m * n, 0.0);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
        m_info.cond_num = 2;
        m_info.rank = rank_A;
        m_info.exponent = 2.0;
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), gen_state);

        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;

        std::vector<double> Q(A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        auto state = RandBLAS::RNGState();
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        int64_t rank = CQRRPT_GPU.rank;
        ASSERT_EQ(rank, rank_A);

        std::vector<double> Q_pad(lda * n, pad), R_pad(ldr * n, pad);
        lapack::lacpy(MatrixType::General, m, n, A.data(), m, Q_pad.data(), lda);
        lapack::laset(MatrixType::General, n, n, 0.0, 0.0, R_pad.data(), ldr);
        std::vector<int64_t> J_pad(n, 0);
        state = RandBLAS::RNGState();
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q_pad.data(), lda, R_pad.data(), ldr, J_pad.data(), 2.0, state), 0);
        EXPECT_EQ(CQRRPT_GPU.rank, rank);
        EXPECT_EQ(J_pad, J);

        double max_q = 0.0, max_r = 0.0, scale_r = 0.0;
        int64_t bad_pad = 0;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i)
                max_q = std::max(max_q, std::abs(Q_pad[i + j * lda] - Q[i + j * m]));
            for (int64_t i = m; i < lda; ++i)
                bad_pad += Q_pad[i + j * lda] != pad;
            for (int64_t i = 0; i < n; ++i) {
                max_r = std::max(max_r, std::abs(R_pad[i + j * ldr] - R[i + j * n]));
                scale_r = std::max(scale_r, std::abs(R[i + j * n]));
            }
            for (int64_t i = n; i < ldr; ++i)
                bad_pad += R_pad[i + j * ldr] != pad;
        }
        EXPECT_LE(max_q, 1e-13) << "max |Q_pad - Q|";
        EXPECT_LE(max_r, 1e-13 * scale_r) << "max |R_pad - R|";
        EXPECT_EQ(bad_pad, 0) << "padding entries changed";
    }
}

// With timing on, each step's time must cover its device work, not just the kernel launches.
// The trsm step must last at least as long as the same trsm timed alone, and total must
// include the final trsm, so the call may outlast total by less than one trsm.
// Timing must not change the results.
TEST_F(TestCQRRPT, CQRRPT_GPU_timing_covers_device_work) {
    int64_t m = 5000;
    int64_t n = 1000;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();
    std::vector<double> A(m * n);
    RandBLAS::DenseDist D(m, n);
    state = RandBLAS::fill_dense(D, A.data(), state);

    // Reference: the preconditioning trsm (m-by-n against an n-by-n triangle), fastest of 3 runs.
    long t_trsm = std::numeric_limits<long>::max();
    {
        std::vector<double> T(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            T[i + i * n] = 1.0;
        double *A_device, *T_device;
        cudaMalloc(&A_device, m * n * sizeof(double));
        cudaMalloc(&T_device, n * n * sizeof(double));
        cudaMemcpy(A_device, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(T_device, T.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
        lapack::Queue queue(0);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, T_device, n, A_device, m, queue);
        queue.sync();
        for (int rep = 0; rep < 3; ++rep) {
            auto start = std::chrono::steady_clock::now();
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, T_device, n, A_device, m, queue);
            queue.sync();
            t_trsm = std::min<long>(t_trsm, std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        }
        cudaFree(A_device);
        cudaFree(T_device);
    }

    auto run = [&](bool timing, std::vector<double>& Q, std::vector<double>& R, std::vector<int64_t>& J, long& wall) {
        Q = A;
        R.assign(n * n, 0.0);
        J.assign(n, 0);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, timing, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;
        auto s = RandBLAS::RNGState();
        auto start = std::chrono::steady_clock::now();
        EXPECT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, s), 0);
        wall = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
        return CQRRPT_GPU.times;
    };
    std::vector<double> Q_on, R_on, Q_off, R_off;
    std::vector<int64_t> J_on, J_off;
    long wall = 0;
    run(false, Q_off, R_off, J_off, wall);
    auto times = run(true, Q_on, R_on, J_on, wall);

    // times = {saso, qrcp, rank_reveal, cholqr, a_mod_piv, a_mod_trsm, rest, total}
    ASSERT_EQ(times.size(), 8u);
    EXPECT_GE(times[5], t_trsm) << "trsm step " << times[5] << " us, trsm alone " << t_trsm << " us";
    EXPECT_LT(wall - times[7], t_trsm) << "call " << wall << " us, total " << times[7] << " us, trsm alone " << t_trsm << " us";
    EXPECT_EQ(Q_on, Q_off);
    EXPECT_EQ(R_on, R_off);
    EXPECT_EQ(J_on, J_off);
}
