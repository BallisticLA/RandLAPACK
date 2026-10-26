#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "cqrrpt_test_inputs.hh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <RandBLAS.hh>
#include <algorithm>
#include <fstream>
#include <memory>
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

// After a failed potrf, CQRRPT cuts the rank a posteriori.
// - At tol = u^0.85 and 1e-12 the ratio test cuts it (at 15 and 24) well before the failed
//   column (about 45), with the ratios at least 4% from the threshold. There CQRRPT_GPU must
//   report CQRRPT's rank, with Q and R at least as accurate as CQRRPT's.
// - At tol = 1e-6 the cut falls at the failed column, which depends on rounding in potrf, so
//   the factors need only be finite and bounded: CholQR loses orthogonality to O(1) here.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_failure_rank_matches_cpu) {
    int64_t m = 600;
    int64_t n = 60;
    uint32_t seed = 7;
    std::vector<double> A = sketch_defeating_input(m, n, 2, 2.0, seed, 1.0);

    auto run_gpu = [&](double tol, std::vector<double>& Q, std::vector<double>& R, std::vector<int64_t>& J) {
        Q = A;
        R.assign(n * n, 0.0);
        J.assign(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;
        EXPECT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        return CQRRPT_GPU.rank;
    };

    int64_t ratio_rank = 0;
    for (double tol : {std::pow(std::numeric_limits<double>::epsilon(), 0.85), 1e-12}) {
        SCOPED_TRACE(tol);
        std::vector<double> Q_cpu(A), R_cpu(n * n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state_cpu(seed);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
        ASSERT_LT(CQRRPT.rank, n) << "input no longer makes CholQR fail; the test does not exercise the failure path";

        std::vector<double> Q, R;
        std::vector<int64_t> J;
        int64_t rank = run_gpu(tol, Q, R, J);
        EXPECT_EQ(rank, CQRRPT.rank);
        auto c = factor_checks(m, n, rank, A, Q, R, n, J);
        auto c_cpu = factor_checks(m, n, CQRRPT.rank, A, Q_cpu, R_cpu, n, J_cpu);
        ASSERT_TRUE(c.finite);
        EXPECT_LE(c.err_orth, 100 * c_cpu.err_orth) << "CPU ||Q'Q - I|| = " << c_cpu.err_orth;
        EXPECT_LE(c.err_proj, 100 * c_cpu.err_proj) << "CPU ||R - Q'AP|| = " << c_cpu.err_proj;
        ratio_rank = std::max(ratio_rank, rank);
    }

    // The cut at the failed column guarantees no accuracy (||Q'Q - I|| reaches 2 here), so the
    // bounds only reject the NaN of keeping columns past it, or the ||R - Q'AP|| >= 60 of taking
    // the coupling block from potrf's partial update.
    SCOPED_TRACE(1e-6);
    std::vector<double> Q, R;
    std::vector<int64_t> J;
    int64_t rank = run_gpu(1e-6, Q, R, J);
    EXPECT_GT(rank, ratio_rank) << "the cut at the failed column must come after the ratio test's cuts";
    EXPECT_LT(rank, n) << "potrf failed, so the rank must be cut";
    auto c = factor_checks(m, n, rank, A, Q, R, n, J);
    ASSERT_TRUE(c.finite) << "rank " << rank << ": Q or R contains NaN or Inf";
    EXPECT_LE(c.err_orth, 10.0) << "rank " << rank;
    EXPECT_LE(c.err_proj, 10.0) << "rank " << rank;
}

// With tol <= u the ratio threshold sqrt(tol/u) is at most 1, so the scan's ratio test holds from
// its first entry. When potrf succeeds there is no scan, and the rank is the sketch's. When it
// fails, the scan cuts at its first allowed index, i = 2, so the rank is 1, as in CQRRPT; it must
// not cut at i = 0, which gives rank -1.
TEST_F(TestCQRRPT, CQRRPT_GPU_tol_at_most_unit_roundoff) {
    double tol = std::numeric_limits<double>::epsilon() / 2;
    struct Input { const char* name; int64_t m, n; std::vector<double> A; uint32_t seed; int64_t rank; };
    std::vector<Input> inputs;
    {
        int64_t m = 2000, n = 50;
        std::vector<double> A(m * n, 0.0);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
        m_info.cond_num = 2;
        m_info.rank = n;
        m_info.exponent = 2.0;
        auto gen_state = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), gen_state);
        inputs.push_back({"potrf succeeds", m, n, A, 0, n});
    }
    inputs.push_back({"potrf fails", 600, 60, sketch_defeating_input(600, 60, 2, 2.0, 7, 1.0), 7, 1});

    for (auto& in : inputs) {
        SCOPED_TRACE(in.name);
        int64_t m = in.m, n = in.n;
        std::vector<double> Q_cpu(in.A), R_cpu(n * n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state_cpu(in.seed);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
        ASSERT_EQ(CQRRPT.rank, in.rank);

        std::vector<double> Q(in.A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(in.seed);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        EXPECT_EQ(CQRRPT_GPU.rank, in.rank);
        auto c = factor_checks(m, n, CQRRPT_GPU.rank, in.A, Q, R, n, J);
        auto c_cpu = factor_checks(m, n, CQRRPT.rank, in.A, Q_cpu, R_cpu, n, J_cpu);
        ASSERT_TRUE(c.finite);
        EXPECT_LE(c.err_orth, 100 * c_cpu.err_orth + 1e-14) << "CPU ||Q'Q - I|| = " << c_cpu.err_orth;
        EXPECT_LE(c.err_proj, 100 * c_cpu.err_proj + 1e-14) << "CPU ||R - Q'AP|| = " << c_cpu.err_proj;
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

        // std::max skips a NaN difference, so non-finite entries are counted separately.
        double max_q = 0.0, max_r = 0.0, scale_r = 0.0;
        int64_t bad_pad = 0, nonfinite = 0;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                max_q = std::max(max_q, std::abs(Q_pad[i + j * lda] - Q[i + j * m]));
                nonfinite += !std::isfinite(Q_pad[i + j * lda]) + !std::isfinite(Q[i + j * m]);
            }
            for (int64_t i = m; i < lda; ++i)
                bad_pad += Q_pad[i + j * lda] != pad;
            for (int64_t i = 0; i < n; ++i) {
                max_r = std::max(max_r, std::abs(R_pad[i + j * ldr] - R[i + j * n]));
                scale_r = std::max(scale_r, std::abs(R[i + j * n]));
                nonfinite += !std::isfinite(R_pad[i + j * ldr]) + !std::isfinite(R[i + j * n]);
            }
            for (int64_t i = n; i < ldr; ++i)
                bad_pad += R_pad[i + j * ldr] != pad;
        }
        EXPECT_EQ(nonfinite, 0) << "NaN or Inf entries in Q, R, Q_pad or R_pad";
        EXPECT_LE(max_q, 1e-13) << "max |Q_pad - Q|";
        EXPECT_LE(max_r, 1e-13 * scale_r) << "max |R_pad - R|";
        EXPECT_EQ(bad_pad, 0) << "padding entries changed";
    }
}

// With timing on, each step's time must cover its device work, not just the kernel launches.
// The trsm step must last at least as long as the same trsm timed alone. Total must include the
// final trsm and trmm, so the call may outlast total only by its tail: the copies back to the
// host, the frees and the queue's teardown. Leaving the final trsm out of total adds a whole
// trsm to that, so the time past total may exceed the tail, redone alone right after the call,
// by at most 3/4 of a trsm. Where a trsm is shorter than the copies (fast FP64), that slack
// cannot cover their noise, and the bound is skipped. Timing must not change the results.
TEST_F(TestCQRRPT, CQRRPT_GPU_timing_covers_device_work) {
    int64_t m = 5000;
    int64_t n = 1000;
    int64_t d = 2 * n;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();
    std::vector<double> A(m * n);
    RandBLAS::DenseDist D(m, n);
    state = RandBLAS::fill_dense(D, A.data(), state);
    auto elapsed = [](std::chrono::steady_clock::time_point start) {
        return (long) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    };

    // Reference trsm: the preconditioning trsm, m-by-n against an n-by-n triangle.
    std::vector<double> T(n * n, 0.0), A_host(m * n), T_host(n * n);
    for (int64_t i = 0; i < n; ++i)
        T[i + i * n] = 1.0;
    double *A_device, *T_device;
    cudaMalloc(&A_device, m * n * sizeof(double));
    cudaMalloc(&T_device, n * n * sizeof(double));
    cudaMemcpy(A_device, A.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(T_device, T.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);
    lapack::Queue queue(0);
    long t_trsm = std::numeric_limits<long>::max();
    auto time_trsm = [&]() {
        for (int rep = 0; rep < 3; ++rep) {
            auto start = std::chrono::steady_clock::now();
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, T_device, n, A_device, m, queue);
            queue.sync();
            t_trsm = std::min(t_trsm, elapsed(start));
        }
    };
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, T_device, n, A_device, m, queue);
    queue.sync();
    time_trsm();

    // The driver's tail, alone: on a queue that has run trsm and potrf, copy Q (m-by-n) and R
    // (n-by-n) back to the host, then free the device buffers and host workspaces of the same
    // sizes, and destroy the queue. Sets t_copy to the copies' time and returns the whole.
    long t_copy = std::numeric_limits<long>::max();
    auto time_tail = [&]() {
        auto q = std::make_unique<lapack::Queue>(0);
        auto* d_info = blas::device_malloc<lapack::device_info_int>(1, *q);
        double *Q_device, *R_device, *R_sp_device;
        cudaMalloc(&Q_device, m * n * sizeof(double));
        cudaMalloc(&R_device, n * n * sizeof(double));
        cudaMalloc(&R_sp_device, n * n * sizeof(double));
        double* A_hat = new double[d * n]();
        double* R_sp = new double[n * n]();
        blas::device_copy_matrix(m, n, A.data(), m, Q_device, m, *q);
        blas::device_copy_matrix(n, n, T.data(), n, R_sp_device, n, *q);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sp_device, n, Q_device, m, *q);
        lapack::potrf(Uplo::Upper, n, R_sp_device, n, d_info, *q);
        q->sync();
        auto start = std::chrono::steady_clock::now();
        blas::device_copy_matrix(m, n, Q_device, m, A_host.data(), m, *q);
        blas::device_copy_matrix(n, n, R_device, n, T_host.data(), n, *q);
        q->sync();
        t_copy = std::min(t_copy, elapsed(start));
        cudaFree(Q_device);
        cudaFree(R_device);
        cudaFree(R_sp_device);
        blas::device_free(d_info, *q);
        delete[] A_hat;
        delete[] R_sp;
        q.reset();
        return elapsed(start);
    };

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
        wall = elapsed(start);
        return CQRRPT_GPU.times;
    };
    std::vector<double> Q_on, R_on, Q_off, R_off;
    std::vector<int64_t> J_on, J_off;
    long wall = 0;
    run(false, Q_off, R_off, J_off, wall);
    // A stall on a busy machine can break a single timed call or tail, while the defects show
    // in every call: keep the longest trsm step, and the median excess, of 3 timed calls.
    long step_trsm = 0;
    std::vector<long> excesses;
    for (int rep = 0; rep < 3; ++rep) {
        auto times = run(true, Q_on, R_on, J_on, wall);
        // times = {saso, qrcp, rank_reveal, cholqr, a_mod_piv, a_mod_trsm, rest, total}
        ASSERT_EQ(times.size(), 8u);
        long past = wall - times[7];
        excesses.push_back(past - time_tail());
        step_trsm = std::max(step_trsm, times[5]);
        EXPECT_EQ(Q_on, Q_off);
        EXPECT_EQ(R_on, R_off);
        EXPECT_EQ(J_on, J_off);
    }
    std::sort(excesses.begin(), excesses.end());
    long excess = excesses[1];
    // The GPU's clocks, or other work on it, can slow either set of reference trsm runs.
    time_trsm();
    cudaFree(A_device);
    cudaFree(T_device);

    EXPECT_GE(step_trsm, t_trsm) << "trsm step " << step_trsm << " us, trsm alone " << t_trsm << " us";
    if (t_trsm < t_copy)
        GTEST_SKIP() << "trsm alone " << t_trsm << " us, copies back " << t_copy << " us: too close to bound the time past total";
    EXPECT_LT(excess, 3 * t_trsm / 4) << "time past total exceeds the tail alone by " << excess << " us; trsm alone " << t_trsm << " us";
}

// R's contents on entry must not matter: with R filled with NaN, CQRRPT_GPU must return exactly
// the R it returns for a zeroed R, and that R is zero below the diagonal and past the rank.
TEST_F(TestCQRRPT, CQRRPT_GPU_R_contents_on_entry_ignored) {
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
    inputs.push_back({"failed potrf", 600, 60, sketch_defeating_input(600, 60, 2, 2.0, 7, 1.0), 7});

    for (auto& in : inputs) {
        SCOPED_TRACE(in.name);
        int64_t m = in.m, n = in.n;
        std::vector<double> Q0(in.A), R0(n * n, 0.0), Q1(in.A), R1(n * n, std::numeric_limits<double>::quiet_NaN());
        std::vector<int64_t> J0(n, 0), J1(n, 0);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.no_hqrrp = 1;
        RandBLAS::RNGState<r123::Philox4x32> state0(in.seed), state1(in.seed);
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q0.data(), m, R0.data(), n, J0.data(), 2.0, state0), 0);
        int64_t rank = CQRRPT_GPU.rank;
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q1.data(), m, R1.data(), n, J1.data(), 2.0, state1), 0);
        ASSERT_EQ(CQRRPT_GPU.rank, rank);

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

// On an all-zero input CQRRPT_GPU must report rank 0, whatever rank held before the call, and
// return R as documented (zero) with A left zero, as CQRRPT does.
TEST_F(TestCQRRPT, CQRRPT_GPU_zero_input) {
    int64_t m = 100;
    int64_t n = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(m * n, 0.0), R(n * n, std::numeric_limits<double>::quiet_NaN());
    std::vector<int64_t> J(n, 0);

    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    EXPECT_EQ(CQRRPT_GPU.rank, 0);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.no_hqrrp = 1;
    CQRRPT_GPU.rank = -7;
    auto state = RandBLAS::RNGState();
    ASSERT_EQ(CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    EXPECT_EQ(CQRRPT_GPU.rank, 0);
    EXPECT_EQ(R, std::vector<double>(n * n, 0.0));
    EXPECT_EQ(A, std::vector<double>(m * n, 0.0));
}
