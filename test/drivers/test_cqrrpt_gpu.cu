#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "cqrrpt_test_inputs.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <RandBLAS.hh>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <gtest/gtest.h>
#include <cusolverDn.h>

// The GPU drivers and kernels come in with RandLAPACK.hh, under its __CUDACC__ guard.

using cqrrpt_test::sketch_defeating_input;
using RandLAPACK::testing::QR_test_data;
using RandLAPACK::testing::norm_and_copy_computational_helper;

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
        QR_test_data<T> &all_data,
        alg_type &CQRRPT_GPU,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;

        CQRRPT_GPU.call(m, n, all_data.A, m, all_data.R, n, all_data.J, d_factor, state);

        all_data.rank = CQRRPT_GPU.rank;
        std::cout << "RANK AS RETURNED BY CQRRPT " << all_data.rank << "\n";

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy1, m, all_data.J);
        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy2, m, all_data.J);

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

    /// Runs alg on the m-by-n A0 with lda = m + pad and ldr = n + pad, A in pageable or pinned memory, and checks
    /// what every QRCP of the sketch must give: ret 0; J a permutation of 1..n; R zero below the diagonal and in
    /// rows past the rank; the padding untouched; for rank 0, A unchanged and J = 1..n; otherwise Q and R finite,
    /// with ||A0[:, J] - QR||_F / ||A0[:, J]||_F and ||Q'Q - I||_F / sqrt(rank) within bound. With cut, the first
    /// error is taken over the first rank columns of A0[:, J] only, the columns the factorization keeps. Returns
    /// the rank.
    template <typename T>
    static int64_t check_properties(RandLAPACK::CQRRPT_GPU<T, r123::Philox4x32>& alg, int64_t m, int64_t n, const std::vector<T>& A0,
                                    int64_t pad, bool pinned, uint32_t seed, T bound, bool cut) {
        int64_t lda = m + pad, ldr = n + pad;
        T sentinel = -777;
        T* Q = nullptr;
        std::vector<T> Q_pageable;
        if (pinned) {
            EXPECT_EQ(cudaMallocHost((void**) &Q, lda * n * sizeof(T)), cudaSuccess);
        } else {
            Q_pageable.resize(lda * n);
            Q = Q_pageable.data();
        }
        std::fill(Q, Q + lda * n, sentinel);
        lapack::lacpy(MatrixType::General, m, n, A0.data(), m, Q, lda);
        std::vector<T> R(ldr * n, std::numeric_limits<T>::quiet_NaN());
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = n; i < ldr; ++i)
                R[i + j * ldr] = sentinel;
        std::vector<int64_t> J(n, -3);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        alg.rank = -7;
        EXPECT_EQ(alg.call(m, n, Q, lda, R.data(), ldr, J.data(), (T) 2.0, state), 0);
        int64_t r = alg.rank;
        EXPECT_GE(r, 0);
        EXPECT_LE(r, n);

        std::vector<int64_t> J_sorted(J), identity(n);
        std::sort(J_sorted.begin(), J_sorted.end());
        std::iota(identity.begin(), identity.end(), (int64_t) 1);
        EXPECT_EQ(J_sorted, identity) << "J is not a permutation of 1..n";
        int64_t bad_R = 0, bad_pad = 0;
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i)
                bad_R += (i > j || i >= r) && R[i + j * ldr] != (T) 0;
            for (int64_t i = n; i < ldr; ++i)
                bad_pad += R[i + j * ldr] != sentinel;
            for (int64_t i = m; i < lda; ++i)
                bad_pad += Q[i + j * lda] != sentinel;
        }
        EXPECT_EQ(bad_R, 0) << "nonzero entries of R below the diagonal or past rank " << r;
        EXPECT_EQ(bad_pad, 0) << "padding entries changed";

        if (r == 0 || J_sorted != identity) {
            if (r == 0) {
                EXPECT_EQ(J, identity) << "rank 0 must come with J = 1..n";
                int64_t changed = 0;
                for (int64_t j = 0; j < n; ++j)
                    for (int64_t i = 0; i < m; ++i)
                        changed += !(Q[i + j * lda] == A0[i + j * m]);
                EXPECT_EQ(changed, 0) << "A changed on a rank-0 return";
            }
            if (pinned)
                cudaFreeHost(Q);
            return r;
        }
        std::vector<T> Qc(m * n), Rc(n * n), AP(A0), G(r * r, (T) 0);
        lapack::lacpy(MatrixType::General, m, n, Q, lda, Qc.data(), m);
        lapack::lacpy(MatrixType::General, n, n, R.data(), ldr, Rc.data(), n);
        if (pinned)
            cudaFreeHost(Q);
        bool finite = true;
        for (int64_t i = 0; i < m * r; ++i)
            finite = finite && std::isfinite(Qc[i]);
        for (int64_t i = 0; i < n * n; ++i)
            finite = finite && std::isfinite(Rc[i]);
        EXPECT_TRUE(finite) << "rank " << r << ": Q or R contains NaN or Inf";
        if (!finite)
            return r;
        std::vector<int64_t> Jc(J);
        RandLAPACK::util::col_swap(m, n, n, AP.data(), m, Jc.data());
        int64_t n_qr = cut ? r : n;
        T norm_AP = lapack::lange(Norm::Fro, m, n_qr, AP.data(), m);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n_qr, r, (T) -1.0, Qc.data(), m, Rc.data(), n, (T) 1.0, AP.data(), m);
        T err_qr = lapack::lange(Norm::Fro, m, n_qr, AP.data(), m) / norm_AP;
        RandLAPACK::util::eye(r, r, G.data());
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, (T) 1.0, Qc.data(), m, (T) -1.0, G.data(), r);
        T err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, G.data(), r) / std::sqrt((T) r);
        EXPECT_LE(err_qr, bound) << "rank " << r << ": ||A[:, J] - QR|| / ||A[:, J]||" << (cut ? " on the first rank columns" : "");
        EXPECT_LE(err_orth, bound) << "rank " << r << ": ||Q'Q - I|| / sqrt(rank)";
        return r;
    }

    struct PropertyInput { const char* name; int64_t m, n; uint32_t seed; int64_t rank; int64_t nnz; bool cut; };

    /// check_properties on each input with compact storage, and with padded storage and A pageable and pinned,
    /// all within u^0.75, with a sparse sketch of nnz nonzeros per column. rank >= 0 marks an input of known rank,
    /// which CQRRPT (with GEQP3) and CQRRPT_GPU must both find. cut marks an input on which the rank can come out
    /// below A's, after a failed potrf or from a sketch that misses most of A: there A[:, J] = QR is checked on the
    /// columns kept. On the failed-potrf input, rank 3 shows that the failure path ran.
    template <typename T>
    static void run_properties(RandLAPACK::CQRRPT_GPU_Subroutines::QRCP qrcp, int64_t bqrrp_block_size,
                               const std::vector<PropertyInput>& inputs, const std::vector<std::vector<T>>& As) {
        T tol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.85);
        T atol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.75);
        for (size_t c = 0; c < inputs.size(); ++c) {
            const auto& in = inputs[c];
            SCOPED_TRACE(::testing::Message() << in.name << (sizeof(T) == 8 ? ", double" : ", float"));
            int64_t m = in.m, n = in.n;
            if (in.rank >= 0) {
                std::vector<T> Q_cpu(As[c]), R_cpu(n * n, (T) 0);
                std::vector<int64_t> J_cpu(n, 0);
                RandBLAS::RNGState<r123::Philox4x32> state_cpu(in.seed);
                RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT(false, tol);
                CQRRPT.nnz = in.nnz;
                ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), (T) 2.0, state_cpu), 0);
                ASSERT_EQ(CQRRPT.rank, in.rank) << "CQRRPT does not find the input's rank";
            }
            for (auto [pad, pinned] : {std::pair<int64_t, bool>{0, false}, {5, false}, {5, true}}) {
                SCOPED_TRACE(::testing::Message() << "pad " << pad << (pinned ? ", pinned" : ", pageable"));
                RandLAPACK::CQRRPT_GPU<T, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
                CQRRPT_GPU.nnz = in.nnz;
                CQRRPT_GPU.qrcp = qrcp;
                CQRRPT_GPU.bqrrp_block_size = bqrrp_block_size;
                int64_t r = check_properties<T>(CQRRPT_GPU, m, n, As[c], pad, pinned, in.seed, atol, in.cut);
                if (in.rank >= 0)
                    EXPECT_EQ(r, in.rank);
            }
        }
    }

    template <typename T>
    static std::vector<T> generated(int64_t m, int64_t n, RandLAPACK::gen::mat_type type, T cond, int64_t rank, uint32_t seed) {
        std::vector<T> A(m * n, (T) 0);
        RandLAPACK::gen::mat_gen_info<T> m_info(m, n, type);
        m_info.cond_num = cond;
        m_info.rank = rank;
        m_info.exponent = 2.0;
        auto gen_state = RandBLAS::RNGState(seed);
        RandLAPACK::gen::mat_gen<T, r123::Philox4x32>(m_info, A.data(), gen_state);
        return A;
    }

    // Gaussian columns scaled by 2^(j / 10), so that the column norms differ and the pivots reorder them.
    template <typename T>
    static std::vector<T> scaled_gaussian(int64_t m, int64_t n, uint32_t seed) {
        std::vector<T> A(m * n);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandBLAS::DenseDist D(m, n);
        RandBLAS::fill_dense(D, A.data(), state);
        for (int64_t j = 0; j < n; ++j)
            blas::scal(m, (T) std::pow(2.0, (double) j / 10), &A[m * j], 1);
        return A;
    }

    /// Runs run_properties for one QRCP of the sketch on its double and float inputs.
    static void qrcp_properties(RandLAPACK::CQRRPT_GPU_Subroutines::QRCP qrcp, int64_t bqrrp_block_size = 0) {
        using RandLAPACK::gen::polynomial;
        using RandLAPACK::gen::exponential;
        std::vector<PropertyInput> in_d = {
            {"polynomial, full rank", 2000, 100, 1, 100, 2, false},
            {"polynomial, rank 60", 2000, 100, 2, 60, 2, false},
            {"polynomial, rank 1", 2000, 100, 6, 1, 2, false},
            {"exponential to 1e-12, full rank", 2000, 100, 3, 100, 2, false},
            {"scaled Gaussian", 3000, 120, 4, 120, 2, false},
            {"sketch-defeating", 600, 60, 7, -1, 2, true},
            {"failed potrf", 64, 8, 1, 3, 1, true},
            {"zero", 500, 40, 5, 0, 2, false}};
        std::vector<std::vector<double>> A_d = {
            generated<double>(2000, 100, polynomial, 2.0, 100, 1),
            generated<double>(2000, 100, polynomial, 2.0, 60, 2),
            generated<double>(2000, 100, polynomial, 2.0, 1, 6),
            generated<double>(2000, 100, exponential, 1e12, 100, 3),
            scaled_gaussian<double>(3000, 120, 4),
            sketch_defeating_input(600, 60, 2, 2.0, 7, 1.0),
            cqrrpt_test::cholesky_failure_input(64, 8, 16, 1, 4, 6),
            std::vector<double>(500 * 40, 0.0)};
        run_properties<double>(qrcp, bqrrp_block_size, in_d, A_d);
        // In float, the rounding residue past rank 60 can exceed the rank threshold 2 u^0.95, so that input is
        // checked for its properties only.
        std::vector<PropertyInput> in_s = {
            {"polynomial, full rank", 2000, 100, 1, 100, 2, false},
            {"polynomial, rank 60", 2000, 100, 2, -1, 2, false},
            {"polynomial, rank 1", 2000, 100, 6, 1, 2, false},
            {"scaled Gaussian", 3000, 120, 4, 120, 2, false},
            {"zero", 500, 40, 5, 0, 2, false}};
        std::vector<std::vector<float>> A_s = {
            generated<float>(2000, 100, polynomial, 2.0f, 100, 1),
            generated<float>(2000, 100, polynomial, 2.0f, 60, 2),
            generated<float>(2000, 100, polynomial, 2.0f, 1, 6),
            scaled_gaussian<float>(3000, 120, 4),
            std::vector<float>(500 * 40, 0.0f)};
        run_properties<float>(qrcp, bqrrp_block_size, in_s, A_s);
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_GPU_full_rank) {
    int64_t m = 1000;
    int64_t n = 200;
    int64_t k = 200;
    double d_factor = 2.0;
    double norm_A = 0;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<double> all_data(m, n, k);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general<double, r123::Philox4x32, RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32>>(d_factor, norm_A, all_data, CQRRPT_GPU, state);
}

// Single precision instantiates CQRRPT_GPU<float>, which must compile and factor to float accuracy.
TEST_F(TestCQRRPT, CQRRPT_GPU_full_rank_float) {
    int64_t m = 1000;
    int64_t n = 200;
    int64_t k = 200;
    float d_factor = 2.0;
    float norm_A = 0;
    float tol = std::pow(std::numeric_limits<float>::epsilon(), (float) 0.85);
    auto state = RandBLAS::RNGState();

    QR_test_data<float> all_data(m, n, k);
    RandLAPACK::CQRRPT_GPU<float, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.num_threads = 4;

    RandLAPACK::gen::mat_gen_info<float> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = k;
    m_info.exponent = 2.0;
    RandLAPACK::gen::mat_gen<float, r123::Philox4x32>(m_info, all_data.A, state);

    norm_and_copy_computational_helper(norm_A, all_data);
    test_CQRRPT_general<float, r123::Philox4x32, RandLAPACK::CQRRPT_GPU<float, r123::Philox4x32>>(d_factor, norm_A, all_data, CQRRPT_GPU, state);
    EXPECT_EQ(all_data.rank, k);
}

// GPU counterpart of CQRRPT_dirty_J_rank_deficient in test_cqrrpt.cc: dirty J buffers
// must not influence pivoting.
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
        QR_test_data<double> all_data(m, n, k);
        lapack::lacpy(MatrixType::General, m, n, A_orig.data(), m, all_data.A, m);

        // Trial 0 keeps the zero-initialized J; trials 1 and 2 dirty it with
        // different nonzero garbage.
        if (trial > 0) {
            for (int64_t i = 0; i < n; ++i)
                all_data.J[i] = 1 + ((7919 * trial + 31 * i) % n);
        }

        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.num_threads = 4;

        double norm_A = 0;
        norm_and_copy_computational_helper(norm_A, all_data);
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

// After a failed potrf, CQRRPT_GPU must return a valid factorization of the columns it keeps. On
// cholesky_failure_input with p = 3 and q = 7, potrf fails at column 8, the last, exactly and on any BLAS, and with
// tol = 1e-6 the condition test cuts at the large diagonal entry of column 4: rank 2, which shows that the failure
// path ran. J must be 1..n, R zero below the diagonal and past the rank, Q(:, 0:2) orthonormal, R(0:2, :) =
// Q(:, 0:2)' A[:, J], and A[:, J(0:2)] = Q(:, 0:2) R(0:2, 0:2), to 10 u; with both QRCPs of the sketch.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_failure_handled) {
    int64_t m = 128, n = 8, d = 16;
    uint32_t seed = 2;
    double tol = 1e-6;
    double u = std::numeric_limits<double>::epsilon();
    std::vector<double> A = cqrrpt_test::cholesky_failure_input(m, n, d, seed, 3, 7);
    ASSERT_EQ((int64_t) A.size(), m * n) << "the sketch from this seed cannot carry the input";

    for (auto qrcp : {RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::luqr, RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::bqrrp}) {
        SCOPED_TRACE(qrcp == RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::luqr ? "luqr" : "bqrrp");
        std::vector<double> Q(A), R(n * n, std::numeric_limits<double>::quiet_NaN());
        std::vector<int64_t> J(n, 0), identity(n);
        std::iota(identity.begin(), identity.end(), (int64_t) 1);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 1;
        CQRRPT_GPU.qrcp = qrcp;
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        int64_t r = CQRRPT_GPU.rank;
        ASSERT_EQ(r, 2) << "potrf no longer fails on this input, or the rank is cut elsewhere";
        ASSERT_EQ(J, identity) << "the sketch's QRCP must keep the columns in order";
        int64_t nonzero = 0;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i)
                nonzero += (i > j || i >= r) && R[i + j * n] != 0.0;
        EXPECT_EQ(nonzero, 0) << "nonzero entries of R below the diagonal or past rank " << r;
        auto c = factor_checks(m, n, r, A, Q, R, n, J);
        ASSERT_TRUE(c.finite) << "rank " << r << ": Q or R contains NaN or Inf";
        EXPECT_LE(c.err_orth, 10 * u);
        EXPECT_LE(c.err_proj, 10 * u);
        std::vector<double> AP(A);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, r, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
        EXPECT_LE(lapack::lange(Norm::Fro, m, r, AP.data(), m), 10 * u * lapack::lange(Norm::Fro, m, r, A.data(), m))
            << "||A[:, J(0:2)] - Q R(0:2, 0:2)||";
    }
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
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    EXPECT_EQ(CQRRPT_GPU.rank, n);
    auto c = factor_checks(m, n, CQRRPT_GPU.rank, A, Q, R, n, J);
    ASSERT_TRUE(c.finite);
    // CholQR loses orthogonality like u * cond^2 here; stay within 100x of the CPU driver.
    EXPECT_LE(c.err_orth, 100 * c_cpu.err_orth + 1e-12) << "CPU ||Q'Q - I|| = " << c_cpu.err_orth;
}

// After a failed potrf, CQRRPT cuts the rank a posteriori, with the rule that CQRRPT_GPU shares. On an input on
// which potrf fails at column 7 exactly, on any BLAS, and whose sketch both drivers' pivots keep in order
// (cholesky_failure_input), CQRRPT_GPU must report CQRRPT's rank, with Q orthonormal and R = Q' A[:, J]:
// - for tol from u^0.85 to 1e-3, the condition test cuts at the large diagonal entry of column 5: rank 3;
// - for tol = u / 2, it cuts at its first tested block, of order 3: rank 1;
// - for tol = 32 no block within the completed columns fails it: rank 5.
// The sketch-defeating inputs no longer make potrf fail here: luqr's pivots reorder their columns.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_failure_rank_matches_cpu) {
    int64_t m = 64, n = 8, d = 16;
    uint32_t seed = 1;
    std::vector<double> A = cqrrpt_test::cholesky_failure_input(m, n, d, seed, 4, 6);
    ASSERT_EQ((int64_t) A.size(), m * n) << "the sketch from this seed cannot carry the input";
    std::vector<int64_t> identity(n);
    std::iota(identity.begin(), identity.end(), (int64_t) 1);
    double u = std::numeric_limits<double>::epsilon();
    struct Case { double tol; int64_t rank; };
    for (auto c : {Case{std::pow(u, 0.85), 3}, Case{1e-12, 3}, Case{1e-6, 3}, Case{1e-3, 3}, Case{u / 2, 1}, Case{32.0, 5}}) {
        SCOPED_TRACE(c.tol);
        std::vector<double> Q_cpu(A), R_cpu(n * n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state_cpu(seed);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, c.tol);
        CQRRPT.nnz = 1;
        ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
        ASSERT_EQ(J_cpu, identity) << "CQRRPT's pivots must keep the columns in order";
        ASSERT_EQ(CQRRPT.rank, c.rank);

        std::vector<double> Q(A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, c.tol);
        CQRRPT_GPU.nnz = 1;
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        ASSERT_EQ(J, identity) << "CQRRPT_GPU's pivots must keep the columns in order";
        EXPECT_EQ(CQRRPT_GPU.rank, CQRRPT.rank);
        auto f = factor_checks(m, n, CQRRPT_GPU.rank, A, Q, R, n, J);
        ASSERT_TRUE(f.finite) << "rank " << CQRRPT_GPU.rank << ": Q or R contains NaN or Inf";
        EXPECT_LE(f.err_orth, 10 * u) << "rank " << CQRRPT_GPU.rank;
        EXPECT_LE(f.err_proj, 10 * u) << "rank " << CQRRPT_GPU.rank;
    }
}

// The two-column input of CQRRPT_cholesky_failure_k2_drops_failed_column, on which potrf fails at
// column 2, exactly and on any BLAS. With k = 2 the condition test, which starts at order 3, cannot cut the
// rank, so only the cap at the failed column keeps CQRRPT_GPU from keeping it: rank 1, as in CQRRPT.
TEST_F(TestCQRRPT, CQRRPT_GPU_cholesky_failure_k2_drops_failed_column) {
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
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 1;
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    ASSERT_EQ(J, (std::vector<int64_t>{1, 2})) << "the sketch's QRCP must not pivot";
    int64_t r = CQRRPT_GPU.rank;
    EXPECT_EQ(r, 1);

    // Q(:, 0:rank) orthonormal, and A[:, J] = Q R up to the dropped direction, of relative size 2e-9.
    auto c = factor_checks(m, n, r, A, Q, R, n, J);
    ASSERT_TRUE(c.finite) << "rank " << r << ": Q or R contains NaN or Inf";
    EXPECT_LE(c.err_orth, 10 * std::numeric_limits<double>::epsilon()) << "rank " << r;
    std::vector<double> AP(A);
    std::vector<int64_t> J_identity{1, 2};
    RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J_identity.data());
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
    double err_qr = lapack::lange(Norm::Fro, m, n, AP.data(), m) / lapack::lange(Norm::Fro, m, n, A.data(), m);
    EXPECT_LE(err_qr, 1e-8) << "rank " << r;
}

// With tol = u / 2 the threshold sqrt(tol/u) is below 1, so every block fails the condition test.
// When potrf succeeds there is no test, and the rank is the sketch's. When it fails, the rule
// cuts at its first tested block, of order 3, so the rank is 1, as in CQRRPT; it must not return
// a negative rank.
TEST_F(TestCQRRPT, CQRRPT_GPU_tol_at_most_unit_roundoff) {
    double tol = std::numeric_limits<double>::epsilon() / 2;
    struct Input { const char* name; int64_t m, n; std::vector<double> A; uint32_t seed; int64_t rank; int64_t nnz; };
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
        inputs.push_back({"potrf succeeds", m, n, A, 0, n, 2});
    }
    inputs.push_back({"potrf fails", 64, 8, cqrrpt_test::cholesky_failure_input(64, 8, 16, 1, 4, 6), 1, 1, 1});

    for (auto& in : inputs) {
        SCOPED_TRACE(in.name);
        int64_t m = in.m, n = in.n;
        std::vector<double> Q_cpu(in.A), R_cpu(n * n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state_cpu(in.seed);
        ASSERT_EQ((int64_t) in.A.size(), m * n);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = in.nnz;
        ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
        ASSERT_EQ(CQRRPT.rank, in.rank);

        std::vector<double> Q(in.A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(in.seed);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = in.nnz;
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
// device; rank deficiency (sketch rank k < n) also fills R's trailing columns.
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

// With timing on, results must not change; the trsm step (one trsm, between syncs) must last at least half a trsm
// alone, which a timer that does not wait for the device misses by far; and total, which covers the whole call, may
// fall short of the call's wall time by at most a quarter of the copies back of Q and R, timed alone. That catches a
// total that leaves out the copies back, the release of the device memory and queue, or a final device step.
TEST_F(TestCQRRPT, CQRRPT_GPU_timing_covers_device_work) {
    // Labeled exclusive and run alone by ctest (gpu_exclusive_tests.cmake): wall-clock bounds need an idle host and GPU.
    int64_t m = 5000;
    int64_t n = 1000;
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

    // The copies back alone: on a queue that has run trsm and potrf, copy Q (m-by-n) and R (n-by-n)
    // back to the host. Sets t_copy to the shortest of the calls.
    long t_copy = std::numeric_limits<long>::max();
    auto time_copies = [&]() {
        auto q = std::make_unique<lapack::Queue>(0);
        auto* d_info = blas::device_malloc<lapack::device_info_int>(1, *q);
        double *Q_device, *R_device, *R_sp_device;
        cudaMalloc(&Q_device, m * n * sizeof(double));
        cudaMalloc(&R_device, n * n * sizeof(double));
        cudaMalloc(&R_sp_device, n * n * sizeof(double));
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
    };

    auto run = [&](bool timing, std::vector<double>& Q, std::vector<double>& R, std::vector<int64_t>& J, long& wall) {
        Q = A;
        R.assign(n * n, 0.0);
        J.assign(n, 0);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, timing, tol);
        CQRRPT_GPU.nnz = 2;
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
    // A stall on a busy machine can break a single timed call, while the defects show in every
    // call: keep the longest trsm step, and the median time past total, of 3 timed calls.
    long step_trsm = 0;
    std::vector<long> pasts;
    for (int rep = 0; rep < 3; ++rep) {
        auto times = run(true, Q_on, R_on, J_on, wall);
        // times = {sketch, qrcp, rank_est, pivot, cholqr, trsm, rest, total}
        ASSERT_EQ(times.size(), 8u);
        for (int i = 0; i < 8; ++i)
            EXPECT_GE(times[i], 0) << "times[" << i << "]";
        pasts.push_back(wall - times[7]);
        step_trsm = std::max(step_trsm, times[5]);
        EXPECT_EQ(Q_on, Q_off);
        EXPECT_EQ(R_on, R_off);
        EXPECT_EQ(J_on, J_off);
        time_copies();
    }
    std::sort(pasts.begin(), pasts.end());
    long past = pasts[1];
    // The GPU's clocks, or other work on it, can slow either set of reference trsm runs.
    time_trsm();
    cudaFree(A_device);
    cudaFree(T_device);

    std::cout << "TRSM STEP " << step_trsm << " us, TRSM ALONE " << t_trsm << " us, COPIES BACK " << t_copy
              << " us, TIME PAST TOTAL " << past << " us\n";
    EXPECT_GE(step_trsm, t_trsm / 2) << "trsm step " << step_trsm << " us, trsm alone " << t_trsm << " us";
    EXPECT_LT(past, t_copy / 4) << "the call outlasts total by " << past << " us; the copies back alone take " << t_copy << " us";
}

// R's contents on entry must not matter: with R filled with NaN, CQRRPT_GPU must return exactly
// the R it returns for a zeroed R, and that R is zero below the diagonal and past the rank. On the
// failed-potrf input, rank 3 shows that the failure path ran.
TEST_F(TestCQRRPT, CQRRPT_GPU_R_contents_on_entry_ignored) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    struct Input { const char* name; int64_t m, n; std::vector<double> A; uint32_t seed; int64_t nnz; int64_t rank; };
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
        inputs.push_back({rank_A == n ? "full rank" : "rank 40", m, n, A, 0, 2, rank_A});
    }
    inputs.push_back({"failed potrf", 64, 8, cqrrpt_test::cholesky_failure_input(64, 8, 16, 1, 4, 6), 1, 1, 3});

    for (auto& in : inputs) {
        SCOPED_TRACE(in.name);
        int64_t m = in.m, n = in.n;
        std::vector<double> Q0(in.A), R0(n * n, 0.0), Q1(in.A), R1(n * n, std::numeric_limits<double>::quiet_NaN());
        std::vector<int64_t> J0(n, 0), J1(n, 0);
        ASSERT_EQ((int64_t) in.A.size(), m * n);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = in.nnz;
        RandBLAS::RNGState<r123::Philox4x32> state0(in.seed), state1(in.seed);
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q0.data(), m, R0.data(), n, J0.data(), 2.0, state0), 0);
        int64_t rank = CQRRPT_GPU.rank;
        EXPECT_EQ(rank, in.rank);
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
// return R as documented (zero) with A left zero, as CQRRPT does, and J = 1..n.
TEST_F(TestCQRRPT, CQRRPT_GPU_zero_input) {
    int64_t m = 100;
    int64_t n = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(m * n, 0.0), R(n * n, std::numeric_limits<double>::quiet_NaN());
    std::vector<int64_t> J(n, 0);

    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    EXPECT_EQ(CQRRPT_GPU.rank, 0);
    CQRRPT_GPU.nnz = 2;
    CQRRPT_GPU.rank = -7;
    auto state = RandBLAS::RNGState();
    ASSERT_EQ(CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    EXPECT_EQ(CQRRPT_GPU.rank, 0);
    EXPECT_EQ(R, std::vector<double>(n * n, 0.0));
    EXPECT_EQ(A, std::vector<double>(m * n, 0.0));
    std::vector<int64_t> identity(n);
    std::iota(identity.begin(), identity.end(), (int64_t) 1);
    EXPECT_EQ(J, identity);
}

// As in CQRRPT: an Inf or a NaN in A is reported (code 2), for both QRCPs of the sketch and in an otherwise
// zero A, and so is a NaN on the diagonal of the Gram matrix, planted by the fault point.
TEST_F(TestCQRRPT, CQRRPT_GPU_nonfinite_input_reported) {
    using QRCP = RandLAPACK::CQRRPT_GPU_Subroutines::QRCP;
    int64_t m = 100;
    int64_t n = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A_full(m * n, 0.0);
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
    m_info.cond_num = 2;
    m_info.rank = n;
    m_info.exponent = 2.0;
    auto gen_state = RandBLAS::RNGState();
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A_full.data(), gen_state);

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    struct Case { const char* name; QRCP qrcp; bool zero_A; double value; bool gram_fault; int code; };
    for (const Case& c : {Case{"Inf, luqr", QRCP::luqr, false, inf, false, 2},
                          Case{"NaN, luqr", QRCP::luqr, false, nan, false, 2},
                          Case{"Inf, bqrrp", QRCP::bqrrp, false, inf, false, 2},
                          Case{"NaN, bqrrp", QRCP::bqrrp, false, nan, false, 2},
                          Case{"NaN in a zero A", QRCP::luqr, true, nan, false, 2},
                          Case{"finite A, no fault", QRCP::luqr, false, 1.0, false, 0},
                          Case{"NaN planted in the Gram matrix", QRCP::luqr, false, 1.0, true, 2}}) {
        SCOPED_TRACE(c.name);
        std::vector<double> A = c.zero_A ? std::vector<double>(m * n, 0.0) : A_full;
        A[37 + 6 * m] = c.value;
        std::vector<double> R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        CQRRPT_GPU.qrcp = c.qrcp;
        CQRRPT_GPU.rank = -7;
        auto state = RandBLAS::RNGState();
        if (c.gram_fault)
            RandLAPACK::test_hooks::fault = [](const char* name) { return std::strcmp(name, "CQRRPT_GPU.gram_nan") == 0; };
        int code = CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, J.data(), 2.0, state);
        RandLAPACK::test_hooks::fault = nullptr;
        EXPECT_EQ(code, c.code);
        EXPECT_EQ(CQRRPT_GPU.rank, c.code == 0 ? n : 0);
    }
}

// As in CQRRPT: with d > m a row of A can get no weight in the sketch, and a NaN there is still reported.
TEST_F(TestCQRRPT, CQRRPT_GPU_nonfinite_in_an_unweighted_row_reported) {
    int64_t m = 8;
    int64_t n = 8;
    int64_t nnz = 2;
    double d_factor = 1.25;
    uint32_t seed = 0;
    int64_t row = cqrrpt_test::unweighted_row((int64_t) (d_factor * n), m, nnz, seed);
    ASSERT_GE(row, 0) << "no seed below 1000 leaves a row of A unweighted";
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    for (bool zero_A : {true, false}) {
        SCOPED_TRACE(zero_A ? "zero A" : "full-rank A");
        std::vector<double> A(m * n, 0.0), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        if (!zero_A)
            for (int64_t i = 0; i < m; ++i)
                A[i + i * m] = 1.0 + i;
        A[row + 3 * m] = std::numeric_limits<double>::quiet_NaN();
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = nnz;
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        EXPECT_EQ(CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, J.data(), d_factor, state), 2);
        EXPECT_EQ(CQRRPT_GPU.rank, 0);
    }
}

// A finite A whose sketch overflows is reported too, also for d > m, where A itself is scanned as well.
TEST_F(TestCQRRPT, CQRRPT_GPU_sketch_overflow_reported) {
    int64_t m = 2;
    int64_t n = 1;
    int64_t nnz = 2;
    double d_factor = 3.0;
    std::vector<double> A0(m * n, std::numeric_limits<double>::max());
    int64_t seed = cqrrpt_test::overflowing_sketch_seed(m, n, nnz, d_factor, A0);
    ASSERT_GE(seed, 0) << "no seed below 1000 overflows the sketch";
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(A0), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> alg(false, false, tol);
    alg.nnz = nnz;
    RandBLAS::RNGState<r123::Philox4x32> state((uint32_t) seed);
    EXPECT_EQ(alg.call(m, n, A.data(), m, R.data(), n, J.data(), d_factor, state), 2);
    EXPECT_EQ(alg.rank, 0);
}

// A finite sketch can still have a column norm above the largest double: with two nonzeros per column of S,
// the sketch of 0.8 * DBL_MAX has norm about 1.13 * DBL_MAX, and the QRCP leaves R_sk(0, 0) non-finite.
TEST_F(TestCQRRPT, CQRRPT_GPU_sketch_column_norm_overflow_reported) {
    int64_t m = 4;
    int64_t n = 1;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    for (auto qrcp : {RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::luqr, RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::bqrrp}) {
        SCOPED_TRACE((int) qrcp);
        std::vector<double> A(m * n, 0.0), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        A[0] = 0.8 * std::numeric_limits<double>::max();
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> alg(false, false, tol);
        alg.nnz = 2;
        alg.qrcp = qrcp;
        auto state = RandBLAS::RNGState();
        EXPECT_EQ(alg.call(m, n, A.data(), m, R.data(), n, J.data(), 3.0, state), 2);
        EXPECT_EQ(alg.rank, 0);
    }
}

// A in pinned host memory must give exactly the factorization of the same A in pageable memory, with
// compact and padded storage, at full rank, at rank 150, and after a failed potrf. The copy of a pinned A
// runs asynchronously, so it may overlap host work.
TEST_F(TestCQRRPT, CQRRPT_GPU_pinned_matches_pageable) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    // rank >= 0 is the rank both calls must give. On the rank-150 input, luqr's estimate can come out a little higher.
    struct Input { const char* name; int64_t m, n; std::vector<double> A; uint32_t seed; int64_t nnz; int64_t rank; };
    std::vector<Input> inputs;
    for (int64_t rank_A : {(int64_t) 200, (int64_t) 150}) {
        // A is 32 MB, so that its copy lasts milliseconds.
        int64_t m = 20000, n = 200;
        std::vector<double> A(m * n, 0.0);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::polynomial);
        m_info.cond_num = 2;
        m_info.rank = rank_A;
        m_info.exponent = 2.0;
        auto gen_state = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), gen_state);
        inputs.push_back({rank_A == n ? "full rank" : "rank 150", m, n, A, 0, 2, rank_A == n ? n : -1});
    }
    // potrf fails, and the rank is cut to 3; see CQRRPT_GPU_cholesky_failure_rank_matches_cpu.
    inputs.push_back({"failed potrf", 64, 8, cqrrpt_test::cholesky_failure_input(64, 8, 16, 1, 4, 6), 1, 1, 3});

    for (auto& in : inputs) {
        for (int64_t pad : {(int64_t) 0, (int64_t) 5}) {
            SCOPED_TRACE(::testing::Message() << in.name << ", lda = m + " << pad);
            int64_t m = in.m, n = in.n, lda = m + pad;
            std::vector<double> Q(lda * n, -777.0), R(n * n, 0.0), R_pin(n * n, 0.0);
            lapack::lacpy(MatrixType::General, m, n, in.A.data(), m, Q.data(), lda);
            double* Q_pin = nullptr;
            ASSERT_EQ(cudaMallocHost((void**) &Q_pin, lda * n * sizeof(double)), cudaSuccess);
            std::copy(Q.begin(), Q.end(), Q_pin);
            std::vector<int64_t> J(n, 0), J_pin(n, 0);
            RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
            CQRRPT_GPU.nnz = in.nnz;
            RandBLAS::RNGState<r123::Philox4x32> state(in.seed), state_pin(in.seed);
            ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), lda, R.data(), n, J.data(), 2.0, state), 0);
            int64_t rank = CQRRPT_GPU.rank;
            if (in.rank >= 0)
                EXPECT_EQ(rank, in.rank);
            ASSERT_EQ(CQRRPT_GPU.call(m, n, Q_pin, lda, R_pin.data(), n, J_pin.data(), 2.0, state_pin), 0);
            EXPECT_EQ(CQRRPT_GPU.rank, rank);
            EXPECT_EQ(J_pin, J);
            // Bytes, padding included: NaN entries would compare unequal as values.
            EXPECT_EQ(std::memcmp(Q_pin, Q.data(), lda * n * sizeof(double)), 0) << "Q differs";
            EXPECT_EQ(std::memcmp(R_pin.data(), R.data(), n * n * sizeof(double)), 0) << "R differs";
            cudaFreeHost(Q_pin);
        }
    }
}

// More rows than one launch of a 2D device kernel with 8-row thread blocks covers (8 * 65,535 =
// 524,280), with columns of distinct norms so that the pivots reorder them. Every row must be
// pivoted and orthogonalized.
TEST_F(TestCQRRPT, CQRRPT_GPU_very_tall) {
    int64_t m = 600000;
    int64_t n = 8;
    double tol  = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    std::vector<double> A(m * n);
    auto state = RandBLAS::RNGState();
    RandBLAS::DenseDist D(m, n);
    state = RandBLAS::fill_dense(D, A.data(), state);
    for (int64_t j = 0; j < n; ++j)
        blas::scal(m, std::pow(2.0, (double) j), &A[m * j], 1);

    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    std::vector<double> Q(A), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
    int64_t r = CQRRPT_GPU.rank;
    ASSERT_EQ(r, n);
    std::vector<int64_t> J_sorted(J), identity(n);
    std::sort(J_sorted.begin(), J_sorted.end());
    std::iota(identity.begin(), identity.end(), (int64_t) 1);
    ASSERT_EQ(J_sorted, identity) << "J is not a permutation";
    ASSERT_NE(J, identity) << "the pivots must reorder the columns for this test to check the pivoting of A";

    auto c = factor_checks(m, n, r, A, Q, R, n, J);
    ASSERT_TRUE(c.finite);
    EXPECT_LE(c.err_orth, atol);
    // ||A[:, J] - QR||_F / ||A||_F, over all rows and over the rows past 524,280 alone.
    std::vector<double> AP(A);
    RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
    double norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);
    int64_t tail = 524280;
    EXPECT_LE(lapack::lange(Norm::Fro, m, n, AP.data(), m) / norm_A, atol);
    EXPECT_LE(lapack::lange(Norm::Fro, m - tail, n, &AP[tail], m) / norm_A, atol);
}

// A sketch of more rows than one launch of the gather kernel covers (8 * 65,535 = 524,280): the pivoted sketch is
// gathered in blocks of rows, as A is. On a 600,000 x 8 A with d = 560,000, CQRRPT_GPU must find rank 8 with both
// errors within u^0.75. Column 0 of A is nonzero only in rows that the sketch adds into its rows past 524,280 alone,
// so a gather of the first 524,280 rows only would leave that column of the pivoted sketch zero, and the rank short.
// Column 1 is nonzero only in rows that the sketch adds into its first d - 524,280 rows alone, which a second block
// gathered to the start of the pivoted sketch, rather than past 524,280, would overwrite. On a zero A with
// d = m = 600,000, it must return rank 0, J = 1..n and R = 0. BQRRP_GPU gathers the sketch in one launch: with
// bqrrp, d = 524,280 must factor A within the same bounds, and d = 560,000 must be rejected (RandLAPACK::Error)
// with A and R unchanged.
TEST_F(TestCQRRPT, CQRRPT_GPU_sketch_taller_than_one_gather) {
    int64_t m = 600000, n = 8, tail = 524280;
    double d_factor = 70000;
    int64_t d = d_factor * n;
    uint32_t seed = 3;
    double tol  = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);

    // The rows of A whose nonzeros in the sparse sketching operator, drawn as CQRRPT_GPU draws it, all lie in
    // rows of the sketch past tail, and those whose nonzeros all lie in its first d - tail rows.
    std::vector<bool> tail_only(m, true), head_only(m, true);
    {
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandBLAS::SparseDist DS(d, m, 2);
        RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, state);
        RandBLAS::fill_sparse(S);
        for (int64_t e = 0; e < S.nnz; ++e) {
            if (S.rows[e] < tail)
                tail_only[S.cols[e]] = false;
            if (S.rows[e] >= d - tail)
                head_only[S.cols[e]] = false;
        }
    }
    std::vector<double> A(m * n);
    RandBLAS::RNGState<r123::Philox4x32> gen_state(1);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::fill_dense(D, A.data(), gen_state);
    int64_t rows_0 = 0, rows_1 = 0;
    for (int64_t i = 0; i < m; ++i) {
        rows_0 += tail_only[i];
        rows_1 += head_only[i];
        if (!tail_only[i])
            A[i] = 0.0;
        if (!head_only[i])
            A[i + m] = 0.0;
    }
    ASSERT_GE(rows_0, 100) << "too few rows of A reach only the tail of the sketch";
    ASSERT_GE(rows_1, 100) << "too few rows of A reach only the first d - tail rows of the sketch";
    for (int64_t j = 1; j < n; ++j)
        blas::scal(m, std::pow(2.0, (double) j), &A[m * j], 1);

    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    CQRRPT_GPU.nnz = 2;
    std::vector<double> Q(A), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state(seed);
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), d_factor, state), 0);
    int64_t r = CQRRPT_GPU.rank;
    EXPECT_EQ(r, n);
    auto c = factor_checks(m, n, r, A, Q, R, n, J);
    ASSERT_TRUE(c.finite) << "rank " << r << ": Q or R contains NaN or Inf";
    EXPECT_LE(c.err_orth, atol);
    std::vector<double> AP(A);
    RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
    EXPECT_LE(lapack::lange(Norm::Fro, m, n, AP.data(), m) / lapack::lange(Norm::Fro, m, n, A.data(), m), atol);

    std::vector<double> Z(m * n, 0.0), R_zero(n * n, std::numeric_limits<double>::quiet_NaN());
    std::vector<int64_t> J_zero(n, 0), identity(n);
    std::iota(identity.begin(), identity.end(), (int64_t) 1);
    RandBLAS::RNGState<r123::Philox4x32> state_zero(seed);
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Z.data(), m, R_zero.data(), n, J_zero.data(), 75000.0, state_zero), 0);
    EXPECT_EQ(CQRRPT_GPU.rank, 0);
    EXPECT_EQ(J_zero, identity);
    EXPECT_EQ(R_zero, std::vector<double>(n * n, 0.0));

    CQRRPT_GPU.qrcp = RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::bqrrp;
    std::vector<double> Q_b(A), R_b(n * n, 0.0);
    std::vector<int64_t> J_b(n, 0);
    RandBLAS::RNGState<r123::Philox4x32> state_b(seed);
    ASSERT_EQ(CQRRPT_GPU.call(m, n, Q_b.data(), m, R_b.data(), n, J_b.data(), (double) tail / n, state_b), 0);
    int64_t r_b = CQRRPT_GPU.rank;
    EXPECT_EQ(r_b, n);
    auto c_b = factor_checks(m, n, r_b, A, Q_b, R_b, n, J_b);
    ASSERT_TRUE(c_b.finite) << "bqrrp, rank " << r_b << ": Q or R contains NaN or Inf";
    EXPECT_LE(c_b.err_orth, atol) << "bqrrp";
    std::vector<double> AP_b(A);
    RandLAPACK::util::col_swap(m, n, n, AP_b.data(), m, J_b.data());
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r_b, -1.0, Q_b.data(), m, R_b.data(), n, 1.0, AP_b.data(), m);
    EXPECT_LE(lapack::lange(Norm::Fro, m, n, AP_b.data(), m) / lapack::lange(Norm::Fro, m, n, A.data(), m), atol) << "bqrrp";

    std::vector<double> Q_t(A), R_t(n * n, -777.0);
    RandBLAS::RNGState<r123::Philox4x32> state_t(seed);
    EXPECT_THROW(CQRRPT_GPU.call(m, n, Q_t.data(), m, R_t.data(), n, J_b.data(), d_factor, state_t), RandLAPACK::Error);
    EXPECT_EQ(Q_t, A) << "A changed before the throw";
    EXPECT_EQ(R_t, std::vector<double>(n * n, -777.0)) << "R changed before the throw";
}

// On inputs of known rank, CQRRPT_GPU must find CQRRPT's rank, which is the input's, with
// ||A[:, J] - QR|| / ||A|| and ||Q'Q - I|| / sqrt(rank) within u^0.75.
TEST_F(TestCQRRPT, CQRRPT_GPU_rank_matches_cpu) {
    double tol  = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    double atol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    struct Input { int64_t m, n, rank; RandLAPACK::gen::mat_type type; double cond; };
    for (auto in : {Input{2000, 50, 40, RandLAPACK::gen::polynomial, 2.0}, Input{1500, 120, 80, RandLAPACK::gen::polynomial, 2.0},
                    Input{3000, 300, 300, RandLAPACK::gen::polynomial, 2.0}, Input{4000, 200, 120, RandLAPACK::gen::polynomial, 1e3},
                    Input{2000, 100, 100, RandLAPACK::gen::exponential, 1e12}}) {
        SCOPED_TRACE(::testing::Message() << in.m << " x " << in.n << ", rank " << in.rank);
        int64_t m = in.m, n = in.n;
        std::vector<double> A(m * n, 0.0);
        RandLAPACK::gen::mat_gen_info<double> m_info(m, n, in.type);
        m_info.cond_num = in.cond;
        m_info.rank = in.rank;
        m_info.exponent = 2.0;
        auto gen_state = RandBLAS::RNGState(3);
        RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, A.data(), gen_state);

        std::vector<double> Q_cpu(A), R_cpu(n * n, 0.0);
        std::vector<int64_t> J_cpu(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state_cpu(11);
        RandLAPACK::CQRRPT<double, r123::Philox4x32> CQRRPT(false, tol);
        CQRRPT.nnz = 2;
        ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), 2.0, state_cpu), 0);
        ASSERT_EQ(CQRRPT.rank, in.rank);

        std::vector<double> Q(A), R(n * n, 0.0);
        std::vector<int64_t> J(n, 0);
        RandBLAS::RNGState<r123::Philox4x32> state(11);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = 2;
        ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, state), 0);
        int64_t r = CQRRPT_GPU.rank;
        EXPECT_EQ(r, CQRRPT.rank);

        auto c = factor_checks(m, n, r, A, Q, R, n, J);
        ASSERT_TRUE(c.finite);
        EXPECT_LE(c.err_orth, atol);
        std::vector<double> AP(A);
        RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, -1.0, Q.data(), m, R.data(), n, 1.0, AP.data(), m);
        EXPECT_LE(lapack::lange(Norm::Fro, m, n, AP.data(), m) / lapack::lange(Norm::Fro, m, n, A.data(), m), atol);
    }
}

// Repeated CQRRPT_GPU calls must not grow device memory use, on the full path and on the exits
// before CholQR: an all-zero input, and an exception from the sketch (nnz = 0, which RandBLAS
// rejects). cudaMemGetInfo is device wide, so this check assumes no other process allocates on the
// GPU meanwhile. A device copy of A kept per call would grow memory by about 120 MiB per round.
TEST_F(TestCQRRPT, CQRRPT_GPU_repeated_calls_no_leak) {
    // Labeled exclusive in ctest (gpu_exclusive_tests.cmake): free memory is device wide, so other GPU processes skew it.
    int64_t m = 20000;
    int64_t n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    int num_rounds = 3;
    int calls_per_round = 16;
    auto free_mib = []() {
        size_t free_bytes, total_bytes;
        cudaDeviceSynchronize();
        cudaMemGetInfo(&free_bytes, &total_bytes);
        return (double) free_bytes / (1 << 20);
    };
    std::vector<double> A_rand(m * n), A_zero(m * n, 0.0);
    auto state = RandBLAS::RNGState();
    RandBLAS::DenseDist D(m, n);
    RandBLAS::fill_dense(D, A_rand.data(), state);

    struct Case { const char* name; const std::vector<double>* A; int64_t nnz; int64_t rank; };
    const Case cases[] = {{"full rank", &A_rand, 2, n}, {"zero input", &A_zero, 2, 0}, {"exception from the sketch", &A_rand, 0, -1}};
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.nnz = c.nnz;
        std::vector<double> Q(m * n), R(n * n);
        std::vector<int64_t> J(n);
        auto run = [&]() {
            std::copy(c.A->begin(), c.A->end(), Q.begin());
            auto s = RandBLAS::RNGState();
            if (c.rank < 0) {
                EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, s));
            } else {
                EXPECT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), 2.0, s), 0);
                EXPECT_EQ(CQRRPT_GPU.rank, c.rank);
            }
        };

        // A first call absorbs one-time allocations.
        run();
        double min_growth = std::numeric_limits<double>::infinity();
        for (int r = 0; r < num_rounds; ++r) {
            double before = free_mib();
            for (int i = 0; i < calls_per_round; ++i)
                run();
            min_growth = std::min(min_growth, before - free_mib());
        }
        cudaError_t ierr = cudaGetLastError();
        ASSERT_EQ(ierr, cudaSuccess) << cudaGetErrorString(ierr);
        std::cout << std::setw(26) << c.name << ":  SMALLEST GROWTH PER ROUND " << min_growth << " MiB\n";
        EXPECT_LE(min_growth, 32.0) << "device memory grew by at least " << min_growth << " MiB per " << calls_per_round << " calls";
    }
}

// An empty A (0 x 16 or 16 x 0) has nothing to factor: CQRRPT_GPU must set rank 0, J to the identity,
// which is what GEQP3 returns for an empty matrix, and R to zero.
TEST_F(TestCQRRPT, CQRRPT_GPU_empty_input) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    for (auto [m, n] : {std::pair<int64_t, int64_t>{0, 16}, std::pair<int64_t, int64_t>{16, 0}}) {
        SCOPED_TRACE(::testing::Message() << m << " x " << n);
        std::vector<double> A(std::max<int64_t>(m * n, 1), 0.0), R(std::max<int64_t>(n * n, 1), std::numeric_limits<double>::quiet_NaN());
        std::vector<int64_t> J(std::max<int64_t>(n, 1), 0);
        RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
        CQRRPT_GPU.rank = -7;
        auto state = RandBLAS::RNGState();
        ASSERT_EQ(CQRRPT_GPU.call(m, n, A.data(), std::max<int64_t>(m, 1), R.data(), std::max<int64_t>(n, 1), J.data(), 2.0, state), 0);
        EXPECT_EQ(CQRRPT_GPU.rank, 0);
        for (int64_t j = 0; j < n; ++j)
            EXPECT_EQ(J[j], j + 1) << "j = " << j;
        for (int64_t i = 0; i < n * n; ++i)
            EXPECT_EQ(R[i], 0.0) << "i = " << i;
    }
}

// The arguments CQRRPT rejects, CQRRPT_GPU must reject too, before it reads A.
TEST_F(TestCQRRPT, CQRRPT_GPU_invalid_arguments) {
    int64_t m = 100, n = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(m * n, 1.0), R(n * n, 0.0);
    std::vector<int64_t> J(n, 0);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
    auto state = RandBLAS::RNGState();
    EXPECT_ANY_THROW(CQRRPT_GPU.call(-1, n, A.data(), m, R.data(), n, J.data(), 2.0, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, -1, A.data(), m, R.data(), n, J.data(), 2.0, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, A.data(), m - 1, R.data(), n, J.data(), 2.0, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n - 1, J.data(), 2.0, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, J.data(), 0.5, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, nullptr, m, R.data(), n, J.data(), 2.0, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, A.data(), m, nullptr, n, J.data(), 2.0, state));
    EXPECT_ANY_THROW(CQRRPT_GPU.call(m, n, A.data(), m, R.data(), n, nullptr, 2.0, state));
    EXPECT_EQ(A, std::vector<double>(m * n, 1.0)) << "A changed";
}

// The default nnz is CQRRPT's, 2: a CQRRPT_GPU that does not set it must return exactly what one
// that sets nnz = 2 returns.
TEST_F(TestCQRRPT, CQRRPT_GPU_default_nnz) {
    int64_t m = 2000, n = 50;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(m * n);
    auto gen_state = RandBLAS::RNGState();
    RandBLAS::DenseDist D(m, n);
    RandBLAS::fill_dense(D, A.data(), gen_state);

    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU_default(false, false, tol);
    EXPECT_EQ(CQRRPT_GPU_default.nnz, 2);
    RandLAPACK::CQRRPT_GPU<double, r123::Philox4x32> CQRRPT_GPU_nnz2(false, false, tol);
    CQRRPT_GPU_nnz2.nnz = 2;
    std::vector<double> Q0(A), Q1(A), R0(n * n), R1(n * n);
    std::vector<int64_t> J0(n), J1(n);
    auto state0 = RandBLAS::RNGState(), state1 = RandBLAS::RNGState();
    ASSERT_EQ(CQRRPT_GPU_default.call(m, n, Q0.data(), m, R0.data(), n, J0.data(), 2.0, state0), 0);
    ASSERT_EQ(CQRRPT_GPU_nnz2.call(m, n, Q1.data(), m, R1.data(), n, J1.data(), 2.0, state1), 0);
    EXPECT_EQ(CQRRPT_GPU_default.rank, n);
    EXPECT_EQ(CQRRPT_GPU_nnz2.rank, n);
    EXPECT_EQ(J0, J1);
    EXPECT_EQ(Q0, Q1);
    EXPECT_EQ(R0, R1);
}

// Every QRCP of the sketch must give a valid factorization: see check_properties and run_properties.
TEST_F(TestCQRRPT, CQRRPT_GPU_luqr_properties) {
    qrcp_properties(RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::luqr);
}

// Inputs that stall LU with partial pivoting on the sparse sketch: the rows of A that the sketch sums into one of
// its rows are all zero, so that row of the sketch is zero, and some columns of A are zero or exact multiples of
// others. Such an LU keeps the next column in place, zero or dependent as it may be. In double and float, and
// with both QRCPs of the sketch, CQRRPT_GPU must find CQRRPT's rank, A's, with finite factors and
// ||A[:, J] - QR|| / ||A|| and ||Q'Q - I|| / sqrt(rank) within u^0.75.
// - The sketch of seed 11 sums into its row 0 rows that are zero in A, and A's column 0 is zero.
// - A has 60 nonzero rows of 2000, so that about a third of the sketch's rows are zero, and three zero columns,
//   or three columns that are exact multiples of others; over the sketches of seeds 1 to 10.
TEST_F(TestCQRRPT, CQRRPT_GPU_lu_stall_inputs) {
    int64_t m = 2000, n = 50, d = 100, nnz = 2;
    auto gaussian = [](int64_t rows, int64_t cols, uint32_t seed) {
        std::vector<double> G(rows * cols);
        RandBLAS::RNGState<r123::Philox4x32> state(seed);
        RandBLAS::DenseDist D(rows, cols);
        RandBLAS::fill_dense(D, G.data(), state);
        return G;
    };
    // The rows of A that the sketch of seed sums into its row 0: sketch the identity, as CQRRPT does A.
    std::vector<int64_t> rows_in_row_0;
    {
        RandBLAS::RNGState<r123::Philox4x32> state(11);
        RandBLAS::SparseDist DS(d, m, nnz);
        RandBLAS::SparseSkOp<double, r123::Philox4x32> S(DS, state);
        RandBLAS::fill_sparse(S);
        std::vector<double> I(m * m, 0.0), SI(d * m, 0.0);
        for (int64_t i = 0; i < m; ++i)
            I[i + i * m] = 1.0;
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, m, 1.0, S, 0, 0, I.data(), m, 0.0, SI.data(), d);
        for (int64_t i = 0; i < m; ++i)
            if (SI[i * d] != 0.0)
                rows_in_row_0.push_back(i);
    }
    ASSERT_FALSE(rows_in_row_0.empty());
    std::vector<double> A_r00 = gaussian(m, n, 3);
    for (int64_t i : rows_in_row_0)
        for (int64_t j = 0; j < n; ++j)
            A_r00[i + j * m] = 0.0;
    for (int64_t i = 0; i < m; ++i)
        A_r00[i] = 0.0;
    // 60 nonzero rows, then three zero columns or three columns that copy others exactly (in any precision).
    std::vector<double> A_rows(m * n, 0.0), G = gaussian(60, n, 8);
    for (int64_t i = 0; i < 60; ++i)
        for (int64_t j = 0; j < n; ++j)
            A_rows[(37 * i + 11) % m + j * m] = G[i + j * 60];
    std::vector<double> A_zero_cols(A_rows), A_dep_cols(A_rows);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j : {5, 17, 30})
            A_zero_cols[i + j * m] = 0.0;
        A_dep_cols[i + 10 * m] = A_dep_cols[i + 3 * m];
        A_dep_cols[i + 20 * m] = -A_dep_cols[i + 7 * m];
        A_dep_cols[i + 33 * m] = 2 * A_dep_cols[i + 1 * m];
    }
    struct Input { const char* name; const std::vector<double>* A; int64_t rank; std::vector<uint32_t> seeds; };
    std::vector<Input> inputs = {
        {"sketch row 0 and column 0 zero", &A_r00, n - 1, {11}},
        {"60 nonzero rows, 3 zero columns", &A_zero_cols, n - 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
        {"60 nonzero rows, 3 dependent columns", &A_dep_cols, n - 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}};

    auto run = [&](auto zero, RandLAPACK::CQRRPT_GPU_Subroutines::QRCP qrcp) {
        using T = decltype(zero);
        T tol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.85);
        T atol = std::pow(std::numeric_limits<T>::epsilon(), (T) 0.75);
        for (auto& in : inputs) {
            std::vector<T> A(in.A->begin(), in.A->end());
            for (uint32_t seed : in.seeds) {
                SCOPED_TRACE(::testing::Message() << in.name << ", seed " << seed << (sizeof(T) == 8 ? ", double" : ", float")
                             << (qrcp == RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::luqr ? ", luqr" : ", bqrrp"));
                std::vector<T> Q_cpu(A), R_cpu(n * n, (T) 0);
                std::vector<int64_t> J_cpu(n, 0);
                RandBLAS::RNGState<r123::Philox4x32> state_cpu(seed);
                RandLAPACK::CQRRPT<T, r123::Philox4x32> CQRRPT(false, tol);
                CQRRPT.nnz = nnz;
                ASSERT_EQ(CQRRPT.call(m, n, Q_cpu.data(), m, R_cpu.data(), n, J_cpu.data(), (T) 2.0, state_cpu), 0);
                ASSERT_EQ(CQRRPT.rank, in.rank) << "CQRRPT does not find A's rank";

                std::vector<T> Q(A), R(n * n, (T) 0);
                std::vector<int64_t> J(n, 0);
                RandBLAS::RNGState<r123::Philox4x32> state(seed);
                RandLAPACK::CQRRPT_GPU<T, r123::Philox4x32> CQRRPT_GPU(false, false, tol);
                CQRRPT_GPU.nnz = nnz;
                CQRRPT_GPU.qrcp = qrcp;
                ASSERT_EQ(CQRRPT_GPU.call(m, n, Q.data(), m, R.data(), n, J.data(), (T) 2.0, state), 0);
                int64_t r = CQRRPT_GPU.rank;
                EXPECT_EQ(r, CQRRPT.rank);
                bool finite = true;
                for (int64_t i = 0; i < m * r; ++i)
                    finite = finite && std::isfinite(Q[i]);
                for (int64_t i = 0; i < n * n; ++i)
                    finite = finite && std::isfinite(R[i]);
                EXPECT_TRUE(finite) << "rank " << r << ": Q or R contains NaN or Inf";
                if (!finite || r <= 0)
                    continue;
                std::vector<T> AP(A), Gram(r * r, (T) 0);
                RandLAPACK::util::col_swap(m, n, n, AP.data(), m, J.data());
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, r, (T) -1.0, Q.data(), m, R.data(), n, (T) 1.0, AP.data(), m);
                T err_qr = lapack::lange(Norm::Fro, m, n, AP.data(), m) / lapack::lange(Norm::Fro, m, n, A.data(), m);
                RandLAPACK::util::eye(r, r, Gram.data());
                blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, r, m, (T) 1.0, Q.data(), m, (T) -1.0, Gram.data(), r);
                T err_orth = lapack::lansy(lapack::Norm::Fro, Uplo::Upper, r, Gram.data(), r) / std::sqrt((T) r);
                EXPECT_LE(err_qr, atol) << "rank " << r << ": ||A[:, J] - QR|| / ||A||";
                EXPECT_LE(err_orth, atol) << "rank " << r << ": ||Q'Q - I|| / sqrt(rank)";
            }
        }
    };
    for (auto qrcp : {RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::luqr, RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::bqrrp}) {
        run(0.0, qrcp);
        run(0.0f, qrcp);
    }
}

// BQRRP_GPU on the sketch, with the default block size (one block of n columns here) and with blocks of 32
// columns, which take BQRRP_GPU through several iterations.
TEST_F(TestCQRRPT, CQRRPT_GPU_bqrrp_properties) {
    for (int64_t b_sz : {(int64_t) 0, (int64_t) 32}) {
        SCOPED_TRACE(::testing::Message() << "bqrrp_block_size " << b_sz);
        qrcp_properties(RandLAPACK::CQRRPT_GPU_Subroutines::QRCP::bqrrp, b_sz);
    }
}
