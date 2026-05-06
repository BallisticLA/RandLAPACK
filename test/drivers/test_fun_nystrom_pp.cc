#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <numeric>

namespace linops = RandLAPACK::linops;

class TestFunNystromPP : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    // Exact reference for tr(f(A)) via dense syevd: sum of f(λ_i).
    // A_copy passed by value because syevd overwrites it with eigenvectors.
    static double true_trace_fa(
        int64_t n, std::vector<double> A_copy,
        std::function<double(double)> f
    ) {
        std::vector<double> eigvals(n);
        lapack::syevd(lapack::Job::NoVec, lapack::Uplo::Upper, n,
                      A_copy.data(), n, eigvals.data());
        double tr = 0.0;
        for (int64_t i = 0; i < n; ++i)
            tr += f(eigvals[i]);
        return tr;
    }

    // Full algorithm stack. error_est_power_iters=0 → fixed-rank single pass in NystromEVD.
    template <typename RNG>
    struct Algs {
        using SYPS_t   = RandLAPACK::SYPS<double, RNG>;
        using Orth_t   = RandLAPACK::HQRQ<double>;
        using SYRF_t   = RandLAPACK::SYRF<SYPS_t, Orth_t>;
        using NystromEVD_t  = RandLAPACK::NystromEVD<SYRF_t>;
        using LFA_t    = RandLAPACK::LanczosFA<double, RNG>;
        using Hutch_t  = RandLAPACK::Hutchinson<double, RNG>;
        using Driver_t = RandLAPACK::FunNystromPP<NystromEVD_t, LFA_t, Hutch_t>;

        SYPS_t   syps;
        Orth_t   orth;
        SYRF_t   syrf;
        NystromEVD_t  nystrom_evd;
        LFA_t    lfa;
        Hutch_t  hutch;
        Driver_t driver;

        Algs() :
            syps(3, 1, false, false),
            orth(false, false),
            syrf(syps, orth),
            nystrom_evd(syrf, 0),
            driver(nystrom_evd, lfa, hutch) {}
    };

    // Common test body: run the full pipeline, print and assert relative error.
    template <typename RNG, typename F>
    static void test_FunNystromPP_general(
        Algs<RNG>& algs,
        linops::ExplicitSymLinOp<double>& A_op,
        F f, double f_zero, double true_tr,
        int64_t k, int64_t s, int64_t d, double tol,
        RandBLAS::RNGState<RNG>& state
    ) {
        double est = algs.driver.call(A_op, f, f_zero, k, s, d, state);
        double rel_err = std::abs(est - true_tr) / true_tr;
        printf("FunNystromPP: est=%e, true=%e, rel_err=%e\n", est, true_tr, rel_err);
        ASSERT_LT(rel_err, tol);
    }
};


// Random dense PSD matrix A = B'B + n*I, f=sqrt. Reference via syevd.
// k=10 << n=30: Nyström is coarse, Hutchinson correction carries real weight. Tol = 10%.
TEST_F(TestFunNystromPP, DensePSDSqrt) {
    using RNG = r123::Philox4x32;
    int64_t n = 30, k = 10, s = 200, d = 15;
    auto state = RandBLAS::RNGState(0);

    std::vector<double> B_raw(n * n);
    RandBLAS::DenseDist DB(n, n);
    state = RandBLAS::fill_dense(DB, B_raw.data(), state);

    std::vector<double> A(n * n, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, B_raw.data(), n, 0.0, A.data(), n);
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] += (double)n;
    // Symmetrize so both syevd (Uplo::Upper) and true_trace_fa see a consistent matrix.
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            A[i + j * n] = A[j + i * n];

    auto f_sqrt = [](double x){ return std::sqrt(std::max(x, 0.0)); };
    double true_tr = true_trace_fa(n, A, f_sqrt);

    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    test_FunNystromPP_general(algs, A_op, f_sqrt, 0.0, true_tr, k, s, d, 0.1, state);
}


// Diagonal A = diag(1,...,n), f=sqrt. True tr(sqrt(A)) = Σ sqrt(i) requires no eigensolver.
// k=15 of n=50 captures the dominant eigenvalues; tail is (n-k)*f(0)=0. Tol = 5%.
TEST_F(TestFunNystromPP, DiagonalSqrt) {
    using RNG = r123::Philox4x32;
    int64_t n = 50, k = 15, s = 300, d = 20;
    auto state = RandBLAS::RNGState(1);

    std::vector<double> A(n * n, 0.0);
    double true_tr = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (double)(i + 1);
        true_tr += std::sqrt((double)(i + 1));
    }

    auto f_sqrt = [](double x){ return std::sqrt(x); };
    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    test_FunNystromPP_general(algs, A_op, f_sqrt, 0.0, true_tr, k, s, d, 0.05, state);
}


// A = gen_shifted_lowrank_psd(noise=1): eigenvalues {lambda_j+1} ∪ {1 repeated n-k_mat times}.
// f=log; all eigenvalues ≥ 1 so log is well-defined. f(noise) = log(1) = 0 → f_zero=0.
// true_tr = Σ log(lambda_j + 1); (n-k_mat) tail terms vanish. Tol = 20%.
TEST_F(TestFunNystromPP, ShiftedLowRankLog) {
    using RNG = r123::Philox4x32;
    int64_t n = 40, k_mat = 20, k = 15, s = 500, d = 20;
    double noise = 1.0;
    auto state = RandBLAS::RNGState(2);

    std::vector<double> eigvals(k_mat);
    RandLAPACK::gen::gen_alg_decay_singvals(k_mat, 100.0, 2.0, eigvals.data());

    std::vector<double> A(n * n, 0.0);
    RandLAPACK::gen::gen_shifted_lowrank_psd(n, k_mat, A.data(), n, eigvals.data(), noise, state);

    auto f_log = [](double x){ return std::log(x); };
    double true_tr = 0.0;
    for (int64_t j = 0; j < k_mat; ++j)
        true_tr += f_log(eigvals[j] + noise);
    // tail: (n - k_mat) * log(1) = 0

    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    test_FunNystromPP_general(algs, A_op, f_log, 0.0, true_tr, k, s, d, 0.20, state);
}


// A = gen_sym_psd_lowrank; f=x*(x+1), degree-2 polynomial. d=2 Lanczos steps is exact for
// degree-2 trace estimation via Gauss quadrature (exact for degree ≤ 2d-1 = 3 quadratic forms).
// true_tr = Σ lambda_j*(lambda_j+1); (n-k_mat) tail terms: 0*(0+1)=0. Tol = 10%.
TEST_F(TestFunNystromPP, LowRankPoly) {
    using RNG = r123::Philox4x32;
    int64_t n = 40, k_mat = 20, k = 15, s = 200, d = 2;
    double lam = 1.0;
    auto state = RandBLAS::RNGState(3);

    std::vector<double> eigvals(k_mat);
    RandLAPACK::gen::gen_alg_decay_singvals(k_mat, 100.0, 2.0, eigvals.data());

    std::vector<double> A(n * n, 0.0);
    RandLAPACK::gen::gen_sym_psd_lowrank(n, k_mat, A.data(), n, eigvals.data(), state);

    auto f_poly = [lam](double x){ return x * (x + lam); };
    double true_tr = 0.0;
    for (int64_t j = 0; j < k_mat; ++j)
        true_tr += f_poly(eigvals[j]);
    // tail: (n - k_mat) * f(0) = 0

    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    test_FunNystromPP_general(algs, A_op, f_poly, 0.0, true_tr, k, s, d, 0.10, state);
}


// A = RBF kernel matrix via gen_kernel_matrix (n=30, d_dim=5). Eigenvalues in (0,1].
// f=sqrt; reference via syevd. k=10, s=300, d=15. Tol = 15%.
TEST_F(TestFunNystromPP, KernelRBFSqrt) {
    using RNG = r123::Philox4x32;
    int64_t n = 30, d_dim = 5, k = 10, s = 300, d = 15;
    auto state = RandBLAS::RNGState(4);

    std::vector<double> K(n * n, 0.0);
    double bandwidth = std::sqrt((double)d_dim);
    RandLAPACK::gen::gen_kernel_matrix<double, RNG>(
        n, d_dim, K.data(), n, 0, bandwidth, 0.0, 0, state);

    auto f_sqrt = [](double x){ return std::sqrt(std::max(x, 0.0)); };
    double true_tr = true_trace_fa(n, K, f_sqrt);

    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, K.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    test_FunNystromPP_general(algs, A_op, f_sqrt, 0.0, true_tr, k, s, d, 0.15, state);
}


// At k=k_mat on a rank-k_mat matrix, NystromEVD captures the entire spectrum, so Phase 1
// computes tr(f(A)) to machine precision and Phase 2's true correction is zero.
// However, Phase 2 (Hutchinson+LanczosFA) produces a small nonzero t2 because LanczosFA
// and the GEMM-based f(Â)*Omega in ResidualOp follow different floating-point arithmetic
// paths, disagreeing at ~1e-14 per column. Hutchinson accumulates this into a systematic
// bias of roughly 1e-5 relative. This test documents that expected behavior:
//   - Phase 1 (t1) is machine-precision accurate: err < 1e-10
//   - Total error is bounded by the Phase 2 floor: err < 1e-3
//   - t2 is nonzero but small (the two-path arithmetic residual)
// To hit machine precision on the total, skip Phase 2 when k >= effective rank.
TEST_F(TestFunNystromPP, Phase1MachinePrecisionAtKmat) {
    using RNG = r123::Philox4x32;
    int64_t n = 200, k_mat = 10, k = 10, s = 200, d = 20;
    auto state = RandBLAS::RNGState(7);

    std::vector<double> eigvals(k_mat);
    RandLAPACK::gen::gen_alg_decay_singvals(k_mat, 100.0, 2.0, eigvals.data());

    std::vector<double> A(n * n, 0.0);
    RandLAPACK::gen::gen_sym_psd_lowrank(n, k_mat, A.data(), n, eigvals.data(), state);

    auto f = [](double x){ return std::sqrt(std::max(x, 0.0)); };
    double f_zero = 0.0;
    double true_tr = 0.0;
    for (int64_t j = 0; j < k_mat; ++j)
        true_tr += f(eigvals[j]);

    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    double est = algs.driver.call(A_op, f, f_zero, k, s, d, state);

    double t1 = 0.0;
    for (int64_t i = 0; i < k; ++i)
        t1 += algs.driver.F_vec[i];
    double t2 = est - t1;

    double err_t1  = std::abs(t1  - true_tr) / true_tr;
    double err_tot = std::abs(est - true_tr) / true_tr;

    printf("Phase1MachinePrecision: t1=%e  t2=%e  err_t1=%e  err_tot=%e\n",
           t1, t2, err_t1, err_tot);

    ASSERT_LT(err_t1,  1e-10);  // Phase 1 machine precision at k=k_mat
    ASSERT_LT(err_tot, 1e-3);   // total bounded by Phase 2 LFA floor
}
