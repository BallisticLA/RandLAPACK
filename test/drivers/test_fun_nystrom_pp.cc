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

    // Compute tr(f(A)) exactly via dense eigendecomposition.
    // A is n×n column-major symmetric; returns sum of f(λ_i).
    static double true_trace_fa(
        int64_t n, std::vector<double> A_copy,
        std::function<double(double)> f
    ) {
        std::vector<double> eigvals(n);
        // dsyevd overwrites A with eigenvectors; we only need eigenvalues
        lapack::syevd(lapack::Job::NoVec, lapack::Uplo::Upper, n,
                      A_copy.data(), n, eigvals.data());
        double tr = 0.0;
        for (int64_t i = 0; i < n; ++i)
            tr += f(eigvals[i]);
        return tr;
    }

    // Construct algorithm stack for FunNystromPP
    template <typename RNG>
    struct Algs {
        using SYPS_t    = RandLAPACK::SYPS<double, RNG>;
        using Orth_t    = RandLAPACK::HQRQ<double>;
        using SYRF_t    = RandLAPACK::SYRF<SYPS_t, Orth_t>;
        using REVD2_t   = RandLAPACK::REVD2<SYRF_t>;
        using LFA_t     = RandLAPACK::LanczosFA<double, RNG>;
        using Hutch_t   = RandLAPACK::Hutchinson<double, RNG>;
        using Driver_t  = RandLAPACK::FunNystromPP<REVD2_t, LFA_t, Hutch_t>;

        SYPS_t  syps;
        Orth_t  orth;
        SYRF_t  syrf;
        REVD2_t revd2;
        LFA_t   lfa;
        Hutch_t hutch;
        Driver_t driver;

        Algs() :
            syps(3, 1, false, false),
            orth(false, false),
            syrf(syps, orth),
            revd2(syrf, 0),      // error_est_power_iters=0 → fixed-rank single pass
            driver(revd2, lfa, hutch) {}
    };
};


// FunNystromPP on a small dense PSD matrix with f=sqrt.
// Compare estimate to tr(sqrt(A)) from direct eigendecomposition.
TEST_F(TestFunNystromPP, DensePSDSqrt) {
    using RNG = r123::Philox4x32;
    int64_t n = 30, k = 10, s = 200, d = 15;
    auto state = RandBLAS::RNGState(0);

    // Generate a random PSD matrix: A = B^T B + I (ensures strict PD)
    std::vector<double> B_raw(n * n);
    RandBLAS::DenseDist DB(n, n);
    state = RandBLAS::fill_dense(DB, B_raw.data(), state);

    std::vector<double> A(n * n, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, B_raw.data(), n, 0.0, A.data(), n);
    // Add identity to ensure strict positive definiteness
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] += (double)n;
    // Fill lower triangle from upper (ExplicitSymLinOp only reads one triangle,
    // but true_trace_fa uses syevd which needs a full triangular half)
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            A[i + j * n] = A[j + i * n];

    // True tr(sqrt(A)) via dense eigendecomp
    auto f_sqrt = [](double x){ return std::sqrt(std::max(x, 0.0)); };
    double true_tr = true_trace_fa(n, A, f_sqrt);

    // FunNystromPP estimate
    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    Algs<RNG> algs;
    double est = algs.driver.call(A_op, f_sqrt, 0.0, k, s, d, state);

    double rel_err = std::abs(est - true_tr) / true_tr;
    printf("FunNystromPP sqrt: est=%e, true=%e, rel_err=%e\n", est, true_tr, rel_err);
    ASSERT_LT(rel_err, 0.1);
}


// FunNystromPP on a diagonal matrix with f=sqrt — easier to control.
// Diagonal A = diag(1, 2, ..., n), so tr(sqrt(A)) = Σ sqrt(i) exactly.
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
    double est = algs.driver.call(A_op, f_sqrt, 0.0, k, s, d, state);

    double rel_err = std::abs(est - true_tr) / true_tr;
    printf("FunNystromPP diagonal sqrt: est=%e, true=%e, rel_err=%e\n", est, true_tr, rel_err);
    ASSERT_LT(rel_err, 0.05);
}


// Verify that REVD2 with error_est_power_iters=0 and tol=0 does not increase k.
// REVD2 takes k by reference and may grow it adaptively; the fixed-rank setting
// must leave k unchanged so FunNystromPP's (n-k)*f(0) tail correction is correct.
TEST_F(TestFunNystromPP, REVD2FixedKBypass) {
    using RNG = r123::Philox4x32;
    int64_t n = 40, k = 10;
    auto state = RandBLAS::RNGState(5);

    // Simple diagonal PSD matrix
    std::vector<double> A(n * n, 0.0);
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] = (double)(i + 1);

    Algs<RNG> algs;
    std::vector<double> V_out(n * k, 0.0), eigvals_out(k, 0.0);
    int64_t k_before = k;
    algs.revd2.call(blas::Uplo::Upper, n, A.data(), k, 0.0, V_out, eigvals_out, state);

    printf("REVD2 fixed-k: k_before=%lld, k_after=%lld\n", (long long)k_before, (long long)k);
    ASSERT_EQ(k, k_before);
}
