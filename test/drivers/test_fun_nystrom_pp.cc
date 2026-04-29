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

    // Full algorithm stack. error_est_power_iters=0 → fixed-rank single pass in REVD2.
    template <typename RNG>
    struct Algs {
        using SYPS_t   = RandLAPACK::SYPS<double, RNG>;
        using Orth_t   = RandLAPACK::HQRQ<double>;
        using SYRF_t   = RandLAPACK::SYRF<SYPS_t, Orth_t>;
        using REVD2_t  = RandLAPACK::REVD2<SYRF_t>;
        using LFA_t    = RandLAPACK::LanczosFA<double, RNG>;
        using Hutch_t  = RandLAPACK::Hutchinson<double, RNG>;
        using Driver_t = RandLAPACK::FunNystromPP<REVD2_t, LFA_t, Hutch_t>;

        SYPS_t   syps;
        Orth_t   orth;
        SYRF_t   syrf;
        REVD2_t  revd2;
        LFA_t    lfa;
        Hutch_t  hutch;
        Driver_t driver;

        Algs() :
            syps(3, 1, false, false),
            orth(false, false),
            syrf(syps, orth),
            revd2(syrf, 0),
            driver(revd2, lfa, hutch) {}
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
