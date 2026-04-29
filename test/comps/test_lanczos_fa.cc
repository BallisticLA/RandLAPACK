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

class TestLanczosFA : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    static std::vector<double> make_diag_matrix(const std::vector<double>& diag) {
        int64_t n = diag.size();
        std::vector<double> A(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            A[i + i * n] = diag[i];
        return A;
    }

    // Exact reference for f(A)B when A = diag(d): result[:,j] = f(d) .* B[:,j].
    static std::vector<double> diag_fa_ref(
        const std::vector<double>& d, const std::vector<double>& B,
        int64_t n, int64_t s, std::function<double(double)> f
    ) {
        std::vector<double> out(n * s, 0.0);
        for (int64_t j = 0; j < s; ++j)
            for (int64_t i = 0; i < n; ++i)
                out[j * n + i] = f(d[i]) * B[j * n + i];
        return out;
    }

    // Exact reference for f(A)B for general symmetric A via dense eigendecomposition:
    // A = Q diag(λ) Q' (syevd, reads upper triangle), then f(A)B = Q diag(f(λ)) Q'B.
    // A_copy passed by value because syevd overwrites it with eigenvectors.
    static std::vector<double> dense_fa_ref(
        std::vector<double> A_copy,
        const std::vector<double>& B,
        int64_t n, int64_t s,
        std::function<double(double)> f
    ) {
        std::vector<double> eigvals(n);
        lapack::syevd(lapack::Job::Vec, lapack::Uplo::Upper, n,
                      A_copy.data(), n, eigvals.data());
        std::vector<double> tmp(n * s);
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                   n, s, n, 1.0, A_copy.data(), n, B.data(), n, 0.0, tmp.data(), n);
        for (int64_t j = 0; j < s; ++j)
            for (int64_t i = 0; i < n; ++i)
                tmp[j * n + i] *= f(eigvals[i]);
        std::vector<double> out(n * s);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   n, s, n, 1.0, A_copy.data(), n, tmp.data(), n, 0.0, out.data(), n);
        return out;
    }

    // A = diag(1,...,n); reference is diag_fa_ref (free, exact).
    // With d < n steps the output is approximate; tolerance encodes expected accuracy.
    template <typename F>
    static void run_diagonal_fa_test(F f, int64_t n, int64_t s, int64_t d,
                                     double tol, uint64_t seed) {
        using RNG = r123::Philox4x32;
        auto state = RandBLAS::RNGState(seed);
        std::vector<double> diag_vec(n);
        std::iota(diag_vec.begin(), diag_vec.end(), 1.0);
        std::vector<double> A_mat = make_diag_matrix(diag_vec);
        std::vector<double> B(n * s);
        RandBLAS::DenseDist DB(n, s);
        state = RandBLAS::fill_dense(DB, B.data(), state);
        std::vector<double> out(n * s, 0.0);
        linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A_mat.data(), n, Layout::ColMajor);
        RandLAPACK::LanczosFA<double, RNG> lfa;
        lfa.reorth = 1;
        lfa.call(A_op, B.data(), n, s, f, d, out.data());
        auto ref = diag_fa_ref(diag_vec, B, n, s, f);
        double err = 0.0, ref_norm = 0.0;
        for (int64_t i = 0; i < n * s; ++i) {
            double r = out[i] - ref[i];
            err      += r * r;
            ref_norm += ref[i] * ref[i];
        }
        double rel_err = std::sqrt(err / ref_norm);
        printf("||f(A)B - ref||/||ref|| = %e\n", rel_err);
        ASSERT_LT(rel_err, tol);
    }

    // A = B'B + n*I (κ ≈ 5, λ_min ≥ n); reference is dense_fa_ref (syevd).
    // Upper triangle only — ExplicitSymLinOp and syevd both read Uplo::Upper.
    template <typename F>
    static void run_dense_fa_test(F f, int64_t n, int64_t s, int64_t d,
                                  double tol, uint64_t seed) {
        using RNG = r123::Philox4x32;
        auto state = RandBLAS::RNGState(seed);
        std::vector<double> B_raw(n * n);
        RandBLAS::DenseDist D1(n, n);
        state = RandBLAS::fill_dense(D1, B_raw.data(), state);
        std::vector<double> A(n * n, 0.0);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, n, 1.0, B_raw.data(), n, 0.0, A.data(), n);
        for (int64_t i = 0; i < n; ++i)
            A[i + i * n] += (double)n;
        std::vector<double> B(n * s);
        RandBLAS::DenseDist D2(n, s);
        state = RandBLAS::fill_dense(D2, B.data(), state);
        std::vector<double> out(n * s, 0.0);
        linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
        RandLAPACK::LanczosFA<double, RNG> lfa;
        lfa.reorth = 1;
        lfa.call(A_op, B.data(), n, s, f, d, out.data());
        auto ref = dense_fa_ref(A, B, n, s, f);
        double err = 0.0, ref_norm = 0.0;
        for (int64_t i = 0; i < n * s; ++i) {
            double r = out[i] - ref[i];
            err      += r * r;
            ref_norm += ref[i] * ref[i];
        }
        double rel_err = std::sqrt(err / ref_norm);
        printf("||f(A)B - ref||/||ref|| = %e\n", rel_err);
        ASSERT_LT(rel_err, tol);
    }
};


// f=sqrt on diag(1,...,50), d=20 < n=50. Good but not exact (Krylov subspace smaller than spectrum).
TEST_F(TestLanczosFA, DiagonalSqrt) {
    run_diagonal_fa_test([](double x){ return std::sqrt(x); }, 50, 5, 20, 1e-4, 42);
}

// f=log on diag(1,...,50). Entries ≥ 1 so log is well-defined. Tolerance relaxed to 1e-3
// (log has slower polynomial convergence on [1,50] than sqrt).
TEST_F(TestLanczosFA, DiagonalLog) {
    run_diagonal_fa_test([](double x){ return std::log(x); }, 50, 4, 20, 1e-3, 7);
}

// f=sqrt on a random dense PSD matrix. d=n/2=25 steps → near-exact convergence on κ≈5.
TEST_F(TestLanczosFA, DensePSDSqrt) {
    run_dense_fa_test([](double x){ return std::sqrt(std::max(x, 0.0)); }, 50, 4, 25, 1e-6, 31);
}

// f=exp on the same dense PSD setup. Eigenvalues in [50,~250] stay within double range.
// Exercises a function not covered by the diagonal tests (per the reference implementation).
TEST_F(TestLanczosFA, DensePSDExp) {
    run_dense_fa_test([](double x){ return std::exp(x); }, 50, 4, 25, 1e-6, 77);
}

// Both reorth modes agree on a well-conditioned diagonal matrix (κ=40).
// Tests that reorthogonalization doesn't introduce a bug, not that it helps
// (for that one would need κ >> 1 where vanilla loses orthogonality).
TEST_F(TestLanczosFA, ReorthVsVanilla) {
    using RNG = r123::Philox4x32;
    int64_t n = 40, s = 3, d = 15;
    auto state = RandBLAS::RNGState(99);

    std::vector<double> diag(n);
    std::iota(diag.begin(), diag.end(), 1.0);
    std::vector<double> A_mat = make_diag_matrix(diag);

    std::vector<double> B(n * s);
    RandBLAS::DenseDist DB(n, s);
    state = RandBLAS::fill_dense(DB, B.data(), state);

    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A_mat.data(), n, Layout::ColMajor);
    auto f_sqrt = [](double x){ return std::sqrt(x); };

    using LFA = RandLAPACK::LanczosFA<double, RNG>;
    std::vector<double> out_full(n * s, 0.0), out_van(n * s, 0.0);
    LFA lfa_full, lfa_van;
    lfa_full.reorth = 1;
    lfa_van.reorth  = 0;

    lfa_full.call(A_op, B.data(), n, s, f_sqrt, d, out_full.data());
    lfa_van.call( A_op, B.data(), n, s, f_sqrt, d, out_van.data());

    double diff = 0.0, norm = 0.0;
    for (int64_t i = 0; i < n * s; ++i) {
        double r = out_full[i] - out_van[i];
        diff += r * r;
        norm += out_full[i] * out_full[i];
    }
    double rel_diff = std::sqrt(diff / norm);
    printf("Full reorth vs vanilla relative diff: %e\n", rel_diff);
    ASSERT_LT(rel_diff, 1e-6);
}


// ---------------------------------------------------------------------------
// Hutchinson tests live here because Hutchinson is a building block of
// FunNystromPP alongside LanczosFA, not a standalone algorithm with its own file.

class TestHutchinson : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// M = diag(1,...,100): tr(M) = n*(n+1)/2 = 5050 exactly.
// s=500 Rademacher samples gives std/mean ≈ 0.5%; 5% tolerance is conservative.
// Uses a custom DiagonalOp to test the SymmetricLinearOperator interface independently
// of ExplicitSymLinOp.
TEST_F(TestHutchinson, DiagonalTrace) {
    using RNG = r123::Philox4x32;
    int64_t n = 100, s = 500;
    auto state = RandBLAS::RNGState(13);

    std::vector<double> d(n);
    std::iota(d.begin(), d.end(), 1.0);
    double true_trace = n * (n + 1.0) / 2.0;

    struct DiagonalOp {
        using scalar_t = double;
        const int64_t dim;
        const std::vector<double>& diag;
        void operator()([[maybe_unused]] Layout layout, int64_t n_vecs, double alpha,
                        double* const B, int64_t ldb, double beta, double* C, int64_t ldc) {
            for (int64_t j = 0; j < n_vecs; ++j)
                for (int64_t i = 0; i < dim; ++i) {
                    double val = alpha * diag[i] * B[j * ldb + i];
                    C[j * ldc + i] = (beta == 0.0) ? val : val + beta * C[j * ldc + i];
                }
        }
    };

    DiagonalOp M{n, d};
    RandLAPACK::Hutchinson<double, RNG> hutch;
    double est = hutch.call(M, s, state);

    double rel_err = std::abs(est - true_trace) / true_trace;
    printf("Hutchinson trace estimate: est=%e, true=%e, rel_err=%e\n", est, true_trace, rel_err);
    ASSERT_LT(rel_err, 0.05);
}
