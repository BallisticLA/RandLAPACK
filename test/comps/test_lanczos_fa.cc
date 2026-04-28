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

    // Build a full symmetric diagonal matrix from a vector of diagonal entries.
    // Returns column-major n×n storage with all off-diagonals zero.
    static std::vector<double> make_diag_matrix(const std::vector<double>& diag) {
        int64_t n = diag.size();
        std::vector<double> A(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            A[i + i * n] = diag[i];
        return A;
    }

    // Direct reference: f(A)*B for diagonal A = diag(d).
    // Returns n×s result where result[:,j] = d .* B[:,j] component-wise via f.
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
};


// LanczosFA on a diagonal matrix with f=sqrt.
// A = diag(1,...,50), d=20 Lanczos steps approximate f(A)B from a 20-dimensional
// Krylov subspace; good but not machine-precision accuracy (d < n = 50).
TEST_F(TestLanczosFA, DiagonalSqrt) {
    using RNG = r123::Philox4x32;
    int64_t n = 50, s = 5, d = 20;
    auto state = RandBLAS::RNGState(42);

    // Diagonal entries: 1, 2, ..., n
    std::vector<double> diag(n);
    std::iota(diag.begin(), diag.end(), 1.0);
    std::vector<double> A_mat = make_diag_matrix(diag);

    // Random B (n×s)
    std::vector<double> B(n * s);
    RandBLAS::DenseDist DB(n, s);
    state = RandBLAS::fill_dense(DB, B.data(), state);

    // LanczosFA output
    std::vector<double> out(n * s, 0.0);
    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A_mat.data(), n, Layout::ColMajor);

    using LFA = RandLAPACK::LanczosFA<double, RNG>;
    LFA lfa;
    lfa.reorth = -1;  // full reorthogonalization
    auto f_sqrt = [](double x){ return std::sqrt(x); };
    lfa.call(A_op, B.data(), n, s, f_sqrt, d, out.data());

    // Reference
    auto ref = diag_fa_ref(diag, B, n, s, f_sqrt);

    // Relative error per column
    double err = 0.0, ref_norm = 0.0;
    for (int64_t i = 0; i < n * s; ++i) {
        double r = out[i] - ref[i];
        err      += r * r;
        ref_norm += ref[i] * ref[i];
    }
    double rel_err = std::sqrt(err / ref_norm);
    printf("LanczosFA diagonal sqrt: relative error = %e\n", rel_err);
    ASSERT_LT(rel_err, 1e-4);
}


// LanczosFA on a diagonal matrix with f=log.
TEST_F(TestLanczosFA, DiagonalLog) {
    using RNG = r123::Philox4x32;
    int64_t n = 50, s = 4, d = 20;
    auto state = RandBLAS::RNGState(7);

    // Diagonal entries: 1, 2, ..., n (all positive, so log is well-defined)
    std::vector<double> diag(n);
    std::iota(diag.begin(), diag.end(), 1.0);
    std::vector<double> A_mat = make_diag_matrix(diag);

    std::vector<double> B(n * s);
    RandBLAS::DenseDist DB(n, s);
    state = RandBLAS::fill_dense(DB, B.data(), state);

    std::vector<double> out(n * s, 0.0);
    linops::ExplicitSymLinOp<double> A_op(n, blas::Uplo::Upper, A_mat.data(), n, Layout::ColMajor);

    using LFA = RandLAPACK::LanczosFA<double, RNG>;
    LFA lfa;
    auto f_log = [](double x){ return std::log(x); };
    lfa.call(A_op, B.data(), n, s, f_log, d, out.data());

    auto ref = diag_fa_ref(diag, B, n, s, f_log);

    double err = 0.0, ref_norm = 0.0;
    for (int64_t i = 0; i < n * s; ++i) {
        double r = out[i] - ref[i];
        err      += r * r;
        ref_norm += ref[i] * ref[i];
    }
    double rel_err = std::sqrt(err / ref_norm);
    printf("LanczosFA diagonal log: relative error = %e\n", rel_err);
    ASSERT_LT(rel_err, 1e-3);
}


// Hutchinson standalone: estimate tr(M) for an explicit M.
// Use M = diag(1, 2, ..., n) so tr(M) = n*(n+1)/2 is known exactly.
TEST_F(TestLanczosFA, HutchinsonStandalone) {
    using RNG = r123::Philox4x32;
    int64_t n = 100, s = 500;
    auto state = RandBLAS::RNGState(13);

    std::vector<double> d(n);
    std::iota(d.begin(), d.end(), 1.0);
    double true_trace = n * (n + 1.0) / 2.0;

    // apply_M: multiply by diagonal matrix (element-wise per column)
    auto apply_M = [&](const double* Omega, double* Z, int64_t n_, int64_t s_) {
        for (int64_t j = 0; j < s_; ++j)
            for (int64_t i = 0; i < n_; ++i)
                Z[j * n_ + i] = d[i] * Omega[j * n_ + i];
    };

    using Hutch = RandLAPACK::Hutchinson<double, RNG>;
    Hutch hutch;
    double est = hutch.call(apply_M, n, s, state);

    double rel_err = std::abs(est - true_trace) / true_trace;
    printf("Hutchinson trace estimate: est=%e, true=%e, rel_err=%e\n", est, true_trace, rel_err);
    // With 500 Rademacher samples, variance should be small enough for 5% tolerance
    ASSERT_LT(rel_err, 0.05);
}


// Vanilla (no reorth) vs full reorth on a well-conditioned diagonal matrix.
// Both should give the same answer — this checks the reorth code path doesn't break things.
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
    lfa_full.reorth = -1;  // full
    lfa_van.reorth  =  0;  // vanilla

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
