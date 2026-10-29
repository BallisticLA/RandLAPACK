// Tests for RandLAPACK::restarted_pcg_ne (restarted PCG on the right-preconditioned normal
// equations) and its inner kernel RandLAPACK::pcg_inner: every termination, warm starts, the
// stable round residual, the inner_abs_tol and outer stagnation controls, the oracle and its
// in-round poll, timing attribution, and argument checks.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include "../lsq_test_helpers.hh"

#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <thread>
#include <vector>

using RandLAPACK::linops::DenseLinOp;
using RandLAPACK::PCGInnerTermination;
using Term = RandLAPACK::RestartedPCGNETermination;
using blas::Layout;
using lsq_test::consistent_problem;
using lsq_test::make_problem;
using lsq_test::gels_reference;
using lsq_test::qr_r_factor;
using lsq_test::rel_err;

template <typename T> using Controls = RandLAPACK::RestartedPCGNEControls<T>;
template <typename T> using Report   = RandLAPACK::RestartedPCGNEReport<T>;


class TestRestartedPCGNE : public ::testing::Test {
protected:
    template <typename T>
    static Controls<T> controls(T tol, int max_iters) {
        Controls<T> c;
        c.tol = tol;
        c.max_iters = max_iters;
        return c;
    }
};


// ---- Solves, budgets and floors --------------------------------------------

// Unpreconditioned restarted PCG (H = A^T A) reproduces the LAPACK solution.
TEST_F(TestRestartedPCGNE, unpreconditioned_matches_gels) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 41, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-12, 2000);
    ctl.round_drop = 1e-2;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(), ctl, h), Term::converged);
    EXPECT_GT(h.iters_done, 0);
    EXPECT_LE(h.final_relres, 1e-12);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}

// With R from QR(A), H = I: a few iterations in total, and the reference solution.
TEST_F(TestRestartedPCGNE, preconditioned_converges_fast) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 43, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    auto R = qr_r_factor(A.data(), m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                              controls<T>(1e-12, 2000), h), Term::converged);
    EXPECT_LE(h.iters_done, 6);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}

// A loose drop (1e-1) cannot reach 1e-12 in one round, so the run restarts several times
// against the true residual and still converges.
TEST_F(TestRestartedPCGNE, restarts_until_tolerance) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 47, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-12, 2000);
    ctl.round_drop = 1e-1;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(), ctl, h), Term::converged);
    EXPECT_GT(h.rounds_done, 1);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}

// Exhausting the total iteration budget, shared across rounds, ends the run.
TEST_F(TestRestartedPCGNE, reports_iteration_budget) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 53, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                              controls<T>(1e-15, 3), h), Term::iteration_budget);
    EXPECT_LE(h.iters_done, 3);
    EXPECT_GT(h.iters_done, 0);
}

// max_restarts counts rounds after the first (0 = one round, negative = unlimited). With a
// loose drop and max_restarts = 0 the run stops after one round with the tolerance unmet.
TEST_F(TestRestartedPCGNE, honors_max_restarts) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 59, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-12, 2000);
    ctl.round_drop = 1e-1;

    Report<T> h_free;
    std::vector<T> x_free(n, 0);
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_free.data(), ctl, h_free),
              Term::converged);
    ASSERT_GT(h_free.rounds_done, 1);

    ctl.max_restarts = 0;
    Report<T> h_one;
    std::vector<T> x_one(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_one.data(), ctl, h_one),
              Term::round_budget);
    EXPECT_EQ(h_one.rounds_done, 1);
    EXPECT_LT(h_one.iters_done, h_free.iters_done);
    EXPECT_GT(h_one.final_relres, 1e-12);
}

// Each round's inner CG exits once its residual flatlines instead of grinding out its cap, and
// the outer stagnation exit ends the run at the floor before the round budget.
TEST_F(TestRestartedPCGNE, stagnation_exits_early) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b;
    lsq_test::floor_problem(m, n, A, b);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-15, 2000);
    ctl.round_drop = 1e-30;
    ctl.max_restarts = 3;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(), ctl, h), Term::ls_floor);
    EXPECT_LT(h.rounds_done, 4);   // before the round budget
    EXPECT_GT(h.rounds_done, 1);   // the flatline takes two rounds to detect
    bool any_stagnated = false;
    for (auto s : h.cg_status) any_stagnated = any_stagnated || (s == PCGInnerTermination::stagnated);
    EXPECT_TRUE(any_stagnated);    // rounds ended on the window, not the per-round cap
    EXPECT_LT(h.iters_done, ctl.max_iters_per_round * h.rounds_done);
    EXPECT_TRUE(std::isfinite((double)h.final_relres));
}

// outer_stag_window <= 0 disables the floor exit: the same problem then runs every round up
// to max_restarts.
TEST_F(TestRestartedPCGNE, outer_stag_window_disabled_runs_to_round_budget) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b;
    lsq_test::floor_problem(m, n, A, b);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-15, 2000);
    ctl.round_drop = 1e-30;
    ctl.max_restarts = 3;
    ctl.outer_stag_window = 0;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(), ctl, h), Term::round_budget);
    EXPECT_EQ(h.rounds_done, 4);
    EXPECT_TRUE(std::isfinite((double)h.final_relres));
}

// b = 0: x = 0 with no iterations.
TEST_F(TestRestartedPCGNE, zero_rhs_returns_zero) {
    using T = double;
    int64_t m = 60, n = 6;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 211, A, b, x_true);
    std::fill(b.begin(), b.end(), (T)0);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    Report<T> h;
    std::vector<T> x(n, (T)3);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                              controls<T>(1e-12, 100), h), Term::converged);
    EXPECT_EQ(h.iters_done, 0);
    EXPECT_EQ(h.final_relres, (T)0);
    for (T v : x) EXPECT_EQ(v, (T)0);
}

// b orthogonal to range(A): g = A^T b = 0, so the normal-equation residual is zero while the
// true residual is not; that is the floor, reached with no iterations.
TEST_F(TestRestartedPCGNE, zero_normal_equation_residual_is_the_floor) {
    using T = double;
    int64_t m = 6, n = 2;
    std::vector<T> A(m * n, (T)0), b(m, (T)0);
    A[0] = 1; A[1 + m] = 1;
    b[4] = 2;
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    Report<T> h;
    std::vector<T> x(n, (T)5);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                              controls<T>(1e-12, 100), h), Term::ls_floor);
    EXPECT_EQ(h.iters_done, 0);
    EXPECT_EQ(h.rounds_done, 0);
    EXPECT_EQ(h.final_relres, (T)1);
    for (T v : x) EXPECT_EQ(v, (T)0);
}

// A tiny nonzero diagonal entry of R overflows the computed apply H p (trsv on the way in and
// out) to inf/NaN, so CG breaks down on its first direction: x is still the cold start.
TEST_F(TestRestartedPCGNE, reports_breakdown_on_unusable_R) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 151, A, b, x_true);
    auto R = qr_r_factor(A.data(), m, n);
    R[(n - 1) + (n - 1) * n] = (T)1e-300;
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    Report<T> h;
    std::vector<T> x(n, (T)999);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                              controls<T>(1e-10, 2000), h), Term::breakdown);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(x[i], (T)0) << "component " << i;
    EXPECT_NEAR(h.final_relres, (T)1, 1e-12);
}

// A NaN in b or in x0 never counts as converged: the residual is NaN, the inner CG breaks down
// on its first direction, and the run reports a breakdown.
TEST_F(TestRestartedPCGNE, nan_data_ends_in_breakdown) {
    using T = double;
    auto p = consistent_problem<T>(80, 10, 155, 156);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T nan = std::numeric_limits<T>::quiet_NaN();
    const auto ctl = controls<T>(1e-10, 100);
    Report<T> h;
    std::vector<T> x(p.n, 0);

    std::vector<T> b_nan(p.b);
    b_nan[3] = nan;
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, b_nan.data(), x.data(), ctl, h),
              Term::breakdown);
    EXPECT_EQ(h.rounds_done, 1);
    EXPECT_EQ(h.iters_done, 0);
    EXPECT_TRUE(std::isnan(h.final_relres));

    std::vector<T> x0(p.x_true);
    x0[p.n - 1] = nan;
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h,
                                              x0.data()), Term::breakdown);
    EXPECT_EQ(h.rounds_done, 1);
    EXPECT_EQ(h.iters_done, 0);
    EXPECT_TRUE(std::isnan(h.final_relres));
}

// ---- Warm starts and the round residual ------------------------------------

// A converged x0 costs no iterations, a perturbed x0 is refined back to the reference, and warm
// and cold starts reach the same solution.
TEST_F(TestRestartedPCGNE, warm_start_refines_an_existing_iterate) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 71, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    auto R = qr_r_factor(A.data(), m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-12, 2000);

    std::vector<T> x_cold(n, 0);
    Report<T> hc;
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x_cold.data(), ctl, hc),
              Term::converged);
    std::vector<T> x_warm(n, 0);
    Report<T> hw;
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x_warm.data(), ctl, hw,
                                              /*x0=*/x_cold.data()), Term::converged);
    EXPECT_EQ(hw.iters_done, 0);
    EXPECT_LT(rel_err(x_warm, x_ref), 1e-8);

    std::vector<T> x_bad(x_ref);
    for (int64_t i = 0; i < n; ++i) x_bad[i] *= (T)1.5;
    std::vector<T> x_fixed(n, 0);
    Report<T> hf;
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x_fixed.data(), ctl, hf,
                                              /*x0=*/x_bad.data()), Term::converged);
    EXPECT_GT(hf.iters_done, 0);
    EXPECT_LT(rel_err(x_fixed, x_ref), 1e-8);
    EXPECT_LT(rel_err(x_fixed, x_cold), 1e-8);
}

// Round residuals are recomputed as R^{-T}(A^T(b - A x)) rather than g - H z, whose
// cancellation floors the accuracy: a consistent ill-conditioned problem with a healthy
// preconditioner reaches a tight tolerance instead of stalling at eps * kappa.
TEST_F(TestRestartedPCGNE, reaches_tight_tol_on_ill_conditioned) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 67, A, b, x_true);
    for (int64_t j = 0; j < n; ++j) {
        T s = std::pow((T)10.0, (T)(-11.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    auto R = qr_r_factor(A.data(), m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-12, 2000);
    ctl.max_iters_per_round = 500;
    ctl.round_drop = 1e-2;
    ctl.max_restarts = 3;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(), ctl, h), Term::converged);
    EXPECT_LE(h.final_relres, 1e-12);
}

// Prolate regime: an augmented prolate Toeplitz matrix (numerically rank-deficient, tiny
// sqrt(lambda) identity rows, noisy data) converges to 1e-9 from a shifted Cholesky-QR-style
// factor. (This dense problem does not separate the two residual forms; it pins a floor no
// change may regress.)
TEST_F(TestRestartedPCGNE, converges_on_regularized_prolate) {
    using T = double;
    // T = 800 x 400 prolate (omega 0.14), A = [T; sqrt(lambda) I] with lambda = 1e-20,
    // rhs = [b; 0], b = T x_true + 1e-11 relative noise.
    int64_t mt = 800, n = 400, m = mt + n;
    const double omega = 0.14;
    auto prolate = [&](int64_t k) -> T {
        if (k == 0) return (T)(2.0 * omega);
        return (T)(std::sin(2.0 * std::numbers::pi * omega * (double)k) / (std::numbers::pi * (double)k));
    };
    std::vector<T> A(m * n, 0.0);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < mt; ++i)
            A[i + j * m] = prolate(i - j);
    const T sqrt_lambda = (T)1e-10;
    for (int64_t j = 0; j < n; ++j) A[(mt + j) + j * m] = sqrt_lambda;
    std::vector<T> x_true(n), b(m, 0.0);
    for (int64_t i = 0; i < n; ++i)
        x_true[i] = std::sin(0.7 * (double)i) / (1.0 + 0.01 * (double)i);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               mt, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    {
        std::vector<T> e(mt);
        for (int64_t i = 0; i < mt; ++i) e[i] = std::sin(3.1 * (double)i);
        T en = blas::nrm2(mt, e.data(), 1), bn = blas::nrm2(mt, b.data(), 1);
        for (int64_t i = 0; i < mt; ++i) b[i] += e[i] / en * (T)1e-11 * bn;
    }
    // Cholesky-QR-style R with an escalating diagonal shift; a weak R is enough.
    std::vector<T> G(n * n, 0.0), R(n * n);
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, m, (T)1.0, A.data(), m, (T)0.0, G.data(), n);
    T gnorm = 0; for (int64_t j = 0; j < n; ++j) gnorm = std::max(gnorm, std::abs(G[j + j * n]));
    T shift = (T)10 * std::numeric_limits<T>::epsilon() * gnorm;
    int info = 1;
    for (int t = 0; t < 40 && info != 0; ++t, shift *= (T)10) {
        std::copy(G.begin(), G.end(), R.begin());
        for (int64_t j = 0; j < n; ++j) R[j + j * n] += shift;
        info = (int)lapack::potrf(blas::Uplo::Upper, n, R.data(), n);
    }
    ASSERT_EQ(info, 0);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-9, 2000);
    ctl.max_iters_per_round = 500;
    ctl.round_drop = 1e-2;
    ctl.max_restarts = 3;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(), ctl, h), Term::converged);
    EXPECT_LE(h.final_relres, 1e-9);
}

// inner_abs_tol: a round stops at the absolute floor inner_abs_tol * ||g|| instead of the full
// round_drop factor. Round 1 starts at exactly ||g||, so with inner_abs_tol < round_drop it is
// unaffected; round 2 starts lower, where the floor binds. Two rounds isolate the mechanism.
TEST_F(TestRestartedPCGNE, inner_abs_tol_shortens_later_rounds) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 163, A, b, x_true);
    for (int64_t j = 0; j < n; ++j) {   // mild ill-conditioning: more than one round is needed
        T s = std::pow((T)10.0, (T)(-4.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-9, 5000);
    ctl.max_iters_per_round = 500;
    ctl.round_drop = 1e-2;
    ctl.max_restarts = 1;

    Report<T> h_base, h_guard;
    std::vector<T> x_base(n, 0), x_guard(n, 0);
    RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_base.data(), ctl, h_base);
    ASSERT_EQ(h_base.cg_iters.size(), (size_t)2);
    ctl.inner_abs_tol = (T)1e-3;
    RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_guard.data(), ctl, h_guard);
    ASSERT_EQ(h_guard.cg_iters.size(), (size_t)2);
    EXPECT_EQ(h_guard.cg_iters[0], h_base.cg_iters[0]);   // round 1 is unaffected
    EXPECT_EQ(h_guard.cg_status[0], h_base.cg_status[0]);
    EXPECT_NEAR(h_guard.ls_relres[0], h_base.ls_relres[0], 1e-12 * h_base.ls_relres[0]);
    EXPECT_EQ(h_guard.cg_status[1], PCGInnerTermination::converged);
    EXPECT_LT(h_guard.cg_iters[1], h_base.cg_iters[1]);
}

// Float, preconditioned, with the timing totals filled.
TEST_F(TestRestartedPCGNE, float_preconditioned_with_times) {
    using T = float;
    int64_t m = 150, n = 12;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 223, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    auto R = qr_r_factor(A.data(), m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    auto ctl = controls<T>(1e-5f, 500);
    ctl.round_drop = 1e-2f;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(), ctl, h), Term::converged);
    EXPECT_LT(rel_err(x, x_ref), 1e-4f);
    for (long t : h.times) EXPECT_GE(t, 0L);
}


// The normal-equation residual of a round's end is formed only when another round runs: a
// converged cold run applies A^T once for g, once per inner iteration and once per round
// after the first.
TEST_F(TestRestartedPCGNE, last_round_skips_the_adjoint_apply) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 515, 516, 517, (T)0.30);   // imperfect R: several rounds
    DenseLinOp<T> A_dense(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    struct CountingOp {
        using scalar_t = T;
        const int64_t n_rows, n_cols;
        DenseLinOp<T>& A;
        int adjoint_applies = 0;
        void operator()(Layout layout, blas::Op tA, blas::Op tB, int64_t mm, int64_t nn, int64_t kk,
                        T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc) {
            if (tA == blas::Op::Trans) ++adjoint_applies;
            A(layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
        }
    } J{p.m, p.n, A_dense};
    auto ctl = controls<T>(1e-10, 4000);
    ctl.outer_stag_window = 0;
    Report<T> h;
    std::vector<T> x(p.n, 0);
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h),
              Term::converged);
    ASSERT_GE(h.rounds_done, 2);
    EXPECT_EQ(J.adjoint_applies, 1 + h.iters_done + (h.rounds_done - 1));
}

// ---- The oracle ------------------------------------------------------------
// These oracles return the least-squares relative residual itself, so every recorded value can
// be checked against the solver's own ls_relres.

// The run stops as soon as the oracle is at or below be_tol, records the value for every round,
// and calls it exactly once per round.
TEST_F(TestRestartedPCGNE, oracle_stops_the_run) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 515, 516, 517, (T)0.30);   // imperfect R: several rounds
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    int calls = 0;
    auto ctl = controls<T>(0, 4000);
    ctl.outer_stag_window = 0;
    ctl.oracle = [&](const T*, const T* r, const T*) -> T { ++calls; return blas::nrm2(p.m, r, 1) / b_norm; };
    ctl.be_tol = (T)1e-6;
    Report<T> h;
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h),
              Term::oracle_met);
    EXPECT_GE(h.rounds_done, 1);
    ASSERT_EQ(h.be.size(), (size_t)h.rounds_done);
    EXPECT_EQ(calls, h.rounds_done);
    EXPECT_LE(h.be.back(), (T)1e-6);
    for (size_t k = 0; k + 1 < h.be.size(); ++k) EXPECT_GT(h.be[k], (T)1e-6);
    for (size_t k = 0; k < h.be.size(); ++k)
        EXPECT_NEAR(h.be[k], h.ls_relres[k], 1e-13 * std::max((T)1, h.ls_relres[k]));
    EXPECT_EQ(h.be_x0, (T)-1);   // cold start: no x0 evaluation
}

// At an oracle exit the returned x is bitwise the iterate the oracle last saw, iters_done is
// the sum of the per-round counts, final_relres is the last round's, and times[3] excludes the
// oracle's own time.
TEST_F(TestRestartedPCGNE, oracle_exit_returns_the_evaluated_iterate_and_consistent_counts) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 555, 556, 557, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    std::vector<T> x_seen(p.n, 0);
    auto ctl = controls<T>(0, 4000);
    ctl.outer_stag_window = 0;
    ctl.be_tol = (T)1e-6;
    ctl.oracle = [&](const T* x, const T* r, const T*) -> T {
        std::copy(x, x + p.n, x_seen.begin());
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // make t_be measurable
        return blas::nrm2(p.m, r, 1) / b_norm;
    };
    Report<T> h;
    std::vector<T> x(p.n, 0);
    auto t0 = std::chrono::steady_clock::now();
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h),
              Term::oracle_met);
    long wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0).count();
    for (int64_t i = 0; i < p.n; ++i) EXPECT_EQ(x[i], x_seen[i]) << "element " << i;
    int sum_iters = 0; for (int v : h.cg_iters) sum_iters += v;
    EXPECT_EQ(h.iters_done, sum_iters);
    ASSERT_FALSE(h.ls_relres.empty());
    EXPECT_EQ(h.final_relres, h.ls_relres.back());
    EXPECT_GE(h.t_be_us, 20000L * h.rounds_done);
    EXPECT_LT(h.times[3], wall_us - h.t_be_us / 2);
}

// An active oracle that is never met leaves the termination, counts, x and residual history
// identical to a run without one.
TEST_F(TestRestartedPCGNE, active_but_unmet_oracle_changes_nothing) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 565, 566, 567, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    int calls = 0;
    auto ctl = controls<T>(1e-10, 4000);
    Report<T> ha, hb;
    std::vector<T> xa(p.n, 0), xb(p.n, 0);
    auto sa = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), xa.data(), ctl, ha);
    ctl.oracle = [&](const T*, const T* r, const T*) -> T { ++calls; return blas::nrm2(p.m, r, 1) / b_norm; };
    ctl.be_tol = (T)0;   // active, never satisfied
    auto sb = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), xb.data(), ctl, hb);
    EXPECT_EQ(calls, hb.rounds_done);
    EXPECT_EQ(sa, sb);
    EXPECT_NE(sb, Term::oracle_met);
    EXPECT_EQ(ha.iters_done, hb.iters_done);
    EXPECT_EQ(ha.rounds_done, hb.rounds_done);
    for (int64_t i = 0; i < p.n; ++i) EXPECT_EQ(xa[i], xb[i]) << "element " << i;
    EXPECT_EQ(ha.ls_relres, hb.ls_relres);
    for (T v : hb.be) EXPECT_GT(v, (T)0);
}

// Meeting tol and the oracle in the same round reports converged, never oracle_met.
TEST_F(TestRestartedPCGNE, tolerance_outranks_the_oracle_in_the_same_round) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 575, 576, 577, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    const T shared_tol = (T)1e-6;
    auto ctl = controls<T>(shared_tol, 4000);
    ctl.outer_stag_window = 0;
    ctl.oracle = [&](const T*, const T* r, const T*) -> T { return blas::nrm2(p.m, r, 1) / b_norm; };
    ctl.be_tol = shared_tol;
    Report<T> h;
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h),
              Term::converged);
    ASSERT_FALSE(h.be.empty());
    EXPECT_LE(h.be.back(), shared_tol);   // the oracle was met too, and lost
}

// A breakdown in the same round outranks an oracle that always reports success.
TEST_F(TestRestartedPCGNE, breakdown_outranks_the_oracle_in_the_same_round) {
    using T = double;
    auto p = consistent_problem<T>(90, 12, 585, 586);
    std::vector<T> negA(p.A.size());
    for (size_t i = 0; i < p.A.size(); ++i) negA[i] = -p.A[i];
    lsq_test::TwoFacedOp<T> J(p.m, p.n, p.A.data(), negA.data());
    int calls = 0;
    auto ctl = controls<T>(0, 4000);
    ctl.outer_stag_window = 0;
    ctl.oracle = [&](const T*, const T*, const T*) -> T { ++calls; return (T)0; };
    ctl.be_tol = (T)1;
    Report<T> h;
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h),
              Term::breakdown);
    EXPECT_EQ(h.rounds_done, 1);
    EXPECT_EQ(calls, 1);
    ASSERT_EQ(h.cg_status.size(), (size_t)1);
    EXPECT_EQ(h.cg_status[0], PCGInnerTermination::breakdown);
    ASSERT_EQ(h.be.size(), (size_t)1);
    EXPECT_LE(h.be[0], (T)1);
}

// With be_tol < 0 the oracle is never called and the run is identical to one without it.
TEST_F(TestRestartedPCGNE, oracle_with_negative_be_tol_is_never_called) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 525, 526, 527, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    int calls = 0;
    auto ctl = controls<T>(1e-10, 4000);
    Report<T> ha, hb;
    std::vector<T> xa(p.n, 0), xb(p.n, 0);
    auto sa = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), xa.data(), ctl, ha);
    ctl.oracle = [&](const T*, const T*, const T*) -> T { ++calls; return (T)0; };
    ctl.be_tol = (T)-1;
    ctl.be_poll_every = 3;   // inactive with the oracle
    auto sb = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), xb.data(), ctl, hb);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(sa, sb);
    EXPECT_EQ(ha.iters_done, hb.iters_done);
    EXPECT_EQ(ha.rounds_done, hb.rounds_done);
    for (int64_t i = 0; i < p.n; ++i) EXPECT_EQ(xa[i], xb[i]) << "element " << i;
    ASSERT_EQ(hb.be.size(), (size_t)hb.rounds_done);
    for (T v : hb.be) EXPECT_EQ(v, (T)-1);
    for (int k : hb.polls) EXPECT_EQ(k, 0);
    EXPECT_EQ(hb.t_be_us, 0L);
    EXPECT_EQ(hb.t_poll_us, 0L);
}

// A warm start that already meets be_tol ends with oracle_met after zero rounds, and the x0
// evaluation is recorded.
TEST_F(TestRestartedPCGNE, oracle_met_by_the_warm_start) {
    using T = double;
    auto p = consistent_problem<T>(100, 15, 535, 536);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    auto ctl = controls<T>(0, 4000);
    ctl.outer_stag_window = 0;
    ctl.oracle = [&](const T*, const T* r, const T*) -> T { return blas::nrm2(p.m, r, 1) / b_norm; };
    ctl.be_tol = (T)1e-8;
    Report<T> h;
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h,
                                              /*x0=*/p.x_true.data()), Term::oracle_met);
    EXPECT_EQ(h.iters_done, 0);
    EXPECT_EQ(h.rounds_done, 0);
    EXPECT_TRUE(h.be.empty());
    EXPECT_GE(h.be_x0, (T)0);
    EXPECT_LE(h.be_x0, (T)1e-8);
    for (int64_t i = 0; i < p.n; ++i) EXPECT_NEAR(x[i], p.x_true[i], 1e-10);
}

// A warm start that meets both tol and be_tol ends as converged: the tolerance outranks the
// oracle at the x0 evaluation too.
TEST_F(TestRestartedPCGNE, tolerance_outranks_the_oracle_at_the_warm_start) {
    using T = double;
    auto p = consistent_problem<T>(100, 15, 537, 538);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    int calls = 0;
    auto ctl = controls<T>(1e-8, 4000);
    ctl.oracle = [&](const T*, const T*, const T*) -> T { ++calls; return (T)0; };
    ctl.be_tol = (T)1;
    Report<T> h;
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h,
                                              /*x0=*/p.x_true.data()), Term::converged);
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(h.be_x0, (T)0);
    EXPECT_EQ(h.rounds_done, 0);
    EXPECT_LE(h.final_relres, (T)1e-8);
}

// The oracle's arguments, checked independently with a nontrivial R at every call site (the
// warm start, round ends and in-round polls): r = b - A x and A^T r for the x it is shown.
TEST_F(TestRestartedPCGNE, oracle_arguments_are_x_residual_and_adjoint_residual) {
    using T = double;
    auto p = consistent_problem<T>(150, 20, 701, 702, 703, (T)0.40);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    std::vector<T> x0(p.x_true);
    for (int64_t i = 0; i < p.n; ++i) x0[i] *= (T)1.3;
    int calls = 0;
    T worst_r = 0, worst_atr = 0;
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    const T a_norm = blas::nrm2(p.m * p.n, p.A.data(), 1);
    auto ctl = controls<T>(0, 600);
    ctl.max_iters_per_round = 50;
    ctl.round_drop = (T)1e-8;
    ctl.outer_stag_window = 0;
    ctl.be_tol = (T)0;           // never met: every call site is exercised
    ctl.be_poll_every = 3;
    ctl.oracle = [&](const T* x, const T* r, const T* ATr) -> T {
        ++calls;
        std::vector<T> r_ref(p.b), atr_ref(p.n, 0);
        blas::gemv(Layout::ColMajor, blas::Op::NoTrans, p.m, p.n, (T)-1, p.A.data(), p.m, x, 1, (T)1, r_ref.data(), 1);
        blas::gemv(Layout::ColMajor, blas::Op::Trans, p.m, p.n, (T)1, p.A.data(), p.m, r_ref.data(), 1, (T)0, atr_ref.data(), 1);
        T dr = 0, da = 0;
        for (int64_t i = 0; i < p.m; ++i) dr = std::max(dr, std::abs(r[i] - r_ref[i]));
        for (int64_t i = 0; i < p.n; ++i) da = std::max(da, std::abs(ATr[i] - atr_ref[i]));
        worst_r   = std::max(worst_r,   dr / b_norm);              // rounding in b - A x is relative to ||b||
        worst_atr = std::max(worst_atr, da / (a_norm * b_norm));
        return (T)1;
    };
    Report<T> h;
    std::vector<T> x(p.n, 0);
    RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h, x0.data());
    int polls = 0; for (int k : h.polls) polls += k;
    EXPECT_GE(h.be_x0, (T)0);        // the warm start was evaluated
    EXPECT_GT(polls, 0);             // and so were in-round polls
    EXPECT_EQ(calls, 1 + polls + h.rounds_done);
    EXPECT_LT(worst_r, 1e-13);
    EXPECT_LT(worst_atr, 1e-13);
}


// ---- The in-round poll -----------------------------------------------------

// A weak R, a tiny per-round drop and a loose target: the first round runs far past the point
// where the residual meets be_tol. Polling every 5 iterations ends the run there: one round
// ending polled, an iteration count that is a multiple of 5 and no larger than the unpolled
// count, x equal to the iterate the firing poll evaluated (the round reuses that evaluation
// instead of calling the oracle again), and consistent counts.
TEST_F(TestRestartedPCGNE, poll_ends_the_run_mid_round) {
    using T = double;
    auto p = consistent_problem<T>(300, 60, 611, 612, 613, (T)0.60);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    auto run = [&](int poll_every, std::vector<T>& x, Report<T>& h, int& calls,
                   std::vector<std::vector<T>>& seen) {
        auto ctl = controls<T>(0, 4000);
        ctl.max_iters_per_round = 400;
        ctl.round_drop = (T)1e-12;
        ctl.outer_stag_window = 0;
        ctl.be_tol = (T)1e-6;
        ctl.be_poll_every = poll_every;
        ctl.oracle = [&](const T* xx, const T* r, const T*) -> T {
            ++calls; seen.emplace_back(xx, xx + p.n);
            return blas::nrm2(p.m, r, 1) / b_norm;
        };
        return RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h);
    };
    std::vector<T> xa(p.n, 0), xb(p.n, 0);
    std::vector<std::vector<T>> seen_a, seen_b;
    int ca = 0, cb = 0;
    Report<T> ha, hb;
    EXPECT_EQ(run(0, xa, ha, ca, seen_a), Term::oracle_met);
    EXPECT_EQ(run(5, xb, hb, cb, seen_b), Term::oracle_met);
    EXPECT_EQ(hb.rounds_done, 1);
    ASSERT_EQ(hb.cg_status.size(), (size_t)1);
    EXPECT_EQ(hb.cg_status[0], PCGInnerTermination::polled);
    EXPECT_EQ(hb.iters_done % 5, 0);
    EXPECT_LE(hb.iters_done, ha.iters_done);
    EXPECT_EQ(hb.polls[0], hb.iters_done / 5);   // one poll per 5 iterations; the last one fired
    EXPECT_EQ(cb, hb.polls[0]);   // the polled round's end reuses its last poll
    EXPECT_LE(hb.be.back(), (T)1e-6);
    EXPECT_LE(hb.final_relres, (T)1e-6);
    EXPECT_EQ(hb.final_relres, hb.ls_relres.back());
    // x is the iterate the firing poll evaluated.
    ASSERT_EQ(seen_b.size(), (size_t)cb);
    ASSERT_GE(cb, 1);
    for (int64_t i = 0; i < p.n; ++i)
        EXPECT_EQ(xb[i], seen_b[cb - 1][i]) << "poll iterate, element " << i;
    for (int k : ha.polls) EXPECT_EQ(k, 0);
    EXPECT_EQ(ha.t_poll_us, 0L);
}

// An active, never satisfied oracle polled every 5 iterations gives the same x, counts and
// per-round records as no polling; only the call count differs. The poll cadence counts
// iterations within each round, and a round that converges makes no poll at its last iteration.
TEST_F(TestRestartedPCGNE, active_but_unmet_poll_changes_nothing_but_the_call_count) {
    using T = double;
    auto p = consistent_problem<T>(300, 60, 621, 622, 623, (T)0.60);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    int calls = 0;
    auto run = [&](int poll_every, std::vector<T>& x, Report<T>& h) {
        auto ctl = controls<T>(1e-10, 4000);
        ctl.oracle = [&](const T*, const T*, const T*) -> T { ++calls; return (T)1; };
        ctl.be_tol = (T)0;
        ctl.be_poll_every = poll_every;
        return RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h);
    };
    std::vector<T> xa(p.n, 0), xb(p.n, 0);
    Report<T> ha, hb;
    auto sa = run(0, xa, ha);
    int calls_a = calls; calls = 0;
    auto sb = run(5, xb, hb);
    int calls_b = calls;
    EXPECT_EQ(sa, sb);
    EXPECT_NE(sb, Term::oracle_met);
    EXPECT_EQ(ha.iters_done, hb.iters_done);
    EXPECT_EQ(ha.rounds_done, hb.rounds_done);
    for (int64_t i = 0; i < p.n; ++i) EXPECT_EQ(xa[i], xb[i]) << "element " << i;
    ASSERT_EQ(ha.cg_iters.size(), hb.cg_iters.size());
    for (size_t k = 0; k < ha.cg_iters.size(); ++k) {
        EXPECT_EQ(ha.cg_iters[k], hb.cg_iters[k]);
        EXPECT_EQ(ha.cg_status[k], hb.cg_status[k]);
        EXPECT_EQ(ha.ls_relres[k], hb.ls_relres[k]);
        EXPECT_EQ(ha.be[k], hb.be[k]);
        EXPECT_EQ(ha.polls[k], 0);
        EXPECT_LE(hb.polls[k], hb.cg_iters[k] / 5);
        EXPECT_GE(hb.polls[k], (hb.cg_iters[k] - 1) / 5);
    }
    int polls_b = 0; for (int k : hb.polls) polls_b += k;
    EXPECT_GT(polls_b, 0);
    EXPECT_EQ(calls_a, ha.rounds_done);
    EXPECT_EQ(calls_b, polls_b + hb.rounds_done);
    // With the period equal to the first round's length, that round (which converged) makes no
    // poll at its last iteration.
    ASSERT_FALSE(ha.cg_iters.empty());
    ASSERT_EQ(ha.cg_status[0], PCGInnerTermination::converged);
    std::vector<T> xc(p.n, 0);
    Report<T> hc;
    run(ha.cg_iters[0], xc, hc);
    ASSERT_FALSE(hc.polls.empty());
    EXPECT_EQ(hc.polls[0], 0);
    EXPECT_EQ(hc.cg_iters[0], ha.cg_iters[0]);
}

// A sleeping oracle shows up in t_be_us (all calls) and t_poll_us (polls only) and is excluded
// from times[3]; the polls' applies are inside the kernel slice.
TEST_F(TestRestartedPCGNE, poll_time_is_recorded_and_excluded_from_the_total) {
    using T = double;
    auto p = consistent_problem<T>(300, 60, 631, 632, 633, (T)0.60);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    int calls = 0;
    auto ctl = controls<T>(0, 4000);
    ctl.max_iters_per_round = 400;
    ctl.round_drop = (T)1e-12;
    ctl.outer_stag_window = 0;
    ctl.be_tol = (T)1e-6;
    ctl.be_poll_every = 5;
    ctl.oracle = [&](const T*, const T* r, const T*) -> T {
        ++calls;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return blas::nrm2(p.m, r, 1) / b_norm;
    };
    Report<T> h;
    std::vector<T> x(p.n, 0);
    auto t0 = std::chrono::steady_clock::now();
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x.data(), ctl, h),
              Term::oracle_met);
    long wall_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    int polls = 0; for (int k : h.polls) polls += k;
    EXPECT_GT(polls, 0);
    EXPECT_GE(h.t_be_us, 5000L * calls);
    EXPECT_GE(h.t_poll_us, 5000L * polls);
    EXPECT_LT(h.times[3], wall_us - h.t_be_us / 2);
    EXPECT_LE(h.t_inner_us, h.times[3]);
    EXPECT_LE(h.t_fwd_inner_us + h.t_adj_inner_us + h.t_trsv_inner_us, h.t_inner_us);
}


// ---- Argument checks -------------------------------------------------------

TEST_F(TestRestartedPCGNE, invalid_arguments_throw) {
    using T = double;
    auto p = consistent_problem<T>(40, 5, 841, 842);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    std::vector<T> x(p.n, 0);
    Report<T> h;
    auto run = [&](const Controls<T>& c, int64_t m, int64_t n, int64_t ldr) {
        return RandLAPACK::restarted_pcg_ne<T>(J, m, n, p.R.data(), ldr, p.b.data(), x.data(), c, h);
    };
    const auto ok = controls<T>(1e-10, 100);
    EXPECT_NO_THROW(run(ok, p.m, p.n, p.n));
    auto c = ok; c.round_drop = 0;           EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.round_drop = 1;                EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.max_iters_per_round = 0;       EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.be_poll_every = -1;            EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.max_iters = 0;                 EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.tol = -1e-3;                   EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.tol = std::numeric_limits<T>::quiet_NaN();
                                             EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.tol = std::numeric_limits<T>::infinity();
                                             EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.stag_rel_improve = -0.1;       EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.stag_rel_improve = 1.5;        EXPECT_THROW(run(c, p.m, p.n, p.n), RandLAPACK::Error);
    c = ok; c.stag_rel_improve = 1;          EXPECT_NO_THROW(run(c, p.m, p.n, p.n));
    EXPECT_THROW(run(ok, p.m, p.n, p.n - 1), RandLAPACK::Error);   // ldr < n with R
    std::vector<T> R_sing(p.R);
    R_sing[2 + 2 * p.n] = (T)0;   // a zero diagonal entry
    EXPECT_THROW(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, R_sing.data(), p.n, p.b.data(), x.data(), ok, h),
                 RandLAPACK::Error);
    EXPECT_NO_THROW(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, nullptr, 0, p.b.data(), x.data(), ok, h));
    EXPECT_THROW(run(ok, p.m - 1, p.n, p.n), RandLAPACK::Error);
    EXPECT_THROW(run(ok, p.m, p.n - 1, p.n), RandLAPACK::Error);
    std::vector<T> W(5 * 8, (T)1);   // wide: m < n
    DenseLinOp<T> Jw(5, 8, W.data(), 5, Layout::ColMajor);
    std::vector<T> bw(5, (T)1), xw(8, 0);
    EXPECT_THROW(RandLAPACK::restarted_pcg_ne<T>(Jw, 5, 8, nullptr, 0, bw.data(), xw.data(), ok, h),
                 RandLAPACK::Error);
}

// R stored with ldr > n gives the tight copy's result up to rounding (the triangular solves
// may block differently for a different ldr).
TEST_F(TestRestartedPCGNE, padded_ldr_matches_tight_storage) {
    using T = double;
    auto p = consistent_problem<T>(120, 15, 851, 852, 853, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const int64_t ldr = p.n + 4;
    std::vector<T> R_pad(ldr * p.n, (T)77);
    for (int64_t j = 0; j < p.n; ++j)
        for (int64_t i = 0; i < p.n; ++i) R_pad[i + j * ldr] = p.R[i + j * p.n];
    auto ctl = controls<T>(1e-10, 2000);
    Report<T> ha, hb;
    std::vector<T> xa(p.n, 0), xb(p.n, 0);
    auto sa = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), xa.data(), ctl, ha);
    auto sb = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, R_pad.data(), ldr, p.b.data(), xb.data(), ctl, hb);
    EXPECT_EQ(sa, Term::converged);
    EXPECT_EQ(sa, sb);
    EXPECT_LT(rel_err(xb, xa), 1e-12);
}

// A warm start without a preconditioner: z = x0 directly.
TEST_F(TestRestartedPCGNE, warm_start_without_preconditioner) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 861, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x0(x_ref);
    for (int64_t i = 0; i < n; ++i) x0[i] *= (T)0.8;
    Report<T> h;
    std::vector<T> x(n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                              controls<T>(1e-12, 2000), h, x0.data()), Term::converged);
    EXPECT_GT(h.iters_done, 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}

// One report reused across two runs is reset on entry: the second run's report equals a fresh
// report of the same run.
TEST_F(TestRestartedPCGNE, report_is_reset_on_entry) {
    using T = double;
    auto p1 = consistent_problem<T>(120, 20, 871, 872, 873, (T)0.30);
    auto p2 = consistent_problem<T>(100, 10, 874, 875, 876, (T)0.20);
    DenseLinOp<T> J1(p1.m, p1.n, p1.A.data(), p1.m, Layout::ColMajor);
    DenseLinOp<T> J2(p2.m, p2.n, p2.A.data(), p2.m, Layout::ColMajor);
    auto ctl = controls<T>(1e-10, 2000);
    Report<T> reused, fresh;
    std::vector<T> x1(p1.n, 0), x2(p2.n, 0), x2f(p2.n, 0);
    RandLAPACK::restarted_pcg_ne<T>(J1, p1.m, p1.n, p1.R.data(), p1.n, p1.b.data(), x1.data(), ctl, reused);
    ASSERT_GT(reused.rounds_done, 0);
    RandLAPACK::restarted_pcg_ne<T>(J2, p2.m, p2.n, p2.R.data(), p2.n, p2.b.data(), x2.data(), ctl, reused);
    RandLAPACK::restarted_pcg_ne<T>(J2, p2.m, p2.n, p2.R.data(), p2.n, p2.b.data(), x2f.data(), ctl, fresh);
    EXPECT_EQ(reused.rounds_done, fresh.rounds_done);
    EXPECT_EQ(reused.iters_done, fresh.iters_done);
    EXPECT_EQ(reused.cg_iters, fresh.cg_iters);
    EXPECT_EQ(reused.cg_status, fresh.cg_status);
    EXPECT_EQ(reused.ls_relres, fresh.ls_relres);
    EXPECT_EQ(reused.be_x0, (T)-1);
    EXPECT_EQ(x2, x2f);
}

TEST_F(TestRestartedPCGNE, to_string_names_every_termination) {
    using RandLAPACK::to_string;
    EXPECT_STREQ(to_string(Term::converged), "converged");
    EXPECT_STREQ(to_string(Term::iteration_budget), "iteration_budget");
    EXPECT_STREQ(to_string(Term::breakdown), "breakdown");
    EXPECT_STREQ(to_string(Term::round_budget), "round_budget");
    EXPECT_STREQ(to_string(Term::ls_floor), "ls_floor");
    EXPECT_STREQ(to_string(Term::oracle_met), "oracle_met");
    EXPECT_STREQ(to_string(PCGInnerTermination::converged), "converged");
    EXPECT_STREQ(to_string(PCGInnerTermination::hit_cap), "hit_cap");
    EXPECT_STREQ(to_string(PCGInnerTermination::breakdown), "breakdown");
    EXPECT_STREQ(to_string(PCGInnerTermination::stagnated), "stagnated");
    EXPECT_STREQ(to_string(PCGInnerTermination::polled), "polled");
}


// ---- pcg_inner on its own --------------------------------------------------

// A zero right-hand side gives z = 0; a non-SPD operator breaks down at once and hands back the
// starting z = 0.
TEST_F(TestRestartedPCGNE, pcg_inner_zero_rhs_and_immediate_breakdown) {
    using T = double;
    const int64_t n = 4;
    std::vector<T> diag = {4, 3, 2, 1};
    auto apply_spd = [&](const T* v, T* out) { for (int64_t i = 0; i < n; ++i) out[i] = diag[i] * v[i]; };
    std::vector<T> c = {4, 3, 2, 1}, z(n), r(n), pp(n), Mp(n), zb(n);
    RandLAPACK::PCGInnerControls<T> ctl;
    ctl.tol = (T)1e-12;
    ctl.max_iters = 50;
    RandLAPACK::PCGInnerReport<T> rep;

    std::vector<T> c0(n, (T)0);
    std::fill(z.begin(), z.end(), (T)7);
    ASSERT_EQ(RandLAPACK::pcg_inner<T>(apply_spd, c0.data(), n, z.data(), r.data(), pp.data(),
                                       Mp.data(), zb.data(), ctl, rep), PCGInnerTermination::converged);
    for (T v : z) EXPECT_EQ(v, (T)0);
    EXPECT_EQ(rep.status, PCGInnerTermination::converged);

    auto apply_neg = [&](const T* v, T* out) { for (int64_t i = 0; i < n; ++i) out[i] = -v[i]; };
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(apply_neg, c.data(), n, z.data(), r.data(), pp.data(),
                                       Mp.data(), zb.data(), ctl, rep), PCGInnerTermination::breakdown);
    EXPECT_EQ(rep.status, PCGInnerTermination::breakdown);
    EXPECT_EQ(rep.iters, 0);
    for (T v : z) EXPECT_EQ(v, (T)0);
}

// The returned iterate on hit_cap and stagnated is the best one, not the last: on
// M = diag(1, 100), c = (1, 0.1) the first CG step (alpha = 0.505, r1 = (0.495, -4.95)) raises
// the residual to ||r1|| = 4.95 ||c||, so the best iterate is the start z = 0 while the last is
// z1 = 0.505 c.
TEST_F(TestRestartedPCGNE, pcg_inner_returns_the_best_iterate_not_the_last) {
    using T = double;
    const int64_t n = 2;
    auto apply_M = [&](const T* v, T* out) { out[0] = v[0]; out[1] = (T)100 * v[1]; };
    std::vector<T> c = {1, 0.1}, z(n), r(n), pp(n), Mp(n), zb(n);
    auto recomputed_relres = [&]() {
        T r0 = c[0] - z[0], r1 = c[1] - (T)100 * z[1];
        return std::hypot(r0, r1) / std::hypot(c[0], c[1]);
    };
    RandLAPACK::PCGInnerReport<T> rep;

    RandLAPACK::PCGInnerControls<T> cap;
    cap.tol = (T)1e-12; cap.max_iters = 1; cap.stag_window = 0;
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(apply_M, c.data(), n, z.data(), r.data(), pp.data(),
                                       Mp.data(), zb.data(), cap, rep), PCGInnerTermination::hit_cap);
    EXPECT_EQ(rep.status, PCGInnerTermination::hit_cap);
    EXPECT_EQ(rep.iters, 1);
    EXPECT_EQ(rep.best_iter, 0);
    EXPECT_EQ(z[0], (T)0); EXPECT_EQ(z[1], (T)0);
    EXPECT_DOUBLE_EQ(rep.relres, recomputed_relres());

    RandLAPACK::PCGInnerControls<T> stag;
    stag.tol = (T)1e-12; stag.max_iters = 10; stag.stag_window = 1; stag.stag_rel_improve = (T)1;
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(apply_M, c.data(), n, z.data(), r.data(), pp.data(),
                                       Mp.data(), zb.data(), stag, rep), PCGInnerTermination::stagnated);
    EXPECT_EQ(rep.status, PCGInnerTermination::stagnated);
    EXPECT_EQ(rep.iters, 1);
    EXPECT_EQ(z[0], (T)0); EXPECT_EQ(z[1], (T)0);
    EXPECT_DOUBLE_EQ(rep.relres, recomputed_relres());
}

// A breakdown after progress (the operator turns indefinite after three applies) returns the
// best iterate of the iterations completed before it.
TEST_F(TestRestartedPCGNE, pcg_inner_breakdown_after_progress_returns_the_best_iterate) {
    using T = double;
    const int64_t n = 6;
    std::vector<T> d = {1, 2, 3, 4, 5, 6}, c = {1, -1, 2, -2, 3, -3};
    int applies = 0;
    auto apply_M = [&](const T* v, T* out) {
        T s = (applies++ < 3) ? (T)1 : (T)-1;
        for (int64_t i = 0; i < n; ++i) out[i] = s * d[i] * v[i];
    };
    auto apply_spd = [&](const T* v, T* out) { for (int64_t i = 0; i < n; ++i) out[i] = d[i] * v[i]; };
    std::vector<T> z(n), r(n), pp(n), Mp(n), zb(n), z_ref(n);
    RandLAPACK::PCGInnerControls<T> ctl;
    ctl.tol = (T)1e-14; ctl.max_iters = 50; ctl.stag_window = 0;
    RandLAPACK::PCGInnerReport<T> rep, rep_ref;
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(apply_M, c.data(), n, z.data(), r.data(), pp.data(),
                                       Mp.data(), zb.data(), ctl, rep), PCGInnerTermination::breakdown);
    EXPECT_EQ(rep.status, PCGInnerTermination::breakdown);
    EXPECT_EQ(rep.iters, 3);
    EXPECT_GE(rep.best_iter, 1);
    // Reference: the same SPD solve capped at the three completed iterations.
    RandLAPACK::PCGInnerControls<T> ref = ctl;
    ref.max_iters = 3;
    RandLAPACK::pcg_inner<T>(apply_spd, c.data(), n, z_ref.data(), r.data(), pp.data(),
                             Mp.data(), zb.data(), ref, rep_ref);
    EXPECT_EQ(rep.best_iter, rep_ref.best_iter);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(z[i], z_ref[i]) << "element " << i;
}

// The poll hook ends the solve at the polled iteration and leaves z at the current iterate (not
// the best-iterate snapshot), bit for bit what the hook saw.
TEST_F(TestRestartedPCGNE, pcg_inner_poll_ends_the_solve_with_the_current_iterate) {
    using T = double;
    const int64_t n = 40;
    std::vector<T> B(n * n), M(n * n, 0), c(n);
    lsq_test::fill_random(B, 601);
    lsq_test::fill_random(c, 602);
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans, n, n, (T)1, B.data(), n, (T)0, M.data(), n);
    for (int64_t j = 0; j < n; ++j) {   // M = B^T B + 1e-3 I, full symmetric storage
        M[j + j * n] += (T)1e-3;
        for (int64_t i = j + 1; i < n; ++i) M[i + j * n] = M[j + i * n];
    }
    auto apply_M = [&](const T* v, T* out) {
        blas::gemv(Layout::ColMajor, blas::Op::NoTrans, n, n, (T)1, M.data(), n, v, 1, (T)0, out, 1);
    };
    std::vector<T> z(n), r(n), pp(n), Mp(n), zb(n), snapshot;
    std::vector<int> polled_at;
    RandLAPACK::PCGInnerControls<T> ctl;
    ctl.tol = (T)1e-12; ctl.max_iters = 500; ctl.stag_window = 0;
    ctl.poll_every = 3;
    ctl.poll = [&](const T* zc, int it) -> bool {
        polled_at.push_back(it);
        if (it == 6) { snapshot.assign(zc, zc + n); return true; }
        return false;
    };
    RandLAPACK::PCGInnerReport<T> rep;
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(apply_M, c.data(), n, z.data(), r.data(), pp.data(),
                                       Mp.data(), zb.data(), ctl, rep), PCGInnerTermination::polled);
    EXPECT_EQ(rep.status, PCGInnerTermination::polled);
    EXPECT_EQ(rep.iters, 6);
    ASSERT_EQ(polled_at, (std::vector<int>{3, 6}));
    ASSERT_EQ(snapshot.size(), (size_t)n);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(z[i], snapshot[i]) << "element " << i;
    EXPECT_GT(rep.relres, (T)1e-12);
}

// A poll that always declines leaves the solve identical to one without a hook, and it is never
// consulted on the iteration at which the solve converges (convergence is tested first).
TEST_F(TestRestartedPCGNE, pcg_inner_poll_that_never_fires_changes_nothing) {
    using T = double;
    const int64_t n = 40;
    std::vector<T> B(n * n), M(n * n, 0), c(n);
    lsq_test::fill_random(B, 603);
    lsq_test::fill_random(c, 604);
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans, n, n, (T)1, B.data(), n, (T)0, M.data(), n);
    for (int64_t j = 0; j < n; ++j) {
        M[j + j * n] += (T)1e-3;
        for (int64_t i = j + 1; i < n; ++i) M[i + j * n] = M[j + i * n];
    }
    auto apply_M = [&](const T* v, T* out) {
        blas::gemv(Layout::ColMajor, blas::Op::NoTrans, n, n, (T)1, M.data(), n, v, 1, (T)0, out, 1);
    };
    auto solve = [&](int poll_every, bool answer, int& calls, RandLAPACK::PCGInnerReport<T>& rep, std::vector<T>& z) {
        std::vector<T> r(n), pp(n), Mp(n), zb(n);
        RandLAPACK::PCGInnerControls<T> ctl;
        ctl.tol = (T)1e-10; ctl.max_iters = 500; ctl.stag_window = 0;
        if (poll_every > 0) {
            ctl.poll_every = poll_every;
            ctl.poll = [&](const T*, int) -> bool { ++calls; return answer; };
        }
        return RandLAPACK::pcg_inner<T>(apply_M, c.data(), n, z.data(), r.data(), pp.data(),
                                        Mp.data(), zb.data(), ctl, rep);
    };
    int ca = 0, cb = 0;
    RandLAPACK::PCGInnerReport<T> ra, rb;
    std::vector<T> za(n), zbv(n);
    EXPECT_EQ(solve(0, false, ca, ra, za), solve(3, false, cb, rb, zbv));
    EXPECT_EQ(ra.status, PCGInnerTermination::converged);
    EXPECT_EQ(rb.status, ra.status);
    EXPECT_EQ(rb.iters, ra.iters);
    EXPECT_EQ(rb.relres, ra.relres);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(za[i], zbv[i]) << "element " << i;
    EXPECT_EQ(cb, (ra.iters - 1) / 3);   // multiples of 3 strictly below the converging iteration
    // A poll due exactly on the converging iteration is never made, even one that would fire.
    int cc = 0;
    RandLAPACK::PCGInnerReport<T> rc;
    std::vector<T> zc(n);
    solve(ra.iters, true, cc, rc, zc);
    EXPECT_EQ(cc, 0);
    EXPECT_EQ(rc.status, PCGInnerTermination::converged);
    EXPECT_EQ(rc.iters, ra.iters);
}

// The non-finite guards. A finite positive p^T M p with an overflowing step: M = 1e-310 I gives
// p^T M p of about 1e-310 ||p||^2, so alpha = rs / pMp is about 1e310 = inf at any scale of c. An
// overflowing residual update: M = diag(1e300, 1e-100), c = (1e-160, 1) give p^T M p of about
// 1e-20 and a finite alpha of about 1e20, but the new residual is about -1e160 in its first
// entry, so ||r||^2 overflows. Both end as a breakdown before any update is kept, returning the
// start z = 0 with the starting report.
TEST_F(TestRestartedPCGNE, pcg_inner_non_finite_step_or_residual_is_a_breakdown) {
    using T = double;
    const int64_t n = 2;
    std::vector<T> z(n), r(n), pp(n), Mp(n), zb(n);
    RandLAPACK::PCGInnerControls<T> ctl;
    ctl.tol = (T)1e-12; ctl.max_iters = 10;
    RandLAPACK::PCGInnerReport<T> rep;
    auto check_start_report = [&]() {
        EXPECT_EQ(rep.status, PCGInnerTermination::breakdown);
        EXPECT_EQ(rep.iters, 0);
        EXPECT_EQ(rep.best_iter, 0);
        EXPECT_EQ(rep.best_relres, (T)1);
        EXPECT_EQ(rep.relres, (T)1);
        for (T v : z) EXPECT_EQ(v, (T)0);
    };

    std::vector<T> c1 = {(T)1e150, (T)1e150};
    auto tiny = [&](const T* v, T* out) { for (int64_t i = 0; i < n; ++i) out[i] = (T)1e-310 * v[i]; };
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(tiny, c1.data(), n, z.data(), r.data(), pp.data(), Mp.data(),
                                       zb.data(), ctl, rep), PCGInnerTermination::breakdown);
    check_start_report();

    std::vector<T> c2 = {(T)1e-160, (T)1};
    auto skewed = [&](const T* v, T* out) { out[0] = (T)1e300 * v[0]; out[1] = (T)1e-100 * v[1]; };
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(skewed, c2.data(), n, z.data(), r.data(), pp.data(), Mp.data(),
                                       zb.data(), ctl, rep), PCGInnerTermination::breakdown);
    check_start_report();
}

// The curvature guard on its own: for the scalar M = 1e308 and c = 1.5 (of unit scale, so not
// rescaled), M c = 1.5e308 and c^T c = 2.25 are finite but p^T M p = 2.25e308 overflows.
// Unguarded, alpha = c^T c / inf = 0 and CG would idle to its cap; guarded, it breaks down
// before any update.
TEST_F(TestRestartedPCGNE, pcg_inner_overflowing_curvature_is_a_breakdown) {
    using T = double;
    std::vector<T> c = {(T)1.5}, z(1, (T)7), r(1), pp(1), Mp(1), zb(1);
    auto apply_M = [](const T* v, T* out) { out[0] = (T)1e308 * v[0]; };
    RandLAPACK::PCGInnerControls<T> ctl;
    ctl.tol = (T)1e-12; ctl.max_iters = 10;
    RandLAPACK::PCGInnerReport<T> rep;
    EXPECT_EQ(RandLAPACK::pcg_inner<T>(apply_M, c.data(), 1, z.data(), r.data(), pp.data(), Mp.data(),
                                       zb.data(), ctl, rep), PCGInnerTermination::breakdown);
    EXPECT_EQ(rep.status, PCGInnerTermination::breakdown);
    EXPECT_EQ(rep.iters, 0);
    EXPECT_EQ(rep.best_iter, 0);
    EXPECT_EQ(rep.best_relres, (T)1);
    EXPECT_EQ(rep.relres, (T)1);
    EXPECT_EQ(z[0], (T)0);
}

// pcg_inner runs on c scaled by a power of two: 2^k c gives 2^k z bit for bit, with the same
// iterations, and the poll sees the unscaled iterate. In float, k = -70 makes the
// first residual's squared norm underflow (unscaled CG would report converged after one
// iteration at a relres of about 5e-5) and k = 70 makes c^T c overflow (an immediate breakdown).
TEST_F(TestRestartedPCGNE, pcg_inner_is_exact_under_power_of_two_scaling_float) {
    using T = float;
    const int64_t n = 2;
    auto apply_M = [](const T* v, T* out) { out[0] = v[0]; out[1] = (T)1.0001 * v[1]; };
    struct Run { PCGInnerTermination term; RandLAPACK::PCGInnerReport<T> rep; std::vector<T> z, z_poll; };
    auto solve = [&](int k) {
        Run run;
        std::vector<T> c = {std::ldexp((T)1, k), std::ldexp((T)1, k)}, r(n), p(n), Mp(n), zb(n);
        run.z.assign(n, (T)0);
        RandLAPACK::PCGInnerControls<T> ictl;
        ictl.tol = (T)1e-6;
        ictl.max_iters = 10;
        ictl.poll_every = 1;
        ictl.poll = [&](const T* z, int it) {
            if (it == 1) run.z_poll.assign(z, z + n);
            return false;
        };
        run.term = RandLAPACK::pcg_inner<T>(apply_M, c.data(), n, run.z.data(), r.data(), p.data(),
                                            Mp.data(), zb.data(), ictl, run.rep);
        return run;
    };
    const Run base = solve(0);
    ASSERT_EQ(base.term, PCGInnerTermination::converged);
    ASSERT_EQ(base.rep.iters, 2);
    ASSERT_EQ(base.z_poll.size(), (size_t)n);
    for (int k : {-70, 70}) {
        const Run run = solve(k);
        EXPECT_EQ(run.term, base.term) << "k=" << k;
        EXPECT_EQ(run.rep.iters, base.rep.iters) << "k=" << k;
        // relres divides by ||c||, which a BLAS may round differently at another scale.
        EXPECT_FLOAT_EQ(run.rep.relres, base.rep.relres) << "k=" << k;
        EXPECT_FLOAT_EQ(run.rep.best_relres, base.rep.best_relres) << "k=" << k;
        ASSERT_EQ(run.z_poll.size(), (size_t)n) << "k=" << k;
        for (int64_t i = 0; i < n; ++i) {
            EXPECT_EQ(run.z[i], std::ldexp(base.z[i], k)) << "k=" << k << ", i=" << i;
            EXPECT_EQ(run.z_poll[i], std::ldexp(base.z_poll[i], k)) << "poll, k=" << k << ", i=" << i;
        }
    }
}

TEST_F(TestRestartedPCGNE, pcg_inner_requires_max_iters) {
    using T = double;
    std::vector<T> c = {1, 2}, z(2), r(2), pp(2), Mp(2), zb(2);
    auto apply = [&](const T* v, T* out) { out[0] = v[0]; out[1] = v[1]; };
    RandLAPACK::PCGInnerControls<T> ctl;   // max_iters defaults to 0
    RandLAPACK::PCGInnerReport<T> rep;
    EXPECT_THROW(RandLAPACK::pcg_inner<T>(apply, c.data(), 2, z.data(), r.data(), pp.data(), Mp.data(),
                                          zb.data(), ctl, rep), RandLAPACK::Error);
}
