// Unit tests for RandLAPACK::IterRefineLSQ — iterative-refinement LSQ
// using R as a right preconditioner.
//
// Strategy: wrap a small dense tall A as a DenseLinOp, build R from
// lapack::geqrf on a copy of A, then verify the IR-LSQ solution matches
// the closed-form lapack::gels reference across well-conditioned,
// residualful, and imperfect-R cases.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <random>
#include <vector>


using RandLAPACK::IterRefineLSQ;
using RandLAPACK::linops::DenseLinOp;
using blas::Layout;


template <typename T>
static void fill_random(std::vector<T>& v, uint32_t seed, T scale = 1.0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    for (auto& x : v) x = scale * dist(rng);
}


// Build R from QR of A: destructive geqrf on a copy, then take the upper-
// triangular n x n factor via lacpy(Upper) + laset(Lower) (raw pointers).
template <typename T>
static void build_R_from_A(const T* A, int64_t m, int64_t n, T* R, int64_t ldr) {
    T* A_copy = new T[m * n];
    std::copy(A, A + m * n, A_copy);
    T* tau = new T[n];
    lapack::geqrf(m, n, A_copy, m, tau);
    lapack::lacpy(lapack::MatrixType::Upper, n, n, A_copy, m, R, ldr);
    if (n > 1)
        lapack::laset(lapack::MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R + 1, ldr);
    delete[] A_copy;
    delete[] tau;
}


class TestIterRefineLSQ : public ::testing::Test {};


// Well-conditioned synthetic problem; exact R from QR(A). M should be
// numerically identity, so CG converges in ~1 iteration.
TEST_F(TestIterRefineLSQ, dense_well_conditioned) {
    using T = double;
    int64_t m = 80, n = 12;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 42);
    fill_random(x_true, 99);

    // b = A * x_true (exact RHS, in the column space of A).
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    // Reference: gels destroys both A and b.
    std::vector<T> A_ref(A.begin(), A.end()), b_ref(b.begin(), b.end());
    lapack::gels(blas::Op::NoTrans, m, n, 1, A_ref.data(), m, b_ref.data(), m);
    std::vector<T> x_ref(b_ref.begin(), b_ref.begin() + n);

    // Build R from QR(A) — perfect preconditioner.
    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);

    // Solve via IR-LSQ.
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*tol=*/1e-12, /*max_inner=*/50, /*n_steps=*/2);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    // x_ir should match x_ref (within accumulated rounding error).
    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-10);

    // CG should converge fast with a perfect preconditioner.
    EXPECT_LE(ir.inner_iters_per_step.front(), 5);
}


// LS with a real residual (b not exactly in range(A)).
TEST_F(TestIterRefineLSQ, dense_with_residual) {
    using T = double;
    int64_t m = 100, n = 7;

    std::vector<T> A(m * n), b(m), noise(m), x_true(n);
    fill_random(A, 7);
    fill_random(x_true, 17);
    fill_random(noise, 27, 0.05);

    // b = A * x_true + noise
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    for (int64_t i = 0; i < m; ++i) b[i] += noise[i];

    std::vector<T> A_ref(A.begin(), A.end()), b_ref(b.begin(), b.end());
    lapack::gels(blas::Op::NoTrans, m, n, 1, A_ref.data(), m, b_ref.data(), m);
    std::vector<T> x_ref(b_ref.begin(), b_ref.begin() + n);

    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 50, 2);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-10);
}


// Imperfect R: from QR of a slightly perturbed A. CG should take more iters
// but the final solution must still match gels.
TEST_F(TestIterRefineLSQ, imperfect_preconditioner) {
    using T = double;
    int64_t m = 60, n = 10;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 3);
    fill_random(x_true, 13);

    // b in col space (zero residual).
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    std::vector<T> A_ref(A.begin(), A.end()), b_ref(b.begin(), b.end());
    lapack::gels(blas::Op::NoTrans, m, n, 1, A_ref.data(), m, b_ref.data(), m);
    std::vector<T> x_ref(b_ref.begin(), b_ref.begin() + n);

    // Perturbed A for R-construction (simulates a sketch-based R).
    std::vector<T> A_pert = A, perturb(m * n);
    fill_random(perturb, 99, 0.1);
    for (size_t i = 0; i < A_pert.size(); ++i) A_pert[i] += perturb[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 100, 2);
    std::vector<T> x_ir(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n);
    EXPECT_EQ(status, 0);

    T diff_norm = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = x_ir[i] - x_ref[i];
        diff_norm += d * d;
    }
    diff_norm = std::sqrt(diff_norm);
    T xref_norm = blas::nrm2(n, x_ref.data(), 1);
    EXPECT_LT(diff_norm / xref_norm, 1e-9);
    // Imperfect R: CG should take more than 1 iter but well under the cap.
    EXPECT_LT(ir.inner_iters_per_step.front(), 100);
}


// A capped, non-converged inner solve must be DISTINGUISHABLE from a converged one.
//
// Before the 2026-07-27 instrumentation both looked identical to the caller: inner_cg
// returned 0 whether it converged or exhausted its budget, so a benchmark CSV could not
// tell "converged in 6 iterations" from "gave up at the cap". That ambiguity is exactly
// what made the App-1 iteration-count complaint hard to diagnose.
//
// Here the cap is set absurdly low (2) against a tolerance that cannot be met that fast,
// forcing the HitCap path, and then the same problem is solved with a generous budget to
// confirm the Converged path reports differently.
TEST_F(TestIterRefineLSQ, capped_solve_is_reported) {
    using T = double;
    int64_t m = 120, n = 20;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 4242);
    fill_random(x_true, 777);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    // Deliberately weak preconditioner: R from a heavily perturbed A, so CG needs
    // several iterations and cannot satisfy a tight tolerance in only two.
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 31337, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    // --- capped run: 2 inner iterations against a 1e-14 tolerance ---
    {
        IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/2, /*n_steps=*/2);
        // This test pins SINGLE-attempt mechanics (exact cap count); the default
        // single restart (2026-08-05) would rerun and double the reported iters.
        // Restart semantics get their own test below.
        ir.inner_restarts = 0;
        std::vector<T> x(n, 0);
        int status = ir.call(J, R.data(), n, b.data(), m, x.data(), n);
        EXPECT_EQ(status, 0);   // capping is still not an error return ...
        ASSERT_EQ(ir.inner_status_per_step.size(), 2u);
        // ... but it is now visible.
        EXPECT_EQ(ir.inner_status_per_step.front(),
                  static_cast<int>(RandLAPACK::InnerCGStatus::HitCap));
        EXPECT_EQ(ir.inner_iters_per_step.front(), 2);
        // Achieved residual must be worse than the tolerance it was asked for.
        EXPECT_GT(ir.inner_relres_per_step.front(), 1e-14);
        // Still descending when the cap hit: best residual is at (or near) the last iter.
        EXPECT_EQ(ir.inner_best_iter_per_step.front(), 2);
    }

    // --- generous run: same problem, enough budget to converge ---
    {
        IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/200, /*n_steps=*/2);
        std::vector<T> x(n, 0);
        int status = ir.call(J, R.data(), n, b.data(), m, x.data(), n);
        EXPECT_EQ(status, 0);
        ASSERT_EQ(ir.inner_status_per_step.size(), 2u);
        EXPECT_EQ(ir.inner_status_per_step.front(),
                  static_cast<int>(RandLAPACK::InnerCGStatus::Converged));
        EXPECT_LT(ir.inner_iters_per_step.front(), 200);
        EXPECT_LE(ir.inner_relres_per_step.front(), 1e-14);
    }
}

// Stagnation exit (added 2026-07-29). The ISAAC diagnostic showed an inner CG reaching its
// residual floor at iteration 17 of a 200-iteration step and then grinding out the
// remaining ~183 with a BIT-IDENTICAL best residual, while the outer solution got 11x worse
// when the budget was raised 10x. The driver now detects the flatline, stops, and returns
// the best iterate rather than the last.
//
// HOW THIS IS PROVOKED, and one thing that does NOT work. The obvious trick -- ask for an
// unreachable tolerance via inner_tol = 0 -- fails: on a small well-conditioned system CG
// drives the recursive residual to EXACTLY 0.0 (here by iteration 11), and `r_norm <= 0` is
// then true, so the solve reports Converged. inner_tol = 0 is reachable, not unreachable.
//
// Instead the window logic is driven directly: inner_stag_rel_improve = 1 demands an
// impossible 100% residual drop, so no iteration ever counts as progress and the exit must
// fire at exactly inner_stag_window iterations. That tests the mechanism (window, status,
// best-residual reporting) without needing to manufacture a pathological matrix.
TEST_F(TestIterRefineLSQ, stagnating_solve_exits_early_with_best_iterate) {
    using T = double;
    int64_t m = 120, n = 20;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 4242);
    fill_random(x_true, 777);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    // IMPERFECT preconditioner on purpose. build_R_from_A on the unperturbed A gives the
    // exact Cholesky factor, so M = R^-T A^T A R^-1 = I and CG converges in ONE iteration --
    // which leaves nothing for the contrast case below to measure.
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 31337, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const int kCap    = 400;
    const int kWindow = 3;

    // --- stagnation fires: impossible improvement threshold, small window ---
    std::vector<T> x_stag(n, 0);
    {
        IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/kCap, /*n_steps=*/2);
        ir.inner_stag_window     = kWindow;
        ir.inner_stag_rel_improve = 1.0;      // no drop can ever qualify as progress
        // Single-attempt mechanics under test (exact window count); the default
        // restart would re-stagnate and double the reported iters.
        ir.inner_restarts = 0;
        int status = ir.call(J, R.data(), n, b.data(), m, x_stag.data(), n);
        EXPECT_EQ(status, 0);                 // stagnating is not an error return
        ASSERT_EQ(ir.inner_status_per_step.size(), 2u);
        EXPECT_EQ(ir.inner_status_per_step.front(),
                  static_cast<int>(RandLAPACK::InnerCGStatus::Stagnated));
        // Deterministic: nothing counts as progress, so the window elapses immediately.
        EXPECT_EQ(ir.inner_iters_per_step.front(), kWindow);
        // The reported residual is the BEST seen, not the last.
        EXPECT_DOUBLE_EQ(ir.inner_relres_per_step.front(),
                         ir.inner_best_relres_per_step.front());
        EXPECT_LE(ir.inner_best_iter_per_step.front(), ir.inner_iters_per_step.front());
        for (int64_t i = 0; i < n; ++i) EXPECT_TRUE(std::isfinite(x_stag[i]));
    }

    // --- contrast: the SAME problem with the exit disabled behaves as before ---
    {
        IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/kCap, /*n_steps=*/2);
        ir.inner_stag_window = 0;             // disable the exit entirely
        std::vector<T> x(n, 0);
        int status = ir.call(J, R.data(), n, b.data(), m, x.data(), n);
        EXPECT_EQ(status, 0);
        // Reaches the tolerance on its own, and takes MORE iterations than the window did.
        EXPECT_EQ(ir.inner_status_per_step.front(),
                  static_cast<int>(RandLAPACK::InnerCGStatus::Converged));
        EXPECT_GT(ir.inner_iters_per_step.front(), kWindow)
            << "contrast case needs a preconditioner weak enough to require several "
               "CG iterations, otherwise it cannot distinguish the early exit";
        for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(x[i], x_true[i], 1e-8);
    }
}

// Single restart (added 2026-08-05, default inner_restarts = 1). After the inner CG
// terminates, it is rerun once from the iterate it returned, against the TRUE residual
// c - M z (the recursive residual CG tracks internally drifts in finite precision, so
// both the convergence test and the stagnation window can be reading fiction by the
// time they fire).
//
// Three properties pinned here:
//   1. A capped attempt restarts and the reported per-step count is the SUM of both
//      attempts (cap 2 => exactly 4 with the same weak preconditioner).
//   2. The restart never loses ground: its entry snapshot is the incoming iterate, so
//      the aggregated best residual is <= the single-attempt one.
//   3. For a solve whose first attempt genuinely converged, the restart is (near) free:
//      the entry check sees the true residual already under tolerance.
TEST_F(TestIterRefineLSQ, single_restart_reruns_from_returned_iterate) {
    using T = double;
    int64_t m = 120, n = 20;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 4242);
    fill_random(x_true, 777);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    // Same deliberately weak preconditioner as capped_solve_is_reported.
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 31337, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    // Baseline: one attempt, cap 2.
    T best_single = 0;
    {
        IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/2, /*n_steps=*/2);
        ir.inner_restarts = 0;
        std::vector<T> x(n, 0);
        ASSERT_EQ(ir.call(J, R.data(), n, b.data(), m, x.data(), n), 0);
        best_single = ir.inner_best_relres_per_step.front();
    }

    // (1) + (2): default restart on the same capped problem.
    {
        IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/2, /*n_steps=*/2);
        std::vector<T> x(n, 0);
        ASSERT_EQ(ir.call(J, R.data(), n, b.data(), m, x.data(), n), 0);
        EXPECT_EQ(ir.inner_restarts, 1);   // the documented default
        EXPECT_EQ(ir.inner_status_per_step.front(),
                  static_cast<int>(RandLAPACK::InnerCGStatus::HitCap));
        EXPECT_EQ(ir.inner_iters_per_step.front(), 4);   // 2 (attempt) + 2 (restart)
        EXPECT_LE(ir.inner_best_relres_per_step.front(), best_single);
        for (int64_t i = 0; i < n; ++i) EXPECT_TRUE(std::isfinite(x[i]));
    }

    // (3): a converging solve pays at most the one extra M apply, not extra budget.
    {
        IterRefineLSQ<T> ir_no(/*tol=*/1e-12, /*max_inner=*/200, /*n_steps=*/2);
        ir_no.inner_restarts = 0;
        std::vector<T> x0v(n, 0);
        ASSERT_EQ(ir_no.call(J, R.data(), n, b.data(), m, x0v.data(), n), 0);

        IterRefineLSQ<T> ir1(/*tol=*/1e-12, /*max_inner=*/200, /*n_steps=*/2);
        std::vector<T> x1v(n, 0);
        ASSERT_EQ(ir1.call(J, R.data(), n, b.data(), m, x1v.data(), n), 0);
        EXPECT_EQ(ir1.inner_status_per_step.front(),
                  static_cast<int>(RandLAPACK::InnerCGStatus::Converged));
        // The restart may add a couple of true-residual polish iterations when the
        // recursive residual was optimistic, but never a second full solve.
        EXPECT_LE(ir1.inner_iters_per_step.front(),
                  ir_no.inner_iters_per_step.front() + 3);
        for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(x1v[i], x_true[i], 1e-8);
    }
}

// A converging solve must be untouched by the stagnation logic: it reports Converged, and
// the exit does not fire before the tolerance is met.
TEST_F(TestIterRefineLSQ, stagnation_exit_does_not_disturb_a_converging_solve) {
    using T = double;
    int64_t m = 150, n = 25;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 9091);
    fill_random(x_true, 1234);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    IterRefineLSQ<T> ir(/*tol=*/1e-12, /*max_inner=*/200, /*n_steps=*/2);
    std::vector<T> x(n, 0);
    int status = ir.call(J, R.data(), n, b.data(), m, x.data(), n);
    EXPECT_EQ(status, 0);
    for (size_t s = 0; s < ir.inner_status_per_step.size(); ++s) {
        EXPECT_EQ(ir.inner_status_per_step[s],
                  static_cast<int>(RandLAPACK::InnerCGStatus::Converged))
            << "step " << s << " should converge, not stagnate";
    }
    for (int64_t i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], x_true[i], 1e-8);
}
