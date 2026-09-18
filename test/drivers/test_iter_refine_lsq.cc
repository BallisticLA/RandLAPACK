// Unit tests for RandLAPACK::IterRefineLSQ: iterative-refinement LSQ
// using R as a right preconditioner.
//
// Strategy: wrap a small dense tall A as a DenseLinOp, build R from
// lapack::geqrf on a copy of A, then verify the IR-LSQ solution matches
// the closed-form lapack::gels reference across well-conditioned,
// residualful, and imperfect-R cases.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <thread>
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

    // Build R from QR(A): perfect preconditioner.
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
    // Under the paced-restart scheme (round_drop = 1e-4) an
    // imperfect preconditioner needs more than two shallow rounds to reach
    // gels-level accuracy: give the loop room and let outer_tol stop it.
    IterRefineLSQ<T> ir(1e-12, 100, /*n_steps=*/20);
    ir.outer_tol = (T)1e-12;
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
    // The paced loop must exit well before the 20-round cap.
    EXPECT_LT(ir.outer_iters_done, 20);
}


// A capped, non-converged inner solve must be DISTINGUISHABLE from a converged one.
//
// Without this instrumentation both looked identical to the caller: inner_cg
// returned 0 whether it converged or exhausted its budget, so a benchmark CSV could not
// tell "converged in 6 iterations" from "gave up at the cap".
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
        // Legacy fixed-tolerance rounds: this test pins the "asked for 1e-14,
        // capped at 2 iterations" mechanics, which the paced default would
        // change (the round target would be round_drop, not 1e-14).
        ir.round_drop = 0;
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
        ir.round_drop = 0;   // legacy fixed-tolerance rounds, as above
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

// Stagnation exit. The ISAAC diagnostic showed an inner CG reaching its
// residual floor at iteration 17 of a 200-iteration step and then grinding out the
// remaining ~183 with a BIT-IDENTICAL best residual, while the outer solution got 11x worse
// when the budget was raised 10x. The driver now detects the flatline, stops, and returns
// the best iterate rather than the last.
//
// HOW THIS IS PROVOKED, and one thing that does NOT work. The obvious trick, asking for an
// unreachable tolerance via inner_tol = 0, fails: on a small well-conditioned system CG
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
    // exact Cholesky factor, so M = R^-T A^T A R^-1 = I and CG converges in ONE iteration,
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
        ir.round_drop = 0;   // legacy fixed 1e-14 rounds: the contrast needs the
                             // deep solve the paced default would cut short
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

// Paced restarts. Each round's inner CG stops after a round_drop (1e-4)
// residual drop and the outer loop restarts against the TRUE residual; outer_tol
// terminates the loop. Pinned here:
//   1. A weak preconditioner needs several rounds but converges to the target well
//      before the 20-round cap.
//   2. Every round is shallow: no round runs anywhere near a deep fixed-tol solve.
//   3. The final solution matches the direct reference.
TEST_F(TestIterRefineLSQ, paced_rounds_converge_with_early_exit) {
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

    IterRefineLSQ<T> ir(/*tol=*/1e-14, /*max_inner=*/200, /*n_steps=*/20);
    ir.outer_tol = (T)1e-10;
    std::vector<T> x(n, 0);
    ASSERT_EQ(ir.call(J, R.data(), n, b.data(), m, x.data(), n), 0);

    EXPECT_GT(ir.outer_iters_done, 1);        // the weak R genuinely needs rounds
    EXPECT_LT(ir.outer_iters_done, 20);       // but converges before the cap
    EXPECT_LE(ir.final_residual_norm, (T)1e-10);
    for (auto it : ir.inner_iters_per_step)
        EXPECT_LT(it, 200);                   // every round is shallow, none hit the cap
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(x[i], x_true[i], 1e-8);
}

// IterRefineLSQ is an adapter over restarted_pcg_ne, so the same problem with
// equivalently-mapped parameters must produce the IDENTICAL iterate sequence,
// not merely a close one. This is the contract that the FEM2
// and Toeplitz benchmarks now run one solver, not two implementations of it.
TEST_F(TestIterRefineLSQ, delegation_matches_restarted_pcg_ne_exactly) {
    using T = double;
    int64_t m = 120, n = 20;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 4242);
    fill_random(x_true, 777);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 31337, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const T tol_outer = (T)1e-10, tol_abs = (T)1e-14, drop = (T)1e-4;
    const int cap = 200, rounds_max = 20;

    IterRefineLSQ<T> ir(tol_abs, cap, rounds_max);
    ir.outer_tol = tol_outer;
    std::vector<T> x_ir(n, 0);
    ASSERT_EQ(ir.call(J, R.data(), n, b.data(), m, x_ir.data(), n), 0);

    std::vector<T> x_eng(n, 0);
    int iters = 0, rounds = 0;
    int st = RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x_eng.data(),
                                             tol_outer, cap * rounds_max, iters,
                                             /*restart_maxit=*/cap, /*restart_drop=*/drop,
                                             /*max_restarts=*/rounds_max - 1,
                                             &rounds, nullptr, nullptr,
                                             /*stag_window=*/20, /*stag_rel_improve=*/(T)1e-3,
                                             /*inner_abs_tol=*/tol_abs);
    ASSERT_EQ(st, 0);
    EXPECT_EQ(ir.outer_iters_done, rounds);
    for (int64_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(x_ir[i], x_eng[i]) << "element " << i;
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


// Outer early exit (structure unification with restarted_pcg_ne): when
// outer_tol > 0, the refinement loop must stop as soon as the TRUE residual
// ||b - Jx||/||b|| meets it, instead of always running all n_refine_steps. With a
// perfect preconditioner one step reaches far below 1e-8, so a generous 8-step
// budget must be cut short. outer_tol = 0 (the default) preserves the historical
// fixed-step behaviour, pinned by every other test in this file.
TEST_F(TestIterRefineLSQ, outer_tol_stops_refinement_early) {
    using T = double;
    int64_t m = 80, n = 12;

    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 71);
    fill_random(x_true, 72);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);

    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*tol=*/1e-12, /*max_inner=*/50, /*n_steps=*/8);
    ir.outer_tol = (T)1e-8;
    std::vector<T> x(n, 0);
    ASSERT_EQ(ir.call(J, R.data(), n, b.data(), m, x.data(), n), 0);

    EXPECT_LT(ir.outer_iters_done, 8);              // stopped before the step budget
    EXPECT_GT(ir.outer_iters_done, 0);              // but did real work
    EXPECT_LE(ir.final_residual_norm, (T)1e-8);     // and the claim is honest
}


// ---- Backward-error oracle exit (engine status 5) --------------------------
// The oracle is a caller-supplied std::function; these tests feed it the LS
// relative residual itself, so every recorded value can be checked against the
// engine's own ls_relres.

// The engine stops with status 5 as soon as the oracle is at or below be_tol,
// records the oracle value for every round, and calls it exactly once per round.
TEST_F(TestIterRefineLSQ, be_oracle_stops_engine_with_status_5) {
    using T = double;
    int64_t m = 120, n = 20;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 515);
    fill_random(x_true, 516);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);   // imperfect R: several rounds
    fill_random(pert, 517, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const T b_norm = blas::nrm2(m, b.data(), 1);
    int calls = 0;
    RandLAPACK::BackwardErrorOracle<T> oracle = [&](const T* x, const T* r, const T* ATr) -> T {
        (void)x; (void)ATr; ++calls;
        return blas::nrm2(m, r, 1) / b_norm;      // the LS relres itself, as a checkable measure
    };
    std::vector<T> x(n, 0);
    RandLAPACK::PCGRoundHistory<T> hist;
    int iters = 0, rounds = 0;
    int st = RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
        /*tol=*/(T)0, /*max_iters=*/4000, iters, /*restart_maxit=*/200, /*restart_drop=*/(T)1e-4,
        /*max_restarts=*/-1, &rounds, nullptr, nullptr, /*stag_window=*/20,
        /*stag_rel_improve=*/(T)1e-3, /*inner_abs_tol=*/(T)0, &hist, /*x0=*/nullptr,
        /*outer_stag_window=*/0, oracle, /*be_tol=*/(T)1e-6);
    EXPECT_EQ(st, 5);
    EXPECT_GE(rounds, 1);
    ASSERT_EQ(hist.be.size(), (size_t)rounds);
    EXPECT_EQ(calls, rounds);
    EXPECT_LE(hist.be.back(), (T)1e-6);
    for (size_t k = 0; k + 1 < hist.be.size(); ++k) EXPECT_GT(hist.be[k], (T)1e-6);
    for (size_t k = 0; k < hist.be.size(); ++k)        // measured on the same b - A x the engine uses
        EXPECT_NEAR(hist.be[k], hist.ls_relres[k], 1e-13 * std::max((T)1, hist.ls_relres[k]));
    EXPECT_EQ(hist.be_x0, (T)-1);                       // cold start: no x0 evaluation
    EXPECT_GE(hist.t_be_us, 0L);
}

// At a status-5 exit the returned x is bitwise the iterate the oracle last saw, the total
// iteration count equals the sum of the per-round counts, and final_relres is the last
// round's LS relres. Also pins the timing exclusion: times[3] must not contain the
// oracle's own wall time.
TEST_F(TestIterRefineLSQ, be_oracle_exit_returns_the_evaluated_iterate_and_consistent_counts) {
    using T = double;
    int64_t m = 120, n = 20;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 555);
    fill_random(x_true, 556);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 557, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const T b_norm = blas::nrm2(m, b.data(), 1);
    std::vector<T> x_seen(n, 0);
    RandLAPACK::BackwardErrorOracle<T> oracle = [&](const T* x, const T* r, const T*) -> T {
        std::copy(x, x + n, x_seen.begin());             // snapshot of what the oracle was shown
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // make t_be measurable
        return blas::nrm2(m, r, 1) / b_norm;
    };
    std::vector<T> x(n, 0);
    RandLAPACK::PCGRoundHistory<T> hist;
    int iters = 0, rounds = 0;
    long times[4] = {0, 0, 0, 0};
    T final_relres = (T)-1;
    auto t0 = std::chrono::steady_clock::now();
    int st = RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
        (T)0, 4000, iters, 200, (T)1e-4, -1, &rounds, times, &final_relres,
        20, (T)1e-3, (T)0, &hist, nullptr, 0, oracle, (T)1e-6);
    long wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - t0).count();
    ASSERT_EQ(st, 5);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(x[i], x_seen[i]) << "element " << i;
    int sum_iters = 0; for (int v : hist.iters) sum_iters += v;
    EXPECT_EQ(iters, sum_iters);
    ASSERT_FALSE(hist.ls_relres.empty());
    EXPECT_EQ(final_relres, hist.ls_relres.back());
    EXPECT_GE(hist.t_be_us, 20000L * rounds);            // the sleeps were counted
    EXPECT_LT(times[3], wall_us - hist.t_be_us / 2);      // and kept out of the solver total
}

// An ACTIVE oracle that is never met (be_tol = 0 against a strictly positive measure) must
// leave status, counts, x and the LS history identical to a run without an oracle: the
// oracle may only end a run, never change it.
TEST_F(TestIterRefineLSQ, active_but_unmet_oracle_changes_nothing) {
    using T = double;
    int64_t m = 120, n = 20;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 565);
    fill_random(x_true, 566);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 567, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    auto run = [&](RandLAPACK::BackwardErrorOracle<T> oracle, T be_tol, std::vector<T>& x,
                   int& iters, int& rounds, RandLAPACK::PCGRoundHistory<T>& hist) {
        return RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
            (T)1e-10, 4000, iters, 200, (T)1e-4, -1, &rounds, nullptr, nullptr,
            20, (T)1e-3, (T)0, &hist, nullptr, 2, oracle, be_tol);
    };
    const T b_norm = blas::nrm2(m, b.data(), 1);
    int calls = 0;
    RandLAPACK::BackwardErrorOracle<T> positive = [&](const T*, const T* r, const T*) -> T {
        ++calls; return blas::nrm2(m, r, 1) / b_norm;   // > 0 for every iterate here
    };
    std::vector<T> xa(n, 0), xb(n, 0);
    int ia = 0, ra = 0, ib = 0, rb = 0;
    RandLAPACK::PCGRoundHistory<T> ha, hb;
    int sa = run({}, (T)-1, xa, ia, ra, ha);
    int sb = run(positive, (T)0, xb, ib, rb, hb);       // active, never satisfied
    EXPECT_EQ(calls, rb);
    EXPECT_EQ(sa, sb);
    EXPECT_NE(sb, 5);
    EXPECT_EQ(ia, ib);
    EXPECT_EQ(ra, rb);
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(xa[i], xb[i]) << "element " << i;
    ASSERT_EQ(ha.ls_relres.size(), hb.ls_relres.size());
    for (size_t k = 0; k < ha.ls_relres.size(); ++k) EXPECT_EQ(ha.ls_relres[k], hb.ls_relres[k]);
    for (T v : hb.be) EXPECT_GT(v, (T)0);               // recorded, never -1, never met
}

// Precedence: when the LS tolerance and the oracle are met in the same round the run
// reports status 0 (tol), never 5.
TEST_F(TestIterRefineLSQ, ls_tolerance_outranks_the_oracle_in_the_same_round) {
    using T = double;
    int64_t m = 120, n = 20;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 575);
    fill_random(x_true, 576);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 577, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const T b_norm = blas::nrm2(m, b.data(), 1);
    RandLAPACK::BackwardErrorOracle<T> oracle = [&](const T*, const T* r, const T*) -> T {
        return blas::nrm2(m, r, 1) / b_norm;            // identical to the engine's LS relres
    };
    std::vector<T> x(n, 0);
    RandLAPACK::PCGRoundHistory<T> hist;
    int iters = 0, rounds = 0;
    const T shared_tol = (T)1e-6;                       // tol == be_tol: both met in one round
    int st = RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
        shared_tol, 4000, iters, 200, (T)1e-4, -1, &rounds, nullptr, nullptr,
        20, (T)1e-3, (T)0, &hist, nullptr, 0, oracle, shared_tol);
    EXPECT_EQ(st, 0);
    ASSERT_FALSE(hist.be.empty());
    EXPECT_LE(hist.be.back(), shared_tol);              // the oracle was met too, and lost
}

// An operator whose transposed apply is NOT the adjoint of its forward apply makes the
// preconditioned normal-equation operator indefinite, so CG breaks down at once
// (p^T H p <= 0). Used to force the breakdown exit deterministically.
template <typename T>
struct TwoFacedOp {
    using scalar_t = T;
    DenseLinOp<T> fwd;    // applies A
    DenseLinOp<T> adj;    // applies (-A)^T in place of A^T
    const int64_t n_rows, n_cols;
    TwoFacedOp(int64_t m, int64_t n, const T* A, const T* negA)
        : fwd(m, n, A, m, Layout::ColMajor), adj(m, n, negA, m, Layout::ColMajor),
          n_rows(m), n_cols(n) {}
    void operator()(Layout layout, blas::Op tA, blas::Op tB, int64_t mm, int64_t nn, int64_t kk,
                    T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc) {
        if (tA == blas::Op::NoTrans) fwd(layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
        else                         adj(layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
    }
    void operator()(blas::Side side, Layout layout, blas::Op tA, blas::Op tB, int64_t mm, int64_t nn,
                    int64_t kk, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc) {
        if (tA == blas::Op::NoTrans) fwd(side, layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
        else                         adj(side, layout, tA, tB, mm, nn, kk, alpha, B, ldb, beta, C, ldc);
    }
};

// Precedence: a CG breakdown in the triggering round outranks the oracle. The oracle here
// reports success unconditionally; the run must still end with status 2 and record the
// oracle value for the round.
TEST_F(TestIterRefineLSQ, breakdown_outranks_the_oracle_in_the_same_round) {
    using T = double;
    int64_t m = 90, n = 12;
    std::vector<T> A(m * n), negA(m * n), b(m), x_true(n);
    fill_random(A, 585);
    fill_random(x_true, 586);
    for (int64_t i = 0; i < m * n; ++i) negA[i] = -A[i];
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);
    TwoFacedOp<T> J(m, n, A.data(), negA.data());

    int calls = 0;
    RandLAPACK::BackwardErrorOracle<T> always_met = [&](const T*, const T*, const T*) -> T { ++calls; return (T)0; };
    std::vector<T> x(n, 0);
    RandLAPACK::PCGRoundHistory<T> hist;
    int iters = -1, rounds = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
        (T)0, 4000, iters, 200, (T)1e-4, -1, &rounds, nullptr, nullptr,
        20, (T)1e-3, (T)0, &hist, nullptr, 0, always_met, /*be_tol=*/(T)1);
    EXPECT_EQ(st, 2);
    EXPECT_EQ(rounds, 1);
    EXPECT_EQ(calls, 1);
    ASSERT_EQ(hist.status.size(), (size_t)1);
    EXPECT_EQ(hist.status[0], static_cast<int>(RandLAPACK::InnerCGStatus::Breakdown));
    ASSERT_EQ(hist.be.size(), (size_t)1);
    EXPECT_LE(hist.be[0], (T)1);                        // the oracle passed and still lost
}

// With be_tol < 0 the oracle is never called and the run is bit-identical to a run without it.
TEST_F(TestIterRefineLSQ, be_oracle_with_negative_tol_is_never_called_and_changes_nothing) {
    using T = double;
    int64_t m = 120, n = 20;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 525);
    fill_random(x_true, 526);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 527, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    auto run = [&](RandLAPACK::BackwardErrorOracle<T> oracle, T be_tol, std::vector<T>& x,
                   int& iters, int& rounds, RandLAPACK::PCGRoundHistory<T>& hist) {
        return RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
            (T)1e-10, 4000, iters, 200, (T)1e-4, -1, &rounds, nullptr, nullptr,
            20, (T)1e-3, (T)0, &hist, nullptr, 2, oracle, be_tol);
    };
    int calls = 0;
    RandLAPACK::BackwardErrorOracle<T> counting = [&](const T*, const T*, const T*) -> T { ++calls; return (T)0; };
    std::vector<T> xa(n, 0), xb(n, 0);
    int ia = 0, ra = 0, ib = 0, rb = 0;
    RandLAPACK::PCGRoundHistory<T> ha, hb;
    int sa = run({}, (T)-1, xa, ia, ra, ha);
    int sb = run(counting, (T)-1, xb, ib, rb, hb);
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(sa, sb);
    EXPECT_EQ(ia, ib);
    EXPECT_EQ(ra, rb);
    for (int64_t i = 0; i < n; ++i) EXPECT_DOUBLE_EQ(xa[i], xb[i]) << "element " << i;
    ASSERT_EQ(hb.be.size(), (size_t)rb);
    for (T v : hb.be) EXPECT_EQ(v, (T)-1);
    EXPECT_EQ(hb.t_be_us, 0L);
}

// A warm start that already meets be_tol returns 5 with zero rounds and zero iterations,
// and the x0 evaluation is recorded.
TEST_F(TestIterRefineLSQ, be_oracle_warm_start_already_converged_returns_5_without_iterating) {
    using T = double;
    int64_t m = 100, n = 15;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 535);
    fill_random(x_true, 536);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> R(n * n, 0);
    build_R_from_A(A.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const T b_norm = blas::nrm2(m, b.data(), 1);
    RandLAPACK::BackwardErrorOracle<T> oracle = [&](const T*, const T* r, const T*) -> T {
        return blas::nrm2(m, r, 1) / b_norm;
    };
    std::vector<T> x(n, 0);
    RandLAPACK::PCGRoundHistory<T> hist;
    int iters = -1, rounds = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(J, m, n, R.data(), n, b.data(), x.data(),
        (T)0, 4000, iters, 200, (T)1e-4, -1, &rounds, nullptr, nullptr,
        20, (T)1e-3, (T)0, &hist, /*x0=*/x_true.data(), 0, oracle, /*be_tol=*/(T)1e-8);
    EXPECT_EQ(st, 5);
    EXPECT_EQ(iters, 0);
    EXPECT_EQ(rounds, 0);
    EXPECT_TRUE(hist.be.empty());
    EXPECT_GE(hist.be_x0, (T)0);
    EXPECT_LE(hist.be_x0, (T)1e-8);
    for (int64_t i = 0; i < n; ++i) EXPECT_NEAR(x[i], x_true[i], 1e-10);
}

// IterRefineLSQ forwards the oracle to the engine and republishes its per-round values.
// call() keeps its contract (0 unless the inner CG broke down); the oracle exit is
// visible through engine_status, exactly like the LS-floor exit is today.
TEST_F(TestIterRefineLSQ, iter_refine_lsq_forwards_be_oracle) {
    using T = double;
    int64_t m = 120, n = 20;
    std::vector<T> A(m * n), b(m), x_true(n);
    fill_random(A, 545);
    fill_random(x_true, 546);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    std::vector<T> A_pert(A.begin(), A.end()), pert(m * n);
    fill_random(pert, 547, (T)0.30);
    for (int64_t i = 0; i < m * n; ++i) A_pert[i] += pert[i];
    std::vector<T> R(n * n, 0);
    build_R_from_A(A_pert.data(), m, n, R.data(), n);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);

    const T b_norm = blas::nrm2(m, b.data(), 1);
    IterRefineLSQ<T> ir(/*tol=*/(T)0, /*max_inner=*/200, /*n_steps=*/20);
    ir.outer_tol = (T)0;
    ir.outer_stag_window = 0;
    ir.be_oracle = [&](const T*, const T* r, const T*) -> T { return blas::nrm2(m, r, 1) / b_norm; };
    ir.be_tol = (T)1e-6;
    std::vector<T> x(n, 0);
    ASSERT_EQ(ir.call(J, R.data(), n, b.data(), m, x.data(), n), 0);
    EXPECT_EQ(ir.engine_status, 5);
    ASSERT_EQ(ir.be_per_step.size(), (size_t)ir.outer_iters_done);
    EXPECT_LE(ir.be_per_step.back(), (T)1e-6);
    EXPECT_EQ(ir.be_x0, (T)-1);
    EXPECT_GE(ir.t_be_us, 0L);
}
