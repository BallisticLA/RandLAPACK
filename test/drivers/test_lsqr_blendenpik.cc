// Unit tests for RandLAPACK::lsqr and RandLAPACK::Blendenpik_linops.
//
// Both shipped untested (added in commit 578c18d); this file is their first coverage.
// It was written alongside the 2026-07-27 investigation into Blendenpik "stopping too
// early", so the cases deliberately pin the behaviours that investigation depends on:
//   * LSQR agrees with the LAPACK least-squares reference, preconditioned or not.
//   * LSQR reports 1 (not 0) when it exhausts its iteration cap.
//   * Blendenpik solves the LS problem and reports a real residual, not a sentinel.
//   * Blendenpik's sketch-and-solve warm start is mathematically transparent: it must
//     reach the same solution as the cold start, only sooner.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using RandLAPACK::linops::DenseLinOp;
using blas::Layout;


class TestLSQRBlendenpik : public ::testing::Test {
protected:
    using RNG = r123::Philox4x32;

    // Random m x n matrix (column-major) plus a right-hand side, via RandBLAS.
    template <typename T>
    static void make_problem(int64_t m, int64_t n, uint32_t seed,
                             std::vector<T>& A, std::vector<T>& b, std::vector<T>& x_true) {
        A.resize(m * n); b.resize(m); x_true.resize(n);
        RandBLAS::RNGState<RNG> state(seed);
        RandBLAS::DenseDist DA(m, n);
        auto s2 = RandBLAS::fill_dense(DA, A.data(), state);
        RandBLAS::DenseDist DX(n, 1);
        RandBLAS::fill_dense(DX, x_true.data(), s2);
        // Well-conditioned: push the diagonal up so the columns are not near-dependent.
        for (int64_t j = 0; j < n; ++j) A[j + j * m] += (T)n;
        blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    }

    // LAPACK reference solution (gels destroys its inputs, so pass copies).
    template <typename T>
    static std::vector<T> gels_reference(const std::vector<T>& A, const std::vector<T>& b,
                                         int64_t m, int64_t n) {
        std::vector<T> Ac(A), bc(b);
        lapack::gels(blas::Op::NoTrans, m, n, 1, Ac.data(), m, bc.data(), m);
        return std::vector<T>(bc.begin(), bc.begin() + n);
    }

    template <typename T>
    static T rel_err(const std::vector<T>& x, const std::vector<T>& ref) {
        T num = 0;
        for (size_t i = 0; i < ref.size(); ++i) { T d = x[i] - ref[i]; num += d * d; }
        return std::sqrt(num) / blas::nrm2((int64_t)ref.size(), ref.data(), 1);
    }
};


// Unpreconditioned LSQR must reproduce the LAPACK least-squares solution.
TEST_F(TestLSQRBlendenpik, lsqr_unpreconditioned_matches_gels) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 11, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    T relres = -1;
    int st = RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                 1e-14, 1e-14, 2000, iters, nullptr, &relres);
    EXPECT_EQ(st, 0);                 // met a stopping test
    EXPECT_GT(iters, 0);
    EXPECT_GE(relres, (T)0);          // out-param actually populated
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}


// With R from QR(A) as a right preconditioner, A R^-1 is orthonormal, so LSQR should
// need very few iterations and still land on the reference solution.
TEST_F(TestLSQRBlendenpik, lsqr_preconditioned_converges_fast) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 13, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    // R = upper-triangular factor of QR(A).
    std::vector<T> Ac(A), tau(n), R(n * n, 0);
    lapack::geqrf(m, n, Ac.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, Ac.data(), m, R.data(), n);
    if (n > 1)
        lapack::laset(lapack::MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R.data() + 1, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    int st = RandLAPACK::lsqr<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                 1e-14, 1e-14, 2000, iters, nullptr, nullptr);
    EXPECT_EQ(st, 0);
    EXPECT_LE(iters, 5);              // perfect preconditioner => a handful of iterations
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}


// Exhausting the cap must be reported as status 1. This is the signal Blendenpik needs
// in order to tell its caller that the requested tolerance was NOT met.
TEST_F(TestLSQRBlendenpik, lsqr_reports_cap_hit) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 17, A, b, x_true);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    int st = RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                 1e-15, 1e-15, /*max_iters=*/2, iters, nullptr, nullptr);
    EXPECT_EQ(st, 1);                 // capped, not silently "successful"
    EXPECT_EQ(iters, 2);
}


// Blendenpik must solve the LS problem and report a real residual (its solver_relres
// column used to be a structural -1 because it never requested the out-param).
TEST_F(TestLSQRBlendenpik, blendenpik_solves_and_reports_residual) {
    using T = double;
    int64_t m = 400, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 23, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(5);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, (T)1e-13);
    std::vector<T> x(n, 0);
    int status = bp.call(Aop, b.data(), m, x.data(), n, /*d_factor=*/4.0, state);

    EXPECT_EQ(status, 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-7);
    EXPECT_TRUE(bp.converged);
    EXPECT_GE(bp.final_relres, (T)0);            // populated, not the -1 sentinel
    EXPECT_EQ(bp.R_out.size(), (size_t)(n * n));
    ASSERT_EQ(bp.times.size(), (size_t)4);
}


// The sketch-and-solve warm start must be mathematically transparent: same answer as a
// cold start, reached in no more iterations. (Epperly/Meier/Nakatsukasa arXiv:2406.03468
// call this initialization necessary for forward stability; it is off in the cold case
// only so the two can be compared here.)
TEST_F(TestLSQRBlendenpik, blendenpik_warm_start_matches_cold_start) {
    using T = double;
    int64_t m = 400, n = 24;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 29, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    std::vector<T> x_cold(n, 0), x_warm(n, 0);
    int iters_cold = 0, iters_warm = 0;
    {
        RandBLAS::RNGState<RNG> state(31);
        RandLAPACK::Blendenpik_linops<T, RNG> bp(true, (T)1e-13);
        bp.warm_start = false;
        ASSERT_EQ(bp.call(Aop, b.data(), m, x_cold.data(), n, 4.0, state), 0);
        iters_cold = bp.lsqr_iters;
    }
    {
        RandBLAS::RNGState<RNG> state(31);   // same seed => same sketch => same R
        RandLAPACK::Blendenpik_linops<T, RNG> bp(true, (T)1e-13);
        bp.warm_start = true;
        ASSERT_EQ(bp.call(Aop, b.data(), m, x_warm.data(), n, 4.0, state), 0);
        iters_warm = bp.lsqr_iters;
    }

    // Both must be correct solutions of the same problem ...
    EXPECT_LT(rel_err(x_cold, x_ref), 1e-7);
    EXPECT_LT(rel_err(x_warm, x_ref), 1e-7);
    // ... and agree with each other to solver tolerance: the warm start only changes
    // where the iteration begins, never the fixed point it converges to.
    EXPECT_LT(rel_err(x_warm, x_cold), 1e-6);
    // Starting closer to the answer must not cost extra iterations.
    EXPECT_LE(iters_warm, iters_cold);
}


// Calling a Q-less QR driver twice on the same object must not leak its test-mode Q
// buffer. Before 2026-07-27 `this->Q = new T[m*n]` overwrote the previous pointer with
// no delete, so every extra call() leaked an m x n block. Under ASan/valgrind this test
// fails outright on the old code; without a sanitizer it still pins the contract that Q
// is re-materialized and remains correct across repeated calls.
TEST_F(TestLSQRBlendenpik, repeated_call_does_not_leak_Q) {
    using T = double;
    int64_t m = 300, n = 16;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 37, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    RandLAPACK::CholQR_linops<T> qr(/*time_subroutines=*/false, (T)0);
    qr.test_mode = true;

    std::vector<T> R(n * n, 0);
    const T* first_Q = nullptr;
    for (int rep = 0; rep < 3; ++rep) {
        std::fill(R.begin(), R.end(), (T)0);
        ASSERT_EQ(qr.call(Aop, R.data(), n), 0) << "rep " << rep;
        ASSERT_NE(qr.Q, nullptr);
        EXPECT_EQ(qr.Q_rows, m);
        EXPECT_EQ(qr.Q_cols, n);
        if (rep == 0) first_Q = qr.Q;
        // Q must be a usable orthonormal factor on every call, not just the first.
        T orth = RandLAPACK::testing::orthogonality_error<T>(qr.Q, m, n);
        EXPECT_LT(orth, 1e-10) << "rep " << rep;
    }
    (void)first_Q;   // pointer value may legitimately be reused by the allocator
}


// ---------------------------------------------------------------------------
// restarted_pcg_ne: restarted PCG on the right-preconditioned normal equations,
// the second solver of Oleg's reference benchmark (solve_with_lsqr's sibling in
// ar_sysid_toeplitz_qless_qr_benchmark.m). Ported 2026-08-06; the 07-14 Toeplitz
// port shipped LSQR only.
// ---------------------------------------------------------------------------


// Unpreconditioned restarted PCG (H = A^T A) must reproduce the LAPACK LS solution.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_unpreconditioned_matches_gels) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 41, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    T relres = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                             1e-12, 2000, iters,
                                             /*restart_maxit=*/200, /*restart_drop=*/1e-2,
                                             /*max_restarts=*/-1,
                                             nullptr, nullptr, &relres);
    EXPECT_EQ(st, 0);                 // met the LS tolerance
    EXPECT_GT(iters, 0);
    EXPECT_GE(relres, (T)0);          // out-param actually populated
    EXPECT_LE(relres, 1e-12);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}


// With R from QR(A), H = R^{-T} A^T A R^{-1} = I, so PCG needs a handful of total
// iterations across all restarts and still lands on the reference solution.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_preconditioned_converges_fast) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 43, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    std::vector<T> Ac(A), tau(n), R(n * n, 0);
    lapack::geqrf(m, n, Ac.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, Ac.data(), m, R.data(), n);
    if (n > 1)
        lapack::laset(lapack::MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R.data() + 1, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                             1e-12, 2000, iters);
    EXPECT_EQ(st, 0);
    EXPECT_LE(iters, 6);              // perfect preconditioner => a few iterations total
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}


// The restart mechanism itself: a loose per-restart drop (1e-1) cannot reach 1e-12 in
// one inner solve, so the outer loop must restart several times against the TRUE
// residual and still converge. This is the behaviour that distinguishes the solver
// from a single PCG call.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_restarts_until_tolerance) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 47, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0, restarts = 0;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                             1e-12, 2000, iters,
                                             /*restart_maxit=*/200, /*restart_drop=*/1e-1,
                                             /*max_restarts=*/-1,
                                             &restarts, nullptr, nullptr);
    EXPECT_EQ(st, 0);
    EXPECT_GT(restarts, 1);           // the loose drop forces more than one outer round
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}


// Exhausting the total iteration budget must be reported as status 1, mirroring both
// MATLAB's convention and our lsqr. The budget is TOTAL inner iterations, shared
// across restarts, exactly as in the reference (innerMaxit = min(restartMaxit,
// maxit - iter)).
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_reports_cap_hit) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 53, A, b, x_true);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                             /*tol=*/1e-15, /*max_iters=*/3, iters);
    EXPECT_EQ(st, 1);                 // capped, not silently "successful"
    EXPECT_LE(iters, 3);              // never exceeds the total budget
    EXPECT_GT(iters, 0);
}


// max_restarts bounds the OUTER rounds: it counts additional rounds after the first,
// matching IterRefineLSQ's inner_restarts convention (0 = single round, negative =
// unlimited, the reference behaviour). With a loose drop and max_restarts=0 the solver
// must stop after one round, report the budget as not met, and leave the true relative
// residual above the tolerance it could have reached with more rounds.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_honors_max_restarts) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 59, A, b, x_true);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    // Unlimited restarts (reference behaviour) converges to 1e-12.
    std::vector<T> x_free(n, 0);
    int iters_free = 0, restarts_free = 0;
    int st_free = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_free.data(),
                                                  1e-12, 2000, iters_free,
                                                  /*restart_maxit=*/200, /*restart_drop=*/1e-1,
                                                  /*max_restarts=*/-1,
                                                  &restarts_free, nullptr, nullptr);
    ASSERT_EQ(st_free, 0);
    ASSERT_GT(restarts_free, 1);      // the loose drop genuinely needs several rounds

    // A single round (max_restarts=0) must stop early and say so.
    std::vector<T> x_one(n, 0);
    int iters_one = 0, restarts_one = 0;
    T relres_one = -1;
    int st_one = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_one.data(),
                                                 1e-12, 2000, iters_one,
                                                 /*restart_maxit=*/200, /*restart_drop=*/1e-1,
                                                 /*max_restarts=*/0,
                                                 &restarts_one, nullptr, &relres_one);
    EXPECT_EQ(st_one, 1);             // budget (rounds) exhausted, not converged
    EXPECT_EQ(restarts_one, 1);       // exactly one round ran
    EXPECT_LT(iters_one, iters_free); // strictly less work than the free run
    EXPECT_GT(relres_one, 1e-12);     // and the tolerance was honestly not met
}


// Structure unification (2026-08-06, Max): restarted_pcg_ne's inner CG must carry the
// same stagnation-exit + best-iterate machinery as IterRefineLSQ's. On a problem whose
// NE floor sits far above tol, the OLD plain inner CG ground out every round's full
// cap (4 rounds x 200 = exactly 800 iterations); with the stagnation window each round
// exits once the residual flatlines, so the total must come in well under the budget
// while still reporting the honest not-converged flag.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_stagnation_exits_early) {
    using T = double;
    int64_t m = 200, n = 20;
    // Dense ill-conditioned A (seeded, deterministic): random well-conditioned base
    // with geometric column scaling down to 1e-8, so kappa(H) ~ 1e16 on the normal
    // equations. Unlike a diagonal matrix (where CG terminates EXACTLY by Krylov
    // exhaustion), a dense spectrum makes the recursive residual flatline at its
    // finite-precision floor -- the stagnation regime this test pins.
    std::vector<T> A, b, x_true;
    make_problem(m, n, 61, A, b, x_true);
    for (int64_t j = 0; j < n; ++j) {
        T s = std::pow((T)10.0, (T)(-8.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    // Make the system INCONSISTENT (column scaling preserves range, so b was still
    // in range(A) and the solver legitimately converged): the optimal LS residual
    // is now O(1), far above tol, so only the round/iteration budget can end this.
    for (int64_t i = 0; i < m; ++i) b[i] += (T)0.1 / (T)(1 + i);
    std::vector<T> x(n, 0.0);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    int iters = 0, restarts = 0;
    T relres = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                             /*tol=*/1e-15, /*max_iters=*/2000, iters,
                                             /*restart_maxit=*/200, /*restart_drop=*/1e-30,
                                             /*max_restarts=*/3,
                                             &restarts, nullptr, &relres);
    EXPECT_EQ(st, 1);                 // tolerance genuinely unreachable
    EXPECT_EQ(restarts, 4);           // all rounds ran
    EXPECT_LT(iters, 800);            // stagnation exit: strictly under the full budget
    EXPECT_GT(iters, 0);
    EXPECT_TRUE(std::isfinite((double)relres));
}
