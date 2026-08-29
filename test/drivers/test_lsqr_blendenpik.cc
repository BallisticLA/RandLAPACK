// Unit tests for RandLAPACK::lsqr and RandLAPACK::Blendenpik_linops.
//
// Both shipped untested; this file is their first coverage, written alongside
// an investigation into Blendenpik "stopping too early", so the cases
// deliberately pin the behaviours that investigation depends on:
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
    bp.max_iters = 2000;   // mandatory: bp.max_iters has no default
    std::vector<T> x(n, 0);
    int status = bp.call(Aop, b.data(), m, x.data(), n, /*d_factor=*/4.0, state);

    EXPECT_EQ(status, 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-7);
    EXPECT_TRUE(bp.converged);
    EXPECT_GE(bp.final_relres, (T)0);            // populated, not the -1 sentinel
    ASSERT_NE(bp.R_out, nullptr);
    EXPECT_EQ(bp.R_out_sz, n * n);
    // 5 slots: {t_sketch, t_qr, t_lsqr, total, t_x0}.
    ASSERT_EQ(bp.times.size(), (size_t)5);
}


// A rank-deficient sketch (Ask = S A has a zero column, so unpivoted geqrf leaves a
// zero diagonal in R) must be reported as a hard failure (1), with every output member
// left in the just-reset state call() establishes at its top, not the values of some
// earlier call on a reused object.
TEST_F(TestLSQRBlendenpik, blendenpik_rank_deficient_sketch_returns_1) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 101, A, b, x_true);
    // Zero out the first column of A: S * 0 = 0 regardless of the sketch S, so Ask
    // inherits an exact zero column and geqrf's first pivot is exactly zero,
    // so rank deficiency is guaranteed independent of the RNG state.
    for (int64_t i = 0; i < m; ++i) A[i] = (T)0;

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(103);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, (T)1e-10);
    bp.max_iters = 500;
    std::vector<T> x(n, (T)0);
    int status = bp.call(Aop, b.data(), m, x.data(), n, /*d_factor=*/4.0, state);

    EXPECT_EQ(status, 1);
    EXPECT_EQ(bp.R_out, nullptr);
    EXPECT_EQ(bp.R_out_sz, 0);
    EXPECT_FALSE(bp.converged);
    EXPECT_EQ(bp.final_relres, (T)-1);
    EXPECT_EQ(bp.lsqr_iters, 0);
    EXPECT_EQ(bp.lsqr_stop_test, 0);
    EXPECT_TRUE(bp.times.empty());
    EXPECT_TRUE(bp.lsqr_op_times.empty());
}


// init_only stops after the sketch-and-solve x0 and skips LSQR entirely: times[2]
// (the LSQR slot) must read 0, lsqr_iters must be 0, and final_relres must be x0's
// own true residual, not a placeholder.
TEST_F(TestLSQRBlendenpik, blendenpik_init_only_reports_x0_and_skips_lsqr) {
    using T = double;
    int64_t m = 300, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 107, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(109);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, (T)1e-10);
    bp.max_iters = 500;   // unused in init_only, set for safety per the class's own note
    bp.init_only = true;
    std::vector<T> x(n, (T)0);
    int status = bp.call(Aop, b.data(), m, x.data(), n, /*d_factor=*/4.0, state);

    ASSERT_EQ(status, 0);
    EXPECT_EQ(bp.lsqr_iters, 0);
    EXPECT_FALSE(bp.converged);          // no tolerance was pursued
    ASSERT_NE(bp.R_out, nullptr);
    EXPECT_EQ(bp.R_out_sz, n * n);
    ASSERT_EQ(bp.times.size(), (size_t)5);
    EXPECT_EQ(bp.times[2], 0L);          // no LSQR slot in init_only mode

    // final_relres must be x's (== x0's) own true relative residual
    // ||b - A x||/||b||, independently recomputed here.
    std::vector<T> Ax(m, 0);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x.data(), n, (T)0.0, Ax.data(), m);
    T num = 0;
    for (int64_t i = 0; i < m; ++i) { T d = b[i] - Ax[i]; num += d * d; }
    T recomputed_relres = std::sqrt(num) / blas::nrm2(m, b.data(), 1);
    EXPECT_NEAR(bp.final_relres, recomputed_relres, 1e-10 * recomputed_relres + 1e-14);
}


// Warm-path skip: when the sketch-and-solve x0 already meets the caller's tol
// on the TRUE residual, LSQR must not run at all. A consistent, well-conditioned
// system makes this the common case, not a corner case: Sb = Ask x_true exactly,
// so the sketched problem's unique least-squares solution IS x_true (up to
// rounding) whenever Ask has full column rank, regardless of sketch quality:
// ||b - A x0|| sits at the rounding floor, far below any ordinary tol.
TEST_F(TestLSQRBlendenpik, blendenpik_warm_start_skips_lsqr_when_x0_meets_tol) {
    using T = double;
    int64_t m = 400, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 113, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(127);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, (T)1e-6);
    bp.max_iters = 2000;
    std::vector<T> x(n, (T)0);
    int status = bp.call(Aop, b.data(), m, x.data(), n, /*d_factor=*/4.0, state);

    ASSERT_EQ(status, 0);
    EXPECT_EQ(bp.lsqr_iters, 0);
    EXPECT_TRUE(bp.converged);
    EXPECT_EQ(bp.lsqr_stop_test, 1);     // S1 (residual) already satisfied by x0
    EXPECT_GE(bp.final_relres, (T)0);
    EXPECT_LT(bp.final_relres, (T)1e-6);
    EXPECT_LT(rel_err(x, x_ref), 1e-7);
    ASSERT_EQ(bp.lsqr_op_times.size(), (size_t)4);
    for (long t : bp.lsqr_op_times) EXPECT_EQ(t, 0L);
}


// Repeated call() on ONE Blendenpik object across different problems: every call
// must succeed on its own terms (pins the top-of-call() output reset alongside
// R_out's own delete[]/reassign, the no-leak contract for R_out that mirrors
// repeated_call_does_not_leak_Q for CholQR_linops's Q).
TEST_F(TestLSQRBlendenpik, blendenpik_repeated_call_is_self_consistent) {
    using T = double;
    int64_t m = 300, n = 16;
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, (T)1e-12);
    bp.max_iters = 2000;

    const T* R_out_call0 = nullptr;
    for (int rep = 0; rep < 3; ++rep) {
        std::vector<T> A, b, x_true;
        make_problem(m, n, 131 + rep, A, b, x_true);
        auto x_ref = gels_reference(A, b, m, n);
        DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
        RandBLAS::RNGState<RNG> state(137 + rep);
        std::vector<T> x(n, (T)0);

        ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), 0) << "rep " << rep;
        EXPECT_TRUE(bp.converged) << "rep " << rep;
        ASSERT_NE(bp.R_out, nullptr) << "rep " << rep;
        EXPECT_EQ(bp.R_out_sz, n * n) << "rep " << rep;
        EXPECT_LT(rel_err(x, x_ref), 1e-7) << "rep " << rep;
        if (rep == 0) R_out_call0 = bp.R_out;
    }
    // Not a strong free-proof (the allocator may legitimately reuse the same
    // address), but call()'s `delete[] R_out;` at the top of every call is the
    // actual no-double-free/no-leak contract this loop exercises under repeated use.
    (void)R_out_call0;
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
        bp.max_iters = 2000;
        bp.warm_start = false;
        ASSERT_EQ(bp.call(Aop, b.data(), m, x_cold.data(), n, 4.0, state), 0);
        iters_cold = bp.lsqr_iters;
    }
    {
        RandBLAS::RNGState<RNG> state(31);   // same seed => same sketch => same R
        RandLAPACK::Blendenpik_linops<T, RNG> bp(true, (T)1e-13);
        bp.max_iters = 2000;
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
// buffer: `this->Q = new T[m*n]` must delete the previous pointer first, or every
// extra call() leaks an m x n block. Under ASan/valgrind this test would fail
// outright on code missing that delete; without a sanitizer it still pins the
// contract that Q is re-materialized and remains correct across repeated calls.
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
// the second solver of the reference benchmark (solve_with_lsqr's sibling in
// ar_sysid_toeplitz_qless_qr_benchmark.m).
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
    EXPECT_EQ(st_one, 3);             // rounds budget exhausted (status 3), not converged
    EXPECT_EQ(restarts_one, 1);       // exactly one round ran
    EXPECT_LT(iters_one, iters_free); // strictly less work than the free run
    EXPECT_GT(relres_one, 1e-12);     // and the tolerance was honestly not met
}


// Structure unification: restarted_pcg_ne's inner CG must carry the
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
    // finite-precision floor, the stagnation regime this test pins.
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
    EXPECT_EQ(st, 4);                 // tolerance genuinely unreachable: LS-floor exit
    // Outer stagnation exit: two consecutive rounds without meaningful
    // true-residual improvement end the loop BEFORE the round budget; the
    // LS floor is O(1) here, so rounds 2+ cannot help.
    EXPECT_LT(restarts, 4);
    EXPECT_GT(restarts, 1);           // but the flatline needed two rounds to detect
    EXPECT_LT(iters, 800);            // stagnation exits: strictly under the full budget
    EXPECT_GT(iters, 0);
    EXPECT_TRUE(std::isfinite((double)relres));
}


// Warm start. restarted_pcg_ne can refine an EXISTING iterate rather
// than starting from x = 0, so a solver's own answer (Blendenpik's, say) can be fed
// to iterative refinement. Three properties pinned:
//   1. An already-converged x0 is recognized immediately: no work, and it is not
//      made worse.
//   2. A deliberately perturbed x0 is refined back to the reference solution.
//   3. Warm and cold start agree on the final answer (same fixed point).
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_warm_start_refines_an_existing_iterate) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 71, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);

    std::vector<T> Ac(A), tau(n), R(n * n, 0);
    lapack::geqrf(m, n, Ac.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, Ac.data(), m, R.data(), n);
    if (n > 1)
        lapack::laset(lapack::MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R.data() + 1, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    // (1) cold solve to convergence, then warm-restart FROM that answer.
    std::vector<T> x_cold(n, 0);
    int it_c = 0;
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(),
                  x_cold.data(), 1e-12, 2000, it_c), 0);
    std::vector<T> x_warm(x_cold);
    int it_w = 0;
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(),
                  x_warm.data(), 1e-12, 2000, it_w, 200, (T)1e-4, -1,
                  nullptr, nullptr, nullptr, 20, (T)1e-3, (T)0, nullptr,
                  /*x0=*/x_cold.data()), 0);
    EXPECT_EQ(it_w, 0) << "an already-converged x0 must cost no inner iterations";
    EXPECT_LT(rel_err(x_warm, x_ref), 1e-8);

    // (2) perturb x0 substantially; refinement must recover the reference.
    std::vector<T> x_bad(x_ref);
    for (int64_t i = 0; i < n; ++i) x_bad[i] *= (T)1.5;
    std::vector<T> x_fixed(n, 0);
    int it_f = 0;
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(),
                  x_fixed.data(), 1e-12, 2000, it_f, 200, (T)1e-4, -1,
                  nullptr, nullptr, nullptr, 20, (T)1e-3, (T)0, nullptr,
                  /*x0=*/x_bad.data()), 0);
    EXPECT_GT(it_f, 0) << "a perturbed x0 must require real work";
    EXPECT_LT(rel_err(x_fixed, x_ref), 1e-8);

    // (3) warm and cold reach the same fixed point.
    EXPECT_LT(rel_err(x_fixed, x_cold), 1e-8);
}


// Round-residual stability: the reference MATLAB recomputes the
// normal-equation residual as g - H z, a difference of two large kappa-contaminated
// quantities whose cancellation error floors the achievable accuracy. Epperly et al.
// Alg. 1 line 5 computes the same quantity stably from the SMALL true LS residual,
// R^{-T}(A^T(b - A x)). On a consistent ill-conditioned problem with a healthy
// preconditioner, the solver must therefore reach a tight LS tolerance instead of
// stalling at an eps*kappa floor.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_reaches_tight_tol_on_ill_conditioned) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 67, A, b, x_true);
    for (int64_t j = 0; j < n; ++j) {
        T s = std::pow((T)10.0, (T)(-11.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    // Recompute b from the SCALED A so the system is exactly consistent: the
    // optimal residual is 0 and only numerical floors can stop the solver.
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);

    std::vector<T> Ac(A), tau(n), R(n * n, 0);
    lapack::geqrf(m, n, Ac.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, Ac.data(), m, R.data(), n);
    if (n > 1)
        lapack::laset(lapack::MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R.data() + 1, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0.0);
    int iters = 0, restarts = 0;
    T relres = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                             /*tol=*/1e-12, /*max_iters=*/2000, iters,
                                             /*restart_maxit=*/500, /*restart_drop=*/1e-2,
                                             /*max_restarts=*/3,
                                             &restarts, nullptr, &relres);
    EXPECT_EQ(st, 0);                 // must actually converge, not stall at a floor
    EXPECT_LE(relres, 1e-12);
}


// Round-residual regression floor, the PROLATE regime. NOTE on what this test
// can and cannot see: the g - H z vs R^{-T}(A^T(b - A x)) residual-form
// difference that floors the FFT-operator benchmark at 1.75e-6 (see
// rl_restarted_pcg_ne.hh) is NOT reproduced by this dense replica: both forms
// pass it, so the FFT apply's rounding is a necessary ingredient and the m=800
// benchmark case is the discriminating harness. This test still pins a floor no
// implementation may regress: augmented prolate (numerically rank-deficient, tiny
// sqrt(lambda) identity rows, noisy data) must converge to 1e-9 from a shifted
// CholQR-style factor.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_stable_residual_converges_on_prolate) {
    using T = double;
    // The exact measured configuration: T = 800 x 400 prolate (omega 0.14),
    // augmented A = [T; sqrt(lambda) I] with lambda_rel = 1e-20, rhs = [b; 0],
    // b = T x_true + 1e-11 relative noise. The tiny sqrt(lambda) identity rows are
    // what push R's trailing diagonal small enough for the g - H z cancellation to
    // bind; the smaller unaugmented prolate does NOT reproduce the stall.
    int64_t mt = 800, n = 400, m = mt + n;
    const double omega = 0.14;
    auto prolate = [&](int64_t k) -> T {
        if (k == 0) return (T)(2.0 * omega);
        return (T)(std::sin(2.0 * M_PI * omega * (double)k) / (M_PI * (double)k));
    };
    std::vector<T> A(m * n, 0.0);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < mt; ++i)
            A[i + j * m] = prolate(i - j);
    const T sqrt_lambda = (T)1e-10;   // lambda = 1e-20, sigma_max(T) ~ 1
    for (int64_t j = 0; j < n; ++j) A[(mt + j) + j * m] = sqrt_lambda;
    // Deterministic smooth x_true and deterministic 1e-11 relative "noise".
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

    // CholQR-style R with an escalating diagonal shift (the benchmark's approach for
    // numerically rank-deficient Grams). A weak R is fine: the measured benchmark run
    // showed even CholQR's 0.83-orthogonality factor reaching the floor with the
    // stable residual form.
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
    std::vector<T> x(n, 0.0);
    int iters = 0, restarts = 0;
    T relres = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                             /*tol=*/1e-9, /*max_iters=*/2000, iters,
                                             /*restart_maxit=*/500, /*restart_drop=*/1e-2,
                                             /*max_restarts=*/3,
                                             &restarts, nullptr, &relres);
    EXPECT_EQ(st, 0);                 // reaches tol instead of stalling near 1e-6
    EXPECT_LE(relres, 1e-9);
}


// status 2 (breakdown): "R unusable". H = R^{-T} A^T A R^{-1} = (A R^{-1})^T (A R^{-1})
// is a Gram matrix for ANY invertible R, so it is mathematically PSD regardless of
// R's diagonal signs: a sign-flipped diagonal entry on an otherwise-valid Cholesky
// factor cannot make it indefinite. What genuinely breaks the COMPUTED apply is an
// astronomically ill-conditioned (but still nonzero) R: the two trsv calls straddling
// A/A^T amplify by ~1/R[k,k] on the way in AND ~1/R[k,k] on the way out, so a single
// tiny diagonal entry overflows the round trip to inf/NaN.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_reports_breakdown_on_unusable_R) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 151, A, b, x_true);

    std::vector<T> Ac(A), tau(n), R(n * n, 0);
    lapack::geqrf(m, n, Ac.data(), m, tau.data());
    lapack::lacpy(lapack::MatrixType::Upper, n, n, Ac.data(), m, R.data(), n);
    if (n > 1)
        lapack::laset(lapack::MatrixType::Lower, n - 1, n - 1, (T)0, (T)0, R.data() + 1, n);
    // Corrupt only the LAST diagonal entry: tiny but nonzero. trsv on z = 0 (the
    // cold-start recovery of round 1) stays an exact 0/anything = 0, so x recovery
    // is unaffected; trsv on the genuinely nonzero CG direction p overflows.
    R[(n - 1) + (n - 1) * n] = (T)1e-300;

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, (T)999);   // sentinel; must come back finite
    int iters = 0;
    T relres = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, R.data(), n, b.data(), x.data(),
                                             /*tol=*/1e-10, /*max_iters=*/2000, iters,
                                             /*restart_maxit=*/200, /*restart_drop=*/1e-4,
                                             /*max_restarts=*/-1,
                                             /*restarts_done=*/nullptr, /*times=*/nullptr,
                                             &relres);

    EXPECT_EQ(st, 2);   // breakdown: R unusable, not a hard crash
    for (int64_t i = 0; i < n; ++i)
        EXPECT_TRUE(std::isfinite((double)x[i])) << "component " << i;
    // Breakdown fires on round 1's very first CG direction, before any progress:
    // the returned iterate is still the initial cold-start x = 0 exactly, and the
    // reported relres matches an independent recomputation of ||b - A x||/||b||.
    for (int64_t i = 0; i < n; ++i) EXPECT_EQ(x[i], (T)0);
    T bnorm = blas::nrm2(m, b.data(), 1);
    EXPECT_NEAR(relres, (T)1, 1e-12);
    (void)bnorm;
}


// inner_abs_tol guard: once a round's NE residual has already fallen to
// inner_abs_tol * ||g||, that round's effective target loosens to the absolute
// floor instead of grinding out the full restart_drop factor. Round 1's NE
// residual starts at EXACTLY ||g|| (cold start), so floor_rel == inner_abs_tol
// there: with inner_abs_tol < restart_drop, round 1 is provably unaffected.
// Round 2 starts from round 1's SHRUNKEN residual, so the same inner_abs_tol
// maps to a proportionally looser floor_rel there, large enough, with this
// problem's conditioning, to bind. Both runs are capped at max_restarts=1 (two
// rounds) so the comparison isolates the guard's per-round mechanism instead of
// depending on how many rounds a full solve to tight tol would need.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_inner_abs_tol_guard_shortens_later_rounds) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 163, A, b, x_true);
    // Mild ill-conditioning so round 1 alone cannot reach a tight restart_drop
    // target in a handful of iterations (more than one round is genuinely
    // needed), without pushing into the stagnation/LS-floor regime.
    for (int64_t j = 0; j < n; ++j) {
        T s = std::pow((T)10.0, (T)(-4.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    T tol = (T)1e-9, restart_drop = (T)1e-2;
    RandLAPACK::PCGRoundHistory<T> hist_base, hist_guard;

    std::vector<T> x_base(n, 0);
    int iters_base = 0;
    RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_base.data(),
                                    tol, 5000, iters_base,
                                    /*restart_maxit=*/500, restart_drop,
                                    /*max_restarts=*/1,
                                    nullptr, nullptr, nullptr,
                                    /*stag_window=*/20, /*stag_rel_improve=*/(T)1e-3,
                                    /*inner_abs_tol=*/(T)0, &hist_base);
    ASSERT_EQ(hist_base.iters.size(), (size_t)2);   // both budgeted rounds ran

    std::vector<T> x_guard(n, 0);
    int iters_guard = 0;
    RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x_guard.data(),
                                    tol, 5000, iters_guard,
                                    /*restart_maxit=*/500, restart_drop,
                                    /*max_restarts=*/1,
                                    nullptr, nullptr, nullptr,
                                    /*stag_window=*/20, /*stag_rel_improve=*/(T)1e-3,
                                    /*inner_abs_tol=*/(T)1e-4, &hist_guard);
    ASSERT_EQ(hist_guard.iters.size(), (size_t)2);

    // Round 1 (index 0): unaffected, same inner iteration count either way.
    EXPECT_EQ(hist_guard.iters[0], hist_base.iters[0]);
    // Round 2 (index 1): the guard's inner CG hits ITS OWN loosened absolute
    // floor and reports Converged there, using strictly fewer iterations than
    // the unguarded round needed to reach the tighter restart_drop target.
    EXPECT_EQ(hist_guard.status[1], static_cast<int>(RandLAPACK::InnerCGStatus::Converged));
    EXPECT_LT(hist_guard.iters[1], hist_base.iters[1]);
}


// outer_stag_window <= 0 disables the outer LS-floor exit: on the SAME problem
// that restarted_pcg_ne_stagnation_exits_early shows exiting early (status 4,
// restarts < 4), disabling the guard must instead run every round up to
// max_restarts and report the round-budget-exhausted status.
TEST_F(TestLSQRBlendenpik, restarted_pcg_ne_outer_stag_window_disabled_runs_to_round_cap) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 61, A, b, x_true);
    for (int64_t j = 0; j < n; ++j) {
        T s = std::pow((T)10.0, (T)(-8.0 * (double)j / (n - 1)));
        blas::scal(m, s, A.data() + j * m, 1);
    }
    for (int64_t i = 0; i < m; ++i) b[i] += (T)0.1 / (T)(1 + i);
    std::vector<T> x(n, 0.0);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    int iters = 0, restarts = 0;
    T relres = -1;
    int st = RandLAPACK::restarted_pcg_ne<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                             /*tol=*/1e-15, /*max_iters=*/2000, iters,
                                             /*restart_maxit=*/200, /*restart_drop=*/1e-30,
                                             /*max_restarts=*/3,
                                             &restarts, nullptr, &relres,
                                             /*stag_window=*/20, /*stag_rel_improve=*/(T)1e-3,
                                             /*inner_abs_tol=*/(T)0, /*history=*/nullptr,
                                             /*x0=*/nullptr, /*outer_stag_window=*/0);
    EXPECT_EQ(st, 3);             // round budget exhausted, not the LS-floor exit
    EXPECT_EQ(restarts, 4);       // all max_restarts+1 rounds ran
    EXPECT_TRUE(std::isfinite((double)relres));
}
