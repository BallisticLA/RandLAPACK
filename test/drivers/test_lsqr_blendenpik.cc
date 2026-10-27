// Tests for RandLAPACK::lsqr (agreement with gels, every stopping path, float, large scale)
// and RandLAPACK::Blendenpik_linops (residual reporting, failure and reset, init_only, the
// warm start's skip and correction paths, sketch-size limits, float).

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
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
    int iters = 0, test = -1;
    T relres = -1;
    int st = RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                 1e-14, 1e-14, 2000, iters, nullptr, &relres, &test);
    EXPECT_EQ(st, 0);                 // met a stopping test
    EXPECT_EQ(test, 1);               // the residual test, inside the loop
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


// Blendenpik must solve the LS problem and report a real residual, not the -1 sentinel.
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
    bp.max_iters = 500;   // unused in init_only
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
    const T recomputed_relres = (T)RandLAPACK::testing::rel_fro_error(Ax.data(), b.data(), m);
    EXPECT_NEAR(bp.final_relres, recomputed_relres, 1e-10 * recomputed_relres + 1e-14);
}


// Warm-path skip: on a consistent system the sketch-and-solve x0 is x_true up to rounding,
// so it already meets an ordinary tol on the true residual and LSQR must not run.
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


// Repeated call() on one Blendenpik object across different problems: every call must
// succeed on its own terms (outputs are reset and R_out is replaced, not leaked).
TEST_F(TestLSQRBlendenpik, blendenpik_repeated_call_is_self_consistent) {
    using T = double;
    int64_t m = 300, n = 16;
    RandLAPACK::Blendenpik_linops<T, RNG> bp(/*time_subroutines=*/true, (T)1e-12);
    bp.max_iters = 2000;

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
    }
}


// The sketch-and-solve warm start must not change the answer of a cold start, nor cost more
// iterations. (Epperly/Meier/Nakatsukasa arXiv:2406.03468
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


// b == 0: x = 0 immediately, S1 satisfied, zero relative residual.
TEST_F(TestLSQRBlendenpik, lsqr_zero_rhs_returns_zero) {
    using T = double;
    int64_t m = 50, n = 5;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 41, A, b, x_true);
    std::fill(b.begin(), b.end(), (T)0);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, (T)7);
    int iters = -1, test = -1;
    T relres = -1;
    ASSERT_EQ(RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                  1e-12, 1e-12, 100, iters, nullptr, &relres, &test), 0);
    EXPECT_EQ(iters, 0);
    EXPECT_EQ(test, 1);
    EXPECT_EQ(relres, (T)0);
    for (T v : x) EXPECT_EQ(v, (T)0);
}

// b orthogonal to range(A): A^T b = 0, so x = 0 is the minimizer (S2 at once, relres 1).
TEST_F(TestLSQRBlendenpik, lsqr_rhs_orthogonal_to_range) {
    using T = double;
    int64_t m = 6, n = 2;
    std::vector<T> A(m * n, (T)0), b(m, (T)0);
    A[0] = 1; A[1 + m] = 1;   // columns e1, e2
    b[4] = 3;                 // b = 3 e5
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, (T)5);
    int iters = -1, test = -1;
    T relres = -1;
    ASSERT_EQ(RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                  1e-12, 1e-12, 100, iters, nullptr, &relres, &test), 0);
    EXPECT_EQ(iters, 0);
    EXPECT_EQ(test, 2);
    EXPECT_EQ(relres, (T)1);
    for (T v : x) EXPECT_EQ(v, (T)0);
}

// An inconsistent system cannot meet S1 at a tight btol; S2 must end the run at the LS
// solution, report stop_test 2, and fill the timing slots.
TEST_F(TestLSQRBlendenpik, lsqr_inconsistent_system_stops_on_s2) {
    using T = double;
    int64_t m = 200, n = 10;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 43, A, b, x_true);
    RandBLAS::RNGState<RNG> st(44);
    std::vector<T> noise(m);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), noise.data(), st);
    for (int64_t i = 0; i < m; ++i) b[i] += noise[i];
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0, test = -1;
    long times[4] = {-1, -1, -1, -1};
    ASSERT_EQ(RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                  1e-12, 1e-14, 500, iters, times, nullptr, &test), 0);
    EXPECT_EQ(test, 2);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
    for (long t : times) EXPECT_GE(t, 0L);
}

TEST_F(TestLSQRBlendenpik, lsqr_float) {
    using T = float;
    int64_t m = 120, n = 8;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 47, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    ASSERT_EQ(RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                  1e-6f, 1e-6f, 500, iters), 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-4f);
}

// A small problem whose sketch has fewer rows than the requested nonzeros per column: the
// sketch's nonzeros are capped at its row count instead of being rejected by RandBLAS.
TEST_F(TestLSQRBlendenpik, blendenpik_caps_sketch_nonzeros_on_small_problems) {
    using T = double;
    int64_t m = 40, n = 2;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 53, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(59);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-12);
    bp.max_iters = 200;
    bp.nnz = 8;
    std::vector<T> x(n, 0);
    ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, /*d_factor=*/1.5, state), 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-8);
}

// Entries near 1e20 in float: ||A||_F^2 overflows float, so an S2 test built on the
// squared norm would fire after one iteration far from the solution.
TEST_F(TestLSQRBlendenpik, lsqr_float_large_scale_does_not_stop_early) {
    using T = float;
    int64_t m = 2, n = 2;
    std::vector<T> A = {1e20f, 0.0f, 0.0f, 2e20f}, b = {1.0f, 1.0f};
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n, 0);
    int iters = 0;
    T relres = -1;
    ASSERT_EQ(RandLAPACK::lsqr<T>(Aop, m, n, nullptr, 0, b.data(), x.data(),
                                  1e-6f, 1e-6f, 10, iters, nullptr, &relres), 0);
    EXPECT_LT(relres, 1e-5f);
    EXPECT_NEAR(x[0] * 1e20f, 1.0f, 1e-4f);
    EXPECT_NEAR(x[1] * 1e20f, 0.5f, 1e-4f);
}

// An inconsistent, moderately ill-conditioned problem: the sketch-and-solve start does not
// meet a tight tolerance, so LSQR must iterate on the correction. The solution must match
// gels and final_relres must match the residual recomputed from x.
TEST_F(TestLSQRBlendenpik, blendenpik_warm_start_iterates_on_inconsistent_problem) {
    using T = double;
    int64_t m = 400, n = 20;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 61, A, b, x_true);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            A[i + j * m] *= std::pow((T)10, -(T)j / (T)4);   // column scaling, kappa ~ 1e5
    RandBLAS::RNGState<RNG> st(62);
    std::vector<T> noise(m);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), noise.data(), st);
    for (int64_t i = 0; i < m; ++i) b[i] += noise[i];
    auto x_ref = gels_reference(A, b, m, n);

    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(63);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-14);
    bp.max_iters = 500;
    std::vector<T> x(n, 0);
    ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), 0);
    EXPECT_GT(bp.lsqr_iters, 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-6);

    std::vector<T> r(b);
    blas::gemv(Layout::ColMajor, blas::Op::NoTrans, m, n, (T)-1.0, A.data(), m, x.data(), 1, (T)1.0, r.data(), 1);
    T true_relres = blas::nrm2(m, r.data(), 1) / blas::nrm2(m, b.data(), 1);
    EXPECT_NEAR(bp.final_relres, true_relres, 1e-6 * true_relres);
}

// A sketch taller than the operator (d = 4n > m = n): the nonzeros cap must keep the sketch
// valid and the square system must still be solved.
TEST_F(TestLSQRBlendenpik, blendenpik_sketch_taller_than_operator) {
    using T = double;
    int64_t m = 12, n = 12;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 67, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(68);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-12);
    bp.max_iters = 200;
    bp.nnz = 16;
    std::vector<T> x(n, 0);
    ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), 0);
    EXPECT_LT(rel_err(x, x_true), 1e-8);
}

// A failing call after a successful one must not leave the first call's outputs behind.
TEST_F(TestLSQRBlendenpik, blendenpik_failure_after_success_resets_outputs) {
    using T = double;
    int64_t m = 200, n = 10;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 71, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(true, (T)1e-12);
    bp.max_iters = 200;
    std::vector<T> x(n, 0);
    RandBLAS::RNGState<RNG> state(72);
    ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), 0);
    ASSERT_NE(bp.R_out, nullptr);

    for (int64_t i = 0; i < m; ++i) A[i] = (T)0;   // zero first column: rank-deficient sketch
    std::fill(x.begin(), x.end(), (T)42);
    ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), 1);
    for (T v : x) EXPECT_EQ(v, (T)42) << "x must be untouched on failure";
    EXPECT_EQ(bp.R_out, nullptr);
    EXPECT_EQ(bp.R_out_sz, 0);
    EXPECT_FALSE(bp.converged);
    EXPECT_EQ(bp.final_relres, (T)-1);
    EXPECT_EQ(bp.lsqr_iters, 0);
    EXPECT_TRUE(bp.times.empty());
}

TEST_F(TestLSQRBlendenpik, blendenpik_rejects_invalid_sketch_factor) {
    using T = double;
    int64_t m = 50, n = 5;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 73, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-12);
    bp.max_iters = 10;
    std::vector<T> x(n, 0);
    RandBLAS::RNGState<RNG> state(74);
    EXPECT_THROW(bp.call(Aop, b.data(), m, x.data(), n, (T)0, state), RandLAPACK::Error);
    EXPECT_THROW(bp.call(Aop, b.data(), m, x.data(), n, std::numeric_limits<T>::infinity(), state), RandLAPACK::Error);
}

TEST_F(TestLSQRBlendenpik, blendenpik_float) {
    using T = float;
    int64_t m = 300, n = 12;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 77, A, b, x_true);
    auto x_ref = gels_reference(A, b, m, n);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    RandBLAS::RNGState<RNG> state(78);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-6);
    bp.max_iters = 200;
    std::vector<T> x(n, 0);
    ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0f, state), 0);
    EXPECT_TRUE(bp.converged);
    EXPECT_LT(rel_err(x, x_ref), 1e-4f);
}

TEST_F(TestLSQRBlendenpik, dimension_and_setup_checks_throw) {
    using T = double;
    int64_t m = 40, n = 4;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 81, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x(n + 1, 0);
    int iters = 0;
    EXPECT_THROW(RandLAPACK::lsqr<T>(Aop, m - 1, n, nullptr, 0, b.data(), x.data(), 1e-8, 1e-8, 10, iters), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::lsqr<T>(Aop, m, n + 1, nullptr, 0, b.data(), x.data(), 1e-8, 1e-8, 10, iters), RandLAPACK::Error);
    RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-8);
    RandBLAS::RNGState<RNG> state(82);
    EXPECT_THROW(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), RandLAPACK::Error);   // max_iters unset
    bp.max_iters = 10;
    EXPECT_THROW(bp.call(Aop, b.data(), m, x.data(), n + 1, 4.0, state), RandLAPACK::Error);
}

// b == 0 on the cold, warm (skip) and init_only paths: x = 0 and a zero relative residual.
TEST_F(TestLSQRBlendenpik, blendenpik_zero_rhs_on_every_path) {
    using T = double;
    int64_t m = 60, n = 5;
    std::vector<T> A, b, x_true;
    make_problem(m, n, 83, A, b, x_true);
    std::fill(b.begin(), b.end(), (T)0);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    for (int mode = 0; mode < 3; ++mode) {
        RandLAPACK::Blendenpik_linops<T, RNG> bp(false, (T)1e-10);
        bp.max_iters = 50;
        bp.warm_start = (mode != 0);
        bp.init_only  = (mode == 2);
        RandBLAS::RNGState<RNG> state(84);
        std::vector<T> x(n, (T)9);
        ASSERT_EQ(bp.call(Aop, b.data(), m, x.data(), n, 4.0, state), 0) << "mode " << mode;
        EXPECT_EQ(bp.final_relres, (T)0) << "mode " << mode;
        for (T v : x) EXPECT_EQ(v, (T)0) << "mode " << mode;
    }
}
