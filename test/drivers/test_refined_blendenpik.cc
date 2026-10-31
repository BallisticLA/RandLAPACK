// Tests for RandLAPACK::bench::run_refined_blendenpik (benchmark/bench_CQRRTO_linops/
// refined_blendenpik.hh), the Blendenpik_refine and Blendenpik_cold_refine rows of the
// least-squares benchmarks, run here with IterRefineLSQ as the FEM benchmark runs them.

#include "refined_blendenpik.hh"

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include "../lsq_test_helpers.hh"

#include <cmath>
#include <vector>

using RandLAPACK::linops::DenseLinOp;
using RandLAPACK::IterRefineLSQ;
using Term = RandLAPACK::RestartedPCGNETermination;
using blas::Layout;

class TestRunRefinedBlendenpik : public ::testing::Test {
protected:
    using RNG = r123::Philox4x32;

    template <typename T>
    static IterRefineLSQ<T> refinement(T tol) {
        IterRefineLSQ<T> ir((T)0, /*max_inner=*/200, /*n_steps=*/20);
        ir.outer_tol = tol;
        return ir;
    }

    // One refined row solved by ir into x.
    template <typename T>
    static RandLAPACK::bench::RefinedBlendenpikResult<T>
    row(DenseLinOp<T>& A, const std::vector<T>& b, std::vector<T>& x, uint32_t seed, bool warm,
        IterRefineLSQ<T>& ir) {
        const int64_t m = A.n_rows, n = A.n_cols;
        return RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
            A, b.data(), m, n, (T)4.0, 4, RandBLAS::RNGState<RNG>(seed), warm,
            [&](const T* R, const T* x0) { ir.call(A, R, n, b.data(), m, x.data(), n, x0); });
    }
};


// Same seed, same sketch: the warm and cold rows share R exactly, and only the warm row
// reports the x0 build and its residual.
TEST_F(TestRunRefinedBlendenpik, warm_and_cold_share_R_and_only_warm_reports_x0) {
    using T = double;
    int64_t m = 400, n = 20;
    std::vector<T> A, b, x_true;
    lsq_test::make_problem<T>(m, n, 401, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x_warm(n, 0), x_cold(n, 0);
    auto ir_w = refinement<T>(1e-10), ir_c = refinement<T>(1e-10);
    auto res_warm = row<T>(Aop, b, x_warm, 403, true, ir_w);
    auto res_cold = row<T>(Aop, b, x_cold, 403, false, ir_c);

    ASSERT_EQ(res_warm.qr_status, 0);
    ASSERT_EQ(res_cold.qr_status, 0);
    ASSERT_NE(res_warm.R, nullptr);
    ASSERT_NE(res_cold.R, nullptr);
    for (int64_t i = 0; i < n * n; ++i)
        EXPECT_EQ(res_warm.R[i], res_cold.R[i]) << "R entry " << i << " differs";
    EXPECT_GE(res_warm.qr_us, 0);
    EXPECT_EQ(res_cold.setup_us, 0);
    EXPECT_EQ(res_cold.x0_relres, (T)-1);
    EXPECT_GE(res_warm.x0_relres, (T)0);
    EXPECT_LT(res_warm.x0_relres, (T)1e-8);   // consistent problem: x0 is nearly exact
}


// The warm row hands the sketch-and-solve x0 to the solver: on a consistent problem it already
// meets the tolerance, so the warm row runs no round, while the cold row has to iterate.
TEST_F(TestRunRefinedBlendenpik, warm_row_starts_the_solver_from_x0) {
    using T = double;
    int64_t m = 400, n = 20;
    std::vector<T> A, b, x_true;
    lsq_test::make_problem<T>(m, n, 409, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    std::vector<T> x_warm(n, 0), x_cold(n, 0);
    const T tol = 1e-10;
    auto ir_w = refinement<T>(tol), ir_c = refinement<T>(tol);
    ASSERT_EQ(row<T>(Aop, b, x_warm, 419, true, ir_w).qr_status, 0);
    ASSERT_EQ(row<T>(Aop, b, x_cold, 419, false, ir_c).qr_status, 0);

    EXPECT_EQ(ir_w.termination_reason, Term::converged);
    EXPECT_EQ(ir_c.termination_reason, Term::converged);
    EXPECT_EQ(ir_w.report.rounds_done, 0);
    EXPECT_LT(ir_w.report.iters_done, ir_c.report.iters_done);
    EXPECT_LE(ir_w.report.final_relres, tol);
    EXPECT_LE(ir_c.report.final_relres, tol);
}


// On an inconsistent problem both rows refine to the least-squares solution: checked against
// gels and by the normal-equation optimality of the returned x, in double and float.
template <typename T>
static void check_inconsistent_refinement(T tol_x, T tol_ne) {
    using RNG = r123::Philox4x32;
    int64_t m = 300, n = 15;
    std::vector<T> A, b, x_true, noise(m);
    lsq_test::make_problem<T>(m, n, 431, A, b, x_true);
    lsq_test::fill_random<T>(noise, 433, (T)0.5);
    for (int64_t i = 0; i < m; ++i) b[i] += noise[i];
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    const std::vector<T> x_ref = lsq_test::gels_reference(A, b, m, n);
    const T A_fro = lapack::lange(lapack::Norm::Fro, m, n, A.data(), m);

    for (bool warm : {true, false}) {
        IterRefineLSQ<T> ir((T)0, 200, 30);
        ir.outer_tol = (T)0;   // no early exit: the run ends at the least-squares floor
        std::vector<T> x(n, 0);
        auto res = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
            Aop, b.data(), m, n, (T)4.0, 4, RandBLAS::RNGState<RNG>(437), warm,
            [&](const T* R, const T* x0) { ir.call(Aop, R, n, b.data(), m, x.data(), n, x0); });
        ASSERT_EQ(res.qr_status, 0);
        EXPECT_GT(ir.report.rounds_done, 0) << "warm=" << warm;
        EXPECT_LE(lsq_test::rel_err(x, x_ref), tol_x) << "warm=" << warm;
        std::vector<T> r(b), ATr(n);
        blas::gemv(Layout::ColMajor, blas::Op::NoTrans, m, n, (T)-1, A.data(), m, x.data(), 1, (T)1, r.data(), 1);
        blas::gemv(Layout::ColMajor, blas::Op::Trans, m, n, (T)1, A.data(), m, r.data(), 1, (T)0, ATr.data(), 1);
        EXPECT_LE(blas::nrm2(n, ATr.data(), 1) / (A_fro * blas::nrm2(m, r.data(), 1)), tol_ne) << "warm=" << warm;
    }
}

TEST_F(TestRunRefinedBlendenpik, inconsistent_problem_reaches_the_least_squares_solution) {
    check_inconsistent_refinement<double>(1e-10, 1e-12);
}

TEST_F(TestRunRefinedBlendenpik, inconsistent_problem_reaches_the_least_squares_solution_float) {
    check_inconsistent_refinement<float>(1e-4f, 1e-5f);
}


// The solver receives the result's R, and x0 only for the warm row.
TEST_F(TestRunRefinedBlendenpik, solver_gets_R_and_x0_only_when_warm) {
    using T = double;
    int64_t m = 200, n = 10;
    std::vector<T> A, b, x_true;
    lsq_test::make_problem<T>(m, n, 441, A, b, x_true);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);
    for (bool warm : {true, false}) {
        const T* seen_R = nullptr;
        const T* seen_x0 = nullptr;
        int calls = 0;
        auto res = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
            Aop, b.data(), m, n, (T)4.0, 4, RandBLAS::RNGState<RNG>(443), warm,
            [&](const T* R, const T* x0) {
                ++calls; seen_R = R; seen_x0 = x0;
                if (x0 != nullptr) EXPECT_LE(lsq_test::rel_err(std::vector<T>(x0, x0 + n), x_true), (T)1e-10);
            });
        ASSERT_EQ(res.qr_status, 0);
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(seen_R, res.R.get());
        EXPECT_EQ(seen_x0 != nullptr, warm);
    }
}


// A rank-deficient sketch factor fails phase 1, and the solver is not called.
TEST_F(TestRunRefinedBlendenpik, failed_build_skips_the_solver) {
    using T = double;
    int64_t m = 100, n = 8;
    std::vector<T> A(m * n, 0), b(m, 1);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n - 1; ++j) A[i + j * m] = (T)((i * 7 + j * 3) % 11) - 5;
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);   // last column zero
    int calls = 0;
    auto res = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
        Aop, b.data(), m, n, (T)4.0, 4, RandBLAS::RNGState<RNG>(461), true,
        [&](const T*, const T*) { ++calls; });
    EXPECT_NE(res.qr_status, 0);
    EXPECT_EQ(res.R, nullptr);
    EXPECT_EQ(res.qr_us, -1);
    EXPECT_EQ(calls, 0);
}
