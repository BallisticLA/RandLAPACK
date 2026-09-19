// Unit tests for RandLAPACK::bench::run_refined_blendenpik (benchmark/refined_blendenpik.hh),
// the shared dispatch used by both the FEM2 and Toeplitz benchmarks for the
// Blendenpik_refine / Blendenpik_cold_refine rows.
//
// Pins the accounting contract documented at the top of refined_blendenpik.hh: warm
// and cold rows share the SAME Blendenpik phase 1 (same seed => identical R), the
// cold row's setup_us/x0_relres stay at their documented zero/sentinel values, and
// the warm row needs no more engine iterations than the cold row on a well-
// conditioned, consistent problem.

#include "benchmark/refined_blendenpik.hh"

#include "RandLAPACK.hh"
#include <RandBLAS.hh>
#include <gtest/gtest.h>

#include <vector>

using RandLAPACK::linops::DenseLinOp;
using blas::Layout;


class TestRunRefinedBlendenpik : public ::testing::Test {
protected:
    using RNG = r123::Philox4x32;
    using T = double;

    // Well-conditioned, consistent problem (b = A * x_true exactly).
    static void make_problem(int64_t m, int64_t n, uint32_t seed,
                             std::vector<T>& A, std::vector<T>& b) {
        A.resize(m * n); b.resize(m);
        std::vector<T> x_true(n);
        RandBLAS::RNGState<RNG> state(seed);
        RandBLAS::DenseDist DA(m, n);
        auto s2 = RandBLAS::fill_dense(DA, A.data(), state);
        RandBLAS::DenseDist DX(n, 1);
        RandBLAS::fill_dense(DX, x_true.data(), s2);
        for (int64_t j = 0; j < n; ++j) A[j + j * m] += (T)n;
        blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   m, 1, n, (T)1.0, A.data(), m, x_true.data(), n, (T)0.0, b.data(), m);
    }
};


// Warm and cold rows must share an identical R (same seed => same sketch => same
// preconditioner), and the cold row's phase-1 accounting sentinels (setup_us == 0,
// x0_relres == -1) must hold even though its Blendenpik phase runs bit-identically
// to the warm row's under the hood.
TEST_F(TestRunRefinedBlendenpik, warm_and_cold_share_identical_R_and_cold_sentinels_hold) {
    int64_t m = 400, n = 20;
    std::vector<T> A, b;
    make_problem(m, n, 401, A, b);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    std::vector<T> x_warm(n, 0), x_cold(n, 0);
    RandBLAS::RNGState<RNG> state_warm(403), state_cold(403);   // identical seed

    auto res_warm = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
        Aop, b.data(), m, x_warm.data(), n, /*d_factor=*/4.0, /*sketch_nnz=*/4,
        state_warm, /*warm=*/true,
        /*tol=*/1e-10, /*max_iters=*/2000,
        /*restart_maxit=*/200, /*restart_drop=*/1e-4, /*max_restarts=*/20);
    auto res_cold = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
        Aop, b.data(), m, x_cold.data(), n, /*d_factor=*/4.0, /*sketch_nnz=*/4,
        state_cold, /*warm=*/false,
        /*tol=*/1e-10, /*max_iters=*/2000,
        /*restart_maxit=*/200, /*restart_drop=*/1e-4, /*max_restarts=*/20);

    ASSERT_EQ(res_warm.qr_status, 0);
    ASSERT_EQ(res_cold.qr_status, 0);
    ASSERT_NE(res_warm.R, nullptr);
    ASSERT_NE(res_cold.R, nullptr);
    ASSERT_EQ(res_warm.R_sz, n * n);
    ASSERT_EQ(res_cold.R_sz, n * n);
    for (int64_t i = 0; i < n * n; ++i)
        EXPECT_EQ(res_warm.R[i], res_cold.R[i]) << "R entry " << i << " differs";

    EXPECT_EQ(res_cold.setup_us, 0);          // cold row: x0 build excluded from accounting
    EXPECT_EQ(res_cold.x0_relres, (T)-1);     // cold row: sentinel, x0 was never handed off

    EXPECT_GE(res_warm.setup_us, 0);          // warm row: real x0 build ran and was timed
    EXPECT_GE(res_warm.x0_relres, (T)0);      // warm row: real residual, not a sentinel
}


// The engine's inner-CG iteration count must not increase from starting closer to
// the answer: warm (starts from the sketch-and-solve x0) needs no more iterations
// than cold (starts from zero) on the same well-conditioned, consistent problem.
TEST_F(TestRunRefinedBlendenpik, warm_needs_no_more_iters_than_cold) {
    int64_t m = 400, n = 20;
    std::vector<T> A, b;
    make_problem(m, n, 409, A, b);
    DenseLinOp<T> Aop(m, n, A.data(), m, Layout::ColMajor);

    std::vector<T> x_warm(n, 0), x_cold(n, 0);
    RandBLAS::RNGState<RNG> state_warm(419), state_cold(419);

    T tol = 1e-11;
    auto res_warm = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
        Aop, b.data(), m, x_warm.data(), n, 4.0, 4, state_warm, true,
        tol, 2000, 200, 1e-4, 20);
    auto res_cold = RandLAPACK::bench::run_refined_blendenpik<T, RNG>(
        Aop, b.data(), m, x_cold.data(), n, 4.0, 4, state_cold, false,
        tol, 2000, 200, 1e-4, 20);

    ASSERT_EQ(res_warm.qr_status, 0);
    ASSERT_EQ(res_cold.qr_status, 0);
    ASSERT_EQ(res_warm.status, 0);    // both must actually reach tol
    ASSERT_EQ(res_cold.status, 0);
    EXPECT_LE(res_warm.iters, res_cold.iters);
    EXPECT_LE(res_warm.solver_relres, tol);
    EXPECT_LE(res_cold.solver_relres, tol);
}
