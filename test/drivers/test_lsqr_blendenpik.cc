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
