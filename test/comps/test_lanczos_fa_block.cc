#include "rl_lanczos_fa_block.hh"
#include "lanczos_test_support.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace linops = RandLAPACK::linops;

class TestBlockLanczosFA : public RandLAPACK::testing::LanczosTestSupport {};

// ===== Block oracles vs the exact oracle ====================================
// BlockQFAmatchesBlockFA compares two consumers of the SAME block tridiagonal,
// so an error confined to T cancels there identically. These independent
// full-depth tests discriminate: at full block Krylov depth (d*s == n) the FA
// vector output here and the QFA quadratic form in test_lanczos_qfa_block.cc
// must reproduce the EXACT oracle to
// roundoff, and any defect in the recurrence, the T assembly, or the
// eigendecomposition surfaces directly.
TEST_F(TestBlockLanczosFA, BlockFAMatchesExactAtFullDepth) {
    using T = double;
    const int64_t n = 36, s = 3, d = 12;   // d*s == n: invariant Krylov space

    T *G0 = randn<T>(n, n, /*seed=*/101);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/103);
    auto fscalar = [](T x) { return std::sqrt(x); };

    auto exact = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    T *ref = new T[n * s];
    exact(n, s, Bmat, ref);                       // ref = f(A)·B
    // FA at full depth.
    RandLAPACK::BlockLanczosFA<T> fa;
    T *Gout = new T[n * s];
    fa.call(A_op, Bmat, n, s, fscalar, d, Gout);
    T maxfa = 0, sclfa = 0;
    for (int64_t e = 0; e < n * s; ++e) {
        maxfa = std::max(maxfa, std::abs(Gout[e] - ref[e]));
        sclfa = std::max(sclfa, std::abs(ref[e]));
    }
    EXPECT_LT(maxfa / sclfa, 1e-10);

    delete[] G0; delete[] A; delete[] Bmat; delete[] ref;
    delete[] Gout;
}

// ===== BlockLanczosFA: early-stopped recurrence evaluates at the run depth ==
// When a stop_after callback ends the recurrence at k < d, the interrupted
// step has already written its off-diagonal block, so evaluating at the full
// d would read a corrupted tail. apply_f must evaluate at steps_run and match
// a fresh full run at that depth.
TEST_F(TestBlockLanczosFA, BlockFAearlyStopAppliesAtRunDepth) {
    using T = double;
    const int64_t n = 60, s = 3, d = 10, k_stop = 4;

    T *G0 = randn<T>(n, n, /*seed=*/163);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/167);
    auto fscalar = [](T x) { return std::sqrt(x); };

    RandLAPACK::BlockLanczosFA<T> fa_stop;
    fa_stop.run_lanczos(A_op, Bmat, n, s, d,
                        [&](int64_t k) { return k >= k_stop; });
    ASSERT_EQ(fa_stop.steps_run, k_stop);
    T *out_stop = new T[n * s];
    fa_stop.apply_f(fscalar, n, s, d, out_stop);

    RandLAPACK::BlockLanczosFA<T> fa_ref;
    T *out_ref = new T[n * s];
    fa_ref.call(A_op, Bmat, n, s, fscalar, k_stop, out_ref);

    T maxdiff = 0, scale = 0;
    for (int64_t e = 0; e < n * s; ++e) {
        maxdiff = std::max(maxdiff, std::abs(out_stop[e] - out_ref[e]));
        scale   = std::max(scale, std::abs(out_ref[e]));
    }
    std::printf("early-stop apply_f vs fresh depth-%ld run: reldiff=%.3e\n",
                (long)k_stop, maxdiff / scale);
    EXPECT_LT(maxdiff / scale, 1e-12);
    delete[] G0; delete[] A; delete[] Bmat; delete[] out_stop; delete[] out_ref;
}

} // namespace
