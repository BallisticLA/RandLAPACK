#include "rl_lanczos_fa.hh"
#include "rl_lanczos_qfa.hh"
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

class TestLanczosQFA : public RandLAPACK::testing::LanczosTestSupport {};

// ===== Scalar Lanczos-QFA equals the per-column FA dots =====================
// The Gauss-quadrature identity bᵀ·LanczosFA(A, f, b) = Lanczos-QFA(A, f, b),
// per column: the fixed-depth scalar QFA vector out[j] must match
// ⟨B[:,j], (LanczosFA output)[:,j]⟩ computed by the reorth-0 scalar FA at the
// same depth (identical recurrence in exact arithmetic), and both must match
// the exact quadratic forms.
TEST_F(TestLanczosQFA, ScalarQFAmatchesScalarFAdots) {
    using T = double;
    const int64_t n = 60, s = 8, d = 30;

    T *G0 = randn<T>(n, n, /*seed=*/59);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    T *Bmat = randn<T>(n, s, /*seed=*/71);
    auto fscalar = [](T x) { return std::sqrt(x); };

    // FA path (vanilla Lanczos to match QFA's no-reorth recurrence).
    RandLAPACK::LanczosFA<T> lfa; lfa.reorth = 0;
    T *Gout = new T[n * s];
    lfa.call(A_op, Bmat, n, s, fscalar, d, Gout);

    // QFA path.
    RandLAPACK::LanczosQFA<T> qfa;
    T *qf = new T[s];
    qfa.call(A_op, Bmat, n, s, fscalar, d, qf);
    EXPECT_EQ(qfa.d_used, d);
    EXPECT_EQ(qfa.matvecs, s * d);

    // Exact quadratic forms for the absolute reference.
    auto exact = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    T *ref = new T[n * s];
    exact(n, s, Bmat, ref);

    T maxrel_fa = 0, maxrel_exact = 0;
    for (int64_t j = 0; j < s; ++j) {
        T dot_fa    = blas::dot(n, Bmat + j * n, 1, Gout + j * n, 1);
        T dot_exact = blas::dot(n, Bmat + j * n, 1, ref  + j * n, 1);
        maxrel_fa    = std::max(maxrel_fa,    std::abs(qf[j] - dot_fa)    / std::abs(dot_fa));
        maxrel_exact = std::max(maxrel_exact, std::abs(qf[j] - dot_exact) / std::abs(dot_exact));
    }
    std::printf("scalar QFA vs FA dots: maxrel=%.3e  vs exact: maxrel=%.3e\n",
                maxrel_fa, maxrel_exact);
    EXPECT_LT(maxrel_fa,    1e-10);
    EXPECT_LT(maxrel_exact, 1e-10);
    delete[] G0; delete[] A; delete[] Bmat; delete[] Gout; delete[] qf; delete[] ref;
}

// ===== Gauss/Gauss-Radau bracket on a diagonal matrix =======================
// On A = diag(1..n) the quadratic form bᵀf(A)b = Σᵢ f(i)·bᵢ² is exact and
// cheap. At several truncation depths the Gauss value (gauss_val) and the
// Gauss-Radau value (radau_val, node pinned at 0) must bracket the truth -
// this is the entire foundation of the certified stopping rule. Depths are
// probed by running adaptive mode with an unreachable tolerance so every
// column reports its (unclosed) bracket at the cap.
TEST_F(TestLanczosQFA, ScalarQFAradauBracketsDiagonal) {
    using T = double;
    const int64_t n = 80, s = 6;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/73);
    // Column 1 = e₉: on the diagonal A the three-term residual is exactly zero
    // in floating point, exercising the true breakdown path (certifies at t=1
    // with the exact value √10 despite the unreachable tolerance below).
    std::fill(Bmat + 1 * n, Bmat + 2 * n, (T)0);
    Bmat[9 + 1 * n] = (T)1;
    auto fscalar = [](T x) { return std::sqrt(x); };

    T truth[s];
    for (int64_t j = 0; j < s; ++j) {
        truth[j] = 0;
        for (int64_t i = 0; i < n; ++i) {
            T bij = Bmat[i + j * n];
            truth[j] += std::sqrt((T)(i + 1)) * bij * bij;
        }
    }

    T *qf = new T[s];
    for (int64_t depth : {4, 8, 16}) {
        RandLAPACK::LanczosQFA<T> qfa;
        qfa.adaptive = true;
        qfa.adaptive_rtol = std::numeric_limits<T>::min();  // never fires
        qfa.call(A_op, Bmat, n, s, fscalar, depth, qf);
        for (int64_t j = 0; j < s; ++j) {
            T hi = std::max(qfa.gauss_val[j], qfa.radau_val[j]);
            T lo = std::min(qfa.gauss_val[j], qfa.radau_val[j]);
            T slack = 1e-12 * std::abs(truth[j]);
            EXPECT_LE(lo - slack, truth[j]) << "depth " << depth << " col " << j;
            EXPECT_LE(truth[j], hi + slack) << "depth " << depth << " col " << j;
        }
        // Breakdown column: certified exactly at t = 1 regardless of tolerance.
        EXPECT_TRUE(qfa.certified[1]);
        EXPECT_EQ(qfa.t_used[1], 1);
        EXPECT_NEAR(qf[1], std::sqrt((T)10), 1e-14);
        T w0 = std::abs(qfa.gauss_val[0] - qfa.radau_val[0]) / std::abs(truth[0]);
        std::printf("Radau bracket d=%2ld: col0 gap/truth=%.3e (U=%.6e L=%.6e true=%.6e)\n",
                    depth, w0, qfa.gauss_val[0], qfa.radau_val[0], truth[0]);
    }
    delete[] A; delete[] Bmat; delete[] qf;
}

// ===== Certified relative error =============================================
// With adaptive stopping at eps, a certified column's Gauss value must be
// within eps of the true quadratic form (up to a small roundoff factor) -
// eps is a guarantee, not a target scale.
TEST_F(TestLanczosQFA, ScalarQFAcertifiedRelErr) {
    using T = double;
    const int64_t n = 80, s = 8, d_cap = 79;
    const T eps = 1e-6;

    T *G0 = randn<T>(n, n, /*seed=*/79);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/83);
    auto fscalar = [](T x) { return std::sqrt(x); };

    auto exact = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    T *ref = new T[n * s];
    exact(n, s, Bmat, ref);

    RandLAPACK::LanczosQFA<T> qfa;
    qfa.adaptive = true; qfa.adaptive_rtol = eps;
    T *qf = new T[s];
    qfa.call(A_op, Bmat, n, s, fscalar, d_cap, qf);

    EXPECT_TRUE(qfa.all_certified);
    int64_t sum_t = 0;
    T maxrel = 0;
    for (int64_t j = 0; j < s; ++j) {
        T tj = blas::dot(n, Bmat + j * n, 1, ref + j * n, 1);
        maxrel = std::max(maxrel, std::abs(qf[j] - tj) / std::abs(tj));
        sum_t += qfa.t_used[j];
        EXPECT_TRUE(qfa.certified[j]) << "col " << j;
    }
    std::printf("certified relerr: max=%.3e (eps=%.0e)  d_used=%ld  matvecs=%ld=Σt\n",
                maxrel, eps, (long)qfa.d_used, (long)qfa.matvecs);
    EXPECT_LT(maxrel, 2 * eps);          // certified bound + roundoff slack
    EXPECT_EQ(qfa.matvecs, sum_t);       // honest per-column accounting
    delete[] G0; delete[] A; delete[] Bmat; delete[] ref; delete[] qf;
}

// ===== Adaptive stopping with heterogeneous per-column depths ===============
// One probe column is an exact eigenvector of A: its Krylov space is
// 1-dimensional, so it breaks down (β = 0) and certifies exactly at t = 1
// while random columns run deeper - exercising the retire/compaction
// bookkeeping (shrinking batched matvec) that a uniform-depth run never hits.
TEST_F(TestLanczosQFA, ScalarQFAadaptiveStopsEarly) {
    using T = double;
    const int64_t n = 80, s = 6, d_max = 70;

    T *G0 = randn<T>(n, n, /*seed=*/89);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    // Eigendecomposition of A: plant V[:,0] (smallest eigenvalue) as column 2.
    T *Vfull = new T[n * n];
    T *ev    = new T[n];
    std::copy(A, A + n * n, Vfull);
    lapack::syevd(lapack::Job::Vec, lapack::Uplo::Upper, n, Vfull, n, ev);
    T *Bmat = randn<T>(n, s, /*seed=*/97);
    blas::copy(n, Vfull, 1, Bmat + 2 * n, 1);

    auto fscalar = [](T x) { return std::sqrt(x); };

    // Fixed-depth reference at the cap.
    RandLAPACK::LanczosQFA<T> qfa_fixed;
    T *qf_fixed = new T[s];
    qfa_fixed.call(A_op, Bmat, n, s, fscalar, d_max, qf_fixed);

    // Adaptive.
    RandLAPACK::LanczosQFA<T> qfa;
    qfa.adaptive = true; qfa.adaptive_rtol = 1e-8;
    T *qf = new T[s];
    qfa.call(A_op, Bmat, n, s, fscalar, d_max, qf);

    EXPECT_TRUE(qfa.all_certified);
    EXPECT_LT(qfa.d_used, d_max);            // stopped early
    EXPECT_LT(qfa.matvecs, s * d_max);       // spent less than the uniform cost
    // Eigenvector column: the three-term residual is roundoff (~1e-15), so it
    // certifies via the bracket at t = 2 rather than the exact-breakdown path.
    EXPECT_LE(qfa.t_used[2], 2);
    EXPECT_NEAR(qf[2], fscalar(ev[0]), 1e-8 * std::abs(fscalar(ev[0])));

    T maxrel = 0;
    for (int64_t j = 0; j < s; ++j)
        maxrel = std::max(maxrel, std::abs(qf[j] - qf_fixed[j]) / std::abs(qf_fixed[j]));
    std::printf("adaptive scalar QFA: d_used=%ld/%ld matvecs=%ld/%ld  t_used=[",
                (long)qfa.d_used, (long)d_max, (long)qfa.matvecs, (long)(s * d_max));
    for (int64_t j = 0; j < s; ++j) std::printf("%ld ", (long)qfa.t_used[j]);
    std::printf("]  vs fixed maxrel=%.3e\n", maxrel);
    EXPECT_LT(maxrel, 1e-7);                 // matches the converged value
    delete[] G0; delete[] A; delete[] Vfull; delete[] ev;
    delete[] Bmat; delete[] qf_fixed; delete[] qf;
}

// ---------------------------------------------------------------------------
// Panel-kernel DECOMPOSITION invariants (pure arithmetic; no OpenMP, no timing).
//
// This exists because two shipped versions of LanczosQFA's panel kernels had
// parallelization defects that EVERY correctness test passed bit-identically:
//   (1) a fixed 4096-element row block gave ONE block at n = 3000, so one thread
//       worked and the rest idled (measured 2x slower);
//   (2) after that was "fixed", a 512-element lower clamp still pinned the block
//       count at 6 for any n <= 114688, so at the auto tier's 4-column probe the
//       trip count was 24 and 88 of 112 threads idled.
// A parallelization defect has no numerical signature, so no accuracy test can
// see it. These assertions check the DECOMPOSITION instead, and would have
// failed on both versions above.
TEST_F(TestLanczosQFA, PanelChunkPlanInvariants) {
    using QFA = RandLAPACK::LanczosQFA<double>;
    const int64_t Ns[]     = {1, 17, 1000, 3000, 100000, 1000000};
    const int64_t NCOLS[]  = {1, 2, 4, 8, 32, 96, 128};
    const int     THREADS[] = {1, 2, 16, 112};

    for (int64_t n : Ns)
    for (int64_t nc : NCOLS)
    for (int p : THREADS) {
        auto cp = QFA::chunk_plan(n, nc, p);
        const int64_t total = n * nc;

        // 1. Never request more threads than exist, and always at least one.
        ASSERT_GE(cp.n_threads, 1)  << "n=" << n << " nc=" << nc << " P=" << p;
        ASSERT_LE(cp.n_threads, p)  << "n=" << n << " nc=" << nc << " P=" << p;

        // 2. THE INVARIANT BOTH BUGS VIOLATED: every requested thread gets work.
        //    The chunk count is defined as a multiple of the team size, so it can
        //    never fall below it for any (n, ncols, nthreads).
        ASSERT_GE(cp.n_chunks, (int64_t)cp.n_threads)
            << "starved team: n=" << n << " ncols=" << nc << " threads=" << p
            << " -> chunks=" << cp.n_chunks << " team=" << cp.n_threads;

        // 3. Never fork a team for trivial work (the opposite failure: satisfying
        //    invariant 2 by splitting a 24 KB panel across 112 threads).
        if (cp.n_threads > 1) {
            ASSERT_GE(total / cp.n_threads, QFA::MIN_ELEMS_PER_THREAD)
                << "forked for trivial work: n=" << n << " ncols=" << nc;
        }

        // 4. The chunk ranges must tile [0, total) exactly: contiguous, no gaps,
        //    no overlap, covering everything (correctness of the decomposition).
        int64_t prev_hi = 0;
        for (int64_t c = 0; c < cp.n_chunks; ++c) {
            int64_t lo, hi;
            QFA::chunk_range(total, cp.n_chunks, c, lo, hi);
            ASSERT_EQ(lo, prev_hi) << "gap/overlap at chunk " << c;
            ASSERT_LE(lo, hi);
            prev_hi = hi;
        }
        ASSERT_EQ(prev_hi, total) << "chunks do not cover the panel";
    }
}

// Per-thread partial slots must be cache-line padded, else threads writing
// adjacent slots ping-pong one line - worst exactly when ncols is small, which
// is the retirement tail this kernel exists to serve.
TEST_F(TestLanczosQFA, PanelPartialStrideIsCacheLinePadded) {
    using QFA = RandLAPACK::LanczosQFA<double>;
    constexpr int64_t LINE = 64 / (int64_t)sizeof(double);
    for (int64_t nc : {1, 2, 4, 7, 8, 9, 32, 96, 100}) {
        const int64_t st = QFA::partial_stride(nc);
        ASSERT_GE(st, nc);
        ASSERT_EQ(st % LINE, 0) << "stride " << st << " for ncols=" << nc
                                << " is not a multiple of a cache line";
    }
}

// ===== Float instantiation ==================================================
// The certified scalar QFA must instantiate and behave at T = float on a
// well-conditioned diagonal SPD matrix (kappa = 1e3). The corresponding
// expert driver check is FloatExpertPath in test_fun_nystrom_pp.cc.
// Bounds are float-loose by design.
TEST_F(TestLanczosQFA, FloatCertifiedQFA) {
    using T = float;
    const int64_t n = 200;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)1 + (T)999 * (T)i / (T)(n - 1);   // linear in [1, 1000]
    }
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    // Certified adaptive scalar QFA at float: certifies and brackets the exact
    // per-column quadratic forms on the same diagonal matrix.
    const int64_t sq = 6, d_cap = 150;
    const T rtol = (T)1e-3;
    T *Bq = randn<T>(n, sq, /*seed=*/181);
    T truth[sq];
    for (int64_t j = 0; j < sq; ++j) {
        truth[j] = 0;
        for (int64_t i = 0; i < n; ++i) {
            T bij = Bq[i + j * n];
            truth[j] += std::sqrt(A[i + i * n]) * bij * bij;
        }
    }
    RandLAPACK::LanczosQFA<T> qfa;
    qfa.adaptive = true; qfa.adaptive_rtol = rtol;
    T *qf = new T[sq];
    qfa.call(A_op, Bq, n, sq, fscalar, d_cap, qf);
    EXPECT_TRUE(qfa.all_certified);
    EXPECT_LT(qfa.d_used, d_cap);
    T maxrel = 0;
    for (int64_t j = 0; j < sq; ++j) {
        EXPECT_TRUE(qfa.certified[j]) << "col " << j;
        // Bracket property, float slack for accumulated roundoff.
        T hi = std::max(qfa.gauss_val[j], qfa.radau_val[j]);
        T lo = std::min(qfa.gauss_val[j], qfa.radau_val[j]);
        T slack = (T)1e-4 * std::abs(truth[j]);
        EXPECT_LE(lo - slack, truth[j]) << "col " << j;
        EXPECT_LE(truth[j], hi + slack) << "col " << j;
        maxrel = std::max(maxrel, std::abs(qf[j] - truth[j]) / std::abs(truth[j]));
    }
    std::printf("f32 certified QFA: d_used=%ld matvecs=%ld maxrel=%.3e (rtol=%.0e)\n",
                (long)qfa.d_used, (long)qfa.matvecs, maxrel, rtol);
    EXPECT_LT(maxrel, 3 * rtol);
    delete[] A; delete[] Bq; delete[] qf;
}

// ===== Scalar QFA: all-zero input column ====================================
// A zero column has no Krylov space: it must retire at t = 0 with value 0,
// certified, before the first matvec - so the batched matvec never pays for
// it and `matvecs` equals the sum of the OTHER columns' depths exactly.
TEST_F(TestLanczosQFA, ScalarQFAZeroColumnRetiresAtZero) {
    using T = double;
    const int64_t n = 50, s = 4, d_cap = 40;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/241);
    std::fill(Bmat + 1 * n, Bmat + 2 * n, (T)0);   // column 1 all-zero
    auto fscalar = [](T x) { return std::sqrt(x); };

    RandLAPACK::LanczosQFA<T> qfa;
    qfa.adaptive = true; qfa.adaptive_rtol = 1e-8;
    T *qf = new T[s];
    qfa.call(A_op, Bmat, n, s, fscalar, d_cap, qf);

    EXPECT_EQ(qfa.t_used[1], 0);
    EXPECT_EQ(qf[1], (T)0);
    EXPECT_TRUE(qfa.certified[1]);
    int64_t sum_others = 0;
    for (int64_t j = 0; j < s; ++j) {
        EXPECT_TRUE(std::isfinite(qf[j]))            << "col " << j;
        EXPECT_TRUE(std::isfinite(qfa.gauss_val[j])) << "col " << j;
        EXPECT_TRUE(std::isfinite(qfa.radau_val[j])) << "col " << j;
        if (j != 1) sum_others += qfa.t_used[j];
    }
    EXPECT_EQ(qfa.matvecs, sum_others);   // the zero column cost zero matvecs
    std::printf("zero-column QFA: t_used=[%ld %ld %ld %ld] matvecs=%ld\n",
                (long)qfa.t_used[0], (long)qfa.t_used[1],
                (long)qfa.t_used[2], (long)qfa.t_used[3], (long)qfa.matvecs);
    delete[] A; delete[] Bmat; delete[] qf;
}

// ===== Scalar QFA: depth d == 1 =============================================
// At depth 1 the tridiagonal is the scalar alpha_1 = q1' A q1, so the Gauss
// value is ||b||^2 * f(alpha_1) exactly - checkable to roundoff on a diagonal
// matrix. Both modes must run without crashing; adaptive cannot certify (the
// bracket needs t >= 2) and must still return the depth-1 Gauss value.
TEST_F(TestLanczosQFA, ScalarQFADepthOne) {
    using T = double;
    const int64_t n = 30, s = 3;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/251);
    auto fscalar = [](T x) { return std::sqrt(x); };

    T expected[s];
    for (int64_t j = 0; j < s; ++j) {
        T nb2 = 0, wsum = 0;
        for (int64_t i = 0; i < n; ++i) {
            T bij = Bmat[i + j * n];
            nb2  += bij * bij;
            wsum += (T)(i + 1) * bij * bij;
        }
        expected[j] = nb2 * std::sqrt(wsum / nb2);   // ||b||^2 * f(alpha_1)
    }

    T *qf = new T[s];
    {   // fixed depth 1
        RandLAPACK::LanczosQFA<T> qfa;
        qfa.call(A_op, Bmat, n, s, fscalar, 1, qf);
        EXPECT_EQ(qfa.d_used, 1);
        EXPECT_EQ(qfa.matvecs, s);
        for (int64_t j = 0; j < s; ++j)
            EXPECT_NEAR(qf[j], expected[j], 1e-13 * std::abs(expected[j])) << "col " << j;
    }
    {   // adaptive with cap 1: no certificate possible, same value, no crash
        RandLAPACK::LanczosQFA<T> qfa;
        qfa.adaptive = true; qfa.adaptive_rtol = 1e-6;
        qfa.call(A_op, Bmat, n, s, fscalar, 1, qf);
        EXPECT_FALSE(qfa.all_certified);
        for (int64_t j = 0; j < s; ++j) {
            EXPECT_EQ(qfa.certified[j], 0) << "col " << j;
            EXPECT_EQ(qfa.t_used[j], 1)    << "col " << j;
            EXPECT_NEAR(qf[j], expected[j], 1e-13 * std::abs(expected[j])) << "col " << j;
        }
    }
    std::printf("depth-1 QFA: values match ||b||^2 f(alpha_1) to roundoff (s=%ld)\n", (long)s);
    delete[] A; delete[] Bmat; delete[] qf;
}

// ===== Scalar QFA: fixed check stride (check_every > 1) =====================
// check_every = 5 replaces the geometric ladder with a fixed stride: in-run
// certificate checks land only at t % 5 == 0 (plus the final at-cap check),
// so certified depths sit on that grid - possibly different from the ladder's
// - while the certified value stays within tolerance. check_every = 0 is
// rejected up front.
TEST_F(TestLanczosQFA, ScalarQFACheckEveryStride) {
    using T = double;
    const int64_t n = 80, s = 6, d_cap = 79;
    const T eps = 1e-6;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/257);
    auto fscalar = [](T x) { return std::sqrt(x); };

    T truth[s];
    for (int64_t j = 0; j < s; ++j) {
        truth[j] = 0;
        for (int64_t i = 0; i < n; ++i) {
            T bij = Bmat[i + j * n];
            truth[j] += std::sqrt((T)(i + 1)) * bij * bij;
        }
    }

    RandLAPACK::LanczosQFA<T> qfa;
    qfa.adaptive = true; qfa.adaptive_rtol = eps; qfa.check_every = 5;
    T *qf = new T[s];
    qfa.call(A_op, Bmat, n, s, fscalar, d_cap, qf);

    EXPECT_TRUE(qfa.all_certified);
    T maxrel = 0;
    for (int64_t j = 0; j < s; ++j) {
        EXPECT_TRUE(qfa.certified[j]) << "col " << j;
        // Certified depths sit on the stride grid, or at the cap (final check).
        EXPECT_TRUE(qfa.t_used[j] % 5 == 0 || qfa.t_used[j] == d_cap)
            << "col " << j << " t_used=" << qfa.t_used[j];
        maxrel = std::max(maxrel, std::abs(qf[j] - truth[j]) / std::abs(truth[j]));
    }
    std::printf("check_every=5: t_used=[");
    for (int64_t j = 0; j < s; ++j) std::printf("%ld ", (long)qfa.t_used[j]);
    std::printf("] maxrel=%.3e (eps=%.0e)\n", maxrel, eps);
    EXPECT_LT(maxrel, 2 * eps);

    RandLAPACK::LanczosQFA<T> qfa_bad;
    qfa_bad.check_every = 0;
    EXPECT_THROW(qfa_bad.call(A_op, Bmat, n, s, fscalar, d_cap, qf),
                 std::invalid_argument);
    delete[] A; delete[] Bmat; delete[] qf;
}

// ===== Scalar QFA: shrinking reuse is bit-identical =========================
// The internal buffers grow and never shrink, and ws_depth carries across
// calls (reset only in adaptive mode). A second, SMALLER call (s and d both
// shrink) on a reused instance must be bit-for-bit the run a fresh instance
// produces - pinning that no oversized-buffer state leaks into the values.
TEST_F(TestLanczosQFA, ScalarQFAShrinkingReuseBitIdentical) {
    using T = double;
    const int64_t n = 70;

    T *G0 = randn<T>(n, n, /*seed=*/263);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(x); };
    T *B1 = randn<T>(n, 8, /*seed=*/269);
    T *B2 = randn<T>(n, 3, /*seed=*/271);

    // Reused instance: big adaptive call, then the small one.
    RandLAPACK::LanczosQFA<T> shared;
    shared.adaptive = true; shared.adaptive_rtol = 1e-8;
    T out1[8], out2[3];
    shared.call(A_op, B1, n, 8, fscalar, 40, out1);
    shared.call(A_op, B2, n, 3, fscalar, 12, out2);

    // Fresh instance: only the small call.
    RandLAPACK::LanczosQFA<T> fresh;
    fresh.adaptive = true; fresh.adaptive_rtol = 1e-8;
    T out2_ref[3];
    fresh.call(A_op, B2, n, 3, fscalar, 12, out2_ref);

    EXPECT_EQ(shared.d_used,  fresh.d_used);
    EXPECT_EQ(shared.matvecs, fresh.matvecs);
    for (int64_t j = 0; j < 3; ++j) {
        EXPECT_EQ(out2[j], out2_ref[j])                     << "col " << j;
        EXPECT_EQ(shared.t_used[j],    fresh.t_used[j])     << "col " << j;
        EXPECT_EQ(shared.gauss_val[j], fresh.gauss_val[j])  << "col " << j;
        EXPECT_EQ(shared.radau_val[j], fresh.radau_val[j])  << "col " << j;
        EXPECT_EQ(shared.certified[j], fresh.certified[j])  << "col " << j;
    }
    std::printf("shrinking reuse: second call (s=3, d=12) bit-identical to fresh "
                "(d_used=%ld matvecs=%ld)\n", (long)fresh.d_used, (long)fresh.matvecs);
    delete[] G0; delete[] A; delete[] B1; delete[] B2;
}

// ===== Scalar QFA: indefinite matrix disables the certificate ===============
// The Gauss-Radau certificate requires T_t positive definite (the LDL' pivot
// chain). On an INDEFINITE symmetric matrix the pivots go non-positive as the
// tridiagonal picks up the negative spectrum, so with a tolerance too tight
// to certify beforehand the pivot guard must disable every column's
// certificate - uncertified, finite values, no crash. f = x^2 is finite
// everywhere (NB quad_e1 clamps Ritz values to >= 0 by the A >= 0 contract,
// so the VALUES are not accurate here; this test pins only the guard).
//
// The first sub-case's rtol = 1e-12 is UNREACHABLE at d_cap = 30 regardless
// of definiteness, so on its own it cannot distinguish "the pivot guard
// fired" from "the tolerance was simply too tight for any matrix of this
// size" - a control at a LOOSE rtol (1e-2) is needed: on a well-conditioned
// SPD matrix of the same n/d_cap that tolerance certifies quickly (the same
// regime ScalarQFAcertifiedRelErr already exercises, eps = 1e-6 well within
// 79 steps), so if the indefinite matrix ALSO fails to certify at the loose
// tolerance, the failure is attributable to indefiniteness (the pivot guard),
// not to an unreachable target.
TEST_F(TestLanczosQFA, ScalarQFAIndefiniteUncertified) {
    using T = double;
    const int64_t n = 60, s = 5, d_cap = 30;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1) - (T)30;  // -29..30
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/277);
    auto fscalar = [](T x) { return x * x; };

    {   // Unreachably tight rtol: pins the guard's OUTPUT contract (finite,
        // uncertified, no crash), but by itself can't isolate the reason.
        RandLAPACK::LanczosQFA<T> qfa;
        qfa.adaptive = true; qfa.adaptive_rtol = 1e-12;
        T *qf = new T[s];
        EXPECT_NO_THROW(qfa.call(A_op, Bmat, n, s, fscalar, d_cap, qf));

        EXPECT_FALSE(qfa.all_certified);
        for (int64_t j = 0; j < s; ++j) {
            EXPECT_EQ(qfa.certified[j], 0)               << "col " << j;
            EXPECT_TRUE(std::isfinite(qf[j]))            << "col " << j;
            EXPECT_TRUE(std::isfinite(qfa.gauss_val[j])) << "col " << j;
            EXPECT_TRUE(std::isfinite(qfa.radau_val[j])) << "col " << j;
        }
        std::printf("indefinite QFA (rtol=1e-12, unreachable): all %ld columns "
                    "uncertified (d_used=%ld), values finite\n", (long)s, (long)qfa.d_used);
        delete[] qf;
    }
    {   // Control: the SAME loose rtol = 1e-2 certifies readily on a well-
        // conditioned SPD matrix of the same n/d_cap (n=60, d_cap=30 comfortably
        // exceeds what ScalarQFAcertifiedRelErr needs at a far tighter eps).
        T *A_spd = new T[n * n]();
        for (int64_t i = 0; i < n; ++i) A_spd[i + i * n] = (T)(i + 1);   // 1..60, SPD
        linops::ExplicitSymLinOp<T> A_spd_op(n, blas::Uplo::Upper, A_spd, n, Layout::ColMajor);
        RandLAPACK::LanczosQFA<T> qfa_ctrl;
        qfa_ctrl.adaptive = true; qfa_ctrl.adaptive_rtol = 1e-2;
        T *qf_ctrl = new T[s];
        EXPECT_NO_THROW(qfa_ctrl.call(A_spd_op, Bmat, n, s, fscalar, d_cap, qf_ctrl));
        EXPECT_TRUE(qfa_ctrl.all_certified)
            << "control precondition failed: rtol=1e-2 should certify readily on "
               "a well-conditioned SPD matrix of this size - if not, the loose "
               "rtol below isn't actually loose enough to isolate indefiniteness";
        std::printf("control SPD QFA (rtol=1e-2): all_certified=%d d_used=%ld\n",
                    (int)qfa_ctrl.all_certified, (long)qfa_ctrl.d_used);
        delete[] A_spd; delete[] qf_ctrl;

        // The actual pin: the SAME indefinite matrix, at the SAME loose rtol
        // the control just showed certifies easily on an SPD matrix, still
        // fails to certify - isolating "indefinite -> never certifies even
        // when the tolerance is easy" from "tight tolerance -> never
        // certifies regardless of definiteness".
        RandLAPACK::LanczosQFA<T> qfa_loose;
        qfa_loose.adaptive = true; qfa_loose.adaptive_rtol = 1e-2;
        T *qf_loose = new T[s];
        EXPECT_NO_THROW(qfa_loose.call(A_op, Bmat, n, s, fscalar, d_cap, qf_loose));
        EXPECT_FALSE(qfa_loose.all_certified)
            << "indefinite matrix certified at a LOOSE rtol - the pivot guard "
               "did not fire, or is not the reason the tight-rtol sub-case failed";
        for (int64_t j = 0; j < s; ++j) {
            EXPECT_TRUE(std::isfinite(qf_loose[j])) << "col " << j;
        }
        std::printf("indefinite QFA (rtol=1e-2, loose): all_certified=%d d_used=%ld "
                    "- failure isolated to the pivot guard, not tolerance\n",
                    (int)qfa_loose.all_certified, (long)qfa_loose.d_used);
        delete[] qf_loose;
    }
    delete[] A; delete[] Bmat;
}

TEST_F(TestLanczosQFA, ScalarQFAPropagatesParallelEvaluationException) {
    const int64_t n = 64, s = 4;
    std::vector<double> lambda(n), out(s);
    for (int64_t i = 0; i < n; ++i) lambda[i] = 1. + i;
    linops::DiagSymLinOp<double> op(n, lambda.data());
    double* B = randn<double>(n, s, 346002);
    RandLAPACK::LanczosQFA<double> q;
    for (bool adaptive : {false, true}) {
        q.adaptive = adaptive;
        auto fails = [](double) -> double { throw std::runtime_error("fixture: failed evaluation"); };
        EXPECT_THROW(q.call(op, B, n, s, fails, 8, out.data()), std::runtime_error);
        EXPECT_LE(q.matvecs, s * 8);
        auto f = [](double x) { return std::log1p(x); };
        EXPECT_NO_THROW(q.call(op, B, n, s, f, 8, out.data()));
        for (double x : out) EXPECT_TRUE(std::isfinite(x));
    }
    delete[] B;
}

} // namespace
