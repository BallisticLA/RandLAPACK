#include "RandLAPACK.hh"
#include "../comps/lanczos_test_support.hh"

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

class TestFunNystromPP : public RandLAPACK::testing::LanczosTestSupport {};

// Counting wrapper around ExplicitSymLinOp, satisfying the same
// SymmetricLinearOperator concept plus the RandBLAS SketchingOperator matvec
// overload NystromEVD dispatches through for its internal SASO sketch.
// Increments `apply_count` by the number of columns applied (n / n_vecs) on
// EVERY invocation, whichever overload is used - independent, externally-
// counted ground truth for I7: every accounting assertion elsewhere in this
// file checks one self-reported driver counter against another self-reported
// counter or a budget-derived bound; this wrapper checks the driver's
// reported telemetry against how many times A was actually applied.
template <typename T>
struct CountingSymLinOp {
    using scalar_t = T;
    RandLAPACK::linops::ExplicitSymLinOp<T> inner;
    const int64_t dim;
    int64_t apply_count = 0;

    CountingSymLinOp(int64_t dim_, blas::Uplo uplo, const T* A_buff, int64_t lda, Layout layout)
        : inner(dim_, uplo, A_buff, lda, layout), dim(dim_) {}

    void operator()(Layout layout, int64_t n, T alpha, T* const B, int64_t ldb,
                     T beta, T* C, int64_t ldc) {
        apply_count += n;
        inner(layout, n, alpha, B, ldb, beta, C, ldc);
    }

    template <RandBLAS::SketchingOperator SkOp>
    void operator()(Layout layout, int64_t n_vecs, T alpha, SkOp& S, T beta, T* C, int64_t ldc) {
        apply_count += n_vecs;
        inner(layout, n_vecs, alpha, S, beta, C, ldc);
    }
};

// Phase 1 tests. The fAfun oracle is "exact dense f(A) · B" computed
// once per test from an explicit eigendecomposition of A; that lets
// each test isolate the v2 driver's behavior from Krylov truncation.
// Phase 4 will add a block-Lanczos fAfun and re-run an analogous set.
//
// All buffers are raw new[]/delete[] (house rule: no std::vector for
// matrix/vector data) and all randomness goes through RandBLAS
// (Philox4x32; house rule: no std::mt19937).

// ===== Phase 1 accuracy tests ================================================

// Diagonal A = diag(1..n), f = sqrt. True trace is Σ √i, no eigensolver
// needed. k = 15 < n = 50; Hutchinson correction does real work.
TEST_F(TestFunNystromPP, DiagonalSqrt) {
    using T = double;
    const int64_t n = 50, k = 15, s = 300, q = 2;

    T *A = new T[n * n]();   // zero-init required: only the diagonal is written below
    T true_tr = 0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)(i + 1);
        true_tr += std::sqrt((T)(i + 1));
    }

    auto fscalar = [](T x) { return std::sqrt(x); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);

    // Phase-1 sketch is kernel-internal (SASO drawn from this state);
    // only the Phase-2 probes are supplied explicitly.
    RandBLAS::RNGState<RNG> state(1);
    T *Omega2 = randn<T>(n, s, /*seed=*/2);

    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fAfun, fscalar, k, s, q,
                        state, Omega2, t1, t2);
    T err = std::abs(est - true_tr) / true_tr;
    std::printf("v2 Diagonal sqrt: est=%.10e true=%.10e err=%.3e (t1=%.3e t2=%.3e)\n",
                est, true_tr, err, t1, t2);
    EXPECT_LT(err, 5e-2);
    delete[] A;
    delete[] Omega2;
}

// Low-rank PSD with k_mat = 10 distinct eigenvalues and an n - k_mat tail
// of zeros. With k = k_mat, NystromEVD captures the full effective
// rank, so t1 matches the analytical Σ √λⱼ to ~ε_mach.
//
// The total estimate, however, carries a ~1e-6 bias even at full-rank
// capture: in exact arithmetic the Phase 2 residual `f(A)Ω − f(Â)Ω` is
// identically zero (U and λ̂ span the same subspace as V and λ), but the
// two GEMM paths (V·diag(f(λ))·Vᵀ·Ω vs U·diag(f(λ̂))·Uᵀ·Ω) accumulate
// different floating-point error per column, and Hutchinson sums those
// per-column residuals into a systematic ~s · ε_mach bias. The relaxed
// `err_tot < 1e-5` threshold documents this realistic floor; the tight
// `err_t1 < 1e-12` threshold is what's actually load-bearing.
TEST_F(TestFunNystromPP, FullRankCapture) {
    using T = double;
    const int64_t n = 80, k_mat = 10, k = 10, s = 200, q = 2;

    // Eigenvalues 100 / j² (algebraic decay, like Persson's setup).
    T eigvals[k_mat];
    for (int64_t j = 0; j < k_mat; ++j) eigvals[j] = (T)100.0 / (T)((j + 1) * (j + 1));

    // Construct A = V · diag(eigvals) · Vᵀ with V a random orthonormal m × k_mat.
    T *V_raw = randn<T>(n, k_mat, /*seed=*/7);
    T *tau   = new T[k_mat];
    lapack::geqrf(n, k_mat, V_raw, n, tau);
    lapack::ungqr(n, k_mat, k_mat, V_raw, n, tau);

    // A = V · D · Vᵀ
    T *Vd = new T[n * k_mat];
    for (int64_t j = 0; j < k_mat; ++j)
        for (int64_t i = 0; i < n; ++i)
            Vd[i + j * n] = V_raw[i + j * n] * eigvals[j];
    T *A = new T[n * n];   // no zero-init: gemm(beta=0) writes every entry
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               n, n, k_mat, (T)1, Vd, n, V_raw, n, (T)0, A, n);
    // symmetrize (drop fp asymmetry)
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            A[i + j * n] = A[j + i * n];

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    T true_tr = 0;
    for (int64_t j = 0; j < k_mat; ++j) true_tr += fscalar(eigvals[j]);

    auto fAfun = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    RandBLAS::RNGState<RNG> state(11);
    T *Omega2 = randn<T>(n, s, /*seed=*/13);

    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fAfun, fscalar, k, s, q,
                        state, Omega2, t1, t2);
    T err_t1  = std::abs(t1  - true_tr) / true_tr;
    T err_tot = std::abs(est - true_tr) / true_tr;
    std::printf("v2 FullRankCapture: t1=%.10e t2=%.3e est=%.10e true=%.10e (err_t1=%.3e err_tot=%.3e)\n",
                t1, t2, est, true_tr, err_t1, err_tot);
    EXPECT_LT(err_t1,  1e-12);   // Phase 1 captures full rank → ε_mach
    EXPECT_LT(err_tot, 1e-5);    // two-path arithmetic floor (see comment above)
    delete[] V_raw;
    delete[] tau;
    delete[] Vd;
    delete[] A;
    delete[] Omega2;
}

// Random dense PSD, f = sqrt. k = 10, k_mat unknown - Phase 1 captures
// only the top subspace, Phase 2's Hutchinson carries real load. Tol = 15%.
TEST_F(TestFunNystromPP, RandomPSDSqrt) {
    using T = double;
    const int64_t n = 40, k = 10, s = 400, q = 2;

    // A = BᵀB + n·I  (well-conditioned random PSD)
    T *B_raw = randn<T>(n, n, /*seed=*/17);
    T *A = new T[n * n];   // no zero-init: syrk(beta=0) + the mirror loop write every entry
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, B_raw, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    // symmetrize
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            A[i + j * n] = A[j + i * n];

    auto fscalar = [](T x) { return std::sqrt(x); };
    T true_tr = true_trace_fa<T>(n, A, fscalar);
    auto fAfun = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);

    RandBLAS::RNGState<RNG> state(19);
    T *Omega2 = randn<T>(n, s, /*seed=*/23);

    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fAfun, fscalar, k, s, q,
                        state, Omega2, t1, t2);
    T err = std::abs(est - true_tr) / true_tr;
    std::printf("v2 RandomPSDSqrt: est=%.10e true=%.10e err=%.3e (t1=%.3e t2=%.3e)\n",
                est, true_tr, err, t1, t2);
    EXPECT_LT(err, 0.15);
    delete[] B_raw;
    delete[] A;
    delete[] Omega2;
}

// The knob-free overload call(A, f, m, eps, state, ...) must (a) never
// overspend the matvec budget - with the certified scalar-QFA oracle the
// closure is an upper bound, probe + q*k + oracle_mv <= m, since columns stop
// at their own certified depths (probe-sample REUSE folds certified probe
// columns into the Phase-2 average but costs zero extra matvecs, so the
// closure invariant is unchanged) - (b) bound the probe's spend by the
// auto_probe_frac cap (default 1/8 of the budget), (c) allocate rank-heavy
// (k >> s; on this easy spectrum the n/2 rank cap binds and the surplus goes
// to probes), and (d) deliver a sane estimate with both certification flags
// reported. Well-conditioned SPD so the probe certifies at a small MEDIAN
// depth (the redesigned depth policy) and k = n/2 stays feasible.
TEST_F(TestFunNystromPP, AutoBudgetClosesAndEstimates) {
    using T = double;
    const int64_t n = 400;
    const int64_t m_budget = 700;
    const T eps = 1e-3;

    T *G0 = randn<T>(n, n, /*seed=*/61);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    T true_tr = true_trace_fa(n, A, fscalar);

    RandBLAS::RNGState<RNG> state(29);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);

    const int64_t spend = driver.auto_probe_matvecs
                        + driver.auto_k + driver.auto_oracle_matvecs;
    T err = std::abs(est - true_tr) / std::abs(true_tr);
    std::printf("auto: m=%ld spend=%ld (probe=%ld k=%ld s=%ld t=%ld oracle_mv=%ld conv=%d)  relerr=%.3e\n",
                (long)m_budget, (long)spend, (long)driver.auto_probe_matvecs,
                (long)driver.auto_k, (long)driver.auto_s, (long)driver.auto_t,
                (long)driver.auto_oracle_matvecs,
                (int)driver.auto_probe_converged, err);
    EXPECT_TRUE(driver.auto_probe_converged);
    EXPECT_TRUE(driver.auto_phase2_certified);   // easy spectrum: Phase-2 oracle
                                                 // certifies at the median-depth cap
    EXPECT_LE(driver.auto_k, n / 2);          // rank cap (fragile k -> n Gram corner)
    EXPECT_LE(spend, m_budget);               // certified stopping never overspends
    // Probe-fraction cap: the probe may spend at most ~1/8 of the budget
    // (b columns, depth cap max(2, floor(0.125*B/b))); on this easy spectrum
    // it certifies far below even that.
    EXPECT_LE(driver.auto_probe_matvecs,
              4 * std::max((int64_t)2, (int64_t)(0.125 * m_budget / 4)));
    EXPECT_GT(driver.t_probe_ms, 0.0);        // the probe's wall-clock is attributed
    EXPECT_GT(driver.auto_oracle_matvecs, 0);
    EXPECT_LE(driver.auto_oracle_matvecs,    // per-column depths never exceed the cap
              driver.auto_s * driver.auto_t);
    EXPECT_GT(driver.auto_k, driver.auto_s);  // rank-heavy split
    EXPECT_LT(err, 1e-2);
    delete[] G0; delete[] A;
}

// ===== Auto tier: probe-reuse fold mechanics (independent-arithmetic pin) ===
// Structurally analogous to AdaptiveProbeReuseFolds (below), but for the
// scalar auto tier's fold, at the fold_probe_reuse call site inside
// FunNystromPP::call(auto) in rl_fun_nystrom_pp.hh (name, not a line number,
// since the extraction of fold_probe_reuse as a shared helper already moved
// this call site once). After call(),
// driver.Omega2_buf holds the Phase-2 probe block, driver.auto_probe_buf /
// auto_probe_gauss / auto_probe_cert hold the depth probe's block and its
// certified per-column quadratic forms - enough to reconstruct, from
// outside the driver, both t2 BEFORE the fold (an independent scalar
// LanczosQFA run - same adaptive settings (adaptive=true, rtol=eps) and
// depth cap t the driver's own Phase-2 fAfun uses, deterministic Lanczos on
// the same Omega2) and the fold's probe_sum term, then verify the driver's
// post-fold t2 equals t2 = (t2_pre*s + probe_sum) / (s + b_cert). Reuses
// AutoBudgetClosesAndEstimates's setup, whose easy spectrum certifies the
// probe (b_cert == b).
TEST_F(TestFunNystromPP, AutoProbeReuseFolds) {
    using T = double;
    const int64_t n = 400;
    const int64_t m_budget = 700;
    const T eps = 1e-3;

    T *G0 = randn<T>(n, n, /*seed=*/61);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandBLAS::RNGState<RNG> state(29);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);

    ASSERT_TRUE(driver.auto_probe_converged) << "reuse mechanics require a certified probe";
    const int64_t k = driver.auto_k, s = driver.auto_s, t = driver.auto_t;
    const int64_t b = driver.auto_probe_block;
    ASSERT_LT(k, n);
    ASSERT_NE(driver.Omega2_buf, nullptr);
    ASSERT_NE(driver.auto_probe_buf, nullptr);

    // t2 BEFORE the fold: independent scalar LanczosQFA at the SAME adaptive
    // settings (adaptive=true, rtol=eps) and depth cap t the driver's own
    // Phase-2 fAfun invokes on this->auto_sqfa - deterministic Lanczos on
    // identical inputs must reproduce it.
    RandLAPACK::LanczosQFA<T> sq_ref;
    sq_ref.adaptive = true; sq_ref.adaptive_rtol = eps;
    T *qf_ref = new T[s];
    sq_ref.call(A_op, driver.Omega2_buf, n, s, fscalar, t, qf_ref);
    T tr_AOmega = 0;
    for (int64_t j = 0; j < s; ++j) tr_AOmega += qf_ref[j];

    T *Y2 = new T[k * s];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, s, n, (T)1, driver.U, n, driver.Omega2_buf, n, (T)0, Y2, k);
    T tr_AhatOmega = 0;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < k; ++i) {
            T v = Y2[i + j * k];
            tr_AhatOmega += fscalar(driver.lambda[i]) * v * v;
        }
    T t2_pre = (tr_AOmega - tr_AhatOmega) / (T)s;

    // probe_sum: the fold's contribution from the CERTIFIED probe columns
    // (mask = auto_probe_cert; all b here, since auto_probe_converged).
    T *Yp = new T[k * b];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, b, n, (T)1, driver.U, n, driver.auto_probe_buf, n, (T)0, Yp, k);
    T probe_sum = 0;
    int64_t b_cert = 0;
    for (int64_t j = 0; j < b; ++j) {
        if (!driver.auto_probe_cert[j]) continue;
        T ghat = 0;
        for (int64_t i = 0; i < k; ++i) {
            T v = Yp[i + j * k];
            ghat += fscalar(driver.lambda[i]) * v * v;
        }
        probe_sum += driver.auto_probe_gauss[j] - ghat;
        ++b_cert;
    }
    ASSERT_EQ(b_cert, b) << "auto_probe_converged implies every column certified";

    T t2_expected = (t2_pre * (T)s + probe_sum) / (T)(s + b_cert);
    T rel = std::abs(t2_expected - t2) / std::max(std::abs(t2), (T)1e-12);
    std::printf("auto probe reuse mechanics: t2_pre=%.10e probe_sum=%.10e t2_expected=%.10e "
                "t2_driver=%.10e rel=%.3e (s=%ld b=%ld)\n",
                t2_pre, probe_sum, t2_expected, t2, rel, (long)s, (long)b);
    EXPECT_LT(rel, 1e-9);
    delete[] G0; delete[] A; delete[] qf_ref; delete[] Y2; delete[] Yp;
}

// ===== C1: f_zero exercised through the auto tier's probe-reuse fold =======
// Both knob-free tiers accept f_zero and, when the probe certifies, apply a
// SECOND, distinct zero-fill correction inside the reuse-fold arithmetic
// (fold_probe_reuse's apply_fzero branch, rl_fun_nystrom_pp.hh:539-546) -
// separate from the zero-fill term the expert call() itself applies
// (FZeroPathMatchesDenseTruth's coverage). Extends AutoProbeReuseFolds's
// independent-reproduction pattern with a finite f_zero: f = log(x+2),
// f_zero = log(2) (same convention as FZeroPathMatchesDenseTruth), and the
// manual t2_pre / probe_sum computations add the same fz*(g_sq - y_sq)
// zero-fill terms the driver's expert call() and fold_probe_reuse apply, so
// a sign error, a g_sq/y_sq swap, or a missing apply_fzero guard in the fold
// would be caught bit-for-bit rather than diluted into a loose end-to-end
// accuracy bound.
TEST_F(TestFunNystromPP, AutoProbeReuseFoldsWithFZero) {
    using T = double;
    const int64_t n = 400;
    const int64_t m_budget = 700;
    const T eps = 1e-3;

    T *G0 = randn<T>(n, n, /*seed=*/61);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::log(x + (T)2); };
    const T f_zero = std::log((T)2);

    RandBLAS::RNGState<RNG> state(29);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    driver.call(A_op, fscalar, m_budget, eps, state, t1, t2, std::optional<T>(f_zero));

    ASSERT_TRUE(driver.auto_probe_converged) << "reuse mechanics require a certified probe";
    const int64_t k = driver.auto_k, s = driver.auto_s, t = driver.auto_t;
    const int64_t b = driver.auto_probe_block;
    ASSERT_LT(k, n);
    ASSERT_NE(driver.Omega2_buf, nullptr);
    ASSERT_NE(driver.auto_probe_buf, nullptr);

    RandLAPACK::LanczosQFA<T> sq_ref;
    sq_ref.adaptive = true; sq_ref.adaptive_rtol = eps;
    T *qf_ref = new T[s];
    sq_ref.call(A_op, driver.Omega2_buf, n, s, fscalar, t, qf_ref);
    T tr_AOmega = 0;
    for (int64_t j = 0; j < s; ++j) tr_AOmega += qf_ref[j];

    T *Y2 = new T[k * s];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, s, n, (T)1, driver.U, n, driver.Omega2_buf, n, (T)0, Y2, k);
    T tr_AhatOmega = 0;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < k; ++i) {
            T v = Y2[i + j * k];
            tr_AhatOmega += fscalar(driver.lambda[i]) * v * v;
        }
    // Expert call()'s OWN zero-fill term on the Phase-2 Omega2 block
    // (rl_fun_nystrom_pp.hh:730-733) - distinct from the fold's term below.
    {
        T omega_fro_sq = blas::dot(n * s, driver.Omega2_buf, 1, driver.Omega2_buf, 1);
        T y2_fro_sq    = blas::dot(k * s, Y2, 1, Y2, 1);
        tr_AhatOmega += f_zero * (omega_fro_sq - y2_fro_sq);
    }
    T t2_pre = (tr_AOmega - tr_AhatOmega) / (T)s;

    // probe_sum WITH the fold's own zero-fill term (fold_probe_reuse:539-546).
    T *Yp = new T[k * b];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, b, n, (T)1, driver.U, n, driver.auto_probe_buf, n, (T)0, Yp, k);
    T probe_sum = 0;
    int64_t b_cert = 0;
    for (int64_t j = 0; j < b; ++j) {
        if (!driver.auto_probe_cert[j]) continue;
        const T *yj = Yp + j * k;
        T ghat = 0;
        for (int64_t i = 0; i < k; ++i) ghat += fscalar(driver.lambda[i]) * yj[i] * yj[i];
        const T *gj  = driver.auto_probe_buf + j * n;
        T g_sq = blas::dot(n, gj, 1, gj, 1);
        T y_sq = blas::dot(k, yj, 1, yj, 1);
        ghat += f_zero * (g_sq - y_sq);
        probe_sum += driver.auto_probe_gauss[j] - ghat;
        ++b_cert;
    }
    ASSERT_EQ(b_cert, b) << "auto_probe_converged implies every column certified";

    T t2_expected = (t2_pre * (T)s + probe_sum) / (T)(s + b_cert);
    T rel = std::abs(t2_expected - t2) / std::max(std::abs(t2), (T)1e-12);
    std::printf("auto f_zero probe reuse: t2_pre=%.10e probe_sum=%.10e t2_expected=%.10e "
                "t2_driver=%.10e rel=%.3e (s=%ld b=%ld f0=%.4f)\n",
                t2_pre, probe_sum, t2_expected, t2, rel, (long)s, (long)b, f_zero);
    EXPECT_LT(rel, 1e-9);
    delete[] G0; delete[] A; delete[] qf_ref; delete[] Y2; delete[] Yp;
}

// Regression for the fixed depth cap of 200 (removed 2026-08): on a hard
// spectrum with a tight eps the certified probe must be free to go deeper
// than 200 when n and the budget allow it. With the old cap this probe
// pinned at exactly 200 and the oracle bias floored above eps, so no budget
// could recover the target accuracy (the kappa >= 1e6 cells of the 2026-07
// campaign). Under the redesigned allocation the probe's cap is
// min(n, max(2, floor(auto_probe_frac*B/b))) = min(400, 625) = 400 here, so
// the probe runs to the full n = 400 (t is then the MEDIAN certified depth,
// or the reached depth capped by m_rem/(2*s_min) when uncertified - both
// exceed 200 on this spectrum). Geometric spectrum kappa = 1e6,
// f = log(1+x): the certified depth wants several hundred at this eps.
TEST_F(TestFunNystromPP, AutoProbeDepthNotFixedCapped) {
    using T = double;
    const int64_t n = 400;
    const int64_t m_budget = 20000;
    const T eps = 1e-6;
    const T kappa = 1e6;

    // Diagonal A with a geometric spectrum, lambda_i = kappa^{i/(n-1)} in [1, kappa].
    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] = std::pow(kappa, (T)i / (T)(n - 1));
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::log1p(std::max(x, (T)0)); };

    RandBLAS::RNGState<RNG> state(31);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    (void)driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);

    std::printf("auto depth regression: probed t=%ld (old fixed cap was 200)\n",
                (long)driver.auto_t);
    EXPECT_GT(driver.auto_t, 200);   // the old cap would pin this at exactly 200
    // Probe-fraction cap still binds: b * min(n, floor(0.125*B/b)) matvecs at most.
    EXPECT_LE(driver.auto_probe_matvecs,
              4 * std::min(n, (int64_t)(0.125 * m_budget / 4)));
    EXPECT_GE(driver.auto_s, 4);     // the s_min floor holds even at this depth
    const int64_t spend = driver.auto_probe_matvecs
                        + driver.auto_k + driver.auto_oracle_matvecs;
    EXPECT_LE(spend, m_budget);      // budget closure unchanged by the deeper probe
    delete[] A;
}

// Infeasible inputs must throw with a descriptive message, not proceed. The
// redesigned tier's feasibility floor is
//   B_min = max(2b + 2*s_min, ceil(2*s_min / (1 - auto_probe_frac)))
// (probe block b at the depth-2 floor, plus s_min = 4 depth-1 Hutchinson
// probes and one unit of rank surviving the probe-fraction cut). With the
// defaults b = 4, s_min = 4, frac = 0.125: B_min = max(16, 10) = 16. The
// boundary is tested exactly: B = 15 throws, B = 16 runs.
TEST_F(TestFunNystromPP, AutoInfeasibleThrows) {
    using T = double;
    const int64_t n = 100;
    T *G0 = randn<T>(n, n, /*seed=*/67);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    {   // far below the floor
        RandBLAS::RNGState<RNG> state(5);
        EXPECT_THROW(driver.call(A_op, fscalar, (int64_t)5, (T)1e-3, state, t1, t2),
                     std::invalid_argument);
    }
    {   // exactly one below the B_min = 16 floor
        RandBLAS::RNGState<RNG> state(5);
        EXPECT_THROW(driver.call(A_op, fscalar, (int64_t)15, (T)1e-3, state, t1, t2),
                     std::invalid_argument);
    }
    {   // exactly at the floor: must run (degenerate but funded allocation)
        RandBLAS::RNGState<RNG> state(5);
        T est = 0;
        EXPECT_NO_THROW(est = driver.call(A_op, fscalar, (int64_t)16, (T)1e-3,
                                          state, t1, t2));
        EXPECT_TRUE(std::isfinite(est));
        EXPECT_GE(driver.auto_s, 1);
        EXPECT_GE(driver.auto_k, 1);
        std::printf("auto at B_min=16: k=%ld s=%ld t=%ld probe=%ld oracle=%ld est=%.3e\n",
                    (long)driver.auto_k, (long)driver.auto_s, (long)driver.auto_t,
                    (long)driver.auto_probe_matvecs, (long)driver.auto_oracle_matvecs, est);
    }
    {   // eps outside (0, 1)
        RandBLAS::RNGState<RNG> state(5);
        EXPECT_THROW(driver.call(A_op, fscalar, (int64_t)1000, (T)0, state, t1, t2),
                     std::invalid_argument);
    }
    delete[] G0; delete[] A;
}

// ===== Auto tier: a throw from inside the probe leaves converged FALSE =====
// M7 regression. The defensive reset block (just before the depth probe)
// must leave auto_probe_converged FALSE, not its declared class default of
// TRUE (:184) - the real value is only set from auto_sqfa.all_certified
// AFTER the probe call returns. If something throws in between, a caller
// catching it must not read "converged == true, probe_matvecs == 0", which
// looks like "converged instantly".
// None of the budget/eps guards reach that window (they all fire before the
// reset), and LanczosQFA::call never throws through them either, so the
// throw here is forced via auto_sqfa.check_every = 0 - validated deep inside
// the very oracle call the reset is protecting against.
TEST_F(TestFunNystromPP, AutoProbeThrowLeavesConvergedFalse) {
    using T = double;
    const int64_t n = 50;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(x); };

    RandLAPACK::FunNystromPP<T> driver;
    driver.auto_sqfa.check_every = 0;   // invalid; caught inside auto_sqfa.call()
    RandBLAS::RNGState<RNG> state(101);
    T t1 = 0, t2 = 0;
    try {
        driver.call(A_op, fscalar, (int64_t)1000, (T)1e-3, state, t1, t2);
        FAIL() << "expected std::invalid_argument (check_every)";
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("check_every"), std::string::npos) << msg;
        EXPECT_FALSE(driver.auto_probe_converged)
            << "reset must leave converged FALSE so a caller reading state "
               "after this throw does not see \"converged instantly\"";
        EXPECT_EQ(driver.auto_probe_matvecs, 0);
    }
    delete[] A;
}

// ===== Driver reuse across calls ============================================
//
// The benchmark makes thousands of calls against one matrix. Hoisting the
// driver out of the per-call path (persistent-handle MEX) is only sound if a
// reused FunNystromPP carries no state between calls. util::upsize buffers grow
// but never shrink, so the risk is real: an oversized buffer from a previous
// larger (k, s), or a timer/counter left over from a previous branch.
//
// These tests pin the invariant in the library's own CI, independent of MATLAB.

// A reused driver must produce BIT-IDENTICAL results to fresh instances, for a
// call sequence whose (k, s) GROWS THEN SHRINKS. Monotone-growing k never
// exercises the oversized-buffer path, which is exactly where the bug would be.
TEST_F(TestFunNystromPP, ReuseAcrossCallsIsBitIdentical) {
    using T = double;
    const int64_t n = 60, q = 2;
    T *A = randn<T>(n, n, /*seed=*/21);
    for (int64_t j = 0; j < n; ++j)                     // make it PSD-ish + symmetric
        for (int64_t i = 0; i < n; ++i)
            A[i + j * n] = A[i + j * n] + A[j + i * n];
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)2 * n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    // grow, shrink, grow again. k stays >= the default vec_nnz (8): the SASO
    // test matrix requires vec_nnz <= k, so smaller ranks are not a legal
    // configuration rather than an untested one.
    const std::vector<std::pair<int64_t,int64_t>> tuples = {
        {10, 20}, {25, 60}, {8, 12}, {30, 40}, {9, 10}, {25, 60}
    };

    std::vector<T> fresh_est, fresh_t1, fresh_t2;
    for (auto [k, s] : tuples) {
        RandLAPACK::FunNystromPP<T> d1;
        RandBLAS::RNGState<RNG> st(101);
        T t1 = 0, t2 = 0;
        fresh_est.push_back(d1.call(A_op, fAfun, fscalar, k, s, q, st, nullptr, t1, t2));
        fresh_t1.push_back(t1); fresh_t2.push_back(t2);
    }

    RandLAPACK::FunNystromPP<T> shared;
    for (size_t i = 0; i < tuples.size(); ++i) {
        auto [k, s] = tuples[i];
        RandBLAS::RNGState<RNG> st(101);
        T t1 = 0, t2 = 0;
        T est = shared.call(A_op, fAfun, fscalar, k, s, q, st, nullptr, t1, t2);
        EXPECT_EQ(est, fresh_est[i]) << "reuse diverged at tuple " << i
                                     << " (k=" << k << ", s=" << s << ")";
        EXPECT_EQ(t1, fresh_t1[i]) << "t1 diverged at tuple " << i;
        EXPECT_EQ(t2, fresh_t2[i]) << "t2 diverged at tuple " << i;
    }
    delete[] A;
}

// ===== Knob-free tiers: driver reuse with a CHANGING n across calls ========
// ReuseAcrossCallsIsBitIdentical (above) and ScalarQFAShrinkingReuseBitIdentical
// (below) both hold the operator dimension n FIXED across a call sequence -
// only the algorithmic knobs shrink/grow. This matters specifically for the
// knob-free tiers, whose n-sized scratch (auto_probe_buf at n*b, Omega2_buf
// at m*s, etc.) is grown via util::upsize (grow-only) and whose per-call
// arithmetic (n/t_safe, n-k, n/b block-Krylov caps) reads n fresh from
// A_op.dim each call - nothing so far proves that a LARGER previous n's
// buffer contents don't leak into a SMALLER n's run (e.g. stale tail data
// past the new, shorter column length being read by something that indexes
// past the new n without a full re-fill). A reused driver, called first on a
// large-n operator then a smaller, independent one, must match a FRESH
// driver run only on the small operator - same bit-identical-reuse pattern
// as ReuseAcrossCallsIsBitIdentical, with a fresh RNGState of the SAME seed
// reconstructed per call (isolating buffer-reuse effects from RNG-stream
// continuation, exactly as that test does).
TEST_F(TestFunNystromPP, AutoTierReuseAcrossDifferentN) {
    using T = double;
    const T eps = 1e-2;
    const int64_t m_big = 300, m_small = 100;

    const int64_t n_big = 200;
    T *G0_big = randn<T>(n_big, n_big, /*seed=*/601);
    T *A_big  = new T[n_big * n_big];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n_big, n_big, (T)1, G0_big, n_big, (T)0, A_big, n_big);
    for (int64_t i = 0; i < n_big; ++i) A_big[i + i * n_big] += (T)n_big;
    for (int64_t j = 0; j < n_big; ++j)
        for (int64_t i = j + 1; i < n_big; ++i) A_big[i + j * n_big] = A_big[j + i * n_big];
    linops::ExplicitSymLinOp<T> A_big_op(n_big, blas::Uplo::Upper, A_big, n_big, Layout::ColMajor);

    const int64_t n_small = 50;
    T *G0_small = randn<T>(n_small, n_small, /*seed=*/607);
    T *A_small  = new T[n_small * n_small];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n_small, n_small, (T)1, G0_small, n_small, (T)0, A_small, n_small);
    for (int64_t i = 0; i < n_small; ++i) A_small[i + i * n_small] += (T)n_small;
    for (int64_t j = 0; j < n_small; ++j)
        for (int64_t i = j + 1; i < n_small; ++i) A_small[i + j * n_small] = A_small[j + i * n_small];
    linops::ExplicitSymLinOp<T> A_small_op(n_small, blas::Uplo::Upper, A_small, n_small, Layout::ColMajor);

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    // Shared driver: big call (grows every n-sized buffer to n_big), THEN a
    // small, independent call.
    RandLAPACK::FunNystromPP<T> shared;
    {
        RandBLAS::RNGState<RNG> st(611);
        T bt1 = 0, bt2 = 0;
        shared.call(A_big_op, fscalar, m_big, eps, st, bt1, bt2);
    }
    RandBLAS::RNGState<RNG> st_small(617);
    T sh_t1 = 0, sh_t2 = 0;
    T sh_est = shared.call(A_small_op, fscalar, m_small, eps, st_small, sh_t1, sh_t2);

    // Fresh driver: only the small call, same seed.
    RandLAPACK::FunNystromPP<T> fresh;
    RandBLAS::RNGState<RNG> st_fresh(617);
    T fr_t1 = 0, fr_t2 = 0;
    T fr_est = fresh.call(A_small_op, fscalar, m_small, eps, st_fresh, fr_t1, fr_t2);

    std::printf("auto reuse across n: shared(n_big=%ld then n_small=%ld) est=%.10e vs "
                "fresh(n_small only) est=%.10e\n", (long)n_big, (long)n_small, sh_est, fr_est);
    EXPECT_EQ(sh_est, fr_est);
    EXPECT_EQ(sh_t1, fr_t1);
    EXPECT_EQ(sh_t2, fr_t2);
    EXPECT_EQ(shared.auto_k, fresh.auto_k);
    EXPECT_EQ(shared.auto_s, fresh.auto_s);
    EXPECT_EQ(shared.auto_t, fresh.auto_t);
    EXPECT_EQ(shared.auto_probe_matvecs,  fresh.auto_probe_matvecs);
    EXPECT_EQ(shared.auto_oracle_matvecs, fresh.auto_oracle_matvecs);
    EXPECT_EQ(shared.auto_probe_converged,  fresh.auto_probe_converged);
    EXPECT_EQ(shared.auto_phase2_certified, fresh.auto_phase2_certified);
    delete[] G0_big; delete[] A_big; delete[] G0_small; delete[] A_small;
}

TEST_F(TestFunNystromPP, AdaptiveTierReuseAcrossDifferentN) {
    using T = double;
    const T eps = 5e-2;

    const int64_t n_big = 200;
    T *G0_big = randn<T>(n_big, n_big, /*seed=*/619);
    T *A_big  = new T[n_big * n_big];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n_big, n_big, (T)1, G0_big, n_big, (T)0, A_big, n_big);
    for (int64_t i = 0; i < n_big; ++i) A_big[i + i * n_big] += (T)n_big;
    for (int64_t j = 0; j < n_big; ++j)
        for (int64_t i = j + 1; i < n_big; ++i) A_big[i + j * n_big] = A_big[j + i * n_big];
    linops::ExplicitSymLinOp<T> A_big_op(n_big, blas::Uplo::Upper, A_big, n_big, Layout::ColMajor);

    const int64_t n_small = 60;
    T *G0_small = randn<T>(n_small, n_small, /*seed=*/631);
    T *A_small  = new T[n_small * n_small];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n_small, n_small, (T)1, G0_small, n_small, (T)0, A_small, n_small);
    for (int64_t i = 0; i < n_small; ++i) A_small[i + i * n_small] += (T)n_small;
    for (int64_t j = 0; j < n_small; ++j)
        for (int64_t i = j + 1; i < n_small; ++i) A_small[i + j * n_small] = A_small[j + i * n_small];
    linops::ExplicitSymLinOp<T> A_small_op(n_small, blas::Uplo::Upper, A_small, n_small, Layout::ColMajor);

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> shared;
    {
        RandBLAS::RNGState<RNG> st(641);
        T bt1 = 0, bt2 = 0;
        shared.call(A_big_op, fscalar, eps, st, bt1, bt2);
    }
    RandBLAS::RNGState<RNG> st_small(643);
    T sh_t1 = 0, sh_t2 = 0;
    T sh_est = shared.call(A_small_op, fscalar, eps, st_small, sh_t1, sh_t2);

    RandLAPACK::FunNystromPP<T> fresh;
    RandBLAS::RNGState<RNG> st_fresh(643);
    T fr_t1 = 0, fr_t2 = 0;
    T fr_est = fresh.call(A_small_op, fscalar, eps, st_fresh, fr_t1, fr_t2);

    std::printf("adaptive reuse across n: shared(n_big=%ld then n_small=%ld) est=%.10e vs "
                "fresh(n_small only) est=%.10e\n", (long)n_big, (long)n_small, sh_est, fr_est);
    EXPECT_EQ(sh_est, fr_est);
    EXPECT_EQ(sh_t1, fr_t1);
    EXPECT_EQ(sh_t2, fr_t2);
    EXPECT_EQ(shared.adaptive_k, fresh.adaptive_k);
    EXPECT_EQ(shared.adaptive_s, fresh.adaptive_s);
    EXPECT_EQ(shared.adaptive_t, fresh.adaptive_t);
    EXPECT_EQ(shared.adaptive_probe_matvecs,  fresh.adaptive_probe_matvecs);
    EXPECT_EQ(shared.adaptive_oracle_matvecs, fresh.adaptive_oracle_matvecs);
    EXPECT_EQ(shared.adaptive_probe_certified,  fresh.adaptive_probe_certified);
    EXPECT_EQ(shared.adaptive_phase2_certified, fresh.adaptive_phase2_certified);
    delete[] G0_big; delete[] A_big; delete[] G0_small; delete[] A_small;
}

// spend_cap_split: admissible; rank at least half the budget unless that would
// leave fewer than s_min probes; leftover below one probe unless a dimension
// clamp binds; s < s_min exactly when no admissible pair with s >= s_min exists.
TEST_F(TestFunNystromPP, SpendCapSplitInvariants) {
    const int64_t s_min = 4;
    int64_t cases = 0;
    for (int64_t n : {5, 6, 8, 30, 97, 1000})
    for (int64_t t : {1, 3, 7, 20})
    for (int64_t avail : {5, 6, 17, 40, 128, 333, 2000})
    for (int64_t q : {1, 2})
    for (bool blk : {false, true}) {
        const auto sp = RandLAPACK::detail::spend_cap_split(avail, t, n, q, s_min, blk);
        bool any = false;   // does any (k >= 1, s >= s_min) pair fit?
        for (int64_t k = 1; k <= std::max((int64_t)1, n / 2) && !any; ++k) {
            int64_t s = std::min((avail - q * k) / t, n - k);
            if (blk) s = std::min(s, n / t);
            any = s >= s_min;
        }
        if (sp.s < s_min) { EXPECT_FALSE(any) << "n=" << n << " t=" << t << " avail=" << avail; continue; }
        EXPECT_TRUE(any);
        EXPECT_GE(sp.k, 1);
        EXPECT_LE(sp.k, std::max((int64_t)1, n / 2));
        EXPECT_LE(sp.k + sp.s, n);
        EXPECT_LE(q * sp.k + sp.s * t, avail);
        if (blk) EXPECT_LE(sp.s * t, n);
        const int64_t half = std::min({std::max((int64_t)1, n / 2), n - s_min, avail / (2 * q), (avail - s_min * t) / q});
        EXPECT_GE(sp.k, std::max((int64_t)1, half));
        const bool clamped = (sp.k == std::max((int64_t)1, n / 2)) || (sp.k + sp.s == n) || (blk && sp.s == n / t);
        if (!clamped) EXPECT_LT(avail - q * sp.k - sp.s * t, std::max(t, q)) << "n=" << n << " t=" << t << " avail=" << avail;
        ++cases;
    }
    EXPECT_GT(cases, 0);
}

// adaptive_spend_cap: spends the cap up to integer rounding, runs every Phase-2
// probe to the probe's depth, reports Phase 2 as unchecked, and needs a cap.
// The default path is left untouched (compared against a fresh object).
TEST_F(TestFunNystromPP, AdaptiveSpendCapSpendsTheCap) {
    using T = double;
    const int64_t n = 200, cap = 512;
    T *G0 = randn<T>(n, n, /*seed=*/701);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::log1p(std::max(x, (T)0)); };

    for (bool scalar : {false, true})
    for (T eps : {(T)1e-3, (T)0.3}) {      // 0.3: the target's request fits under the cap
        RandLAPACK::FunNystromPP<T> shipped, full;
        shipped.adaptive_use_scalar = full.adaptive_use_scalar = scalar;
        shipped.adaptive_cap_rank_fraction = full.adaptive_cap_rank_fraction = (T)0.5;
        full.adaptive_spend_cap = true;
        T a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        RandBLAS::RNGState<RNG> st1(709), st2(709);
        const T e1 = shipped.call(A_op, fscalar, eps, st1, a1, a2, cap);
        const T e2 = full.call(A_op, fscalar, eps, st2, b1, b2, cap);
        const int64_t spent_full = full.adaptive_probe_matvecs + full.adaptive_k + full.adaptive_oracle_matvecs;
        const int64_t spent_ship = shipped.adaptive_probe_matvecs + shipped.adaptive_k + shipped.adaptive_oracle_matvecs;
        std::printf("spend_cap scalar=%d eps=%.0e: shipped spent %ld (k=%ld s=%ld t=%ld), full spent %ld (k=%ld s=%ld t=%ld) of %ld; est %.8e vs %.8e\n",
                    (int)scalar, (double)eps, (long)spent_ship, (long)shipped.adaptive_k, (long)shipped.adaptive_s,
                    (long)shipped.adaptive_t, (long)spent_full, (long)full.adaptive_k, (long)full.adaptive_s,
                    (long)full.adaptive_t, (long)cap, e1, e2);
        EXPECT_TRUE(std::isfinite(e1)); EXPECT_TRUE(std::isfinite(e2));
        EXPECT_EQ(full.adaptive_probe_matvecs, shipped.adaptive_probe_matvecs);   // same probe
        EXPECT_EQ(full.adaptive_t, shipped.adaptive_t);
        EXPECT_LE(spent_full, cap);
        const bool dim_bound = full.adaptive_k == n / 2 || full.adaptive_k + full.adaptive_s == n ||
                               (!scalar && full.adaptive_s == n / full.adaptive_t);
        if (!dim_bound) EXPECT_GT(spent_full, cap - full.adaptive_t);          // leftover < one probe
        EXPECT_GE(spent_full, spent_ship);
        const int64_t a_post = cap - full.adaptive_probe_matvecs;
        EXPECT_GE(full.adaptive_k, std::min({n / 2, a_post / 2, a_post - 4 * full.adaptive_t}));   // half to the rank
        EXPECT_EQ(full.adaptive_oracle_matvecs, full.adaptive_s * full.adaptive_t);   // no early stop
        EXPECT_FALSE(full.adaptive_phase2_checked);
        EXPECT_FALSE(full.adaptive_phase2_certified);
        EXPECT_TRUE(shipped.adaptive_phase2_checked);
        EXPECT_TRUE(full.adaptive_bqfa.adaptive);        // restored after the call
        EXPECT_TRUE(full.auto_sqfa.adaptive);
    }
    {
        RandLAPACK::FunNystromPP<T> d; d.adaptive_spend_cap = true;
        RandBLAS::RNGState<RNG> st(1); T t1 = 0, t2 = 0;
        EXPECT_THROW(d.call(A_op, fscalar, (T)1e-3, st, t1, t2), std::invalid_argument);
    }
    {   // n = 6 and a small target: the target-derived split leaves fewer than four probes, which
        // must not stop the switch, since the cap funds (k, s) = (2, 4) (it threw before)
        const int64_t m = 6;
        T Asm[36] = {0};
        for (int64_t i = 0; i < m; ++i) Asm[i + i * m] = (T)(i + 1);
        linops::ExplicitSymLinOp<T> S_op(m, blas::Uplo::Upper, Asm, m, Layout::ColMajor);
        RandLAPACK::FunNystromPP<T> d; d.adaptive_use_scalar = true; d.adaptive_spend_cap = true;
        d.adaptive_cap_rank_fraction = (T)0.5;
        RandBLAS::RNGState<RNG> st(3); T t1 = 0, t2 = 0;
        EXPECT_NO_THROW(d.call(S_op, fscalar, (T)1e-6, st, t1, t2, (int64_t)200));
        EXPECT_GE(d.adaptive_s, 4); EXPECT_GE(d.adaptive_k, 1); EXPECT_LE(d.adaptive_k + d.adaptive_s, m);
    }
    delete[] G0; delete[] A;
}

// probe_dist = Rademacher must produce ONLY +-1 entries (sign of a RandBLAS
// Uniform fill) with column norm exactly sqrt(n) by construction, and it must
// route through the SAME fill_probe_block the auto-tier probe fill uses (no
// per-site drift). Checked at the expert-overload Omega2 site since Omega2_buf
// is inspectable after call().
TEST_F(TestFunNystromPP, ProbeDistRademacherSignEntriesUnitNorm) {
    using T = double;
    const int64_t n = 50, k = 10, s = 6, q = 1;
    T *A = randn<T>(n, n, /*seed=*/41);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            A[i + j * n] = A[i + j * n] + A[j + i * n];
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)2 * n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    RandLAPACK::FunNystromPP<T> driver;
    driver.probe_dist = RandLAPACK::ProbeDist::Rademacher;
    RandBLAS::RNGState<RNG> st(43);
    T t1 = 0, t2 = 0;
    driver.call(A_op, fAfun, fscalar, k, s, q, st, nullptr, t1, t2);

    ASSERT_NE(driver.Omega2_buf, nullptr);
    const T sqrt_n = std::sqrt((T)n);
    for (int64_t j = 0; j < s; ++j) {
        T *col = driver.Omega2_buf + j * n;
        T ssq = 0;
        for (int64_t i = 0; i < n; ++i) {
            EXPECT_TRUE(col[i] == (T)1 || col[i] == (T)-1)
                << "col " << j << " row " << i << " = " << col[i];
            ssq += col[i] * col[i];
        }
        EXPECT_NEAR(std::sqrt(ssq), sqrt_n, 1e-10) << "col " << j << " norm";
        T nrm = blas::nrm2(n, col, 1);
        EXPECT_NEAR(nrm, sqrt_n, 1e-10) << "col " << j << " blas nrm2";
    }
    delete[] A;
}

// t_fafun_ms must be CLEARED on the k == n path, not left at the previous
// call's value. Consumers compute assembly = t_phase2_ms - t_fafun_ms, so a
// stale value makes that negative. Regression test for the fix in
// rl_fun_nystrom_pp.hh's Phase-2 skip branch.
TEST_F(TestFunNystromPP, FafunTimerResetAtKEqualsN) {
    using T = double;
    const int64_t n = 40, q = 2;
    T *A = randn<T>(n, n, /*seed=*/23);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            A[i + j * n] = A[i + j * n] + A[j + i * n];
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)2 * n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;

    // First a k < n call, which sets a nonzero t_fafun_ms ...
    RandBLAS::RNGState<RNG> s1(31);
    driver.call(A_op, fAfun, fscalar, /*k=*/10, /*s=*/20, q, s1, nullptr, t1, t2);
    ASSERT_GT(driver.t_fafun_ms, 0.0) << "precondition: k<n call should time the oracle";

    // ... then a k == n call, which skips Phase 2 entirely.
    RandBLAS::RNGState<RNG> s2(31);
    driver.call(A_op, fAfun, fscalar, /*k=*/n, /*s=*/20, q, s2, nullptr, t1, t2);
    EXPECT_EQ(driver.t_phase2_ms, 0.0);
    EXPECT_EQ(driver.t_fafun_ms,  0.0) << "stale t_fafun_ms leaked across the k==n branch";
    EXPECT_GE(driver.t_phase2_ms - driver.t_fafun_ms, 0.0) << "assembly time went negative";
    delete[] A;
}

// A rank below the sketch's vec_nnz must DEGRADE (dense sketch columns), not
// throw. Regression for "(vec_nnz <= dim_major) was required, but did not hold,
// in function SparseDist", which killed 47 of 221 rungs in the 2026-07-28
// rehearsal and would have hit the real campaign at its smallest budgets
// (k = B/2 = 5 at B = 10) as well as the knob-free auto tier.
TEST_F(TestFunNystromPP, SmallRankBelowVecNnzDoesNotThrow) {
    using T = double;
    const int64_t n = 40, q = 1;
    T *A = randn<T>(n, n, /*seed=*/29);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i) A[i + j * n] = A[i + j * n] + A[j + i * n];
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)2 * n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];

    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T true_tr = true_trace_fa<T>(n, A, fscalar);

    for (int64_t k : {1, 2, 5, 7, 8}) {          // default vec_nnz is 8
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> st(41);
        T t1 = 0, t2 = 0;
        T est = 0;
        ASSERT_NO_THROW(est = driver.call(A_op, fAfun, fscalar, k, /*s=*/12, q,
                                          st, nullptr, t1, t2))
            << "k=" << k << " (< vec_nnz) must degrade, not throw";
        EXPECT_TRUE(std::isfinite(est)) << "k=" << k;
        EXPECT_LT(std::abs(est - true_tr) / true_tr, 0.5) << "k=" << k;
    }
    delete[] A;
}

// ===== Float instantiation ==================================================
// The expert driver path must instantiate and behave at T = float. The
// corresponding scalar certificate check is FloatCertifiedQFA in
// test_lanczos_qfa.cc. Well-conditioned
// diagonal SPD (kappa = 1e3) so the f32 Nystrom shift nu ~ n*eps_f*||A||_2 is
// harmless against lambda_min = 1 (NystromEVD emits its one-time f32 stderr
// NOTE here; expected). Bounds are float-loose by design.
TEST_F(TestFunNystromPP, FloatExpertPath) {
    using T = float;
    const int64_t n = 200, k = 40, s = 100, q = 2;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    T true_tr = 0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)1 + (T)999 * (T)i / (T)(n - 1);   // linear in [1, 1000]
        true_tr += std::sqrt(A[i + i * n]);
    }
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);

    RandBLAS::RNGState<RNG> state(173);
    T *Omega2 = randn<T>(n, s, /*seed=*/179);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fAfun, fscalar, k, s, q, state, Omega2, t1, t2);
    T err = std::abs(est - true_tr) / true_tr;
    std::printf("f32 expert: est=%.7e true=%.7e relerr=%.3e (t1=%.3e t2=%.3e)\n",
                est, true_tr, err, t1, t2);
    EXPECT_TRUE(std::isfinite(est));
    EXPECT_LT(err, 0.1);   // loose statistical bound; f32 shift harmless at kappa 1e3

    delete[] A; delete[] Omega2;
}

// ===== Float instantiation: knob-free tiers + block Lanczos-QFA/FA =========
// Extends float coverage beyond the expert path (FloatExpertPath
// above) to the "new adaptive-tier
// surface" this audit was scoped to: both knob-free overloads' depth-probe
// arithmetic, median-depth selection, probe-reuse folding, and the
// compute_adaptive_split clamps. The float BlockLanczosFA / BlockLanczosQFA
// comparison is in test_lanczos_qfa_block.cc. Well-conditioned
// spectra so float roundoff doesn't dominate the signal; tolerances are
// float-loose, following FloatExpertPath's precedent above.
TEST_F(TestFunNystromPP, FloatAutoTierBudgetCloses) {
    using T = float;
    const int64_t n = 150;
    const int64_t m_budget = 500;
    const T eps = (T)1e-2;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    T true_tr = 0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)1 + (T)9 * (T)i / (T)(n - 1);   // linear in [1, 10]
        true_tr += std::sqrt(A[i + i * n]);
    }
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandBLAS::RNGState<RNG> state(419);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);
    T err = std::abs(est - true_tr) / std::abs(true_tr);
    std::printf("f32 auto tier: k=%ld s=%ld t=%ld probe=%ld oracle=%ld conv=%d p2cert=%d relerr=%.3e\n",
                (long)driver.auto_k, (long)driver.auto_s, (long)driver.auto_t,
                (long)driver.auto_probe_matvecs, (long)driver.auto_oracle_matvecs,
                (int)driver.auto_probe_converged, (int)driver.auto_phase2_certified, (double)err);
    EXPECT_TRUE(std::isfinite(est));
    const int64_t spend = driver.auto_probe_matvecs + driver.auto_k + driver.auto_oracle_matvecs;
    EXPECT_LE(spend, m_budget);
    EXPECT_LT(err, (T)0.1);
    delete[] A;
}

TEST_F(TestFunNystromPP, FloatAdaptiveEpsCloses) {
    using T = float;
    const int64_t n = 150;
    const T eps = (T)5e-2;

    T *A = new T[n * n]();
    T true_tr = 0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)1 + (T)9 * (T)i / (T)(n - 1);
        true_tr += std::sqrt(A[i + i * n]);
    }
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandBLAS::RNGState<RNG> state(421);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fscalar, eps, state, t1, t2);
    T err = std::abs(est - true_tr) / std::abs(true_tr);
    std::printf("f32 adaptive tier: k=%ld s=%ld t=%ld probe_mv=%ld oracle_mv=%ld probe_cert=%d p2cert=%d relerr=%.3e\n",
                (long)driver.adaptive_k, (long)driver.adaptive_s, (long)driver.adaptive_t,
                (long)driver.adaptive_probe_matvecs, (long)driver.adaptive_oracle_matvecs,
                (int)driver.adaptive_probe_certified, (int)driver.adaptive_phase2_certified, (double)err);
    EXPECT_TRUE(std::isfinite(est));
    EXPECT_LE(driver.adaptive_k, n / 2);
    EXPECT_GE(driver.adaptive_s, 4);
    EXPECT_LT(err, (T)0.1);
    delete[] A;
}

// ===== f_zero zero-fill path ================================================
// f = log(x + 2) has f(0) = log(2) != 0, so the Persson-anchor convention
// (implicit f(0) = 0) would misestimate tr(f(A)) through the (n - k)-dim
// complement. Passing f_zero opts in to the zero-fill correction: t1 gains
// (n - k) f(0) and t2 subtracts the projector-complement term, and the
// estimate must land on the dense truth. The documented invalid_argument on
// a non-finite f_zero is pinned alongside.
TEST_F(TestFunNystromPP, FZeroPathMatchesDenseTruth) {
    using T = double;
    const int64_t n = 40, k = 10, s = 300, q = 2;

    T *G0 = randn<T>(n, n, /*seed=*/191);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    auto fscalar = [](T x) { return std::log(x + (T)2); };
    const T f_zero = std::log((T)2);
    T true_tr = true_trace_fa<T>(n, A, fscalar);
    auto fAfun = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);

    RandBLAS::RNGState<RNG> state(193);
    T *Omega2 = randn<T>(n, s, /*seed=*/197);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fAfun, fscalar, k, s, q, state, Omega2, t1, t2,
                        std::optional<T>(f_zero));
    T err = std::abs(est - true_tr) / std::abs(true_tr);
    std::printf("f_zero path: est=%.10e true=%.10e relerr=%.3e (t1=%.3e t2=%.3e, f0=%.4f)\n",
                est, true_tr, err, t1, t2, f_zero);
    EXPECT_LT(err, 0.1);

    // Non-finite f_zero must throw (documented; no silent auto-resolve).
    for (T bad : {std::numeric_limits<T>::infinity(),
                  std::numeric_limits<T>::quiet_NaN()}) {
        RandBLAS::RNGState<RNG> st(193);
        EXPECT_THROW(driver.call(A_op, fAfun, fscalar, k, s, q, st, Omega2,
                                 t1, t2, std::optional<T>(bad)),
                     std::invalid_argument);
    }
    delete[] G0; delete[] A; delete[] Omega2;
}

// ===== Expert use_qfa = true ================================================
// The QFA oracle convention: fAfun fills the s x s quadratic form (the driver
// reads ONLY its diagonal), skipping the f(A)*Omega2 mapback. With the same
// RNG state and the same explicit Omega2, Phase 1 is identical between the
// exact-oracle and QFA runs (t1 bit-equal), and the estimates differ only by
// the deep fixed-depth QFA's truncation - near machine precision on this
// well-conditioned spectrum.
TEST_F(TestFunNystromPP, ExpertUseQfaMatchesExactOracle) {
    using T = double;
    const int64_t n = 60, k = 12, s = 40, q = 2, d = 40;

    T *G0 = randn<T>(n, n, /*seed=*/199);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(x); };
    T *Omega2 = randn<T>(n, s, /*seed=*/211);

    // Exact-oracle run.
    auto fAfun_exact = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    RandLAPACK::FunNystromPP<T> dr_exact;
    RandBLAS::RNGState<RNG> st1(223);
    T t1_e = 0, t2_e = 0;
    T est_exact = dr_exact.call(A_op, fAfun_exact, fscalar, k, s, q, st1,
                                Omega2, t1_e, t2_e);

    // Scalar-QFA oracle run: per-column quadratic forms onto the diagonal.
    RandLAPACK::LanczosQFA<T> sqfa;
    auto fAfun_qfa = [&](int64_t m, int64_t ss, const T *Bblk, T *Y) {
        T *vals = new T[ss];
        sqfa.call(A_op, Bblk, m, ss, fscalar, d, vals);
        for (int64_t j = 0; j < ss; ++j) Y[j + j * ss] = vals[j];
        delete[] vals;
    };
    RandLAPACK::FunNystromPP<T> dr_qfa;
    dr_qfa.use_qfa = true;
    RandBLAS::RNGState<RNG> st2(223);
    T t1_q = 0, t2_q = 0;
    T est_qfa = dr_qfa.call(A_op, fAfun_qfa, fscalar, k, s, q, st2,
                            Omega2, t1_q, t2_q);

    T reldiff = std::abs(est_qfa - est_exact) / std::abs(est_exact);
    std::printf("use_qfa vs exact: est_qfa=%.12e est_exact=%.12e reldiff=%.3e\n",
                est_qfa, est_exact, reldiff);
    EXPECT_EQ(t1_q, t1_e);        // identical Phase 1 (same state, same k, q)
    EXPECT_LT(reldiff, 1e-10);    // depth-40 QFA truncation, well-conditioned
    delete[] G0; delete[] A; delete[] Omega2;
}

// ===== End-to-end driver + Krylov (LanczosFA) oracle ========================
// The production configuration in miniature: the expert driver with a scalar
// Lanczos-FA fAfun at a depth deep enough to converge on this easy spectrum,
// against the same run with the exact dense oracle (same state, same Omega2:
// Phase 1 identical, so the whole difference is the Krylov truncation).
TEST_F(TestFunNystromPP, ExpertLanczosFAOracleMatchesExact) {
    using T = double;
    const int64_t n = 60, k = 12, s = 40, q = 2, d = 50;

    T *G0 = randn<T>(n, n, /*seed=*/227);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(x); };
    T *Omega2 = randn<T>(n, s, /*seed=*/229);

    auto fAfun_exact = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    RandLAPACK::FunNystromPP<T> dr_exact;
    RandBLAS::RNGState<RNG> st1(233);
    T t1_e = 0, t2_e = 0;
    T est_exact = dr_exact.call(A_op, fAfun_exact, fscalar, k, s, q, st1,
                                Omega2, t1_e, t2_e);

    RandLAPACK::LanczosFA<T> lfa;
    auto fAfun_kry = [&](int64_t m, int64_t ss, const T *Bblk, T *Y) {
        lfa.call(A_op, Bblk, m, ss, fscalar, d, Y);
    };
    RandLAPACK::FunNystromPP<T> dr_kry;
    RandBLAS::RNGState<RNG> st2(233);
    T t1_k = 0, t2_k = 0;
    T est_kry = dr_kry.call(A_op, fAfun_kry, fscalar, k, s, q, st2,
                            Omega2, t1_k, t2_k);

    T reldiff = std::abs(est_kry - est_exact) / std::abs(est_exact);
    std::printf("LanczosFA oracle vs exact: est_kry=%.12e est_exact=%.12e reldiff=%.3e\n",
                est_kry, est_exact, reldiff);
    EXPECT_EQ(t1_k, t1_e);        // identical Phase 1
    EXPECT_LT(reldiff, 1e-6);     // depth-50 Krylov converged on this spectrum
    delete[] G0; delete[] A; delete[] Omega2;
}

// ===== Auto tier contracts on a hard spectrum ===============================
// The 2026-08 redesign's pins where the probe CANNOT certify within its
// fraction-capped depth (geometric kappa = 1e6 at eps = 1e-6; the certified
// depth wants ~1000 while the cap allows at most 0.125*B/b):
//   (a) auto_probe_converged == false and auto_s >= 4 (the s_min floor: the
//       uncertified branch caps t at m_rem/(2*s_min), so the split always
//       funds >= 4 probes - never the old s == 2 lock);
//   (b) more budget buys more probes: auto_s nondecreasing across B1 < B2 < B3;
//   (c) the probe never spends past its fraction cap:
//       probe_mv <= ceil(0.125*B) + b slack.
TEST_F(TestFunNystromPP, AutoContractsHardSpectrum) {
    using T = double;
    const int64_t n = 600;
    const T eps = 1e-6, kappa = 1e6;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] = std::pow(kappa, (T)i / (T)(n - 1));
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    int64_t s_prev = 0;
    for (int64_t m_budget : {(int64_t)800, (int64_t)1600, (int64_t)3200}) {
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> state(401);
        T t1 = 0, t2 = 0;
        T est = driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);
        const int64_t spend = driver.auto_probe_matvecs
                            + driver.auto_k + driver.auto_oracle_matvecs;
        std::printf("auto hard B=%ld: probe=%ld conv=%d k=%ld s=%ld t=%ld oracle=%ld spend=%ld est=%.4e\n",
                    (long)m_budget, (long)driver.auto_probe_matvecs,
                    (int)driver.auto_probe_converged, (long)driver.auto_k,
                    (long)driver.auto_s, (long)driver.auto_t,
                    (long)driver.auto_oracle_matvecs, (long)spend, est);
        EXPECT_TRUE(std::isfinite(est));
        EXPECT_FALSE(driver.auto_probe_converged) << "B=" << m_budget;   // (a)
        EXPECT_GE(driver.auto_s, 4)               << "B=" << m_budget;   // (a) s_min floor
        EXPECT_GE(driver.auto_s, s_prev)          << "B=" << m_budget;   // (b)
        s_prev = driver.auto_s;
        EXPECT_LE(driver.auto_probe_matvecs,                             // (c)
                  (int64_t)std::ceil(0.125 * (double)m_budget) + 4) << "B=" << m_budget;
        EXPECT_LE(spend, m_budget) << "B=" << m_budget;
    }
    delete[] A;
}

// Easy-spectrum counterpart (the redesign's certification pins): with a
// near-flat spectrum the probe certifies at a small uniform depth, so both
// flags must report success - auto_probe_converged (the depth probe) AND
// auto_phase2_certified (every Phase-2 oracle column certified at the
// median-depth cap; captured after the oracle runs, since Phase 2 reuses the
// probe's LanczosQFA instance).
TEST_F(TestFunNystromPP, AutoEasySpectrumCertifiesBothPhases) {
    using T = double;
    const int64_t n = 400;
    const int64_t m_budget = 700;
    const T eps = 1e-3;

    T *A = new T[n * n]();
    T true_tr = 0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)1 + (T)i / (T)(n - 1);   // linear in [1, 2]
        true_tr += std::sqrt(A[i + i * n]);
    }
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> driver;
    RandBLAS::RNGState<RNG> state(409);
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);
    T err = std::abs(est - true_tr) / true_tr;
    std::printf("auto easy: k=%ld s=%ld t=%ld probe=%ld oracle=%ld conv=%d p2cert=%d relerr=%.3e\n",
                (long)driver.auto_k, (long)driver.auto_s, (long)driver.auto_t,
                (long)driver.auto_probe_matvecs, (long)driver.auto_oracle_matvecs,
                (int)driver.auto_probe_converged, (int)driver.auto_phase2_certified, err);
    EXPECT_TRUE(driver.auto_probe_converged);
    EXPECT_TRUE(driver.auto_phase2_certified);
    EXPECT_LT(err, 1e-2);
    delete[] A;
}

// ===== I7: auto tier's matvec telemetry vs an independently-counted total ==
// Every accounting assertion elsewhere in this file (e.g.
// AutoBudgetClosesAndEstimates's spend <= m_budget) checks one self-reported
// driver counter against another self-reported counter or a budget-derived
// bound - never against how many times A was ACTUALLY applied. Wraps A_op in
// CountingSymLinOp (satisfies SymmetricLinearOperator plus the SketchingOperator
// overload NystromEVD's internal SASO sketch needs) and asserts the counter's
// final value equals the documented invariant (rl_fun_nystrom_pp.hh:178-181)
// auto_probe_matvecs + q*auto_k + auto_oracle_matvecs (q = 1, the auto tier's
// fixed single-pass convention) EXACTLY - closing the loop from outside the
// driver rather than trusting its own self-report.
TEST_F(TestFunNystromPP, AutoMatvecTelemetryMatchesGroundTruth) {
    using T = double;
    const int64_t n = 400;
    const int64_t m_budget = 700;
    const T eps = 1e-3;

    T *G0 = randn<T>(n, n, /*seed=*/61);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    CountingSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandBLAS::RNGState<RNG> state(29);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    driver.call(A_op, fscalar, m_budget, eps, state, t1, t2);

    const int64_t q = 1;   // the auto tier's fixed single-pass Nystrom convention
    const int64_t ground_truth = A_op.apply_count;
    const int64_t reported = driver.auto_probe_matvecs + q * driver.auto_k
                            + driver.auto_oracle_matvecs;
    std::printf("auto matvec telemetry: ground_truth=%ld reported=%ld "
                "(probe=%ld k=%ld oracle=%ld)\n",
                (long)ground_truth, (long)reported, (long)driver.auto_probe_matvecs,
                (long)driver.auto_k, (long)driver.auto_oracle_matvecs);
    EXPECT_EQ(ground_truth, reported);
    delete[] G0; delete[] A;
}

// ===== Eps-targeted adaptive tier: split-helper arithmetic ==================
// detail::compute_adaptive_split is pure arithmetic (no driver, no RNG, no A):
// k = clamp(ceil(k_const*sqrt(t)/eps), 1, n/2), s = max(s_min, ceil(s_const /
// (sqrt(t)*eps))), then s = min(s, n-k), then the block-Krylov guard
// s = min(s, n/t). Five hand-computed (t, eps, n) triples exercise: a
// baseline with no clamp active, the k -> n/2 saturation, the s*t <= n
// clamp cutting s below its s_min floor, non-default k_const/s_const, and
// the s_min floor itself binding.
TEST_F(TestFunNystromPP, AdaptiveSplitHelperMatchesFormula) {
    using T = double;
    struct Case {
        int64_t t; T eps; int64_t n; T k_const; T s_const; int64_t s_min;
        int64_t exp_k, exp_s; const char *note;
    };
    const Case cases[] = {
        // t=4, eps=0.1, n=1000: k=ceil(2/0.1)=20, s=ceil(1/0.2)=5; nothing clamps.
        {4, (T)0.1, 1000, (T)1, (T)1, 4, 20, 5, "baseline, no clamp"},
        // t=4, eps=0.01, n=100: k=ceil(2/0.01)=200 -> saturates at n/2=50;
        // s=ceil(1/0.04)=25, s<=n-k=50, s<=n/t=25: stays 25.
        {4, (T)0.01, 100, (T)1, (T)1, 4, 50, 25, "k saturates at n/2"},
        // t=20, eps=0.9, n=50: k=ceil(sqrt(20)/0.9)=ceil(4.969)=5;
        // s=ceil(1/(sqrt(20)*0.9))=ceil(0.2485)=1 -> s_min floor to 4;
        // s<=n-k=45 (no cut); s<=n/t=50/20=2 - block-Krylov guard cuts s
        // below its own s_min floor.
        {20, (T)0.9, 50, (T)1, (T)1, 4, 5, 2, "s*t<=n clamp undercuts s_min"},
        // t=9, eps=0.2, n=200, k_const=2, s_const=0.5:
        // k=ceil(2*3/0.2)=ceil(30)=30; s=ceil(0.5/(3*0.2))=ceil(0.833)=1 -> 4;
        // s<=n-k=170, s<=n/t=22: stays 4.
        {9, (T)0.2, 200, (T)2, (T)0.5, 4, 30, 4, "non-default k_const/s_const"},
        // t=16, eps=0.05, n=500, s_min=8: k=ceil(4/0.05)=80;
        // s=ceil(1/(4*0.05))=ceil(5)=5 -> floored up to s_min=8;
        // s<=n-k=420, s<=n/t=31: stays 8.
        {16, (T)0.05, 500, (T)1, (T)1, 8, 80, 8, "s_min floor binds"},
    };
    for (const auto &c : cases) {
        auto split = RandLAPACK::detail::compute_adaptive_split(
            c.t, c.eps, c.n, c.k_const, c.s_const, c.s_min);
        std::printf("split[%s]: t=%ld eps=%.3f n=%ld -> k=%ld (exp %ld), s=%ld (exp %ld)\n",
                    c.note, (long)c.t, (double)c.eps, (long)c.n,
                    (long)split.k, (long)c.exp_k, (long)split.s, (long)c.exp_s);
        EXPECT_EQ(split.k, c.exp_k) << c.note;
        EXPECT_EQ(split.s, c.exp_s) << c.note;
    }
}

// ===== Eps-targeted adaptive tier: closes and estimates on an easy spectrum =
// call(A, f, eps, state, ...) with no matvec_cap: the depth probe (block
// Gauss-Radau, Rademacher columns) must certify on this well-conditioned
// spectrum, the derived rank must respect the n/2 cap, the probe count must
// clear the s_min = 4 floor, the Phase-2 oracle's matvec spend must not
// exceed its allocated s*t budget (certified early stopping), and the
// resulting estimate must be accurate.
TEST_F(TestFunNystromPP, AdaptiveEpsClosesAndEstimates) {
    using T = double;
    const int64_t n = 300;
    const T eps = 1e-2;

    T *G0 = randn<T>(n, n, /*seed=*/311);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    T true_tr = true_trace_fa(n, A, fscalar);

    RandBLAS::RNGState<RNG> state(313);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    T est = driver.call(A_op, fscalar, eps, state, t1, t2);

    T err = std::abs(est - true_tr) / std::abs(true_tr);
    std::printf("adaptive: k=%ld s=%ld t=%ld probe_mv=%ld oracle_mv=%ld "
                "probe_cert=%d p2_cert=%d relerr=%.3e\n",
                (long)driver.adaptive_k, (long)driver.adaptive_s, (long)driver.adaptive_t,
                (long)driver.adaptive_probe_matvecs, (long)driver.adaptive_oracle_matvecs,
                (int)driver.adaptive_probe_certified, (int)driver.adaptive_phase2_certified, err);

    EXPECT_TRUE(driver.adaptive_probe_certified);
    EXPECT_TRUE(driver.adaptive_phase2_certified);
    EXPECT_LE(driver.adaptive_k, n / 2);
    EXPECT_GE(driver.adaptive_s, 4);
    // The probe never runs past n and reports its actual depth in adaptive_t;
    // absent an n-cap, matvecs = block_size * depth exactly (one joint block
    // recurrence, not per-column early retirement).
    EXPECT_EQ(driver.adaptive_probe_matvecs, driver.adaptive_probe_block * driver.adaptive_t);
    EXPECT_LE(driver.adaptive_oracle_matvecs, driver.adaptive_s * driver.adaptive_t);
    EXPECT_LT(err, 1e-2);
    delete[] G0; delete[] A;
}

// ===== I7: adaptive tier's matvec telemetry vs an independently-counted total
// Adaptive-tier counterpart to AutoMatvecTelemetryMatchesGroundTruth: wraps
// A_op in CountingSymLinOp and asserts the counter's final value equals
// adaptive_probe_matvecs + q*adaptive_k + adaptive_oracle_matvecs (q = 1)
// exactly, closing the loop from outside the driver.
TEST_F(TestFunNystromPP, AdaptiveMatvecTelemetryMatchesGroundTruth) {
    using T = double;
    const int64_t n = 300;
    const T eps = 1e-2;

    T *G0 = randn<T>(n, n, /*seed=*/311);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    CountingSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandBLAS::RNGState<RNG> state(313);
    RandLAPACK::FunNystromPP<T> driver;
    T t1 = 0, t2 = 0;
    driver.call(A_op, fscalar, eps, state, t1, t2);

    const int64_t q = 1;   // the adaptive tier's fixed single-pass Nystrom convention
    const int64_t ground_truth = A_op.apply_count;
    const int64_t reported = driver.adaptive_probe_matvecs + q * driver.adaptive_k
                            + driver.adaptive_oracle_matvecs;
    std::printf("adaptive matvec telemetry: ground_truth=%ld reported=%ld "
                "(probe=%ld k=%ld oracle=%ld)\n",
                (long)ground_truth, (long)reported, (long)driver.adaptive_probe_matvecs,
                (long)driver.adaptive_k, (long)driver.adaptive_oracle_matvecs);
    EXPECT_EQ(ground_truth, reported);
    delete[] G0; delete[] A;
}

// ===== F16: exact closed form at k == m (small diagonal) ====================
// k == m is the analytic skip point for Phase 2 (rl_fun_nystrom_pp.hh's
// `if (k < m)` guard around the Hutchinson correction): once Phase 1 captures
// the FULL spectrum, f(A) - f(Aat) is exactly zero, t2 == 0 identically, and
// the estimate reduces to Sum f(lambda_i) to roundoff. The eps-targeted
// adaptive overload cannot reach this point on its own - compute_adaptive_
// split caps k at n/2 by construction (see AdaptiveSplitHelperMatchesFormula
// above), so k == n never happens through that path regardless of eps.
// Reached here directly through the EXPERT overload instead: k = n = m, with
// an fAfun that asserts it is never invoked (the guard means it can't be).
TEST_F(TestFunNystromPP, ExactClosedFormSmallDiagonal) {
    using T = double;
    const int64_t n = 8, q = 1;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    T true_sqrt = 0, true_log1p = 0;
    for (int64_t i = 0; i < n; ++i) {
        A[i + i * n] = (T)(i + 1);
        true_sqrt  += std::sqrt((T)(i + 1));
        true_log1p += std::log1p((T)(i + 1));
    }
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    auto never_called = [](int64_t, int64_t, const T*, T*) {
        ADD_FAILURE() << "fAfun must not be invoked when k == m (Phase 2 is skipped)";
    };

    {
        auto fscalar = [](T x) { return std::sqrt(x); };
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> state(347);
        T t1 = 0, t2 = 0;
        T est = driver.call(A_op, never_called, fscalar, /*k=*/n, /*s=*/0, q,
                            state, /*Omega2=*/nullptr, t1, t2);
        T err = std::abs(est - true_sqrt) / std::abs(true_sqrt);
        std::printf("F16 sqrt: est=%.15e true=%.15e relerr=%.3e t2=%.3e\n",
                    est, true_sqrt, err, t2);
        EXPECT_EQ(driver.k_out, n);
        EXPECT_EQ(t2, (T)0);
        EXPECT_LT(err, 1e-13);
    }
    {
        auto fscalar = [](T x) { return std::log1p(x); };
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> state(349);
        T t1 = 0, t2 = 0;
        T est = driver.call(A_op, never_called, fscalar, /*k=*/n, /*s=*/0, q,
                            state, /*Omega2=*/nullptr, t1, t2);
        T err = std::abs(est - true_log1p) / std::abs(true_log1p);
        std::printf("F16 log1p: est=%.15e true=%.15e relerr=%.3e t2=%.3e\n",
                    est, true_log1p, err, t2);
        EXPECT_EQ(driver.k_out, n);
        EXPECT_EQ(t2, (T)0);
        EXPECT_LT(err, 1e-13);
    }
    delete[] A;
}

// ===== Eps-targeted adaptive tier: hard spectrum -> depth actually discovered
// On a geometric kappa = 1e6 spectrum the certified probe depth is data-
// driven, not a small fixed value: it must come out well above what a trivial
// (uncapped-but-effectively-shallow) implementation would produce, and the
// derived split must still be feasible (no throw) at a loose-enough eps.
TEST_F(TestFunNystromPP, AdaptiveHardSpectrumDepthDiscovered) {
    using T = double;
    const int64_t n = 400;
    const T eps = 1e-3, kappa = 1e6;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] = std::pow(kappa, (T)i / (T)(n - 1));
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::log1p(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> driver;
    RandBLAS::RNGState<RNG> state(353);
    T t1 = 0, t2 = 0;
    T est = 0;
    EXPECT_NO_THROW(est = driver.call(A_op, fscalar, eps, state, t1, t2));
    std::printf("adaptive hard spectrum: k=%ld s=%ld t=%ld probe_mv=%ld oracle_mv=%ld "
                "probe_cert=%d est=%.6e\n",
                (long)driver.adaptive_k, (long)driver.adaptive_s, (long)driver.adaptive_t,
                (long)driver.adaptive_probe_matvecs, (long)driver.adaptive_oracle_matvecs,
                (int)driver.adaptive_probe_certified, est);
    EXPECT_TRUE(std::isfinite(est));
    EXPECT_GT(driver.adaptive_t, 50);   // genuinely discovered depth, not a trivial floor
    EXPECT_LE(driver.adaptive_k, n / 2);
    EXPECT_GE(driver.adaptive_s, 4);
    delete[] A;
}

// A hard cap bounds the pilot too. Below the uncapped allocation boundary,
// the method may return an explicitly uncertified shallower estimate.
TEST_F(TestFunNystromPP, AdaptiveMatvecCapInfeasibleThrows) {
    using T = double;
    const int64_t n = 200;
    const T eps = 5e-2;
    const int64_t s_min = 4;

    T *G0 = randn<T>(n, n, /*seed=*/359);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    // Baseline: discover the natural (probe_mv, t) with no cap.
    RandLAPACK::FunNystromPP<T> baseline;
    RandBLAS::RNGState<RNG> st0(41);
    T t1 = 0, t2 = 0;
    baseline.call(A_op, fscalar, eps, st0, t1, t2);
    const int64_t probe_mv = baseline.adaptive_probe_matvecs;
    const int64_t t        = baseline.adaptive_t;
    ASSERT_GE(baseline.adaptive_s, s_min);   // precondition: the baseline itself is feasible
    std::printf("cap boundary baseline: probe_mv=%ld t=%ld k=%ld s=%ld\n",
                (long)probe_mv, (long)t, (long)baseline.adaptive_k, (long)baseline.adaptive_s);

    {   // exactly at the minimum feasible clamp boundary: must run.
        const int64_t feasible_cap = probe_mv + 1 + s_min * t;
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> st(41);
        T a = 0, b = 0;
        T est = 0;
        EXPECT_NO_THROW(est = driver.call(A_op, fscalar, eps, st, a, b, feasible_cap));
        EXPECT_TRUE(std::isfinite(est));
        EXPECT_GE(driver.adaptive_s, s_min);
        EXPECT_EQ(driver.adaptive_k, 1);        // the clamp reduces k to the affordable floor
        EXPECT_EQ(driver.adaptive_s, s_min);
        std::printf("cap=%ld (feasible boundary): k=%ld s=%ld est=%.4e\n",
                    (long)feasible_cap, (long)driver.adaptive_k, (long)driver.adaptive_s, est);
    }
    {   // Below the old boundary, shortening the pilot can make a run feasible.
        const int64_t infeasible_cap = probe_mv + s_min * t;
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> st(41);
        T a = 0, b = 0;
        try {
            driver.call(A_op, fscalar, eps, st, a, b, infeasible_cap);
            EXPECT_LE(driver.adaptive_probe_matvecs + driver.adaptive_k +
                      driver.adaptive_oracle_matvecs, infeasible_cap);
        } catch (const std::invalid_argument &e) {
            std::string msg = e.what();
            EXPECT_NE(msg.find("infeasible"), std::string::npos) << msg;
        }
    }
    {   // far below any feasible split: must also throw.
        RandLAPACK::FunNystromPP<T> driver;
        RandBLAS::RNGState<RNG> st(41);
        T a = 0, b = 0;
        EXPECT_THROW(driver.call(A_op, fscalar, eps, st, a, b, (int64_t)1),
                     std::invalid_argument);
    }
    delete[] G0; delete[] A;
}

// ===== Eps-targeted adaptive tier: block-Krylov limit throws, distinct msg ==
// Since R5 the probe's OWN depth cap already respects d*b <= n (probe_cap =
// min(user_cap, n, n/b)), so n/t >= b whenever b >= s_min = 4 (the default,
// adaptive_probe_block = 4): the block-Krylov guard can no longer starve the
// split below s_min through an uncapped probe alone. It still CAN when the
// probe block is narrower than s_min (adaptive_probe_block < 4): then
// n/t >= b < s_min is reachable. Forced here with adaptive_probe_block = 2
// (probe_cap = n/b = 20) and eps so tight (1e-12) the probe cannot certify
// to that tolerance in any depth this cap allows (whether it stops at the
// cap uncertified or the pivot chain deflates first is immaterial - R6:
// neither is itself an error): the reached t still divides n at most ~b
// ways, well under s_min, and the block-Krylov throw fires. The message
// must be distinct from the matvec_cap infeasibility
// message (AdaptiveMatvecCapInfeasibleThrows): it names the s*t <= n limit
// specifically and recommends the scalar `auto` tier, which has no joint
// block Krylov constraint.
TEST_F(TestFunNystromPP, AdaptiveKrylovLimitThrows) {
    using T = double;
    const int64_t n = 40;
    const T eps = 1e-12;

    T *G0 = randn<T>(n, n, /*seed=*/383);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> driver;
    driver.adaptive_probe_block = 2;
    RandBLAS::RNGState<RNG> state(367);
    T t1 = 0, t2 = 0;
    try {
        driver.call(A_op, fscalar, eps, state, t1, t2);
        FAIL() << "expected std::invalid_argument (block-Krylov limit); adaptive_t="
               << driver.adaptive_t << " adaptive_probe_certified="
               << driver.adaptive_probe_certified;
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        std::printf("Krylov-limit message: %s\n", msg.c_str());
        EXPECT_NE(msg.find("s*t <= n"), std::string::npos) << msg;
        EXPECT_NE(msg.find("auto"), std::string::npos)     << msg;
        EXPECT_EQ(msg.find("matvec_cap"), std::string::npos)
            << "must be distinct from the matvec_cap infeasibility message: " << msg;
    }
    delete[] G0; delete[] A;
}

// ===== Eps-targeted adaptive tier: uncertified probe proceeds, labeled ======
// R6: a depth probe that never certifies within its (now n/b-respecting,
// see AdaptiveKrylovLimitThrows) cap is a LABELED DEGRADATION, not a thrown
// error - the driver proceeds with t = the depth it actually reached
// (== probe_cap here), and adaptive_probe_certified / adaptive_phase2_
// certified come out false so the caller can see the reduced confidence.
// Forced with a small n (so the probe's own d*b <= n cap is tiny) and an
// eps far tighter than a handful of Lanczos steps can certify to, on a
// generic (no special structure) spectrum.
TEST_F(TestFunNystromPP, AdaptiveUncertifiedProbeProceedsLabeled) {
    using T = double;
    const int64_t n = 20;
    const T eps = 1e-10;

    T *A = new T[n * n]();   // zero-init: only the diagonal is written
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);   // generic, no gaps
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(x); };

    RandLAPACK::FunNystromPP<T> driver;
    RandBLAS::RNGState<RNG> state(389);
    T t1 = 0, t2 = 0;
    T est = 0;
    EXPECT_NO_THROW(est = driver.call(A_op, fscalar, eps, state, t1, t2));
    std::printf("uncertified probe: k=%ld s=%ld t=%ld probe_mv=%ld probe_cert=%d "
                "p2_cert=%d est=%.4e\n",
                (long)driver.adaptive_k, (long)driver.adaptive_s, (long)driver.adaptive_t,
                (long)driver.adaptive_probe_matvecs, (int)driver.adaptive_probe_certified,
                (int)driver.adaptive_phase2_certified, est);
    EXPECT_TRUE(std::isfinite(est));
    EXPECT_FALSE(driver.adaptive_probe_certified);
    EXPECT_FALSE(driver.adaptive_phase2_certified);
    // The probe ran to its own d*b <= n cap without closing the bracket.
    const int64_t b = driver.adaptive_probe_block;
    EXPECT_EQ(driver.adaptive_t, std::max((int64_t)1, n / b));
    delete[] A;
}

// ===== Eps-targeted adaptive tier: probe-reuse fold mechanics ===============
// After call(), driver.Omega2_buf holds the exact Phase-2 probe block that
// was used internally, and driver.adaptive_probe_buf / adaptive_M_buf hold
// the depth-probe's Rademacher block and its certified quadratic form - all
// public and untouched by anything after Phase 2 (adaptive_M_buf is the
// PROBE's b x b output; Phase 2 writes into a separate fAOmega buffer). That
// is enough to reconstruct, from outside the driver, both t2 BEFORE the
// reuse fold (an independent BlockLanczosQFA run on the exact same Omega2,
// same depth cap, deterministic Lanczos) and the fold's probe_sum term, and
// verify the driver's post-fold t2 equals
//   t2 = (t2_pre*s + probe_sum) / (s + b).
TEST_F(TestFunNystromPP, AdaptiveProbeReuseFolds) {
    using T = double;
    const int64_t n = 200;
    const T eps = 3e-2;

    T *G0 = randn<T>(n, n, /*seed=*/373);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> driver;
    RandBLAS::RNGState<RNG> state(379);
    T t1 = 0, t2 = 0;
    driver.call(A_op, fscalar, eps, state, t1, t2);

    ASSERT_TRUE(driver.adaptive_probe_certified) << "reuse mechanics require a certified probe";
    const int64_t k = driver.adaptive_k, s = driver.adaptive_s, t = driver.adaptive_t;
    const int64_t b = driver.adaptive_probe_block;
    ASSERT_LT(k, n);
    ASSERT_NE(driver.Omega2_buf, nullptr);
    ASSERT_NE(driver.adaptive_probe_buf, nullptr);

    // t2 BEFORE the fold: same expert-overload computation (same Omega2,
    // same U/lambda, same depth-t certified block QFA) via an independent
    // BlockLanczosQFA instance - deterministic Lanczos on identical inputs
    // must reproduce it.
    RandLAPACK::BlockLanczosQFA<T> bq_ref;
    bq_ref.adaptive      = true;
    bq_ref.stop_rule     = RandLAPACK::BlockQFAStop::Radau;
    bq_ref.return_mode   = RandLAPACK::BlockQFAReturn::Midpoint;
    bq_ref.adaptive_rtol = eps;
    T *M_ref = new T[s * s];
    bq_ref.call(A_op, driver.Omega2_buf, n, s, fscalar, t, M_ref);
    T tr_AOmega = 0;
    for (int64_t i = 0; i < s; ++i) tr_AOmega += M_ref[i + i * s];

    T *Y2 = new T[k * s];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, s, n, (T)1, driver.U, n, driver.Omega2_buf, n, (T)0, Y2, k);
    T tr_AhatOmega = 0;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < k; ++i) {
            T v = Y2[i + j * k];
            tr_AhatOmega += fscalar(driver.lambda[i]) * v * v;
        }
    T t2_pre = (tr_AOmega - tr_AhatOmega) / (T)s;

    // probe_sum: the fold's contribution from the b probe columns.
    T *Yp = new T[k * b];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, b, n, (T)1, driver.U, n, driver.adaptive_probe_buf, n, (T)0, Yp, k);
    T probe_sum = 0;
    for (int64_t j = 0; j < b; ++j) {
        T ghat = 0;
        for (int64_t i = 0; i < k; ++i) {
            T v = Yp[i + j * k];
            ghat += fscalar(driver.lambda[i]) * v * v;
        }
        probe_sum += driver.adaptive_M_buf[j + j * b] - ghat;
    }

    T t2_expected = (t2_pre * (T)s + probe_sum) / (T)(s + b);
    T rel = std::abs(t2_expected - t2) / std::max(std::abs(t2), (T)1e-12);
    std::printf("probe reuse mechanics: t2_pre=%.10e probe_sum=%.10e t2_expected=%.10e "
                "t2_driver=%.10e rel=%.3e (s=%ld b=%ld)\n",
                t2_pre, probe_sum, t2_expected, t2, rel, (long)s, (long)b);
    EXPECT_LT(rel, 1e-9);
    delete[] G0; delete[] A; delete[] M_ref; delete[] Y2; delete[] Yp;
}

// ===== C1: f_zero exercised through the adaptive tier's probe-reuse fold ===
// Adaptive-tier counterpart to AutoProbeReuseFoldsWithFZero: extends
// AdaptiveProbeReuseFolds's independent-reproduction pattern with a finite
// f_zero (f = log(x+2), f_zero = log(2)), adding the SAME zero-fill terms
// the driver applies at the expert call()'s Omega2 site
// (rl_fun_nystrom_pp.hh:730-733) and inside fold_probe_reuse
// (:539-546, src_stride = b+1 reading the block certificate's diagonal) to
// the manual t2_pre / probe_sum computations, then comparing bit-for-bit
// against the driver's actual post-fold t2.
TEST_F(TestFunNystromPP, AdaptiveProbeReuseFoldsWithFZero) {
    using T = double;
    const int64_t n = 200;
    const T eps = 3e-2;

    T *G0 = randn<T>(n, n, /*seed=*/373);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::log(x + (T)2); };
    const T f_zero = std::log((T)2);

    RandLAPACK::FunNystromPP<T> driver;
    RandBLAS::RNGState<RNG> state(379);
    T t1 = 0, t2 = 0;
    driver.call(A_op, fscalar, eps, state, t1, t2, /*matvec_cap=*/std::nullopt,
               std::optional<T>(f_zero));

    ASSERT_TRUE(driver.adaptive_probe_certified) << "reuse mechanics require a certified probe";
    const int64_t k = driver.adaptive_k, s = driver.adaptive_s, t = driver.adaptive_t;
    const int64_t b = driver.adaptive_probe_block;
    ASSERT_LT(k, n);
    ASSERT_NE(driver.Omega2_buf, nullptr);
    ASSERT_NE(driver.adaptive_probe_buf, nullptr);

    // t2 BEFORE the fold: independent BlockLanczosQFA on the same Omega2,
    // same depth cap t, same certificate settings the driver's Phase-2
    // fAfun uses.
    RandLAPACK::BlockLanczosQFA<T> bq_ref;
    bq_ref.adaptive      = true;
    bq_ref.stop_rule     = RandLAPACK::BlockQFAStop::Radau;
    bq_ref.return_mode   = RandLAPACK::BlockQFAReturn::Midpoint;
    bq_ref.adaptive_rtol = eps;
    T *M_ref = new T[s * s];
    bq_ref.call(A_op, driver.Omega2_buf, n, s, fscalar, t, M_ref);
    T tr_AOmega = 0;
    for (int64_t i = 0; i < s; ++i) tr_AOmega += M_ref[i + i * s];

    T *Y2 = new T[k * s];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, s, n, (T)1, driver.U, n, driver.Omega2_buf, n, (T)0, Y2, k);
    T tr_AhatOmega = 0;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < k; ++i) {
            T v = Y2[i + j * k];
            tr_AhatOmega += fscalar(driver.lambda[i]) * v * v;
        }
    // Expert call()'s OWN zero-fill term on the Phase-2 Omega2 block
    // (rl_fun_nystrom_pp.hh:730-733) - distinct from the fold's term below.
    {
        T omega_fro_sq = blas::dot(n * s, driver.Omega2_buf, 1, driver.Omega2_buf, 1);
        T y2_fro_sq    = blas::dot(k * s, Y2, 1, Y2, 1);
        tr_AhatOmega += f_zero * (omega_fro_sq - y2_fro_sq);
    }
    T t2_pre = (tr_AOmega - tr_AhatOmega) / (T)s;

    // probe_sum WITH the fold's own zero-fill term. src_stride = b+1 reads
    // the diagonal of the b x b block certificate matrix adaptive_M_buf.
    T *Yp = new T[k * b];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               k, b, n, (T)1, driver.U, n, driver.adaptive_probe_buf, n, (T)0, Yp, k);
    T probe_sum = 0;
    for (int64_t j = 0; j < b; ++j) {
        const T *yj = Yp + j * k;
        T ghat = 0;
        for (int64_t i = 0; i < k; ++i) ghat += fscalar(driver.lambda[i]) * yj[i] * yj[i];
        const T *gj  = driver.adaptive_probe_buf + j * n;
        T g_sq = blas::dot(n, gj, 1, gj, 1);
        T y_sq = blas::dot(k, yj, 1, yj, 1);
        ghat += f_zero * (g_sq - y_sq);
        probe_sum += driver.adaptive_M_buf[j + j * b] - ghat;
    }

    T t2_expected = (t2_pre * (T)s + probe_sum) / (T)(s + b);
    T rel = std::abs(t2_expected - t2) / std::max(std::abs(t2), (T)1e-12);
    std::printf("adaptive f_zero probe reuse: t2_pre=%.10e probe_sum=%.10e t2_expected=%.10e "
                "t2_driver=%.10e rel=%.3e (s=%ld b=%ld f0=%.4f)\n",
                t2_pre, probe_sum, t2_expected, t2, rel, (long)s, (long)b, f_zero);
    EXPECT_LT(rel, 1e-9);
    delete[] G0; delete[] A; delete[] M_ref; delete[] Y2; delete[] Yp;
}

// ===== I2: probe certifies, but Phase-2 does NOT (flags diverge) ===========
// Every existing test that checks both auto_probe_converged and
// auto_phase2_certified shows them moving TOGETHER (AutoBudgetClosesAndEstimates,
// AutoEasySpectrumCertifiesBothPhases: both true; AutoContractsHardSpectrum:
// both effectively false, probe never even certifies). The combination "probe
// (small block b=4) certifies, but the larger Phase-2 oracle block (different
// random columns, run at the probe-derived MEDIAN depth cap t) fails to
// certify" is plausible by construction: t is the median of only b=4 probe
// columns' certified depths, so roughly half the PROBE's own columns needed
// depth > t: an independent Phase-2 batch of s columns run capped at that
// same t has no structural reason to all converge by then.
//
// Found by a parameter sweep (not hand-derived): a geometric spectrum with
// kappa = 1e5 gives enough per-column depth variance that at n=800, eps=1e-3,
// matvec_budget=6000, seed=401 the probe certifies while at least one of the
// independently-drawn Phase-2 oracle columns does not, at the SAME depth cap
// the probe committed to.
TEST_F(TestFunNystromPP, AutoProbeCertifiesPhase2DoesNot) {
    using T = double;
    const int64_t n = 800;
    const T kappa = 1e5;
    const T eps = 1e-3;
    const int64_t m_budget = 6000;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i)
        A[i + i * n] = std::pow(kappa, (T)i / (T)(n - 1));
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::FunNystromPP<T> driver;
    RandBLAS::RNGState<RNG> state(401);
    T t1 = 0, t2 = 0;
    T est = 0;
    EXPECT_NO_THROW(est = driver.call(A_op, fscalar, m_budget, eps, state, t1, t2));

    std::printf("probe-certifies-phase2-does-not: k=%ld s=%ld t=%ld probe_mv=%ld "
                "oracle_mv=%ld probe_conv=%d p2cert=%d est=%.4e\n",
                (long)driver.auto_k, (long)driver.auto_s, (long)driver.auto_t,
                (long)driver.auto_probe_matvecs, (long)driver.auto_oracle_matvecs,
                (int)driver.auto_probe_converged, (int)driver.auto_phase2_certified, est);
    EXPECT_TRUE(std::isfinite(est));
    EXPECT_TRUE(driver.auto_probe_converged)   << "precondition: the depth probe must certify";
    EXPECT_FALSE(driver.auto_phase2_certified) << "the flags must DIVERGE: a certified probe "
                                                   "does not imply Phase 2 certifies at the same depth cap";
    delete[] A;
}

// ===== DiagSymLinOp: interchangeable with a dense diag(lambda) in every tier ==
// Same RNG state (and, for the expert tier, the same explicit Omega2 and the
// same exact oracle), so both operators see identical sketches and probes. Only
// the matvec arithmetic differs (blas::symm / right_spmm against a stored
// diagonal versus a row scaling), so the estimates must agree to rounding.
TEST_F(TestFunNystromPP, DiagSymLinOpMatchesDenseDiagAllTiers) {
    using T = double;
    const int64_t n = 300, k = 40, s = 30, q = 1;
    const T kappa = 1e3, eps = 1e-3;
    const int64_t budget = 600;

    T *A   = build_hard_psd<T>(n, kappa);
    T *lam = new T[n];
    for (int64_t i = 0; i < n; ++i) lam[i] = A[i + i * n];
    linops::ExplicitSymLinOp<T> A_dense(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    linops::DiagSymLinOp<T>     A_diag(n, lam);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto fAfun   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    T *Omega2 = randn<T>(n, s, /*seed=*/91);
    const T rtol = 1e-10;

    // Expert tier.
    {
        RandLAPACK::FunNystromPP<T> d1, d2;
        RandBLAS::RNGState<RNG> s1(501), s2(501);
        T a1 = 0, b1 = 0, a2 = 0, b2 = 0;
        T e1 = d1.call(A_dense, fAfun, fscalar, k, s, q, s1, Omega2, a1, b1);
        T e2 = d2.call(A_diag,  fAfun, fscalar, k, s, q, s2, Omega2, a2, b2);
        std::printf("diag-vs-dense expert: %.12e vs %.12e\n", e1, e2);
        EXPECT_LE(std::abs(e1 - e2), rtol * std::abs(e1));
        EXPECT_LE(std::abs(a1 - a2), rtol * std::abs(a1));
    }
    // Auto tier (internal probes drawn from the state).
    {
        RandLAPACK::FunNystromPP<T> d1, d2;
        RandBLAS::RNGState<RNG> s1(502), s2(502);
        T a1 = 0, b1 = 0, a2 = 0, b2 = 0;
        T e1 = d1.call(A_dense, fscalar, budget, eps, s1, a1, b1);
        T e2 = d2.call(A_diag,  fscalar, budget, eps, s2, a2, b2);
        std::printf("diag-vs-dense auto:   %.12e vs %.12e (k=%ld/%ld d=%ld/%ld)\n", e1, e2,
                    (long)d1.auto_k, (long)d2.auto_k, (long)d1.auto_sqfa.d_used, (long)d2.auto_sqfa.d_used);
        EXPECT_LE(std::abs(e1 - e2), rtol * std::abs(e1));
        EXPECT_EQ(d1.auto_k, d2.auto_k);
    }
    // Adaptive tier (eps-targeted block tier).
    {
        RandLAPACK::FunNystromPP<T> d1, d2;
        RandBLAS::RNGState<RNG> s1(503), s2(503);
        T a1 = 0, b1 = 0, a2 = 0, b2 = 0;
        T e1 = d1.call(A_dense, fscalar, eps, s1, a1, b1);
        T e2 = d2.call(A_diag,  fscalar, eps, s2, a2, b2);
        std::printf("diag-vs-dense adapt:  %.12e vs %.12e (k=%ld/%ld t=%ld/%ld)\n", e1, e2,
                    (long)d1.adaptive_k, (long)d2.adaptive_k, (long)d1.adaptive_t, (long)d2.adaptive_t);
        EXPECT_LE(std::abs(e1 - e2), rtol * std::abs(e1));
        EXPECT_EQ(d1.adaptive_k, d2.adaptive_k);
        EXPECT_EQ(d1.adaptive_t, d2.adaptive_t);
    }
    delete[] A;
    delete[] lam;
    delete[] Omega2;
}

// ===== Adaptive tier: adaptive_rademacher selects the probe distribution =====
// On a diagonal matrix a Rademacher quadratic form g' f(A) g equals tr f(A)
// exactly, which makes a diagonal test of the eps-targeted tier degenerate.
// The flag (default true, the paper's choice) must switch BOTH probe fills,
// the depth probe block and the delegated Phase-2 block, to the sphere family.
// Checked on the buffers themselves, not on the error, which the estimator's
// structure does not bound tightly enough for a deterministic threshold.
TEST_F(TestFunNystromPP, AdaptiveRademacherFlagControlsProbes) {
    using T = double;
    const int64_t n = 600;
    const T eps = 1e-3, kappa = 1e3;

    T *lam = new T[n];
    T tr = 0;
    for (int64_t i = 0; i < n; ++i) {
        lam[i] = std::pow(kappa, (T)i / (T)(n - 1));
        tr += std::sqrt(lam[i]);
    }
    linops::DiagSymLinOp<T> A_op(n, lam);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };
    auto all_pm1 = [](const T* buf, int64_t len) {
        for (int64_t e = 0; e < len; ++e)
            if (std::abs(buf[e]) != (T)1) return false;
        return true;
    };

    // Default: Rademacher everywhere, so the probe block certificate brackets
    // the exact block trace b * tr f(A).
    RandLAPACK::FunNystromPP<T> rad;
    ASSERT_TRUE(rad.adaptive_rademacher);
    RandBLAS::RNGState<RNG> st1(777);
    T t1 = 0, t2 = 0;
    T est_rad = rad.call(A_op, fscalar, eps, st1, t1, t2);
    const int64_t b = rad.adaptive_probe_block;
    EXPECT_TRUE(all_pm1(rad.adaptive_probe_buf, n * b));
    EXPECT_TRUE(all_pm1(rad.Omega2_buf, n * rad.adaptive_s));
    ASSERT_TRUE(rad.adaptive_probe_certified);
    T diag_sum = 0;
    for (int64_t i = 0; i < b; ++i) diag_sum += rad.adaptive_M_buf[i + i * b];
    EXPECT_LE(std::abs(diag_sum - (T)b * tr), (T)2 * eps * (T)b * tr);
    EXPECT_EQ(rad.probe_dist, RandLAPACK::ProbeDist::SphereGaussian);   // restored after the call

    // Flag off: sphere probes in both fills, same seed, different estimate.
    RandLAPACK::FunNystromPP<T> sph;
    sph.adaptive_rademacher = false;
    RandBLAS::RNGState<RNG> st2(777);
    T est_sph = sph.call(A_op, fscalar, eps, st2, t1, t2);
    EXPECT_FALSE(all_pm1(sph.adaptive_probe_buf, n * b));
    EXPECT_FALSE(all_pm1(sph.Omega2_buf, n * sph.adaptive_s));
    EXPECT_NE(est_rad, est_sph);
    EXPECT_EQ(sph.probe_dist, RandLAPACK::ProbeDist::SphereGaussian);
    std::printf("adaptive probes: rademacher est=%.8e sphere est=%.8e true=%.8e\n", est_rad, est_sph, tr);
    EXPECT_LT(std::abs(est_rad - tr) / tr, 5e-2);
    EXPECT_LT(std::abs(est_sph - tr) / tr, 5e-2);
    delete[] lam;
}

TEST_F(TestFunNystromPP, TargetAutoHardCapIncludesPilotOnEveryExit) {
    using T = double;
    const int64_t n = 100;
    std::vector<T> A(n*n, 0.0);
    for (int64_t i = 0; i < n; ++i) A[i+i*n] = std::pow(1e-6, T(i)/(n-1));
    auto f = [](T x) { return std::sqrt(std::max(x, T(0))); };
    for (bool scalar : {false, true}) {
        for (int64_t cap : {1, 16, 17, 32, 100, 400}) {
            CountingSymLinOp<T> op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
            RandLAPACK::FunNystromPP<T> driver;
            driver.adaptive_use_scalar = scalar;
            driver.adaptive_rademacher = false;
            driver.adaptive_reuse_pilot = false;
            driver.adaptive_gauss_return = true;
            RandBLAS::RNGState<RNG> state(991);
            T t1 = 0, t2 = 0;
            try {
                T est = driver.call(op, f, T(1e-3), state, t1, t2, cap);
                EXPECT_TRUE(std::isfinite(est));
                EXPECT_EQ(op.apply_count, driver.adaptive_probe_matvecs +
                          driver.adaptive_k + driver.adaptive_oracle_matvecs);
            } catch (const std::invalid_argument&) {
                EXPECT_LT(cap, 17);
            }
            EXPECT_LE(op.apply_count, cap) << "scalar=" << scalar << " cap=" << cap;
            if (cap < 17) EXPECT_EQ(op.apply_count, 0);
        }
    }
}

TEST_F(TestFunNystromPP, TargetAutoScalarDoesNotInheritBlockDimensionLimit) {
    const auto scalar = RandLAPACK::detail::compute_adaptive_split(
        (int64_t)80, 0.01, (int64_t)100, 1.0, 1.0, (int64_t)4, false);
    const auto block = RandLAPACK::detail::compute_adaptive_split(
        (int64_t)80, 0.01, (int64_t)100, 1.0, 1.0, (int64_t)4, true);
    EXPECT_GE(scalar.s, 4);
    EXPECT_EQ(block.s, 1);
    EXPECT_EQ(scalar.k, block.k);
}

TEST_F(TestFunNystromPP, TargetAutoTinyEpsSaturatesBeforeIntegerConversion) {
    for (bool block : {false, true}) {
        const auto split = RandLAPACK::detail::compute_adaptive_split(
            (int64_t)4, std::numeric_limits<double>::min(), (int64_t)100,
            1.0, 1.0, (int64_t)4, block);
        EXPECT_EQ(split.k, 50);
        EXPECT_EQ(split.s, block ? 25 : 50);
    }
}

// Independently count products when the cap forces a rank/probe tradeoff.
TEST_F(TestFunNystromPP, TargetAutoAllocationPoliciesRespectActualCap) {
    using T = double;
    const int64_t n = 100;
    T* A = new T[n*n]();
    for (int64_t i = 0; i < n; ++i) A[i+i*n] = std::pow(1e-4, T(i)/(n-1));
    auto f = [](T x) { return std::sqrt(std::max(x,T(0))); };
    for (bool scalar : {false,true}) {
        for (T fraction : {1.,.5,.25,.125}) {
            for (T quadrature : {1.,.25}) {
                CountingSymLinOp<T> op(n,blas::Uplo::Upper,A,n,Layout::ColMajor);
                RandLAPACK::FunNystromPP<T> d;
                d.adaptive_use_scalar=scalar;
                d.adaptive_rademacher=false;
                d.adaptive_reuse_pilot=false;
                d.adaptive_gauss_return=true;
                d.adaptive_cap_rank_fraction=fraction;
                d.adaptive_quadrature_fraction=quadrature;
                RandBLAS::RNGState<RNG> state(994);
                T a=0,b=0;
                const int64_t cap=180;
                EXPECT_TRUE(std::isfinite(d.call(op,f,T(1e-3),state,a,b,cap)));
                EXPECT_EQ(op.apply_count,d.adaptive_probe_matvecs+d.adaptive_k+d.adaptive_oracle_matvecs);
                EXPECT_LE(op.apply_count,cap);
                EXPECT_GE(d.adaptive_s,4);
                if (!scalar) EXPECT_LE(d.adaptive_s*d.adaptive_t,n);
            }
        }
    }
    delete[] A;
}

TEST_F(TestFunNystromPP, TargetAutoRejectsInvalidFractionsBeforeProducts) {
    using T = double;
    const int64_t n=20;
    T A[n*n]={};
    for (int64_t i=0;i<n;++i) A[i+i*n]=T(i+1)/n;
    for (T bad : {0.,-1.,1.01,std::numeric_limits<T>::infinity(),std::numeric_limits<T>::quiet_NaN()}) {
        for (bool rank : {false,true}) {
            CountingSymLinOp<T> op(n,blas::Uplo::Upper,A,n,Layout::ColMajor);
            RandLAPACK::FunNystromPP<T> d;
            if (rank) d.adaptive_cap_rank_fraction=bad;
            else d.adaptive_quadrature_fraction=bad;
            RandBLAS::RNGState<RNG> state(995);
            T a=0,b=0;auto f=[](T x){return std::sqrt(x);};
            EXPECT_THROW(d.call(op,f,T(1e-3),state,a,b,100),std::invalid_argument);
            EXPECT_EQ(op.apply_count,0);
        }
    }
}

} // namespace
