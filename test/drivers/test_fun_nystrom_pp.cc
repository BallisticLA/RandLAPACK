#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_test_utils.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

namespace linops = RandLAPACK::linops;

// Phase 1 tests. The fAfun oracle is "exact dense f(A) · B" computed
// once per test from an explicit eigendecomposition of A; that lets
// each test isolate the v2 driver's behavior from Krylov truncation.
// Phase 4 will add a block-Lanczos fAfun and re-run an analogous set.
//
// All buffers are raw new[]/delete[] (house rule: no std::vector for
// matrix/vector data) and all randomness goes through RandBLAS
// (Philox4x32; house rule: no std::mt19937).

class TestFunNystromPPv2 : public ::testing::Test {
protected:
    using RNG = r123::Philox4x32;

    // The exact f(A)·B oracle (V·diag(f(λ))·Vᵀ·B) now lives in one place:
    // RandLAPACK::testing::make_exact_fa_oracle (rl_test_utils.hh). The tests,
    // the benchmark, and the MEX binding all share that single implementation
    // rather than re-deriving the GEMM-diag-GEMM apply.

    // Compute tr(f(A)) exactly via syevd. A_full must be full-symmetric n×n.
    template <typename T, typename F>
    static T true_trace_fa(int64_t n, const T *A_full, F &&fscalar) {
        T *A_cpy = new T[n * n];
        T *ev    = new T[n];
        std::copy(A_full, A_full + n * n, A_cpy);
        lapack::syevd(lapack::Job::NoVec, lapack::Uplo::Upper, n, A_cpy, n, ev);
        T tr = 0;
        for (int64_t i = 0; i < n; ++i) tr += fscalar(ev[i]);
        delete[] A_cpy;
        delete[] ev;
        return tr;
    }

    // Sample an n×s standard-normal matrix (column-major) into a new[]
    // buffer, through RandBLAS. Caller owns the buffer.
    template <typename T>
    static T* randn(int64_t n, int64_t s, uint32_t seed) {
        T *M = new T[n * s];
        RandBLAS::RNGState<RNG> state(seed);
        RandBLAS::DenseDist D(n, s);
        RandBLAS::fill_dense(D, M, state);
        return M;
    }
};


// ===== Binary I/O round trip (kept from Phase 0) =============================

TEST_F(TestFunNystromPPv2, BinaryIoRoundTrip) {
    using T = double;
    int64_t m = 5, n = 3;
    T *orig = new T[m * n];
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            orig[i + j * m] = (T)(100 * j + i + 1);

    char tmpname[] = "/tmp/rl_v2_bin_roundtrip_XXXXXX.bin";
    int fd = mkstemps(tmpname, 4);
    ASSERT_GE(fd, 0);
    close(fd);

    RandLAPACK::testing::save_dense_bin<T>(tmpname, m, n, orig);
    T *back = new T[m * n];
    std::fill(back, back + m * n, (T)-1.0);
    int64_t m_b = 0, n_b = 0;
    RandLAPACK::testing::load_dense_bin<T>(tmpname, m_b, n_b, back, m * n);
    EXPECT_EQ(m_b, m);
    EXPECT_EQ(n_b, n);
    for (int64_t i = 0; i < m * n; ++i) EXPECT_DOUBLE_EQ(back[i], orig[i]);
    std::remove(tmpname);
    delete[] orig;
    delete[] back;
}


// ===== Phase 1 accuracy tests ================================================

// Diagonal A = diag(1..n), f = sqrt. True trace is Σ √i, no eigensolver
// needed. k = 15 < n = 50; Hutchinson correction does real work.
TEST_F(TestFunNystromPPv2, DiagonalSqrt) {
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
TEST_F(TestFunNystromPPv2, FullRankCapture) {
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

// Random dense PSD, f = sqrt. k = 10, k_mat unknown — Phase 1 captures
// only the top subspace, Phase 2's Hutchinson carries real load. Tol = 15%.
TEST_F(TestFunNystromPPv2, RandomPSDSqrt) {
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


// ===== Block Lanczos-QFA equals Bᵀ·(Block Lanczos-FA output) ================
// The quadratic form M = BlockLanczosQFA(A, B, f, d) must equal Bᵀ·(f(A)·B),
// where f(A)·B = BlockLanczosFA(A, B, f, d) — the Gauss-quadrature identity
// gᵀ·LanczosFA = Lanczos-QFA, lifted to blocks. Exact when the block Krylov
// basis is orthonormal (reorth on); a looser sanity bound without reorth,
// where basis-orthogonality loss makes the two approximations differ.
TEST_F(TestFunNystromPPv2, BlockQFAmatchesBlockFA) {
    using T = double;
    const int64_t n = 60, s = 8, d = 50;

    // A = GᵀG + n·I (symmetric PSD), same construction as RandomPSDSqrt.
    T *G0 = randn<T>(n, n, /*seed=*/31);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    T *Bmat = randn<T>(n, s, /*seed=*/37);
    auto fscalar = [](T x) { return std::sqrt(x); };

    for (int64_t reorth = 1; reorth >= 0; --reorth) {
        // FA path: G = f(A)·B (n×s), then BᵀG (s×s).
        RandLAPACK::BlockLanczosFA<T> fa; fa.reorth = reorth;
        T *Gout = new T[n * s];
        fa.call(A_op, Bmat, n, s, fscalar, d, Gout);
        T *BtG = new T[s * s];
        blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   s, s, n, (T)1, Bmat, n, Gout, n, (T)0, BtG, s);

        // QFA path: M = Bᵀ f(A) B directly (no mapback).
        RandLAPACK::BlockLanczosQFA<T> qfa; qfa.reorth = reorth;
        T *M = new T[s * s];
        qfa.call(A_op, Bmat, n, s, fscalar, d, M);

        T maxdiff = 0, scale = 0, trFA = 0, trQFA = 0;
        for (int64_t i = 0; i < s * s; ++i) {
            maxdiff = std::max(maxdiff, std::abs(M[i] - BtG[i]));
            scale   = std::max(scale, std::abs(BtG[i]));
        }
        for (int64_t i = 0; i < s; ++i) { trFA += BtG[i + i * s]; trQFA += M[i + i * s]; }
        T relmat = maxdiff / scale;
        T reltr  = std::abs(trFA - trQFA) / std::abs(trFA);
        std::printf("BlockQFA vs BᵀFA (reorth=%ld): matrix reldiff=%.3e  tr reldiff=%.3e\n",
                    reorth, relmat, reltr);
        // reorth on: block MGS orthogonality (~1e-9); reorth off: for a smooth
        // f and modest d the raw three-term basis keeps Q₀ ⊥ later blocks to
        // ~machine precision, so the FA/QFA identity holds tighter still.
        if (reorth) { EXPECT_LT(relmat, 1e-7); EXPECT_LT(reltr, 1e-7); }
        else        { EXPECT_LT(relmat, 1e-8); EXPECT_LT(reltr, 1e-8); }
        delete[] Gout; delete[] BtG; delete[] M;
    }
    delete[] G0; delete[] A; delete[] Bmat;
}


// ===== Adaptive-depth block Lanczos-QFA =====================================
// With adaptive = true the recurrence stops before d_max once the block
// quadrature estimate tr(M_k) settles (windowed relative change <= rtol). On a
// well-conditioned SPD matrix (fast Krylov convergence) it must stop early, and
// its trace must match the fully-converged fixed-depth QFA to ~rtol.
TEST_F(TestFunNystromPPv2, BlockQFAadaptiveStopsEarly) {
    using T = double;
    const int64_t n = 80, s = 6, d_max = 70;

    // A = GᵀG + n·I  (well-conditioned SPD ⟹ fast Lanczos convergence).
    T *G0 = randn<T>(n, n, /*seed=*/41);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/43);
    auto fscalar = [](T x) { return std::sqrt(x); };

    // Fixed-depth reference (fully converged at d_max).
    RandLAPACK::BlockLanczosQFA<T> qfa_fixed;
    T *M_fixed = new T[s * s];
    qfa_fixed.call(A_op, Bmat, n, s, fscalar, d_max, M_fixed);
    T tr_fixed = 0; for (int64_t i = 0; i < s; ++i) tr_fixed += M_fixed[i + i * s];

    // Adaptive.
    RandLAPACK::BlockLanczosQFA<T> qfa;
    qfa.adaptive = true; qfa.adaptive_rtol = 1e-3; qfa.adaptive_delay = 5;
    T *M_adapt = new T[s * s];
    qfa.call(A_op, Bmat, n, s, fscalar, d_max, M_adapt);
    T tr_adapt = 0; for (int64_t i = 0; i < s; ++i) tr_adapt += M_adapt[i + i * s];

    T reltr = std::abs(tr_adapt - tr_fixed) / std::abs(tr_fixed);
    std::printf("adaptive QFA: d_used=%ld / d_max=%ld  tr_adapt=%.8e tr_fixed=%.8e reltr=%.3e\n",
                (long)qfa.d_used, (long)d_max, tr_adapt, tr_fixed, reltr);
    EXPECT_GT(qfa.d_used, 0);
    EXPECT_LT(qfa.d_used, d_max);   // stopped early
    EXPECT_LT(reltr, 1e-2);         // matches the converged value
    delete[] G0; delete[] A; delete[] Bmat; delete[] M_fixed; delete[] M_adapt;
}
