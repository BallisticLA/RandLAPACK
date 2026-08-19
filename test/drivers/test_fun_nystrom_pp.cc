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

    // Portable temp path: gtest's per-run TempDir works on Windows too, where
    // POSIX mkstemps/close do not exist. save_dense_bin creates the file.
    std::string tmpname = ::testing::TempDir() + "rl_v2_bin_roundtrip.bin";

    RandLAPACK::testing::save_dense_bin<T>(tmpname, m, n, orig);
    T *back = new T[m * n];
    std::fill(back, back + m * n, (T)-1.0);
    int64_t m_b = 0, n_b = 0;
    RandLAPACK::testing::load_dense_bin<T>(tmpname, m_b, n_b, back, m * n);
    EXPECT_EQ(m_b, m);
    EXPECT_EQ(n_b, n);
    for (int64_t i = 0; i < m * n; ++i) EXPECT_DOUBLE_EQ(back[i], orig[i]);
    std::remove(tmpname.c_str());
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


// ===== Scalar Lanczos-FA vs the exact oracle ================================
// First direct coverage of the scalar (per-column) LanczosFA: on a
// well-conditioned SPD matrix the depth-d Krylov approximation of f(A)B must
// match the exact V·diag(f(λ))·Vᵀ·B oracle to near machine precision, with and
// without reorthogonalization (Lanczos-FA tolerates orthogonality loss,
// Paige-Greenbaum).
TEST_F(TestFunNystromPPv2, ScalarLanczosFAMatchesExact) {
    using T = double;
    const int64_t n = 60, s = 8, d = 30;

    // A = GᵀG + n·I (symmetric PSD, well-conditioned ⟹ fast Krylov convergence).
    T *G0 = randn<T>(n, n, /*seed=*/47);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);

    T *Bmat = randn<T>(n, s, /*seed=*/53);
    auto fscalar = [](T x) { return std::sqrt(x); };
    auto exact   = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    T *ref = new T[n * s];
    exact(n, s, Bmat, ref);
    T ref_nrm = blas::nrm2(n * s, ref, 1);

    for (int64_t reorth = 1; reorth >= 0; --reorth) {
        RandLAPACK::LanczosFA<T> lfa; lfa.reorth = reorth;
        T *out = new T[n * s];
        lfa.call(A_op, Bmat, n, s, fscalar, d, out);
        for (int64_t i = 0; i < n * s; ++i) out[i] -= ref[i];
        T relF = blas::nrm2(n * s, out, 1) / ref_nrm;
        std::printf("scalar LanczosFA vs exact (reorth=%ld): rel Frobenius diff=%.3e\n",
                    reorth, relF);
        EXPECT_LT(relF, 1e-10);
        delete[] out;
    }
    delete[] G0; delete[] A; delete[] Bmat; delete[] ref;
}


// ===== Scalar Lanczos-QFA equals the per-column FA dots =====================
// The Gauss-quadrature identity bᵀ·LanczosFA(A, f, b) = Lanczos-QFA(A, f, b),
// per column: the fixed-depth scalar QFA vector out[j] must match
// ⟨B[:,j], (LanczosFA output)[:,j]⟩ computed by the reorth-0 scalar FA at the
// same depth (identical recurrence in exact arithmetic), and both must match
// the exact quadratic forms.
TEST_F(TestFunNystromPPv2, ScalarQFAmatchesScalarFAdots) {
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
// Gauss-Radau value (radau_val, node pinned at 0) must bracket the truth —
// this is the entire foundation of the certified stopping rule. Depths are
// probed by running adaptive mode with an unreachable tolerance so every
// column reports its (unclosed) bracket at the cap.
TEST_F(TestFunNystromPPv2, ScalarQFAradauBracketsDiagonal) {
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
// within eps of the true quadratic form (up to a small roundoff factor) —
// eps is a guarantee, not a target scale.
TEST_F(TestFunNystromPPv2, ScalarQFAcertifiedRelErr) {
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
// while random columns run deeper — exercising the retire/compaction
// bookkeeping (shrinking batched matvec) that a uniform-depth run never hits.
TEST_F(TestFunNystromPPv2, ScalarQFAadaptiveStopsEarly) {
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

// The knob-free overload call(A, f, m, eps, state, ...) must (a) never
// overspend the matvec budget — with the certified scalar-QFA oracle the
// closure is an upper bound, probe + q*k + oracle_mv <= m, since columns stop
// at their own certified depths — (b) allocate rank-heavy (k >> s, the
// paper's advocacy, automatic from the cost split), and (c) deliver a sane
// estimate. Well-conditioned SPD so the probe depth stays small and k = ~m/2
// stays below n.
TEST_F(TestFunNystromPPv2, AutoBudgetClosesAndEstimates) {
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
    EXPECT_LE(driver.auto_k, n / 2);          // rank cap (fragile k -> n Gram corner)
    EXPECT_LE(spend, m_budget);               // certified stopping never overspends
    EXPECT_GT(driver.auto_oracle_matvecs, 0);
    EXPECT_LE(driver.auto_oracle_matvecs,    // per-column depths never exceed the cap
              driver.auto_s * driver.auto_t);
    EXPECT_GT(driver.auto_k, driver.auto_s);  // rank-heavy split
    EXPECT_LT(err, 1e-2);
    delete[] G0; delete[] A;
}

// Infeasible inputs must throw with a descriptive message, not proceed.
TEST_F(TestFunNystromPPv2, AutoInfeasibleThrows) {
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
    {   // budget too small to fund probe + one probe + one unit of rank
        RandBLAS::RNGState<RNG> state(5);
        EXPECT_THROW(driver.call(A_op, fscalar, (int64_t)5, (T)1e-3, state, t1, t2),
                     std::invalid_argument);
    }
    {   // eps outside (0, 1)
        RandBLAS::RNGState<RNG> state(5);
        EXPECT_THROW(driver.call(A_op, fscalar, (int64_t)1000, (T)0, state, t1, t2),
                     std::invalid_argument);
    }
    delete[] G0; delete[] A;
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
TEST_F(TestFunNystromPPv2, PanelChunkPlanInvariants) {
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
// adjacent slots ping-pong one line -- worst exactly when ncols is small, which
// is the retirement tail this kernel exists to serve.
TEST_F(TestFunNystromPPv2, PanelPartialStrideIsCacheLinePadded) {
    using QFA = RandLAPACK::LanczosQFA<double>;
    constexpr int64_t LINE = 64 / (int64_t)sizeof(double);
    for (int64_t nc : {1, 2, 4, 7, 8, 9, 32, 96, 100}) {
        const int64_t st = QFA::partial_stride(nc);
        ASSERT_GE(st, nc);
        ASSERT_EQ(st % LINE, 0) << "stride " << st << " for ncols=" << nc
                                << " is not a multiple of a cache line";
    }
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
TEST_F(TestFunNystromPPv2, ReuseAcrossCallsIsBitIdentical) {
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

// t_fafun_ms must be CLEARED on the k == n path, not left at the previous
// call's value. Consumers compute assembly = t_phase2_ms - t_fafun_ms, so a
// stale value makes that negative. Regression test for the fix in
// rl_fun_nystrom_pp.hh's Phase-2 skip branch.
TEST_F(TestFunNystromPPv2, FafunTimerResetAtKEqualsN) {
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
TEST_F(TestFunNystromPPv2, SmallRankBelowVecNnzDoesNotThrow) {
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
