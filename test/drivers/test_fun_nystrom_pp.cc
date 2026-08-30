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

    // Adaptive, legacy Window rule (this test pins the heuristic; the Radau
    // certificate has its own tests below). Two runs: the shipped default
    // delay (2), then the historical delay = 5, both must stop early and land
    // on the converged value.
    for (int64_t delay : {RandLAPACK::BlockLanczosQFA<T>::default_adaptive_delay, (int64_t)5}) {
        RandLAPACK::BlockLanczosQFA<T> qfa;
        qfa.adaptive = true;
        qfa.stop_rule = RandLAPACK::BlockQFAStop::Window;
        qfa.adaptive_rtol = 1e-3; qfa.adaptive_delay = delay;
        T *M_adapt = new T[s * s];
        qfa.call(A_op, Bmat, n, s, fscalar, d_max, M_adapt);
        T tr_adapt = 0; for (int64_t i = 0; i < s; ++i) tr_adapt += M_adapt[i + i * s];

        T reltr = std::abs(tr_adapt - tr_fixed) / std::abs(tr_fixed);
        std::printf("adaptive QFA (window, delay=%ld): d_used=%ld / d_max=%ld  tr_adapt=%.8e tr_fixed=%.8e reltr=%.3e\n",
                    (long)delay, (long)qfa.d_used, (long)d_max, tr_adapt, tr_fixed, reltr);
        EXPECT_GT(qfa.d_used, 0);
        EXPECT_LT(qfa.d_used, d_max);   // stopped early
        EXPECT_LT(reltr, 1e-2);         // matches the converged value
        EXPECT_FALSE(qfa.certified);    // the window rule carries no certificate
        delete[] M_adapt;
    }
    delete[] G0; delete[] A; delete[] Bmat; delete[] M_fixed;
}

// The knob-free overload call(A, f, m, eps, state, ...) must (a) never
// overspend the matvec budget — with the certified scalar-QFA oracle the
// closure is an upper bound, probe + q*k + oracle_mv <= m, since columns stop
// at their own certified depths (probe-sample REUSE folds certified probe
// columns into the Phase-2 average but costs zero extra matvecs, so the
// closure invariant is unchanged) — (b) bound the probe's spend by the
// auto_probe_frac cap (default 1/8 of the budget), (c) allocate rank-heavy
// (k >> s; on this easy spectrum the n/2 rank cap binds and the surplus goes
// to probes), and (d) deliver a sane estimate with both certification flags
// reported. Well-conditioned SPD so the probe certifies at a small MEDIAN
// depth (the redesigned depth policy) and k = n/2 stays feasible.
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

// Regression for the fixed depth cap of 200 (removed 2026-08): on a hard
// spectrum with a tight eps the certified probe must be free to go deeper
// than 200 when n and the budget allow it. With the old cap this probe
// pinned at exactly 200 and the oracle bias floored above eps, so no budget
// could recover the target accuracy (the kappa >= 1e6 cells of the 2026-07
// campaign). Under the redesigned allocation the probe's cap is
// min(n, max(2, floor(auto_probe_frac*B/b))) = min(400, 625) = 400 here, so
// the probe runs to the full n = 400 (t is then the MEDIAN certified depth,
// or the reached depth capped by m_rem/(2*s_min) when uncertified — both
// exceed 200 on this spectrum). Geometric spectrum kappa = 1e6,
// f = log(1+x): the certified depth wants several hundred at this eps.
TEST_F(TestFunNystromPPv2, AutoProbeDepthNotFixedCapped) {
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


// ===== Block oracles vs the exact oracle ====================================
// BlockQFAmatchesBlockFA compares two consumers of the SAME block tridiagonal,
// so an error confined to T cancels there identically. This test is the
// discriminating one: at full block Krylov depth (d*s == n) both the FA vector
// output and the QFA quadratic form must reproduce the EXACT oracle to
// roundoff, and any defect in the recurrence, the T assembly, or the
// eigendecomposition surfaces directly.
TEST_F(TestFunNystromPPv2, BlockOraclesMatchExactAtFullDepth) {
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
    T *Mref = new T[s * s];
    blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
               s, s, n, (T)1, Bmat, n, ref, n, (T)0, Mref, s);   // Bᵀf(A)B

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

    // QFA at full depth.
    RandLAPACK::BlockLanczosQFA<T> qfa;
    T *M = new T[s * s];
    qfa.call(A_op, Bmat, n, s, fscalar, d, M);
    T maxq = 0, sclq = 0;
    for (int64_t e = 0; e < s * s; ++e) {
        maxq = std::max(maxq, std::abs(M[e] - Mref[e]));
        sclq = std::max(sclq, std::abs(Mref[e]));
    }
    std::printf("block-vs-exact full depth: FA reldiff=%.3e  QFA reldiff=%.3e\n",
                maxfa / sclfa, maxq / sclq);
    EXPECT_LT(maxq / sclq, 1e-10);
    delete[] G0; delete[] A; delete[] Bmat; delete[] ref; delete[] Mref;
    delete[] Gout; delete[] M;
}


// ===== Block Gauss-Radau: s = 1 must reproduce the scalar certificate =======
// At s = 1 the block recurrence is the scalar recurrence (up to a sign the
// quadratic form is invariant to). The Gauss value and the Radau CORNER must
// match the scalar oracle to roundoff at every depth. The Radau VALUE itself
// is compared tightly only for f = log1p: with f = sqrt the pinned-at-0 node
// makes tr_L reproducible only to ~sqrt(roundoff) across algebraically
// equivalent corner computations (f' is infinite at the node, so a
// roundoff-level shift delta in the near-zero Ritz value moves tr_L by
// ~ w * sqrt(delta)); log1p has f'(0) = 1 and no such amplification.
TEST_F(TestFunNystromPPv2, BlockQFAradauS1MatchesScalar) {
    using T = double;
    const int64_t n = 80;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *b = randn<T>(n, 1, /*seed=*/107);
    auto f_sqrt = [](T x) { return std::sqrt(x); };
    auto f_log  = [](T x) { return std::log1p(x); };

    T qf_s, M_b;
    for (int64_t depth : {4, 8, 16}) {
        for (int fcase = 0; fcase < 2; ++fcase) {
            RandLAPACK::LanczosQFA<T> sq;
            sq.adaptive = true;
            sq.adaptive_rtol = std::numeric_limits<T>::min();   // run to the cap
            RandLAPACK::BlockLanczosQFA<T> bq;
            bq.reorth = 0;   // scalar LanczosQFA has no reorthogonalization
            bq.adaptive = true;
            bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
            bq.adaptive_rtol = std::numeric_limits<T>::min();   // never fires
            if (fcase == 0) {
                sq.call(A_op, b, n, 1, f_sqrt, depth, &qf_s);
                bq.call(A_op, b, n, 1, f_sqrt, depth, &M_b);
            } else {
                sq.call(A_op, b, n, 1, f_log, depth, &qf_s);
                bq.call(A_op, b, n, 1, f_log, depth, &M_b);
            }

            T relU = std::abs(bq.tr_U - sq.gauss_val[0]) / std::abs(sq.gauss_val[0]);
            T relL = std::abs(bq.tr_L - sq.radau_val[0]) / std::abs(sq.radau_val[0]);
            // Corner comparison is f-independent: block corner = A_d - D_d
            // (1x1 tiles at s = 1) vs the scalar's exact saved corner.
            const int64_t m = depth * 1;
            const T A_d = bq.fa.T_blk[(m - 1) + (m - 1) * m];
            T relC = std::abs((A_d - bq.D_buf[0]) - sq.radau_corner[0])
                     / std::abs(sq.radau_corner[0]);
            std::printf("block s=1 vs scalar d=%2ld %s: relU=%.3e relL=%.3e relC=%.3e\n",
                        depth, fcase == 0 ? "sqrt " : "log1p", relU, relL, relC);
            EXPECT_LT(relU, 1e-12) << "depth " << depth << " fcase " << fcase;
            EXPECT_LT(relC, 1e-11) << "depth " << depth << " fcase " << fcase;
            if (fcase == 0) EXPECT_LT(relL, 3e-8)  << "sqrt depth "  << depth;
            else            EXPECT_LT(relL, 1e-12) << "log1p depth " << depth;
        }
    }
    delete[] A; delete[] b;
}


// ===== Block Gauss-Radau: bracket property on a diagonal matrix =============
// For operator-monotone f the Gauss and Radau-at-0 block quadratures err on
// opposite sides, so [min(trU,trL), max(trU,trL)] must trap the exact
// tr(Bᵀ f(A) B) at every depth, for both benchmark f's.
TEST_F(TestFunNystromPPv2, BlockQFAradauBracketsDiagonal) {
    using T = double;
    const int64_t n = 90, s = 4;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/109);

    auto run_case = [&](auto fscalar, const char *fname) {
        T truth = 0;
        for (int64_t j = 0; j < s; ++j)
            for (int64_t i = 0; i < n; ++i) {
                T bij = Bmat[i + j * n];
                truth += fscalar((T)(i + 1)) * bij * bij;
            }
        T *M = new T[s * s];
        for (int64_t depth : {4, 8, 16}) {
            RandLAPACK::BlockLanczosQFA<T> bq;
            bq.adaptive = true;
            bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
            bq.adaptive_rtol = std::numeric_limits<T>::min();   // never fires
            bq.call(A_op, Bmat, n, s, fscalar, depth, M);
            T hi = std::max(bq.tr_U, bq.tr_L), lo = std::min(bq.tr_U, bq.tr_L);
            T slack = 1e-12 * std::abs(truth);
            EXPECT_LE(lo - slack, truth) << fname << " depth " << depth;
            EXPECT_LE(truth, hi + slack) << fname << " depth " << depth;
            std::printf("block Radau bracket %s d=%2ld: U=%.8e L=%.8e true=%.8e gap/true=%.3e\n",
                        fname, depth, bq.tr_U, bq.tr_L, truth,
                        std::abs(bq.tr_U - bq.tr_L) / std::abs(truth));
        }
        delete[] M;
    };
    run_case([](T x) { return std::sqrt(x); },  "sqrt");
    run_case([](T x) { return std::log1p(x); }, "log1p");
    delete[] A; delete[] Bmat;
}


// ===== Block Gauss-Radau: certified adaptive stop delivers eps ==============
// d*s stays below n so the run avoids the no-deflation degenerate regime.
TEST_F(TestFunNystromPPv2, BlockQFAcertifiedRelErr) {
    using T = double;
    const int64_t n = 320, s = 4, d_cap = 79;   // d_cap*s = 316 < n
    const T eps = 1e-6;

    T *G0 = randn<T>(n, n, /*seed=*/113);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/127);
    auto fscalar = [](T x) { return std::sqrt(x); };

    auto exact = RandLAPACK::testing::make_exact_fa_oracle<T>(n, A, fscalar);
    T *ref = new T[n * s];
    exact(n, s, Bmat, ref);
    T truth = 0;
    for (int64_t j = 0; j < s; ++j)
        truth += blas::dot(n, Bmat + j * n, 1, ref + j * n, 1);

    RandLAPACK::BlockLanczosQFA<T> bq;
    bq.adaptive = true;
    bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
    bq.adaptive_rtol = eps;
    T *M = new T[s * s];
    bq.call(A_op, Bmat, n, s, fscalar, d_cap, M);
    T trM = 0; for (int64_t i = 0; i < s; ++i) trM += M[i + i * s];

    T rel = std::abs(trM - truth) / std::abs(truth);
    std::printf("block Radau certified: d_used=%ld/%ld rel=%.3e (eps=%.0e) matvecs=%ld certified=%d\n",
                (long)bq.d_used, (long)d_cap, rel, eps, (long)bq.matvecs, (int)bq.certified);
    EXPECT_TRUE(bq.certified);
    EXPECT_LT(bq.d_used, d_cap);              // stopped early on this easy spectrum
    EXPECT_LT(rel, 2 * eps);                  // certified bound + roundoff slack
    EXPECT_EQ(bq.matvecs, s * bq.d_used);     // s column-applications per block step
    delete[] G0; delete[] A; delete[] Bmat; delete[] ref; delete[] M;
}


// ===== Block Gauss-Radau: rank-deficient initial block / breakdown ==========
// A zero column and a duplicated column make R0 singular; an invariant-subspace
// block (two exact eigenvector columns of a diagonal A) collapses the Krylov
// space at the first step. Neither may crash, produce NaN, or report a
// certificate that the pivot chain cannot support.
TEST_F(TestFunNystromPPv2, BlockQFArankDeficientInitialBlock) {
    using T = double;
    const int64_t n = 60;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    auto fscalar = [](T x) { return std::sqrt(x); };

    {   // zero column + duplicated column, s = 4
        const int64_t s = 4;
        T *Bmat = randn<T>(n, s, /*seed=*/131);
        std::fill(Bmat + 1 * n, Bmat + 2 * n, (T)0);            // col 1 = 0
        std::copy(Bmat, Bmat + n, Bmat + 3 * n);                // col 3 = col 0
        RandLAPACK::BlockLanczosQFA<T> bq;
        bq.adaptive = true;
        bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
        bq.adaptive_rtol = 1e-6;
        T *M = new T[s * s];
        EXPECT_NO_THROW(bq.call(A_op, Bmat, n, s, fscalar, 20, M));
        for (int64_t e = 0; e < s * s; ++e)
            EXPECT_TRUE(std::isfinite(M[e])) << "entry " << e;
        std::printf("rank-deficient R0: d_used=%ld certified=%d trU=%.6e trL=%.6e\n",
                    (long)bq.d_used, (int)bq.certified, bq.tr_U, bq.tr_L);
        delete[] Bmat; delete[] M;
    }
    {   // invariant-subspace block: two eigenvector columns, s = 2
        const int64_t s = 2;
        T *Bmat = new T[n * s]();
        Bmat[3 + 0 * n] = (T)1;    // e_3
        Bmat[7 + 1 * n] = (T)1;    // e_7
        RandLAPACK::BlockLanczosQFA<T> bq;
        bq.adaptive = true;
        bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
        bq.adaptive_rtol = 1e-6;
        T *M = new T[s * s];
        EXPECT_NO_THROW(bq.call(A_op, Bmat, n, s, fscalar, 10, M));
        // The quadratic form itself stays exact regardless of certification:
        // tr = f(4) + f(8) for these unit eigenvector columns.
        T tr = M[0] + M[3];
        EXPECT_TRUE(std::isfinite(tr));
        EXPECT_NEAR(tr, std::sqrt((T)4) + std::sqrt((T)8), 1e-10);
        std::printf("invariant block: d_used=%ld certified=%d tr=%.12e\n",
                    (long)bq.d_used, (int)bq.certified, tr);
        delete[] Bmat; delete[] M;
    }
    delete[] A;
}


// ===== Block Gauss-Radau: midpoint return ===================================
// The midpoint lies inside the bracket by construction, and its error is at
// most half the bracket width whenever the bracket traps the truth.
TEST_F(TestFunNystromPPv2, BlockQFAmidpointWithinBracket) {
    using T = double;
    const int64_t n = 90, s = 4, depth = 8;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/137);
    auto fscalar = [](T x) { return std::sqrt(x); };

    T truth = 0;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < n; ++i) {
            T bij = Bmat[i + j * n];
            truth += std::sqrt((T)(i + 1)) * bij * bij;
        }

    RandLAPACK::BlockLanczosQFA<T> bq;
    bq.adaptive = true;
    bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
    bq.return_mode = RandLAPACK::BlockQFAReturn::Midpoint;
    bq.adaptive_rtol = 1e-2;   // certifies at a truncated depth on this spectrum
    T *M = new T[s * s];
    bq.call(A_op, Bmat, n, s, fscalar, depth, M);
    T trMid = 0; for (int64_t i = 0; i < s; ++i) trMid += M[i + i * s];

    ASSERT_TRUE(bq.certified);
    T hi = std::max(bq.tr_U, bq.tr_L), lo = std::min(bq.tr_U, bq.tr_L);
    EXPECT_LE(lo - 1e-12 * std::abs(truth), trMid);
    EXPECT_LE(trMid, hi + 1e-12 * std::abs(truth));
    EXPECT_NEAR(trMid, (T)0.5 * (bq.tr_U + bq.tr_L), 1e-12 * std::abs(truth));
    EXPECT_LE(std::abs(trMid - truth), (T)0.5 * (hi - lo) + 1e-12 * std::abs(truth));
    std::printf("midpoint: tr=%.8e in [%.8e, %.8e], |err|=%.3e <= half-width=%.3e\n",
                trMid, lo, hi, std::abs(trMid - truth), (T)0.5 * (hi - lo));
    delete[] A; delete[] Bmat; delete[] M;
}


// ===== Block QFA: reorthogonalization time propagated =======================
TEST_F(TestFunNystromPPv2, BlockQFAreorthTimingPropagated) {
    using T = double;
    const int64_t n = 400, s = 4, depth = 30;

    T *G0 = randn<T>(n, n, /*seed=*/139);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/149);
    auto fscalar = [](T x) { return std::sqrt(x); };
    T *M = new T[s * s];

    RandLAPACK::BlockLanczosQFA<T> bq_on;
    bq_on.reorth = 1; bq_on.timing = true;
    bq_on.call(A_op, Bmat, n, s, fscalar, depth, M);
    ASSERT_EQ((int64_t)bq_on.times.size(), 6);
    EXPECT_GT(bq_on.times[5], 0L);   // block MGS is real, measured work

    RandLAPACK::BlockLanczosQFA<T> bq_off;
    bq_off.reorth = 0; bq_off.timing = true;
    bq_off.call(A_op, Bmat, n, s, fscalar, depth, M);
    ASSERT_EQ((int64_t)bq_off.times.size(), 6);
    EXPECT_EQ(bq_off.times[5], 0L);
    std::printf("block QFA reorth time: on=%ld us, off=%ld us\n",
                bq_on.times[5], bq_off.times[5]);
    delete[] G0; delete[] A; delete[] Bmat; delete[] M;
}


// ===== Block Gauss-Radau: pivot recurrence vs a dense solve =================
// The maintained corner A_t - D_t must equal B_{t-1} (Eᵀ T_{t-1}⁻¹ E) B_{t-1}ᵀ
// computed by an explicit dense solve on the leading (t-1)s block — the
// identity the whole O(s³) recurrence rests on.
TEST_F(TestFunNystromPPv2, BlockQFApivotRecurrenceMatchesDenseSolve) {
    using T = double;
    const int64_t n = 60, s = 3, d = 10;

    T *G0 = randn<T>(n, n, /*seed=*/151);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/157);
    auto fscalar = [](T x) { return std::sqrt(x); };

    // Run to the cap with an unreachable tolerance: D_buf then holds D_d and
    // fa.T_blk survives intact (the at-cap bracket check uses preserving copies).
    RandLAPACK::BlockLanczosQFA<T> bq;
    bq.adaptive = true;
    bq.stop_rule = RandLAPACK::BlockQFAStop::Radau;
    bq.adaptive_rtol = std::numeric_limits<T>::min();
    T *M = new T[s * s];
    bq.call(A_op, Bmat, n, s, fscalar, d, M);
    ASSERT_EQ(bq.d_used, d);

    const int64_t m  = d * s;         // T_blk leading dimension
    const int64_t m1 = (d - 1) * s;   // T_{d-1} dimension

    // Dense T_{d-1} (symmetrize from the stored lower triangle).
    T *Tm = new T[m1 * m1];
    for (int64_t j = 0; j < m1; ++j)
        for (int64_t i = 0; i < m1; ++i) {
            T v = (i >= j) ? bq.fa.T_blk[i + j * m] : bq.fa.T_blk[j + i * m];
            Tm[i + j * m1] = v;
        }
    // X = T_{d-1}⁻¹ E, E = last s columns of the identity.
    T *X = new T[m1 * s]();
    for (int64_t j = 0; j < s; ++j) X[(m1 - s + j) + j * m1] = (T)1;
    lapack::posv(blas::Uplo::Lower, m1, s, Tm, m1, X, m1);   // T_{d-1} is PD here
    // corner_dense = Btile · X_bottom · Btileᵀ, Btile = math B_{d-1} (upper
    // triangular s×s at rows (d-1)s.., cols (d-2)s.. of T_blk).
    const T *Btile = bq.fa.T_blk + ((d - 2) * s) * m + ((d - 1) * s);
    T *Xb = new T[s * s];   // bottom s rows of X
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = 0; i < s; ++i)
            Xb[i + j * s] = X[(m1 - s + i) + j * m1];
    T *tmp = new T[s * s], *corner_dense = new T[s * s];
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               s, s, s, (T)1, const_cast<T*>(Btile), m, Xb, s, (T)0, tmp, s);
    // tmp = Btile·Xb reads Btile as a general matrix; its strict lower is zero.
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               s, s, s, (T)1, tmp, s, const_cast<T*>(Btile), m, (T)0, corner_dense, s);

    // Maintained corner: A_d − D_d (lower triangles are the valid data).
    const T *Ad = bq.fa.T_blk + ((d - 1) * s) * m + ((d - 1) * s);
    T maxdiff = 0, scale = 0;
    for (int64_t j = 0; j < s; ++j)
        for (int64_t i = j; i < s; ++i) {
            T maintained = Ad[i + j * m] - bq.D_buf[i + j * s];
            T dense      = corner_dense[i + j * s];
            maxdiff = std::max(maxdiff, std::abs(maintained - dense));
            scale   = std::max(scale, std::abs(dense));
        }
    std::printf("pivot recurrence vs dense solve: reldiff=%.3e\n", maxdiff / scale);
    EXPECT_LT(maxdiff / scale, 1e-12);
    delete[] G0; delete[] A; delete[] Bmat; delete[] M;
    delete[] Tm; delete[] X; delete[] Xb; delete[] tmp; delete[] corner_dense;
}


// ===== BlockLanczosFA: early-stopped recurrence evaluates at the run depth ==
// When a stop_after callback ends the recurrence at k < d, the interrupted
// step has already written its off-diagonal block, so evaluating at the full
// d would read a corrupted tail. apply_f must evaluate at steps_run and match
// a fresh full run at that depth.
TEST_F(TestFunNystromPPv2, BlockFAearlyStopAppliesAtRunDepth) {
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


// ===== Float instantiation ==================================================
// First float coverage in this file: the expert driver path and the certified
// scalar QFA must instantiate and behave at T = float. Well-conditioned
// diagonal SPD (kappa = 1e3) so the f32 Nystrom shift nu ~ n*eps_f*||A||_2 is
// harmless against lambda_min = 1 (NystromEVD emits its one-time f32 stderr
// NOTE here; expected). Bounds are float-loose by design.
TEST_F(TestFunNystromPPv2, FloatExpertPathAndCertifiedQFA) {
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
    delete[] A; delete[] Omega2; delete[] Bq; delete[] qf;
}


// ===== f_zero zero-fill path ================================================
// f = log(x + 2) has f(0) = log(2) != 0, so the Persson-anchor convention
// (implicit f(0) = 0) would misestimate tr(f(A)) through the (n - k)-dim
// complement. Passing f_zero opts in to the zero-fill correction: t1 gains
// (n - k) f(0) and t2 subtracts the projector-complement term, and the
// estimate must land on the dense truth. The documented invalid_argument on
// a non-finite f_zero is pinned alongside.
TEST_F(TestFunNystromPPv2, FZeroPathMatchesDenseTruth) {
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
// the deep fixed-depth QFA's truncation — near machine precision on this
// well-conditioned spectrum.
TEST_F(TestFunNystromPPv2, ExpertUseQfaMatchesExactOracle) {
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
TEST_F(TestFunNystromPPv2, ExpertLanczosFAOracleMatchesExact) {
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
//       funds >= 4 probes — never the old s == 2 lock);
//   (b) more budget buys more probes: auto_s nondecreasing across B1 < B2 < B3;
//   (c) the probe never spends past its fraction cap:
//       probe_mv <= ceil(0.125*B) + b slack.
TEST_F(TestFunNystromPPv2, AutoContractsHardSpectrum) {
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
// flags must report success — auto_probe_converged (the depth probe) AND
// auto_phase2_certified (every Phase-2 oracle column certified at the
// median-depth cap; captured after the oracle runs, since Phase 2 reuses the
// probe's LanczosQFA instance).
TEST_F(TestFunNystromPPv2, AutoEasySpectrumCertifiesBothPhases) {
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


// ===== Scalar QFA: all-zero input column ====================================
// A zero column has no Krylov space: it must retire at t = 0 with value 0,
// certified, before the first matvec — so the batched matvec never pays for
// it and `matvecs` equals the sum of the OTHER columns' depths exactly.
TEST_F(TestFunNystromPPv2, ScalarQFAZeroColumnRetiresAtZero) {
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
// value is ||b||^2 * f(alpha_1) exactly — checkable to roundoff on a diagonal
// matrix. Both modes must run without crashing; adaptive cannot certify (the
// bracket needs t >= 2) and must still return the depth-1 Gauss value.
TEST_F(TestFunNystromPPv2, ScalarQFADepthOne) {
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
// so certified depths sit on that grid — possibly different from the ladder's
// — while the certified value stays within tolerance. check_every = 0 is
// rejected up front.
TEST_F(TestFunNystromPPv2, ScalarQFACheckEveryStride) {
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
// produces — pinning that no oversized-buffer state leaks into the values.
TEST_F(TestFunNystromPPv2, ScalarQFAShrinkingReuseBitIdentical) {
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
// certificate — uncertified, finite values, no crash. f = x^2 is finite
// everywhere (NB quad_e1 clamps Ritz values to >= 0 by the A >= 0 contract,
// so the VALUES are not accurate here; this test pins only the guard).
TEST_F(TestFunNystromPPv2, ScalarQFAIndefiniteUncertified) {
    using T = double;
    const int64_t n = 60, s = 5, d_cap = 30;

    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1) - (T)30;  // -29..30
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/277);
    auto fscalar = [](T x) { return x * x; };

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
    std::printf("indefinite QFA: all %ld columns uncertified via the pivot guard "
                "(d_used=%ld), values finite\n", (long)s, (long)qfa.d_used);
    delete[] A; delete[] Bmat; delete[] qf;
}
