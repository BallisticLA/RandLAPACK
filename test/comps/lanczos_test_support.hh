#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_test_utils.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace RandLAPACK::testing {

// Shared matrix construction and exact-trace helpers for the Lanczos component
// and FunNystromPP driver tests. Matrix buffers are owned by each caller.
class LanczosTestSupport : public ::testing::Test {
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

    // Two spectra reused by the OPT-1 banded-vs-dense eigensolve A/B
    // equivalence tests (BlockQFABandedVsDense*): "easy" is the
    // well-conditioned GᵀG + n·I construction used throughout these tests
    // (e.g. BlockQFAmatchesBlockFA); "hard" is the diagonal geometric
    // spectrum (kappa = 1e6) AutoContractsHardSpectrum /
    // AdaptiveHardSpectrumDepthDiscovered use. Caller owns the returned
    // n×n buffer (new[]).
    template <typename T>
    static T* build_easy_psd(int64_t n, uint32_t seed) {
        T *G0 = randn<T>(n, n, seed);
        T *A  = new T[n * n];
        blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
                   n, n, (T)1, G0, n, (T)0, A, n);
        for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
        delete[] G0;
        return A;
    }
    template <typename T>
    static T* build_hard_psd(int64_t n, T kappa) {
        T *A = new T[n * n]();
        for (int64_t i = 0; i < n; ++i)
            A[i + i * n] = std::pow(kappa, (T)i / (T)(n - 1));
        return A;
    }
};

} // namespace RandLAPACK::testing
