#include "rl_lanczos_fa.hh"
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

class TestLanczosFA : public RandLAPACK::testing::LanczosTestSupport {};

// ===== Scalar Lanczos-FA vs the exact oracle ================================
// First direct coverage of the scalar (per-column) LanczosFA: on a
// well-conditioned SPD matrix the depth-d Krylov approximation of f(A)B must
// match the exact V·diag(f(λ))·Vᵀ·B oracle to near machine precision, with and
// without reorthogonalization (Lanczos-FA tolerates orthogonality loss,
// Paige-Greenbaum).
TEST_F(TestLanczosFA, ScalarLanczosFAMatchesExact) {
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

} // namespace
