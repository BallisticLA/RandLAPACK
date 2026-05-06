#pragma once

#include "rl_blaspp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <concepts>

namespace RandLAPACK {


/// Hutchinson stochastic trace estimator.
///
/// Estimates tr(M) using the identity E[ω^T M ω] = tr(M) for zero-mean
/// unit-variance ω. Draws s independent Rademacher vectors (iid Unif{±1}),
/// applies M to all at once, and returns (1/s) * <Ω, Z>_F where Z = M*Ω.
///
/// M must satisfy linops::SymmetricLinearOperator<T>.
///
/// @tparam T    Floating-point scalar type.
/// @tparam RNG  Random number generator type.
template <typename T, typename RNG>
class Hutchinson {
public:
    // Internal buffers — grown with new/delete[], never shrunk.
    T* Omega = nullptr; int64_t Omega_sz = 0;
    T* Z     = nullptr; int64_t Z_sz     = 0;

    Hutchinson()                             = default;
    Hutchinson(const Hutchinson&)            = delete;
    Hutchinson& operator=(const Hutchinson&) = delete;

    ~Hutchinson() { delete[] Omega; delete[] Z; }

    // ------------------------------------------------------------------
    /// Low-level estimator: given precomputed Ω (n×s) and Z = M*Ω (n×s),
    /// returns <Ω, Z>_F / s = (1/s) * Σ_j ω_j^T (M ω_j).
    /// Uses blas::dot on flattened n*s arrays — no allocation.
    ///
    /// @param[in] Omega_buf  n×s sketch matrix (column-major).
    /// @param[in] Z          n×s result of M applied to Omega_buf (column-major).
    /// @param[in] n          Ambient dimension.
    /// @param[in] s          Number of samples.
    /// @returns   Frobenius inner product <Ω, Z>_F / s.
    T estimate(const T* Omega_buf, const T* Z, int64_t n, int64_t s) const {
        // Frobenius inner product of two n×s matrices, treated as flat n*s vectors
        return blas::dot(n * s, Omega_buf, 1, Z, 1) / static_cast<T>(s);
    }

    // ------------------------------------------------------------------
    /// High-level estimator: draws Ω internally, applies M, returns trace estimate.
    /// n is taken from M.dim.
    ///
    /// @param[in] M      Operator satisfying SymmetricLinearOperator<T>.
    /// @param[in] s      Number of Hutchinson samples.
    /// @param[in] state  RandBLAS RNG state; advanced on return.
    template <linops::SymmetricLinearOperator SLO>
    T call(SLO& M, int64_t s, RandBLAS::RNGState<RNG>& state) {
        int64_t n = M.dim;

        util::resize(Omega, Omega_sz, n * s);

        // Draw Ω with iid Rademacher entries (Unif{±1}).
        // RandBLAS has no ScalarDist::Rademacher, but ScalarDist::Uniform fills
        // with Unif[-1,1] via r123ext::uneg11; sign-transforming gives exact ±1.
        RandBLAS::DenseDist D(n, s, RandBLAS::ScalarDist::Uniform);
        state = RandBLAS::fill_dense(D, Omega, state);
        for (int64_t i = 0; i < n * s; ++i)
            Omega[i] = (Omega[i] >= 0) ? (T)1 : (T)-1;

        util::resize(Z, Z_sz, n * s);
        M(Layout::ColMajor, s, (T)1.0, Omega, n, (T)0.0, Z, n);

        return estimate(Omega, Z, n, s);
    }
};


} // end namespace RandLAPACK
