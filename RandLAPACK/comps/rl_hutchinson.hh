#pragma once

#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <concepts>

namespace RandLAPACK {


/// Hutchinson stochastic trace estimator.
///
/// Estimates tr(M) using the identity E[ω^T M ω] = tr(M) for any zero-mean
/// unit-variance ω. Draws s independent Gaussian vectors, applies M to all at once, then
/// returns the averaged quadratic form (1/s) * <Ω, Z>_F where Z = M*Ω.
///
/// Standalone and reusable: the operator M is provided as a callable rather
/// than a fixed type, so this component can be used for any trace estimation
/// problem in RandLAPACK without modification.
///
/// @tparam T    Floating-point scalar type.
/// @tparam RNG  Random number generator type for RandBLAS.
template <typename T, typename RNG>
class Hutchinson {
public:
    // Internal sketch buffer — grown with new/delete[], never shrunk.
    T* Omega   = nullptr; int64_t Omega_sz = 0;

    ~Hutchinson() { delete[] Omega; }

    // ------------------------------------------------------------------
    /// Low-level estimator: given precomputed Ω (n×s) and Z = M*Ω (n×s),
    /// returns <Ω, Z>_F / s = (1/s) * Σ_j ω_j^T (M ω_j).
    /// Uses blas::dot on flattened n*s arrays — no allocation.
    ///
    /// @param[in] Omega_buf  n×s sketch matrix (column-major).
    /// @param[in] Z          n×s result of M applied to Omega_buf (column-major).
    /// @param[in] n          Ambient dimension.
    /// @param[in] s          Number of samples.
    T estimate(const T* Omega_buf, const T* Z, int64_t n, int64_t s) const {
        // Frobenius inner product of two n×s matrices, treated as flat n*s vectors
        return blas::dot(n * s, Omega_buf, 1, Z, 1) / static_cast<T>(s);
    }

    // ------------------------------------------------------------------
    /// High-level estimator: draws Ω internally, calls apply_M to fill Z,
    /// then returns the trace estimate.
    ///
    /// apply_M must have signature: (const T* Omega, T* Z, int64_t n, int64_t s)
    /// It receives the n×s Rademacher matrix and must overwrite Z with M*Ω.
    ///
    /// @param[in] apply_M  Callable that applies M to an n×s matrix.
    /// @param[in] n        Ambient dimension.
    /// @param[in] s        Number of Hutchinson samples.
    /// @param[in] state    RandBLAS RNG state; advanced on return.
    template <typename ApplyM>
    T call(ApplyM apply_M, int64_t n, int64_t s,
           RandBLAS::RNGState<RNG>& state) {
        // Grow Omega buffer if needed (reuse across repeated calls)
        if (n * s > Omega_sz) {
            delete[] Omega;
            Omega = new T[n * s];
            Omega_sz = n * s;
        }

        // Draw Ω with iid Rademacher entries (Unif{±1}).
        // RandBLAS has no ScalarDist::Rademacher, but ScalarDist::Uniform fills
        // with Unif[-1,1] via r123ext::uneg11; sign-transforming gives exact ±1.
        RandBLAS::DenseDist D(n, s, RandBLAS::ScalarDist::Uniform);
        state = RandBLAS::fill_dense(D, Omega, state);
        for (int64_t i = 0; i < n * s; ++i)
            Omega[i] = (Omega[i] >= 0) ? (T)1 : (T)-1;

        // Allocate Z and apply the operator
        T* Z = new T[n * s];
        apply_M(Omega, Z, n, s);

        T result = estimate(Omega, Z, n, s);
        delete[] Z;
        return result;
    }
};


} // end namespace RandLAPACK
