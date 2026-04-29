#pragma once

#include "rl_blaspp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <cassert>
#include <concepts>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace RandLAPACK {


/// funNyström++ trace estimator: estimates tr(f(A)) for symmetric PSD A.
///
/// Algorithm (3 phases):
///   Phase 1 — Nyström approximation via REVD2 (k matvecs):
///     (V, λ) = REVD2(A, k)  →  Â = V diag(λ) V^T
///     tr(f(Â)) = Σ f(λ_i) + (n-k) * f(0)   [free from the eigenvalues]
///
///   Phase 2 — Hutchinson correction on the residual (d*s matvecs):
///     Ω = Rademacher(n×s)
///     Z1 = f(A)*Ω   via LanczosFA  [d batch matvecs]
///     Z2 = f(Â)*Ω = V * (f(λ) ⊙ (V^T Ω))  [two GEMMs, free]
///     correction = <Ω, Z1 - Z2>_F / s
///
///   return tr(f(Â)) + correction
///
/// Complexity: O(√κ/ε) matvecs total, versus O(√κ/ε²) for naive Hutchinson.
/// The variance reduction comes from subtracting the control variate f(Â)Ω:
/// because Â ≈ A in the dominant eigenspace, f(A)-f(Â) has small Frobenius
/// norm and the Hutchinson estimator converges in far fewer samples.
///
/// Requirements on A:
///   - Symmetric positive definite (λ_min > 0). The spectral interval [a,b]
///     with 0 < a < b is assumed throughout (see paper).
///   - For f = log, A must be strictly PD (log(0) = -∞). Pass f_zero = 0 only
///     for functions where f(0) is well-defined (e.g., sqrt, x^α with α > 0).
///
/// Parameter selection (f = sqrt, κ = λ_max/λ_min):
///   d = O(√κ log(n/ε))   Lanczos steps — caller's responsibility
///   s = O(1/(√κ ε))      Hutchinson samples — caller's responsibility
///   k = O(√κ/ε)          Nyström rank — caller's responsibility
/// Use error_est_power_iters=0 in REVD2 and tol=0 to get fixed-rank behavior
/// (k never changes; the adaptive loop exits after one iteration).
///
/// @tparam REVD2_t       Type of the REVD2 Nyström component.
/// @tparam LanczosFA_t   Type of the LanczosFA component.
/// @tparam Hutchinson_t  Type of the Hutchinson trace estimator component.
template <typename REVD2_t, typename LanczosFA_t, typename Hutchinson_t>
class FunNystromPP {
public:
    using T   = typename REVD2_t::T;
    using RNG = typename REVD2_t::RNG;

    REVD2_t&      revd2;
    LanczosFA_t&  lanczos_fa;
    Hutchinson_t& hutchinson;

    // Persistent output/working buffers — grown with new/delete[], never shrunk.
    // V, eigvals: REVD2 outputs reused across calls with same (n, k).
    // F_vec:      f applied elementwise to eigvals.
    // tmp, Z1, Z2: workspace owned here and lent to linops::ResidualOp each call.
    T* V_buf      = nullptr; int64_t V_buf_sz      = 0;
    T* eigvals_buf = nullptr; int64_t eigvals_buf_sz = 0;
    T* F_vec      = nullptr; int64_t F_vec_sz      = 0;
    T* tmp        = nullptr; int64_t tmp_sz        = 0;
    T* Z1         = nullptr; int64_t Z1_sz         = 0;
    T* Z2         = nullptr; int64_t Z2_sz         = 0;

    ~FunNystromPP() {
        delete[] V_buf; delete[] eigvals_buf;
        delete[] F_vec; delete[] tmp; delete[] Z1; delete[] Z2;
    }

    FunNystromPP(REVD2_t& r, LanczosFA_t& l, Hutchinson_t& h)
        : revd2(r), lanczos_fa(l), hutchinson(h) {}

    // ------------------------------------------------------------------
    /// Estimate tr(f(A)).
    ///
    /// @param[in]  A       SymmetricLinearOperator — matvec oracle for A.
    /// @param[in]  f       Scalar function T→T (e.g., sqrt, log, pow(x,α)).
    /// @param[in]  f_zero  f(0): value at zero. Use 0.0 for sqrt/power.
    ///                     Must be finite; assert fires if infinite.
    ///                     For log, A must be strictly PD — pass any finite
    ///                     placeholder and ensure λ_min > 0.
    /// @param[in]  k       Nyström rank. Caller chooses based on κ, ε.
    /// @param[in]  s       Hutchinson samples. Caller chooses based on κ, ε.
    /// @param[in]  d       Lanczos steps per sample. Caller chooses based on κ.
    /// @param[in]  state   RandBLAS RNG state; advanced on return.
    /// @returns   Estimate of tr(f(A)).
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    T call(
        SLO& A,
        F f,
        T f_zero,
        int64_t k,
        int64_t s,
        int64_t d,
        RandBLAS::RNGState<RNG>& state
    ) {
        assert(std::isfinite(f_zero) && "f_zero must be finite; for log, ensure A is strictly PD");

        int64_t n = A.dim;

        // ------------------------------------------------------------------
        // Phase 1: Nyström approximation
        // REVD2 with error_est_power_iters=0 and tol=0 runs a single pass
        // with fixed rank k. V_buf (n×k) and eigvals_buf (k) are the outputs.
        // ------------------------------------------------------------------
        // REVD2::call takes rank by reference and may increase it adaptively.
        // We pass k_in (a copy) so the caller's k stays unchanged; we still
        // need the original k below for the (n-k)*f(0) tail correction.
        int64_t k_in = k;
        revd2.call(A, k_in, (T)0.0, V_buf, V_buf_sz, eigvals_buf, eigvals_buf_sz, state);

        // f(λ): apply scalar f to k eigenvalues
        util::resize(F_vec, F_vec_sz, k);
        std::transform(eigvals_buf, eigvals_buf + k, F_vec, f);

        // tr(f(Â)) = Σ f(λ_i) + (n-k)*f(0)
        T tr_Ahat = (T)0.0;
        for (int64_t i = 0; i < k; ++i)
            tr_Ahat += F_vec[i];
        tr_Ahat += static_cast<T>(n - k) * f_zero;

        // ------------------------------------------------------------------
        // Phase 2: Hutchinson correction on residual f(A) - f(Â)
        // linops::ResidualOp wraps the correction as a SymmetricLinearOperator
        // so Hutchinson::call can draw Ω and compute <Ω, (f(A)-f(Â))Ω>_F / s.
        // ------------------------------------------------------------------
        util::resize(tmp, tmp_sz, k * s);
        util::resize(Z1,  Z1_sz,  n * s);
        util::resize(Z2,  Z2_sz,  n * s);

        linops::ResidualOp<T, SLO, LanczosFA_t, F> res_op(
            n, A, lanczos_fa, f, d, k, V_buf, F_vec, tmp, Z1, Z2
        );
        T correction = hutchinson.call(res_op, s, state);

        return tr_Ahat + correction;
    }
};


} // end namespace RandLAPACK
