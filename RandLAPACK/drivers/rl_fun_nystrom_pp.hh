#pragma once

#include "rl_blaspp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <concepts>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace RandLAPACK {

/// Per-call timing breakdown for FunNystromPP::call.
/// Pass a pointer to call() to collect Phase 1 and Phase 2 wall-clock times.
struct FunNystromPP_timing {
    long phase1_us = 0;  // NystromEVD Nyström approximation (Phase 1)
    long phase2_us = 0;  // Hutchinson trace correction (Phase 2)
};

/// funNyström++ trace estimator: estimates tr(f(A)) for symmetric PSD A.
///
/// Algorithm (2 phases):
///   Phase 1 — Nyström approximation via NystromEVD (O(k) matvecs):
///     (V, λ) = NystromEVD(A, k)  →  Â = V diag(λ) V^T
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
///   - Symmetric positive semidefinite (λ_min ≥ 0) for most functions
///     (e.g., sqrt, x^α with α > 0 where f(0) is finite).
///   - For f = log, A must be strictly positive definite (λ_min > 0),
///     since log(0) = -∞. Pass any finite f_zero placeholder and ensure
///     λ_min > 0 at the call site.
///
/// Parameter selection (f = sqrt, κ = λ_max/λ_min):
///   d = O(√κ log(n/ε))   Lanczos steps — caller's responsibility
///   s = O(1/(√κ ε))      Hutchinson samples — caller's responsibility
///   k = O(√κ/ε)          Nyström rank — caller's responsibility
/// Use error_est_power_iters=0 in NystromEVD and tol=0 to get fixed-rank behavior
/// (k never changes; the adaptive loop exits after one iteration).
///
/// @tparam NystromEVD_t       Type of the NystromEVD Nyström component.
/// @tparam LanczosFA_t   Type of the LanczosFA component.
/// @tparam Hutchinson_t  Type of the Hutchinson trace estimator component.
template <typename NystromEVD_t, typename LanczosFA_t, typename Hutchinson_t>
class FunNystromPP {
public:
    using T   = typename NystromEVD_t::T;
    using RNG = typename NystromEVD_t::RNG;

    NystromEVD_t&      nystrom_evd;       // Phase 1: Nyström approximation
    LanczosFA_t&  lanczos_fa;  // Phase 2: f(A)*Ω oracle
    Hutchinson_t& hutchinson;  // Phase 2: trace correction estimator

    // Persistent output/working buffers — grown with new/delete[], never shrunk.
    // V, eigvals: NystromEVD outputs reused across calls with same (n, k).
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

    // ------------------------------------------------------------------
    /// Construct a FunNystromPP estimator from pre-built components.
    ///
    /// @param r  NystromEVD instance for Phase 1 (Nyström approximation).
    /// @param l  LanczosFA instance for computing f(A)*Ω in Phase 2.
    /// @param h  Hutchinson instance for Phase 2 trace correction.
    ///
    /// All three components are stored by reference and must outlive this object.
    FunNystromPP(NystromEVD_t& r, LanczosFA_t& l, Hutchinson_t& h)
        : nystrom_evd(r), lanczos_fa(l), hutchinson(h) {}

    // ------------------------------------------------------------------
    /// Estimate tr(f(A)).
    ///
    /// @param[in]  A       SymmetricLinearOperator — matvec oracle for A.
    /// @param[in]  f       Scalar function T→T (e.g., sqrt, log, pow(x,α)).
    /// @param[in]  f_zero  f(0): value at zero. Use 0.0 for sqrt/power.
    ///                     Must be finite; throws std::invalid_argument if not.
    ///                     For log, A must be strictly PD — pass any finite
    ///                     placeholder and ensure λ_min > 0.
    /// @param[in]  k       Nyström rank. Caller chooses based on κ, ε.
    /// @param[in]  s       Hutchinson samples. Caller chooses based on κ, ε.
    /// @param[in]  d       Lanczos steps per sample. Caller chooses based on κ.
    /// @param[in]  state   RandBLAS RNG state; advanced on return.
    /// @param[out] timing  Optional. If non-null, receives Phase 1 and Phase 2
    ///                     wall-clock times in microseconds.
    /// @returns   Estimate of tr(f(A)).
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    T call(
        SLO& A,
        F f,
        T f_zero,
        int64_t k,
        int64_t s,
        int64_t d,
        RandBLAS::RNGState<RNG>& state,
        FunNystromPP_timing* timing = nullptr
    ) {
        if (!std::isfinite(f_zero))
            throw std::invalid_argument("f_zero must be finite; for log, ensure A is strictly PD");

        int64_t n = A.dim;

        // ------------------------------------------------------------------
        // Phase 1: Nyström approximation
        // NystromEVD with error_est_power_iters=0 and tol=0 runs a single pass
        // with fixed rank k. V_buf (n×k_in) and eigvals_buf (k_in) are the outputs.
        // ------------------------------------------------------------------
        // NystromEVD::call takes rank by reference and may increase it adaptively.
        // We pass k_in (a copy) so the caller's k stays unchanged.
        // All post-call uses reference k_in (not k) so they stay consistent
        // with the actual rank used, even in adaptive mode.
        int64_t k_in = k;
        auto t_phase1_start = std::chrono::steady_clock::now();
        nystrom_evd.call(A, k_in, (T)0.0, V_buf, V_buf_sz, eigvals_buf, eigvals_buf_sz, state);
        auto t_phase1_end = std::chrono::steady_clock::now();

        // f(λ): apply scalar f to k_in eigenvalues
        util::resize(F_vec, F_vec_sz, k_in);
        std::transform(eigvals_buf, eigvals_buf + k_in, F_vec, f);

        // tr(f(Â)) = Σ f(λ_i) + (n-k_in)*f(0)
        T tr_Ahat = (T)0.0;
        for (int64_t i = 0; i < k_in; ++i)
            tr_Ahat += F_vec[i];
        tr_Ahat += static_cast<T>(n - k_in) * f_zero;

        // ------------------------------------------------------------------
        // Phase 2: Hutchinson correction on residual f(A) - f(Â)
        // linops::ResidualOp wraps the correction as a SymmetricLinearOperator
        // so Hutchinson::call can draw Ω and compute <Ω, (f(A)-f(Â))Ω>_F / s.
        // ------------------------------------------------------------------
        util::resize(tmp, tmp_sz, k_in * s);
        util::resize(Z1,  Z1_sz,  n * s);
        util::resize(Z2,  Z2_sz,  n * s);

        linops::ResidualOp<T, SLO, LanczosFA_t, F> res_op(
            n, A, lanczos_fa, f, d, k_in, V_buf, F_vec, tmp, Z1, Z2
        );
        auto t_phase2_start = std::chrono::steady_clock::now();
        T correction = hutchinson.call(res_op, s, state);
        auto t_phase2_end = std::chrono::steady_clock::now();

        if (timing) {
            using us = std::chrono::microseconds;
            timing->phase1_us = std::chrono::duration_cast<us>(t_phase1_end - t_phase1_start).count();
            timing->phase2_us = std::chrono::duration_cast<us>(t_phase2_end - t_phase2_start).count();
        }

        return tr_Ahat + correction;
    }
};


} // end namespace RandLAPACK
