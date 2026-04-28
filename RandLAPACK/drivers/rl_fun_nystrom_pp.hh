#pragma once

#include "rl_blaspp.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <cassert>
#include <concepts>
#include <stdexcept>
#include <limits>

namespace RandLAPACK {


/// Residual operator (f(A) - f(Â)) for the funNyström++ Hutchinson correction.
///
/// Satisfies SymmetricLinearOperator so it can be passed directly to Hutchinson.
/// Computes C = α*(f(A)*B - f(Â)*B) + β*C via:
///   Z1 = f(A)*B  using LanczosFA (d batch matvecs)
///   Z2 = f(Â)*B = V*(diag(F_vec)*(V^T B))  using two GEMMs
/// tmp, Z1, Z2 are workspace owned by FunNystromPP and borrowed here by pointer.
///
/// @tparam T      Scalar type.
/// @tparam SLO_t  Type of the operator A.
/// @tparam LFA_t  Type of the LanczosFA component.
/// @tparam F_t    Scalar function type T→T.
template <typename T, typename SLO_t, typename LFA_t, typename F_t>
struct ResidualOp {
    using scalar_t = T;
    const int64_t dim;

    SLO_t&  A;
    LFA_t&  lanczos_fa;
    F_t     f;
    int64_t d;
    int64_t k;
    T*      V;
    T*      F_vec;
    T*      tmp;   // k×s workspace
    T*      Z1;    // n×s workspace: f(A)*Ω
    T*      Z2;    // n×s workspace: f(Â)*Ω

    ResidualOp(int64_t n_, SLO_t& A_, LFA_t& lfa_, F_t f_,
               int64_t d_, int64_t k_, T* V_, T* Fv_, T* tmp_, T* Z1_, T* Z2_)
        : dim(n_), A(A_), lanczos_fa(lfa_), f(f_),
          d(d_), k(k_), V(V_), F_vec(Fv_), tmp(tmp_), Z1(Z1_), Z2(Z2_) {}

    void operator()(Layout layout, int64_t n_vecs, T alpha,
                    T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        // Z1 = f(A)*B via Lanczos-FA (d batch matvecs)
        lanczos_fa.call(A, B, dim, n_vecs, f, d, Z1);

        // Z2 = f(Â)*B = V * (diag(F_vec) * (V^T B))
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                   k, n_vecs, dim, (T)1.0, V, dim, B, ldb, (T)0.0, tmp, k);
        for (int64_t j = 0; j < n_vecs; ++j)
            for (int64_t i = 0; i < k; ++i)
                tmp[j * k + i] *= F_vec[i];
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   dim, n_vecs, k, (T)1.0, V, dim, tmp, k, (T)0.0, Z2, dim);

        // C = alpha*(Z1 - Z2) + beta*C
        if (beta == (T)0) {
            for (int64_t i = 0; i < dim * n_vecs; ++i)
                C[i] = alpha * (Z1[i] - Z2[i]);
        } else {
            blas::scal(dim * n_vecs, beta, C, 1);
            blas::axpy(dim * n_vecs,  alpha, Z1, 1, C, 1);
            blas::axpy(dim * n_vecs, -alpha, Z2, 1, C, 1);
        }
    }
};


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
    // tmp, Z1, Z2: workspace owned here and lent to ResidualOp each call.
    std::vector<T> V_buf, eigvals_buf;   // std::vector for REVD2's adaptive resize
    T* F_vec = nullptr; int64_t F_vec_sz = 0;
    T* tmp   = nullptr; int64_t tmp_sz   = 0;
    T* Z1    = nullptr; int64_t Z1_sz    = 0;
    T* Z2    = nullptr; int64_t Z2_sz    = 0;

    ~FunNystromPP() { delete[] F_vec; delete[] tmp; delete[] Z1; delete[] Z2; }

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
        revd2.call(A, k_in, (T)0.0, V_buf, eigvals_buf, state);
        T* V        = V_buf.data();
        T* eigvals  = eigvals_buf.data();

        // f(λ): apply scalar f to k eigenvalues
        if (k > F_vec_sz) {
            delete[] F_vec;
            F_vec = new T[k];
            F_vec_sz = k;
        }
        for (int64_t i = 0; i < k; ++i)
            F_vec[i] = f(eigvals[i]);

        // tr(f(Â)) = Σ f(λ_i) + (n-k)*f(0)
        T tr_Ahat = (T)0.0;
        for (int64_t i = 0; i < k; ++i)
            tr_Ahat += F_vec[i];
        tr_Ahat += static_cast<T>(n - k) * f_zero;

        // ------------------------------------------------------------------
        // Phase 2: Hutchinson correction on residual f(A) - f(Â)
        // ResidualOp wraps the correction computation as a SymmetricLinearOperator
        // so Hutchinson::call can draw Ω and compute <Ω, (f(A)-f(Â))Ω>_F / s.
        // ------------------------------------------------------------------
        if (k * s > tmp_sz) {
            delete[] tmp;
            tmp = new T[k * s];
            tmp_sz = k * s;
        }
        if (n * s > Z1_sz) {
            delete[] Z1;
            Z1 = new T[n * s];
            Z1_sz = n * s;
        }
        if (n * s > Z2_sz) {
            delete[] Z2;
            Z2 = new T[n * s];
            Z2_sz = n * s;
        }

        ResidualOp<T, SLO, LanczosFA_t, F> res_op(
            n, A, lanczos_fa, f, d, k, V, F_vec, tmp, Z1, Z2
        );
        T correction = hutchinson.call(res_op, s, state);

        return tr_Ahat + correction;
    }
};


} // end namespace RandLAPACK
