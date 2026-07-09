#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"
#include "rl_lanczos_fa_block.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <concepts>
#include <algorithm>

namespace RandLAPACK {


/// d-step block Lanczos-QFA: the *quadratic form* Bᵀ f(A) B (s×s), NOT the
/// vector block f(A)B. This is the draft's Algorithm 1 line 5 primitive
/// (Lanczos-QFA, the ♠ RAM note): by the Gauss-quadrature identity
///   gᵢᵀ · LanczosFA(A, f, gᵢ) = Lanczos-QFA(A, f, gᵢ),
/// so the funNyström++ Phase-2 term tr(Ω₂ᵀ f(A) Ω₂) can be formed WITHOUT ever
/// materializing f(A)·Ω₂.
///
/// Cost vs BlockLanczosFA: identical recurrence (same d matvecs, same T_k), but
/// the reconstruction replaces the O(n·d·s²) mapback  out = Q_basis·f(T_k)·E₁·R₀
/// (BlockLanczosFA line 11d) with an O(d·s³) small-matrix computation and NO
/// n×s output. Derivation, with B = Q₀·R₀ (the initial block QR) and
/// f(A)B ≈ Q_basis·f(T_k)·E₁·R₀:
///   Bᵀ f(A) B ≈ R₀ᵀ·(Q₀ᵀ Q_basis)·f(T_k)·E₁·R₀ = R₀ᵀ·E₁ᵀ f(T_k) E₁·R₀
///             = R₀ᵀ · [f(T_k)]_{1:s,1:s} · R₀,
/// using Q₀ᵀ Q_p = δ_{0p} I (the block basis is orthonormal — exact with
/// reorthogonalization, approximate without, same level as BlockLanczosFA).
///
/// The recurrence is reused from a held BlockLanczosFA instance (single-sourced
/// so the CGS/reorth switch and matvec accounting stay in one place). Only the
/// tridiagonal T_blk and the initial factor R₀ are consumed here; the Krylov
/// basis K_big is built by the recurrence but never read by apply_qf — that is
/// exactly the mapback that QFA avoids.
///
/// @tparam T  Floating-point scalar type.
template <typename T>
class BlockLanczosQFA {
public:
    /// Reorthogonalization control (forwarded to the recurrence).
    ///  1 = full block CGS against all previous blocks;  0 = none.
    int64_t reorth = 1;

    /// Reused block recurrence + its buffers (K_big, R0_buf, T_blk, ...).
    BlockLanczosFA<T> fa;

    // apply_qf scratch: eig_vals (d*s) followed by W (s × d*s, col-major ld=s).
    T* workspace = nullptr; int64_t workspace_sz = 0;

    bool timing = false;
    long _t_matvec_us = 0;

    BlockLanczosQFA()                                  = default;
    BlockLanczosQFA(const BlockLanczosQFA&)            = delete;
    BlockLanczosQFA& operator=(const BlockLanczosQFA&) = delete;

    ~BlockLanczosQFA() { delete[] workspace; }

    // ------------------------------------------------------------------
    /// Compute M = Bᵀ f(A) B (s×s, col-major, ld = s) into `out`.
    /// B is n×s col-major. Calls A exactly d times (on n×s blocks).
    /// For the funNyström++ correction only tr(M) is needed, but the full
    /// s×s matrix is returned for generality.
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        fa.reorth = this->reorth;
        fa.timing = this->timing;
        fa.run_lanczos(A, B, n, s, d);
        this->_t_matvec_us = fa._t_matvec_us;
        apply_qf(f, s, d, out);
    }

    // ------------------------------------------------------------------
    /// M = R₀ᵀ · [f(T_k)]_{1:s,1:s} · R₀ from the recurrence's T_blk / R0_buf.
    /// CONSUMES fa.T_blk (the syevd overwrites it with eigenvectors), so one
    /// run_lanczos supports exactly one apply_qf — the call() pairing.
    template <std::invocable<T> F>
    void apply_qf(F f, int64_t s, int64_t d, T* out) {
        const int64_t m = d * s;
        util::upsize(workspace, workspace_sz, m + s * m);
        T* eig_vals = workspace;
        T* W        = eig_vals + m;   // s × m col-major (ld = s)
        T* V        = fa.T_blk;       // eigenvectors overwrite the block tridiagonal

        // (V, λ) ← syevd(T_k) (reads Uplo::Lower, as the recurrence populated).
        lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, m, V, m, eig_vals);

        // W[i,j] = f(λⱼ)·V[i,j] for i = 0..s-1 (first s rows of V, column-scaled)
        // — identical to BlockLanczosFA line 11a. Clamp negatives to 0 for √.
        for (int64_t j = 0; j < m; ++j) {
            T fev = f(std::max(eig_vals[j], (T)0));
            const T* V_col = V + j * m;
            T*       W_col = W + j * s;
            for (int64_t i = 0; i < s; ++i)
                W_col[i] = fev * V_col[i];
        }

        // [f(T_k)]_{1:s,1:s} = P·Wᵀ, P = first s rows of V (s×m, lda = m):
        //   out[a,b] = Σⱼ V[a,j]·f(λⱼ)·V[b,j].  GEMM(NoTrans, Trans, s, s, m).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                   s, s, m, (T)1.0, V, m, W, s, (T)0.0, out, s);

        // M = R₀ᵀ · out · R₀  (R₀ upper triangular s×s in fa.R0_buf, ld = s).
        blas::trmm(Layout::ColMajor, Side::Left,  Uplo::Upper, Op::Trans,   Diag::NonUnit,
                   s, s, (T)1.0, fa.R0_buf, s, out, s);
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   s, s, (T)1.0, fa.R0_buf, s, out, s);
    }
};


} // end namespace RandLAPACK
