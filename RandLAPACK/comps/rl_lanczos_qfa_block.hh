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
#include <cmath>
#include <limits>
#include <vector>

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
/// n×s output. With B = Q₀·R₀ (the initial block QR) and the block basis
/// orthonormal (Q₀ᵀ Q_p = δ_{0p} I),
///   Bᵀ f(A) B ≈ R₀ᵀ · [f(T_k)]_{1:s,1:s} · R₀,
/// the top-left s×s block of f(T_k) sandwiched by R₀.
///
/// Adaptive depth (optional, `adaptive = true`): instead of a fixed d, run the
/// recurrence while monitoring the block quadrature estimate tr(M_k) as the
/// depth k grows (the paper's Alg. 2 "qfa" certificate, lifted to blocks). Stop
/// at the first k where the change over a delay window δ is below a relative
/// tolerance: |tr(M_k) − tr(M_{k−δ})| ≤ rtol · |tr(M_k)|. This defines the depth
/// online (no offline calibration) and stops the recurrence early, spending
/// fewer matvecs. The actual depth used is reported in `d_used`.
///
/// The recurrence is reused from a held BlockLanczosFA instance (single-sourced,
/// so the reorth switch and matvec accounting stay in one place); only T_blk and
/// the initial factor R₀ are read here. The Krylov basis K_big is built by the
/// recurrence but never read by the reconstruction — exactly the mapback QFA
/// avoids.
///
/// @tparam T  Floating-point scalar type.
template <typename T>
class BlockLanczosQFA {
public:
    /// Reorthogonalization control (forwarded to the recurrence).
    ///  1 = full block reorthogonalization; 0 = none.
    int64_t reorth = 1;

    // ---- adaptive depth controls (used only when `adaptive`) --------------
    bool    adaptive        = false;   ///< choose depth online from the certificate
    T       adaptive_rtol   = (T)1e-2; ///< relative-change tolerance on tr(M_k)
    int64_t adaptive_delay  = 5;       ///< δ: compare depth k against k − δ checks
    int64_t adaptive_min    = 2;       ///< do not test convergence before this depth
    int64_t d_used          = 0;       ///< block steps actually used by the last call

    /// Reused block recurrence + its buffers (K_big, R0_buf, T_blk, ...).
    BlockLanczosFA<T> fa;

    // Scratch: apply/compute reuse `workspace` (eig_vals + W + T-copy); the
    // adaptive certificate uses a separate s×s M buffer so it never disturbs the
    // caller's output.
    T* workspace  = nullptr; int64_t workspace_sz  = 0;
    T* M_scratch  = nullptr; int64_t M_scratch_sz  = 0;

    bool timing = false;
    long _t_matvec_us = 0;

    BlockLanczosQFA()                                  = default;
    BlockLanczosQFA(const BlockLanczosQFA&)            = delete;
    BlockLanczosQFA& operator=(const BlockLanczosQFA&) = delete;

    ~BlockLanczosQFA() { delete[] workspace; delete[] M_scratch; }

    // ------------------------------------------------------------------
    /// Compute M = Bᵀ f(A) B (s×s, col-major, ld = s) into `out`. B is n×s
    /// col-major. Calls A up to d times (on n×s blocks); with `adaptive` it may
    /// stop earlier (see d_used). For the funNyström++ correction only tr(M) is
    /// needed, but the full s×s matrix is returned for generality.
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        fa.reorth = this->reorth;
        fa.timing = this->timing;

        if (this->adaptive) {
            util::upsize(this->M_scratch, this->M_scratch_sz, s * s);
            std::vector<T> hist;                 // tr(M_k) at tested depths
            const int64_t dmax = d;
            auto stop_after = [&](int64_t kdepth) -> bool {
                if (kdepth < this->adaptive_min || kdepth >= dmax) return false;
                compute_M(f, s, kdepth, dmax, this->M_scratch);
                T tr = (T)0;
                for (int64_t i = 0; i < s; ++i) tr += this->M_scratch[i + i * s];
                hist.push_back(tr);
                if ((int64_t)hist.size() > this->adaptive_delay) {
                    T prev = hist[hist.size() - 1 - this->adaptive_delay];
                    T scale = std::max(std::abs(tr), std::numeric_limits<T>::min());
                    if (std::abs(tr - prev) <= this->adaptive_rtol * scale) return true;
                }
                return false;
            };
            fa.run_lanczos(A, B, n, s, d, stop_after);
            this->d_used = fa.steps_run;
        } else {
            fa.run_lanczos(A, B, n, s, d);
            this->d_used = d;
        }

        this->_t_matvec_us = fa._t_matvec_us;
        compute_M(f, s, this->d_used, d, out);   // final M from the leading d_used block
    }

    // ------------------------------------------------------------------
    /// M = R₀ᵀ · [f(T_{1:kdepth})]_{1:s,1:s} · R₀ (s×s) into `out`, from the
    /// leading kdepth*s block of fa.T_blk (whose leading dimension is dmax*s).
    /// Copies that block out first, so fa.T_blk is NOT modified — the recurrence
    /// can continue after an adaptive certificate check.
    template <std::invocable<T> F>
    void compute_M(F f, int64_t s, int64_t kdepth, int64_t dmax, T* out) {
        const int64_t m      = kdepth * s;   // active tridiagonal dimension
        const int64_t src_ld = dmax   * s;   // leading dim of fa.T_blk
        util::upsize(workspace, workspace_sz, m + s * m + m * m);
        T* eig_vals = workspace;
        T* W        = eig_vals + m;   // s × m col-major (ld = s)
        T* V        = W + s * m;      // m × m: leading T block, then its eigenvectors

        // Copy the leading kdepth block (lower triangle; syevd reads Uplo::Lower)
        // into a compact m×m buffer so the source T_blk stays intact.
        lapack::lacpy(lapack::MatrixType::Lower, m, m, fa.T_blk, src_ld, V, m);
        lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, m, V, m, eig_vals);

        // W[i,j] = f(λⱼ)·V[i,j] for i = 0..s-1 (first s rows of V, column-scaled).
        for (int64_t j = 0; j < m; ++j) {
            T fev = f(std::max(eig_vals[j], (T)0));
            const T* V_col = V + j * m;
            T*       W_col = W + j * s;
            for (int64_t i = 0; i < s; ++i)
                W_col[i] = fev * V_col[i];
        }

        // [f(T_k)]_{1:s,1:s} = P·Wᵀ, P = first s rows of V (s×m, lda = m).
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
