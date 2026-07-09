#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"
#include "rl_lanczos_fa.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <vector>
#include <cstring>

namespace RandLAPACK {


/// d-step block Lanczos for matrix function application f(A)B.
/// Builds a single joint block Krylov subspace using BLAS-3 throughout,
/// replacing the s independent scalar Lanczos sequences in LanczosFA.
///
/// Algorithm reference: T. Chen, "A handbook for matrix-function-based Krylov methods",
///   arXiv:2410.11090 (2024). Algorithm 9.2 (block Lanczos recurrence),
///   Definition 9.6 (block Lanczos-FA).
///
/// At each recurrence step all updates (alpha, beta, Z) are BLAS-3 (GEMM + GEQRF).
/// apply_f calls a single syevd on the (d*s)×(d*s) block tridiagonal instead of
/// s separate stevd calls of size d.
///
/// Block Lanczos-FA formula:  out ≈ Q_basis * f(T_k) * E₁ * R₀
///   Q_basis = [Q₀|...|Q_{d-1}] (n×d*s), T_k = d*s×d*s block tridiagonal,
///   E₁ = first d*s×s columns of identity, B = Q₀*R₀ (initial QR).
///
/// Pseudocode (Chen 2024 Alg. 9.2 recurrence + Def. 9.6 reconstruction). Code
/// blocks in run_lanczos / apply_f are tagged [line N] against this listing:
///    (1)  Q₀, R₀ ← qr(B)                                (initial block QR)
///    (2)  for i = 0..d−1:
///    (3)      Z ← A·Q_i                                 (1 batch matvec)
///    (4)      if i > 0:  Z ← Z − Q_{i−1}·B_{i−1}ᵀ
///    (5)      A_i ← Q_iᵀ·Z;  symmetrize                 (s×s block-α)
///    (6)      Z ← Z − Q_i·A_i
///    (7)      if reorth:  for p = 0..i:  Z ← Z − Q_p·(Q_pᵀ·Z)   (block MGS)
///    (8)      if i < d−1:  Q_{i+1}, B_i ← qr(Z)         (s×s block-β = R factor)
///    (9)  T_k ← blocktridiag({A_i}, {B_i})              (d·s × d·s; in code this is
///                                                        fused into lines 2-8: the blocks
///                                                        are written into T_blk in place)
///   (10)  (V, λ) ← syevd(T_k)
///   (11)  F ← Q_basis · [ V·diag(f(λ))·V[1:s,:]ᵀ ] · R₀ (= Q_basis·f(T_k)·E₁·R₀)
///
/// Known limitation (v1): deflation is not implemented.  When the block Krylov
/// space fills before d steps (B_step develops near-zero singular values), accuracy
/// degrades but the algorithm does not crash.  For problems where d*s approaches the
/// effective rank of A, use a smaller d or the scalar LanczosFA.
///
/// @tparam T    Floating-point scalar type.
template <typename T>
class BlockLanczosFA {
public:
    /// Reorthogonalization control.
    ///  1 = full (project each new block Z out of all previous Krylov blocks).
    ///  0 = none.
    int64_t reorth = 1;

    // Internal buffers — grown with new/delete[], never shrunk between calls.
    // Dimension key: n = operator dimension, s = block size, d = Lanczos steps.
    //
    // K_big:     (d+1)*n*s  — block Krylov basis + matvec scratch.
    //   Layout: K_big[step*n*s .. (step+1)*n*s-1] = Q_step (n×s col-major, ld=n).
    //   First d blocks form Q_basis (n×d*s col-major, ld=n) used by apply_f.
    //   Block d is scratch for the current-step matvec output.
    // R0_buf:    s*s         — upper triangular factor from initial QR of B.
    // tau_buf:   s           — Householder scalars (geqrf/orgqr on n×s panels
    //   need min(n, s) = s of them); reused at each geqrf/orgqr call.
    // T_blk:     (d*s)^2     — the block tridiagonal T_k, assembled IN PLACE by
    //   the recurrence (no separate A_blk/B_blk copies): block alpha A_step is
    //   GEMM'd directly into the diagonal-block position (b0, b0), block beta
    //   B_step (upper triangular) is copied once from the QR'd Z into the lower
    //   off-diagonal position (b0+s, b0); ld = d*s throughout. Only the lower
    //   triangle is populated (syevd reads Uplo::Lower). apply_f's syevd then
    //   overwrites T_blk with the eigenvectors, so one run_lanczos supports
    //   exactly one apply_f (the call() pairing; re-applying a different f
    //   requires re-running the recurrence).
    // workspace: apply_f scratch — eig_vals (d*s) + G (d*s×s) + C1 (d*s×s).
    // proj_buf:  s*s — reorthogonalization scratch (Q_p^T * Y projection); reused across steps.
    T* K_big     = nullptr; int64_t K_big_sz     = 0;
    T* R0_buf    = nullptr; int64_t R0_sz        = 0;
    T* tau_buf   = nullptr; int64_t tau_buf_sz   = 0;
    T* T_blk     = nullptr; int64_t T_blk_sz     = 0;
    T* workspace = nullptr; int64_t workspace_sz = 0;
    T* proj_buf  = nullptr; int64_t proj_buf_sz  = 0;

    bool timing = false;
    std::vector<long> times;
    long _t_matvec_us = 0;

    BlockLanczosFA()                                 = default;
    BlockLanczosFA(const BlockLanczosFA&)            = delete;
    BlockLanczosFA& operator=(const BlockLanczosFA&) = delete;

    ~BlockLanczosFA() {
        delete[] K_big; delete[] R0_buf; delete[] tau_buf;
        delete[] T_blk; delete[] workspace; delete[] proj_buf;
    }

    // ------------------------------------------------------------------
    /// Run the d-step block Lanczos recurrence on B (n×s col-major).
    /// Fills K_big, R0_buf, and T_blk (the block tridiagonal, assembled in
    /// place at its final positions — see the member comment).
    /// Calls A exactly d times, each applied to an n×s block.
    ///
    /// T_k layout built here (lower triangle only; syevd reads Uplo::Lower):
    ///
    ///         ┌  A₀                        ┐
    ///         │  B₀   A₁                   │
    ///   T_k = │       B₁   A₂              │      Aᵢ s×s symmetric (GEMM'd in),
    ///         │              ⋱   ⋱         │      Bᵢ s×s upper triangular (R of
    ///         └                  B_{d-2}  A_{d-1}┘  the QR of Z at step i+1)
    template <linops::SymmetricLinearOperator SLO>
    void run_lanczos(SLO& A, const T* B, int64_t n, int64_t s, int64_t d) {
        using namespace std::chrono;
        steady_clock::time_point _mv_t0, _mv_t1;
        _t_matvec_us = 0;
        const int64_t m = d * s;   // T_k dimension / its leading dimension

        util::upsize(K_big,   K_big_sz,   (d + 1) * n * s);
        util::upsize(R0_buf,  R0_sz,      s * s);
        util::upsize(tau_buf, tau_buf_sz, s);
        util::upsize(T_blk,   T_blk_sz,   m * m);
        if (reorth) util::upsize(proj_buf, proj_buf_sz, s * s);

        // Zero T_blk's background once per run (persistent workspace: it holds
        // the previous call's eigenvectors); the recurrence then writes only
        // the band blocks.
        std::memset(T_blk, 0, m * m * sizeof(T));

        // [line 1] Q₀, R₀ ← qr(B).  Q0 overwrites K_big[0..n*s-1].
        T* Q0 = K_big;
        lapack::lacpy(lapack::MatrixType::General, n, s, B, n, Q0, n);
        lapack::geqrf(n, s, Q0, n, tau_buf);
        lapack::laset(lapack::MatrixType::General, s, s, (T)0, (T)0, R0_buf, s);
        lapack::lacpy(lapack::MatrixType::Upper, s, s, Q0, n, R0_buf, s);
        lapack::orgqr(n, s, s, Q0, n, tau_buf);

        // [line 2] for i = 0..d−1 (i ≡ step)
        for (int64_t step = 0; step < d; ++step) {
            T* Q_step = K_big + step       * n * s;
            T* Q_prev = (step > 0) ? K_big + (step - 1) * n * s : nullptr;
            T* Y      = K_big + (step + 1) * n * s;   // matvec output, then Z in-place
            const int64_t b0 = step * s;
            // Block positions inside T_blk (ld = m): A_step at (b0, b0),
            // B_{step-1} at (b0, b0 - s) — written directly, no staging copies.
            T* A_step = T_blk + b0 * m + b0;
            T* B_prev = (step > 0) ? T_blk + (b0 - s) * m + b0 : nullptr;

            // [line 3] Z ← A·Q_i  (1 batch matvec; Z lives in Y)
            if (this->timing) _mv_t0 = steady_clock::now();
            A(Layout::ColMajor, s, (T)1.0, Q_step, n, (T)0.0, Y, n);
            if (this->timing) { _mv_t1 = steady_clock::now(); _t_matvec_us += duration_cast<microseconds>(_mv_t1 - _mv_t0).count(); }

            // [line 4] if i > 0:  Z ← Z − Q_{i−1}·B_{i−1}ᵀ
            if (step > 0)
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                           n, s, s, (T)-1.0, Q_prev, n, B_prev, m, (T)1.0, Y, n);

            // [line 5] A_i ← Q_iᵀ·Z; symmetrize  (block alpha, s×s), GEMM'd
            // directly into its diagonal-block slot of T_blk.
            // Symmetric in exact arithmetic since Q_stepᵀ·A·Q_step is, but the
            // general GEMM's two triangles disagree at roundoff. The syevd that
            // eventually consumes the block tridiagonal reads a single triangle,
            // so average with util::symmetrize to factor the symmetric part.
            // Same rationale as the Gram symmetrize in rl_nystrom_evd.hh.
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                       s, s, n, (T)1.0, Q_step, n, Y, n, (T)0.0, A_step, m);
            util::symmetrize(s, A_step, m);

            // [line 6] Z ← Z − Q_i·A_i  (in-place in Y)
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, s, s, (T)-1.0, Q_step, n, A_step, m, (T)1.0, Y, n);

            // [line 7] if reorth: for p = 0..i:  Z ← Z − Q_p·(Q_pᵀ·Z)
            // (full block modified Gram-Schmidt against every previous block).
            // NB a batched block-CLASSICAL-GS variant (one GEMM pair against the
            // whole Q_basis[0..i] at once) was attempted for BLAS-3 locality but
            // regressed orthogonality (left ‖QᵀQ−I‖ ~ 1); reverted pending a
            // correct CGS. Toggle reorthogonalization with `reorth`.
            if (reorth) {
                for (int64_t prev = 0; prev <= step; ++prev) {
                    T* Q_p = K_big + prev * n * s;
                    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                               s, s, n, (T)1.0, Q_p, n, Y, n, (T)0.0, proj_buf, s);
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                               n, s, s, (T)-1.0, Q_p, n, proj_buf, s, (T)1.0, Y, n);
                }
            }

            // [line 8] if i < d−1:  Q_{i+1}, B_i ← qr(Z)
            // (Q_{step+1} overwrites Y. B_step = upper R factor, copied once
            //  from the QR'd Z into its lower off-diagonal slot (b0+s, b0) of
            //  T_blk — the tile's strict lower stays zero from the memset.
            //  Skipped at the last step: Q_d is never needed by apply_f.)
            if (step < d - 1) {
                T* B_step = T_blk + b0 * m + (b0 + s);
                lapack::geqrf(n, s, Y, n, tau_buf);
                lapack::lacpy(lapack::MatrixType::Upper, s, s, Y, n, B_step, m);
                lapack::orgqr(n, s, s, Y, n, tau_buf);
            }
        }
    }

    // ------------------------------------------------------------------
    /// Evaluate f(A)B from precomputed Krylov data (K_big, R0_buf, T_blk)
    /// — lines 9-11 of the class-doc pseudocode. Line 9 (assembling T_k) has
    /// already happened: the recurrence wrote the blocks into T_blk in place.
    ///
    /// CONSUMES T_blk: the syevd overwrites it with the eigenvectors, so one
    /// run_lanczos supports exactly one apply_f (the call() pairing).
    ///
    /// Computation:
    ///  [line 10]  syevd: T_blk → eigenvectors V (in-place), eigenvalues λ.
    ///  [line 11a] W (s×m): W[i,j] = f(λⱼ)*V[i,j] for i=0..s-1 (first s rows of V, col-scaled).
    ///  [line 11b] C1 (d*s × s) = V * W^T  — this equals f(T_k) * E₁.
    ///  [line 11c] C1 *= R₀  (TRMM: right-multiply by upper-triangular R₀).
    ///  [line 11d] out (n × s) = Q_basis * C1  (GEMM).
    template <std::invocable<T> F>
    void apply_f(F f, int64_t n, int64_t s, int64_t d, T* out) {
        int64_t m = d * s;
        util::upsize(workspace, workspace_sz, m + 2 * m * s);

        T* eig_vals = workspace;
        T* G        = eig_vals + m;
        T* C1       = G + m * s;
        T* T_dense  = T_blk;   // eigenvectors V overwrite the block tridiagonal

        // [line 10] (V, λ) ← syevd(T_k): eigenvectors V overwrite T_blk, eig_vals → λ.
        lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, m, T_dense, m, eig_vals);

        // [line 11a] W (s×m col-major, ld=s; stored in the G buffer): W[i,j] = f(λⱼ)*V[i,j]
        //    for i=0..s-1, j=0..m-1.  Each column j of W is the first s elements of
        //    column j of V, scaled by f(λⱼ).
        //
        //    Derivation: f(T_k)*E₁ = V*diag(f(λ))*V^T*E₁.
        //    V^T*E₁ = first s columns of V^T = {row i of V} for i=0..s-1 stacked as cols.
        //    diag(f(λ))*(V^T*E₁): scale row j → f(λⱼ)*(row j of V^T*E₁) = f(λⱼ)*V[:,j][0:s].
        //    That intermediate is W^T (m×s), so f(T_k)*E₁ = V * W^T.
        T* W = G;   // reuse G buffer; W is s×m col-major (ld=s)
        for (int64_t j = 0; j < m; ++j) {
            T fev = f(std::max(eig_vals[j], (T)0));
            const T* V_col = T_dense + j * m;   // col j of V (contiguous, length m)
            T*       W_col = W       + j * s;   // col j of W (contiguous, length s)
            for (int64_t i = 0; i < s; ++i)
                W_col[i] = fev * V_col[i];      // first s rows of V[:,j], scaled
        }

        // [line 11b] C1 (m × s) = V * W^T  — equals f(T_k) * E₁.
        //    GEMM(NoTrans, Trans, m, s, m): C = V(m×m) * W^T  where W is s×m (ld=s).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                   m, s, m, (T)1.0, T_dense, m, W, s, (T)0.0, C1, m);

        // [line 11c] C1 *= R₀  (TRMM: C1 = C1 * R₀, right upper triangular).
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   m, s, (T)1.0, R0_buf, s, C1, m);

        // [line 11d] out (n × s) = Q_basis (n × m) * C1 (m × s): F = Q_basis·f(T_k)·E₁·R₀.
        //    Q_basis = K_big[0..m*n-1] (n×m col-major, ld=n).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   n, s, m, (T)1.0, K_big, n, C1, m, (T)0.0, out, n);
    }

    // ------------------------------------------------------------------
    /// Combined run + apply.
    ///
    /// Drop-in replacement for LanczosFA::call — same signature, slots into
    /// FunNystromPP and ResidualOp as the LanczosFA_t template parameter.
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        detail::lanczos_fa_timed_call<T>(*this, A, B, n, s, f, d, out);
    }
};


} // end namespace RandLAPACK
