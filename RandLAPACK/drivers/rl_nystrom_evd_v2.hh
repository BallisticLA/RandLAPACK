#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace RandLAPACK {

// --- Phase 1: free-standing NystromEVD_v2 ----------------------------------
//
// Reference-aligned sketched Nyström spectral recovery. Used by
// `FunNystromPP_v2` in `rl_fun_nystrom_pp_v2.hh`.

/// Reference-aligned sketched Nyström spectral recovery.
/// Direct port of davpersson/funNystrom/Other/nystrom.m with the Phase 7a
/// Cholesky-fast / eig(YᵀY) fast path bolted on (HMT §5.1).
///
/// Computes a rank-k approximation Â = U · diag(λ) · Uᵀ via:
///   Ω ← qr(Ω, 0)
///   for iter = 1..q-1:  Ω ← qr(A·Ω, 0)
///   Y ← A · Ω
///   G ← Ωᵀ · Y
///   try potrf(G) → if success, fast path (TRSM + SYRK + syevd of k×k Gram);
///                  if fail, fall-back (syevd of G + pinv + gesdd of m×k B).
///
/// Stores U (m × k) and λ (length k, descending) in the caller-supplied
/// vectors. Workspace is owned by the caller through the `workspace` struct
/// so it can be reused across calls.
template <typename T>
struct NystromEVD_v2_workspace {
    std::vector<T> Omega;      // m × k
    std::vector<T> Y;          // m × k
    std::vector<T> G;          // k × k
    std::vector<T> G_backup;   // k × k (Gram backup for Cholesky-fast fall-back; Phase 7)
    std::vector<T> V_G;        // k × k (left singular vectors of G, fall-back path)
    std::vector<T> VT_G;       // k × k (right singular vectors of G, fall-back path)
    std::vector<T> D;          // k (singular values of G, fall-back path; also σ² in fast path)
    std::vector<T> tmp_kk;     // k × k (V_G · pinv(√D) · V_Gᵀ on fall-back; unused in fast)
    std::vector<T> Sigma;      // k (singular values of B, fall-back; σ² of Y in fast)
    std::vector<T> VT_B;       // k × k (gesdd VT output on fall-back; unused in fast)
    std::vector<T> tau;        // k (geqrf reflectors)
};

template <typename T, linops::SymmetricLinearOperator SLO>
void NystromEVD_v2(
    SLO &A_op,
    int64_t k,
    int64_t q,
    const T *Omega1_in,
    std::vector<T> &U_out,            // populated to size m·k, column-major
    std::vector<T> &lambda_out,        // populated to size k, descending
    NystromEVD_v2_workspace<T> &ws,
    bool force_fallback = false,       // Phase 7a perf knob: skip Cholesky-fast, take SVD-pinv path. Default false (normal dual-path behavior).
    double *t_specrec_ms_out = nullptr  // optional: write wall-clock ms of just the dual-path spectral-recovery block (post Y = A·Ω).
) {
    using namespace blas;
    int64_t m = A_op.dim;

    // Allocate / resize workspace.
    ws.Omega.assign(m * k, (T)0);
    ws.Y.assign(m * k, (T)0);
    ws.G.assign(k * k, (T)0);
    ws.G_backup.assign(k * k, (T)0);
    ws.V_G.assign(k * k, (T)0);
    ws.VT_G.assign(k * k, (T)0);
    ws.D.assign(k, (T)0);
    ws.tmp_kk.assign(k * k, (T)0);
    ws.Sigma.assign(k, (T)0);
    ws.VT_B.assign(k * k, (T)0);
    ws.tau.assign(k, (T)0);
    U_out.assign(m * k, (T)0);
    lambda_out.assign(k, (T)0);

    // Step 1: Ω ← qr(Ω₁_in, 0).
    std::copy(Omega1_in, Omega1_in + m * k, ws.Omega.data());
    lapack::geqrf(m, k, ws.Omega.data(), m, ws.tau.data());
    lapack::ungqr(m, k, k, ws.Omega.data(), m, ws.tau.data());

    // Steps 2–3: q − 1 subspace-iteration passes, each: Ω ← qr(A · Ω, 0).
    for (int64_t iter = 1; iter < q; ++iter) {
        A_op(Layout::ColMajor, k, (T)1, ws.Omega.data(), m, (T)0, ws.Y.data(), m);
        std::copy(ws.Y.begin(), ws.Y.end(), ws.Omega.begin());
        lapack::geqrf(m, k, ws.Omega.data(), m, ws.tau.data());
        lapack::ungqr(m, k, k, ws.Omega.data(), m, ws.tau.data());
    }

    // Step 4: Y ← A · Ω.
    A_op(Layout::ColMajor, k, (T)1, ws.Omega.data(), m, (T)0, ws.Y.data(), m);

    // ---- Phase 7: dual-path spectral recovery ----
    //
    // Try Cholesky on the k×k Gram first. If the Gram is well-conditioned
    // (the common case), use the Halko-Martinsson-Tropp §5.1 eig(YᵀY)
    // trick to avoid the n×k gesdd of B. If Cholesky fails (rank-
    // deficient Gram), fall back to the SVD-pseudoinverse path we used
    // before — that path is mathematically identical to Persson's
    // nystrom.m and what we cross-validated against in Phase 2.
    //
    // Both paths produce the same Â and the same (U, λ) up to ε_mach
    // floating-point ordering. Verified bit-equal in Phase 2's cross-
    // validation harness when the Gram is far from singular.

    // Tight timer for the spectral-recovery block (Phase 7a A/B). Starts
    // at Gram formation; ends after λ_out is populated. Excludes the
    // shared QR / subspace-iter / final-matvec costs which are identical
    // on both paths.
    auto t_specrec_start = std::chrono::steady_clock::now();

    // Step 5: Form Gram G = Ωᵀ · Y (k×k, sym PSD). Symmetrize FP noise.
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
               (T)1, ws.Omega.data(), m, ws.Y.data(), m, (T)0, ws.G.data(), k);
    RandLAPACK::util::symmetrize(k, ws.G.data(), k);

    // Backup G into G_backup before potrf destroys it (lacpy General copies
    // both triangles).
    lapack::lacpy(lapack::MatrixType::General, k, k, ws.G.data(), k,
                  ws.G_backup.data(), k);

    // Try Cholesky (skipped when force_fallback set — Phase 7a perf knob).
    int chol_status = force_fallback
        ? 1   // any non-zero so we take the fall-back branch
        : lapack::potrf(Uplo::Upper, k, ws.G.data(), k);

    const T eps_mach = std::numeric_limits<T>::epsilon();

    if (chol_status == 0) {
        // ---------- FAST PATH (Cholesky-fast + eig(YᵀY)) ----------
        //
        //   Y    := Y · R⁻¹                          (TRSM; R is upper-tri Cholesky factor)
        //   G_y  = Yᵀ · Y                            (k×k syrk)
        //   [V, σ²] = syevd(G_y)                      (eigendecomp; ASC order)
        //   reverse V cols + σ² to descending
        //   U    = Y · V                              (m×k gemm)
        //   scale U[:,j] by 1/σⱼ (threshold tiny σ² as 0)
        //   λ    = σ²
        //
        // No n×k gesdd needed (vs the fall-back's gesdd of B). Big win
        // when k ≈ n; same answer up to FP ordering.

        // Zero strict lower of the Cholesky factor R.
        RandLAPACK::util::get_U(k, k, ws.G.data(), k);

        // Y := Y · R⁻¹.
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper,
                   Op::NoTrans, Diag::NonUnit, m, k,
                   (T)1, ws.G.data(), k, ws.Y.data(), m);

        // G_y = Yᵀ · Y (k×k, sym PSD). Overwrite ws.G (the Cholesky factor
        // is no longer needed).
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                   k, m, (T)1, ws.Y.data(), m, (T)0, ws.G.data(), k);

        // syevd(G_y) → ws.G now holds eigenvectors, ws.D holds σ² (ASC).
        lapack::syevd(lapack::Job::Vec, Uplo::Upper, k,
                      ws.G.data(), k, ws.D.data());

        // Reverse to descending order so eigvals[0] = λ_max (the rest of
        // the FunNystromPP pipeline assumes descending).
        for (int64_t ii = 0; ii < k / 2; ++ii) {
            std::swap(ws.D[ii], ws.D[k - 1 - ii]);
        }
        for (int64_t jj = 0; jj < k / 2; ++jj) {
            blas::swap(k, ws.G.data() + jj * k, 1,
                          ws.G.data() + (k - 1 - jj) * k, 1);
        }

        // U = Y · V (unscaled left singular vectors of B).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
                   (T)1, ws.Y.data(), m, ws.G.data(), k, (T)0,
                   U_out.data(), m);

        // Scale columns of U by 1/√σ². Threshold tiny σ² to 0 (column zeroed).
        T sig_sq_max    = (k > 0) ? std::max(ws.D[0], (T)0) : (T)0;
        T sig_sq_thresh = (T)4 * eps_mach * sig_sq_max;
        for (int64_t j = 0; j < k; ++j) {
            if (ws.D[j] > sig_sq_thresh) {
                T inv_sig = (T)1 / std::sqrt(ws.D[j]);
                blas::scal(m, inv_sig, U_out.data() + j * m, 1);
            } else {
                std::fill(U_out.data() + j * m,
                          U_out.data() + (j + 1) * m, (T)0);
            }
        }

        // λ = σ² (clamp negatives to 0).
        for (int64_t i = 0; i < k; ++i) {
            lambda_out[i] = std::max(ws.D[i], (T)0);
        }
    } else {
        // ---------- FALL-BACK (eig-pseudoinverse) ----------
        //
        // Restore Gram into ws.G from backup.
        lapack::lacpy(lapack::MatrixType::General, k, k,
                      ws.G_backup.data(), k, ws.G.data(), k);

        // syevd on the symmetric (and intended-PSD) k×k Gram. Standard
        // LAPACK call for symmetric eigendecomp; preferred over gesdd
        // on a symmetric input because the algorithm exploits structure
        // (~2× faster, tighter ε guarantees). After the call:
        //   ws.G  holds eigenvectors V_G as columns
        //   ws.D  holds eigenvalues (ASCENDING order)
        // Aligns with the production driver in PR #132 (rl_nystrom_evd.hh).
        lapack::syevd(lapack::Job::Vec, Uplo::Upper, k,
                      ws.G.data(), k, ws.D.data());

        // Pinv threshold: drop eigenvalues below 2·ε·D_max where
        // D_max = D[k-1] (largest, ascending). Matches the production
        // driver's constant; tighter than the previous 5e-16 MATLAB
        // copy and indistinguishable on well-conditioned fixtures.
        const T D_max  = (k > 0) ? std::max(ws.D[k - 1], (T)0) : (T)0;
        const T thresh = (T)2 * eps_mach * D_max;

        // Form B = Y · V_G · diag(1/√D) · V_Gᵀ. (3 stages)
        //   1) U_out = Y · V_G                                (m × k)
        //   2) scale columns of U_out by 1/√D[j] (or 0)
        //   3) ws.Y = U_out · V_Gᵀ                            (m × k)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
                   (T)1, ws.Y.data(), m, ws.G.data(), k, (T)0,
                   U_out.data(), m);
        for (int64_t j = 0; j < k; ++j) {
            T scale = (ws.D[j] > thresh) ? (T)1 / std::sqrt(ws.D[j]) : (T)0;
            blas::scal(m, scale, U_out.data() + j * m, 1);
        }
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, k, k,
                   (T)1, U_out.data(), m, ws.G.data(), k, (T)0,
                   ws.Y.data(), m);

        // Step 8: [U, Σ, _] ← svd(B).
        lapack::gesdd(lapack::Job::SomeVec, m, k, ws.Y.data(), m,
                      ws.Sigma.data(), U_out.data(), m, ws.VT_B.data(), k);

        // Step 9: λ ← Σ². Zero columns of U_out for tiny singular values
        // (rank-deficient B) to match the production driver.
        const T Sig_max    = (k > 0) ? ws.Sigma[0] : (T)0;
        const T Sig_thresh = (T)2 * eps_mach * Sig_max;
        int64_t r = 0;
        for (int64_t i = 0; i < k; ++i) {
            lambda_out[i] = ws.Sigma[i] * ws.Sigma[i];
            if (ws.Sigma[i] > Sig_thresh) ++r;
        }
        std::fill(U_out.data() + m * r, U_out.data() + m * k, (T)0);
    }

    auto t_specrec_end = std::chrono::steady_clock::now();
    if (t_specrec_ms_out) {
        *t_specrec_ms_out = std::chrono::duration<double, std::milli>(
            t_specrec_end - t_specrec_start).count();
    }
}


} // namespace RandLAPACK
