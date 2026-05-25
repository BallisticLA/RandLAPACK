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

// --- Phase 1: free-standing NystromEVD ------------------------------------
//
// Reference-aligned sketched Nyström spectral recovery. Used by
// `FunNystromPP_v2` in `rl_fun_nystrom_pp_v2.hh`.

// Heap-owned workspace for `NystromEVD`. Buffers grow on demand via
// `util::upsize`; existing contents are not preserved across calls
// (each call re-fills what it reads). The struct is deliberately a
// plain pointer-bag rather than `std::vector`-of-`T` so that callers
// (FunNystromPP_v2) can keep it alive across many `call()` invocations
// at amortised allocation cost.
template <typename T>
struct NystromEVD_workspace {
    T* Omega    = nullptr; int64_t Omega_sz    = 0;   // m × k
    T* Y        = nullptr; int64_t Y_sz        = 0;   // m × k
    T* G        = nullptr; int64_t G_sz        = 0;   // k × k
    T* G_backup = nullptr; int64_t G_backup_sz = 0;   // k × k (Phase 7 dual-path)
    T* D        = nullptr; int64_t D_sz        = 0;   // k       (eigenvalues of G in fall-back; σ² in fast)
    T* Sigma    = nullptr; int64_t Sigma_sz    = 0;   // k       (gesdd singular values of B, fall-back only)
    T* VT_B     = nullptr; int64_t VT_B_sz     = 0;   // k × k   (gesdd VT, fall-back only)
    T* tau      = nullptr; int64_t tau_sz      = 0;   // k       (geqrf reflectors)

    // 11-slot timing vector (microseconds), populated when `times_enabled`.
    // Layout mirrors PR #132's NystromEVD: 0 alloc, 1 syrf, 2 matvec, 3 gram,
    // 4 potrf, 5 trsm, 6 svd, 7 post_svd, 8 error_est, 9 rest, 10 total.
    // Slot 8 is unused in this fixed-k variant (no power_error_est); kept
    // at index 8 to preserve V1's layout for cross-driver tooling.
    bool times_enabled = false;
    std::vector<long> times = std::vector<long>(11, 0L);

    NystromEVD_workspace() = default;
    NystromEVD_workspace(const NystromEVD_workspace&) = delete;
    NystromEVD_workspace& operator=(const NystromEVD_workspace&) = delete;

    ~NystromEVD_workspace() {
        delete[] Omega;
        delete[] Y;
        delete[] G;
        delete[] G_backup;
        delete[] D;
        delete[] Sigma;
        delete[] VT_B;
        delete[] tau;
    }
};


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
/// Outputs `U_out` (m × k, column-major) and `lambda_out` (length k,
/// descending) as raw heap buffers managed via `util::upsize`. The
/// caller owns the buffers and is responsible for `delete[]`-ing them
/// (FunNystromPP_v2 holds them as class members and frees in its dtor).
template <typename T, linops::SymmetricLinearOperator SLO>
void NystromEVD(
    SLO &A_op,
    int64_t k,
    int64_t q,
    const T *Omega1_in,
    T*& U_out,        int64_t& U_out_sz,
    T*& lambda_out,   int64_t& lambda_out_sz,
    NystromEVD_workspace<T> &ws,
    bool force_fallback = false,       // Phase 7a perf knob: skip Cholesky-fast, take SVD-pinv path. Default false (normal dual-path behavior).
    double *t_specrec_ms_out = nullptr  // optional: write wall-clock ms of just the dual-path spectral-recovery block (post Y = A·Ω).
) {
    using namespace blas;
    int64_t m = A_op.dim;

    using clk = std::chrono::steady_clock;
    auto t_total_start = clk::now();
    long t_alloc = 0, t_syrf = 0, t_matvec = 0, t_gram = 0;
    long t_potrf = 0, t_trsm = 0, t_svd = 0, t_post = 0, t_rest = 0;

    // --- alloc ---
    auto t0 = clk::now();
    util::upsize(ws.Omega,    ws.Omega_sz,    m * k);
    util::upsize(ws.Y,        ws.Y_sz,        m * k);
    util::upsize(ws.G,        ws.G_sz,        k * k);
    util::upsize(ws.G_backup, ws.G_backup_sz, k * k);
    util::upsize(ws.D,        ws.D_sz,        k);
    util::upsize(ws.Sigma,    ws.Sigma_sz,    k);
    util::upsize(ws.VT_B,     ws.VT_B_sz,     k * k);
    util::upsize(ws.tau,      ws.tau_sz,      k);
    util::upsize(U_out,       U_out_sz,       m * k);
    util::upsize(lambda_out,  lambda_out_sz,  k);
    t_alloc = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();

    // --- syrf: Ω ← qr(Ω₁_in, 0) followed by q−1 subspace-iter passes ---
    auto t1 = clk::now();
    std::copy(Omega1_in, Omega1_in + m * k, ws.Omega);
    lapack::geqrf(m, k, ws.Omega, m, ws.tau);
    lapack::ungqr(m, k, k, ws.Omega, m, ws.tau);

    for (int64_t iter = 1; iter < q; ++iter) {
        A_op(Layout::ColMajor, k, (T)1, ws.Omega, m, (T)0, ws.Y, m);
        std::copy(ws.Y, ws.Y + m * k, ws.Omega);
        lapack::geqrf(m, k, ws.Omega, m, ws.tau);
        lapack::ungqr(m, k, k, ws.Omega, m, ws.tau);
    }
    t_syrf = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t1).count();

    // --- matvec: Y ← A · Ω ---
    auto t2 = clk::now();
    A_op(Layout::ColMajor, k, (T)1, ws.Omega, m, (T)0, ws.Y, m);
    t_matvec = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t2).count();

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

    auto t_specrec_start = clk::now();

    // --- gram: G = Ωᵀ · Y, symmetrize, backup for potrf-failure fall-back ---
    auto t3 = clk::now();
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
               (T)1, ws.Omega, m, ws.Y, m, (T)0, ws.G, k);
    RandLAPACK::util::symmetrize(k, ws.G, k);
    lapack::lacpy(lapack::MatrixType::General, k, k, ws.G, k,
                  ws.G_backup, k);
    t_gram = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t3).count();

    // --- potrf: try Cholesky (skipped when force_fallback set) ---
    auto t4 = clk::now();
    int chol_status = force_fallback
        ? 1   // any non-zero so we take the fall-back branch
        : lapack::potrf(Uplo::Upper, k, ws.G, k);
    t_potrf = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t4).count();

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
        RandLAPACK::util::get_U(k, k, ws.G, k);

        // --- trsm: Y := Y · R⁻¹ ---
        auto t5 = clk::now();
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper,
                   Op::NoTrans, Diag::NonUnit, m, k,
                   (T)1, ws.G, k, ws.Y, m);
        t_trsm = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t5).count();

        // --- svd: syrk + syevd ---
        auto t6 = clk::now();
        // G_y = Yᵀ · Y (k×k, sym PSD). Overwrite ws.G (Cholesky factor done).
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                   k, m, (T)1, ws.Y, m, (T)0, ws.G, k);
        lapack::syevd(lapack::Job::Vec, Uplo::Upper, k,
                      ws.G, k, ws.D);
        t_svd = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t6).count();

        // --- post_svd: descending sort + scale columns + lambda compute ---
        auto t7 = clk::now();
        for (int64_t ii = 0; ii < k / 2; ++ii) {
            std::swap(ws.D[ii], ws.D[k - 1 - ii]);
        }
        for (int64_t jj = 0; jj < k / 2; ++jj) {
            blas::swap(k, ws.G + jj * k, 1,
                          ws.G + (k - 1 - jj) * k, 1);
        }

        // U = Y · V (unscaled left singular vectors of B).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
                   (T)1, ws.Y, m, ws.G, k, (T)0, U_out, m);

        // Scale columns of U by 1/√σ². Threshold tiny σ² to 0 (column zeroed).
        T sig_sq_max    = (k > 0) ? std::max(ws.D[0], (T)0) : (T)0;
        T sig_sq_thresh = (T)4 * eps_mach * sig_sq_max;
        for (int64_t j = 0; j < k; ++j) {
            if (ws.D[j] > sig_sq_thresh) {
                T inv_sig = (T)1 / std::sqrt(ws.D[j]);
                blas::scal(m, inv_sig, U_out + j * m, 1);
            } else {
                std::fill(U_out + j * m, U_out + (j + 1) * m, (T)0);
            }
        }

        // λ = σ² (clamp negatives to 0).
        for (int64_t i = 0; i < k; ++i) {
            lambda_out[i] = std::max(ws.D[i], (T)0);
        }
        t_post = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t7).count();
    } else {
        // ---------- FALL-BACK (eig-pseudoinverse) ----------

        // Restore Gram from backup; potrf clobbered ws.G.
        lapack::lacpy(lapack::MatrixType::General, k, k,
                      ws.G_backup, k, ws.G, k);

        // --- svd: syevd on k×k Gram + gesdd on m×k B ---
        auto t6 = clk::now();
        // syevd on the symmetric (and intended-PSD) k×k Gram. Standard
        // LAPACK call for symmetric eigendecomp; preferred over gesdd
        // on a symmetric input because the algorithm exploits structure
        // (~2× faster, tighter ε guarantees). After the call:
        //   ws.G  holds eigenvectors V_G as columns
        //   ws.D  holds eigenvalues (ASCENDING order)
        // Aligns with the production driver in PR #132 (rl_nystrom_evd.hh).
        lapack::syevd(lapack::Job::Vec, Uplo::Upper, k, ws.G, k, ws.D);

        // Pinv threshold: drop eigenvalues below 2·ε·D_max where
        // D_max = D[k-1] (largest, ascending). Matches the production
        // driver's constant.
        const T D_max  = (k > 0) ? std::max(ws.D[k - 1], (T)0) : (T)0;
        const T thresh = (T)2 * eps_mach * D_max;

        // Form B = Y · V_G · diag(1/√D) · V_Gᵀ. (3 stages)
        //   1) U_out = Y · V_G                                (m × k)
        //   2) scale columns of U_out by 1/√D[j] (or 0)
        //   3) ws.Y = U_out · V_Gᵀ                            (m × k)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k,
                   (T)1, ws.Y, m, ws.G, k, (T)0, U_out, m);
        for (int64_t j = 0; j < k; ++j) {
            T scale = (ws.D[j] > thresh) ? (T)1 / std::sqrt(ws.D[j]) : (T)0;
            blas::scal(m, scale, U_out + j * m, 1);
        }
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, k, k,
                   (T)1, U_out, m, ws.G, k, (T)0, ws.Y, m);

        // [U, Σ, _] ← svd(B).
        lapack::gesdd(lapack::Job::SomeVec, m, k, ws.Y, m,
                      ws.Sigma, U_out, m, ws.VT_B, k);
        t_svd = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t6).count();

        // --- post_svd: λ ← Σ²; zero columns of U_out for tiny σ ---
        auto t7 = clk::now();
        const T Sig_max    = (k > 0) ? ws.Sigma[0] : (T)0;
        const T Sig_thresh = (T)2 * eps_mach * Sig_max;
        int64_t r = 0;
        for (int64_t i = 0; i < k; ++i) {
            lambda_out[i] = ws.Sigma[i] * ws.Sigma[i];
            if (ws.Sigma[i] > Sig_thresh) ++r;
        }
        std::fill(U_out + m * r, U_out + m * k, (T)0);
        t_post = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t7).count();
    }

    auto t_specrec_end = clk::now();
    if (t_specrec_ms_out) {
        *t_specrec_ms_out = std::chrono::duration<double, std::milli>(
            t_specrec_end - t_specrec_start).count();
    }

    long t_total = std::chrono::duration_cast<std::chrono::microseconds>(
                       clk::now() - t_total_start).count();
    t_rest = t_total - (t_alloc + t_syrf + t_matvec + t_gram + t_potrf
                        + t_trsm + t_svd + t_post);

    if (ws.times_enabled) {
        ws.times[0]  = t_alloc;
        ws.times[1]  = t_syrf;
        ws.times[2]  = t_matvec;
        ws.times[3]  = t_gram;
        ws.times[4]  = t_potrf;
        ws.times[5]  = t_trsm;
        ws.times[6]  = t_svd;
        ws.times[7]  = t_post;
        ws.times[8]  = 0L;       // error_est: unused in fixed-k variant
        ws.times[9]  = t_rest;
        ws.times[10] = t_total;
    }
}


} // namespace RandLAPACK
