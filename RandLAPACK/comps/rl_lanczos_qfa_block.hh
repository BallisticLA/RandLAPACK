#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"
#include "rl_lanczos_fa_block.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace RandLAPACK {


/// Adaptive stopping rule for BlockLanczosQFA (used only when `adaptive`).
enum class BlockQFAStop : int {
    Window = 0,  ///< legacy heuristic: relative change of tr(M_k) over a delay window
    Radau  = 1   ///< certified: block Gauss vs Gauss-Radau bracket on tr(M_k)
};

/// Which value BlockLanczosQFA returns when the Radau certificate is active.
enum class BlockQFAReturn : int {
    Gauss    = 0,  ///< the plain block Gauss value (matches the scalar oracle's default)
    Midpoint = 1   ///< (M_U + M_L)/2 — for operator-monotone f the two quadratures err on
                   ///< opposite sides, so the midpoint halves the one-sided Gauss bias for free
};

/// d-step block Lanczos-QFA: the *quadrature form* Bᵀ f(A) B (s×s), NOT the
/// vector block f(A)B. By the Gauss-quadrature identity
///   gᵢᵀ · LanczosFA(A, f, gᵢ) = Lanczos-QFA(A, f, gᵢ),
/// the funNyström++ Phase-2 term tr(Ω₂ᵀ f(A) Ω₂) can be formed WITHOUT ever
/// materializing f(A)·Ω₂.
///
/// Cost vs BlockLanczosFA: identical recurrence (same d matvecs, same T_k) and
/// the same (d·s)-sized syevd of T_k, but the reconstruction replaces the
/// O(n·d·s²) mapback  out = Q_basis·f(T_k)·E₁·R₀  (BlockLanczosFA line 11d)
/// with an O(d·s³) small-matrix computation and NO n×s output. With B = Q₀·R₀
/// (the initial block QR) and the block basis orthonormal (Q₀ᵀ Q_p = δ_{0p} I),
///   Bᵀ f(A) B ≈ R₀ᵀ · [f(T_k)]_{1:s,1:s} · R₀,
/// the top-left s×s block of f(T_k) sandwiched by R₀.
///
/// Adaptive depth (optional, `adaptive = true`) with two stopping rules:
///
/// stop_rule = Radau (default) — CERTIFIED block Gauss/Gauss-Radau bracket.
///   At check depth t form the plain block Gauss value tr(M_U) from T_t, and
///   the Gauss-Radau value tr(M_L) from T̂_t: T_t with its bottom-right
///   diagonal block replaced by
///     Â_t = B_{t−1} · (T_{t−1}⁻¹ evaluated at its trailing s×s corner) · B_{t−1}ᵀ
///         = B_{t−1} · D_{t−1}⁻¹ · B_{t−1}ᵀ  =  A_t − D_t,
///   which pins s quadrature nodes at 0. Here D_i is the block-LDLᵀ pivot of
///   T_i, maintained by the O(s³)-per-step recurrence
///     D_1 = A_1,   D_i = A_i − B_{i−1} D_{i−1}⁻¹ B_{i−1}ᵀ,
///   the exact block analogue of the scalar pivot trick in LanczosQFA
///   (α̂_t = α_t − d_t): with T_{t−1} = L D Lᵀ and L unit-lower
///   block-bidiagonal, forward substitution on the trailing block-column of
///   the identity gives Eᵀ T_{t−1}⁻¹ E = D_{t−1}⁻¹ exactly — no (t·s)-sized
///   solve is ever formed. For the operator-monotone f this class targets,
///   Gauss and Radau-at-0 err on opposite sides (Golub–Meurant), so
///     |tr(M_U) − tr(M_L)| ≤ adaptive_rtol · |tr(M_L)|
///   is a certified relative-error stop (Raphael's block criterion; the scale
///   is guarded as max(|tr_U|, |tr_L|, tiny) against roundoff inversion at
///   the convergence floor). A non-PD pivot (potrf failure on D) means T is
///   not positive definite — indefinite A or breakdown — and disables the
///   certificate for the rest of the run: the recurrence runs to the cap and
///   the result is reported UNcertified rather than dividing by a bad pivot,
///   the same policy as the scalar class. The bracket is also evaluated once
///   AT the cap when the ladder skipped it, so a bracket that closes between
///   the last ladder check and the cap is still reported certified (the
///   scalar class historically missed this).
///
/// stop_rule = Window — legacy heuristic, kept selectable: stop at the first
///   check where |tr(M_k) − tr(M_{k−δ})| ≤ rtol · |tr(M_k)| over a delay
///   window δ. No certificate; certified stays false.
///
/// Check cadence: util::qfa_check_due (shared with the scalar oracle) — a
/// ~1.5× geometric ladder by default, fixed stride for check_every > 1. Each
/// check costs syevd work of size (t·s) and zero matvecs.
///
/// The recurrence is reused from a held BlockLanczosFA instance (single-sourced,
/// so the reorth switch and matvec accounting stay in one place); only T_blk and
/// the initial factor R₀ are read here. The Krylov basis K_big is built by the
/// recurrence but never read by the reconstruction, which is exactly the mapback
/// QFA avoids. (The recurrence still allocates and fills K_big; with reorth = 0
/// only three n×s blocks of it are mathematically needed — an accepted
/// memory-over-speed tradeoff of reusing the FA recurrence unchanged.)
///
/// @tparam T  Floating-point scalar type.
template <typename T>
class BlockLanczosQFA {
public:
    /// Reorthogonalization control (forwarded to the recurrence).
    ///  1 = full block reorthogonalization; 0 = none.
    int64_t reorth = 1;

    // ---- adaptive depth controls (used only when `adaptive`) --------------
    bool           adaptive      = false;                ///< choose depth online
    BlockQFAStop   stop_rule     = BlockQFAStop::Radau;  ///< certificate variant
    BlockQFAReturn return_mode   = BlockQFAReturn::Gauss;///< value returned on a certified stop
    T              adaptive_rtol = (T)1e-2;              ///< Radau: certified rel err; Window: rel-change tol
    int64_t        check_every   = 1;                    ///< 1 = geometric ladder; >1 = fixed stride

    // Window-rule knobs (stop_rule = Window only). First convergence test is
    // at depth (adaptive_min + adaptive_delay), so these set the floor on
    // d_used. delay=2 was chosen empirically (2026-07-10): across easy-to-
    // moderate spectra it matched delay=5's accuracy exactly while stopping
    // ~3 steps sooner. Named constants so callers that reuse an instance
    // across calls can RESTORE them rather than relying on fresh construction.
    static constexpr int64_t default_adaptive_delay = 2;
    static constexpr int64_t default_adaptive_min   = 2;
    int64_t adaptive_delay  = default_adaptive_delay;  ///< δ: compare depth k against k − δ checks
    int64_t adaptive_min    = default_adaptive_min;    ///< do not test convergence before this depth

    // ---- outputs of the last call -----------------------------------------
    int64_t d_used    = 0;      ///< block steps actually used
    int64_t matvecs   = 0;      ///< A column-applications spent: s per block step = s·d_used
    bool    certified = false;  ///< Radau bracket closed within adaptive_rtol
    T       tr_U      = (T)0;   ///< final block Gauss trace (tr of the Gauss M)
    T       tr_L      = (T)0;   ///< final block Radau trace (== tr_U when no certificate ran)

    /// Reused block recurrence + its buffers (K_big, R0_buf, T_blk, ...).
    BlockLanczosFA<T> fa;

    // Scratch (raw buffers, grown never shrunk — house workspace rule):
    //  workspace : compute_M's preserving path (eig_vals + W + T-copy).
    //  M_scratch : s×s Gauss-side M from certificate checks (never disturbs `out`).
    //  ML_scratch: s×s Radau-side M from certificate checks.
    //  D_buf     : s×s current block-LDLᵀ pivot D_t (lower triangle valid).
    //  Dchol_buf : s×s scratch — Cholesky factor of D_{t−1}, then reused.
    //  Bt_buf    : s×s scratch — B_{t−1}ᵀ, overwritten by L⁻¹B_{t−1}ᵀ (trsm).
    //  hist_buf  : Window rule's tr(M_k) history (one entry per check, ≤ d).
    T* workspace  = nullptr; int64_t workspace_sz  = 0;
    T* M_scratch  = nullptr; int64_t M_scratch_sz  = 0;
    T* ML_scratch = nullptr; int64_t ML_scratch_sz = 0;
    T* D_buf      = nullptr; int64_t D_buf_sz      = 0;
    T* Dchol_buf  = nullptr; int64_t Dchol_buf_sz  = 0;
    T* Bt_buf     = nullptr; int64_t Bt_buf_sz     = 0;
    T* hist_buf   = nullptr; int64_t hist_buf_sz   = 0;

    // Profiling, matching the LanczosFA / BlockLanczosFA surface: set `timing`
    // and read `times` after call(). Six slots, microseconds:
    //   {matvec, run_lanczos, apply, rest, total, reorth}
    // where `apply` is all compute_M work (certificate checks + the final M),
    // `run_lanczos` is the recurrence net of certificate time, and `reorth`
    // is the recurrence's block-MGS time (propagated from the held
    // BlockLanczosFA — this oracle DOES pay reorthogonalization cost whenever
    // reorth = 1, unlike the basis-free scalar LanczosQFA).
    bool timing = false;
    std::vector<long> times;
    long _t_matvec_us = 0;

    BlockLanczosQFA()                                  = default;
    BlockLanczosQFA(const BlockLanczosQFA&)            = delete;
    BlockLanczosQFA& operator=(const BlockLanczosQFA&) = delete;

    ~BlockLanczosQFA() {
        delete[] workspace; delete[] M_scratch; delete[] ML_scratch;
        delete[] D_buf; delete[] Dchol_buf; delete[] Bt_buf; delete[] hist_buf;
    }

    // ------------------------------------------------------------------
    /// Compute M = Bᵀ f(A) B (s×s, col-major, ld = s) into `out`. B is n×s
    /// col-major. Calls A up to d times (on n×s blocks); with `adaptive` it may
    /// stop earlier (see d_used). For the funNyström++ correction only tr(M) is
    /// needed, but the full s×s matrix is returned for generality (note the
    /// Radau certificate brackets the TRACE only; off-diagonal entries carry no
    /// certificate).
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        using namespace std::chrono;
        if (check_every <= 0)
            throw std::invalid_argument("BlockLanczosQFA::call: check_every must be >= 1");
        fa.reorth = this->reorth;
        fa.timing = this->timing;

        steady_clock::time_point t_start, t_lanczos_end, t_end;
        long cert_us = 0;   // compute_M time spent inside the certificate
        if (this->timing) t_start = steady_clock::now();

        this->certified = false;
        this->tr_U = (T)0;
        this->tr_L = (T)0;

        bool have_M  = false;  // certificate fired: M_scratch holds M_U at d_used
        bool cert_ok = true;   // block pivot chain still positive definite
        if (this->adaptive && this->stop_rule == BlockQFAStop::Radau) {
            util::upsize(this->M_scratch,  this->M_scratch_sz,  s * s);
            util::upsize(this->ML_scratch, this->ML_scratch_sz, s * s);
            util::upsize(this->D_buf,      this->D_buf_sz,      s * s);
            util::upsize(this->Dchol_buf,  this->Dchol_buf_sz,  s * s);
            util::upsize(this->Bt_buf,     this->Bt_buf_sz,     s * s);
            const int64_t dmax = d;
            auto stop_after = [&](int64_t kdepth) -> bool {
                // Maintain the pivot D_kdepth FIRST (every step, cheap O(s³));
                // only then decide whether this depth is a check depth.
                if (cert_ok) cert_ok = update_pivot(kdepth, s, dmax);
                if (!cert_ok || kdepth < 2 || kdepth >= dmax) return false;
                if (!util::qfa_check_due(kdepth, this->check_every)) return false;
                steady_clock::time_point c0, c1;
                if (this->timing) c0 = steady_clock::now();
                bool closed = radau_bracket_check(f, s, kdepth, dmax);
                if (this->timing) { c1 = steady_clock::now(); cert_us += duration_cast<microseconds>(c1 - c0).count(); }
                return closed;
            };
            fa.run_lanczos(A, B, n, s, d, stop_after);
            this->d_used = fa.steps_run;
            have_M = (this->d_used < d);   // early stop ⇒ scratches hold the pair at d_used
            if (have_M) this->certified = true;

            if (!have_M) {
                // Ran to the cap. If the pivot chain survived, evaluate the
                // bracket once AT the cap — the ladder rarely lands exactly on
                // d, and a bracket that closed in the gap must still be
                // reported certified. Order matters: the Radau side needs
                // T_blk intact (preserving copy), the Gauss side can then
                // consume T_blk in place.
                if (cert_ok && this->d_used >= 2) {
                    steady_clock::time_point c0, c1;
                    if (this->timing) c0 = steady_clock::now();
                    this->certified = radau_bracket_check(f, s, this->d_used, d);
                    if (this->timing) { c1 = steady_clock::now(); cert_us += duration_cast<microseconds>(c1 - c0).count(); }
                    have_M = true;   // scratches hold the at-cap pair
                }
            }
        } else if (this->adaptive) {   // stop_rule == Window (legacy heuristic)
            util::upsize(this->M_scratch, this->M_scratch_sz, s * s);
            util::upsize(this->hist_buf,  this->hist_buf_sz,  d);
            int64_t hist_n = 0;
            const int64_t dmax = d;
            auto stop_after = [&](int64_t kdepth) -> bool {
                if (kdepth < this->adaptive_min || kdepth >= dmax) return false;
                steady_clock::time_point c0, c1;
                if (this->timing) c0 = steady_clock::now();
                // preserve_T: the recurrence continues past this check.
                compute_M(f, s, kdepth, dmax, this->M_scratch, true);
                if (this->timing) { c1 = steady_clock::now(); cert_us += duration_cast<microseconds>(c1 - c0).count(); }
                T tr = (T)0;
                for (int64_t i = 0; i < s; ++i) tr += this->M_scratch[i + i * s];
                this->hist_buf[hist_n++] = tr;
                if (hist_n > this->adaptive_delay) {
                    T prev  = this->hist_buf[hist_n - 1 - this->adaptive_delay];
                    T scale = std::max(std::abs(tr), std::numeric_limits<T>::min());
                    if (std::abs(tr - prev) <= this->adaptive_rtol * scale) return true;
                }
                return false;
            };
            fa.run_lanczos(A, B, n, s, d, stop_after);
            this->d_used = fa.steps_run;
            have_M = (this->d_used < d);
        } else {
            fa.run_lanczos(A, B, n, s, d);
            this->d_used = d;
        }
        if (this->timing) t_lanczos_end = steady_clock::now();

        if (have_M && this->certified && this->return_mode == BlockQFAReturn::Midpoint) {
            // out = (M_U + M_L)/2, entrywise.
            for (int64_t e = 0; e < s * s; ++e)
                out[e] = (T)0.5 * (this->M_scratch[e] + this->ML_scratch[e]);
        } else if (have_M) {
            lapack::lacpy(lapack::MatrixType::General, s, s, this->M_scratch, s, out, s);
        } else {
            // No live scratch copy (fixed depth, Window ran to the cap, or the
            // pivot chain died): final M from the leading d_used block.
            // preserve_T = false, since the recurrence is over and the next
            // run_lanczos re-initializes T_blk; the syevd works in place on
            // fa.T_blk (no (d·s)² copy).
            compute_M(f, s, this->d_used, d, out, false);
        }
        // Final traces. When a certificate pair is live both were set inside
        // radau_bracket_check; otherwise report the Gauss trace on both sides.
        if (!(this->adaptive && this->stop_rule == BlockQFAStop::Radau && have_M)) {
            T tr = (T)0;
            const T* src = have_M ? this->M_scratch : out;
            for (int64_t i = 0; i < s; ++i) tr += src[i + i * s];
            this->tr_U = tr;
            this->tr_L = tr;
        }
        this->matvecs = s * this->d_used;

        this->_t_matvec_us = fa._t_matvec_us;
        if (this->timing) {
            t_end = steady_clock::now();
            long total_us   = duration_cast<microseconds>(t_end         - t_start).count();
            long span_us    = duration_cast<microseconds>(t_lanczos_end - t_start).count();
            long final_us   = duration_cast<microseconds>(t_end         - t_lanczos_end).count();
            long lanczos_us = span_us - cert_us;      // recurrence net of certificate
            long apply_us   = cert_us + final_us;     // all compute_M work
            long rest_us    = total_us - lanczos_us - apply_us;
            this->times = {this->_t_matvec_us, lanczos_us, apply_us, rest_us, total_us,
                           fa._t_reorth_us};
        }
    }

    // ------------------------------------------------------------------
    /// M = R₀ᵀ · [f(T_{1:kdepth})]_{1:s,1:s} · R₀ (s×s) into `out`, from the
    /// leading kdepth*s block of fa.T_blk (whose leading dimension is dmax*s).
    ///
    /// preserve_T = true:  copy that block into a compact m×m buffer first, so
    ///   fa.T_blk stays intact and the recurrence can continue after an adaptive
    ///   certificate check (costs an extra m×m workspace + lacpy).
    /// preserve_T = false: syevd directly in fa.T_blk (eigenvectors overwrite
    ///   the block tridiagonal, exactly like BlockLanczosFA::apply_f). Only
    ///   valid once the recurrence is finished.
    template <std::invocable<T> F>
    void compute_M(F f, int64_t s, int64_t kdepth, int64_t dmax, T* out, bool preserve_T) {
        const int64_t m      = kdepth * s;   // active tridiagonal dimension
        const int64_t src_ld = dmax   * s;   // leading dim of fa.T_blk
        const int64_t v_ld   = preserve_T ? m : src_ld;
        util::upsize(workspace, workspace_sz, m + s * m + (preserve_T ? m * m : 0));
        T* eig_vals = workspace;
        T* W        = eig_vals + m;   // s × m col-major (ld = s)
        T* V;                         // m × m (ld = v_ld): T block, then its eigenvectors
        if (preserve_T) {
            V = W + s * m;
            // General copy (not just the lower triangle): the strict upper of
            // the leading block of T_blk is zero from the recurrence's memset,
            // and copying it keeps V fully initialized — syevd only contracts
            // to read one triangle, and an uninitialized upper half trips
            // memory sanitizers for no saving.
            lapack::lacpy(lapack::MatrixType::General, m, m, fa.T_blk, src_ld, V, m);
        } else {
            V = fa.T_blk;
        }
        reduce_fT_to_M(f, s, m, V, v_ld, eig_vals, W, out);
    }

    /// Radau-side variant: same reduction, but from T̂ = T with the trailing
    /// diagonal block replaced by Â_t = A_t − D_t (D_last = the maintained
    /// pivot D_t, s×s, lower triangle valid). Always preserving — the Gauss
    /// side and/or the continuing recurrence still need T_blk.
    template <std::invocable<T> F>
    void compute_M_radau(F f, int64_t s, int64_t kdepth, int64_t dmax,
                         const T* D_last, T* out) {
        const int64_t m      = kdepth * s;
        const int64_t src_ld = dmax   * s;
        util::upsize(workspace, workspace_sz, m + s * m + m * m);
        T* eig_vals = workspace;
        T* W        = eig_vals + m;
        T* V        = W + s * m;
        lapack::lacpy(lapack::MatrixType::General, m, m, fa.T_blk, src_ld, V, m);
        // Corner block (rows/cols m-s .. m-1): Â_t = A_t − D_t. V's corner
        // holds A_t; subtract D_t's lower triangle and mirror, so V stays
        // fully symmetric (its upper triangle was copied from A_t too).
        const int64_t c0 = m - s;
        for (int64_t j = 0; j < s; ++j) {
            for (int64_t i = j; i < s; ++i) {
                const T v = V[(c0 + i) + (c0 + j) * m] - D_last[i + j * s];
                V[(c0 + i) + (c0 + j) * m] = v;
                V[(c0 + j) + (c0 + i) * m] = v;
            }
        }
        reduce_fT_to_M(f, s, m, V, m, eig_vals, W, out);
    }

private:
    // ------------------------------------------------------------------
    /// Shared tail of the two compute_M variants: given the m×m symmetric V
    /// (ld = v_ld; destroyed — eigenvectors overwrite it), form
    /// out = R₀ᵀ · [f(V)]_{1:s,1:s} · R₀ (s×s).
    template <std::invocable<T> F>
    void reduce_fT_to_M(F f, int64_t s, int64_t m, T* V, int64_t v_ld,
                        T* eig_vals, T* W, T* out) {
        lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, m, V, v_ld, eig_vals);
        // W[i,j] = f(λⱼ)·V[i,j] for i = 0..s-1 (first s rows of V, col-scaled).
        // Ritz values are clamped to ≥ 0 before f: A ⪰ 0 by assumption, and
        // the Radau construction pins nodes at 0 ± roundoff, so f must be
        // finite at 0 (use the shifted log(x+1), never a raw log).
        for (int64_t j = 0; j < m; ++j) {
            T fev = f(std::max(eig_vals[j], (T)0));
            const T* V_col = V + j * v_ld;
            T*       W_col = W + j * s;
            for (int64_t i = 0; i < s; ++i)
                W_col[i] = fev * V_col[i];
        }
        // [f(T)]_{1:s,1:s} = P·Wᵀ, P = first s rows of V (s×m, lda = v_ld).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                   s, s, m, (T)1.0, V, v_ld, W, s, (T)0.0, out, s);
        // M = R₀ᵀ · out · R₀  (R₀ upper triangular s×s in fa.R0_buf, ld = s).
        blas::trmm(Layout::ColMajor, Side::Left,  Uplo::Upper, Op::Trans,   Diag::NonUnit,
                   s, s, (T)1.0, fa.R0_buf, s, out, s);
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   s, s, (T)1.0, fa.R0_buf, s, out, s);
    }

    /// Advance the block-LDLᵀ pivot to depth `kdepth`:
    ///   kdepth == 1: D ← A_1;   kdepth ≥ 2: D ← A_t − B_{t−1} D⁻¹ B_{t−1}ᵀ.
    /// D_buf's LOWER triangle is the valid data throughout (all consumers —
    /// potrf here and the corner subtraction in compute_M_radau — read lower
    /// only). Returns false when D fails Cholesky (T not PD: indefinite A or
    /// breakdown), which permanently disables the certificate for this run.
    bool update_pivot(int64_t kdepth, int64_t s, int64_t dmax) {
        const int64_t m = dmax * s;
        // Block positions (math 1-based, code 0-based): A_t's tile starts at
        // (t−1)s; math B_{t−1} was written by loop step t−2 at tile
        // (row (t−1)s, col (t−2)s), upper triangular with zero strict lower.
        const T* A_t = fa.T_blk + ((kdepth - 1) * s) * m + ((kdepth - 1) * s);
        if (kdepth == 1) {
            lapack::lacpy(lapack::MatrixType::General, s, s, A_t, m, D_buf, s);
            return true;
        }
        const T* B_tm1 = fa.T_blk + ((kdepth - 2) * s) * m + ((kdepth - 1) * s);
        // Cholesky of D_{t−1} (lower).
        lapack::lacpy(lapack::MatrixType::General, s, s, D_buf, s, Dchol_buf, s);
        int64_t info = lapack::potrf(Uplo::Lower, s, Dchol_buf, s);
        if (info != 0) return false;
        // Bt_buf ← B_{t−1}ᵀ (general s×s; source tile's strict lower is zero).
        for (int64_t j = 0; j < s; ++j)
            for (int64_t i = 0; i < s; ++i)
                Bt_buf[i + j * s] = B_tm1[j + i * m];
        // Y = L⁻¹ B_{t−1}ᵀ, then correction B_{t−1} D⁻¹ B_{t−1}ᵀ = Yᵀ Y.
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit,
                   s, s, (T)1.0, Dchol_buf, s, Bt_buf, s);
        // D ← A_t, then D.lower −= (YᵀY).lower.
        lapack::lacpy(lapack::MatrixType::General, s, s, A_t, m, D_buf, s);
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans,
                   s, s, (T)-1.0, Bt_buf, s, (T)1.0, D_buf, s);
        return true;
    }

    /// Evaluate the Gauss/Radau pair at depth `kdepth` into M_scratch /
    /// ML_scratch, set tr_U / tr_L, and return whether the bracket closed
    /// within adaptive_rtol. Both computations preserve fa.T_blk.
    template <std::invocable<T> F>
    bool radau_bracket_check(F f, int64_t s, int64_t kdepth, int64_t dmax) {
        compute_M(f, s, kdepth, dmax, M_scratch, true);
        compute_M_radau(f, s, kdepth, dmax, D_buf, ML_scratch);
        T trU = (T)0, trL = (T)0;
        for (int64_t i = 0; i < s; ++i) {
            trU += M_scratch [i + i * s];
            trL += ML_scratch[i + i * s];
        }
        this->tr_U = trU;
        this->tr_L = trL;
        const T scale = std::max({std::abs(trU), std::abs(trL),
                                  std::numeric_limits<T>::min()});
        return std::abs(trU - trL) <= this->adaptive_rtol * scale;
    }
};


} // end namespace RandLAPACK
