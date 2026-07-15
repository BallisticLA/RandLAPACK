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


/// d-step block Lanczos-QFA: the *quadratic form* Bᵀ f(A) B (s×s), NOT the
/// vector block f(A)B. This is the draft's Algorithm 1 line 5 primitive
/// (Lanczos-QFA, the ♠ RAM note): by the Gauss-quadrature identity
///   gᵢᵀ · LanczosFA(A, f, gᵢ) = Lanczos-QFA(A, f, gᵢ),
/// so the funNyström++ Phase-2 term tr(Ω₂ᵀ f(A) Ω₂) can be formed WITHOUT ever
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
/// Adaptive depth (optional, `adaptive = true`): instead of a fixed d, run the
/// recurrence while monitoring the block quadrature estimate tr(M_k) as the
/// depth k grows (the paper's Alg. 2 "qfa" certificate, lifted to blocks). Stop
/// at the first k where the change over a delay window δ is below a relative
/// tolerance: |tr(M_k) − tr(M_{k−δ})| ≤ rtol · |tr(M_k)|. This defines the depth
/// online (no offline calibration) and stops the recurrence early, spending
/// fewer matvecs. The actual depth used is reported in `d_used`. Certificate
/// cost note: each check at depth k is a (k·s)-sized syevd, O((k·s)³) flops and
/// zero matvecs; the checks are what buy the early stop, and they dominate the
/// adaptive variant's non-matvec wall-clock.
///
/// The recurrence is reused from a held BlockLanczosFA instance (single-sourced,
/// so the reorth switch and matvec accounting stay in one place); only T_blk and
/// the initial factor R₀ are read here. The Krylov basis K_big is built by the
/// recurrence but never read by the reconstruction, which is exactly the mapback
/// QFA avoids.
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
    // First convergence test is at depth (adaptive_min + adaptive_delay), so these
    // set the floor on d_used. delay=2 was chosen empirically (2026-07-10): across
    // easy-to-moderate spectra it matched delay=5's accuracy exactly while stopping
    // ~3 steps sooner (fewer matvecs). Raise it if a stalling spectrum trips a
    // premature stop (tr(M_k) plateauing before it has truly converged).
    int64_t adaptive_delay  = 2;       ///< δ: compare depth k against k − δ checks
    int64_t adaptive_min    = 2;       ///< do not test convergence before this depth
    int64_t d_used          = 0;       ///< block steps actually used by the last call

    /// Reused block recurrence + its buffers (K_big, R0_buf, T_blk, ...).
    BlockLanczosFA<T> fa;

    // Scratch: compute_M's preserving path uses `workspace` (eig_vals + W +
    // T-copy); the adaptive certificate writes into a separate s×s M buffer so
    // it never disturbs the caller's output.
    T* workspace  = nullptr; int64_t workspace_sz  = 0;
    T* M_scratch  = nullptr; int64_t M_scratch_sz  = 0;

    // Profiling, matching the LanczosFA / BlockLanczosFA surface: set `timing`
    // and read `times` after call(). Slots: {matvec, run_lanczos, apply, rest,
    // total} in microseconds, where `apply` is all compute_M work (adaptive
    // certificate checks + the final M) and `run_lanczos` is the recurrence
    // excluding certificate time, so the slots stay comparable across oracles.
    bool timing = false;
    std::vector<long> times;
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
        using namespace std::chrono;
        fa.reorth = this->reorth;
        fa.timing = this->timing;

        steady_clock::time_point t_start, t_lanczos_end, t_end;
        long cert_us = 0;   // compute_M time spent inside the certificate
        if (this->timing) t_start = steady_clock::now();

        bool have_M = false;   // certificate fired: M_scratch holds M at d_used
        if (this->adaptive) {
            util::upsize(this->M_scratch, this->M_scratch_sz, s * s);
            std::vector<T> hist;                 // tr(M_k) at tested depths
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
            // The certificate only returns true right after computing M at the
            // depth it stopped at, so an early stop (d_used < d) means M_scratch
            // already holds the answer; recomputing it would repeat the largest
            // syevd of the run.
            have_M = (this->d_used < d);
        } else {
            fa.run_lanczos(A, B, n, s, d);
            this->d_used = d;
        }
        if (this->timing) t_lanczos_end = steady_clock::now();

        if (have_M) {
            lapack::lacpy(lapack::MatrixType::General, s, s, this->M_scratch, s, out, s);
        } else {
            // No live certificate copy (fixed depth, or adaptive ran to the cap):
            // final M from the leading d_used block. preserve_T = false, since the
            // recurrence is over and the next run_lanczos re-initializes T_blk;
            // the syevd works in place on fa.T_blk (no (d·s)² copy).
            compute_M(f, s, this->d_used, d, out, false);
        }

        this->_t_matvec_us = fa._t_matvec_us;
        if (this->timing) {
            t_end = steady_clock::now();
            long total_us   = duration_cast<microseconds>(t_end         - t_start).count();
            long span_us    = duration_cast<microseconds>(t_lanczos_end - t_start).count();
            long final_us   = duration_cast<microseconds>(t_end         - t_lanczos_end).count();
            long lanczos_us = span_us - cert_us;      // recurrence net of certificate
            long apply_us   = cert_us + final_us;     // all compute_M work
            long rest_us    = total_us - lanczos_us - apply_us;
            this->times = {this->_t_matvec_us, lanczos_us, apply_us, rest_us, total_us};
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
            // Copy the leading kdepth block (lower triangle; syevd reads
            // Uplo::Lower) so the source T_blk stays intact.
            lapack::lacpy(lapack::MatrixType::Lower, m, m, fa.T_blk, src_ld, V, m);
        } else {
            V = fa.T_blk;
        }
        lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, m, V, v_ld, eig_vals);

        // W[i,j] = f(λⱼ)·V[i,j] for i = 0..s-1 (first s rows of V, column-scaled).
        for (int64_t j = 0; j < m; ++j) {
            T fev = f(std::max(eig_vals[j], (T)0));
            const T* V_col = V + j * v_ld;
            T*       W_col = W + j * s;
            for (int64_t i = 0; i < s; ++i)
                W_col[i] = fev * V_col[i];
        }

        // [f(T_k)]_{1:s,1:s} = P·Wᵀ, P = first s rows of V (s×m, lda = v_ld).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                   s, s, m, (T)1.0, V, v_ld, W, s, (T)0.0, out, s);
        // M = R₀ᵀ · out · R₀  (R₀ upper triangular s×s in fa.R0_buf, ld = s).
        blas::trmm(Layout::ColMajor, Side::Left,  Uplo::Upper, Op::Trans,   Diag::NonUnit,
                   s, s, (T)1.0, fa.R0_buf, s, out, s);
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   s, s, (T)1.0, fa.R0_buf, s, out, s);
    }
};


} // end namespace RandLAPACK
