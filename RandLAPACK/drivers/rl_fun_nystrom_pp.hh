#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_nystrom_evd.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace RandLAPACK {

/// FunNyström++ v2 — reference-aligned trace estimator.
///
/// From-scratch C++ port of the Persson-Kressner MATLAB reference
/// (`davpersson/funNystrom/Other/{nystrom,funnystrompp}.m`), intended as
/// a bit-exact baseline against MATLAB at fixed RNG. Deliberately does
/// not share infrastructure with the production funnystrompp PR — no
/// SYPS, no SYRF, no internal_stab plumbing — so that each algorithmic
/// knob can be added back one at a time with verification.
///
/// Phase 1 (`NystromEVD` in `rl_nystrom_evd.hh`): SparseStack/SASO
/// sketch generated inside the kernel from the caller's RNG state
/// (vec_nnz nonzeros per column; matches the writeup's Alg. 1 line 1)
/// + q − 1 subspace-iteration passes (with QR stabilization between
/// passes) + shifted Nyström spectral recovery on the k × k Gram
/// (see that file's [Alg. 2, line N] tags).
///
/// Phase 2 (this class's `call`): Hutchinson on the residual
/// f(A) − f(Â) using a caller-supplied fAfun oracle. Mirrors
/// `funnystrompp.m` line-for-line. Uses Gaussian Ω₂ to match the
/// reference. A Rademacher option will be added in Phase 6 of the
/// project plan.
///
/// The driver is numbered to match Algorithm 1 of the funNyström++
/// writeup (*Accelerating trace estimation*). Code blocks in `call()`
/// are tagged [Alg. 1, line N] with the step they implement:
///   (1)  draw a SparseStack Ω ∈ R^{m×k}      generated inside NystromEVD from
///                                            the caller's `state` (`vec_nnz`
///                                            nonzeros per column)
///   (2)  Â = U·Λ·Uᵀ ← Nyström(A^q·Ω)         NystromEVD. NB the writeup's q
///                                            counts extra A-passes on the
///                                            sketch (it advocates q = 0);
///                                            ours counts total A-applications
///                                            in Phase 1, so writeup-q = our q − 1
///   (3)  tr_top ← Σᵢ₌₁ᵏ f(Λᵢᵢ)
///   (4)  sample g₁,…,g_ℓ ~ N(0, I)           caller-supplied Omega2 (ℓ ≡ s)
///   (5)  tr_bot ← (1/ℓ)·Σᵢ Lanczos-QFA(A, f, gᵢ, t)
///                                            realized as tr(Ω₂ᵀ·fAfun(Ω₂))/s:
///                                            gᵢᵀ·LanczosFA(A, f, gᵢ) equals
///                                            Lanczos-QFA(A, f, gᵢ) by the
///                                            Gauss-quadrature identity
///   (6)  tr_cor ← (1/ℓ)·Σᵢ gᵢᵀ·U·f(Λ)·Uᵀ·gᵢ
///   (7)  return tr̃_f++ = tr_top + tr_bot − tr_cor
///                                            computed as t1 + t2 with
///                                            t1 = tr_top, t2 = tr_bot − tr_cor
/// The optional f_zero "zero-fill" correction (documented on `call`) is an
/// extension beyond Algorithm 1, which implicitly takes f(0) = 0.
///
/// The Phase-1 sketch is kernel-internal (SASO from `state`); only Ω₂
/// (Phase 2 Hutchinson probes) is taken externally. NB this drops the
/// original bit-exact Ω₁ cross-validation contract with the MATLAB
/// reference (which uses its own sketching anyway); Phase-2 fixtures
/// via Ω₂ and tolerance-level accuracy checks remain.
///
/// Reference: the Persson-Kressner funNyström++ algorithm and its MATLAB
/// reference implementation (davpersson/funNystrom), cited above.
template <typename T>
class FunNystromPP {
public:
    bool verbose = false;
    // Vestige of the removed dual-path recovery: NystromEVD now has a single
    // (shifted) path and no longer takes this flag. Kept only so the benchmark
    // CLI / CSV schema stays stable mid-campaign; remove with the next
    // benchmark schema change.
    bool force_fallback = false;

    // Nonzeros per column of the Phase-1 SASO sketch (forwarded to
    // NystromEVD's internal sketch generation). 8 is the project-wide
    // SASO default (matches the benchmark CLI default).
    int64_t vec_nnz = 8;

    // Phase-2 oracle convention. false = Lanczos-FA (fAfun fills f(A)·Ω₂, m×s,
    // and the driver dots it with Ω₂). true = Lanczos-QFA (fAfun fills the s×s
    // quadratic form Ω₂ᵀ f(A) Ω₂ directly, via BlockLanczosQFA, and the driver
    // takes its trace); QFA skips the f(A)·Ω₂ mapback entirely.
    bool use_qfa = false;

    // After call(), these hold Phase 1's eigenpairs of Â. Exposed as
    // public so tests can inspect them; in production code you'd treat
    // them as read-only output. Heap-owned; grown via util::upsize and
    // freed in the destructor.
    T* U      = nullptr; int64_t U_sz      = 0;   // m × k_out, column-major
    T* lambda = nullptr; int64_t lambda_sz = 0;   // k_out descending eigenvalues
    int64_t k_out = 0;

    // Persistent Phase 1 workspace; grown via util::upsize on each call.
    NystromEVD_workspace<T> nystrom_ws;

    // Persistent Phase 2 buffers; grown via util::upsize on each call.
    T* Y_2     = nullptr; int64_t Y_2_sz     = 0;   // k × s
    T* fAOmega = nullptr; int64_t fAOmega_sz = 0;   // m × s (FA) or s × s (QFA)
    // Internal Ω₂ (used only when the caller passes Omega2 == nullptr): a
    // Gaussian n × s block with each column normalized to ‖·‖₂ = √n.
    T* Omega2_buf = nullptr; int64_t Omega2_buf_sz = 0;

    // Phase-split wall-clock timings populated by call() (ms).
    double t_phase1_ms = 0.0;
    double t_phase2_ms = 0.0;
    // Phase-2 sub-split: wall-clock spent inside the caller-supplied fAfun
    // oracle (Lanczos-FA / exact apply). t_phase2_ms − t_fafun_ms is the
    // driver-side trace assembly (Y₂ GEMM + weighted Frobenius sums).
    double t_fafun_ms  = 0.0;
    // Benchmarking aid: inner wall-clock of just the shifted spectral-
    // recovery block inside NystromEVD (Alg. 2 lines 3-8); the QR +
    // subspace-iter + final matvec costs that precede it contribute to
    // t_phase1_ms instead.
    double t_specrec_ms = 0.0;

    FunNystromPP() = default;
    FunNystromPP(const FunNystromPP&) = delete;
    FunNystromPP& operator=(const FunNystromPP&) = delete;

    ~FunNystromPP() {
        delete[] U;
        delete[] lambda;
        delete[] Y_2;
        delete[] fAOmega;
        delete[] Omega2_buf;
    }

    /// Returns the estimate t = t1 + t2 of tr(f(A)).
    ///
    /// fAfun(int64_t s, const T* B, T* Y) is a callable that computes
    /// Y := f(A) * B for column-major B, Y of shape m × s with lda = m.
    /// In Phase 1 tests we pass a dense exact-f(A) oracle; in Phase 4
    /// we will swap it for block_lanczos_fa.
    ///
    /// fscalar and fAfun are the SAME function f in two representations,
    /// consumed in different places:
    ///   - fscalar (T -> T) acts on scalars. Used wherever eigenvalues are
    ///     already in hand: t1 = Σᵢ fscalar(λ̂ᵢ) [Alg. 1, line 3] and the
    ///     correction Σᵢⱼ fscalar(λ̂ᵢ)·Y₂[i,j]² [Alg. 1, line 6]. Â is
    ///     diagonalized by Phase 1, so f(Â) is never formed as a matrix;
    ///     f on k scalars is exact and essentially free.
    ///   - fAfun acts on the operator: B ↦ f(A)·B [Alg. 1, line 5], the one
    ///     place f must be applied to A itself, whose eigendecomposition we
    ///     do not have (Lanczos-FA in production; exact dense oracle in
    ///     tests). Expensive and approximate.
    /// CALLER OBLIGATION: the two must realize the same f. If they disagree,
    /// t2 silently estimates tr(f₁(A)) − tr(f₂(Â)) — nonsense with no error
    /// raised. In practice Lanczos-FA also evaluates f only on scalars (the
    /// Ritz values of its tridiagonal), so pass the same lambda once directly
    /// as fscalar and once captured inside the fAfun wrapper.
    ///
    /// @param[in]  A_op     Symmetric linop providing A * X.
    /// @param[in]  fAfun    Callable B ↦ f(A) * B.
    /// @param[in]  fscalar  Scalar f operating on each eigenvalue; must
    ///                      realize the same f as fAfun (see the contract
    ///                      note above).
    /// @param[in]  k        Phase 1 Nyström rank.
    /// @param[in]  s        Phase 2 Hutchinson sample count.
    /// @param[in]  q        Phase 1 number of A applications (q = 1 single
    ///                      pass; q = 2 = 1 subspace-iter pass; etc.).
    /// @param[in,out] state RNG state for the Phase-1 SASO sketch (generated
    ///                      inside NystromEVD with `this->vec_nnz` nonzeros
    ///                      per column); advanced past the draw.
    /// @param[in]  Omega2   Phase-2 Hutchinson probes (m × s, col-major).
    ///                      Pass nullptr to generate them inside the kernel: a
    ///                      Gaussian block drawn from `state`, each column
    ///                      normalized to ‖·‖₂ = √n. Pass an explicit block to
    ///                      override (e.g. test fixtures). When k == m Phase 2
    ///                      is skipped and Omega2 is unread.
    /// @param[in]  f_zero   Optional f(0). Default std::nullopt produces
    ///                      Persson-MATLAB-aligned output (no zero-fill
    ///                      correction; bit-exact cross-validation anchor).
    ///                      Pass a finite f(0) to opt in to PR-#132-style
    ///                      "zero-fill" semantics: tr(f(Â)) is computed
    ///                      assuming the rank-k Â is treated as an n×n
    ///                      operator with f(0) on the orthogonal
    ///                      complement; t1 gains (n − k) f(0) and t2 is
    ///                      adjusted by the projector-complement term.
    ///                      Must be finite; throws std::invalid_argument
    ///                      otherwise. No auto-resolve from fscalar(0):
    ///                      for f = log that would produce -∞.
    /// @param[out] t1_out   Phase 1 contribution Σ f(λ̂ᵢ), with the
    ///                      optional (n − k) f(0) term added when
    ///                      f_zero is supplied.
    /// @param[out] t2_out   Phase 2 stochastic correction; 0 when k == m.
    /// @return     t = t1 + t2.
    template <linops::SymmetricLinearOperator SLO, typename RNG,
              typename FAFun, typename FScalar>
    T call(
        SLO &A_op,
        FAFun &&fAfun,
        FScalar &&fscalar,
        int64_t k,
        int64_t s,
        int64_t q,
        RandBLAS::RNGState<RNG> &state,
        const T *Omega2,
        T &t1_out,
        T &t2_out,
        std::optional<T> f_zero = std::nullopt
    );
};



// --- Phase 2: FunNystromPP::call ---------------------------------------

template <typename T>
template <linops::SymmetricLinearOperator SLO, typename RNG,
          typename FAFun, typename FScalar>
T FunNystromPP<T>::call(
    SLO &A_op,
    FAFun &&fAfun,
    FScalar &&fscalar,
    int64_t k,
    int64_t s,
    int64_t q,
    RandBLAS::RNGState<RNG> &state,
    const T *Omega2,
    T &t1_out,
    T &t2_out,
    std::optional<T> f_zero
) {
    int64_t m = A_op.dim;

    if (f_zero.has_value() && !std::isfinite(*f_zero))
        throw std::invalid_argument(
            "FunNystromPP::call: f_zero must be finite when provided");

    // ---- Phase 1 (Algorithm 1, lines 1-3) ----
    //
    // [Alg. 1, line 1] Ω (SparseStack/SASO, vec_nnz nonzeros per column) is
    //   drawn inside NystromEVD from `state`.
    // [Alg. 1, line 2] Â = U·Λ·Uᵀ ← Nyström(A^{q−1}·Ω)  (shifted recovery;
    //   see the [Alg. 2, line N] tags inside NystromEVD).
    auto t_p1_start = std::chrono::steady_clock::now();
    NystromEVD<T>(A_op, k, q, this->vec_nnz, state,
                  this->U, this->U_sz,
                  this->lambda, this->lambda_sz,
                  this->nystrom_ws,
                  &this->t_specrec_ms);
    this->k_out = k;

    // f(0) for the optional zero-fill correction term in tr(f(Â)). Full
    // semantics (nullopt = Persson-MATLAB anchor; finite = PR-#132 zero-fill;
    // no auto-resolve from fscalar(0), which would give -∞ for f = log) are
    // documented on the f_zero @param above. The two conventions differ, for
    // a fixed Ω, by  f(0)·[(m−k) − (‖Ω‖²_F − ‖VᵀΩ‖²_F)/s],  which vanishes in
    // expectation for iid Gaussian / Rademacher Ω.
    const bool   apply_fzero = f_zero.has_value() && (k < m);
    const T      fz          = apply_fzero ? *f_zero : (T)0;

    // [Alg. 1, line 3] tr_top ← Σᵢ₌₁ᵏ f(Λᵢᵢ)  (= t1; plus the beyond-Alg.-1
    //   zero-fill term (m − k)·f(0) when f_zero is supplied).
    t1_out = (T)0;
    for (int64_t i = 0; i < k; ++i) t1_out += fscalar(this->lambda[i]);
    if (apply_fzero) t1_out += static_cast<T>(m - k) * fz;
    auto t_p1_end = std::chrono::steady_clock::now();
    this->t_phase1_ms = std::chrono::duration<double, std::milli>(t_p1_end - t_p1_start).count();

    // ---- Phase 2 (Algorithm 1, lines 4-7) ----
    //
    // t2 = ( tr(Ω₂ᵀ · fAfun(Ω₂)) − tr(Yᵀ · diag(f(λ)) · Y) ) / s
    //   where Y = Uᵀ · Ω₂ (shape k × s); i.e. t2 = tr_bot − tr_cor with the
    //   two 1/ℓ probe averages of lines 5-6 computed jointly (ℓ ≡ s).
    //
    // [Alg. 1, line 4] g₁,…,g_ℓ ~ N(0, I) are caller-supplied as the columns
    //   of Omega2.
    //
    // Skip Phase 2 when k == m: Phase 1 has captured the full spectrum
    // exactly (Â = A), so f(A) − f(Â) is analytically zero. Running the
    // Hutchinson correction anyway would leave an O(ε · s · LFA_residual)
    // misleading noise floor because Z1 (LFA approximation) and Z2
    // (V · diag(f(λ)) · Vᵀ · Ω) follow different floating-point paths.
    // The matvec-budget constraint k + s ≤ m implies s == 0 at k = m, so
    // this is also the only sensible call when the caller respects it.
    t2_out = (T)0;
    if (k < m) {
        auto t_p2_start = std::chrono::steady_clock::now();

        // [Alg. 1, line 4] Resolve Ω₂. Caller-supplied when Omega2 != nullptr;
        //   otherwise generated inside the kernel: a Gaussian n×s block drawn
        //   from `state`, each column normalized to ‖·‖₂ = √n (uniformly-random
        //   direction, fixed norm √n).
        const T* Om2 = Omega2;
        if (Om2 == nullptr) {
            util::upsize(this->Omega2_buf, this->Omega2_buf_sz, m * s);
            RandBLAS::DenseDist D2(m, s);                 // Gaussian (default family)
            state = RandBLAS::fill_dense(D2, this->Omega2_buf, state);
            const T target = std::sqrt((T)m);
            for (int64_t j = 0; j < s; ++j) {
                T* col = this->Omega2_buf + j * m;
                T nrm  = blas::nrm2(m, col, 1);
                if (nrm > (T)0) blas::scal(m, target / nrm, col, 1);
            }
            Om2 = this->Omega2_buf;
        }

        // [Alg. 1, line 6 — prep] Y_2 ← Uᵀ · Ω₂  (k × s), so that
        //   gᵢᵀ·U·f(Λ)·Uᵀ·gᵢ = Σⱼ f(λⱼ)·Y_2[j,i]².
        util::upsize(this->Y_2, this->Y_2_sz, k * s);
        blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   k, s, m, (T)1, this->U, m, Om2, m, (T)0, this->Y_2, k);

        // [Alg. 1, line 5 — oracle] tr_AΩ ← tr(Ω₂ᵀ·f(A)·Ω₂). Two conventions,
        //   selected by use_qfa (see the member doc):
        //     FA  : fAfun fills fAOmega = f(A)·Ω₂ (m×s); trace = Σⱼ⟨Ω₂ⱼ, fAOmegaⱼ⟩.
        //     QFA : fAfun fills the s×s matrix M = Ω₂ᵀf(A)Ω₂ directly, skipping
        //           the mapback (gᵢᵀ·LanczosFA ≡ Lanczos-QFA); trace = Σᵢ M[i,i].
        T tr_AOmega = (T)0;
        auto t_fafun_start = std::chrono::steady_clock::now();
        if (this->use_qfa) {
            util::upsize(this->fAOmega, this->fAOmega_sz, s * s);
            fAfun(m, s, Om2, this->fAOmega);
            for (int64_t i = 0; i < s; ++i) tr_AOmega += this->fAOmega[i + i * s];
        } else {
            util::upsize(this->fAOmega, this->fAOmega_sz, m * s);
            fAfun(m, s, Om2, this->fAOmega);
            for (int64_t j = 0; j < s; ++j)
                tr_AOmega += blas::dot(m, Om2 + j * m, 1, this->fAOmega + j * m, 1);
        }
        this->t_fafun_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_fafun_start).count();

        // [Alg. 1, line 6] ℓ·tr_cor = Σᵢ gᵢᵀ·U·f(Λ)·Uᵀ·gᵢ:
        //   tr_AhatΩ ← tr(Ω₂ᵀ · f(Â) · Ω₂) = Σᵢⱼ f(λᵢ)·Y_2[i,j]².
        // When apply_fzero, the rank-k Â is "zero-filled" to an n×n
        // operator: f(Â) = V·diag(f(λ))·Vᵀ + f(0)·(I − V·Vᵀ). The
        // projector-complement term contributes
        //   f(0) · [‖Ω₂‖²_F − ‖Y_2‖²_F]
        // to tr_AhatΩ, computed alongside the rank-k diagonal sum.
        T tr_AhatOmega = (T)0;
        for (int64_t j = 0; j < s; ++j) {
            for (int64_t i = 0; i < k; ++i) {
                T v = this->Y_2[i + j * k];
                tr_AhatOmega += fscalar(this->lambda[i]) * v * v;
            }
        }
        if (apply_fzero) {
            const T omega_fro_sq = blas::dot(m * s, Om2, 1, Om2, 1);
            const T y2_fro_sq    = blas::dot(k * s, this->Y_2, 1, this->Y_2, 1);
            tr_AhatOmega += fz * (omega_fro_sq - y2_fro_sq);
        }

        // [Alg. 1, lines 5-6 — probe average] t2 = tr_bot − tr_cor
        //   = (tr_AΩ − tr_AhatΩ)/ℓ  (single division; ℓ ≡ s).
        t2_out = (tr_AOmega - tr_AhatOmega) / (T)s;
        auto t_p2_end = std::chrono::steady_clock::now();
        this->t_phase2_ms = std::chrono::duration<double, std::milli>(t_p2_end - t_p2_start).count();
    } else {
        this->t_phase2_ms = 0.0;
    }
    // [Alg. 1, line 7] return tr̃_f++ = tr_top + tr_bot − tr_cor = t1 + t2.
    return t1_out + t2_out;
}


} // namespace RandLAPACK
