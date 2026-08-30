#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_nystrom_evd.hh"
#include "rl_lanczos_qfa.hh"       // scalar LanczosQFA: the driver's actual QFA oracle (auto_sqfa)

#include <RandBLAS.hh>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

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
/// (vec_nnz nonzeros per ROW of the m × k sketch, nnz = vec_nnz·m;
/// matches the writeup's Alg. 1 line 1) + q − 1 subspace-iteration
/// passes (with QR stabilization between passes) + shifted Nyström
/// spectral recovery on the k × k Gram (see that file's
/// [Alg. 2, line N] tags).
///
/// Phase 2 (this class's `call`): Hutchinson on the residual
/// f(A) − f(Â) using a caller-supplied fAfun oracle. Mirrors
/// `funnystrompp.m` line-for-line. The probes Ω₂ are generated
/// internally by default: Gaussian columns normalized to ‖·‖₂ = √m
/// (uniformly-random direction, fixed norm — NOT plain Gaussian); an
/// explicit Ω₂ block can be passed to override (test fixtures).
///
/// The driver is numbered to match Algorithm 1 of the funNyström++
/// writeup (*Accelerating trace estimation*). Code blocks in `call()`
/// are tagged [Alg. 1, line N] with the step they implement:
///   (1)  draw a SparseStack Ω ∈ R^{m×k}      generated inside NystromEVD from
///                                            the caller's `state` (`vec_nnz`
///                                            nonzeros per row)
///   (2)  Â = U·Λ·Uᵀ ← Nyström(A^q·Ω)         NystromEVD. NB the writeup's q
///                                            counts extra A-passes on the
///                                            sketch (it advocates q = 0);
///                                            ours counts total A-applications
///                                            in Phase 1, so writeup-q = our q − 1
///   (3)  tr_top ← Σᵢ₌₁ᵏ f(Λᵢᵢ)
///   (4)  sample g₁,…,g_ℓ ~ N(0, I)           kernel-internal Ω₂ by default
///                                            (sphere-normalized, ‖gᵢ‖ = √m),
///                                            caller-overridable (ℓ ≡ s)
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

    // Nonzeros per ROW of the m × k Phase-1 SASO sketch (forwarded to
    // NystromEVD's internal sketch generation; nnz = vec_nnz·m). 8 is the
    // project-wide SASO default (matches the benchmark CLI default). 0 means
    // "auto": NystromEVD resolves it to ~log(k) at sketch time, which keeps
    // the probability of an exactly-zero sketch column (a genuine potrf
    // failure at large k) negligible.
    int64_t vec_nnz = 8;

    // Phase-2 oracle convention. false = Lanczos-FA (fAfun fills f(A)·Ω₂, m×s,
    // and the driver dots it with Ω₂). true = Lanczos-QFA (fAfun fills the s×s
    // quadratic form Ω₂ᵀ f(A) Ω₂ — via BlockLanczosQFA, or diagonal-only via
    // the scalar LanczosQFA — and the driver takes its trace, which reads ONLY
    // the diagonal); QFA skips the f(A)·Ω₂ mapback entirely.
    bool use_qfa = false;

    // ---- knob-free tier ("auto"): state for the (matvec_budget, eps) overload ----
    // The auto overload owns its Phase-2 oracle (the expert overload takes fAfun
    // from the caller, but choosing the depth is the auto tier's whole job).
    // Scalar Lanczos-QFA with the Gauss-Radau certified stop: eps is a certified
    // per-probe relative error, columns stop at their own depths, and the
    // batched matvec shrinks as they do (see rl_lanczos_qfa.hh).
    LanczosQFA<T> auto_sqfa;
    // Tunables. auto_probe_block is the probe block size b: the depth probe
    // costs at most b*probe_cap matvecs, so small b is cheap. auto_probe_frac
    // bounds the probe's spend: the probe may spend at most this fraction of
    // the matvec budget (probe_cap <= auto_probe_frac * matvec_budget / b,
    // floored at depth 2 so tiny budgets still probe at all). Must lie in
    // (0, 1). auto_depth_cap optionally bounds the probe depth further: 0
    // (default) means no fixed cap, so the probe is bounded only by n and by
    // the probe-fraction cap. A fixed cap must be opt-in: the old default of
    // 200 silently floored the oracle bias whenever the certified depth
    // exceeded it (kappa >= 1e6 in the 2026-07 campaign), and no budget could
    // buy the accuracy back.
    int64_t auto_probe_block = 4;
    T       auto_probe_frac  = (T)0.125;
    int64_t auto_depth_cap   = 0;
    // VESTIGIAL: Phase-2 batching existed to dodge BlockLanczosQFA's one-block
    // (t*s)^2 dense eigenproblem. The scalar oracle has per-column tridiagonals
    // (no block eigenproblem), so batching buys nothing; the member is kept only
    // so existing call sites stay stable. Remove with the next MEX-side
    // schema change.
    int64_t auto_qfa_batch   = 100;
    // Outputs of the last auto call, for cost accounting: the chosen rank/probe
    // count/oracle depth cap, the matvecs the probe actually spent
    // (auto_probe_matvecs = sum of per-column depths <= b*probe_cap), the
    // matvecs Phase 2 actually spent (auto_oracle_matvecs = sum of per-probe
    // depths <= auto_s*auto_t), and whether every probe column certified. With
    // certified early stopping the budget closes as an UPPER BOUND:
    //   auto_probe_matvecs + q*auto_k + auto_oracle_matvecs <= matvec_budget
    // (q = 1); the allocation still charges the worst case s*t up front, so the
    // certified savings surface as underspend, not overspend.
    int64_t auto_k = 0, auto_s = 0, auto_t = 0, auto_probe_matvecs = 0;
    int64_t auto_oracle_matvecs = 0;
    bool    auto_probe_converged = true;
    // Phase-2 certification of the last auto call: whether every Phase-2
    // oracle column certified at its depth cap. Distinct from
    // auto_probe_converged, which names ONLY the depth probe (Phase 2 reuses
    // the same LanczosQFA instance and overwrites its all_certified flag, so
    // it must be captured after the oracle runs).
    bool    auto_phase2_certified = false;
    // Wall-clock of the depth probe (Gaussian fill + certified QFA run), ms.
    // The probe runs before the expert call's Phase-1 clock starts, so
    // without this its cost would be attributed to nothing.
    double  t_probe_ms = 0.0;
    // Probe scratch (normalized-Gaussian block + probe QFA output).
    T* auto_probe_buf = nullptr; int64_t auto_probe_buf_sz = 0;
    T* auto_M_buf     = nullptr; int64_t auto_M_buf_sz     = 0;
    // Probe-sample snapshots for Hutchinson reuse: per-column certified
    // quadratic forms gᵢᵀf(A)gᵢ, certification flags, and depths, copied out
    // of auto_sqfa before Phase 2 overwrites its arrays.
    T*       auto_probe_gauss = nullptr; int64_t auto_probe_gauss_sz = 0;
    uint8_t* auto_probe_cert  = nullptr; int64_t auto_probe_cert_sz  = 0;
    int64_t* auto_probe_depth = nullptr; int64_t auto_probe_depth_sz = 0;

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
        delete[] auto_probe_buf;
        delete[] auto_M_buf;
        delete[] auto_probe_gauss;
        delete[] auto_probe_cert;
        delete[] auto_probe_depth;
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
    /// @param[in]  k        Phase 1 Nyström rank. Precondition: 1 <= k <= m,
    ///                      and for a sparse SASO sketch keep k <~ m/2 — as k
    ///                      approaches m the probability of an exactly-zero
    ///                      sketch column grows, the Gram Ωᵀ(A+νI)Ω becomes
    ///                      exactly singular, and NystromEVD's potrf throws;
    ///                      raising vec_nnz (or vec_nnz = 0 auto) suppresses
    ///                      this (the knob-free auto tier caps k at m/2;
    ///                      k == m is exact and safe).
    /// @param[in]  s        Phase 2 Hutchinson sample count.
    /// @param[in]  q        Phase 1 number of A applications (q = 1 single
    ///                      pass; q = 2 = 1 subspace-iter pass; etc.).
    /// @param[in,out] state RNG state for the Phase-1 SASO sketch (generated
    ///                      inside NystromEVD with `this->vec_nnz` nonzeros
    ///                      per row; 0 = auto ~log(k)); advanced past the draw.
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

    /// Knob-free overload: same estimator, but the caller sets only a total
    /// A-matvec budget B and a target accuracy; the rank k, probe count s,
    /// and oracle depth t are chosen inside. Allocation:
    ///
    ///   1. Depth probe: run the adaptive scalar Lanczos-QFA certificate on a
    ///      small internal probe block (b = auto_probe_block sphere-normalized
    ///      Gaussian columns, rtol = eps) to find the depth an accurate probe
    ///      needs. The probe's spend is BOUNDED: its depth cap is
    ///      min(auto_depth_cap if > 0, n, max(2, auto_probe_frac*B/b)), so the
    ///      probe spends at most ~auto_probe_frac of the budget (default 1/8)
    ///      and can never eat the half of it the old B/(2b) cap allowed.
    ///   2. Depth t. When the probe CERTIFIES, t is the MEDIAN of the b
    ///      per-column certified depths (the max over columns biases depth up
    ///      and starves the probe count; the certificate tolerance already
    ///      carries headroom). When the probe does NOT certify, the probe cap
    ///      is not promoted into t blindly: t = min(probe-reached depth,
    ///      m_rem/(2*s_min), n) with m_rem = B - probe spend and a probe-count
    ///      floor s_min = 4, so the 50/50 split below always yields
    ///      s >= s_min — more budget always buys more probes AND more rank
    ///      (never the old s == 2 lock). Always t = max(1, min(t, n)).
    ///   3. Split: 50/50 in matvecs, s = floor(m_rem/(2t)),
    ///      k = (m_rem - s*t)/q. The rank is capped at n/2 (the k -> n
    ///      sparse-sketch corner risks an exactly-zero SASO column, and n/2
    ///      matches the benchmark's rank-heavy split); when the cap binds the
    ///      surplus goes into probes. s is finally clamped to n - k (the
    ///      estimator's k + s <= n convention); past that point the tier
    ///      saturates and the leftover budget goes unspent.
    ///   4. Delegate to the expert call() above with a QFA oracle (use_qfa
    ///      path) and kernel-internal Omega2. The oracle stays ADAPTIVE
    ///      (certificate rtol = eps), depth-capped at t: per-column early
    ///      stopping is the knob-free tier's efficiency feature.
    ///   5. Probe-sample reuse: the probe's CERTIFIED columns are valid
    ///      Hutchinson samples drawn with the same sphere-normalized Gaussian
    ///      convention as Omega2, so their quadratic forms gᵢᵀf(A)gᵢ (and the
    ///      matching gᵢᵀf(Â)gᵢ from the Phase-1 eigenpairs — zero extra
    ///      matvecs) join the Phase-2 average: the effective sample count is
    ///      s + (# certified probe columns) and t2 is re-averaged over it.
    ///      Uncertified probe columns carry no accuracy statement and are
    ///      never reused. The probe's matvecs were already charged to the
    ///      budget, so accounting is unchanged.
    ///
    /// Conventions this tier fixes (the expert overload keeps them free):
    /// the budget currency is the A-MATVEC COUNT (not wall-clock); q = 1
    /// (single-pass Nystrom, the paper's advocated setting); eps is a target
    /// scale for the oracle bias (certificate rtol = eps), not an end-to-end
    /// error guarantee (the stochastic probe error is budget-limited).
    /// Chosen values are reported in auto_k / auto_s / auto_t /
    /// auto_probe_matvecs / auto_oracle_matvecs; certification in
    /// auto_probe_converged (probe) and auto_phase2_certified (Phase-2
    /// oracle); the probe's wall-clock in t_probe_ms.
    ///
    /// Throws std::invalid_argument for an infeasible matvec_budget: the
    /// budget must cover the worst-case minimum probe (b columns at the
    /// depth-2 floor) plus a split funding s_min depth-1 probes and at least
    /// one unit of rank, i.e. B >= max(2b + 2*s_min,
    /// ceil(2*s_min/(1 - auto_probe_frac))); running a depth-1 oracle
    /// silently is not an acceptable fallback.
    template <linops::SymmetricLinearOperator SLO, typename RNG, typename FScalar>
    T call(
        SLO &A_op,
        FScalar &&fscalar,
        int64_t matvec_budget,
        T eps,
        RandBLAS::RNGState<RNG> &state,
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

    // Reset the per-call outputs up front so a throw out of NystromEVD cannot
    // leave a previous call's k_out / t_specrec_ms visible on the driver.
    this->k_out        = 0;
    this->t_specrec_ms = 0.0;

    // ---- Phase 1 (Algorithm 1, lines 1-3) ----
    //
    // [Alg. 1, line 1] Ω (SparseStack/SASO, vec_nnz nonzeros per row) is
    //   drawn inside NystromEVD from `state`.
    // [Alg. 1, line 2] Â = U·Λ·Uᵀ ← Nyström(A^{q−1}·Ω)  (shifted recovery;
    //   see the [Alg. 2, line N] tags inside NystromEVD).
    auto t_p1_start = std::chrono::steady_clock::now();
    // Clamp the sketch's nonzeros-per-row to the sketch width k. A SASO with
    // vec_nnz > k asks each row for more nonzeros than it has columns to place
    // them in, and RandBLAS rejects it outright: "(vec_nnz <= dim_major) was
    // required, but did not hold, in function SparseDist". Before this clamp,
    // ANY call with k < vec_nnz (default 8) threw rather than degrading --
    // which is not an exotic corner: the benchmark's smallest budgets give
    // k = B/2 = 5 at B = 10, and the knob-free `auto` tier picks its own k and
    // lands below 8 at small budgets too (47 of 221 rungs in the 2026-07-28
    // rehearsal died this way). vec_nnz = k makes each ROW dense, i.e. Ω fully
    // dense ±1, which is the correct degenerate limit of a SASO. vec_nnz = 0
    // passes through unchanged: NystromEVD resolves it to the ~log(k) auto
    // policy at sketch time.
    const int64_t vnz = (this->vec_nnz == 0)
        ? (int64_t)0
        : std::max((int64_t)1, std::min(this->vec_nnz, k));
    NystromEVD<T>(A_op, k, q, vnz, state,
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
    // [Alg. 1, line 4] the probes g₁,…,g_ℓ are the columns of Ω₂ —
    //   kernel-internal sphere-normalized Gaussians by default, or
    //   caller-supplied when Omega2 != nullptr (resolved below).
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
        // t_fafun_ms must be cleared here too, not just left over from a prior
        // call. Consumers compute assembly = t_phase2_ms - t_fafun_ms; with a
        // stale t_fafun_ms that goes NEGATIVE on the first k == m call after a
        // k < m one. Latent while every call constructed a fresh driver; a real
        // wrong-answer bug as soon as the driver is reused across calls.
        this->t_fafun_ms  = 0.0;
    }
    // [Alg. 1, line 7] return tr̃_f++ = tr_top + tr_bot − tr_cor = t1 + t2.
    return t1_out + t2_out;
}


// --- Knob-free tier: FunNystromPP::call(A, f, matvec_budget, eps, ...) --------

template <typename T>
template <linops::SymmetricLinearOperator SLO, typename RNG, typename FScalar>
T FunNystromPP<T>::call(
    SLO &A_op,
    FScalar &&fscalar,
    int64_t matvec_budget,
    T eps,
    RandBLAS::RNGState<RNG> &state,
    T &t1_out,
    T &t2_out,
    std::optional<T> f_zero
) {
    using clk = std::chrono::steady_clock;
    const int64_t n = A_op.dim;
    const int64_t q = 1;       // single-pass Nystrom (the paper's advocated setting)
    const int64_t s_min = 4;   // probe-count floor when the depth probe fails to certify

    if (!(eps > (T)0) || !(eps < (T)1))
        throw std::invalid_argument(
            "FunNystromPP::call(auto): eps must lie in (0, 1); got " + std::to_string((double)eps));
    if (!(this->auto_probe_frac > (T)0) || !(this->auto_probe_frac < (T)1))
        throw std::invalid_argument(
            "FunNystromPP::call(auto): auto_probe_frac must lie in (0, 1); got " +
            std::to_string((double)this->auto_probe_frac));
    const int64_t b = std::min(this->auto_probe_block, n);
    if (b < 1)
        throw std::invalid_argument(
            "FunNystromPP::call(auto): auto_probe_block must be >= 1; got " +
            std::to_string(this->auto_probe_block));

    // ---- 0. Minimum-budget guard ----
    // Worst case the probe spends 2b matvecs (the probe-depth cap below never
    // drops under 2) or auto_probe_frac * B, whichever binds; the 50/50 split
    // then needs m_rem >= 2*s_min so it funds s >= s_min probes at depth >= 1,
    // and the rank half m_rem - s*t >= ceil(m_rem/2) >= s_min >= q covers
    // k >= 1. Refuse anything smaller rather than silently running a depth-1
    // oracle.
    const int64_t min_budget = std::max<int64_t>(
        2 * b + 2 * s_min,
        (int64_t)std::ceil((double)(2 * s_min) / (1.0 - (double)this->auto_probe_frac)));
    if (matvec_budget < min_budget)
        throw std::invalid_argument(
            "FunNystromPP auto: matvec budget B = " + std::to_string(matvec_budget) +
            " is infeasible; minimum for this configuration is " +
            std::to_string(min_budget) + " (probe block b = " + std::to_string(b) +
            " at the depth-2 floor plus " + std::to_string(s_min) +
            " depth-1 Hutchinson probes and at least one unit of rank).");

    // ---- 1. Depth probe: adaptive QFA certificate on a small probe block ----
    // The probe may spend at most ~auto_probe_frac of the budget (default
    // 1/8): its depth cap is auto_probe_frac*B/b, floored at 2 so tiny
    // budgets still probe, and bounded by n and the opt-in auto_depth_cap.
    const int64_t user_cap  = (this->auto_depth_cap > 0) ? this->auto_depth_cap : n;
    const int64_t frac_cap  = std::max((int64_t)2,
        (int64_t)(this->auto_probe_frac * (T)matvec_budget / (T)b));
    const int64_t probe_cap = std::min({user_cap, n, frac_cap});

    this->t_probe_ms            = 0.0;
    this->auto_phase2_certified = false;
    auto t_probe_start = clk::now();

    // Probe block: Gaussian, columns normalized to sqrt(n) (the same probe
    // convention as the internal Omega2 — which is what makes the certified
    // probe columns reusable as Hutchinson samples in step 5 below).
    util::upsize(this->auto_probe_buf, this->auto_probe_buf_sz, n * b);
    util::upsize(this->auto_M_buf,     this->auto_M_buf_sz,     b);   // scalar QFA writes a length-b (per-column) output, not b*b
    RandBLAS::DenseDist Dp(n, b);
    state = RandBLAS::fill_dense(Dp, this->auto_probe_buf, state);
    const T colnorm_target = std::sqrt((T)n);
    for (int64_t j = 0; j < b; ++j) {
        T* col = this->auto_probe_buf + j * n;
        T nrm  = blas::nrm2(n, col, 1);
        if (nrm > (T)0) blas::scal(n, colnorm_target / nrm, col, 1);
    }

    this->auto_sqfa.adaptive      = true;
    this->auto_sqfa.adaptive_rtol = eps;
    this->auto_sqfa.call(A_op, this->auto_probe_buf, n, b, fscalar, probe_cap,
                         this->auto_M_buf);
    this->auto_probe_matvecs   = this->auto_sqfa.matvecs;   // actual Σ t_j <= b*probe_cap
    this->auto_probe_converged = this->auto_sqfa.all_certified;

    // Snapshot the probe's per-column outputs NOW: Phase 2 reuses the same
    // LanczosQFA instance, which overwrites gauss_val / certified / t_used.
    util::upsize(this->auto_probe_gauss, this->auto_probe_gauss_sz, b);
    util::upsize(this->auto_probe_cert,  this->auto_probe_cert_sz,  b);
    util::upsize(this->auto_probe_depth, this->auto_probe_depth_sz, b);
    std::copy(this->auto_sqfa.gauss_val, this->auto_sqfa.gauss_val + b, this->auto_probe_gauss);
    std::copy(this->auto_sqfa.certified, this->auto_sqfa.certified + b, this->auto_probe_cert);
    std::copy(this->auto_sqfa.t_used,    this->auto_sqfa.t_used    + b, this->auto_probe_depth);

    this->t_probe_ms = std::chrono::duration<double, std::milli>(
        clk::now() - t_probe_start).count();

    // ---- 2. Depth policy ----
    const int64_t m_rem = matvec_budget - this->auto_probe_matvecs;
    int64_t t;
    if (this->auto_probe_converged) {
        // Certified: t is the MEDIAN of the b per-column certified depths.
        // The max over columns biases the depth up — buying per-probe accuracy
        // the O(1/sqrt(s)) stochastic term cannot use, at the cost of halving
        // s — and the certificate tolerance already carries headroom. Upper
        // median for even b (rounds toward the deeper middle depth).
        std::sort(this->auto_probe_depth, this->auto_probe_depth + b);
        t = (b % 2 == 1)
            ? this->auto_probe_depth[b / 2]
            : (this->auto_probe_depth[b / 2 - 1] + this->auto_probe_depth[b / 2] + 1) / 2;
    } else {
        // Uncertified: never promote the probe cap into t blindly. Cap t so
        // the 50/50 split below funds at least s_min probes; then more budget
        // always buys more probes AND more rank (never an s == 2 lock at a
        // problem-independent fixed point).
        t = std::min({this->auto_sqfa.d_used, m_rem / (2 * s_min), n});
    }
    // t is always a valid divisor and never deeper than one probe can afford:
    // d_used (and the certified median) can be 0 for an all-zero probe block,
    // and m_rem/2 caps the degenerate configs the floors above don't cover.
    t = std::min(t, m_rem / 2);
    t = std::max((int64_t)1, std::min(t, n));
    this->auto_t = t;

    // ---- 3. Split: 50/50 in matvecs between the probe side s*t and the rank
    //         side q*k (probe + q*k + s*t <= B; certified early stopping in
    //         Phase 2 turns the s*t term into an upper bound, so the actual
    //         spend is reported, not assumed) ----
    int64_t s = m_rem / (2 * t);                 // floor: probes get ~half of m_rem
    int64_t k = (m_rem - s * t) / q;             // rank absorbs the rounding
    // Cap the rank at n/2 and put the surplus budget into probes. Two reasons:
    // (a) as k approaches n an exactly-zero column of the sparse SASO sketch
    // becomes increasingly likely, making the shifted-Nystrom Gram exactly
    // singular (potrf throws), so the k -> n corner is not reliably available;
    // (b) n/2 is the same rank cap the benchmark's rank-heavy split uses, so
    // the tiers agree.
    // After the cap the leftover r = m_rem - q*k - s*t satisfies 0 <= r < t
    // (not enough for one more probe); the budget is spent up to that remainder.
    const int64_t k_cap = std::max((int64_t)1, n / 2);
    if (k > k_cap) {
        k = k_cap;
        s = (m_rem - q * k) / t;
    }
    // s >= 1 in both branches: pre-cap, t <= m_rem/2 gives s = floor(m_rem/(2t))
    // >= 1; post-cap, the branch is taken only when m_rem - s*t > q*k_cap, so
    // the re-derived s = floor((m_rem - q*k_cap)/t) can only grow.
    assert(s >= 1);
    // Cap the probe count at n - k (the estimator's k + s <= n convention; the
    // block-QFA oracle also requires a block no wider than n). Past this point
    // the estimator SATURATES: a larger budget buys nothing more in this tier,
    // and the leftover goes unspent (visible as matvec_budget minus the
    // auto_* accounting). Batching the probes to spend arbitrarily large
    // budgets is possible but not implemented.
    s = std::min(s, n - k);
    this->auto_k = k;
    this->auto_s = s;

    // ---- 4. Delegate to the expert overload with a certified QFA oracle ----
    // The scalar oracle stays adaptive in Phase 2 (rtol = eps, depth cap t
    // from the depth policy): each probe column stops at its own certified
    // depth, spending auto_oracle_matvecs = Σ_j t_j <= s*t. One call over the
    // whole block; the per-column output vector lands on the diagonal of Y
    // (the only part the use_qfa trace read touches; off-diagonals are left
    // unwritten).
    this->auto_oracle_matvecs   = 0;
    this->auto_phase2_certified = true;   // vacuously true if Phase 2 is skipped (k == n)
    auto fAfun = [&](int64_t fa_n, int64_t fa_s, const T* Bblk, T* Y) {
        util::upsize(this->auto_M_buf, this->auto_M_buf_sz, fa_s);
        this->auto_sqfa.call(A_op, Bblk, fa_n, fa_s, fscalar, t, this->auto_M_buf);
        for (int64_t jj = 0; jj < fa_s; ++jj)
            Y[jj + jj * fa_s] = this->auto_M_buf[jj];
        this->auto_oracle_matvecs += this->auto_sqfa.matvecs;
        // Captured HERE, after the Phase-2 oracle ran: the probe's flag is
        // already saved in auto_probe_converged, and this call overwrites
        // auto_sqfa.all_certified.
        this->auto_phase2_certified =
            this->auto_phase2_certified && this->auto_sqfa.all_certified;
    };
    const bool saved_use_qfa = this->use_qfa;
    this->use_qfa = true;
    T est;
    try {
        est = this->call(A_op, fAfun, fscalar, k, s, q, state,
                         /*Omega2=*/nullptr, t1_out, t2_out, f_zero);
    } catch (...) {
        this->use_qfa = saved_use_qfa;
        throw;
    }
    this->use_qfa = saved_use_qfa;

    // ---- 5. Probe-sample reuse ----
    // The probe's certified columns are valid Hutchinson samples: the same
    // sphere-normalized Gaussian convention as the internal Omega2, with
    // quadratic forms gᵀf(A)g certified to rtol = eps. Fold them into the
    // Phase-2 average,
    //   t2 ← (s·t2 + Σ_cert [gᵀf(A)g − gᵀf(Â)g]) / (s + b_cert),
    // where gᵀf(Â)g comes from the Phase-1 eigenpairs exactly like the other
    // columns (y = Uᵀg; zero extra matvecs). Uncertified columns carry no
    // accuracy statement and are never reused. The probe's matvecs are
    // already charged to the budget, so the accounting is unchanged.
    int64_t b_cert = 0;
    for (int64_t j = 0; j < b; ++j)
        if (this->auto_probe_cert[j]) ++b_cert;
    if (b_cert > 0 && k < n) {
        // Y_2 (k × s from the expert call) is dead scratch at this point;
        // reuse it for the k × b probe projection Uᵀ·G_probe.
        util::upsize(this->Y_2, this->Y_2_sz, k * b);
        blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   k, b, n, (T)1, this->U, n, this->auto_probe_buf, n,
                   (T)0, this->Y_2, k);
        const bool apply_fzero = f_zero.has_value();
        const T    fz          = apply_fzero ? *f_zero : (T)0;
        T probe_sum = (T)0;
        for (int64_t j = 0; j < b; ++j) {
            if (!this->auto_probe_cert[j]) continue;
            const T* yj = this->Y_2 + j * k;
            T ghat = (T)0;
            for (int64_t i = 0; i < k; ++i)
                ghat += fscalar(this->lambda[i]) * yj[i] * yj[i];
            if (apply_fzero) {
                // Zero-fill complement term, same convention as the expert
                // call: f(Â) = U·diag(f(λ))·Uᵀ + f(0)·(I − U·Uᵀ).
                const T* gj  = this->auto_probe_buf + j * n;
                const T g_sq = blas::dot(n, gj, 1, gj, 1);
                const T y_sq = blas::dot(k, yj, 1, yj, 1);
                ghat += fz * (g_sq - y_sq);
            }
            probe_sum += this->auto_probe_gauss[j] - ghat;
        }
        t2_out = (t2_out * (T)s + probe_sum) / (T)(s + b_cert);
        est    = t1_out + t2_out;
    }
    return est;
}


} // namespace RandLAPACK
