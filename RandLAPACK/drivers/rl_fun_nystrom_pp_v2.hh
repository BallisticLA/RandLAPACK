#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_nystrom_evd_v2.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

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
/// Phase 1 (NystromEVD_v2, implemented inline below): Gaussian sketch +
/// q − 1 subspace-iteration passes (with QR stabilization between
/// passes) + SVD-pseudoinverse recovery on the k × k Gram. Mirrors
/// `nystrom.m` line-for-line.
///
/// Phase 2 (this class's `call`): Hutchinson on the residual
/// f(A) − f(Â) using a caller-supplied fAfun oracle. Mirrors
/// `funnystrompp.m` line-for-line. Uses Gaussian Ω₂ to match the
/// reference. A Rademacher option will be added in Phase 6 of the
/// project plan.
///
/// The class takes Ω₁ (Phase 1 sketch) and Ω₂ (Phase 2 Hutchinson)
/// externally — this is the cross-validation harness contract. The
/// caller generates or loads them via the same RNG as the MATLAB side.
/// See util::load_dense_bin for the on-disk format.
///
/// See [funnystrompp-v2-baseline-plan.md](Claude_files/randnla/project-plans/funnystrompp-v2-baseline-plan.md)
/// for the multi-phase development plan.
template <typename T>
class FunNystromPP_v2 {
public:
    bool verbose = false;
    bool force_fallback = false;   // Phase 7a perf knob: skip Cholesky-fast in NystromEVD_v2 and use the SVD-pinv path.

    // After call(), these hold Phase 1's eigenpairs of Â. Exposed as
    // public so tests can inspect them; in production code you'd treat
    // them as read-only output.
    std::vector<T> U;        // m × k_out, column-major
    std::vector<T> lambda;   // k_out descending eigenvalues
    int64_t k_out = 0;

    // Phase-split wall-clock timings populated by call() (ms).
    double t_phase1_ms = 0.0;
    double t_phase2_ms = 0.0;
    // Inner wall-clock of just the dual-path spectral-recovery block
    // inside NystromEVD_v2 (Cholesky-fast vs SVD-pinv fall-back). This
    // is the tight measurement for the Phase 7a A/B comparison; the
    // QR + subspace-iter + final matvec costs that precede it are
    // identical on both paths and contribute to t_phase1_ms.
    double t_specrec_ms = 0.0;

    FunNystromPP_v2() = default;
    FunNystromPP_v2(const FunNystromPP_v2&) = delete;
    FunNystromPP_v2& operator=(const FunNystromPP_v2&) = delete;

    /// Returns the estimate t = t1 + t2 of tr(f(A)).
    ///
    /// fAfun(int64_t s, const T* B, T* Y) is a callable that computes
    /// Y := f(A) * B for column-major B, Y of shape m × s with lda = m.
    /// In Phase 1 tests we pass a dense exact-f(A) oracle; in Phase 4
    /// we will swap it for block_lanczos_fa.
    ///
    /// @param[in]  A_op     Symmetric linop providing A * X.
    /// @param[in]  fAfun    Callable B ↦ f(A) * B.
    /// @param[in]  fscalar  Scalar f operating on each eigenvalue.
    /// @param[in]  k        Phase 1 Nyström rank.
    /// @param[in]  s        Phase 2 Hutchinson sample count.
    /// @param[in]  q        Phase 1 number of A applications (q = 1 single
    ///                      pass; q = 2 = 1 subspace-iter pass; etc.).
    /// @param[in]  Omega1   Caller-supplied m × k sketch for Phase 1.
    /// @param[in]  Omega2   Caller-supplied m × s sketch for Phase 2
    ///                      (Gaussian in the v2 baseline). When k == m
    ///                      Phase 2 is skipped and Omega2 is unread; the
    ///                      caller may pass nullptr.
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
    template <linops::SymmetricLinearOperator SLO, typename FAFun, typename FScalar>
    T call(
        SLO &A_op,
        FAFun &&fAfun,
        FScalar &&fscalar,
        int64_t k,
        int64_t s,
        int64_t q,
        const T *Omega1,
        const T *Omega2,
        T &t1_out,
        T &t2_out,
        std::optional<T> f_zero = std::nullopt
    );

    /// Sparse-sketch overload: Phase 1 sketch is a `RandBLAS::SparseSkOp`
    /// rather than a dense buffer. Routes the first matvec Y0 = A · S
    /// through the SkOp-taking operator() on the SLO (which dispatches
    /// to `RandBLAS::sparse_data::right_spmm` for `ExplicitSymLinOp`),
    /// then delegates to the dense path with `q_effective = q − 1`.
    /// Algorithmically equivalent to the reference; same answer at
    /// fixed RNG as densifying `S` and calling the dense overload.
    ///
    /// PRECONDITIONS:
    ///   - q >= 2. For q == 1 there's no first-matvec to amortize
    ///     (the dense path does QR + one matvec; with sparse Ω₁ that
    ///     means densifying anyway). Throws std::invalid_argument.
    ///   - The SLO supports the SkOp-taking operator() overload.
    ///     For `linops::ExplicitSymLinOp` this requires BOTH triangles
    ///     of A populated (right_spmm doesn't exploit symmetry; the
    ///     `RandBLAS::sparse_symm_spmm` upstream work, when it lands,
    ///     will close the ~2× cost gap).
    ///
    /// Other parameters and semantics match the dense overload above.
    template <linops::SymmetricLinearOperator SLO,
              RandBLAS::SketchingOperator SkOp,
              typename FAFun, typename FScalar>
    T call(
        SLO &A_op,
        FAFun &&fAfun,
        FScalar &&fscalar,
        int64_t k,
        int64_t s,
        int64_t q,
        SkOp &Omega1_sparse,
        const T *Omega2,
        T &t1_out,
        T &t2_out,
        std::optional<T> f_zero = std::nullopt
    );
};



// --- Phase 2: FunNystromPP_v2::call ---------------------------------------

template <typename T>
template <linops::SymmetricLinearOperator SLO, typename FAFun, typename FScalar>
T FunNystromPP_v2<T>::call(
    SLO &A_op,
    FAFun &&fAfun,
    FScalar &&fscalar,
    int64_t k,
    int64_t s,
    int64_t q,
    const T *Omega1,
    const T *Omega2,
    T &t1_out,
    T &t2_out,
    std::optional<T> f_zero
) {
    int64_t m = A_op.dim;

    if (f_zero.has_value() && !std::isfinite(*f_zero))
        throw std::invalid_argument(
            "FunNystromPP_v2::call: f_zero must be finite when provided");

    // Phase 1.
    auto t_p1_start = std::chrono::steady_clock::now();
    NystromEVD_v2_workspace<T> ws;
    NystromEVD_v2<T>(A_op, k, q, Omega1, this->U, this->lambda, ws, this->force_fallback, &this->t_specrec_ms);
    this->k_out = k;

    // Resolve f(0) for the optional (n − k) · f(0) correction term in
    // tr(f(Â)). Default (f_zero = nullopt) matches Persson MATLAB which
    // omits this term — that's the cross-validation anchor. Caller can
    // opt in to production-style behavior (PR #132) by passing a finite
    // f_zero, in which case both t1 (via the (m−k)·f(0) term below) and
    // t2 (via the projector-complement correction below) are adjusted.
    // Both estimators converge to tr(f(A)) in expectation; for a fixed Ω
    // they differ by `f(0)·[(m−k) − (‖Ω‖²_F − ‖VᵀΩ‖²_F)/s]`, a quantity
    // that vanishes on average when Ω is iid Gaussian / Rademacher.
    // No auto-resolve from fscalar(0): for f = log the auto-call would
    // produce -∞, breaking the bit-exact MATLAB anchor on log-style
    // fixtures.
    const bool   apply_fzero = f_zero.has_value() && (k < m);
    const T      fz          = apply_fzero ? *f_zero : (T)0;

    // t1 = Σ fscalar(λᵢ) (+ (m − k) · fz when f_zero supplied).
    t1_out = (T)0;
    for (int64_t i = 0; i < k; ++i) t1_out += fscalar(this->lambda[i]);
    if (apply_fzero) t1_out += static_cast<T>(m - k) * fz;
    auto t_p1_end = std::chrono::steady_clock::now();
    this->t_phase1_ms = std::chrono::duration<double, std::milli>(t_p1_end - t_p1_start).count();

    // Phase 2: t2 = ( tr(Ω₂ᵀ · fAfun(Ω₂)) − tr(Yᵀ · diag(f(λ)) · Y) ) / s
    //   where Y = Uᵀ · Ω₂ (shape k × s).
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

        // Step a: Y_2 ← Uᵀ · Ω₂  (k × s)
        std::vector<T> Y_2(k * s, (T)0);
        blas::gemm(Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   k, s, m, (T)1, this->U.data(), m, Omega2, m, (T)0, Y_2.data(), k);

        // Step b: fAOmega ← f(A) · Ω₂  (m × s, caller-supplied oracle)
        std::vector<T> fAOmega(m * s, (T)0);
        fAfun(m, s, Omega2, fAOmega.data());

        // Step c: tr_AΩ ← tr(Ω₂ᵀ · fAOmega) = Σⱼ ⟨Ω₂[:,j], fAOmega[:,j]⟩.
        T tr_AOmega = (T)0;
        for (int64_t j = 0; j < s; ++j) {
            tr_AOmega += blas::dot(m, Omega2 + j * m, 1, fAOmega.data() + j * m, 1);
        }

        // Step d: tr_AhatΩ ← tr(Ω₂ᵀ · f(Â) · Ω₂).
        // When apply_fzero, the rank-k Â is "zero-filled" to an n×n
        // operator: f(Â) = V·diag(f(λ))·Vᵀ + f(0)·(I − V·Vᵀ). The
        // projector-complement term contributes
        //   f(0) · [‖Ω₂‖²_F − ‖Y_2‖²_F]
        // to tr_AhatΩ, computed alongside the rank-k diagonal sum.
        T tr_AhatOmega = (T)0;
        for (int64_t j = 0; j < s; ++j) {
            for (int64_t i = 0; i < k; ++i) {
                T v = Y_2[i + j * k];
                tr_AhatOmega += fscalar(this->lambda[i]) * v * v;
            }
        }
        if (apply_fzero) {
            const T omega_fro_sq = blas::dot(m * s, Omega2, 1, Omega2, 1);
            const T y2_fro_sq    = blas::dot(k * s, Y_2.data(), 1, Y_2.data(), 1);
            tr_AhatOmega += fz * (omega_fro_sq - y2_fro_sq);
        }

        t2_out = (tr_AOmega - tr_AhatOmega) / (T)s;
        auto t_p2_end = std::chrono::steady_clock::now();
        this->t_phase2_ms = std::chrono::duration<double, std::milli>(t_p2_end - t_p2_start).count();
    } else {
        this->t_phase2_ms = 0.0;
    }
    return t1_out + t2_out;
}


// Sparse-sketch overload (Phase 6 + Gap 5: SASO + SkOp-aware first matvec
// pulled into the driver). Computes Y0 = A · S through the SkOp path on
// the SLO, then delegates to the dense overload with q − 1.
template <typename T>
template <linops::SymmetricLinearOperator SLO,
          RandBLAS::SketchingOperator SkOp,
          typename FAFun, typename FScalar>
T FunNystromPP_v2<T>::call(
    SLO &A_op,
    FAFun &&fAfun,
    FScalar &&fscalar,
    int64_t k,
    int64_t s,
    int64_t q,
    SkOp &Omega1_sparse,
    const T *Omega2,
    T &t1_out,
    T &t2_out,
    std::optional<T> f_zero
) {
    int64_t m = A_op.dim;
    if (q < 2) {
        throw std::invalid_argument(
            "FunNystromPP_v2::call (SkOp overload): q must be >= 2. "
            "For q == 1, densify the sketch caller-side and call the "
            "dense Omega1 overload.");
    }
    // Y0 = A · S via the SkOp-aware operator() overload on the SLO
    // (dispatches to RandBLAS::sparse_data::right_spmm for
    // ExplicitSymLinOp). Caller is responsible for ensuring A has both
    // triangles populated when using ExplicitSymLinOp — right_spmm
    // treats A as generic dense.
    std::vector<T> Y0(m * k, (T)0);
    A_op(Layout::ColMajor, k, (T)1, Omega1_sparse, (T)0, Y0.data(), m);

    // Delegate to the dense path. q_effective = q - 1 because the
    // sparse first matvec replaces the dense path's initial
    // qr(Ω) → A·Ω' step; the dense path will then do q - 2 subspace-iter
    // passes + a final A·Ω matvec, matching the reference's total of
    // q matvecs of A.
    return this->call(A_op,
                      std::forward<FAFun>(fAfun),
                      std::forward<FScalar>(fscalar),
                      k, s, q - 1,
                      Y0.data(), Omega2,
                      t1_out, t2_out, f_zero);
}


} // namespace RandLAPACK
