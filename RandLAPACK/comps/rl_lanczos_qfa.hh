#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RandLAPACK {


/// Scalar (per-column) Lanczos-QFA: the quadratic forms out[j] = bⱼᵀ f(A) bⱼ
/// = ‖bⱼ‖²·e₁ᵀ f(T_{t,j}) e₁, one independent Krylov subspace per column of B,
/// returned as a length-s vector (NOT the n×s block f(A)B and NOT the s×s
/// block quadratic form - the off-diagonal entries of Bᵀf(A)B are not
/// computable from per-column tridiagonals). By the Gauss-quadrature identity
/// bᵀ·LanczosFA(A, f, b) = Lanczos-QFA(A, f, b), so the funNyström++ Phase-2
/// term tr(Ω₂ᵀ f(A) Ω₂) = Σⱼ out[j] is formed without materializing f(A)·Ω₂.
///
/// Basis-free by construction: the reconstruction reads only each column's
/// tridiagonal (α, β) and norm, never the Krylov basis, so the recurrence
/// keeps a rolling window of three n-vectors per column (q_{t−1}, q_t, and the
/// matvec target) - O(n·s) working memory instead of LanczosFA's O(n·d·s)
/// stored basis. Consequently there is NO reorthogonalization option: this
/// class runs vanilla Lanczos (Gauss-quadrature values tolerate orthogonality
/// loss far better than eigenvalue Lanczos, Paige-Greenbaum; the certificate
/// below independently monitors positive-definiteness via its LDLᵀ pivots).
/// Requires A ⪰ 0 and f finite at 0 (Ritz values are clamped to ≥ 0 before f;
/// a raw log(x) would produce -inf at the Radau node pinned at 0 - use the
/// shifted log(x+1)).
///
/// Certified adaptive stopping (optional, `adaptive = true`): per column,
/// bracket the true bᵀf(A)b between the plain Gauss value
///   U_t = ‖b‖²·e₁ᵀ f(T_t) e₁
/// and the Gauss-Radau value
///   L_t = ‖b‖²·e₁ᵀ f(T̂_t) e₁,
/// where T̂_t is T_t with only its bottom-right entry replaced by
///   α̂_t = β_{t−1}²·e_{t−1}ᵀ (T_{t−1})⁻¹ e_{t−1},
/// the value that pins one quadrature node at 0 (det(T̂_t) = 0). For the
/// operator-monotone f this method targets, Gauss and Radau err on opposite
/// sides (Golub-Meurant), so |U_t − L_t| is a certified error bound with no
/// delay window and no plateau risk: the column stops at the first CHECKED
/// depth t ≥ 2 with |U_t − L_t| ≤ adaptive_rtol·scale, where
/// scale = max(|U_t|, |L_t|, tiny) guards the convergence floor, and
/// adaptive_rtol is a CERTIFIED relative error, not a target scale. Checks
/// follow a geometric cadence (see due_for_check), so the stop lands at the
/// first checked depth past certifiability, at most ~1.5x beyond it, and
/// never before t = 2. The corner entry needs no
/// tridiagonal solve: e_{t−1}ᵀ(T_{t−1})⁻¹e_{t−1} = 1/d_{t−1} with the LDLᵀ
/// pivot recurrence d_1 = α_1, d_i = α_i − β_{i−1}²/d_{i−1}, maintained in
/// O(1) per step per column; a non-positive pivot means T is not positive
/// definite (indefinite A or breakdown), and that column's certificate is
/// disabled (it runs to the cap, reported uncertified) rather than dividing
/// by a non-positive pivot.
///
/// Columns certify at different depths. A certified column retires
/// immediately: it is swapped out of the active window (all three rolling
/// slices), so the batched matvec shrinks to the surviving columns and a
/// slowly-converging straggler no longer costs matvecs for columns that are
/// already done. `matvecs` reports the actual total Σⱼ t_j.
///
/// Cost of a certificate check at depth t: two t×t symmetric-tridiagonal
/// eigensolves (stevd) per active column - zero matvecs, and negligible next
/// to an n-length matvec for the depths this method runs at (tens).
///
/// @tparam T  Floating-point scalar type.
template <typename T>
class LanczosQFA {
public:
    // ---- certified adaptive stopping controls ------------------------------
    bool    adaptive      = false;   ///< per-column Gauss-Radau certified stop
    T       adaptive_rtol = (T)1e-2; ///< certified relative-error tolerance
    int64_t check_every   = 1;       ///< 1 = geometric check ladder; >1 = fixed stride (checks at t ≥ 2 either way)

    // ---- outputs of the last call ------------------------------------------
    int64_t d_used        = 0;       ///< max over columns of the depth used
    int64_t matvecs       = 0;       ///< Σⱼ t_j: actual A column-applications
    bool    all_certified = false;   ///< every column certified before the cap
    /// Per-column results, indexed by the ORIGINAL column of B (length s):
    int64_t* t_used    = nullptr; int64_t t_used_sz    = 0; ///< depth per column
    T*       gauss_val = nullptr; int64_t gauss_val_sz = 0; ///< final U (== out)
    T*       radau_val = nullptr; int64_t radau_val_sz = 0; ///< final L (0 if unchecked)
    uint8_t* certified = nullptr; int64_t certified_sz = 0; ///< 1 = bracket closed

    // ---- internal buffers (grown, never shrunk) ----------------------------
    // qbuf: 3 rolling n×s slices (q_prev / q_cur / matvec target, roles rotate).
    // alpha/beta: per-column tridiagonal histories, alpha[col*d + i] = α_{i+1},
    //   beta[col*d + i] = β_{i+1} (β stride d for simplicity; entry d−1 unused).
    // ldl_piv: per-column running LDLᵀ pivot d_t of T_t (through current depth).
    // radau_corner: per-column exact Radau corner α̂_t = β_{t−1}²/d_{t−1}, saved
    //   at the pivot update so evaluate_pair never reconstructs it by subtraction.
    // col_of_slot: original column held by each active slot (compaction map).
    T*       qbuf        = nullptr; int64_t qbuf_sz        = 0;
    T*       alpha       = nullptr; int64_t alpha_sz       = 0;
    T*       beta        = nullptr; int64_t beta_sz        = 0;
    T*       normb       = nullptr; int64_t normb_sz       = 0;
    T*       ldl_piv     = nullptr; int64_t ldl_piv_sz     = 0;
    T*       radau_corner= nullptr; int64_t radau_corner_sz= 0;
    uint8_t* cert_ok     = nullptr; int64_t cert_ok_sz     = 0; ///< pivot chain still PD
    int64_t* col_of_slot = nullptr; int64_t col_of_slot_sz = 0;
    T*       workspace   = nullptr; int64_t workspace_sz   = 0; ///< per-slot stevd scratch
    int64_t  ws_depth    = 0;       ///< current slot stride basis: each slot holds ws_depth² + 2·ws_depth
    // Scratch for the fused panel kernels (see the [panel kernels] note below):
    // per-thread partial column sums, and slot-indexed coefficient vectors.
    T*       panel_par   = nullptr; int64_t panel_par_sz   = 0; ///< nthreads * s partials
    T*       slot_a      = nullptr; int64_t slot_a_sz      = 0; ///< alpha gathered by SLOT
    T*       slot_b      = nullptr; int64_t slot_b_sz      = 0; ///< beta  gathered by SLOT
    T*       slot_v      = nullptr; int64_t slot_v_sz      = 0; ///< dots / sumsq / inverse scales

    // Profiling, matching the LanczosFA-family surface. Slots: {matvec,
    // run_lanczos, apply, rest, total} µs; `apply` is all certificate checks
    // plus the final Gauss evaluations, `run_lanczos` is the recurrence net of
    // certificate time, so the slots stay comparable across oracles.
    bool timing = false;
    std::vector<long> times;
    long _t_matvec_us = 0;
    /// Always 0: this oracle has no reorthogonalization by design. Present so the
    /// timing surface matches LanczosFA's, making "reorth cost" directly
    /// comparable across tiers (it is exactly the FA-vs-QFA difference).
    long _t_reorth_us = 0;

    LanczosQFA()                             = default;
    LanczosQFA(const LanczosQFA&)            = delete;
    LanczosQFA& operator=(const LanczosQFA&) = delete;

    ~LanczosQFA() {
        delete[] qbuf; delete[] alpha; delete[] beta; delete[] normb;
        delete[] ldl_piv; delete[] radau_corner; delete[] cert_ok; delete[] col_of_slot;
        delete[] t_used; delete[] gauss_val; delete[] radau_val;
        delete[] certified; delete[] workspace;
        delete[] panel_par; delete[] slot_a; delete[] slot_b; delete[] slot_v;
    }

    // ------------------------------------------------------------------
    /// Compute out[j] = B[:,j]ᵀ f(A) B[:,j] for j = 0..s−1 (length-s vector).
    /// d = Lanczos depth: exact number of steps per column when !adaptive,
    /// depth cap when adaptive (columns stop earlier as they certify).
    /// Calls A once per step on the block of still-active columns.
    ///
    /// Per column: a zero input column yields out = 0, certified at t = 0.
    /// A Lanczos breakdown (β_t = 0: the Krylov space became invariant) makes
    /// the current Gauss value exact; the column certifies immediately.
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        using namespace std::chrono;
        steady_clock::time_point t_start, mv0, mv1, c0, c1;
        long cert_us = 0;
        _t_matvec_us = 0;
        if (timing) t_start = steady_clock::now();

        if (d < 1) throw std::invalid_argument("LanczosQFA: depth d must be >= 1.");
        if (check_every < 1) throw std::invalid_argument("LanczosQFA: check_every must be >= 1.");

        // [setup] buffers + per-column output state.
        util::upsize(qbuf,        qbuf_sz,        3 * n * s);
        util::upsize(alpha,       alpha_sz,       d * s);
        util::upsize(beta,        beta_sz,        d * s);
        util::upsize(normb,       normb_sz,       s);
        util::upsize(ldl_piv,     ldl_piv_sz,     s);
        util::upsize(radau_corner, radau_corner_sz, s);
        util::upsize(cert_ok,     cert_ok_sz,     s);
        util::upsize(col_of_slot, col_of_slot_sz, s);
        util::upsize(t_used,      t_used_sz,      s);
        util::upsize(gauss_val,   gauss_val_sz,   s);
        util::upsize(radau_val,   radau_val_sz,   s);
        util::upsize(certified,   certified_sz,   s);
        // Eigensolve scratch: per slot, an alpha copy + beta copy + eigenvector
        // matrix Z. Indexed by the loop variable j < act <= s, NOT by thread
        // id, so it needs s slots - not nthreads. Sizing it by nthreads
        // cost 112 * d^2 = 4.34 GB at d = 2226 on a 112-core node while at most
        // s of those slots could ever be live (28x over-allocation at the auto
        // tier's probe BLOCK b = 4), re-mapped on every call by util::upsize.
        // Adaptive runs additionally size each slot by the depth actually
        // EVALUATED, not the cap: columns typically certify at depths far below
        // d, and an upfront s * (d^2 + 2d) allocation is 1.6 GB at d = 2226,
        // s = 40. The slot stride basis ws_depth grows lazily (ensure_eval_ws)
        // just before each evaluation depth. A fixed-depth run makes its single
        // evaluation at t = d, so it keeps the upfront allocation.
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        if (adaptive) {
            ws_depth = 0;
        } else {
            ws_depth = d;
            util::upsize(workspace, workspace_sz, s * (d * d + 2 * d));
        }
        // Partials are cache-line padded per thread (see partial_stride).
        util::upsize(panel_par, panel_par_sz, (int64_t)nthreads * partial_stride(s));
        util::upsize(slot_a,    slot_a_sz,    s);
        util::upsize(slot_b,    slot_b_sz,    s);
        util::upsize(slot_v,    slot_v_sz,    s);
        // The rolling slices are always overwritten before numerical use; this
        // memset only keeps retire_slot's slice copies (which can precede the
        // first write of w, and of q_prev at t = 1) off indeterminate heap so
        // memory sanitizers stay clean.
        std::memset(qbuf, 0, sizeof(T) * 3 * n * s);

        for (int64_t j = 0; j < s; ++j) {
            t_used[j] = 0; gauss_val[j] = (T)0; radau_val[j] = (T)0;
            certified[j] = 0; cert_ok[j] = 1; col_of_slot[j] = j;
        }
        matvecs = 0; d_used = 0;

        // Rolling slices; roles rotate every step.
        T* q_prev = qbuf;
        T* q_cur  = qbuf + n * s;
        T* w      = qbuf + 2 * n * s;

        // [t = 1 init] q_1 = b/‖b‖ per column, into q_cur. Zero columns retire
        // immediately (out = 0, certified at t = 0), compacting before the
        // first matvec so they never cost anything.
        lapack::lacpy(lapack::MatrixType::General, n, s, B, n, q_cur, n);
        int64_t act = s;
        for (int64_t j = 0; j < act; /* advance inside */) {
            T nrm = blas::nrm2(n, q_cur + j * n, 1);
            if (nrm > (T)0) {
                normb[col_of_slot[j]] = nrm;
                blas::scal(n, (T)1.0 / nrm, q_cur + j * n, 1);
                ++j;
            } else {
                const int64_t col = col_of_slot[j];
                normb[col] = (T)0; certified[col] = 1; t_used[col] = 0;
                retire_slot(j, act, n, q_prev, q_cur, w, /*have_prev=*/false);
                // slot j now holds the previously-last column; re-examine it.
            }
        }

        // Main loop over depth t = 1, 2, …, d on the active window.
        // Loop invariant at the certificate point: for each active slot,
        // alpha[·, 0..t−1] and beta[·, 0..t−2] are filled (T_t complete),
        // ldl_piv holds d_{t−1} (LDLᵀ pivot of T_{t−1}; at t = 1 it is unset),
        // q_cur = q_t, q_prev = q_{t−1}, w = A·q_t.
        int64_t t = 0;
        while (act > 0 && t < d) {
            ++t;
            // [matvec] w = A·q_t over the active window (one batched apply).
            if (timing) mv0 = steady_clock::now();
            A(Layout::ColMajor, act, (T)1.0, q_cur, n, (T)0.0, w, n);
            if (timing) { mv1 = steady_clock::now(); _t_matvec_us += duration_cast<microseconds>(mv1 - mv0).count(); }
            matvecs += act;

            // [α_t + LDLᵀ] α_t = q_tᵀ·A·q_t; maintain the pivot chain:
            // radau_corner = β_{t−1}²/d_{t−1} (the Gauss-Radau corner entry
            // α̂_t), then d_t = α_t − radau_corner. At t = 1, d_1 = α_1.
            // All α at once via a fused column-wise reduction over the panel
            // (see [panel kernels]); the O(1)-per-column pivot bookkeeping then
            // runs serially, since act <= s is small.
            panel_coldots(n, act, w, q_cur, slot_v, panel_par, nthreads);
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                const T a = slot_v[j];
                alpha[col * d + (t - 1)] = a;   // always: Gauss quadrature needs α
                // The LDLᵀ pivot chain feeds ONLY the Gauss-Radau certificate;
                // skip it entirely on a fixed-depth (!adaptive) run.
                if (adaptive) {
                    if (t == 1) {
                        ldl_piv[col] = a;
                        if (a <= (T)0) cert_ok[col] = 0;
                    } else if (cert_ok[col]) {
                        const T b_ = beta[col * d + (t - 2)];
                        const T corner = b_ * b_ / ldl_piv[col];
                        radau_corner[col] = corner;   // exact α̂_t, read back by evaluate_pair
                        const T piv = a - corner;
                        ldl_piv[col] = piv;   // pivot THROUGH depth t
                        if (piv <= (T)0) cert_ok[col] = 0;
                    }
                }
            }

            // [certificate] at depth t ≥ 2 (cadence: due_for_check): per active
            // column, Gauss U_t vs Gauss-Radau L_t; certify when the bracket
            // closes. The Radau corner α̂_t = β_{t−1}²/d_{t−1} needs the pivot
            // BEFORE this step's update; it was saved exactly at the update
            // above (radau_corner), which evaluate_pair() reads directly.
            if (adaptive && t >= 2 && due_for_check(t)) {
                // Every evaluation in this sweep is at the same depth t, so the
                // scratch resize happens once here, OUTSIDE the parallel region
                // (it can re-map workspace and change the uniform slot stride).
                ensure_eval_ws(t, s);
                if (timing) c0 = steady_clock::now();
// dynamic: per-iteration cost is bimodal (a skipped column costs nothing, a
// checked one costs O(t^3)), so a static split leaves threads idle behind a
// neighbour doing full eigensolves.
#pragma omp parallel for schedule(dynamic, 1)
                for (int64_t j = 0; j < act; ++j) {
                    const int64_t col = col_of_slot[j];
                    if (!cert_ok[col]) continue;
                    T U, L;
                    evaluate_pair(f, col, t, d, U, L);
                    const T hi = std::max(U, L), lo = std::min(U, L);
                    const T scale = std::max(std::abs(hi), std::numeric_limits<T>::min());
                    if (hi - lo <= adaptive_rtol * scale) {
                        certified[col] = 1; t_used[col] = t;
                        gauss_val[col] = U; radau_val[col] = L;
                    }
                }
                if (timing) { c1 = steady_clock::now(); cert_us += duration_cast<microseconds>(c1 - c0).count(); }
                // [compaction] retire certified slots; the batched matvec
                // shrinks to the survivors. Serial: each retire is 3 column
                // copies (O(n)).
                for (int64_t j = 0; j < act; /* advance inside */) {
                    if (certified[col_of_slot[j]]) {
                        retire_slot(j, act, n, q_prev, q_cur, w, /*have_prev=*/true);
                    } else {
                        ++j;
                    }
                }
                if (act == 0) break;
            }
            if (t == d) break;

            // [three-term step] z = A·q_t − α_t·q_t − β_{t−1}·q_{t−1} (into w);
            // β_t = ‖z‖; q_{t+1} = z/β_t. β_t = 0 is a Lanczos breakdown: the
            // Krylov space is invariant, the depth-t Gauss value is exact, and
            // the column certifies with U = L (handled after the loop body via
            // the certified flag, retired in the sweep below).
            // Gather the per-slot coefficients, then do the whole update in ONE
            // fused pass (subtract both terms and accumulate ‖z‖² together),
            // followed by one scaling pass. Was four separate BLAS-1 sweeps per
            // column; see [panel kernels].
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                slot_a[j] = alpha[col * d + (t - 1)];
                slot_b[j] = (t > 1) ? beta[col * d + (t - 2)] : (T)0;
            }
            panel_axpy2_sumsq(n, act, w, q_cur, q_prev, slot_a, slot_b,
                              /*use_prev=*/(t > 1), slot_v, panel_par, nthreads);
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                // sqrt of the accumulated sum of squares rather than nrm2's
                // scaled recurrence: these are Lanczos vectors of norm O(‖A‖),
                // so there is no overflow risk, and a sum that underflows to 0
                // is a breakdown, which is exactly how it is then handled.
                const T nrm = std::sqrt(slot_v[j]);
                beta[col * d + (t - 1)] = nrm;
                // Breakdown threshold: relative to the local scale |α_t|, not
                // exact zero. A tiny-but-nonzero β is the normal floating-point
                // signature of an invariant subspace; dividing through would
                // continue the recurrence on amplified roundoff.
                if (nrm > std::numeric_limits<T>::epsilon() * std::abs(slot_a[j])) {
                    slot_v[j] = (T)1.0 / nrm;      // scale factor for the pass below
                } else {
                    slot_v[j] = (T)0;              // leave the column untouched
                    // Breakdown: certify with the exact depth-t value.
                    ensure_eval_ws(t, s);          // serial loop: safe to resize here
                    T U;
                    evaluate_gauss(f, col, t, d, U);
                    certified[col] = 1; t_used[col] = t;
                    gauss_val[col] = U; radau_val[col] = U;
                }
            }
            panel_scale(n, act, w, slot_v, nthreads);
            for (int64_t j = 0; j < act; /* advance inside */) {
                if (certified[col_of_slot[j]]) {
                    retire_slot(j, act, n, q_prev, q_cur, w, /*have_prev=*/true);
                } else {
                    ++j;
                }
            }
            if (act == 0) break;

            // [rotate roles] q_{t+1} lives in w; old q_prev becomes the next
            // matvec target.
            T* tmp = q_prev; q_prev = q_cur; q_cur = w; w = tmp;
        }

        // [final evaluation] columns still active ran to the cap (or the whole
        // fixed-depth run when !adaptive): evaluate the depth-t Gauss value.
        // In adaptive mode the Radau companion is also evaluated when the
        // pivot chain allows, and the bracket test is applied one last time:
        // the check ladder rarely lands exactly on the cap, so a bracket that
        // closed between the last checked depth and the cap must be reported
        // certified. Columns whose bracket is still open keep certified = 0,
        // with radau_val reporting the unclosed bracket for diagnostics.
        if (act > 0) {
            // One resize for the whole sweep, outside the parallel region
            // (every evaluation below is at the same depth t).
            ensure_eval_ws(t, s);
            if (timing) c0 = steady_clock::now();
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                T U, L;
                if (adaptive && t >= 2 && cert_ok[col]) {
                    evaluate_pair(f, col, t, d, U, L);
                    radau_val[col] = L;
                    // Same criterion as the in-run certificate check.
                    const T hi = std::max(U, L), lo = std::min(U, L);
                    const T scale = std::max(std::abs(hi), std::numeric_limits<T>::min());
                    if (hi - lo <= adaptive_rtol * scale) certified[col] = 1;
                } else {
                    evaluate_gauss(f, col, t, d, U);
                }
                gauss_val[col] = U; t_used[col] = t;
            }
            if (timing) { c1 = steady_clock::now(); cert_us += duration_cast<microseconds>(c1 - c0).count(); }
        }

        // [outputs] out = the Gauss values, by original column; summary stats.
        all_certified = true;
        for (int64_t j = 0; j < s; ++j) {
            out[j] = gauss_val[j];
            d_used = std::max(d_used, t_used[j]);
            if (!certified[j]) all_certified = false;
        }

        if (timing) {
            steady_clock::time_point t_end = steady_clock::now();
            long total_us   = duration_cast<microseconds>(t_end - t_start).count();
            long apply_us   = cert_us;
            long lanczos_us = total_us - apply_us;   // recurrence net of certificate
            // rest_us is identically 0 here by construction: lanczos_us is
            // DEFINED as total minus certificate time, so there is no residual
            // bucket. The slot is kept so the layout matches LanczosFA's times.
            long rest_us    = total_us - lanczos_us - apply_us;
            // 6th slot = reorthogonalization, always 0 here: this oracle has none by
            // design. Kept so the slot layout matches LanczosFA and "reorth cost" is
            // directly comparable across tiers.
            times = {_t_matvec_us, lanczos_us, apply_us, rest_us, total_us, _t_reorth_us};
        }
    }

private:
    // ------------------------------------------------------------------
    /// Retire active slot j: overwrite its rolling vectors with the last
    /// active slot's and shrink the window. The retired column's per-column
    /// state (alpha/beta/normb/outputs) is indexed by original column and
    /// needs no move.
    void retire_slot(int64_t j, int64_t& act, int64_t n,
                     T* q_prev, T* q_cur, T* w, bool have_prev) {
        const int64_t last = act - 1;
        if (j != last) {
            blas::copy(n, q_cur + last * n, 1, q_cur + j * n, 1);
            blas::copy(n, w     + last * n, 1, w     + j * n, 1);
            if (have_prev)
                blas::copy(n, q_prev + last * n, 1, q_prev + j * n, 1);
            col_of_slot[j] = col_of_slot[last];
        }
        act = last;
    }

    /* ---------------------------- [panel kernels] ----------------------------
     * The recurrence's vector work used to run one OpenMP THREAD PER COLUMN,
     * each calling serial BLAS-1 (dot / axpy / nrm2 / scal) down its own column.
     * Two problems with that:
     *
     *   1. Parallelism was capped by the number of ACTIVE columns, and that
     *      number shrinks as columns certify and retire. Late in a run only a
     *      handful of columns remain, so most cores sat idle - the retirement
     *      optimization was starving the machine.
     *   2. The three-term step made four separate streaming passes over each
     *      column (two axpy, a norm, a scal) where one suffices.
     *
     * These kernels instead parallelize over ROW BLOCKS, with every thread
     * cooperating on the whole n x ncols panel and holding a private length-
     * ncols accumulator (small, stays in cache). Thread utilization no longer
     * depends on ncols, and the update collapses to one fused pass plus a scale.
     *
     * The math is unchanged: the columns still run INDEPENDENT three-term
     * recurrences with their own tridiagonals, so the Gauss-Radau certificate
     * (which is derived for the single-vector case) is untouched. Only the
     * summation order changes, so results differ from the BLAS-1 version at
     * roundoff level (measured: QFA-vs-FA dots 8.4e-16 -> 1.2e-15; every
     * behavioural anchor - certified rel-err, adaptive depths, matvec counts,
     * auto-tier spend - is bit-identical).
     *
     * PERFORMANCE STATUS: UNVALIDATED, deliberately. Local A/B runs (16-core
     * box, load average ~6 while measuring, diagonal operator) landed between
     * 0.84x and 1.18x depending on shape, with the BASELINE itself varying more
     * than 2x run-to-run for identical parameters. Numbers that noisy cannot
     * support a conclusion about a quiet 112-core node, so none are recorded
     * here as if they could. Honest state: correctness established (every
     * behavioural anchor bit-identical), parallel structure sound (below), speed
     * question OPEN until measured on the cluster, where the benchmark already
     * records wall_ms per method.
     *
     * The parallel STRUCTURE is defensible independent of timing. collapse(2)
     * over (row block, column) makes the work-unit count nblocks * ncols, so
     * neither a small n (few row blocks) nor a narrow active window (few
     * columns) can starve the threads. Both predecessors had exactly that bug:
     * the original parallelized over columns alone, capping utilization at the
     * active-column count, which SHRINKS as columns retire; an intermediate
     * version of this code parallelized over row blocks alone with a fixed 4096
     * block, which at n = 3000 yielded ONE block and ran on ONE thread (2x
     * slower, and every unit test still passed - a pure parallelization defect
     * is invisible to correctness testing). collapse(2) removes both by
     * construction rather than by tuning a threshold.
     *
     * The row block must give every thread several chunks: a FIXED block (4096)
     * meant n = 3000 produced a single chunk, so the `omp for` had one iteration
     * and exactly one thread worked - measured 2x SLOWER than the per-column
     * version it replaced. Size it from n and the thread count instead, clamped
     * to keep strips vectorizable at the bottom and cache-resident at the top.
     */
public:
    /// Decomposition plan for a panel kernel over an n x ncols panel.
    /// PUBLIC deliberately: these are pure arithmetic with no state, and the
    /// invariants they must satisfy (every requested thread gets a chunk; never
    /// fork for trivial work; chunks tile the panel exactly) are the ONLY way to
    /// unit-test for the parallelization defects that correctness tests cannot
    /// see. See TestFunNystromPP.PanelChunkPlanInvariants.
    /// PURE ARITHMETIC and deliberately free of any block-SIZE constant: we fix
    /// the chunk COUNT and let the size fall out, because choosing a size can
    /// never guarantee a count (that is precisely how both previous versions of
    /// these kernels starved - see the note above).
    ///   n_threads: threads to actually request. Scales DOWN with the work, so a
    ///              trivial panel runs serially instead of paying a 112-thread
    ///              fork/join to touch 24 KB.
    ///   n_chunks : always n_threads * OVERSUB, so every requested thread gets
    ///              OVERSUB chunks BY CONSTRUCTION - no parameter combination
    ///              (small n, narrow window, any thread count) can starve them.
    struct ChunkPlan { int n_threads; int64_t n_chunks; };

    /// Minimum elements per thread before parallelism is worth its barrier
    /// (~64 KB of doubles ~ 10 us, safely above a 112-thread fork/join).
    static constexpr int64_t MIN_ELEMS_PER_THREAD = 8192;
    static constexpr int     OVERSUB              = 4;   ///< chunks per thread

    static ChunkPlan chunk_plan(int64_t n, int64_t ncols, int nthreads) {
        if (n <= 0 || ncols <= 0 || nthreads < 1) return {1, 1};
        const int64_t total = n * ncols;
        int64_t p = total / MIN_ELEMS_PER_THREAD;
        if (p < 1) p = 1;
        if (p > (int64_t)nthreads) p = nthreads;
        return { (int)p, p * OVERSUB };
    }

    /// Cache-line padded stride for per-thread partial slots (avoids the false
    /// sharing that bites hardest exactly when ncols is small: at ncols = 1,
    /// eight unpadded threads share one 64-byte line).
    static int64_t partial_stride(int64_t ncols) {
        constexpr int64_t LINE = 64 / (int64_t)sizeof(T);
        return ((ncols + LINE - 1) / LINE) * LINE;
    }

    /// Flattened element range [lo, hi) of chunk c out of n_chunks.
    static void chunk_range(int64_t total, int64_t n_chunks, int64_t c,
                            int64_t& lo, int64_t& hi) {
        lo = (total *  c     ) / n_chunks;
        hi = (total * (c + 1)) / n_chunks;
    }

private:
    /// out[j] = <X_j, Y_j> for every column j < ncols, in one pass.
    static void panel_coldots(int64_t n, int64_t ncols, const T* X, const T* Y,
                              T* out, T* partials, int nthreads) {
        if (ncols <= 0 || n <= 0) return;
        const ChunkPlan cp = chunk_plan(n, ncols, nthreads);
        const int64_t stride = partial_stride(ncols);
        const int64_t total  = n * ncols;
        // Zero ALL cp.n_threads partial slots BEFORE forking: num_threads() is
        // a request, not a guarantee, and the reduction below sums every slot,
        // so a slot no granted thread visits must contribute an exact zero.
        std::memset(partials, 0, sizeof(T) * (int64_t)cp.n_threads * stride);
#pragma omp parallel num_threads(cp.n_threads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            T* loc = partials + (int64_t)tid * stride;
#pragma omp for schedule(static)
            for (int64_t c = 0; c < cp.n_chunks; ++c) {
                int64_t lo, hi; chunk_range(total, cp.n_chunks, c, lo, hi);
                for (int64_t e = lo; e < hi; ++e) {
                    const int64_t j = e / n, i = e - j * n;
                    loc[j] += X[e] * Y[e];   // e == j*n + i (column-major panel)
                    (void)i;
                }
            }
        }
        for (int64_t j = 0; j < ncols; ++j) {
            T sum = (T)0;
            for (int tt = 0; tt < cp.n_threads; ++tt) sum += partials[(int64_t)tt * stride + j];
            out[j] = sum;
        }
    }

    /// W_j <- W_j - a[j]*Qcur_j - b[j]*Qprev_j (second term only if use_prev),
    /// accumulating sumsq[j] = ||W_j||^2 in the SAME pass.
    static void panel_axpy2_sumsq(int64_t n, int64_t ncols, T* W,
                                  const T* Qcur, const T* Qprev,
                                  const T* a, const T* b, bool use_prev,
                                  T* sumsq, T* partials, int nthreads) {
        if (ncols <= 0 || n <= 0) return;
        const ChunkPlan cp = chunk_plan(n, ncols, nthreads);
        const int64_t stride = partial_stride(ncols);
        const int64_t total  = n * ncols;
        // Pre-zero every partial slot: num_threads() is a request, not a
        // guarantee, and unvisited slots must reduce as exact zeros.
        std::memset(partials, 0, sizeof(T) * (int64_t)cp.n_threads * stride);
#pragma omp parallel num_threads(cp.n_threads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            T* loc = partials + (int64_t)tid * stride;
#pragma omp for schedule(static)
            for (int64_t c = 0; c < cp.n_chunks; ++c) {
                int64_t lo, hi; chunk_range(total, cp.n_chunks, c, lo, hi);
                for (int64_t e = lo; e < hi; ++e) {
                    const int64_t j = e / n;
                    const T v = use_prev ? (W[e] - a[j] * Qcur[e] - b[j] * Qprev[e])
                                         : (W[e] - a[j] * Qcur[e]);
                    W[e] = v; loc[j] += v * v;
                }
            }
        }
        for (int64_t j = 0; j < ncols; ++j) {
            T sum = (T)0;
            for (int tt = 0; tt < cp.n_threads; ++tt) sum += partials[(int64_t)tt * stride + j];
            sumsq[j] = sum;
        }
    }

    /// W_j *= scale[j]; a zero scale leaves that column untouched (breakdown).
    static void panel_scale(int64_t n, int64_t ncols, T* W, const T* scale, int nthreads) {
        if (ncols <= 0 || n <= 0) return;
        const ChunkPlan cp = chunk_plan(n, ncols, nthreads);
        const int64_t total = n * ncols;
#pragma omp parallel for num_threads(cp.n_threads) schedule(static)
        for (int64_t c = 0; c < cp.n_chunks; ++c) {
            int64_t lo, hi; chunk_range(total, cp.n_chunks, c, lo, hi);
            for (int64_t e = lo; e < hi; ++e) {
                const T sj = scale[e / n];
                if (sj != (T)0) W[e] *= sj;
            }
        }
    }

    // ------------------------------------------------------------------
    /// Gauss value U_t = ‖b‖²·e₁ᵀ f(T_t) e₁ for column `col` at depth t,
    /// from the stored alpha/beta history (copies; stevd destroys its input).
    template <std::invocable<T> F>
    void evaluate_gauss(F f, int64_t col, int64_t t, int64_t d, T& U) {
        T* base = slot_ws(col);
        T* a_c  = base;
        T* b_c  = a_c + ws_depth;
        T* Z    = b_c + ws_depth;
        blas::copy(t, alpha + col * d, 1, a_c, 1);
        if (t > 1) blas::copy(t - 1, beta + col * d, 1, b_c, 1);
        U = quad_e1(f, t, a_c, b_c, Z) * normb[col] * normb[col];
    }

    /// Gauss U_t and Gauss-Radau L_t for column `col` at depth t ≥ 2. The
    /// Radau tridiagonal T̂_t differs from T_t only in its corner entry
    /// α̂_t = β_{t−1}²/d_{t−1}, saved exactly at this depth's pivot update
    /// (radau_corner). Algebraically α̂_t == α_t − d_t by the pivot recurrence
    /// d_t = α_t − β_{t−1}²/d_{t−1}, but that subtraction cancels (relative
    /// error ~ eps·α_t/α̂_t), so the saved value is used instead.
    template <std::invocable<T> F>
    void evaluate_pair(F f, int64_t col, int64_t t, int64_t d, T& U, T& L) {
        T* base = slot_ws(col);
        T* a_c  = base;
        T* b_c  = a_c + ws_depth;
        T* Z    = b_c + ws_depth;
        const T nb2 = normb[col] * normb[col];
        // Gauss.
        blas::copy(t, alpha + col * d, 1, a_c, 1);
        blas::copy(t - 1, beta + col * d, 1, b_c, 1);
        U = quad_e1(f, t, a_c, b_c, Z) * nb2;
        // Radau: same T_t but corner α̂_t (pins a node at 0).
        blas::copy(t, alpha + col * d, 1, a_c, 1);
        blas::copy(t - 1, beta + col * d, 1, b_c, 1);
        a_c[t - 1] = radau_corner[col];
        L = quad_e1(f, t, a_c, b_c, Z) * nb2;
    }

    /// e₁ᵀ f(T) e₁ for the t×t tridiagonal (diag a, subdiag b), via stevd.
    /// a and b are destroyed. Ritz values clamped to ≥ 0 before f (A ⪰ 0 by
    /// assumption; the Radau node sits at 0 ± roundoff).
    template <std::invocable<T> F>
    T quad_e1(F f, int64_t t, T* a, T* b, T* Z) {
        // Nonzero info means stevd did not converge and a/Z hold garbage;
        // fail loudly rather than certify against it.
        const int64_t info = lapack::stevd(lapack::Job::Vec, t, a, b, Z, t);
        if (info != 0)
            throw std::runtime_error("LanczosQFA: stevd failed at depth "
                + std::to_string(t) + " (info = " + std::to_string(info) + ").");
        T acc = (T)0;
        for (int64_t i = 0; i < t; ++i) {
            const T z0 = Z[i * t + 0];
            acc += f(std::max(a[i], (T)0)) * z0 * z0;
        }
        return acc;
    }

    /// Certificate cadence. Checking EVERY step costs two O(t^3) eigensolves per
    /// active column per step, i.e. O(s*d^4) over a run - ~1e15 flops at the
    /// d ~ 2200 the hard cells reach, which dominates everything else. The
    /// default check_every == 1 uses a geometric ladder instead: every depth
    /// through t = 9, then 12, 18, 27, 42, 63, 93, 141, ... (~1.5x steps),
    /// which is O(s*d^3) while overshooting the true stopping depth by at most
    /// ~1.5x, since the bracket closes monotonically. check_every > 1 forces a
    /// plain fixed stride - note the semantics inversion around the default:
    /// a fixed stride (even check_every = 2) checks MORE often than the ladder
    /// once t > 9, not less. Single-sourced in util::qfa_check_due, which is
    /// the authoritative implementation.
    bool due_for_check(int64_t t) const {
        return util::qfa_check_due(t, check_every);
    }

    /// Grow the eigensolve scratch so every slot holds a depth-t evaluation.
    /// Must be called OUTSIDE any parallel region: it can re-map `workspace`,
    /// and slot_ws() uses ws_depth as a uniform stride, so all evaluations in
    /// one parallel sweep must share one ws_depth (they do: each sweep
    /// evaluates at a single depth t, and ws_depth only grows). Contents need
    /// no preservation across the re-map - every evaluation copies fresh
    /// alpha/beta histories in.
    void ensure_eval_ws(int64_t t, int64_t s) {
        if (t > ws_depth) {
            ws_depth = t;
            util::upsize(workspace, workspace_sz, s * (ws_depth * ws_depth + 2 * ws_depth));
        }
    }

    /// Eigensolve scratch for the column `col`. Indexed by COLUMN, not thread
    /// id: every caller sits in a `for (j < act)` loop where col = col_of_slot[j]
    /// is distinct per iteration and col < s, so slots never collide, and the
    /// buffer needs s slots rather than nthreads. The slot stride derives from
    /// ws_depth (the largest depth evaluated so far this call, or the cap d on
    /// a fixed-depth run), never from the cap alone; see ensure_eval_ws.
    T* slot_ws(int64_t col) {
        return workspace + col * (ws_depth * ws_depth + 2 * ws_depth);
    }
};


} // end namespace RandLAPACK
