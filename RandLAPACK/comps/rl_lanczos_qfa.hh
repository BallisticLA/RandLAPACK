#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RandLAPACK {


/// Scalar (per-column) Lanczos-QFA: the quadratic forms out[j] = bⱼᵀ f(A) bⱼ
/// = ‖bⱼ‖²·e₁ᵀ f(T_{t,j}) e₁, one independent Krylov subspace per column of B,
/// returned as a length-s vector (NOT the n×s block f(A)B and NOT the s×s
/// block quadratic form — the off-diagonal entries of Bᵀf(A)B are not
/// computable from per-column tridiagonals). By the Gauss-quadrature identity
/// bᵀ·LanczosFA(A, f, b) = Lanczos-QFA(A, f, b), so the funNyström++ Phase-2
/// term tr(Ω₂ᵀ f(A) Ω₂) = Σⱼ out[j] is formed without materializing f(A)·Ω₂.
///
/// Basis-free by construction: the reconstruction reads only each column's
/// tridiagonal (α, β) and norm, never the Krylov basis, so the recurrence
/// keeps a rolling window of three n-vectors per column (q_{t−1}, q_t, and the
/// matvec target) — O(n·s) working memory instead of LanczosFA's O(n·d·s)
/// stored basis. Consequently there is NO reorthogonalization option: this
/// class runs vanilla Lanczos (Gauss-quadrature values tolerate orthogonality
/// loss far better than eigenvalue Lanczos, Paige-Greenbaum; the certificate
/// below independently monitors positive-definiteness via its LDLᵀ pivots).
/// Requires A ⪰ 0 and f finite at 0 (Ritz values are clamped to ≥ 0 before f;
/// a raw log(x) would produce -inf at the Radau node pinned at 0 — use the
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
/// delay window, no depth floor, and no plateau risk: the column stops at the
/// first depth t with |U_t − L_t| ≤ adaptive_rtol·|U_t|, and adaptive_rtol is
/// a CERTIFIED relative error, not a target scale. The corner entry needs no
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
/// eigensolves (stevd) per active column — zero matvecs, and negligible next
/// to an n-length matvec for the depths this method runs at (tens).
///
/// @tparam T  Floating-point scalar type.
template <typename T>
class LanczosQFA {
public:
    // ---- certified adaptive stopping controls ------------------------------
    bool    adaptive      = false;   ///< per-column Gauss-Radau certified stop
    T       adaptive_rtol = (T)1e-2; ///< certified relative-error tolerance
    int64_t check_every   = 1;       ///< certificate stride (checks at t ≥ 2)

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
    // col_of_slot: original column held by each active slot (compaction map).
    T*       qbuf        = nullptr; int64_t qbuf_sz        = 0;
    T*       alpha       = nullptr; int64_t alpha_sz       = 0;
    T*       beta        = nullptr; int64_t beta_sz        = 0;
    T*       normb       = nullptr; int64_t normb_sz       = 0;
    T*       ldl_piv     = nullptr; int64_t ldl_piv_sz     = 0;
    uint8_t* cert_ok     = nullptr; int64_t cert_ok_sz     = 0; ///< pivot chain still PD
    int64_t* col_of_slot = nullptr; int64_t col_of_slot_sz = 0;
    T*       workspace   = nullptr; int64_t workspace_sz   = 0; ///< per-thread stevd scratch

    // Profiling, matching the LanczosFA-family surface. Slots: {matvec,
    // run_lanczos, apply, rest, total} µs; `apply` is all certificate checks
    // plus the final Gauss evaluations, `run_lanczos` is the recurrence net of
    // certificate time, so the slots stay comparable across oracles.
    bool timing = false;
    std::vector<long> times;
    long _t_matvec_us = 0;

    LanczosQFA()                             = default;
    LanczosQFA(const LanczosQFA&)            = delete;
    LanczosQFA& operator=(const LanczosQFA&) = delete;

    ~LanczosQFA() {
        delete[] qbuf; delete[] alpha; delete[] beta; delete[] normb;
        delete[] ldl_piv; delete[] cert_ok; delete[] col_of_slot;
        delete[] t_used; delete[] gauss_val; delete[] radau_val;
        delete[] certified; delete[] workspace;
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

        // [setup] buffers + per-column output state.
        util::upsize(qbuf,        qbuf_sz,        3 * n * s);
        util::upsize(alpha,       alpha_sz,       d * s);
        util::upsize(beta,        beta_sz,        d * s);
        util::upsize(normb,       normb_sz,       s);
        util::upsize(ldl_piv,     ldl_piv_sz,     s);
        util::upsize(cert_ok,     cert_ok_sz,     s);
        util::upsize(col_of_slot, col_of_slot_sz, s);
        util::upsize(t_used,      t_used_sz,      s);
        util::upsize(gauss_val,   gauss_val_sz,   s);
        util::upsize(radau_val,   radau_val_sz,   s);
        util::upsize(certified,   certified_sz,   s);
        // Per-thread scratch for the tridiagonal eigensolves: alpha copy (d) +
        // beta copy (d) + eigenvector matrix Z (d*d).
        const int64_t ws_per_thread = d * d + 2 * d;
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        util::upsize(workspace, workspace_sz, (int64_t)nthreads * ws_per_thread);

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
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                const T a = blas::dot(n, w + j * n, 1, q_cur + j * n, 1);
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
                        const T piv = a - corner;
                        ldl_piv[col] = piv;   // pivot THROUGH depth t; corner used at depth t reads the pre-update value below
                        if (piv <= (T)0) cert_ok[col] = 0;
                    }
                }
            }

            // [certificate] at depth t ≥ 2 (stride check_every): per active
            // column, Gauss U_t vs Gauss-Radau L_t; certify when the bracket
            // closes. The Radau corner α̂_t = β_{t−1}²/d_{t−1} needs the pivot
            // BEFORE this step's update, but no history is stored: the pivot
            // recurrence d_t = α_t − α̂_t gives α̂_t = α_t − d_t exactly, which
            // is how evaluate_pair() reconstructs it from the updated pivot.
            if (adaptive && t >= 2 && (t % check_every == 0)) {
                if (timing) c0 = steady_clock::now();
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                blas::axpy(n, -alpha[col * d + (t - 1)], q_cur + j * n, 1, w + j * n, 1);
                if (t > 1)
                    blas::axpy(n, -beta[col * d + (t - 2)], q_prev + j * n, 1, w + j * n, 1);
                T nrm = blas::nrm2(n, w + j * n, 1);
                beta[col * d + (t - 1)] = nrm;
                if (nrm > (T)0) {
                    blas::scal(n, (T)1.0 / nrm, w + j * n, 1);
                } else {
                    // Breakdown: certify with the exact depth-t value.
                    T U;
                    evaluate_gauss(f, col, t, d, U);
                    certified[col] = 1; t_used[col] = t;
                    gauss_val[col] = U; radau_val[col] = U;
                }
            }
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
        // fixed-depth run when !adaptive): evaluate the depth-t Gauss value,
        // uncertified in adaptive mode. In adaptive mode the Radau companion
        // is also evaluated when the pivot chain allows, so radau_val reports
        // the (unclosed) bracket at the cap for diagnostics.
        if (act > 0) {
            if (timing) c0 = steady_clock::now();
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < act; ++j) {
                const int64_t col = col_of_slot[j];
                T U, L;
                if (adaptive && t >= 2 && cert_ok[col]) {
                    evaluate_pair(f, col, t, d, U, L);
                    radau_val[col] = L;
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
            long rest_us    = total_us - lanczos_us - apply_us;
            times = {_t_matvec_us, lanczos_us, apply_us, rest_us, total_us};
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

    // ------------------------------------------------------------------
    /// Gauss value U_t = ‖b‖²·e₁ᵀ f(T_t) e₁ for column `col` at depth t,
    /// from the stored alpha/beta history (copies; stevd destroys its input).
    template <std::invocable<T> F>
    void evaluate_gauss(F f, int64_t col, int64_t t, int64_t d, T& U) {
        T* base = thread_ws(d);
        T* a_c  = base;
        T* b_c  = a_c + d;
        T* Z    = b_c + d;
        blas::copy(t, alpha + col * d, 1, a_c, 1);
        if (t > 1) blas::copy(t - 1, beta + col * d, 1, b_c, 1);
        U = quad_e1(f, t, a_c, b_c, Z) * normb[col] * normb[col];
    }

    /// Gauss U_t and Gauss-Radau L_t for column `col` at depth t ≥ 2. The
    /// Radau tridiagonal T̂_t differs from T_t only in its corner entry
    /// α̂_t = β_{t−1}²/d_{t−1} = α_t − d_t (from the pivot recurrence
    /// d_t = α_t − β_{t−1}²/d_{t−1}), so it is recovered O(1) from the
    /// maintained pivot without storing the pivot history.
    template <std::invocable<T> F>
    void evaluate_pair(F f, int64_t col, int64_t t, int64_t d, T& U, T& L) {
        T* base = thread_ws(d);
        T* a_c  = base;
        T* b_c  = a_c + d;
        T* Z    = b_c + d;
        const T nb2 = normb[col] * normb[col];
        // Gauss.
        blas::copy(t, alpha + col * d, 1, a_c, 1);
        blas::copy(t - 1, beta + col * d, 1, b_c, 1);
        U = quad_e1(f, t, a_c, b_c, Z) * nb2;
        // Radau: same T_t but corner α̂_t = α_t − d_t (pins a node at 0).
        blas::copy(t, alpha + col * d, 1, a_c, 1);
        blas::copy(t - 1, beta + col * d, 1, b_c, 1);
        a_c[t - 1] = alpha[col * d + (t - 1)] - ldl_piv[col];
        L = quad_e1(f, t, a_c, b_c, Z) * nb2;
    }

    /// e₁ᵀ f(T) e₁ for the t×t tridiagonal (diag a, subdiag b), via stevd.
    /// a and b are destroyed. Ritz values clamped to ≥ 0 before f (A ⪰ 0 by
    /// assumption; the Radau node sits at 0 ± roundoff).
    template <std::invocable<T> F>
    T quad_e1(F f, int64_t t, T* a, T* b, T* Z) {
        lapack::stevd(lapack::Job::Vec, t, a, b, Z, t);
        T acc = (T)0;
        for (int64_t i = 0; i < t; ++i) {
            const T z0 = Z[i * t + 0];
            acc += f(std::max(a[i], (T)0)) * z0 * z0;
        }
        return acc;
    }

    T* thread_ws(int64_t d) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        return workspace + (int64_t)tid * (d * d + 2 * d);
    }
};


} // end namespace RandLAPACK
