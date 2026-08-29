#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "../comps/rl_cholqr.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <string>

// scholqr3_eps_shift() (the RANDLAPACK_SCHOLQR3_SHIFT env knob) now lives
// in comps/rl_cholqr.hh, shared with the dense sCholQR3 driver.

using namespace std::chrono;

namespace RandLAPACK {

/// Shifted Cholesky QR3 for abstract linear operators (fully-blocked variant).
///
/// Algorithm 3 from the collaborator's spec:
///   iter 1: cholqr_primitive(A) with shift s = eps * ||A||_F^2  -> R_1
///   iter i = 2, 3: cholqr_primitive(A, R_{i-1}, TRSM_IDENTITY)  -> R_i
///   return R_3
///
/// Peak memory O(n^2 + (m+n)*b_eff): never materializes the m × n operator product
/// during the QR iterations. If test_mode is enabled, Q = A * R^{-1} is materialized
/// at the end (m × n, outside the timing region).
///
/// Reference: Shifted Cholesky QR from Fukaya et al. (SISC, 2020).
///
template <typename T>
class sCholQR3_linops {
    public:

        bool timing;
        bool test_mode;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // Timing breakdown (18 entries; layout preserved for matlab plotters):
        // [0]  alloc
        // [1]  fwd1     [2]  adj1    [3]  chol1   [4]  upd1
        // [5]  fwd2     [6]  adj2    [7]  gemm2   [8]  chol2   [9]  upd2
        // [10] fwd3     [11] adj3    [12] gemm3   [13] chol3   [14] upd3
        // [15] q_mat    [16] rest    [17] total
        std::vector<long> times;
        /// Total measured wall-clock (microseconds) of the last call(), or -1 if timing
        /// was off. Every driver in this family packs the total as the LAST times[] entry,
        /// but the entry COUNT differs per driver (6 / 11 / 15 / 18). Callers used to hard-
        /// code that index (times[5], times[10], times[14], times[17]), so adding or
        /// removing one slot silently wrote the wrong number into every CSV with no compile
        /// error. Read the total through here instead.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        int64_t block_size;

        // Adaptive-shift policy (see cholqr_primitive). Shift s = factor * trace(G).
        //
        // iter 1: shifted. shift_factor_iter1 < 0 (the default) resolves at call
        //   time to the paper's prescription 11 * n * eps (FukayaEtAl2020, c = 11),
        //   or to plain eps when RANDLAPACK_SCHOLQR3_SHIFT=eps is set. The eps
        //   variant is a smaller shift kept for A/B campaigns: a shift far above
        //   σ_min²(A) makes the iter-2 Gram rank-deficient in principle, though
        //   with unshifted refinement passes plus the adaptive retry the paper
        //   shift has measured clean on the FEM2 campaigns. A caller-set value
        //   >= 0 is used verbatim.
        //
        // iters 2-3: UNSHIFTED (shift_factor_iter23 = 0). This is the defining
        //   feature of Fukaya shifted-CholeskyQR3: the refinement passes are plain
        //   CholeskyQR2, which is what drives orthogonality down to machine level.
        //   A persistent eps shift on these passes (the old setting) never gets
        //   removed: it floors orth at ~2n*eps (≈1.8e-12 in double) and, in single,
        //   over-regularizes R into a useless preconditioner (CG stalls, 100s of
        //   inner iters). The adaptive retry below still shifts a refinement pass
        //   *only* if its potrf genuinely fails.
        //
        // The retry loop bumps shift × 10 if potrf bails, unboundedly
        // (max_retries = -1) until the Gram is PD.
        T   shift_factor_iter1;
        T   shift_factor_iter23;
        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;   ///< shift retries used on the last call (0 = clean)
        /// Per-pass shift record from the last call (passes 1-3): absolute shift the
        /// successful potrf carried (0 = unshifted) and that pass's Gram trace.
        T chol_applied_shifts[3] = {T(0), T(0), T(0)};
        T chol_gram_traces[3]    = {T(0), T(0), T(0)};

        sCholQR3_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            (void)ep;   // kept in the signature for call-site compatibility; unused
            block_size = kDefaultGramBlockSize;
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
            shift_factor_iter1  = T(-1);  // < 0: resolve default (11*n*eps, or eps via env) at call time
            shift_factor_iter23 = T(0);   // iters 2-3 unshifted (Fukaya); retry covers genuine non-PD
            max_retries         = -1;     // unbounded retries (no ceiling), consistent with CholQR/CholQR2
            shift_growth        = T(10);
        }

        ~sCholQR3_linops() {
            if (Q != nullptr) {
                delete[] Q;
            }
        }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr
        ) {
            steady_clock::time_point t0, t1, total_t_start, total_t_stop;
            long q_mat_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // sCholQR3 = three cholqr_iterate passes; iter 1 carries the shift,
            // iters 2-3 use shift_factor_iter23.
            //
            // The first-pass shift defaults to the paper's prescription
            // s = 11*eps*n*trace(G) (FukayaEtAl2020, c = 11), resolved here because
            // it needs n. RANDLAPACK_SCHOLQR3_SHIFT=eps selects the legacy smaller
            // s = eps*trace(G) instead; a caller-set shift_factor_iter1 >= 0 wins
            // over both.
            long it[15] = {0};
            const T eps_T = std::numeric_limits<T>::epsilon();
            const T sf1 = (this->shift_factor_iter1 >= T(0))
                        ? this->shift_factor_iter1
                        : (scholqr3_eps_shift() ? eps_T : T(11) * T(n) * eps_T);
            int info = cholqr_iterate<T, GLO>(
                A, R, ldr, this->block_size, /*num_iters=*/3,
                sf1, this->shift_factor_iter23,
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries,
                this->chol_applied_shifts, this->chol_gram_traces);
            // 1/2/3 = the 1-based pass that failed (retry exhaustion, singular
            // preconditioner, non-finite shift, or invalid input; see stderr).
            if (info != 0) return info;

            // Test mode: materialize Q = A * R^{-1} (outside timing region).
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();

                T* Q_buf = new T[m * n];
                RandLAPACK::materialize_Q_from_R(A, R, ldr, m, n, b_eff, Q_buf, m);
                this->Q_rows = m;
                this->Q_cols = n;
                // The class owns Q (the destructor frees it), so release any buffer from a
                // previous call() before taking ownership of this one.
                delete[] this->Q;
                this->Q = Q_buf;

                if (this->timing) { t1 = steady_clock::now(); q_mat_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            // ============================================================
            // Finalize timing
            // ============================================================
            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count() - q_mat_dur;
                long iters_sum = 0;
                for (int i = 0; i < 15; ++i) iters_sum += it[i];
                long rest_dur = total_dur - iters_sum;
                // it groups [fwd,adj,gemm,chol,upd] per pass; iter-1 gemm/upd are 0.
                this->times = {0L,
                    it[0], it[1], it[3], it[4],             // fwd1, adj1, chol1, upd1
                    it[5], it[6], it[7], it[8], it[9],      // fwd2, adj2, gemm2, chol2, upd2
                    it[10], it[11], it[12], it[13], it[14], // fwd3, adj3, gemm3, chol3, upd3
                    q_mat_dur, rest_dur, total_dur};
            }

            return 0;
        }
};

/// Non-blocked (basic) sCholQR3: algorithmically identical to sCholQR3_linops with
/// block_size = 0: all three iterations route through cholqr_primitive on the linop
/// (no materialized-Q / dense-syrk shortcut). It exists only as a separate analytic-
/// memory accounting case; the heavy work is the same per-iteration linop Gram.
///
template <typename T>
class sCholQR3_linops_basic {
    public:

        bool timing;
        bool test_mode;

        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // Timing layout (15 entries, kept for matlab CSV-column compatibility):
        // [0]  alloc      [1]  fwd1   [2]  adj1   [3]  chol1   [4]  trsm1=0  [5]  fwd_q=0
        // [6]  syrk2      [7]  chol2  [8]  upd2
        // [9]  syrk3      [10] chol3  [11] upd3
        // [12] q_mat      [13] rest   [14] total
        // (Post-refactor slots 4, 5, 6, 9 stay 0 because the primitives don't expose
        //  syrk vs adj/fwd as separate signals; the heavy lifters are folded into
        //  fwd/adj from blocked_preconditioned_gram and into chol from potrf.)
        std::vector<long> times;
        /// Total measured wall-clock (microseconds) of the last call(), or -1 if timing
        /// was off. Every driver in this family packs the total as the LAST times[] entry,
        /// but the entry COUNT differs per driver (6 / 11 / 15 / 18). Callers used to hard-
        /// code that index (times[5], times[10], times[14], times[17]), so adding or
        /// removing one slot silently wrote the wrong number into every CSV with no compile
        /// error. Read the total through here instead.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        // Adaptive shift policy, shared with sCholQR3_linops.
        T   shift_factor_iter1;
        T   shift_factor_iter23;
        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;   ///< shift retries used on the last call (0 = clean)
        /// Per-pass shift record from the last call (passes 1-3): absolute shift the
        /// successful potrf carried (0 = unshifted) and that pass's Gram trace.
        T chol_applied_shifts[3] = {T(0), T(0), T(0)};
        T chol_gram_traces[3]    = {T(0), T(0), T(0)};

        sCholQR3_linops_basic(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            (void)ep;   // kept in the signature for call-site compatibility; unused
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
            shift_factor_iter1  = T(-1);  // < 0: resolve default (11*n*eps, or eps via env) at call time
            shift_factor_iter23 = T(0);   // iters 2-3 unshifted (Fukaya); retry covers genuine non-PD
            max_retries         = -1;     // unbounded retries (no ceiling), consistent with CholQR/CholQR2
            shift_growth        = T(10);
        }

        ~sCholQR3_linops_basic() {
            if (Q != nullptr) {
                delete[] Q;
            }
        }

        // Non-blocked sCholQR3 expressed via the shared primitives.
        // Algorithmically identical to sCholQR3_linops with block_size=0; the
        // distinction is now purely the analytic-memory accounting (no R_pre /
        // P_prev / Z_buf reuse across iters since the primitives allocate their
        // own G internally per call). Diagnostic prints from cholqr_primitive
        // surface here too.
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr
        ) {
            steady_clock::time_point t0, t1, total_t_start, total_t_stop;
            long q_mat_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = n;   // non-blocked (block_size = 0 → b_eff = n)

            // Non-blocked sCholQR3 = three cholqr_iterate passes with block_size = 0.
            long it[15] = {0};
            // Same shift-default resolution as the blocked variant above.
            const T eps_T = std::numeric_limits<T>::epsilon();
            const T sf1 = (this->shift_factor_iter1 >= T(0))
                        ? this->shift_factor_iter1
                        : (scholqr3_eps_shift() ? eps_T : T(11) * T(n) * eps_T);
            int info = cholqr_iterate<T, GLO>(
                A, R, ldr, /*block_size=*/0, /*num_iters=*/3,
                sf1, this->shift_factor_iter23,
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries,
                this->chol_applied_shifts, this->chol_gram_traces);
            // 1/2/3 = the 1-based pass that failed (retry exhaustion, singular
            // preconditioner, non-finite shift, or invalid input; see stderr).
            if (info != 0) return info;

            // ---- Test mode: materialize Q = A * R^{-1} via blocked linop call ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();
                T* Q_buf = new T[m * n];
                RandLAPACK::materialize_Q_from_R(A, R, ldr, m, n, b_eff, Q_buf, m);
                this->Q_rows = m;
                this->Q_cols = n;
                // The class owns Q (the destructor frees it), so release any buffer from a
                // previous call() before taking ownership of this one.
                delete[] this->Q;
                this->Q = Q_buf;
                if (this->timing) { t1 = steady_clock::now(); q_mat_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count() - q_mat_dur;
                long iters_sum = 0;
                for (int i = 0; i < 15; ++i) iters_sum += it[i];
                long rest_dur = total_dur - iters_sum;
                // Basic layout folds each iter's fwd+adj+gemm into its chol slot.
                long chol2 = it[8] + it[5] + it[6] + it[7];
                long chol3 = it[13] + it[10] + it[11] + it[12];
                this->times = {0L, it[0], it[1], it[3],
                               /*trsm1=*/0L, /*fwd_q=*/0L,
                               /*syrk2=*/0L, chol2, it[9],
                               /*syrk3=*/0L, chol3, it[14],
                               q_mat_dur, rest_dur, total_dur};
            }
            return 0;
        }
};

} // end namespace RandLAPACK
