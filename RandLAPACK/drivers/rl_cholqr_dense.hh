#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_exceptions.hh"
#include "../comps/rl_cholqr.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <limits>

namespace RandLAPACK {

/// Dense (non-LinOp) entry points for the CholQR family.
///
/// These drivers take a raw column-major buffer instead of a LinearOperator, for
/// callers that work through the regular BLAS API and do not want to build an
/// operator first. They run the SAME numerics as their LinOp counterparts in
/// `rl_cholqr_linops.hh` and `rl_scholqr3_linops.hh`: each wraps its input in a
/// `linops::DenseLinOp` and delegates to the shared `cholqr_iterate` engine, so
/// the pass count, the shift policy, and the adaptive-shift retry are inherited
/// rather than reimplemented. The only thing that differs is the interface.
///
/// The three members of the family are distinguished exactly as in the engine:
///   CholQR_dense    num_iters = 1, both shifts 0 (unshifted first attempt)
///   CholQR2_dense   num_iters = 2, both shifts 0
///   sCholQR3_dense  num_iters = 3, iter 1 shifted by eps, iters 2-3 unshifted
///
/// A is not modified. R is the (upper-triangular) output factor. When a Q buffer
/// is supplied, Q = A * R^{-1} is materialized after the factorization and is
/// excluded from the reported timing total, matching the LinOp drivers' test mode.

namespace detail {

/// Shared body for the three dense drivers. Wraps A in a DenseLinOp and runs
/// `num_iters` passes of the CholQR engine.
///
/// @param[in] m        Rows of A. Must be >= 0.
/// @param[in] n        Columns of A. Must be >= 0 and <= m.
/// @param[in] A        Column-major input buffer, not modified.
/// @param[in] lda      Leading dimension of A, must be >= m.
/// @param[out] R       Output factor, n by n, column-major.
/// @param[in] ldr      Leading dimension of R, must be >= n.
/// @param[out] Q       Optional. If non-null, receives Q = A * R^{-1} (m by n).
/// @param[in] ldq      Leading dimension of Q. Must be >= m. Ignored when Q is null.
/// @param[out] applied_shifts  Optional, length num_iters: per-pass absolute shift
///                     the successful potrf carried (0 = unshifted).
/// @param[out] gram_traces     Optional, length num_iters: per-pass Gram trace.
/// @param[out] q_mat_us Optional. When Q is materialized and timing is on, receives
///                     the wall-clock (us) spent materializing Q, so the caller can
///                     exclude it from the reported total exactly as the LinOp
///                     drivers do. Set to 0 when Q is null or timing is off.
/// @return 0 on success, or the 1-based index of the pass whose Cholesky failed.
template <typename T>
int cholqr_dense_body(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* R,
    int64_t ldr,
    T* Q,
    int64_t ldq,
    int64_t block_size,
    int num_iters,
    T shift_iter1,
    T shift_iter_rest,
    int max_retries,
    T shift_growth,
    bool timing,
    long* iter_times,
    int* n_retries_total,
    T* applied_shifts,
    T* gram_traces,
    long* q_mat_us
) {
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
    randlapack_require(n <= m) << "n=" << n << " must be <= m=" << m << " (CholQR needs a tall or square input)";
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (column-major input)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n;
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
    randlapack_require(!(Q != nullptr && ldq < m)) << "ldq=" << ldq << " < m=" << m << " (column-major Q output)";

    if (q_mat_us) *q_mat_us = 0;

    linops::DenseLinOp<T> A_op(m, n, A, lda, Layout::ColMajor);

    int info = cholqr_iterate<T, linops::DenseLinOp<T>>(
        A_op, R, ldr, block_size, num_iters,
        shift_iter1, shift_iter_rest,
        max_retries, shift_growth, timing,
        iter_times, n_retries_total, applied_shifts, gram_traces);
    if (info != 0)
        return info;

    if (Q != nullptr) {
        int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
        if (timing && q_mat_us) {
            std::chrono::steady_clock::time_point qt0 = std::chrono::steady_clock::now();
            RandLAPACK::materialize_Q_from_R(A_op, R, ldr, m, n, b_eff, Q, ldq);
            std::chrono::steady_clock::time_point qt1 = std::chrono::steady_clock::now();
            *q_mat_us = std::chrono::duration_cast<std::chrono::microseconds>(qt1 - qt0).count();
        } else {
            RandLAPACK::materialize_Q_from_R(A_op, R, ldr, m, n, b_eff, Q, ldq);
        }
    }
    return 0;
}

} // end namespace detail


/// Plain CholeskyQR on a dense column-major buffer.
/// One unpreconditioned pass, unshifted first attempt.
template <typename T>
class CholQR_dense {
    public:

        bool timing;
        int64_t block_size;

        // Adaptive-shift safety net: the first attempt is unshifted, and only on
        // potrf breakdown does the primitive seed the shift and grow it. A negative
        // max_retries means unbounded, matching the LinOp drivers.
        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;   ///< shift retries used on the last call (0 = clean)
        /// Per-pass shift record from the last call: the absolute diagonal shift the
        /// successful potrf carried (0 = unshifted) and that pass's Gram trace.
        T chol_applied_shifts[1] = {T(0)};
        T chol_gram_traces[1]    = {T(0)};

        // 6 entries: alloc, fwd, adj, chol, rest, total
        std::vector<long> times;
        long total_us() const { return times.empty() ? -1L : times.back(); }

        CholQR_dense(
            bool time_subroutines
        ) {
            timing = time_subroutines;
            block_size = kDefaultGramBlockSize;
            max_retries = -1;
            shift_growth = T(10);
        }

        int call(
            int64_t m,
            int64_t n,
            const T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            T* Q = nullptr,
            int64_t ldq = 0
        ) {
            std::chrono::steady_clock::time_point total_t_start, total_t_stop;
            if (this->timing) total_t_start = std::chrono::steady_clock::now();

            long it[5] = {0};
            long q_mat_us = 0;
            int info = detail::cholqr_dense_body<T>(
                m, n, A, lda, R, ldr, Q, ldq,
                this->block_size, /*num_iters=*/1,
                /*shift_iter1=*/T(0), /*shift_iter_rest=*/T(0),
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries,
                this->chol_applied_shifts, this->chol_gram_traces,
                this->timing ? &q_mat_us : nullptr);
            if (info != 0) return info;

            if (this->timing) {
                total_t_stop = std::chrono::steady_clock::now();
                long total_dur = std::chrono::duration_cast<std::chrono::microseconds>(total_t_stop - total_t_start).count() - q_mat_us;
                long fwd = it[0], adj = it[1], chol = it[3];
                long rest_dur = total_dur - (fwd + adj + chol);
                this->times = {0L, fwd, adj, chol, rest_dur, total_dur};
            }
            return 0;
        }
};


/// CholeskyQR2 on a dense column-major buffer.
/// Two passes, both starting unshifted; the retry rescues a non-PD Gram.
template <typename T>
class CholQR2_dense {
    public:

        bool timing;
        int64_t block_size;

        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;
        /// Per-pass shift record from the last call (pass 1, pass 2): absolute shift
        /// the successful potrf carried (0 = unshifted) and that pass's Gram trace.
        T chol_applied_shifts[2] = {T(0), T(0)};
        T chol_gram_traces[2]    = {T(0), T(0)};

        // 11 entries: alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, total
        //   upd1 is 0 by convention (iter 1 has no R-update step). Matches
        //   CholQR2_linops's layout exactly (the always-zero iter-1 gemm slot is
        //   dropped, not carried as a 12th entry).
        std::vector<long> times;
        long total_us() const { return times.empty() ? -1L : times.back(); }

        CholQR2_dense(
            bool time_subroutines
        ) {
            timing = time_subroutines;
            block_size = kDefaultGramBlockSize;
            max_retries = -1;
            shift_growth = T(10);
        }

        int call(
            int64_t m,
            int64_t n,
            const T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            T* Q = nullptr,
            int64_t ldq = 0
        ) {
            std::chrono::steady_clock::time_point total_t_start, total_t_stop;
            if (this->timing) total_t_start = std::chrono::steady_clock::now();

            long it[10] = {0};
            long q_mat_us = 0;
            int info = detail::cholqr_dense_body<T>(
                m, n, A, lda, R, ldr, Q, ldq,
                this->block_size, /*num_iters=*/2,
                /*shift_iter1=*/T(0), /*shift_iter_rest=*/T(0),
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries,
                this->chol_applied_shifts, this->chol_gram_traces,
                this->timing ? &q_mat_us : nullptr);
            if (info != 0) return info;

            if (this->timing) {
                total_t_stop = std::chrono::steady_clock::now();
                long total_dur = std::chrono::duration_cast<std::chrono::microseconds>(total_t_stop - total_t_start).count() - q_mat_us;
                // it = [fwd1,adj1,gemm1=0,chol1,upd1, fwd2,adj2,gemm2,chol2,upd2].
                // Drop the always-zero gemm1 slot (it[2]) to match CholQR2_linops.
                this->times = {0L, it[0], it[1], it[3], it[4],
                                    it[5], it[6], it[7], it[8], it[9],
                                    total_dur};
            }
            return 0;
        }
};


/// Shifted CholeskyQR3 on a dense column-major buffer.
/// Three passes; iter 1 carries the shift (default: the paper's 11*n*eps, or
/// eps via RANDLAPACK_SCHOLQR3_SHIFT=eps), iters 2 and 3 are unshifted
/// (Fukaya's prescription). This mirrors sCholQR3_linops exactly, including its
/// RANDLAPACK_SCHOLQR3_SHIFT env knob (shared via scholqr3_eps_shift() in
/// comps/rl_cholqr.hh).
template <typename T>
class sCholQR3_dense {
    public:

        bool timing;
        int64_t block_size;

        T shift_factor_iter1;
        T shift_factor_iter23;

        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;
        /// Per-pass shift record from the last call (passes 1-3): absolute shift the
        /// successful potrf carried (0 = unshifted) and that pass's Gram trace.
        T chol_applied_shifts[3] = {T(0), T(0), T(0)};
        T chol_gram_traces[3]    = {T(0), T(0), T(0)};

        // Timing breakdown (18 entries; matches sCholQR3_linops exactly):
        // [0]  alloc
        // [1]  fwd1     [2]  adj1    [3]  chol1   [4]  upd1
        // [5]  fwd2     [6]  adj2    [7]  gemm2   [8]  chol2   [9]  upd2
        // [10] fwd3     [11] adj3    [12] gemm3   [13] chol3   [14] upd3
        // [15] q_mat    [16] rest    [17] total
        std::vector<long> times;
        long total_us() const { return times.empty() ? -1L : times.back(); }

        sCholQR3_dense(
            bool time_subroutines
        ) {
            timing = time_subroutines;
            block_size = kDefaultGramBlockSize;
            shift_factor_iter1  = T(-1);  // < 0: resolve default (11*n*eps, or eps via env) at call time
            shift_factor_iter23 = T(0);
            max_retries = -1;
            shift_growth = T(10);
        }

        int call(
            int64_t m,
            int64_t n,
            const T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            T* Q = nullptr,
            int64_t ldq = 0
        ) {
            std::chrono::steady_clock::time_point total_t_start, total_t_stop;
            if (this->timing) total_t_start = std::chrono::steady_clock::now();

            // First-pass shift defaults to the paper's s = 11*eps*n*trace(G)
            // (FukayaEtAl2020, c = 11); RANDLAPACK_SCHOLQR3_SHIFT=eps selects the
            // legacy eps*trace(G). A caller-set shift_factor_iter1 >= 0 wins.
            // Same knob resolution as sCholQR3_linops.
            const T eps_T = std::numeric_limits<T>::epsilon();
            const T sf1 = (this->shift_factor_iter1 >= T(0))
                        ? this->shift_factor_iter1
                        : (scholqr3_eps_shift() ? eps_T : T(11) * T(n) * eps_T);

            long it[15] = {0};
            long q_mat_us = 0;
            int info = detail::cholqr_dense_body<T>(
                m, n, A, lda, R, ldr, Q, ldq,
                this->block_size, /*num_iters=*/3,
                sf1, this->shift_factor_iter23,
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries,
                this->chol_applied_shifts, this->chol_gram_traces,
                this->timing ? &q_mat_us : nullptr);
            if (info != 0) return info;   // 1/2/3 = the pass that failed

            if (this->timing) {
                total_t_stop = std::chrono::steady_clock::now();
                long total_dur = std::chrono::duration_cast<std::chrono::microseconds>(total_t_stop - total_t_start).count() - q_mat_us;
                long iters_sum = 0;
                for (int i = 0; i < 15; ++i) iters_sum += it[i];
                long rest_dur = total_dur - iters_sum;
                this->times = {0L,
                    it[0], it[1], it[3], it[4],             // fwd1, adj1, chol1, upd1
                    it[5], it[6], it[7], it[8], it[9],      // fwd2, adj2, gemm2, chol2, upd2
                    it[10], it[11], it[12], it[13], it[14], // fwd3, adj3, gemm3, chol3, upd3
                    q_mat_us, rest_dur, total_dur};
            }
            return 0;
        }
};

} // end namespace RandLAPACK
