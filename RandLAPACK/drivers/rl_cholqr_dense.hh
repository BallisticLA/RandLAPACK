#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "../comps/rl_cholqr.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <limits>

using namespace std::chrono;

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
/// excluded from the reported timing, matching the LinOp drivers' test mode.

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
/// @param[in] ldq      Leading dimension of Q. Ignored when Q is null.
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
    int* n_retries_total
) {
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
    randlapack_require(n <= m) << "n=" << n << " must be <= m=" << m << " (CholQR needs a tall or square input)";
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (column-major input)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n;
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
    randlapack_require(!(Q != nullptr && ldq < m)) << "ldq=" << ldq << " < m=" << m << " (column-major Q output)";

    linops::DenseLinOp<T> A_op(m, n, A, lda, Layout::ColMajor);

    int info = cholqr_iterate<T, linops::DenseLinOp<T>>(
        A_op, R, ldr, block_size, num_iters,
        shift_iter1, shift_iter_rest,
        max_retries, shift_growth, timing,
        iter_times, n_retries_total);
    if (info != 0)
        return info;

    if (Q != nullptr) {
        int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
        RandLAPACK::materialize_Q_from_R(A_op, R, ldr, m, n, b_eff, Q);
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

        // 6 entries: alloc, fwd, adj, chol, rest, total
        std::vector<long> times;
        long total_us() const { return times.empty() ? -1L : times.back(); }

        CholQR_dense(
            bool time_subroutines
        ) {
            timing = time_subroutines;
            block_size = 0;
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
            steady_clock::time_point total_t_start, total_t_stop;
            if (this->timing) total_t_start = steady_clock::now();

            long it[5] = {0};
            int info = detail::cholqr_dense_body<T>(
                m, n, A, lda, R, ldr, Q, ldq,
                this->block_size, /*num_iters=*/1,
                /*shift_iter1=*/T(0), /*shift_iter_rest=*/T(0),
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries);
            if (info != 0) return info;

            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
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

        // 11 entries: alloc, then [fwd, adj, gemm, chol, upd] per pass, then total
        std::vector<long> times;
        long total_us() const { return times.empty() ? -1L : times.back(); }

        CholQR2_dense(
            bool time_subroutines
        ) {
            timing = time_subroutines;
            block_size = 0;
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
            steady_clock::time_point total_t_start, total_t_stop;
            if (this->timing) total_t_start = steady_clock::now();

            long it[10] = {0};
            int info = detail::cholqr_dense_body<T>(
                m, n, A, lda, R, ldr, Q, ldq,
                this->block_size, /*num_iters=*/2,
                /*shift_iter1=*/T(0), /*shift_iter_rest=*/T(0),
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries);
            if (info != 0) return info;

            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                this->times.assign(it, it + 10);
                this->times.insert(this->times.begin(), 0L);
                this->times.push_back(total_dur);
            }
            return 0;
        }
};


/// Shifted CholeskyQR3 on a dense column-major buffer.
/// Three passes; iter 1 carries the eps shift, iters 2 and 3 are unshifted
/// (Fukaya's prescription). This mirrors sCholQR3_linops exactly.
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

        // 17 entries: alloc, then [fwd, adj, gemm, chol, upd] per pass, then total
        std::vector<long> times;
        long total_us() const { return times.empty() ? -1L : times.back(); }

        sCholQR3_dense(
            bool time_subroutines
        ) {
            timing = time_subroutines;
            block_size = 0;
            shift_factor_iter1  = std::numeric_limits<T>::epsilon();
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
            steady_clock::time_point total_t_start, total_t_stop;
            if (this->timing) total_t_start = steady_clock::now();

            long it[15] = {0};
            int info = detail::cholqr_dense_body<T>(
                m, n, A, lda, R, ldr, Q, ldq,
                this->block_size, /*num_iters=*/3,
                this->shift_factor_iter1, this->shift_factor_iter23,
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr, &this->n_chol_retries);
            if (info != 0) return info;   // 1/2/3 = the pass that failed

            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                this->times.assign(it, it + 15);
                this->times.insert(this->times.begin(), 0L);
                this->times.push_back(total_dur);
            }
            return 0;
        }
};

} // end namespace RandLAPACK
