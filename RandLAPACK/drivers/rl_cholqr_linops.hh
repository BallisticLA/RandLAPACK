#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "../comps/rl_cholqr.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std::chrono;

namespace RandLAPACK {

/// Cholesky QR factorization for abstract linear operators.
///
/// Thin wrapper around `cholqr_primitive` (Algorithm 1 from the collaborator's spec):
///     G = A^T A,  G = R^T R via Cholesky.
///
/// A may be any type satisfying the LinearOperator concept (dense, sparse, composite,
/// etc.). Q is not computed by default; when test_mode is enabled, Q = A * R^{-1}
/// is materialized for verification.
///
template <typename T>
class CholQR_linops {
    public:

        bool timing;
        bool test_mode;
        T eps;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // 6 entries: alloc, fwd, adj, chol, rest, total
        std::vector<long> times;

        // Column-block size for Gram and Q materialization. <=0 or >=n means no blocking.
        int64_t block_size;

        // Adaptive-shift safety net (Oleg's prescription): the first attempt is always
        // unshifted (shift_factor is hard-wired to 0 in call()); only if potrf breaks
        // down does the primitive seed the shift at eps*trace(G) and grow it
        // x shift_growth. max_retries < 0 = unbounded (no ceiling) — retry until PD.
        int max_retries;
        T   shift_growth;

        CholQR_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            block_size = 0;
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
            max_retries  = -1;       // unbounded retries (no ceiling)
            shift_growth = T(10);
        }

        ~CholQR_linops() {
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
            long q_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // Plain CholQR = one unpreconditioned cholqr_iterate pass, unshifted-first.
            long it[5] = {0};
            int info = cholqr_iterate<T, GLO>(
                A, R, ldr, this->block_size, /*num_iters=*/1,
                /*shift_iter1=*/T(0), /*shift_iter_rest=*/T(0),
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr);
            if (info != 0) return info;

            // Test mode: materialize Q = A * R^{-1} (outside the timing region).
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();
                T* Q_buf = new T[m * n];
                RandLAPACK::materialize_Q_from_R(A, R, ldr, m, n, b_eff, Q_buf);
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;
                if (this->timing) { t1 = steady_clock::now(); q_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count() - q_dur;
                long fwd = it[0], adj = it[1], chol = it[3];
                long rest_dur = total_dur - (fwd + adj + chol);
                this->times = {0L, fwd, adj, chol, rest_dur, total_dur};   // [alloc, fwd, adj, chol, rest, total]
            }

            return 0;
        }
};


/// CholQR2 for abstract linear operators.
///
/// Two passes of CholQR via the shared primitives:
///   iter 1:  cholqr_primitive(A, shift_factor, max_retries)        -> R_1
///   iter 2:  cholqr_primitive(A, P=R_1, TRSM_IDENTITY, ..., max_retries) -> R
///
/// Both passes start UNSHIFTED (shift_factor = 0); only on potrf breakdown does
/// the primitive seed the shift at eps * trace(G) (= ||A||_F^2 for iter 1, ~ n for
/// the iter-2 preconditioned Gram) and grow it x shift_growth, retrying unboundedly
/// (max_retries < 0) until the Gram is PD. Starting unshifted
/// avoids biasing R_1 on well-conditioned inputs — an always-on eps shift was found
/// to leave CholQR2 *less* orthogonal than a single unshifted CholQR pass; the retry
/// still rescues Gram matrices driven non-PD by rounding.
///
/// Status codes from call():
///   1  if iter-1 cholqr_primitive exhausted retries
///   2  if iter-2 cholqr_primitive exhausted retries
///
template <typename T>
class CholQR2_linops {
    public:
        bool timing;
        bool test_mode;
        T eps;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // 11 entries: alloc, fwd1, adj1, chol1, upd1, fwd2, adj2, gemm2, chol2, upd2, total
        //   upd1 is 0 by convention (iter 1 has no R-update step).
        std::vector<long> times;

        int64_t block_size;

        // Adaptive-shift policy (see cholqr_primitive).
        // shift_factor_iter1 is multiplied by trace(G_1) = ||A||_F^2 on the first
        // attempt. shift_factor_iter2 is multiplied by trace(G_2) (~ n when the
        // preconditioner is well-formed). max_retries bounds the geometric retry
        // loop; final shift is shift_factor * shift_growth^k on the kth retry.
        T   shift_factor_iter1;
        T   shift_factor_iter2;
        int max_retries;
        T   shift_growth;

        CholQR2_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            block_size = 0;
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
            shift_factor_iter1 = T(0);   // unshifted first attempt (shift only on breakdown)
            shift_factor_iter2 = T(0);
            max_retries        = -1;     // unbounded retries (no ceiling)
            shift_growth       = T(10);
        }

        ~CholQR2_linops() {
            if (Q != nullptr) delete[] Q;
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

            // CholQR2 = two cholqr_iterate passes (iter 1 unpreconditioned, iter 2
            // preconditioned on R_1). Both shifts default to 0 (unshifted-first).
            long it[10] = {0};
            int info = cholqr_iterate<T, GLO>(
                A, R, ldr, this->block_size, /*num_iters=*/2,
                this->shift_factor_iter1, this->shift_factor_iter2,
                this->max_retries, this->shift_growth, this->timing,
                this->timing ? it : nullptr);
            if (info != 0) return info;   // 1 or 2 = the pass that failed

            // ---- Test mode: materialize Q = A * R^{-1} via blocked linop calls ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();
                T* Q_buf = new T[m * n];
                RandLAPACK::materialize_Q_from_R(A, R, ldr, m, n, b_eff, Q_buf);
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;
                if (this->timing) { t1 = steady_clock::now(); q_mat_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            if (this->timing) {
                total_t_stop = steady_clock::now();
                long total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count() - q_mat_dur;
                // it = [fwd1,adj1,gemm1=0,chol1,upd1=0, fwd2,adj2,gemm2,chol2,upd2].
                this->times = {0L,                                  // alloc (now inside cholqr_iterate)
                               it[0], it[1], it[3], it[4],          // fwd1, adj1, chol1, upd1
                               it[5], it[6], it[7], it[8], it[9],   // fwd2, adj2, gemm2, chol2, upd2
                               total_dur};
            }

            return 0;
        }
};


// Analytical peak working memory for CholQR2_linops, mirroring the analytic_kb
// helpers used by CholQR_linops / sCholQR3_linops. Sum of class-member scratches:
//   G, R_pre, P_prev (3 n^2) + A_temp (m * b_eff) + Z_buf (n * b_eff)
//   + G_backup (n^2) when max_retries > 0
//   + cholqr_primitive's transient G + A_temp during iter 1 (n^2 + m*b_eff)
// We report the iter-2 peak (which dominates: persistent class scratches +
// active primitive scratches), conservatively assuming max_retries > 0.
template <typename T>
inline long cholqr2_linops_analytical_kb(int64_t m, int64_t n, int64_t block_size) {
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    int64_t bytes = (int64_t)sizeof(T) *
        ( 4 * n * n        // G + R_pre + P_prev + G_backup (peak; iter-2 retry)
        + 2 * m * b_eff    // A_temp (driver) + A_temp (primitive); same allocation in practice
        + n * b_eff        // Z_buf
        );
    return bytes / 1024;
}

} // end namespace RandLAPACK
