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
            long alloc_dur = 0, fwd_dur = 0, adj_dur = 0, chol_dur = 0, total_dur = 0;
            long q_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            int info = cholqr_primitive<T, GLO>(
                A, R, ldr,
                /*shift_factor=*/T(0),                      // unshifted first attempt
                this->block_size,
                fwd_dur, adj_dur, chol_dur, this->timing,
                this->max_retries, this->shift_growth);     // shift only on potrf breakdown

            if (info != 0) {
                return info;
            }

            // Test mode: materialize Q = A * R^{-1}. Recompute A * I into a full m×n buffer
            // (outside the timing region) then apply R^{-1} via trsm.
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();

                T* Q_buf = new T[m * n];
                T* I_mat = new T[n * n]();
                RandLAPACK::util::eye(n, n, I_mat);

                for (int64_t j = 0; j < n; j += b_eff) {
                    int64_t b_j = std::min(b_eff, n - j);
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, I_mat + j * n, n, (T)0.0, Q_buf + j * m, m);
                }
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_buf, m);

                delete[] I_mat;
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;

                if (this->timing) { t1 = steady_clock::now(); q_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                // Subtract test-mode Q materialization (not part of algorithmic cost).
                total_dur -= q_dur;

                long rest_dur = total_dur - (alloc_dur + fwd_dur + adj_dur + chol_dur);
                this->times = {alloc_dur, fwd_dur, adj_dur, chol_dur, rest_dur, total_dur};
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
/// the primitive seed the shift at eps * ||A||_F^2 and grow it x shift_growth,
/// retrying unboundedly (max_retries < 0) until the Gram is PD. Starting unshifted
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
            long alloc_dur = 0;
            long fwd1_dur = 0, adj1_dur = 0, chol1_dur = 0, upd1_dur = 0;
            long fwd2_dur = 0, adj2_dur = 0, gemm2_dur = 0, chol2_dur = 0, upd2_dur = 0;
            long q_mat_dur = 0, total_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // ---- Shared per-call scratch for cholqr_primitive iter-2 ----
            if (this->timing) t0 = steady_clock::now();
            T* G       = new T[n * n]();
            T* R_pre   = new T[n * n]();
            T* P_prev  = new T[n * n]();
            T* A_temp  = new T[m * b_eff];
            T* Z_buf   = new T[n * b_eff];
            if (this->timing) { t1 = steady_clock::now(); alloc_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Iter 1: shifted CholQR via cholqr_primitive -> R_1 in R ----
            int info = cholqr_primitive<T, GLO>(
                A, R, ldr,
                this->shift_factor_iter1,
                this->block_size,
                fwd1_dur, adj1_dur, chol1_dur, this->timing,
                this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 1;
            }

            // ---- Iter 2: PCholQR with P = R_1 (TRSM_IDENTITY precond) ----
            // Stash R_1 in P_prev before cholqr_primitive overwrites R with R_2.
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, P_prev, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), P_prev + 1, n);

            long precond_inv2 = 0, update2 = 0;
            info = cholqr_primitive<T, GLO>(
                A, P_prev, R, ldr,
                PCholQRPrecondMethod::TRSM_IDENTITY,
                this->block_size,
                /*bqrrp_block_ratio=*/T(1.0),
                R_pre, G, A_temp, Z_buf,
                /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
                precond_inv2, fwd2_dur, adj2_dur, gemm2_dur, chol2_dur, update2,
                this->timing,
                this->shift_factor_iter2, this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 2;
            }
            upd2_dur = precond_inv2 + update2;

            // ---- Test mode: materialize Q = A * R^{-1} via blocked linop calls ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();

                RandLAPACK::util::eye(n, n, R_pre);
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, n, n, T(1), R, ldr, R_pre, n);

                T* Q_buf = new T[m * n];
                for (int64_t j = 0; j < n; j += b_eff) {
                    int64_t b_j = std::min(b_eff, n - j);
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, R_pre + j * n, n, (T)0.0, Q_buf + j * m, m);
                }
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;

                if (this->timing) { t1 = steady_clock::now(); q_mat_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            delete[] G; delete[] R_pre; delete[] P_prev;
            delete[] A_temp; delete[] Z_buf;

            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                total_dur -= q_mat_dur;
                this->times = {alloc_dur,
                               fwd1_dur, adj1_dur, chol1_dur, upd1_dur,
                               fwd2_dur, adj2_dur, gemm2_dur, chol2_dur, upd2_dur,
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
