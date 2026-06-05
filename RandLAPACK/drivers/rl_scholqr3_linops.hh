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

using namespace std::chrono;

namespace RandLAPACK {

/// Shifted Cholesky QR3 for abstract linear operators (fully-blocked variant).
///
/// Algorithm 3 from the collaborator's spec:
///   iter 1: cholqr_primitive(A) with shift s = 11 * eps * n * ||A||_F^2  -> R_1
///   iter i = 2, 3: pcholqr_primitive(A, R_{i-1}, TRSM_IDENTITY)  -> R_i
///   return R_3
///
/// Peak memory O(n^2 + (m+n)*b_eff) — never materializes the m × n operator product
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
        T eps;

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

        int64_t block_size;

        // Adaptive-shift policy (see cholqr_primitive / pcholqr_primitive).
        // Lowered from the original `11 * eps * n` to plain `eps` per Oleg: the
        // smaller initial shift avoids the iter-2 collapse where shift ≫ σ_min²(A)
        // makes G_2 effectively rank-deficient. The retry loop bumps shift × 10
        // if potrf still bails, up to max_retries times.
        T   shift_factor_iter1;
        T   shift_factor_iter23;
        int max_retries;
        T   shift_growth;

        sCholQR3_linops(
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
            shift_factor_iter1  = std::numeric_limits<T>::epsilon();
            shift_factor_iter23 = std::numeric_limits<T>::epsilon();
            max_retries         = 10;
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
            long alloc_dur = 0;
            long fwd1_dur = 0, adj1_dur = 0, chol1_dur = 0, upd1_dur = 0;
            long fwd2_dur = 0, adj2_dur = 0, gemm2_dur = 0, chol2_dur = 0, upd2_dur = 0;
            long fwd3_dur = 0, adj3_dur = 0, gemm3_dur = 0, chol3_dur = 0, upd3_dur = 0;
            long q_mat_dur = 0, total_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // ---- Workspaces shared across all 3 iterations ----
            if (this->timing) t0 = steady_clock::now();
            T* G       = new T[n * n]();             // Gram / Cholesky workspace
            T* R_pre   = new T[n * n]();             // preconditioner inverse (used in iters 2, 3)
            T* P_prev  = new T[n * n]();             // previous R_{i-1}, fed as P to pcholqr_primitive
            T* A_temp  = new T[m * b_eff];           // m × b_eff scratch for linop NoTrans
            T* Z_buf   = new T[n * b_eff];           // n × b_eff scratch for linop Trans
            if (this->timing) { t1 = steady_clock::now(); alloc_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ============================================================
            // Iter 1: shifted CholQR -> R_1, written into R
            // Adaptive shift (see cholqr_primitive). Initial shift_factor_iter1
            // defaults to eps; primitive bumps × shift_growth on potrf failure.
            // ============================================================
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
            upd1_dur = 0;

            // ============================================================
            // Iter 2: pcholqr_primitive(A, P = R_1)
            // ============================================================
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, P_prev, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), P_prev + 1, n);

            long precond_inv2 = 0, update2 = 0;
            info = pcholqr_primitive<T, GLO>(
                A, P_prev, R, ldr,
                PCholQRPrecondMethod::TRSM_IDENTITY,
                this->block_size,
                /*bqrrp_block_ratio=*/T(1.0),
                R_pre, G, A_temp, Z_buf,
                /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
                precond_inv2, fwd2_dur, adj2_dur, gemm2_dur, chol2_dur, update2,
                this->timing,
                this->shift_factor_iter23, this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 2;
            }
            upd2_dur = precond_inv2 + update2;

            // ============================================================
            // Iter 3: pcholqr_primitive(A, P = R_2)
            // ============================================================
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, P_prev, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), P_prev + 1, n);

            long precond_inv3 = 0, update3 = 0;
            info = pcholqr_primitive<T, GLO>(
                A, P_prev, R, ldr,
                PCholQRPrecondMethod::TRSM_IDENTITY,
                this->block_size,
                /*bqrrp_block_ratio=*/T(1.0),
                R_pre, G, A_temp, Z_buf,
                /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
                precond_inv3, fwd3_dur, adj3_dur, gemm3_dur, chol3_dur, update3,
                this->timing,
                this->shift_factor_iter23, this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 3;
            }
            upd3_dur = precond_inv3 + update3;

            // ============================================================
            // Test mode: materialize Q = A * R^{-1} (outside timing region).
            // R_pre currently holds R_2^{-1} from iter 3's pcholqr_primitive precond step.
            // We need R^{-1} = R_3^{-1} = R^{chol_3}^{-1} * R_2^{-1}.
            // Simpler: recompute R^{-1} from R via trsm(R, I), then materialize Q via blocked linop.
            // ============================================================
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

            // ============================================================
            // Finalize timing
            // ============================================================
            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                total_dur -= q_mat_dur;

                long rest_dur = total_dur - (alloc_dur +
                    fwd1_dur + adj1_dur + chol1_dur + upd1_dur +
                    fwd2_dur + adj2_dur + gemm2_dur + chol2_dur + upd2_dur +
                    fwd3_dur + adj3_dur + gemm3_dur + chol3_dur + upd3_dur);

                this->times = {alloc_dur,
                    fwd1_dur, adj1_dur, chol1_dur, upd1_dur,
                    fwd2_dur, adj2_dur, gemm2_dur, chol2_dur, upd2_dur,
                    fwd3_dur, adj3_dur, gemm3_dur, chol3_dur, upd3_dur,
                    q_mat_dur, rest_dur, total_dur};
            }

            delete[] G;
            delete[] R_pre;
            delete[] P_prev;
            delete[] A_temp;
            delete[] Z_buf;
            return 0;
        }
};

/// Non-blocked (basic) sCholQR3 — materializes Q = A * R_1^{-1} after iter 1 and uses
/// dense syrk for iters 2, 3.  Iter 1 still routes through cholqr_primitive; iters 2-3
/// stay inline because they operate on the dense Q buffer (not the linop).
///
/// Linop accesses: exactly 3 (NoTrans for Gram-step, Trans for Gram-step, NoTrans to
/// materialize Q after iter 1).
///
template <typename T>
class sCholQR3_linops_basic {
    public:

        bool timing;
        bool test_mode;
        T eps;

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

        // Adaptive shift policy — shared with sCholQR3_linops (Oleg's prescription).
        T   shift_factor_iter1;
        T   shift_factor_iter23;
        int max_retries;
        T   shift_growth;

        sCholQR3_linops_basic(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
            shift_factor_iter1  = std::numeric_limits<T>::epsilon();
            shift_factor_iter23 = std::numeric_limits<T>::epsilon();
            max_retries         = 10;
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
        // own G internally per call). Diagnostic prints from cholqr_primitive /
        // pcholqr_primitive surface here too.
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr
        ) {
            steady_clock::time_point t0, t1, total_t_start, total_t_stop;
            long alloc_dur = 0;
            long fwd1_dur = 0, adj1_dur = 0, chol1_dur = 0;
            long chol2_dur = 0, upd2_dur = 0;
            long chol3_dur = 0, upd3_dur = 0;
            long q_mat_dur = 0, total_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            int64_t b_eff = n;   // non-blocked (block_size = 0 → b_eff = n)

            // Driver-owned scratch for pcholqr_primitive in iters 2 and 3.
            if (this->timing) t0 = steady_clock::now();
            T* G       = new T[n * n]();
            T* R_pre   = new T[n * n]();
            T* P_prev  = new T[n * n]();
            T* A_temp  = new T[m * b_eff];
            T* Z_buf   = new T[n * b_eff];
            if (this->timing) { t1 = steady_clock::now(); alloc_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Iter 1: shifted CholQR via cholqr_primitive -> R_1 ----
            int info = cholqr_primitive<T, GLO>(
                A, R, ldr,
                this->shift_factor_iter1,
                /*block_size=*/0,
                fwd1_dur, adj1_dur, chol1_dur, this->timing,
                this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 1;
            }

            // ---- Iter 2: pcholqr_primitive(A, P = R_1) ----
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, P_prev, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), P_prev + 1, n);

            long precond_inv2 = 0, fwd2 = 0, adj2 = 0, gemm2 = 0, update2 = 0;
            info = pcholqr_primitive<T, GLO>(
                A, P_prev, R, ldr,
                PCholQRPrecondMethod::TRSM_IDENTITY,
                /*block_size=*/0,
                /*bqrrp_block_ratio=*/T(1.0),
                R_pre, G, A_temp, Z_buf,
                /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
                precond_inv2, fwd2, adj2, gemm2, chol2_dur, update2,
                this->timing,
                this->shift_factor_iter23, this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 2;
            }
            upd2_dur = precond_inv2 + update2;
            // Fold iter-2 fwd+adj+gemm into chol2 slot (kept distinct from iter-3
            // timings; matlab plotter sums them downstream).
            chol2_dur += fwd2 + adj2 + gemm2;

            // ---- Iter 3: pcholqr_primitive(A, P = R_2) ----
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, P_prev, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), P_prev + 1, n);

            long precond_inv3 = 0, fwd3 = 0, adj3 = 0, gemm3 = 0, update3 = 0;
            info = pcholqr_primitive<T, GLO>(
                A, P_prev, R, ldr,
                PCholQRPrecondMethod::TRSM_IDENTITY,
                /*block_size=*/0,
                /*bqrrp_block_ratio=*/T(1.0),
                R_pre, G, A_temp, Z_buf,
                /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
                precond_inv3, fwd3, adj3, gemm3, chol3_dur, update3,
                this->timing,
                this->shift_factor_iter23, this->max_retries, this->shift_growth);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 3;
            }
            upd3_dur = precond_inv3 + update3;
            chol3_dur += fwd3 + adj3 + gemm3;

            // ---- Test mode: materialize Q = A * R^{-1} via blocked linop call ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();
                RandLAPACK::util::eye(n, n, R_pre);
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, n, n, T(1), R, ldr, R_pre, n);

                T* Q_buf = new T[m * n];
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, n, n, (T)1.0, R_pre, n, (T)0.0, Q_buf, m);
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
                long rest_dur = total_dur - (alloc_dur + fwd1_dur + adj1_dur + chol1_dur +
                                              chol2_dur + upd2_dur +
                                              chol3_dur + upd3_dur);
                this->times = {alloc_dur, fwd1_dur, adj1_dur, chol1_dur,
                               /*trsm1=*/0L, /*fwd_q=*/0L,
                               /*syrk2=*/0L, chol2_dur, upd2_dur,
                               /*syrk3=*/0L, chol3_dur, upd3_dur,
                               q_mat_dur, rest_dur, total_dur};
            }
            return 0;
        }
};

} // end namespace RandLAPACK
