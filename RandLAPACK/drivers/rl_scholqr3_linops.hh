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
            // shift_factor = 11 * eps * n  so the shift becomes 11*eps*n*||A||_F^2
            // (per the collaborator's Alg 3 step 3).
            // ============================================================
            T shift_factor_iter1 = (T)11.0 * std::numeric_limits<T>::epsilon() * (T)n;
            int info = cholqr_primitive<T, GLO>(
                A, R, ldr,
                shift_factor_iter1,
                this->block_size,
                fwd1_dur, adj1_dur, chol1_dur, this->timing);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 1;
            }
            // upd1 placeholder kept for matlab CSV compatibility (no inverse formed in iter 1).
            upd1_dur = 0;

            // ============================================================
            // Iter 2: pcholqr_primitive(A, P = R_1)
            // ============================================================
            // Stash R_1 in P_prev before pcholqr_primitive overwrites R with R_2.
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
                this->timing);
            if (info != 0) {
                delete[] G; delete[] R_pre; delete[] P_prev;
                delete[] A_temp; delete[] Z_buf;
                return 2;
            }
            // Fold pcholqr's precond_inv into upd2 alongside the trmm update (matches the prior
            // sCholQR3 upd2 semantics = "M update + R update" in one slot).
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
                this->timing);
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

        // Timing breakdown (15 entries; layout preserved for matlab plotters):
        // [0]  alloc      [1]  fwd1   [2]  adj1   [3]  chol1   [4]  trsm1   [5]  fwd_q
        // [6]  syrk2      [7]  chol2  [8]  upd2
        // [9]  syrk3      [10] chol3  [11] upd3
        // [12] q_mat      [13] rest   [14] total
        std::vector<long> times;

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
        }

        ~sCholQR3_linops_basic() {
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
            long fwd1_dur = 0, adj1_dur = 0, chol1_dur = 0, trsm1_dur = 0, fwd_q_dur = 0;
            long syrk2_dur = 0, chol2_dur = 0, upd2_dur = 0;
            long syrk3_dur = 0, chol3_dur = 0, upd3_dur = 0;
            long q_mat_dur = 0, total_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            if (this->timing) t0 = steady_clock::now();
            T* Q_buf = new T[m * n];     // materialized operator, updated in-place through iters
            T* G     = new T[n * n]();   // Gram workspace
            T* M     = new T[n * n]();   // n × n — R1^{-1} for Q materialization
            if (this->timing) { t1 = steady_clock::now(); alloc_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Iter 1: shifted CholQR via cholqr_primitive (no Q_buf yet) ----
            T shift_factor_iter1 = (T)11.0 * std::numeric_limits<T>::epsilon() * (T)n;
            int info = cholqr_primitive<T, GLO>(
                A, R, ldr,
                shift_factor_iter1,
                /*block_size=*/0,   // basic variant is non-blocked
                fwd1_dur, adj1_dur, chol1_dur, this->timing);
            if (info != 0) {
                delete[] Q_buf; delete[] G; delete[] M;
                return 1;
            }

            // ---- Materialize Q_buf = A * R1^{-1} (NoTrans linop call #3) ----
            // First compute M = I * R_1^{-1} = R_1^{-1} via Side::Right TRSM(I, R_1).
            if (this->timing) t0 = steady_clock::now();
            RandLAPACK::util::eye(n, n, M);
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, T(1), R, ldr, M, n);
            if (this->timing) { t1 = steady_clock::now(); trsm1_dur = duration_cast<microseconds>(t1 - t0).count(); }

            if (this->timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              m, n, n, (T)1.0, M, n, (T)0.0, Q_buf, m);
            if (this->timing) { t1 = steady_clock::now(); fwd_q_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Iter 2: dense syrk on Q_buf -> G2 = R_2^T R_2, then Q_buf *= R_2^{-1}, R = R_2 * R ----
            if (this->timing) t0 = steady_clock::now();
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                       n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);
            if (this->timing) { t1 = steady_clock::now(); syrk2_dur = duration_cast<microseconds>(t1 - t0).count(); t0 = t1; }

            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), &G[1], n);
            if (lapack::potrf(Uplo::Upper, n, G, n)) {
                delete[] Q_buf; delete[] G; delete[] M;
                return 2;
            }
            if (this->timing) { t1 = steady_clock::now(); chol2_dur = duration_cast<microseconds>(t1 - t0).count(); t0 = t1; }

            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, m, n, (T)1.0, G, n, Q_buf, m);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R, ldr);
            if (this->timing) { t1 = steady_clock::now(); upd2_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Iter 3: same pattern ----
            if (this->timing) t0 = steady_clock::now();
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                       n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);
            if (this->timing) { t1 = steady_clock::now(); syrk3_dur = duration_cast<microseconds>(t1 - t0).count(); t0 = t1; }

            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), &G[1], n);
            if (lapack::potrf(Uplo::Upper, n, G, n)) {
                delete[] Q_buf; delete[] G; delete[] M;
                return 3;
            }
            if (this->timing) { t1 = steady_clock::now(); chol3_dur = duration_cast<microseconds>(t1 - t0).count(); t0 = t1; }

            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R, ldr);
            if (this->timing) { t1 = steady_clock::now(); upd3_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Test mode: Q_buf *= R_3^{-1} so Q = A * R^{-1} ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();
                // Iter 3 didn't update Q_buf (we skip the m × n trsm there since iter 3 is the last
                // Cholesky polish; the m × n trsm for R_3^{-1} only matters if we want Q).
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, G, n, Q_buf, m);
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;
                if (this->timing) { t1 = steady_clock::now(); q_mat_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            // ---- Finalize timing ----
            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                total_dur -= q_mat_dur;

                long rest_dur = total_dur - (alloc_dur + fwd1_dur + adj1_dur + chol1_dur + trsm1_dur + fwd_q_dur +
                                              syrk2_dur + chol2_dur + upd2_dur +
                                              syrk3_dur + chol3_dur + upd3_dur);
                this->times = {alloc_dur, fwd1_dur, adj1_dur, chol1_dur, trsm1_dur, fwd_q_dur,
                               syrk2_dur, chol2_dur, upd2_dur,
                               syrk3_dur, chol3_dur, upd3_dur,
                               q_mat_dur, rest_dur, total_dur};
            }

            delete[] G;
            delete[] M;
            if (!this->test_mode)
                delete[] Q_buf;
            return 0;
        }
};

} // end namespace RandLAPACK
