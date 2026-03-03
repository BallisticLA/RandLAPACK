#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std::chrono;

namespace RandLAPACK {

/// Shifted Cholesky QR3 algorithm for computing QR factorization via linear operators.
///
/// Fully-blocked implementation: never materializes the full m x n operator product
/// during the QR iterations. All three iterations compute their Gram matrices through
/// blocked linop calls, using an accumulated right-factor M = R1^{-1} R2^{-1} ...
/// to avoid storing Q explicitly.
///
/// Algorithm:
///   1. Shifted CholQR1: G1 = A^T A + s*I,  R1 = chol(G1),  M <- R1^{-1}
///   2. CholQR2:         G2 = M^T A^T A M,   R2 = chol(G2),  R = R2*R1,  M <- M*R2^{-1}
///   3. CholQR3:         G3 = M^T A^T A M,   R3 = chol(G3),  R = R3*R
///
/// Blocked Gram computation for iteration k (M_k = accumulated R-inverse):
///   for each column block j of width b:
///     W = A * M_k[:, j:j+b]          (linop NoTrans, m x b)
///     Z = A^T * W                    (linop Trans,  n x b)
///     G[:, j:j+b] = M_k^T * Z       (gemm, n x b)
///
/// Peak memory: O(m*b + n^2) -- no m x n buffer needed during QR iterations.
/// If test_mode is enabled, Q = A * R^{-1} is materialized at the end (m x n).
///
/// The shift is computed as: shift = 11 * eps * n * ||A||_F^2
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

        // Individual Cholesky factors from each iteration (n x n upper triangular).
        std::vector<T> G1_factor;
        std::vector<T> G2_factor;
        std::vector<T> G3_factor;

        // Timing breakdown (18 entries):
        // [0]  alloc      - buffer allocation
        // [1]  fwd1       - Iter 1 NoTrans: A * M[:, block]
        // [2]  adj1       - Iter 1 Trans: A^T * W (direct to G since M=I)
        // [3]  chol1      - Iter 1 Cholesky (potrf)
        // [4]  upd1       - M = R1^{-1} (n x n trsm)
        // [5]  fwd2       - Iter 2 NoTrans: A * M[:, block]
        // [6]  adj2       - Iter 2 Trans: A^T * W
        // [7]  gemm2      - Iter 2 M^T * Z
        // [8]  chol2      - Iter 2 Cholesky
        // [9]  upd2       - R = R2*R1, M *= R2^{-1}
        // [10] fwd3       - Iter 3 NoTrans
        // [11] adj3       - Iter 3 Trans
        // [12] gemm3      - Iter 3 M^T * Z
        // [13] chol3      - Iter 3 Cholesky
        // [14] upd3       - R = R3*R
        // [15] q_mat      - Q materialization for test mode (0 if not test_mode)
        // [16] rest       - unaccounted time
        // [17] total      - wall-clock total
        std::vector<long> times;

        // Column-block size for blocked Gram computations.
        //
        // Controls the width of column blocks used in all three iterations'
        // Gram matrix computations. Smaller values reduce peak memory
        // (O(m*block_size + n^2) instead of O(m*n)), at the cost of
        // more linop calls (2 * ceil(n/block_size) per iteration).
        //
        // When block_size <= 0 or >= n, uses b_eff = n (single block per loop).
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

        /// Computes the QR factorization A = QR using shifted Cholesky QR3.
        ///
        /// @param[in] A
        ///     The m-by-n linear operator A.
        ///
        /// @param[out] R
        ///     Stores n-by-n matrix with upper-triangular R factor.
        ///     Zero entries are not compressed.
        ///
        /// @param[in] ldr
        ///     Leading dimension of R.
        ///
        /// @return = 0: successful exit
        ///
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr
        ) {
            ///--------------------TIMING VARS--------------------/
            steady_clock::time_point t_start, t_stop;
            steady_clock::time_point total_t_start, total_t_stop;
            long alloc_dur = 0;
            long fwd1_dur = 0, adj1_dur = 0, chol1_dur = 0, upd1_dur = 0;
            long fwd2_dur = 0, adj2_dur = 0, gemm2_dur = 0, chol2_dur = 0, upd2_dur = 0;
            long fwd3_dur = 0, adj3_dur = 0, gemm3_dur = 0, chol3_dur = 0, upd3_dur = 0;
            long q_mat_dur = 0, total_dur = 0;

            if(this->timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            // Determine effective block width.
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            if(this->timing)
                t_start = steady_clock::now();

            // ---- Allocate buffers ----
            // G: n x n Gram matrix / Cholesky workspace (zero-init for lower triangle)
            T* G = new T[n * n]();

            // R_temp: n x n workspace for R accumulation via trmm
            T* R_temp = new T[n * n]();

            // M: n x n accumulated R-inverse product (starts as identity)
            T* M = new T[n * n]();
            RandLAPACK::util::eye(n, n, M);

            // A_temp: m x b_eff buffer for linop NoTrans output
            T* A_temp = new T[m * b_eff];

            // Z_buf: n x b_eff buffer for linop Trans output (used in iterations 2-3)
            T* Z_buf = new T[n * b_eff];

            if(this->timing) {
                t_stop = steady_clock::now();
                alloc_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 1: Shifted Cholesky QR
            //================================================================
            // Blocked Gram: G = A^T A (since M = I, no M^T multiply needed)
            long fwd1_accum = 0, adj1_accum = 0;
            for (int64_t j = 0; j < n; j += b_eff) {
                int64_t b_j = std::min(b_eff, n - j);

                // W = A * M[:, j:j+b]  (= A * I[:, j:j+b] since M = I)
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, A_temp, m);
                if(this->timing) { t_stop = steady_clock::now(); fwd1_accum += duration_cast<microseconds>(t_stop - t_start).count(); }

                // G[:, j:j+b] = A^T * W  (direct to G since M = I)
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                  n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n);
                if(this->timing) { t_stop = steady_clock::now(); adj1_accum += duration_cast<microseconds>(t_stop - t_start).count(); }
            }

            // Compute shift from ||A||_F^2 = trace(G)
            T norm_A_sq = 0;
            for (int64_t i = 0; i < n; ++i)
                norm_A_sq += G[i * (n + 1)];
            T shift = 11 * std::numeric_limits<T>::epsilon() * n * norm_A_sq;

            // Add shift to diagonal: G = G + shift * I
            for (int64_t i = 0; i < n; ++i)
                G[i * (n + 1)] += shift;

            if(this->timing) {
                fwd1_dur = fwd1_accum;
                adj1_dur = adj1_accum;
                t_start = steady_clock::now();
            }

            // Zero lower triangle, Cholesky: G = R1^T * R1
            if (n > 1)
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            lapack::potrf(Uplo::Upper, n, G, n);

            // Save G1 factor
            this->G1_factor.resize(n * n, (T)0.0);
            lapack::lacpy(MatrixType::Upper, n, n, G, n, this->G1_factor.data(), n);

            // Initialize R = R1
            lapack::lacpy(MatrixType::Upper, n, n, G, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                chol1_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Update M: M = I * R1^{-1} = R1^{-1}
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, M, n);

            if(this->timing) {
                t_stop = steady_clock::now();
                upd1_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 2: Cholesky QR
            //================================================================
            // Blocked Gram: G2 = M^T * A^T * A * M  where M = R1^{-1}
            long fwd2_accum = 0, adj2_accum = 0, gemm2_accum = 0;
            for (int64_t j = 0; j < n; j += b_eff) {
                int64_t b_j = std::min(b_eff, n - j);

                // W = A * M[:, j:j+b]
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, A_temp, m);
                if(this->timing) { t_stop = steady_clock::now(); fwd2_accum += duration_cast<microseconds>(t_stop - t_start).count(); }

                // Z = A^T * W
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                  n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);
                if(this->timing) { t_stop = steady_clock::now(); adj2_accum += duration_cast<microseconds>(t_stop - t_start).count(); }

                // G[:, j:j+b] = M^T * Z
                if(this->timing) t_start = steady_clock::now();
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                           n, b_j, n, (T)1.0, M, n, Z_buf, n, (T)0.0, G + j * n, n);
                if(this->timing) { t_stop = steady_clock::now(); gemm2_accum += duration_cast<microseconds>(t_stop - t_start).count(); }
            }

            if(this->timing) {
                fwd2_dur = fwd2_accum;
                adj2_dur = adj2_accum;
                gemm2_dur = gemm2_accum;
                t_start = steady_clock::now();
            }

            // Zero lower triangle, Cholesky: G = R2^T * R2
            if (n > 1)
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            lapack::potrf(Uplo::Upper, n, G, n);

            // Save G2 factor
            this->G2_factor.resize(n * n, (T)0.0);
            lapack::lacpy(MatrixType::Upper, n, n, G, n, this->G2_factor.data(), n);

            if(this->timing) {
                t_stop = steady_clock::now();
                chol2_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // R = R2 * R1
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            // M = M * R2^{-1}
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, M, n);

            if(this->timing) {
                t_stop = steady_clock::now();
                upd2_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 3: Cholesky QR
            //================================================================
            // Blocked Gram: G3 = M^T * A^T * A * M  where M = R1^{-1} * R2^{-1}
            long fwd3_accum = 0, adj3_accum = 0, gemm3_accum = 0;
            for (int64_t j = 0; j < n; j += b_eff) {
                int64_t b_j = std::min(b_eff, n - j);

                // W = A * M[:, j:j+b]
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, A_temp, m);
                if(this->timing) { t_stop = steady_clock::now(); fwd3_accum += duration_cast<microseconds>(t_stop - t_start).count(); }

                // Z = A^T * W
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                  n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);
                if(this->timing) { t_stop = steady_clock::now(); adj3_accum += duration_cast<microseconds>(t_stop - t_start).count(); }

                // G[:, j:j+b] = M^T * Z
                if(this->timing) t_start = steady_clock::now();
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                           n, b_j, n, (T)1.0, M, n, Z_buf, n, (T)0.0, G + j * n, n);
                if(this->timing) { t_stop = steady_clock::now(); gemm3_accum += duration_cast<microseconds>(t_stop - t_start).count(); }
            }

            if(this->timing) {
                fwd3_dur = fwd3_accum;
                adj3_dur = adj3_accum;
                gemm3_dur = gemm3_accum;
                t_start = steady_clock::now();
            }

            // Zero lower triangle, Cholesky: G = R3^T * R3
            if (n > 1)
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            lapack::potrf(Uplo::Upper, n, G, n);

            // Save G3 factor
            this->G3_factor.resize(n * n, (T)0.0);
            lapack::lacpy(MatrixType::Upper, n, n, G, n, this->G3_factor.data(), n);

            if(this->timing) {
                t_stop = steady_clock::now();
                chol3_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // R = R3 * R
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                upd3_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Test mode: materialize Q = A * R^{-1} = A * M * R3^{-1}
            //================================================================
            if(this->test_mode) {
                if(this->timing)
                    t_start = steady_clock::now();

                // M currently holds R1^{-1} R2^{-1}; update to R^{-1} = R1^{-1} R2^{-1} R3^{-1}
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, n, n, (T)1.0, G, n, M, n);

                // Materialize Q = A * M in blocks
                T* Q_buf = new T[m * n]();
                for (int64_t j = 0; j < n; j += b_eff) {
                    int64_t b_j = std::min(b_eff, n - j);
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, Q_buf + j * m, m);
                }

                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;

                if(this->timing) {
                    t_stop = steady_clock::now();
                    q_mat_dur = duration_cast<microseconds>(t_stop - t_start).count();
                }
            }

            //================================================================
            // Finalize timing
            //================================================================
            if(this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q materialization from total (test overhead, not algorithmic cost)
                total_dur -= q_mat_dur;

                long rest_dur = total_dur - (alloc_dur +
                    fwd1_dur + adj1_dur + chol1_dur + upd1_dur +
                    fwd2_dur + adj2_dur + gemm2_dur + chol2_dur + upd2_dur +
                    fwd3_dur + adj3_dur + gemm3_dur + chol3_dur + upd3_dur);

                // 18 entries
                this->times = {alloc_dur,
                    fwd1_dur, adj1_dur, chol1_dur, upd1_dur,
                    fwd2_dur, adj2_dur, gemm2_dur, chol2_dur, upd2_dur,
                    fwd3_dur, adj3_dur, gemm3_dur, chol3_dur, upd3_dur,
                    q_mat_dur, rest_dur, total_dur};
            }

            // Cleanup
            delete[] G;
            delete[] R_temp;
            delete[] M;
            delete[] A_temp;
            delete[] Z_buf;

            return 0;
        }
};

/// Non-blocked (basic) sCholQR3 algorithm for computing QR factorization via linear operators.
///
/// Matches the standard sCholQR3 pseudocode from Fukaya et al. (SISC, 2020) exactly:
///   1. Compute G1 = A^T A via linop, add shift, Cholesky → R1
///   2. Materialize Q = A * R1^{-1} via linop
///   3. Iterations 2-3: G = Q^T Q via dense syrk, Cholesky, Q *= R_k^{-1} via dense trsm
///
/// Accesses the linear operator exactly 3 times:
///   - NoTrans: W = A * I (materialization for Gram computation)
///   - Trans:   G1 = A^T * W (Gram matrix)
///   - NoTrans: Q = A * R1^{-1} (first Q-factor)
///
/// After the first Q-factor, iterations 2-3 use dense syrk on Q (no further linop calls).
/// This is theoretically distinct from sCholQR3_linops (fully-blocked), which recomputes
/// each Gram through the linop and never materializes the m x n operator product.
///
/// Peak memory: O(m*n + n^2) — Q is explicitly stored as m x n dense.
///
/// Reference: Shifted Cholesky QR from Fukaya et al. (SISC, 2020).
///
template <typename T>
class sCholQR3_linops_basic {
    public:

        bool timing;
        bool test_mode;
        T eps;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // Individual Cholesky factors from each iteration (n x n upper triangular).
        std::vector<T> G1_factor;
        std::vector<T> G2_factor;
        std::vector<T> G3_factor;

        // Timing breakdown (15 entries):
        // [0]  alloc      - buffer allocation
        // [1]  fwd1       - NoTrans: W = A * I (m x n)
        // [2]  adj1       - Trans: G = A^T * W (n x n)
        // [3]  chol1      - Iter 1 Cholesky
        // [4]  trsm1      - M = R1^{-1} (n x n trsm)
        // [5]  fwd_q      - NoTrans: Q = A * M (m x n)
        // [6]  syrk2      - G = Q^T Q
        // [7]  chol2      - Iter 2 Cholesky
        // [8]  upd2       - Q *= R2^{-1}, R = R2*R1
        // [9]  syrk3      - G = Q^T Q
        // [10] chol3      - Iter 3 Cholesky
        // [11] upd3       - R = R3*R
        // [12] q_mat      - test mode: Q_buf *= R3^{-1} (m x n trsm), 0 otherwise
        // [13] rest       - unaccounted time
        // [14] total      - wall-clock total
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

        /// Computes the QR factorization A = QR using shifted Cholesky QR3 (basic variant).
        ///
        /// @param[in] A
        ///     The m-by-n linear operator A.
        ///
        /// @param[out] R
        ///     Stores n-by-n matrix with upper-triangular R factor.
        ///     Zero entries are not compressed.
        ///
        /// @param[in] ldr
        ///     Leading dimension of R.
        ///
        /// @return = 0: successful exit
        ///
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr
        ) {
            ///--------------------TIMING VARS--------------------/
            steady_clock::time_point t_start, t_stop;
            steady_clock::time_point total_t_start, total_t_stop;
            long alloc_dur = 0;
            long fwd1_dur = 0, adj1_dur = 0, chol1_dur = 0, trsm1_dur = 0, fwd_q_dur = 0;
            long syrk2_dur = 0, chol2_dur = 0, upd2_dur = 0;
            long syrk3_dur = 0, chol3_dur = 0, upd3_dur = 0;
            long q_mat_dur = 0, total_dur = 0;

            if(this->timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            if(this->timing)
                t_start = steady_clock::now();

            // ---- Allocate buffers ----
            // Q_buf: m x n — materialized operator, updated in-place through iterations
            T* Q_buf = new T[m * n];

            // G: n x n Gram matrix / Cholesky workspace (zero-init for lower triangle)
            T* G = new T[n * n]();

            // R_temp: n x n workspace for R accumulation via trmm
            T* R_temp = new T[n * n]();

            // M: n x n — starts as identity, becomes R1^{-1} for Q materialization
            T* M = new T[n * n]();
            RandLAPACK::util::eye(n, n, M);

            if(this->timing) {
                t_stop = steady_clock::now();
                alloc_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 1: Shifted Cholesky QR
            //================================================================
            // Gram: G1 = A^T * A via linop (2 of 3 total linop accesses)

            // Linop access 1: W = A * I (NoTrans, materializes operator as m x n dense)
            if(this->timing) t_start = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              m, n, n, (T)1.0, M, n, (T)0.0, Q_buf, m);
            if(this->timing) { t_stop = steady_clock::now(); fwd1_dur = duration_cast<microseconds>(t_stop - t_start).count(); }

            // Linop access 2: G1 = A^T * W (Trans, n x n Gram matrix)
            if(this->timing) t_start = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);
            if(this->timing) { t_stop = steady_clock::now(); adj1_dur = duration_cast<microseconds>(t_stop - t_start).count(); }

            // Compute shift from ||A||_F^2 = trace(G)
            T norm_A_sq = 0;
            for (int64_t i = 0; i < n; ++i)
                norm_A_sq += G[i * (n + 1)];
            T shift = 11 * std::numeric_limits<T>::epsilon() * n * norm_A_sq;

            // Add shift to diagonal: G = G + shift * I
            for (int64_t i = 0; i < n; ++i)
                G[i * (n + 1)] += shift;

            if(this->timing) {
                t_start = steady_clock::now();
            }

            // Zero lower triangle, Cholesky: G = R1^T * R1
            if (n > 1)
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            lapack::potrf(Uplo::Upper, n, G, n);

            // Save G1 factor
            this->G1_factor.resize(n * n, (T)0.0);
            lapack::lacpy(MatrixType::Upper, n, n, G, n, this->G1_factor.data(), n);

            // Initialize R = R1
            lapack::lacpy(MatrixType::Upper, n, n, G, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                chol1_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            // Compute M = I * R1^{-1} = R1^{-1}
            if(this->timing) t_start = steady_clock::now();
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, M, n);
            if(this->timing) { t_stop = steady_clock::now(); trsm1_dur = duration_cast<microseconds>(t_stop - t_start).count(); }

            // Linop access 3: Q_buf = A * M = A * R1^{-1} (NoTrans, m x n)
            if(this->timing) t_start = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              m, n, n, (T)1.0, M, n, (T)0.0, Q_buf, m);
            if(this->timing) { t_stop = steady_clock::now(); fwd_q_dur = duration_cast<microseconds>(t_stop - t_start).count(); }

            //================================================================
            // Iteration 2: Cholesky QR (dense syrk on Q_buf)
            //================================================================
            if(this->timing)
                t_start = steady_clock::now();

            // G2 = Q_buf^T * Q_buf (dense syrk, upper triangle only)
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                       n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);

            if(this->timing) {
                t_stop = steady_clock::now();
                syrk2_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Zero lower triangle, Cholesky: G = R2^T * R2
            if (n > 1)
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            lapack::potrf(Uplo::Upper, n, G, n);

            // Save G2 factor
            this->G2_factor.resize(n * n, (T)0.0);
            lapack::lacpy(MatrixType::Upper, n, n, G, n, this->G2_factor.data(), n);

            if(this->timing) {
                t_stop = steady_clock::now();
                chol2_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Q_buf *= R2^{-1} (m x n trsm — update Q in-place)
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, m, n, (T)1.0, G, n, Q_buf, m);

            // R = R2 * R1
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                upd2_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 3: Cholesky QR (dense syrk on Q_buf)
            //================================================================
            if(this->timing)
                t_start = steady_clock::now();

            // G3 = Q_buf^T * Q_buf (dense syrk, upper triangle only)
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                       n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);

            if(this->timing) {
                t_stop = steady_clock::now();
                syrk3_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Zero lower triangle, Cholesky: G = R3^T * R3
            if (n > 1)
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            lapack::potrf(Uplo::Upper, n, G, n);

            // Save G3 factor
            this->G3_factor.resize(n * n, (T)0.0);
            lapack::lacpy(MatrixType::Upper, n, n, G, n, this->G3_factor.data(), n);

            if(this->timing) {
                t_stop = steady_clock::now();
                chol3_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // R = R3 * R
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                upd3_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Test mode: Q = Q_buf * R3^{-1}
            //================================================================
            if(this->test_mode) {
                if(this->timing)
                    t_start = steady_clock::now();

                // Q_buf currently holds A * R1^{-1} * R2^{-1}
                // Apply R3^{-1}: Q_buf = Q_buf * R3^{-1} = A * R^{-1} = Q
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, G, n, Q_buf, m);

                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;  // Take ownership of Q_buf

                if(this->timing) {
                    t_stop = steady_clock::now();
                    q_mat_dur = duration_cast<microseconds>(t_stop - t_start).count();
                }
            }

            //================================================================
            // Finalize timing
            //================================================================
            if(this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q materialization from total (test overhead, not algorithmic cost)
                total_dur -= q_mat_dur;

                long rest_dur = total_dur - (alloc_dur + fwd1_dur + adj1_dur + chol1_dur + trsm1_dur + fwd_q_dur +
                                              syrk2_dur + chol2_dur + upd2_dur +
                                              syrk3_dur + chol3_dur + upd3_dur);

                // 15 entries
                this->times = {alloc_dur, fwd1_dur, adj1_dur, chol1_dur, trsm1_dur, fwd_q_dur,
                               syrk2_dur, chol2_dur, upd2_dur,
                               syrk3_dur, chol3_dur, upd3_dur,
                               q_mat_dur, rest_dur, total_dur};
            }

            // Cleanup
            delete[] G;
            delete[] R_temp;
            delete[] M;
            if(!this->test_mode)
                delete[] Q_buf;

            return 0;
        }
};

} // end namespace RandLAPACK
