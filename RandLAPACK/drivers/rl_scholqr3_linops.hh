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

        // Timing breakdown (13 entries):
        // [0] alloc       - buffer allocation
        // [1] gram1       - iteration 1 blocked Gram (A^T A via linop, includes shift)
        // [2] potrf1      - iteration 1 Cholesky factorization
        // [3] m_update1   - form M = R1^{-1} (n x n trsm)
        // [4] gram2       - iteration 2 blocked Gram (M^T A^T A M via linop)
        // [5] potrf2      - iteration 2 Cholesky factorization
        // [6] update2     - R = R2*R1, M = M*R2^{-1}
        // [7] gram3       - iteration 3 blocked Gram (M^T A^T A M via linop)
        // [8] potrf3      - iteration 3 Cholesky factorization
        // [9] update3     - R = R3*R
        // [10] q_mat      - Q materialization for test mode (0 if not test_mode)
        // [11] rest       - unaccounted time
        // [12] total      - wall-clock total
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
            long alloc_dur = 0, gram1_dur = 0, potrf1_dur = 0, m_update1_dur = 0;
            long gram2_dur = 0, potrf2_dur = 0, update2_dur = 0;
            long gram3_dur = 0, potrf3_dur = 0, update3_dur = 0;
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
            if(this->timing)
                t_start = steady_clock::now();

            for (int64_t j = 0; j < n; j += b_eff) {
                int64_t b_j = std::min(b_eff, n - j);

                // W = A * M[:, j:j+b]  (= A * I[:, j:j+b] since M = I)
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, A_temp, m);

                // G[:, j:j+b] = A^T * W  (direct to G since M = I)
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                  n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n);
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
                t_stop = steady_clock::now();
                gram1_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                potrf1_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Update M: M = I * R1^{-1} = R1^{-1}
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, M, n);

            if(this->timing) {
                t_stop = steady_clock::now();
                m_update1_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 2: Cholesky QR
            //================================================================
            // Blocked Gram: G2 = M^T * A^T * A * M  where M = R1^{-1}
            if(this->timing)
                t_start = steady_clock::now();

            for (int64_t j = 0; j < n; j += b_eff) {
                int64_t b_j = std::min(b_eff, n - j);

                // W = A * M[:, j:j+b]
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, A_temp, m);

                // Z = A^T * W
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                  n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);

                // G[:, j:j+b] = M^T * Z
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                           n, b_j, n, (T)1.0, M, n, Z_buf, n, (T)0.0, G + j * n, n);
            }

            if(this->timing) {
                t_stop = steady_clock::now();
                gram2_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                potrf2_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                update2_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

            //================================================================
            // Iteration 3: Cholesky QR
            //================================================================
            // Blocked Gram: G3 = M^T * A^T * A * M  where M = R1^{-1} * R2^{-1}
            if(this->timing)
                t_start = steady_clock::now();

            for (int64_t j = 0; j < n; j += b_eff) {
                int64_t b_j = std::min(b_eff, n - j);

                // W = A * M[:, j:j+b]
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, b_j, n, (T)1.0, M + j * n, n, (T)0.0, A_temp, m);

                // Z = A^T * W
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                  n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);

                // G[:, j:j+b] = M^T * Z
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                           n, b_j, n, (T)1.0, M, n, Z_buf, n, (T)0.0, G + j * n, n);
            }

            if(this->timing) {
                t_stop = steady_clock::now();
                gram3_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                potrf3_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // R = R3 * R
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                update3_dur = duration_cast<microseconds>(t_stop - t_start).count();
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

                long rest_dur = total_dur - (alloc_dur + gram1_dur + potrf1_dur + m_update1_dur +
                                              gram2_dur + potrf2_dur + update2_dur +
                                              gram3_dur + potrf3_dur + update3_dur);

                // 13 entries: alloc, gram1, potrf1, m_update1, gram2, potrf2, update2,
                //             gram3, potrf3, update3, q_mat, rest, total
                this->times = {alloc_dur, gram1_dur, potrf1_dur, m_update1_dur,
                               gram2_dur, potrf2_dur, update2_dur,
                               gram3_dur, potrf3_dur, update3_dur,
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

        // Timing breakdown (13 entries):
        // [0] alloc       - buffer allocation
        // [1] gram1       - W = A*I (NoTrans) + G1 = A^T*W (Trans) + shift computation
        // [2] potrf1      - iteration 1 Cholesky factorization
        // [3] q_factor    - M = R1^{-1} (n x n trsm) + Q_buf = A*M (NoTrans linop, m x n)
        // [4] syrk2       - G2 = Q_buf^T * Q_buf (dense syrk)
        // [5] potrf2      - iteration 2 Cholesky factorization
        // [6] update2     - Q_buf *= R2^{-1} (m x n trsm), R = R2*R1 (trmm)
        // [7] syrk3       - G3 = Q_buf^T * Q_buf (dense syrk)
        // [8] potrf3      - iteration 3 Cholesky factorization
        // [9] update3     - R = R3*R (trmm)
        // [10] q_mat      - test mode: Q_buf *= R3^{-1} (m x n trsm), 0 otherwise
        // [11] rest       - unaccounted time
        // [12] total      - wall-clock total
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
            long alloc_dur = 0, gram1_dur = 0, potrf1_dur = 0, q_factor_dur = 0;
            long syrk2_dur = 0, potrf2_dur = 0, update2_dur = 0;
            long syrk3_dur = 0, potrf3_dur = 0, update3_dur = 0;
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
            if(this->timing)
                t_start = steady_clock::now();

            // Linop access 1: W = A * I (NoTrans, materializes operator as m x n dense)
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              m, n, n, (T)1.0, M, n, (T)0.0, Q_buf, m);

            // Linop access 2: G1 = A^T * W (Trans, n x n Gram matrix)
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);

            // Compute shift from ||A||_F^2 = trace(G)
            T norm_A_sq = 0;
            for (int64_t i = 0; i < n; ++i)
                norm_A_sq += G[i * (n + 1)];
            T shift = 11 * std::numeric_limits<T>::epsilon() * n * norm_A_sq;

            // Add shift to diagonal: G = G + shift * I
            for (int64_t i = 0; i < n; ++i)
                G[i * (n + 1)] += shift;

            if(this->timing) {
                t_stop = steady_clock::now();
                gram1_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                potrf1_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Compute M = I * R1^{-1} = R1^{-1}
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, M, n);

            // Linop access 3: Q_buf = A * M = A * R1^{-1} (NoTrans, m x n)
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              m, n, n, (T)1.0, M, n, (T)0.0, Q_buf, m);

            if(this->timing) {
                t_stop = steady_clock::now();
                q_factor_dur = duration_cast<microseconds>(t_stop - t_start).count();
            }

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
                potrf2_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                update2_dur = duration_cast<microseconds>(t_stop - t_start).count();
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
                potrf3_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // R = R3 * R
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            if(this->timing) {
                t_stop = steady_clock::now();
                update3_dur = duration_cast<microseconds>(t_stop - t_start).count();
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

                long rest_dur = total_dur - (alloc_dur + gram1_dur + potrf1_dur + q_factor_dur +
                                              syrk2_dur + potrf2_dur + update2_dur +
                                              syrk3_dur + potrf3_dur + update3_dur);

                // 13 entries: alloc, gram1, potrf1, q_factor, syrk2, potrf2, update2,
                //             syrk3, potrf3, update3, q_mat, rest, total
                this->times = {alloc_dur, gram1_dur, potrf1_dur, q_factor_dur,
                               syrk2_dur, potrf2_dur, update2_dur,
                               syrk3_dur, potrf3_dur, update3_dur,
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
