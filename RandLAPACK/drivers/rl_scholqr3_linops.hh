#pragma once

#include "rl_cholqr.hh"
#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <chrono>
#include <cstdint>
#include <limits>
#include <vector>

using namespace std::chrono;

namespace RandLAPACK {

/// Shifted Cholesky QR3 (Fukaya et al., SISC 2020) for abstract linear operators.
///
///   pass 1:    shifted CholQR, s = shift_factor_iter1 * trace(G)          -> R_1
///   pass 2, 3: CholQR preconditioned on the previous factor (trsm_identity) -> R
///
/// The Gram of every pass is formed in column blocks through A * B and A^T * B, so peak memory
/// is O(n^2 + (m + n) b) and at most m x b of A is formed at a time when b = block_size < n.
/// With test_mode on, Q = A * R^{-1} is formed after the factorization, outside the timed region.
///
/// shift_factor_iter1 < 0 (the default) selects 11 * n * eps, main's choice (Fukaya et al.
/// prescribe 11 (m n + n (n + 1)) u ||A||_2^2, which grows with m); any other value is passed on
/// as given, and NaN fails the call with kCholQRInvalidShift.
/// Passes 2 and 3 are unshifted (shift_factor_iter23 = 0) so that they drive orthogonality to
/// machine level; the adaptive retry shifts them only if potrf breaks down.
///
/// call() returns 0 on success, or the 1-based pass that failed (1, 2 or 3; an invalid shift
/// factor fails as pass 1 before any work); chol_status then holds cholqr_primitive's return
/// code for that pass.
///
/// times (18 entries):
/// [0]  alloc (always 0)
/// [1]  fwd1     [2]  adj1    [3]  chol1   [4]  upd1 (always 0; R_1^{-1} is formed in upd2)
/// [5]  fwd2     [6]  adj2    [7]  left2   [8]  chol2   [9]  upd2
/// [10] fwd3     [11] adj3    [12] left3   [13] chol3   [14] upd3
/// [15] q_mat    [16] rest    [17] total
/// For an operator that provides materialize_cols, fwd1 times its column copy.
template <typename T>
class sCholQR3_linops : public CholQRLinopsBase<T, 3> {
    public:
        T    shift_factor_iter1  = T(-1);  ///< pass-1 shift factor; < 0 selects 11 * n * eps, NaN fails
        T    shift_factor_iter23 = T(0);   ///< first-attempt shift factor of passes 2 and 3; no sentinel:
                                           ///< negative or NaN fails with kCholQRInvalidShift
        bool gram_left_gemm      = false;  ///< passes 2-3: apply R_pre^T by GEMM per block instead of by TRSM

        /// @param ep  unused; kept for call-site compatibility with the other QR drivers.
        sCholQR3_linops(bool time_subroutines, T ep, bool enable_test_mode = false)
            : CholQRLinopsBase<T, 3>(time_subroutines, enable_test_mode) { (void)ep; }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, T* R, int64_t ldr) {
            const T sf1 = (shift_factor_iter1 < T(0))
                        ? T(11) * T(A.n_cols) * std::numeric_limits<T>::epsilon()
                        : shift_factor_iter1;
            return this->run_passes(A, R, ldr, sf1, shift_factor_iter23, gram_left_gemm);
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
///   - NoTrans: W = A * I (materialization for Gram computation; A.materialize_cols when A
///     provides it)
///   - Trans:   G1 = A^T * W (Gram matrix)
///   - NoTrans: Q = A * R1^{-1} (first Q-factor)
///
/// After the first Q-factor, iterations 2-3 use dense syrk on Q (no further linop calls).
/// This is theoretically distinct from sCholQR3_linops (fully-blocked), which recomputes
/// each Gram through the linop and forms at most m x block_size of A at a time.
///
/// Peak memory: O(m*n + n^2): Q is explicitly stored as m x n dense.
///
/// Reference: Shifted Cholesky QR from Fukaya et al. (SISC, 2020).
///
template <typename T>
class sCholQR3_linops_basic {
    public:

        bool timing;
        bool test_mode;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // Timing breakdown (15 entries):
        // [0]  alloc      - buffer allocation
        // [1]  fwd1       - NoTrans: W = A * I (m x n), or A.materialize_cols
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
        /// Total wall-clock (us) of the last call(), or -1 if it was untimed or failed.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        sCholQR3_linops_basic(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            (void)ep;   // kept for call-site compatibility with the other QR drivers
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

        sCholQR3_linops_basic(const sCholQR3_linops_basic&) = delete;
        sCholQR3_linops_basic& operator=(const sCholQR3_linops_basic&) = delete;

        /// Computes the QR factorization A = QR using shifted Cholesky QR3 (basic variant).
        ///
        /// @param[in] A
        ///     The m-by-n linear operator (m and n read from A.n_rows, A.n_cols).
        ///
        /// @param[out] R
        ///     Pre-allocated n-by-n buffer. On exit, stores the upper-triangular
        ///     R factor. Zero entries are not compressed.
        ///
        /// @param[in] ldr
        ///     Leading dimension of R.
        ///
        /// @return = 0: successful exit
        ///         = 1, 2, 3: that iteration's Cholesky failed (a non-PD pivot or a non-finite factor)
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

            this->times.clear();
            if(this->timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            if(this->timing)
                t_start = steady_clock::now();

            // ---- Allocate buffers ----
            // Q_buf: m x n, the materialized operator, updated in-place through iterations
            T* Q_buf = new T[m * n];

            // G: n x n Gram matrix / Cholesky workspace (zero-init for lower triangle)
            T* G = new T[n * n]();

            // R_temp: n x n workspace for R accumulation via trmm
            T* R_temp = new T[n * n]();

            // M: n x n, starts as identity, becomes R1^{-1} for Q materialization
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
            if constexpr (RandLAPACK::linops::HasMaterializeCols<GLO>)
                A.materialize_cols(0, n, Q_buf, m);
            else
                A(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  m, n, n, (T)1.0, M, n, (T)0.0, Q_buf, m);
            if(this->timing) { t_stop = steady_clock::now(); fwd1_dur = duration_cast<microseconds>(t_stop - t_start).count(); }

            // Linop access 2: G1 = A^T * W (Trans, n x n Gram matrix)
            if(this->timing) t_start = steady_clock::now();
            A(Layout::ColMajor, Op::Trans, Op::NoTrans,
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
            if (potrf_upper_finite(n, G, n)) {
                delete[] Q_buf; delete[] G; delete[] R_temp; delete[] M;
                return 1;
            }

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
            A(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
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
            if (potrf_upper_finite(n, G, n)) {
                delete[] Q_buf; delete[] G; delete[] R_temp; delete[] M;
                return 2;
            }

            if(this->timing) {
                t_stop = steady_clock::now();
                chol2_dur = duration_cast<microseconds>(t_stop - t_start).count();
                t_start = steady_clock::now();
            }

            // Q_buf *= R2^{-1} (m x n trsm, updates Q in-place)
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
            if (potrf_upper_finite(n, G, n)) {
                delete[] Q_buf; delete[] G; delete[] R_temp; delete[] M;
                return 3;
            }

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
                delete[] this->Q;  // a previous call's Q
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
