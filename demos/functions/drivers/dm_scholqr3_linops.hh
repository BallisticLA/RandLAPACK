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

using namespace std::chrono;

namespace RandLAPACK_demos {

/// Shifted Cholesky QR3 algorithm for computing QR factorization via linear operators.
///
/// This algorithm performs three iterations of Cholesky QR, with the first iteration
/// using a diagonal shift to improve numerical stability:
///
///   1. Shifted CholQR1: G = A^T A + shift*I, R1 = chol(G), Q = A * R1^{-1}
///   2. CholQR2: G = Q^T Q, R2 = chol(G), Q = Q * R2^{-1}, R = R2 * R1
///   3. CholQR3: G = Q^T Q, R3 = chol(G), Q = Q * R3^{-1}, R = R3 * R
///
/// The shift is computed as: shift = 11 * eps * n * ||A||_F^2
///
/// Reference: Shifted Cholesky QR from Fukaya et al. and related work on
/// communication-avoiding QR algorithms.
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

        // Timing: [materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total]
        // 12 entries for detailed breakdown
        std::vector<long> times;

        sCholQR3_linops(
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
            steady_clock::time_point materialize_t_start, materialize_t_stop;
            steady_clock::time_point gram1_t_start, gram1_t_stop;
            steady_clock::time_point potrf1_t_start, potrf1_t_stop;
            steady_clock::time_point trsm1_t_start, trsm1_t_stop;
            steady_clock::time_point syrk2_t_start, syrk2_t_stop;
            steady_clock::time_point potrf2_t_start, potrf2_t_stop;
            steady_clock::time_point update2_t_start, update2_t_stop;
            steady_clock::time_point syrk3_t_start, syrk3_t_stop;
            steady_clock::time_point potrf3_t_start, potrf3_t_stop;
            steady_clock::time_point update3_t_start, update3_t_stop;
            steady_clock::time_point total_t_start, total_t_stop;
            steady_clock::time_point q_t_start, q_t_stop;
            long materialize_t_dur = 0;
            long gram1_t_dur = 0;
            long potrf1_t_dur = 0;
            long trsm1_t_dur = 0;
            long syrk2_t_dur = 0;
            long potrf2_t_dur = 0;
            long update2_t_dur = 0;
            long syrk3_t_dur = 0;
            long potrf3_t_dur = 0;
            long update3_t_dur = 0;
            long total_t_dur = 0;
            long q_t_dur = 0;

            if(this->timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            // Create identity matrix for materializing A
            T* I_mat = new T[n * n]();
            RandLAPACK::util::eye(n, n, I_mat);

            // Allocate buffer for Q (will hold A initially, then Q after each iteration)
            T* Q_buf = new T[m * n]();

            // Allocate buffer for Gram matrix / R-factor updates
            T* G = new T[n * n]();

            // Allocate buffer for temporary R factor
            T* R_temp = new T[n * n]();

            //================================================================
            // Step 1: Shifted Cholesky QR1
            //================================================================
            if(this->timing)
                materialize_t_start = steady_clock::now();

            // Materialize A: Q_buf = A * I
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, I_mat, n, (T)0.0, Q_buf, m);

            if(this->timing) {
                materialize_t_stop = steady_clock::now();
                gram1_t_start = steady_clock::now();
            }

            // Compute ||A||_F for the shift
            T norm_A = lapack::lange(Norm::Fro, m, n, Q_buf, m);

            // Compute shift = 11 * eps * n * ||A||_F^2
            T shift = 11 * std::numeric_limits<T>::epsilon() * n * std::pow(norm_A, 2);

            // Compute Gram matrix: G = A^T * A
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);

            // Add shift to diagonal: G = G + shift * I
            for (int64_t i = 0; i < n; ++i) {
                G[i * (n + 1)] += shift;
            }

            // Zero out lower triangle before Cholesky
            if (n > 1) {
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            }

            if(this->timing) {
                gram1_t_stop = steady_clock::now();
                potrf1_t_start = steady_clock::now();
            }

            // Cholesky factorization: G = R1^T * R1
            lapack::potrf(Uplo::Upper, n, G, n);

            // Copy R1 to R (accumulate R factor)
            lapack::lacpy(MatrixType::Upper, n, n, G, n, R, ldr);

            if(this->timing) {
                potrf1_t_stop = steady_clock::now();
                trsm1_t_start = steady_clock::now();
            }

            // Compute Q = A * R1^{-1}
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_buf, m);

            if(this->timing)
                trsm1_t_stop = steady_clock::now();

            //================================================================
            // Step 2: Cholesky QR2
            //================================================================
            if(this->timing)
                syrk2_t_start = steady_clock::now();

            // Compute Gram matrix: G = Q^T * Q
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);

            // Zero out lower triangle before Cholesky
            if (n > 1) {
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            }

            if(this->timing) {
                syrk2_t_stop = steady_clock::now();
                potrf2_t_start = steady_clock::now();
            }

            // Cholesky factorization: G = R2^T * R2
            lapack::potrf(Uplo::Upper, n, G, n);

            if(this->timing) {
                potrf2_t_stop = steady_clock::now();
                update2_t_start = steady_clock::now();
            }

            // Update R = R2 * R1 (G contains R2)
            // Copy current R to R_temp
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            // R = R2 * R_temp
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            // Compute Q = Q * R2^{-1}
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, m, n, (T)1.0, G, n, Q_buf, m);

            if(this->timing)
                update2_t_stop = steady_clock::now();

            //================================================================
            // Step 3: Cholesky QR3
            //================================================================
            if(this->timing)
                syrk3_t_start = steady_clock::now();

            // Compute Gram matrix: G = Q^T * Q
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T)1.0, Q_buf, m, (T)0.0, G, n);

            // Zero out lower triangle before Cholesky
            if (n > 1) {
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G[1], n);
            }

            if(this->timing) {
                syrk3_t_stop = steady_clock::now();
                potrf3_t_start = steady_clock::now();
            }

            // Cholesky factorization: G = R3^T * R3
            lapack::potrf(Uplo::Upper, n, G, n);

            if(this->timing) {
                potrf3_t_stop = steady_clock::now();
                update3_t_start = steady_clock::now();
            }

            // Update R = R3 * R (G contains R3)
            // Copy current R to R_temp
            lapack::lacpy(MatrixType::Upper, n, n, R, ldr, R_temp, n);
            // R = R3 * R_temp
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                       Diag::NonUnit, n, n, (T)1.0, G, n, R_temp, n);
            lapack::lacpy(MatrixType::Upper, n, n, R_temp, n, R, ldr);

            if(this->timing)
                update3_t_stop = steady_clock::now();

            // Only compute final Q if test mode is enabled (not needed for R-factor only)
            if(this->test_mode) {
                if(this->timing)
                    q_t_start = steady_clock::now();

                // Compute Q = Q * R3^{-1}
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, G, n, Q_buf, m);

                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = Q_buf;  // Take ownership of Q_buf

                if(this->timing)
                    q_t_stop = steady_clock::now();
            }

            if(this->timing) {
                total_t_stop = steady_clock::now();

                materialize_t_dur = duration_cast<microseconds>(materialize_t_stop - materialize_t_start).count();
                gram1_t_dur = duration_cast<microseconds>(gram1_t_stop - gram1_t_start).count();
                potrf1_t_dur = duration_cast<microseconds>(potrf1_t_stop - potrf1_t_start).count();
                trsm1_t_dur = duration_cast<microseconds>(trsm1_t_stop - trsm1_t_start).count();
                syrk2_t_dur = duration_cast<microseconds>(syrk2_t_stop - syrk2_t_start).count();
                potrf2_t_dur = duration_cast<microseconds>(potrf2_t_stop - potrf2_t_start).count();
                update2_t_dur = duration_cast<microseconds>(update2_t_stop - update2_t_start).count();
                syrk3_t_dur = duration_cast<microseconds>(syrk3_t_stop - syrk3_t_start).count();
                potrf3_t_dur = duration_cast<microseconds>(potrf3_t_stop - potrf3_t_start).count();
                update3_t_dur = duration_cast<microseconds>(update3_t_stop - update3_t_start).count();
                total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q-factor computation time if in test mode
                if(this->test_mode) {
                    q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
                    total_t_dur -= q_t_dur;
                }

                long rest_t_dur = total_t_dur - (materialize_t_dur + gram1_t_dur + potrf1_t_dur + trsm1_t_dur +
                                                 syrk2_t_dur + potrf2_t_dur + update2_t_dur +
                                                 syrk3_t_dur + potrf3_t_dur + update3_t_dur);

                // Fill the data vector: [materialize, gram1, potrf1, trsm1, syrk2, potrf2, update2, syrk3, potrf3, update3, rest, total]
                this->times = {materialize_t_dur, gram1_t_dur, potrf1_t_dur, trsm1_t_dur,
                               syrk2_t_dur, potrf2_t_dur, update2_t_dur,
                               syrk3_t_dur, potrf3_t_dur, update3_t_dur,
                               rest_t_dur, total_t_dur};
            }

            // Cleanup
            delete[] I_mat;
            delete[] G;
            delete[] R_temp;

            // Only delete Q_buf if not in test mode (otherwise Q owns it)
            if(!this->test_mode) {
                delete[] Q_buf;
            }

            return 0;
        }
};

} // end namespace RandLAPACK_demos
