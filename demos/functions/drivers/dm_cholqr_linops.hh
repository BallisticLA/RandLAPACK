#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"

#include <cstdint>
#include <vector>
#include <chrono>

using namespace std::chrono;

namespace RandLAPACK_demos {

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

        // 3 entries: gram_t_dur, chol_t_dur, total_t_dur
        std::vector<long> times;

        CholQR_linops(
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

        ~CholQR_linops() {
            if (Q != nullptr) {
                delete[] Q;
            }
        }

        /// Computes an R-factor of the unpivoted QR factorization using unpreconditioned Cholesky QR:
        ///     A = QR,
        /// where Q and R are of size m-by-n and n-by-n.
        ///
        /// This is the baseline unpreconditioned version for comparison with CQRRT.
        /// Algorithm:
        ///   1. Compute Gram matrix: G = A^T * A
        ///   2. Compute Cholesky factorization: G = R^T * R
        ///   3. (Optional) Compute Q = A * R^{-1}
        ///
        /// @note This algorithm expects A to be full-rank (rank = n). Rank-deficient inputs may result
        ///       in loss of orthogonality in the Q-factor (when test_mode=true) and numerical instability
        ///       in the R-factor.
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
            steady_clock::time_point gram_t_start;
            steady_clock::time_point gram_t_stop;
            steady_clock::time_point chol_t_start;
            steady_clock::time_point chol_t_stop;
            steady_clock::time_point total_t_start;
            steady_clock::time_point total_t_stop;
            long gram_t_dur  = 0;
            long chol_t_dur  = 0;
            long total_t_dur = 0;
            long q_t_dur     = 0;
            steady_clock::time_point q_t_start;
            steady_clock::time_point q_t_stop;

            if(this->timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            if(this->timing)
                gram_t_start = steady_clock::now();

            // Compute Gram matrix: R = A^T * A
            // We cannot use syrk since A may be a sparse operator
            // Instead, compute R = A^T * A via two matvec operations:
            // 1. Create identity matrix I (n x n)
            // 2. Compute A_temp = A * I (materializes A as m x n dense)
            // 3. Compute R = A^T * A_temp

            // Create identity matrix
            T* I_mat = new T[n * n]();
            RandLAPACK::util::eye(n, n, I_mat);

            // Allocate buffer for A materialized
            T* A_temp = new T[m * n]();

            // Step 1: Materialize A by computing A_temp = A * I
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, I_mat, n, (T)0.0, A_temp, m);

            // Step 2: Compute R = A^T * A_temp (using the linear operator's transpose)
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, A_temp, m, (T)0.0, R, ldr);

            // Zero out the lower triangle before Cholesky (potrf only uses upper triangle)
            if (n > 1) {
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &R[1], ldr);
            }

            if(this->timing) {
                gram_t_stop = steady_clock::now();
                chol_t_start = steady_clock::now();
            }

            // Compute Cholesky factorization: G = R^T * R
            // On exit, the upper triangle of R contains the R-factor
            lapack::potrf(Uplo::Upper, n, R, ldr);

            if(this->timing)
                chol_t_stop = steady_clock::now();

            // Compute Q-factor if test mode is enabled
            if(this->test_mode) {
                if(this->timing)
                    q_t_start = steady_clock::now();

                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = A_temp;  // Take ownership of A_temp buffer

                // Solve Q * R = A_temp for Q
                // Q = A * R^{-1}
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, R, ldr, this->Q, m);

                if(this->timing)
                    q_t_stop = steady_clock::now();
            }

            if(this->timing) {
                // Stop timing BEFORE cleanup operations to exclude deallocation costs
                total_t_stop = steady_clock::now();

                gram_t_dur  = duration_cast<microseconds>(gram_t_stop  - gram_t_start).count();
                chol_t_dur  = duration_cast<microseconds>(chol_t_stop  - chol_t_start).count();
                total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q-factor computation time if in test mode
                if(this->test_mode) {
                    q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
                    total_t_dur -= q_t_dur;
                }

                // Fill the data vector: [gram_time, chol_time, total_time]
                this->times = {gram_t_dur, chol_t_dur, total_t_dur};
            }

            // Cleanup
            delete[] I_mat;

            // Only delete A_temp if not in test mode (otherwise Q owns it)
            if(!this->test_mode) {
                delete[] A_temp;
            }

            return 0;
        }
};

} // end namespace RandLAPACK_demos
