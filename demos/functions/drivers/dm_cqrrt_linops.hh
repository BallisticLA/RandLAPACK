#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"
#include "rl_bqrrp.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>
#include <iomanip>

using namespace std::chrono;

namespace RandLAPACK_demos {

template <typename T, typename RNG = RandBLAS::DefaultRNG>
class CQRRT_linops {
    public:

        bool timing;
        bool test_mode;
        T eps;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // 10 entries: saso, qr, trtri, linop_precond, linop_gram, trmm_gram, potrf, finalize, rest, total
        std::vector<long> times;

        // tuning SASOS
        int64_t nnz;

        CQRRT_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            nnz = 2;
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
        }

        ~CQRRT_linops() {
            if (Q != nullptr) {
                delete[] Q;
            }
        }

        /// Computes an R-factor of the unpivoted QR factorization of the form:
        ///     A= QR,
        /// where Q and R are of size m-by-n and n-by-n;
        /// operates similarly to ../../../RandLAPACK/drivers/rl_cqrrt.hh, but returns a Q-less factorization
        /// and accepts linear operators.
        ///
        /// @note This algorithm expects A to be full-rank (rank = n). Rank-deficient inputs may result
        ///       in loss of orthogonality in the Q-factor (when test_mode=true) and numerical instability
        ///       in the R-factor.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-n linear operator A.
        ///
        /// @param[in] d
        ///     Embedding dimension of a sketch, m >= d >= n.
        ///
        /// @param[in] R
        ///     Represents the upper-triangular R factor of QR factorization.
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[in] state
        ///     RNG state parameter, required for sketching operator generation.
        ///
        /// @param[out] A
        ///     Same as on input.
        ///
        /// @param[out] R
        ///     Stores n-by-n matrix with upper-triangular R factor.
        ///     Zero entries are not compressed.
        ///
        /// @return = 0: successful exit
        ///

        // Templated CQRRT call that accepts any linear operator.
        // This particular implementation aims to work with the combined linear operator.
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) {
            ///--------------------TIMING VARS--------------------/
            steady_clock::time_point saso_t_start, saso_t_stop;
            steady_clock::time_point qr_t_start, qr_t_stop;
            steady_clock::time_point trtri_t_start, trtri_t_stop;
            steady_clock::time_point linop_precond_t_start, linop_precond_t_stop;
            steady_clock::time_point linop_gram_t_start, linop_gram_t_stop;
            steady_clock::time_point trmm_gram_t_start, trmm_gram_t_stop;
            steady_clock::time_point potrf_t_start, potrf_t_stop;
            steady_clock::time_point finalize_t_start, finalize_t_stop;
            steady_clock::time_point total_t_start, total_t_stop;
            steady_clock::time_point q_t_start, q_t_stop;
            long saso_t_dur         = 0;
            long qr_t_dur           = 0;
            long trtri_t_dur        = 0;
            long linop_precond_t_dur = 0;
            long linop_gram_t_dur   = 0;
            long trmm_gram_t_dur    = 0;
            long potrf_t_dur        = 0;
            long finalize_t_dur     = 0;
            long total_t_dur        = 0;
            long q_t_dur            = 0;

            if(this -> timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            int64_t d = d_factor * n;

            T* A_hat = new T[d * n]();
            T* tau   = new T[n]();

            if(this -> timing)
                saso_t_start = steady_clock::now();

            /// Generating a SASO and applying it to the linear operator
            //
            // Create and fill a sparse sketching operator, then apply it directly.
            // The linear operator's operator() overload for SkOp handles the multiplication.
            RandBLAS::SparseDist DS(d, m, this->nnz);
            RandBLAS::SparseSkOp<T, RNG> S(DS, state);
            state = S.next_state;
            RandBLAS::fill_sparse(S);

            // Compute A_hat = S * A (sketch from the left)
            // Side::Right means the operator A is on the right side: C = op(S) * op(A)
            A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, (T)1.0, S, (T)0.0, A_hat, d);

            if(this -> timing) {
                saso_t_stop = steady_clock::now();
                qr_t_start = steady_clock::now();
            }

            /// Performing QR on a sketch
            lapack::geqrf(d, n, A_hat, d, tau);

            if(this -> timing)
                qr_t_stop = steady_clock::now();

            /// Extracting a k by k R representation
            T* R_sk  = new T[n * n]();
            lapack::lacpy(MatrixType::Upper, n, n, A_hat, d, R_sk, n);

            //blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sk, ldr, A, lda);
            // TRSM, used in the original implementation of CQRRT, is only defined for dense operators.
            // If A is not just a general dense operator, we handle this step via an explicit inverse and a multiplication.

            // Explicitly invert R_sk to get R_sk_inv
            if(this -> timing)
                trtri_t_start = steady_clock::now();

            // Try TRSM with identity rhs
            lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_sk, n);

            if(this -> timing) {
                trtri_t_stop = steady_clock::now();
                linop_precond_t_start = steady_clock::now();
            }

            T* R_sk_inv = R_sk;  // Rename for clarity - R_sk is now inverted

            // Allocate a buffer for A_pre
            T* A_pre = new T[m * n]();

            // Multiply A * R_sk_inv, where A is a GLO and R_sk_inv is a dense matrix.
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, R_sk_inv, n, (T)0.0, A_pre, m);

            if(this -> timing) {
                linop_precond_t_stop = steady_clock::now();
                linop_gram_t_start = steady_clock::now();
            }

            // Do Cholesky QR
            //blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, lda, 0.0, R_sk, ldr);
            // Since SYRK, used in the original implementation of CQRRT, is not defined for non-dense operators, perform an explicit ((R_sk)^-1)^T * A^T * A * (R_sk)^-1

            // A^T * A_pre = A^T * A * R_sk_inv
            // Try CSR format
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, A_pre, m, (T)0.0, R, ldr);

            if(this -> timing) {
                linop_gram_t_stop = steady_clock::now();
                trmm_gram_t_start = steady_clock::now();
            }

            // (R_sk_inv)^T * (A^T * A_pre)
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit, n, n, (T) 1.0, R_sk_inv, n, R, ldr);

            if(this -> timing) {
                trmm_gram_t_stop = steady_clock::now();
                potrf_t_start = steady_clock::now();
            }

            // Cholesky factorization (only reads/writes upper triangle)
            lapack::potrf(Uplo::Upper, n, R, ldr);

            if(this -> timing)
                potrf_t_stop = steady_clock::now();

            // Compute Q-factor if test mode is enabled (NOT included in cholqr timing)
            if(this->test_mode) {
                if(this->timing)
                    q_t_start = steady_clock::now();

                // Reuse A_pre storage for Q (Q = A_pre * R_chol^{-1})
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = A_pre;  // Take ownership of A_pre buffer

                // Solve Q * R_chol = A_pre for Q (R_chol is upper triangular from Cholesky)
                // Q = A * (R_chol * R_sk)^{-1} = A * R_sk^{-1} * R_chol^{-1} = A_pre * R_chol^{-1}
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, R, ldr, this->Q, m);

                if(this->timing)
                    q_t_stop = steady_clock::now();
            }

            // Zero out strictly lower triangle of R before final trmm
            // trmm expects R to be upper triangular on input (it preserves triangular structure)
            // The lower triangle may contain garbage from the Gram matrix computation
            // Use laset with beta=1.0 to preserve diagonal while zeroing strictly lower triangle
            if (n > 1) {
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &R[1], ldr);
            }

            if(this -> timing)
                finalize_t_start = steady_clock::now();

            // Get the final R-factor - undoing the preconditioning
            // R := R_chol * R_sk, where R_sk is the upper triangle of A_hat
            // trmm with Uplo::Upper only reads upper triangle of A_hat (can use ld=d directly)
            // and expects R to be upper triangular on input
            blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, 1.0, A_hat, d, R, ldr);

            if(this -> timing)
                finalize_t_stop = steady_clock::now();

            if(this -> timing) {
                // Stop timing BEFORE cleanup operations to exclude deallocation costs.
                // Note: First iteration may show inflated times due to cold cache effects.
                total_t_stop = steady_clock::now();

                saso_t_dur         = duration_cast<microseconds>(saso_t_stop         - saso_t_start).count();
                qr_t_dur           = duration_cast<microseconds>(qr_t_stop           - qr_t_start).count();
                trtri_t_dur        = duration_cast<microseconds>(trtri_t_stop        - trtri_t_start).count();
                linop_precond_t_dur = duration_cast<microseconds>(linop_precond_t_stop - linop_precond_t_start).count();
                linop_gram_t_dur   = duration_cast<microseconds>(linop_gram_t_stop   - linop_gram_t_start).count();
                trmm_gram_t_dur    = duration_cast<microseconds>(trmm_gram_t_stop    - trmm_gram_t_start).count();
                potrf_t_dur        = duration_cast<microseconds>(potrf_t_stop        - potrf_t_start).count();
                finalize_t_dur     = duration_cast<microseconds>(finalize_t_stop     - finalize_t_start).count();

                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q-factor computation time if in test mode
                if(this->test_mode) {
                    q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
                    total_t_dur -= q_t_dur;
                }

                long t_rest  = total_t_dur - (saso_t_dur + qr_t_dur + trtri_t_dur + linop_precond_t_dur +
                                              linop_gram_t_dur + trmm_gram_t_dur + potrf_t_dur + finalize_t_dur);

                // Fill the data vector (10 entries)
                // Index: 0=saso, 1=qr, 2=trtri, 3=linop_precond, 4=linop_gram, 5=trmm_gram, 6=potrf, 7=finalize, 8=rest, 9=total
                this -> times = {saso_t_dur, qr_t_dur, trtri_t_dur, linop_precond_t_dur,
                                 linop_gram_t_dur, trmm_gram_t_dur, potrf_t_dur, finalize_t_dur,
                                 t_rest, total_t_dur};
            }

            // Cleanup - now outside the timing region to avoid timing artifacts
            delete[] A_hat;
            delete[] tau;
            delete[] R_sk;

            // Only delete A_pre if not in test mode (otherwise Q owns it)
            if(!this->test_mode) {
                delete[] A_pre;
            }

            return 0;
        }
};
} // end namespace RandLAPACK
