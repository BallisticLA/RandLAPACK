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

using namespace std::chrono;

namespace RandLAPACK_demos {

template <typename T, typename RNG = RandBLAS::DefaultRNG>
class CQRRT_linops {
    public:

        bool timing;
        bool test_mode;
        T eps;
        int64_t rank;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // 6 entries
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
            steady_clock::time_point saso_t_stop;
            steady_clock::time_point saso_t_start;
            steady_clock::time_point qr_t_start;
            steady_clock::time_point qr_t_stop;
            steady_clock::time_point cholqr_t_start;
            steady_clock::time_point cholqr_t_stop;
            steady_clock::time_point a_mod_trsm_t_start;
            steady_clock::time_point a_mod_trsm_t_stop;
            steady_clock::time_point total_t_start;
            steady_clock::time_point total_t_stop;
            long saso_t_dur        = 0;
            long qr_t_dur          = 0;
            long rank_reveal_t_dur = 0;
            long cholqr_t_dur      = 0;
            long a_mod_piv_t_dur   = 0;
            long a_mod_trsm_t_dur  = 0;
            long total_t_dur       = 0;

            if(this -> timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            int i;
            int64_t d = d_factor * n;
            // Variables for a posteriori rank estimation.
            int64_t new_rank;
            T running_max, running_min, curr_entry;

            T* A_hat = new T[d * n]();
            T* tau   = new T[n]();

            if(this -> timing)
                saso_t_start = steady_clock::now();

            /// Generating a SASO
            //
            // We want an explicit sparse matrix S instead of just a SparseSkOp object in order
            // to be able to multiply S by a linear operator.
            // Below, is, however, pretty complicated.
            // In the future, I should implement sketching with RandBLAS object types in linops.
            RandBLAS::SparseDist DS(d, m, this->nnz);
            RandBLAS::SparseSkOp<T, RNG> S(DS, state);
            state = S.next_state;
            RandBLAS::fill_sparse(S);
            RandBLAS::CSRMatrix S_csr(S.n_rows, S.n_cols);
            RandBLAS::COOMatrix S_coo_view(S.n_rows, S.n_cols, S.nnz, S.vals, S.rows, S.cols);
            RandBLAS::coo_to_csr(S_coo_view, S_csr);

            // Below expression replaces applying a SASO from the left to a general linear operator (S * A).
            // Below, Side::Right is used because the operator is on the right side of the expression.
            A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, (T)1.0, S_csr, (T)0.0, A_hat, d);

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

            if(this -> timing)
                a_mod_trsm_t_start = steady_clock::now();

            //blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sk, ldr, A, lda);
            // TRSM, used in the original implementation of CQRRT, is only defined for dense operators.
            // If A is not just a general dense operator, we handle this step via an explicit inverse and a multiplication.

            // Explicitly invert R_sk
            lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_sk, n);

            // Allocate a buffer for A_pre
            T* A_pre = new T[m * n]();

            // Multiply A * (R_sk)^-1, where A is a GLO and (R_sk)^-1 is a dense matrix.
            A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, R_sk, n, (T)0.0, A_pre, m);

            if(this -> timing) {
                a_mod_trsm_t_stop = steady_clock::now();
                cholqr_t_start = steady_clock::now();
            }

            // Do Cholesky QR
            //blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, lda, 0.0, R_sk, ldr);
            // Since SYRK, used in the original implementation of CQRRT, is not defined for non-dense operators, perform an explicit ((R_sk)^-1)^T * A^T * A * (R_sk)^-1

            // A^T * A_pre = A^T * A * (R_sk)^-1
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, A_pre, m, (T)0.0, R, ldr);

            // (R_sk^-1)^T * (A^T * A_pre)
            blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit, n, n, (T) 1.0, R_sk, n, R, ldr);
            // Cholesky factorization
            lapack::potrf(Uplo::Upper, n, R, ldr);

            // Estimate rank after we have the R-factor form Cholesky QR.
            // The strategy here is the same as in naive rank estimation.
            // This also automatically takes care of any potential failures in Cholesky factorization.
            // Note that the diagonal of R may not be sorted, so we need to keep the running max/min
            new_rank = n;
            running_max = R[0];
            running_min = R[0];
            T cond_threshold = std::sqrt(this->eps / std::numeric_limits<T>::epsilon());

            for(i = 0; i < n; ++i) {
                curr_entry = std::abs(R[i * ldr + i]);
                running_max = std::max(running_max, curr_entry);
                running_min = std::min(running_min, curr_entry);
                if((running_min * cond_threshold < running_max) && i > 1) {
                    new_rank = i - 1;
                    break;
                }
            }

            // Set the rank parameter to the value computed a posteriori.
            this->rank = new_rank;

            // Compute Q-factor if test mode is enabled
            if(this->test_mode) {
                // Allocate Q: m x n matrix
                this->Q_rows = m;
                this->Q_cols = n;
                this->Q = new T[m * n]();

                // Copy A_pre to Q
                lapack::lacpy(MatrixType::General, m, n, A_pre, m, this->Q, m);

                // Solve Q * R_sk = A_pre for Q, i.e., Q = A_pre * (R_sk)^{-1}
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, m, n, (T)1.0, R_sk, n, this->Q, m);
            }

            if(this -> timing)
                cholqr_t_stop = steady_clock::now();

            // Get the final R-factor - undoing the preconditioning
            blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, 1.0, A_hat, d, R, ldr);

            if(this -> timing) {
                saso_t_dur       = duration_cast<microseconds>(saso_t_stop       - saso_t_start).count();
                qr_t_dur         = duration_cast<microseconds>(qr_t_stop         - qr_t_start).count();
                a_mod_trsm_t_dur = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
                cholqr_t_dur     = duration_cast<microseconds>(cholqr_t_stop     - cholqr_t_start).count();

                total_t_stop = steady_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest  = total_t_dur - (saso_t_dur + qr_t_dur + cholqr_t_dur + a_mod_trsm_t_dur);

                // Fill the data vector
                this -> times = {saso_t_dur, qr_t_dur, cholqr_t_dur, a_mod_trsm_t_dur, t_rest, total_t_dur};
            }

            delete[] A_hat;
            delete[] tau;
            delete[] R_sk;
            delete[] A_pre;

            return 0;
        }
};
} // end namespace RandLAPACK
