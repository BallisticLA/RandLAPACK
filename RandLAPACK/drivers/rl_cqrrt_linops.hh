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

namespace RandLAPACK {

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

        // If true, use a dense Gaussian sketching operator instead of sparse SASO.
        // Dense sketches avoid potential issues with rank-deficient sketches but
        // require O(d*m) storage and O(d*m*n) work for application.
        bool use_dense_sketch;

        // Column-block size for the precondition + Gram computation.
        //
        // When block_size > 0, the two expensive linear operator calls:
        //   (1) A_pre = A * R_sk_inv       (m × n)
        //   (2) R     = A^T * A_pre         (n × n)
        // are fused into a column-block loop that processes b columns at a time:
        //   for each column block j of width b:
        //     buf (m × b) = A * R_sk_inv[:, j*b : (j+1)*b]
        //     R[:, j*b : (j+1)*b] = A^T * buf
        //
        // This reduces peak memory from O(m*n) to O(m*b), which is significant
        // when m is large and n is moderate.  The result is mathematically
        // identical — each column block of R is:
        //   R[:, j_block] = A^T * (A * R_sk_inv[:, j_block])
        // which equals the corresponding columns of A^T * A * R_sk_inv.
        //
        // When block_size <= 0 or block_size >= n, the full m × n buffer is
        // allocated and the original (non-blocked) path is used.
        //
        // When test_mode is enabled and blocking is active, the Q-factor
        // computation (which needs the full m × n A_pre) is handled by
        // recomputing A_pre = A * R_sk_inv after the Gram loop.  This
        // recomputation is outside the timing region, so it does not
        // affect benchmark results.
        int64_t block_size;

        CQRRT_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            nnz = 2;
            use_dense_sketch = false;
            block_size = 0;
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
        /// operates similarly to rl_cqrrt.hh, but returns a Q-less factorization
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

            /// Generate and apply the sketching operator to the linear operator.
            // Side::Right means the operator A is on the right side: C = op(S) * op(A)
            if (this->use_dense_sketch) {
                // Dense Gaussian sketch: allocate d x m buffer, fill with Gaussian entries.
                RandBLAS::DenseDist DD(d, m);
                RandBLAS::DenseSkOp<T, RNG> S(DD, state);
                state = S.next_state;
                RandBLAS::fill_dense(S);
                A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, (T)1.0, S, (T)0.0, A_hat, d);
            } else {
                // Sparse SASO sketch: uses nnz nonzeros per column.
                RandBLAS::SparseDist DS(d, m, this->nnz);
                RandBLAS::SparseSkOp<T, RNG> S(DS, state);
                state = S.next_state;
                RandBLAS::fill_sparse(S);
                A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, (T)1.0, S, (T)0.0, A_hat, d);
            }

            if(this -> timing) {
                saso_t_stop = steady_clock::now();
                qr_t_start = steady_clock::now();
            }

            /// Performing QR on a sketch
            lapack::geqrf(d, n, A_hat, d, tau);

            if(this -> timing)
                qr_t_stop = steady_clock::now();

            //blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sk, ldr, A, lda);
            // TRSM, used in the original implementation of CQRRT, is only defined for dense operators.
            // If A is not just a general dense operator, we handle this step via an explicit inverse and a multiplication.

            // Explicitly invert R_sk to get R_sk_inv
            if(this -> timing)
                trtri_t_start = steady_clock::now();

            // Instead of doing TRTRI to find R_sk_inv, we do TRSM with an identity, since trtri is not optimized in MKL
            //lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_sk, n);
            //T* R_sk_inv = R_sk;  // Rename for clarity - R_sk is now inverted
            T* Eye = new T[n * n]();
            RandLAPACK::util::eye(n, n, Eye);
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, 1.0, A_hat, d, Eye, n);
            if (n > 1) {
                // Clear the below-diagonal
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &Eye[1], n);
            }
            T* R_sk_inv = Eye;

            if(this -> timing) {
                trtri_t_stop = steady_clock::now();
                linop_precond_t_start = steady_clock::now();
            }

            // ================================================================
            // Precondition + Gram computation: R = A^T * A * R_sk_inv
            // ================================================================
            //
            // This section computes the Gram matrix for Cholesky QR:
            //   R = (R_sk_inv)^T * A^T * A * R_sk_inv
            // The (R_sk_inv)^T left-multiply is handled later via TRMM.
            // Here we compute the inner product:  R = A^T * (A * R_sk_inv).
            //
            // Two paths are available:
            //
            // (a) FULL MATERIALIZATION (original):
            //     Allocate m × n buffer A_pre, compute A_pre = A * R_sk_inv,
            //     then R = A^T * A_pre.  Memory: O(m*n).
            //
            // (b) COLUMN-BLOCK PROCESSING (memory-efficient):
            //     Process b columns at a time.  For each column block j:
            //       buf (m × b_j) = A * R_sk_inv[:, j : j+b_j]
            //       R[:, j : j+b_j] = A^T * buf
            //     Memory: O(m*b).  Never forms the full A_pre.
            //     The result is mathematically identical because:
            //       R[:, j_block] = A^T * (A * R_sk_inv[:, j_block])
            //     is the j-th column block of A^T * A * R_sk_inv.
            //
            //     All operator types (Dense, Sparse, Composite, CholSolver)
            //     support arbitrary column counts in operator(), so no
            //     modifications to the linear operator interface are needed.
            //
            //     When test_mode is on, the Q-factor (Q = A_pre * R_chol^{-1})
            //     needs the full m × n A_pre.  In that case, A_pre is
            //     recomputed after the Gram loop, outside the timing region.
            //     This costs an extra operator application but does NOT affect
            //     the benchmark timings.
            //
            // ================================================================

            // Determine effective block width.
            // block_size <= 0 or >= n means "no blocking" (full width).
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // A_pre buffer:
            //   Full path:  m × n  (kept alive for Q-factor in test_mode)
            //   Block path: m × b_eff  (temporary, freed after Gram loop;
            //                           if test_mode, a full m × n buffer is
            //                           allocated later for Q computation)
            T* A_pre = new T[m * b_eff]();

            if (b_eff == n) {
                // --- Full materialization path (original) ---

                // Step 1: A_pre (m × n) = A * R_sk_inv (m × n × n)
                //   A is m × n, R_sk_inv is n × n, A_pre is m × n.
                //   Side::Left: C = alpha * op(A) * op(B) + beta * C
                //   where op(A) = A (m × n), op(B) = R_sk_inv (n × n), C = A_pre (m × n).
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, R_sk_inv, n, (T)0.0, A_pre, m);

                if(this -> timing) {
                    linop_precond_t_stop = steady_clock::now();
                    linop_gram_t_start = steady_clock::now();
                }

                // Step 2: R (n × n) = A^T * A_pre (n × m × m × n = n × n)
                //   Since SYRK is not defined for non-dense operators, we use
                //   an explicit A^T * A_pre to form the Gram matrix.
                //   op(A) = A^T (n × m), op(B) = A_pre (m × n), C = R (n × n).
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, A_pre, m, (T)0.0, R, ldr);

                if(this -> timing) {
                    linop_gram_t_stop = steady_clock::now();
                }
            } else {
                // --- Column-block processing path (memory-efficient) ---
                //
                // Process n columns in blocks of width b_eff.
                // The last block may be narrower if n is not divisible by b_eff.
                //
                // For each block starting at column j with width b_j:
                //   (1) buf (m × b_j) = A * R_sk_inv[:, j : j+b_j]
                //       R_sk_inv is n × n in ColMajor with ld = n.
                //       Column j starts at R_sk_inv + j * n.
                //       We multiply A (m × n) by this n × b_j slice.
                //
                //   (2) R[:, j : j+b_j] (n × b_j) = A^T * buf
                //       A^T is n × m, buf is m × b_j, result is n × b_j.
                //       R is n × n with ld = ldr.
                //       Column j starts at R + j * ldr.
                //
                // Total FLOPs are the same as the full path; only memory differs.

                for (int64_t j = 0; j < n; j += b_eff) {
                    int64_t b_j = std::min(b_eff, n - j);

                    // (1) buf = A * R_sk_inv[:, j : j+b_j]
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, R_sk_inv + j * n, n, (T)0.0, A_pre, m);

                    // (2) R[:, j : j+b_j] = A^T * buf
                    A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                      n, b_j, m, (T)1.0, A_pre, m, (T)0.0, R + j * ldr, ldr);
                }

                if(this -> timing) {
                    // In column-block mode, precondition and Gram are fused
                    // into a single loop.  Report the entire loop as
                    // linop_precond; linop_gram is zero.
                    linop_precond_t_stop = steady_clock::now();
                    linop_gram_t_start  = linop_precond_t_stop;
                    linop_gram_t_stop   = linop_precond_t_stop;
                }
            }

            if(this -> timing) {
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

                if (b_eff < n) {
                    // Column-block Gram was used: A_pre is only m × b_eff,
                    // too small for Q.  Recompute A_pre = A * R_sk_inv in
                    // full.  This is outside the timing region, so the extra
                    // operator application does not affect benchmark results.
                    delete[] A_pre;
                    A_pre = new T[m * n]();
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, n, n, (T)1.0, R_sk_inv, n, (T)0.0, A_pre, m);
                }

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
            delete[] Eye;

            // Only delete A_pre if not in test mode (otherwise Q owns it).
            // When test_mode + blocking: the small block buffer was freed
            // and replaced with a full m × n buffer in the Q section above.
            if(!this->test_mode) {
                delete[] A_pre;
            }

            return 0;
        }
};
} // end namespace RandLAPACK
