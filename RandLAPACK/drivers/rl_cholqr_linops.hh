#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"

#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace std::chrono;

namespace RandLAPACK {

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

        // 6 entries: alloc, materialize, gram, potrf, rest, total
        std::vector<long> times;

        // Column-block size for the materialize + Gram computation.
        //
        // When block_size > 0, the two expensive operations:
        //   (1) A_temp = A * I       (m × n) - materialize the operator
        //   (2) R      = A^T * A_temp (n × n) - Gram matrix
        // are fused into a column-block loop that processes b columns at a time:
        //   for each column block j of width b:
        //     buf (m × b) = A * I[:, j*b : (j+1)*b]
        //     R[:, j*b : (j+1)*b] = A^T * buf
        //
        // This reduces peak memory from O(m*n) to O(m*b), which is significant
        // when m is large and n is moderate.  The result is mathematically
        // identical — each column block of R is:
        //   R[:, j_block] = A^T * (A * I[:, j_block])
        // which equals the corresponding columns of A^T * A.
        //
        // When block_size <= 0 or block_size >= n, the full m × n buffer is
        // allocated and the original (non-blocked) path is used.
        //
        // When test_mode is enabled and blocking is active, the Q-factor
        // computation (which needs the full m × n A_temp) is handled by
        // recomputing A_temp = A * I after the Gram loop.  This
        // recomputation is outside the timing region, so it does not
        // affect benchmark results.
        int64_t block_size;

        CholQR_linops(
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
            steady_clock::time_point alloc_t_start;
            steady_clock::time_point alloc_t_stop;
            steady_clock::time_point materialize_t_start;
            steady_clock::time_point materialize_t_stop;
            steady_clock::time_point gram_t_start;
            steady_clock::time_point gram_t_stop;
            steady_clock::time_point potrf_t_start;
            steady_clock::time_point potrf_t_stop;
            steady_clock::time_point total_t_start;
            steady_clock::time_point total_t_stop;
            long alloc_t_dur = 0;
            long materialize_t_dur = 0;
            long gram_t_dur  = 0;
            long potrf_t_dur = 0;
            long total_t_dur = 0;
            long q_t_dur     = 0;
            steady_clock::time_point q_t_start;
            steady_clock::time_point q_t_stop;

            if(this->timing)
                total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            // Compute Gram matrix: R = A^T * A
            // We cannot use syrk since A may be a sparse operator
            // Instead, compute R = A^T * A via two matvec operations:
            // 1. Create identity matrix I (n x n)
            // 2. Compute A_temp = A * I (materializes A as m x n dense)
            // 3. Compute R = A^T * A_temp

            if(this->timing)
                alloc_t_start = steady_clock::now();

            // Create identity matrix (needs zero-init for off-diagonal elements)
            T* I_mat = new T[n * n]();
            RandLAPACK::util::eye(n, n, I_mat);

            // ================================================================
            // Materialize + Gram computation: R = A^T * A
            // ================================================================
            //
            // Two paths are available:
            //
            // (a) FULL MATERIALIZATION (original):
            //     Allocate m × n buffer A_temp, compute A_temp = A * I,
            //     then R = A^T * A_temp.  Memory: O(m*n).
            //
            // (b) COLUMN-BLOCK PROCESSING (memory-efficient):
            //     Process b columns at a time.  For each column block j:
            //       buf (m × b_j) = A * I[:, j : j+b_j]
            //       R[:, j : j+b_j] = A^T * buf
            //     Memory: O(m*b).  Never forms the full A_temp.
            //     The result is mathematically identical because:
            //       R[:, j_block] = A^T * (A * I[:, j_block])
            //     is the j-th column block of A^T * A.
            //
            //     When test_mode is on, the Q-factor (Q = A_temp * R^{-1})
            //     needs the full m × n A_temp.  In that case, A_temp is
            //     recomputed after the Gram loop, outside the timing region.
            //     This costs an extra operator application but does NOT affect
            //     the benchmark timings.
            //
            // ================================================================

            // Determine effective block width.
            // block_size <= 0 or >= n means "no blocking" (full width).
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // A_temp buffer:
            //   Full path:  m × n  (kept alive for Q-factor in test_mode)
            //   Block path: m × b_eff  (temporary, freed after Gram loop;
            //                           if test_mode, a full m × n buffer is
            //                           allocated later for Q computation)
            // No zero-init needed: first use is with beta=0.0 which overwrites all elements.
            T* A_temp = new T[m * b_eff];

            if(this->timing) {
                alloc_t_stop = steady_clock::now();
                materialize_t_start = steady_clock::now();
            }

            if (b_eff == n) {
                // --- Full materialization path (original) ---

                // Step 1: Materialize A by computing A_temp = A * I
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, I_mat, n, (T)0.0, A_temp, m);

                if(this->timing) {
                    materialize_t_stop = steady_clock::now();
                    gram_t_start = steady_clock::now();
                }

                // Step 2: Compute R = A^T * A_temp (using the linear operator's transpose)
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, A_temp, m, (T)0.0, R, ldr);

                if(this->timing) {
                    gram_t_stop = steady_clock::now();
                }
            } else {
                // --- Column-block processing path (memory-efficient) ---
                //
                // Process n columns in blocks of width b_eff.
                // The last block may be narrower if n is not divisible by b_eff.
                //
                // For each block starting at column j with width b_j:
                //   (1) buf (m × b_j) = A * I[:, j : j+b_j]
                //       I is n × n in ColMajor with ld = n.
                //       Column j starts at I + j * n.
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

                    // (1) buf = A * I[:, j : j+b_j]
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, I_mat + j * n, n, (T)0.0, A_temp, m);

                    // (2) R[:, j : j+b_j] = A^T * buf
                    A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                      n, b_j, m, (T)1.0, A_temp, m, (T)0.0, R + j * ldr, ldr);
                }

                if(this->timing) {
                    // In column-block mode, materialize and Gram are fused
                    // into a single loop.  Report the entire loop as
                    // materialize; gram is zero.
                    materialize_t_stop = steady_clock::now();
                    gram_t_start  = materialize_t_stop;
                    gram_t_stop   = materialize_t_stop;
                }
            }

            // Zero out the lower triangle before Cholesky (potrf only uses upper triangle)
            if (n > 1) {
                lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &R[1], ldr);
            }

            if(this->timing) {
                potrf_t_start = steady_clock::now();
            }

            // Compute Cholesky factorization: G = R^T * R
            // On exit, the upper triangle of R contains the R-factor
            lapack::potrf(Uplo::Upper, n, R, ldr);

            if(this->timing)
                potrf_t_stop = steady_clock::now();

            // Compute Q-factor if test mode is enabled (NOT included in cholqr timing)
            if(this->test_mode) {
                if(this->timing)
                    q_t_start = steady_clock::now();

                if (b_eff < n) {
                    // Column-block Gram was used: A_temp is only m × b_eff,
                    // too small for Q.  Recompute A_temp = A * I in
                    // full.  This is outside the timing region, so the extra
                    // operator application does not affect benchmark results.
                    delete[] A_temp;
                    // No zero-init: beta=0.0 overwrites all elements
                    A_temp = new T[m * n];
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, n, n, (T)1.0, I_mat, n, (T)0.0, A_temp, m);
                }

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

                alloc_t_dur = duration_cast<microseconds>(alloc_t_stop - alloc_t_start).count();
                materialize_t_dur = duration_cast<microseconds>(materialize_t_stop - materialize_t_start).count();
                gram_t_dur  = duration_cast<microseconds>(gram_t_stop  - gram_t_start).count();
                potrf_t_dur = duration_cast<microseconds>(potrf_t_stop - potrf_t_start).count();
                total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q-factor computation time if in test mode
                if(this->test_mode) {
                    q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
                    total_t_dur -= q_t_dur;
                }

                long rest_t_dur = total_t_dur - (alloc_t_dur + materialize_t_dur + gram_t_dur + potrf_t_dur);

                // Fill the data vector: [alloc, materialize, gram, potrf, rest, total]
                this->times = {alloc_t_dur, materialize_t_dur, gram_t_dur, potrf_t_dur, rest_t_dur, total_t_dur};
            }

            // Cleanup - now outside the timing region to avoid timing artifacts
            delete[] I_mat;

            // Only delete A_temp if not in test mode (otherwise Q owns it).
            // When test_mode + blocking: the small block buffer was freed
            // and replaced with a full m × n buffer in the Q section above.
            if(!this->test_mode) {
                delete[] A_temp;
            }

            return 0;
        }
};

} // end namespace RandLAPACK
