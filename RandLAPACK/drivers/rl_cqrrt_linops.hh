#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_bqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>

using namespace std::chrono;

namespace RandLAPACK {

/// Method used to form R_sk^{-1} in CQRRT_linops.
///
/// TRSM_IDENTITY  — original method: TRSM(I, R_sk) → R_sk^{-1}.
///                  Fast (O(n³/2)) but unstable when κ(R_sk) is large.
/// GEQP3          — stabilized method: GEQP3(R_sk) = Q*R_buf*P^T,
///                  then R_sk^{-1} = P * R_buf^{-1} * Q^T via ungqr + TRSM.
///                  More expensive (O(11n³/6)) but accurate for ill-conditioned R_sk.
/// BQRRP          — stabilized method using blocked randomized QRCP instead of GEQP3.
///                  Same output format as GEQP3; may be faster for large n due to
///                  cache-friendly blocked structure and sketched pivot selection.
///                  Block size is chosen adaptively via bqrrp_block_ratio.
enum class CQRRTLinopPrecond {
    TRSM_IDENTITY,
    GEQP3,
    BQRRP
};

/// Sketch-preconditioned Cholesky QR for abstract linear operators.
///
/// Linop analogue of CQRRT (rl_cqrrt.hh). Computes A = QR where A is any
/// type satisfying the LinearOperator concept. The algorithm sketches A to
/// obtain a preconditioner R_sk, then computes the Gram matrix of the
/// preconditioned operator A * R_sk^{-1} via linop calls, and factors it
/// with Cholesky.
///
/// Unlike rl_cqrrt.hh (which overwrites A in place with TRSM), this class
/// cannot modify the operator directly. Instead, it computes R_sk^{-1}
/// explicitly and multiplies through the operator interface.
///
/// The Q factor is not computed by default (Q-less factorization). When
/// test_mode is enabled, Q is materialized for verification.
///
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

        // 11 entries: alloc, sketch, qr, tri_inv, fwd, adj, trmm, chol, finalize, rest, total
        //   fwd      = LinOp NoTrans: A * R_sk_inv (accumulated over blocks)
        //   adj      = LinOp Trans:   A^T * buf    (accumulated over blocks)
        //   trmm     = R_sk_inv^T * G (dense trmm, completes Gram)
        std::vector<long> times;

        // tuning SASOS
        int64_t nnz;

        // If true, use a dense Gaussian sketching operator instead of sparse SASO.
        // Dense sketches avoid potential issues with rank-deficient sketches but
        // require O(d*m) storage and O(d*m*n) work for application.
        bool use_dense_sketch;

        // Method for computing R_sk^{-1}. See CQRRTLinopPrecond enum.
        // Default: TRSM_IDENTITY (original behavior).
        // Set to GEQP3 or BQRRP for the stabilized variants.
        CQRRTLinopPrecond precond_method;

        // Block size ratio for BQRRP preconditioner (precond_method == BQRRP only).
        // The BQRRP block size is set to max(1, n * bqrrp_block_ratio).
        // When precond_method == BQRRP and bqrrp_block_ratio == 1.0 (default),
        // the ratio is overridden adaptively inside call() using the same
        // heuristic as CQRRPT: 1.0 for n ≤ 2000, 0.5 for n ≤ 8000, 1/32 otherwise.
        T bqrrp_block_ratio;

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
            precond_method = CQRRTLinopPrecond::TRSM_IDENTITY;
            bqrrp_block_ratio = (T)1.0;
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

        /// Computes the R-factor of the unpivoted QR factorization A = QR,
        /// where Q is m-by-n and R is n-by-n upper triangular.
        ///
        /// Operates similarly to rl_cqrrt.hh, but accepts any type satisfying
        /// the LinearOperator concept and returns a Q-less factorization
        /// (Q is only computed when test_mode is enabled).
        ///
        /// Algorithm:
        ///   1. Sketch: S*A (d x n), where d = d_factor * n
        ///   2. QR of sketch: S*A = Q_sk * R_sk
        ///   3. Precondition: A_pre = A * R_sk^{-1}
        ///   4. Gram matrix: G = (R_sk^{-1})^T * A^T * A * R_sk^{-1}
        ///   5. Cholesky: G = R_chol^T * R_chol
        ///   6. Final R = R_chol * R_sk
        ///
        /// @note This algorithm expects A to be full-rank (rank = n). Rank-deficient inputs may result
        ///       in loss of orthogonality in the Q-factor (when test_mode=true) and numerical instability
        ///       in the R-factor.
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
        /// @param[in] d_factor
        ///     Sketch embedding factor. The sketch dimension is d = d_factor * n.
        ///     Typically d_factor >= 1; larger values improve numerical stability.
        ///
        /// @param[in,out] state
        ///     RNG state for sketching operator generation. Advanced on exit.
        ///
        /// @return = 0: successful exit
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) {
            ///--------------------TIMING VARS--------------------/
            steady_clock::time_point t_start, t_stop;
            steady_clock::time_point alloc_t_start, alloc_t_stop;
            steady_clock::time_point saso_t_start, saso_t_stop;
            steady_clock::time_point qr_t_start, qr_t_stop;
            steady_clock::time_point trtri_t_start, trtri_t_stop;
            steady_clock::time_point trmm_gram_t_start, trmm_gram_t_stop;
            steady_clock::time_point potrf_t_start, potrf_t_stop;
            steady_clock::time_point finalize_t_start, finalize_t_stop;
            steady_clock::time_point total_t_start, total_t_stop;
            steady_clock::time_point q_t_start, q_t_stop;
            long alloc_t_dur        = 0;
            long saso_t_dur         = 0;
            long qr_t_dur           = 0;
            long trtri_t_dur        = 0;
            long fwd_t_dur          = 0;
            long adj_t_dur          = 0;
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

            if(this -> timing)
                alloc_t_start = steady_clock::now();

            // No zero-init: first use is with beta=0.0 which overwrites all elements
            T* A_hat = new T[d * n];
            // No zero-init: geqrf writes the output
            T* tau   = new T[n];

            if(this -> timing) {
                alloc_t_stop = steady_clock::now();
                saso_t_start = steady_clock::now();
            }

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

            // Compute R_sk^{-1}. Method selected by this->precond_method.
            if(this -> timing)
                trtri_t_start = steady_clock::now();

            T* R_sk_inv = nullptr;

            if (this->precond_method == CQRRTLinopPrecond::TRSM_IDENTITY) {
                // Original: TRSM(I, R_sk) → R_sk^{-1}   (upper triangular result)
                T* Eye = new T[n * n]();
                RandLAPACK::util::eye(n, n, Eye);
                if (!RandLAPACK::util::diag_is_nonzero(n, A_hat, d)) {
                    delete[] A_hat;
                    delete[] tau;
                    delete[] Eye;
                    return 1;
                }
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, (T)1.0, A_hat, d, Eye, n);
                if (n > 1) {
                    lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &Eye[1], n);
                }
                R_sk_inv = Eye;
            } else {
                // Stabilized QRCP-based inversion (GEQP3 or BQRRP).
                // Both methods produce the same output format:
                //   R_sk_copy: Householder reflectors for Q_buf in lower triangle,
                //              R_buf in upper triangle (1-based jpiv, same as LAPACK GEQP3)
                // The ungqr + TRSM + scatter code below is identical for both.
                //
                // Result: R_sk^{-1} = P * R_buf^{-1} * Q_buf^T  (dense, n×n)

                // Extract R_sk from A_hat upper triangle into n×n buffer
                T* R_sk_copy = new T[n * n]();
                for (int64_t j = 0; j < n; ++j)
                    for (int64_t i = 0; i <= j; ++i)
                        R_sk_copy[i + j*n] = A_hat[i + j*d];

                if (!RandLAPACK::util::diag_is_nonzero(n, R_sk_copy, n)) {
                    delete[] A_hat;
                    delete[] tau;
                    delete[] R_sk_copy;
                    return 1;
                }

                int64_t* jpiv   = new int64_t[n]();
                T*       tau_qr = new T[n];

                if (this->precond_method == CQRRTLinopPrecond::GEQP3) {
                    lapack::geqp3(n, n, R_sk_copy, n, jpiv, tau_qr);
                } else {
                    // BQRRP: blocked randomized QRCP on R_sk (n×n).
                    // Adaptive block size matches CQRRPT's heuristic.
                    if (n <= 2000)      this->bqrrp_block_ratio = (T)1.0;
                    else if (n <= 8000) this->bqrrp_block_ratio = (T)0.5;
                    else                this->bqrrp_block_ratio = (T)1.0 / (T)32;
                    RandLAPACK::BQRRP<T, RNG> bqrrp(false, (int64_t)(n * this->bqrrp_block_ratio));
                    bqrrp.call(n, n, R_sk_copy, n, (T)1.0, tau_qr, jpiv, state);
                }

                // Extract upper triangular R_buf before overwriting R_sk_copy with Q_buf
                T* R_buf = new T[n * n]();
                for (int64_t j = 0; j < n; ++j)
                    for (int64_t i = 0; i <= j; ++i)
                        R_buf[i + j*n] = R_sk_copy[i + j*n];

                // Expand Q_buf from Householder reflectors (overwrites R_sk_copy)
                lapack::ungqr(n, n, n, R_sk_copy, n, tau_qr);

                // W = Q_buf^T (explicit transpose), then solve R_buf * W = Q_buf^T in-place
                T* W = new T[n * n];
                for (int64_t i = 0; i < n; ++i)
                    for (int64_t j = 0; j < n; ++j)
                        W[i + j*n] = R_sk_copy[j + i*n];
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                           Diag::NonUnit, n, n, (T)1.0, R_buf, n, W, n);

                // R_sk_inv = P * W: row (jpiv[k]-1) of R_sk_inv gets row k of W
                R_sk_inv = new T[n * n]();
                for (int64_t k = 0; k < n; ++k)
                    for (int64_t j = 0; j < n; ++j)
                        R_sk_inv[(jpiv[k]-1) + j*n] = W[k + j*n];

                delete[] R_sk_copy;
                delete[] R_buf;
                delete[] W;
                delete[] jpiv;
                delete[] tau_qr;
            }

            if(this -> timing) {
                trtri_t_stop = steady_clock::now();
            }

            // Gram computation: R = (R_sk_inv)^T * A^T * A * R_sk_inv
            // The (R_sk_inv)^T left-multiply is handled later via TRMM.
            // Here we compute: R = A^T * (A * R_sk_inv).

            // Determine effective block width.
            // block_size <= 0 or >= n means "no blocking" (full width).
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // A_pre buffer:
            //   Full path:  m × n  (kept alive for Q-factor in test_mode)
            //   Block path: m × b_eff  (temporary, freed after Gram loop;
            //                           if test_mode, a full m × n buffer is
            //                           allocated later for Q computation)
            // No zero-init: first use is with beta=0.0 which overwrites all elements
            T* A_pre = new T[m * b_eff];

            if (b_eff == n) {
                // --- Full materialization path (original) ---

                // Step 1: A_pre (m × n) = A * R_sk_inv (m × n × n)
                //   A is m × n, R_sk_inv is n × n, A_pre is m × n.
                //   Side::Left: C = alpha * op(A) * op(B) + beta * C
                //   where op(A) = A (m × n), op(B) = R_sk_inv (n × n), C = A_pre (m × n).
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, (T)1.0, R_sk_inv, n, (T)0.0, A_pre, m);
                if(this->timing) { t_stop = steady_clock::now(); fwd_t_dur = duration_cast<microseconds>(t_stop - t_start).count(); }

                // Step 2: R (n × n) = A^T * A_pre (n × m × m × n = n × n)
                //   Since SYRK is not defined for non-dense operators, we use
                //   an explicit A^T * A_pre to form the Gram matrix.
                //   op(A) = A^T (n × m), op(B) = A_pre (m × n), C = R (n × n).
                if(this->timing) t_start = steady_clock::now();
                A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, (T)1.0, A_pre, m, (T)0.0, R, ldr);
                if(this->timing) { t_stop = steady_clock::now(); adj_t_dur = duration_cast<microseconds>(t_stop - t_start).count(); }
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

                long fwd_accum = 0, adj_accum = 0;
                for (int64_t j = 0; j < n; j += b_eff) {
                    int64_t b_j = std::min(b_eff, n - j);

                    // (1) buf = A * R_sk_inv[:, j : j+b_j]  (NoTrans = fwd)
                    if(this->timing) t_start = steady_clock::now();
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      m, b_j, n, (T)1.0, R_sk_inv + j * n, n, (T)0.0, A_pre, m);
                    if(this->timing) { t_stop = steady_clock::now(); fwd_accum += duration_cast<microseconds>(t_stop - t_start).count(); }

                    // (2) R[:, j : j+b_j] = A^T * buf  (Trans = adj)
                    if(this->timing) t_start = steady_clock::now();
                    A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                      n, b_j, m, (T)1.0, A_pre, m, (T)0.0, R + j * ldr, ldr);
                    if(this->timing) { t_stop = steady_clock::now(); adj_accum += duration_cast<microseconds>(t_stop - t_start).count(); }
                }

                if(this -> timing) {
                    fwd_t_dur = fwd_accum;
                    adj_t_dur = adj_accum;
                }
            }

            if(this -> timing) {
                trmm_gram_t_start = steady_clock::now();
            }

            // Complete Gram: R := R_sk_inv^T * R  (= R_sk_inv^T * A^T * A * R_sk_inv = A_pre^T * A_pre)
            // TRSM_IDENTITY: R_sk_inv is upper triangular — use TRMM (O(n³/2))
            // GEQP3:         R_sk_inv is dense            — use GEMM (O(n³))
            if (this->precond_method == CQRRTLinopPrecond::TRSM_IDENTITY) {
                blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit, n, n, (T)1.0, R_sk_inv, n, R, ldr);
            } else {
                T* tmp = new T[n * n];
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n,
                           (T)1.0, R_sk_inv, n, R, ldr, (T)0.0, tmp, n);
                for (int64_t j = 0; j < n; ++j)
                    for (int64_t i = 0; i < n; ++i)
                        R[i + j*ldr] = tmp[i + j*n];
                delete[] tmp;
            }

            if(this -> timing) {
                trmm_gram_t_stop = steady_clock::now();
                potrf_t_start = steady_clock::now();
            }

            // Cholesky factorization (only reads/writes upper triangle)
            if (lapack::potrf(Uplo::Upper, n, R, ldr)) {
                delete[] A_hat;
                delete[] tau;
                delete[] R_sk_inv;
                delete[] A_pre;
                return 1;
            }

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
                    // No zero-init: beta=0.0 overwrites all elements
                    A_pre = new T[m * n];
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

                alloc_t_dur        = duration_cast<microseconds>(alloc_t_stop        - alloc_t_start).count();
                saso_t_dur         = duration_cast<microseconds>(saso_t_stop         - saso_t_start).count();
                qr_t_dur           = duration_cast<microseconds>(qr_t_stop           - qr_t_start).count();
                trtri_t_dur        = duration_cast<microseconds>(trtri_t_stop        - trtri_t_start).count();
                // fwd_t_dur and adj_t_dur already set (in both full and blocked paths)
                trmm_gram_t_dur    = duration_cast<microseconds>(trmm_gram_t_stop    - trmm_gram_t_start).count();
                potrf_t_dur        = duration_cast<microseconds>(potrf_t_stop        - potrf_t_start).count();
                finalize_t_dur     = duration_cast<microseconds>(finalize_t_stop     - finalize_t_start).count();

                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                // Subtract Q-factor computation time if in test mode
                if(this->test_mode) {
                    q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
                    total_t_dur -= q_t_dur;
                }

                long t_rest  = total_t_dur - (alloc_t_dur + saso_t_dur + qr_t_dur + trtri_t_dur + fwd_t_dur +
                                              adj_t_dur + trmm_gram_t_dur + potrf_t_dur + finalize_t_dur);

                // Fill the data vector (11 entries)
                // Index: 0=alloc, 1=sketch, 2=qr, 3=tri_inv, 4=fwd, 5=adj, 6=trmm, 7=chol, 8=finalize, 9=rest, 10=total
                this -> times = {alloc_t_dur, saso_t_dur, qr_t_dur, trtri_t_dur, fwd_t_dur,
                                 adj_t_dur, trmm_gram_t_dur, potrf_t_dur, finalize_t_dur,
                                 t_rest, total_t_dur};
            }

            // Cleanup - now outside the timing region to avoid timing artifacts
            delete[] A_hat;
            delete[] tau;
            delete[] R_sk_inv;

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
