#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"
#include "rl_cqrrt.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>
#include <climits>
#include <iomanip>

using namespace std::chrono;

namespace RandLAPACK {

/// BK (Block Krylov) is the computational routine underlying the ABRIK driver.
/// It builds left and right Krylov subspaces (X_ev, Y_od) and band matrices (R, S)
/// via block Krylov iterations with double reorthogonalization.
///
/// The ABRIK driver calls BK to obtain these factored intermediates, then performs
/// SVD on R or S and reconstructs the final U, Sigma, V.
///
/// This follows the same pattern as QB (comps) + RSVD (driver).

// Struct outside of BK class to make symbols shorter
struct BKSubroutines {
    enum QR_explicit {geqrf_ungqr, cqrrt};
};

/// Reason BK terminated its main loop.
enum class BKTermination {
    max_iters_reached, ///< Reached max_krylov_iters without convergence (resumable).
    norm_converged,    ///< norm_R exceeded threshold (A's spectral content exhausted).
    rank_deficient,    ///< A dead block with refilling off or exhausted (or a failed CQRRT).
    saturated          ///< The basis has as many columns as the ambient dimension allows,
                       ///< including a dead block with refilling on but no room to refill.
};

template <typename T, typename RNG>
class BK {
    public:
        using Subroutines = BKSubroutines;
        Subroutines::QR_explicit qr_exp;

        bool verbose;
        bool timing;
        T tol;
        int num_krylov_iters;
        int max_krylov_iters;
        std::vector<long> times;
        T norm_R_end;
        BKTermination termination_reason;
        /// Relative threshold for the numerical-rank criterion: the trailing block is
        /// judged against tau*||A||_F. This criterion does not provide the conditioning
        /// guarantee of Balabanov Thm 5.6; see util::block_numerical_rank in rl_util.hh.
        /// Mirrors CQRRPT's user-facing `eps` (rl_cqrrpt.hh:120). Zero means "derive a
        /// default from the problem size", which is done inside call_impl where n is known.
        T tau;
        /// Width of the terminal block after truncation; equals k unless the rank criterion
        /// rejected part of it. Diagnostic only: end_rows/end_cols are read straight off the
        /// accepted-column counters and no longer need to be reconstructed from it.
        int64_t final_block_width;
        /// Number of blocks the rank criterion narrowed over the run. Lets a test assert
        /// that the mechanism FIRED, not merely that the answer came out right.
        int64_t narrowed_blocks;
        /// Resume state. Everything the loop needs to pick up where it left off, saved on
        /// exit and restored by resume(). Previously the resume path reconstructed all of
        /// this from `iter` by fixed-width arithmetic, which is silently wrong the moment a
        /// block is not exactly k wide. Saving it makes resume a restore rather than a
        /// reconstruction, which removes that hazard instead of fencing it off.
        int64_t saved_x_cols;
        int64_t saved_y_cols;
        int64_t saved_w_last;
        int64_t saved_alloc_X_cols;
        int64_t saved_alloc_Y_cols;
        /// Refill the rejected columns of a narrowed or dead block with Gaussian directions
        /// orthogonalized against the accepted basis, so the run continues where the Krylov
        /// space closed early (the identity, a singular value repeated more than k times,
        /// I + P). The next block on the other side probes them; if their images come back
        /// dead they are retracted and refilling is switched off for the rest of the run.
        /// Default true; false restores prune-and-narrow, where a dead block ends the run.
        bool refill_dead_columns;
        /// True once a probe has fired or a CQRRT factorization failed: no further refills
        /// in this run. Output, and resume state that resume() leaves as the last call set it.
        bool refills_exhausted;
        /// Number of blocks that received refills over the run. Diagnostic only.
        int64_t refilled_blocks;
        /// Resume state: refills in the newest block that no iteration has probed yet. They
        /// are kept in the basis and in saved_x_cols/saved_y_cols, but not in end_rows/end_cols.
        int64_t saved_pending_refills;

        BK(
            bool verb,
            bool time_subroutines,
            T ep
        ) {
            qr_exp = Subroutines::QR_explicit::geqrf_ungqr;
            verbose = verb;
            timing = time_subroutines;
            tol = ep;
            max_krylov_iters = INT_MAX;
            // These three are outputs, but they were left uninitialized, so reading a
            // termination reason before the first call() was undefined behaviour -- and
            // reading it is exactly what a test asserting on termination does.
            num_krylov_iters = 0;
            norm_R_end = 0;
            termination_reason = BKTermination::max_iters_reached;
            tau = 0;                 // 0 => derive n*eps inside call_impl
            final_block_width = 0;
            narrowed_blocks = 0;
            saved_x_cols = 0;
            saved_y_cols = 0;
            saved_w_last = 0;
            saved_alloc_X_cols = 0;
            saved_alloc_Y_cols = 0;
            refill_dead_columns = true;
            refills_exhausted = false;
            refilled_blocks = 0;
            saved_pending_refills = 0;
        }

        /// Builds the block Krylov subspaces and band matrices for a truncated SVD.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     Pointer to the m-by-n matrix A, stored in a column-major format.
        ///
        /// @param[in] lda
        ///     Leading dimension of A.
        ///
        /// @param[in] k
        ///     Block size for Krylov iterations.
        ///
        /// @param[out] X_ev
        ///     Left Krylov basis (m x end_rows), allocated internally with calloc.
        ///     Caller must free(). After an even final iteration the columns just past
        ///     end_rows may hold refills that no iteration has probed yet; they are kept
        ///     for resume() and are not part of the result.
        ///
        /// @param[out] Y_od
        ///     Right Krylov basis (n x end_cols), allocated internally with calloc.
        ///     Caller must free(). After an odd final iteration the columns just past
        ///     end_cols may hold unprobed refills, kept for resume() in the same way.
        ///
        /// @param[out] R
        ///     Band matrix for an odd final iteration: end_rows by end_cols with leading
        ///     dimension n, allocated internally with calloc. Caller must free().
        ///     Stored in the orientation that is consumed directly: the buffer AS STORED
        ///     equals X_ev(:,1:end_rows)' * A * Y_od(:,1:end_cols). That is what
        ///     TestBK.BK_band_equals_XtAY_* measures (7e-16 on a Gaussian input), and it is
        ///     the orientation ABRIK hands to lapack::gesdd without transposing.
        ///     The former phrase "stored transposed" described the per-block transposition
        ///     performed by util::transposition(..., copy_upper_triangle=1), which leaves
        ///     each diagonal block lower triangular. It never meant the band as a whole, and
        ///     it read as the opposite of the truth.
        ///     Where a rejected column was zeroed (and possibly refilled) the band holds a
        ///     zero whose true value is bounded by tau*||A||_F.
        ///
        /// @param[out] S
        ///     Band matrix for an even final iteration: end_rows by end_cols with leading
        ///     dimension n + k, allocated internally with calloc. Caller must free().
        ///     The extra k rows exist because the diagonal block sits one block below the
        ///     diagonal (lower Hessenberg). Same orientation convention as R: the buffer as
        ///     stored is the band. Rejected rows are zeroed as R's rejected columns are.
        ///
        /// @param[out] end_rows
        ///     Number of rows in the band matrix for SVD: the accepted left columns, less
        ///     any refills in the newest left block that no iteration has probed yet.
        ///
        /// @param[out] end_cols
        ///     Number of columns in the band matrix for SVD: the accepted right columns,
        ///     less any refills in the newest right block that no iteration has probed yet.
        ///
        /// @param[out] final_iter_is_odd
        ///     True if the last iteration was odd (use R for SVD), false if even (use S).
        ///
        /// @param[in] state
        ///     RNG state parameter, required for sketching operator generation.
        ///
        /// @return = 0: successful exit, -1: realloc failure

        // BK call that accepts a general dense matrix.
        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            int64_t k,
            T* &X_ev,
            T* &Y_od,
            T* &R,
            T* &S,
            int64_t &end_rows,
            int64_t &end_cols,
            bool &final_iter_is_odd,
            RandBLAS::RNGState<RNG> &state
        ) {
            linops::DenseLinOp<T> A_linop(m, n, A, lda, Layout::ColMajor);
            return this->call(A_linop, k, X_ev, Y_od, R, S, end_rows, end_cols, final_iter_is_odd, state);
        }

        // BK call that accepts sparse matrix.
        template <RandBLAS::sparse_data::SparseMatrix SpMat>
        int call(
            int64_t m,
            int64_t n,
            SpMat &A,
            int64_t lda,
            int64_t k,
            T* &X_ev,
            T* &Y_od,
            T* &R,
            T* &S,
            int64_t &end_rows,
            int64_t &end_cols,
            bool &final_iter_is_odd,
            RandBLAS::RNGState<RNG> &state
        ) {
            linops::SparseLinOp<SpMat> A_linop(m, n, A);
            return this->call(A_linop, k, X_ev, Y_od, R, S, end_rows, end_cols, final_iter_is_odd, state);
        }

        /// Resume a previous BK computation with more iterations.
        /// X_ev, Y_od, R, S must be non-null from a prior call().
        /// Increase max_krylov_iters before calling.
        template <RandLAPACK::linops::LinearOperator GLO>
        int resume(
            GLO& A,
            int64_t k,
            T* &X_ev,
            T* &Y_od,
            T* &R,
            T* &S,
            int64_t &end_rows,
            int64_t &end_cols,
            bool &final_iter_is_odd,
            RandBLAS::RNGState<RNG> &state
        ) {
            return this->call_impl(A, k, X_ev, Y_od, R, S, end_rows, end_cols, final_iter_is_odd, state, true);
        }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            int64_t k,
            T* &X_ev,
            T* &Y_od,
            T* &R,
            T* &S,
            int64_t &end_rows,
            int64_t &end_cols,
            bool &final_iter_is_odd,
            RandBLAS::RNGState<RNG> &state
        ) {
            return this->call_impl(A, k, X_ev, Y_od, R, S, end_rows, end_cols, final_iter_is_odd, state, false);
        }

    private:
        template <RandLAPACK::linops::LinearOperator GLO>
        int call_impl(
            GLO& A,
            int64_t k,
            T* &X_ev,
            T* &Y_od,
            T* &R,
            T* &S,
            int64_t &end_rows,
            int64_t &end_cols,
            bool &final_iter_is_odd,
            RandBLAS::RNGState<RNG> &state,
            bool resuming
        ){
                steady_clock::time_point allocation_t_start;
                steady_clock::time_point allocation_t_stop;
                steady_clock::time_point ungqr_t_start;
                steady_clock::time_point ungqr_t_stop;
                steady_clock::time_point reorth_t_start;
                steady_clock::time_point reorth_t_stop;
                steady_clock::time_point qr_t_start;
                steady_clock::time_point qr_t_stop;
                steady_clock::time_point gemm_A_t_start;
                steady_clock::time_point gemm_A_t_stop;
                steady_clock::time_point main_loop_t_start;
                steady_clock::time_point main_loop_t_stop;
                steady_clock::time_point sketching_t_start;
                steady_clock::time_point sketching_t_stop;
                steady_clock::time_point r_cpy_t_start;
                steady_clock::time_point r_cpy_t_stop;
                steady_clock::time_point s_cpy_t_start;
                steady_clock::time_point s_cpy_t_stop;
                steady_clock::time_point norm_t_start;
                steady_clock::time_point norm_t_stop;
                steady_clock::time_point bk_total_t_start;
                steady_clock::time_point bk_total_t_stop;

                long allocation_t_dur  = 0;
                long ungqr_t_dur       = 0;
                long reorth_t_dur      = 0;
                long qr_t_dur          = 0;
                long gemm_A_t_dur      = 0;
                long main_loop_t_dur   = 0;
                long sketching_t_dur   = 0;
                long r_cpy_t_dur       = 0;
                long s_cpy_t_dur       = 0;
                long norm_t_dur        = 0;
                long bk_total_t_dur    = 0;

                if(this -> timing)
                    bk_total_t_start = steady_clock::now();

                int64_t m = A.n_rows;
                int64_t n = A.n_cols;
                int max_iters = this->max_krylov_iters;

                // Preconditions. k > min(m, n) is not merely unsupported, it is an
                // out-of-bounds READ: the band buffers are sized n*k and (n+k)*k, while the
                // rank-deficiency probes index R_ii[(n + 1) * (k - 1)] and
                // S_ii[((n + k) + 1) * (k - 1)]. With n = 5 and k = 10 the first of those
                // is R[54] against a 50-element allocation. ABRIK only ever checked k > 0
                // (rl_abrik.hh:188, :209), so nothing upstream caught it either. ungqr(n, k,
                // k, ...) at the explicit-QR step would also be an invalid LAPACK call.
                //
                // randlapack_require is not NDEBUG-gated (rl_exceptions.hh:97-98), so these
                // hold in Release and are testable with EXPECT_THROW.
                randlapack_require(m > 0) << "BK: m=" << m << " must be > 0";
                randlapack_require(n > 0) << "BK: n=" << n << " must be > 0";
                randlapack_require(k > 0) << "BK: block size k=" << k << " must be > 0";
                randlapack_require(k <= std::min(m, n))
                    << "BK: block size k=" << k << " exceeds min(m, n)=" << std::min(m, n)
                    << "; the band buffers and the rank-deficiency probes assume k <= min(m, n)";

                // Loop state: initialized differently for fresh start vs resume.
                int64_t iter;
                // ACCEPTED columns of each basis, refills included. They advance only after
                // the rank test of the block they count, never as a reservation, and fall back
                // only when a probe retracts dead refills (the last columns of a block).
                // The old curr_X_cols/curr_Y_cols were pre-advanced before the block they
                // reserved was written, so they meant "accepted plus pending" and flipped
                // meaning twice per iteration cycle. At exit, end_rows and end_cols are these
                // counters less the unprobed refills of the newest block (see the exit).
                int64_t x_cols, y_cols;
                // Width of the most recently accepted block, equivalently the width the next
                // iteration will build. Non-increasing over the run, in [1, k].
                int64_t w_last;
                // Refills at the tail of the most recently accepted block. The next iteration
                // probes them: their images are the trailing rows (odd) or columns (even) of
                // its diagonal band block. Zero once probed.
                int64_t pending_refills;
                // Allocated column counts, tracked apart from the accepted counts.
                int64_t alloc_X_cols, alloc_Y_cols;
                T norm_R;
                T* Y_i;
                T* X_i;
                T* R_i;
                T* R_ii;
                T* S_i;
                T* S_ii;

                // Pre-allocation: when max_krylov_iters is known, allocate all
                // buffers upfront to avoid per-iteration realloc + memset.
                bool prealloc = (max_iters != INT_MAX);
                int64_t max_X_cols = 0, max_Y_cols = 0;
                if (prealloc) {
                    // After max_iters iterations:
                    //   odd iters (1,3,...) grow X_ev; even iters (2,4,...) grow Y_od
                    //   Initial: k cols each. Each relevant iter adds k cols.
                    int64_t n_odd  = (max_iters + 1) / 2;  // ceil(max_iters/2)
                    int64_t n_even = max_iters / 2;
                    max_X_cols = k * (1 + n_odd);
                    max_Y_cols = k * (1 + n_even);
                }

                if (!resuming) {
                    // Fresh start: allocate output buffers and initialize state
                    if(this -> timing)
                        allocation_t_start = steady_clock::now();

                    iter = 0; x_cols = 0; y_cols = 0; w_last = k;
                    pending_refills = 0; this->refills_exhausted = false;
                    this->refilled_blocks = 0; this->narrowed_blocks = 0;
                    end_rows = 0; end_cols = 0;
                    norm_R = 0;

                    if (prealloc) {
                        // Allocate to maximum size upfront, no realloc needed in loop.
                        Y_od  = ( T * ) calloc( n * max_Y_cols, sizeof( T ) );
                        X_ev  = ( T * ) calloc( m * max_X_cols, sizeof( T ) );
                        R     = ( T * ) calloc( n * max_X_cols, sizeof( T ) );
                        S     = ( T * ) calloc( (n + k) * max_Y_cols, sizeof( T ) );
                    } else {
                        // Tolerance-based: start small, realloc as needed.
                        Y_od  = ( T * ) calloc( n * k, sizeof( T ) );
                        X_ev  = ( T * ) calloc( m * k, sizeof( T ) );
                        R     = ( T * ) calloc( n * k, sizeof( T ) );
                        S     = ( T * ) calloc( (n + k) * k, sizeof( T ) );
                    }
                    alloc_Y_cols = prealloc ? max_Y_cols : k;
                    alloc_X_cols = prealloc ? max_X_cols : k;

                    // The prologue below writes its blocks at offset zero. Every band and
                    // block pointer used inside the loop is derived at the top of its own
                    // branch from x_cols/y_cols, so there is nothing else to initialize.
                    Y_i  = Y_od;
                    X_i  = X_ev;
                    R_i  = NULL;
                    R_ii = NULL;
                    S_i  = NULL;
                    S_ii = NULL;

                    if(this -> timing) {
                        allocation_t_stop  = steady_clock::now();
                        allocation_t_dur   = duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                    }
                } else {
                    // Resume: RESTORE loop state, do not reconstruct it.
                    // Only valid after a prior call() that terminated with max_iters_reached.
                    //
                    // This used to derive iter_od, iter_ev, both column counts and all six
                    // pointers from `iter` by fixed-width arithmetic, which is silently wrong
                    // the moment any earlier block is narrower than k. Saving the four scalars
                    // instead removes that hazard rather than fencing it off, and the pointers
                    // need no reconstruction at all because every branch derives its own.
                    iter         = this->num_krylov_iters;
                    norm_R       = this->norm_R_end;
                    x_cols       = this->saved_x_cols;
                    y_cols       = this->saved_y_cols;
                    w_last       = this->saved_w_last;
                    alloc_X_cols = this->saved_alloc_X_cols;
                    alloc_Y_cols = this->saved_alloc_Y_cols;
                    // Unprobed refills are restored and probed by the next iteration;
                    // refills_exhausted is left as the previous call set it.
                    pending_refills = this->saved_pending_refills;

                    // Grow buffers if the new max_krylov_iters requires more space
                    // than was allocated in the previous call.
                    if (prealloc) {
                        if (max_X_cols > alloc_X_cols) {
                            X_ev = ( T * ) realloc(X_ev, m * max_X_cols * sizeof( T ));
                            R    = ( T * ) realloc(R,    n * max_X_cols * sizeof( T ));
                            if (!X_ev || !R) {
                                free(X_ev); free(Y_od); free(R); free(S);
                                X_ev = nullptr; Y_od = nullptr; R = nullptr; S = nullptr;
                                return -1;
                            }
                            std::fill(&X_ev[m * alloc_X_cols], &X_ev[m * max_X_cols], T(0));
                            std::fill(&R[n * alloc_X_cols],    &R[n * max_X_cols],    T(0));
                            alloc_X_cols = max_X_cols;
                        }
                        if (max_Y_cols > alloc_Y_cols) {
                            Y_od = ( T * ) realloc(Y_od, n * max_Y_cols * sizeof( T ));
                            S    = ( T * ) realloc(S,    (n + k) * max_Y_cols * sizeof( T ));
                            if (!Y_od || !S) {
                                free(X_ev); free(Y_od); free(R); free(S);
                                X_ev = nullptr; Y_od = nullptr; R = nullptr; S = nullptr;
                                return -1;
                            }
                            std::fill(&Y_od[n * alloc_Y_cols],   &Y_od[n * max_Y_cols],   T(0));
                            std::fill(&S[(n + k) * alloc_Y_cols], &S[(n + k) * max_Y_cols], T(0));
                            alloc_Y_cols = max_Y_cols;
                        }
                    }

                    // Advance past the completed iteration so the while-loop starts at the next one.
                    ++iter;
                }

                // Internal temporaries: shared for both paths.
                // These are pure scratch buffers (beta=0.0 GEMM outputs), no need to zero-initialize.
                T* Y_orth_buf = ( T * ) malloc( k * n * sizeof( T ) );
                T* X_orth_buf = ( T * ) malloc( k * (n + k) * sizeof( T ) );
                // tau space for QR (geqrf fully overwrites it)
                T* tau = ( T * ) malloc( k * sizeof( T ) );
                // Declared here (before cleanup lambda) so cleanup can free it.
                // Conditionally allocated below only when CQRRT is used.
                T* R_11_trans = nullptr;

                // Cleanup lambda for realloc failure: frees all buffers and nulls output pointers.
                // free(nullptr) is a no-op, so no guards needed.
                auto cleanup_and_fail = [&]() -> int {
                    free(Y_od);       Y_od = nullptr;
                    free(X_ev);       X_ev = nullptr;
                    free(R);          R    = nullptr;
                    free(S);          S    = nullptr;
                    free(tau);
                    free(Y_orth_buf);
                    free(X_orth_buf);
                    free(R_11_trans);
                    return -1;
                };

                // Refill rejected slots of a block with Gaussian directions orthonormal to the
                // accepted basis. dst: rows x n_ref slot inside the block (ld = rows); basis:
                // rows x c accepted-plus-kept columns, contiguous with dst; G: scratch, ldg >= c.
                // Never touches the operator, so one application per iteration stays true.
                auto refill = [&](int64_t rows, const T* basis, int64_t c, T* dst, int64_t n_ref, T* G, int64_t ldg) {
                    if(this -> timing)
                        sketching_t_start = steady_clock::now();
                    RandBLAS::DenseDist D(rows, n_ref);
                    state = RandBLAS::fill_dense(D, dst, state);
                    if(this -> timing) {
                        sketching_t_stop  = steady_clock::now();
                        sketching_t_dur  += duration_cast<microseconds>(sketching_t_stop - sketching_t_start).count();
                        reorth_t_start    = steady_clock::now();
                    }
                    // Two projection passes, as for the Krylov blocks.
                    for (int pass = 0; pass < 2 && c > 0; ++pass) {
                        blas::gemm(Layout::ColMajor, Op::Trans,   Op::NoTrans, c, n_ref, rows,  1.0, basis, rows, dst, rows, 0.0, G, ldg);
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, rows, n_ref, c, -1.0, basis, rows, G, ldg, 1.0, dst, rows);
                    }
                    if(this -> timing) {
                        reorth_t_stop  = steady_clock::now();
                        reorth_t_dur  += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
                        qr_t_start     = steady_clock::now();
                    }
                    // Always Householder, whatever qr_exp says: unlike CQRRT it cannot fail here.
                    lapack::geqrf(rows, n_ref, dst, rows, tau);
                    if(this -> timing) {
                        qr_t_stop      = steady_clock::now();
                        qr_t_dur      += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                        ungqr_t_start  = steady_clock::now();
                    }
                    lapack::ungqr(rows, n_ref, n_ref, dst, rows, tau);
                    if(this -> timing) {
                        ungqr_t_stop   = steady_clock::now();
                        ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
                    }
                };

                // Termination criteria (both checked inside the main loop below).
                //
                // 1. Frobenius-content convergence: stop once norm_R = ||R||_F exceeds
                //    threshold = sqrt(1 - tol^2) * ||M||_F. R is the coordinate representation
                //    X'MY of M in the accumulated Krylov bases, and hat(M) = X (X'MY) Y' is the
                //    two-sided orthogonal projection of M onto those bases. Since X, Y have
                //    orthonormal columns, ||R||_F = ||hat(M)||_F, and the projection residual is
                //    Frobenius-orthogonal to hat(M), so ||M||_F^2 = ||hat(M)||_F^2 + ||M - hat(M)||_F^2.
                //    Hence norm_R > sqrt(1 - tol^2)||M||_F is equivalent to the relative bound
                //    ||M - hat(M)||_F <= tol * ||M||_F, obtained without any SVD. norm_R is
                //    recomputed only on odd iterations (where R is the current triangular factor).
                //    Exact in exact arithmetic; holds to working precision thanks to the double
                //    reorthogonalization in the qr_add steps.
                // 2. Rank deficiency: stop if the numerical-rank criterion retains no
                //    columns of the new block and no refill can take their place. A positive
                //    reduced width narrows the next block instead. See the per-branch checks.
                // The bounded loop also stops at max_krylov_iters (termination_reason set per case;
                // the ABRIK driver only resumes when that was the reason).
                T norm_A = A.fro_nrm();
                // tau = 0 selects the heuristic default n*eps. This default does not
                // establish a conditioning guarantee; callers can set tau explicitly.
                const T tau_eff = (this->tau > 0) ? this->tau
                                                  : (T)n * std::numeric_limits<T>::epsilon();
                this->final_block_width = k;
                T sq_tol = std::pow(this->tol, 2);
                T threshold =  std::sqrt(1 - sq_tol) * norm_A;

                // Creating the CQRRT object in case it is to be used for explicit QR.
                std::optional<RandLAPACK::CQRRT<T, RNG>> CQRRT;
                T d_factor = 1.25;
                // Conditional initialization
                if(this -> qr_exp == Subroutines::QR_explicit::cqrrt) {
                    CQRRT.emplace(false, tol);
                    CQRRT->nnz = 2;
                    R_11_trans = ( T * ) calloc( k * k, sizeof( T ) );
                }

                if (!resuming) {
                    // Fresh start: sketch generation, first GEMM, first QR
                    if(this -> timing)
                        sketching_t_start  = steady_clock::now();

                    // Generate a dense Gaussian random matrix.
                    RandBLAS::DenseDist D(n, k);
                    state = RandBLAS::fill_dense(D, Y_i, state);

                    if(this -> timing) {
                        sketching_t_stop  = steady_clock::now();
                        sketching_t_dur   = duration_cast<microseconds>(sketching_t_stop - sketching_t_start).count();
                        gemm_A_t_start = steady_clock::now();
                    }

                    // [X_ev, ~] = qr(A * Y_i, 0)
                    A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, Y_i, n, 0.0, X_i, m);

                    if(this -> timing) {
                        gemm_A_t_stop = steady_clock::now();
                        gemm_A_t_dur  = duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                    }

                    if(this -> qr_exp == Subroutines::QR_explicit::cqrrt) {
                        if(this -> timing)
                            qr_t_start = steady_clock::now();

                        std::fill(R_11_trans, R_11_trans + k * k, (T)0.0);
                        (void) CQRRT -> call(m, k, X_i, m, R_11_trans, k, d_factor, state);

                        if(this -> timing) {
                            qr_t_stop = steady_clock::now();
                            qr_t_dur  = duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                        }
                    } else {

                        if(this -> timing)
                            qr_t_start = steady_clock::now();

                        lapack::geqrf(m, k, X_i, m, tau);

                        if(this -> timing) {
                            qr_t_stop = steady_clock::now();
                            qr_t_dur  = duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                            ungqr_t_start  = steady_clock::now();
                        }

                        // Convert X_i into an explicit form. It is now stored in X_ev as it should be.
                        lapack::ungqr(m, k, k, X_i, m, tau);

                        if(this -> timing) {
                            ungqr_t_stop  = steady_clock::now();
                            ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
                        }
                    }

                    // Advance odd iteration count.
                    // The prologue's X block is accepted unconditionally: it is orth(A*Omega)
                    // and is never rank-probed.
                    x_cols = k;
                    w_last = k;
                    // Advance iteration count.
                    ++iter;
                }

                // Main loop: shared for both fresh start and resume.
                while(1) {
                    if(this -> timing)
                        main_loop_t_start = steady_clock::now();

                    if (iter % 2 != 0) {
                        // ODD: build a Y block from the last accepted X block, write the R band.
                        //
                        // Pointers are DERIVED here rather than advanced at the end of the
                        // previous branch. That collapses three copies of the same arithmetic
                        // (fresh init, resume reconstruction, end-of-branch advance) into one.
                        //   x_row = start row of the source X block, also the band's row offset
                        //   R_i   = w by y_cols off-diagonal block, ld n
                        //   R_ii  = w by w diagonal block at (x_row, y_cols), ld n
                        const int64_t w     = w_last;
                        const int64_t x_row = x_cols - w;

                        // Growth FIRST, before the pointers are derived and before the GEMM
                        // below writes into the new block. The old code grew X_ev in this
                        // branch as a pre-advance for the NEXT (even) iteration, which kept
                        // the destination in bounds by staying one block ahead. Growth is now
                        // regrouped by index space (odd owns Y_od and R, even owns X_ev and S),
                        // so it has to happen before the write rather than an iteration early.
                        if (!prealloc && y_cols + w > alloc_Y_cols) {
                            int64_t want = y_cols + w;
                            T* Y_new = ( T * ) realloc(Y_od, n * want * sizeof( T ));
                            T* R_new = ( T * ) realloc(R,    n * want * sizeof( T ));
                            if (Y_new) Y_od = Y_new;
                            if (R_new) R    = R_new;
                            if (!Y_new || !R_new) return cleanup_and_fail();
                            std::fill(&Y_od[n * alloc_Y_cols], &Y_od[n * want], T(0));
                            std::fill(&R[n * alloc_Y_cols],    &R[n * want],    T(0));
                            alloc_Y_cols = want;
                        }

                        X_i  = &X_ev[m * x_row];
                        Y_i  = &Y_od[n * y_cols];
                        R_i  = &R[x_row];
                        R_ii = &R[n * y_cols + x_row];

                        if(this -> timing)
                            gemm_A_t_start = steady_clock::now();
                        // Y_i = A' * X_i
                        A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, w, m, 1.0, X_i, m, 0.0, Y_i, n);

                        if(this -> timing) {
                            gemm_A_t_stop = steady_clock::now();
                            gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                            allocation_t_start  = steady_clock::now();
                        }



                        if(this -> timing) {
                            allocation_t_stop  = steady_clock::now();
                            allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                            reorth_t_start  = steady_clock::now();
                        }

                        if (y_cols > 0) {
                            // R_i' = Y_i' * Y_od
                            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, w, y_cols, n, 1.0, Y_i, n, Y_od, n, 0.0, R_i, n);

                            // Y_i = Y_i - Y_od * R_i
                            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, w, y_cols, -1.0, Y_od, n, R_i, n, 1.0, Y_i, n);

                            // Reorthogonalization
                            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, w, y_cols, n, 1.0, Y_i, n, Y_od, n, 0.0, Y_orth_buf, k);
                            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, w, y_cols, -1.0, Y_od, n, Y_orth_buf, k, 1.0, Y_i, n);
                        }

                        if(this -> timing) {
                            reorth_t_stop  = steady_clock::now();
                            reorth_t_dur   += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
                        }

                        // Perform explicit QR via a method of choice
                        if(this -> qr_exp == Subroutines::QR_explicit::cqrrt) {
                            if(this -> timing)
                                qr_t_start = steady_clock::now();

                            // R_11_trans is allocated once and REUSED every iteration, so
                            // a failed CQRRT would leave the previous block's healthy
                            // diagonal in place and the rank criterion below would read
                            // stale values and detect nothing. Clear it, and honour the
                            // status: CQRRT returns nonzero on rank deficiency
                            // (rl_cqrrt.hh, diag_is_nonzero / potrf failure), which is
                            // precisely the condition we must not discard.
                            std::fill(R_11_trans, R_11_trans + k * k, (T)0.0);
                            int cq_status = CQRRT -> call(n, w, Y_i, n, R_11_trans, k, d_factor, state);
                            if (cq_status != 0) {
                                // A potrf failure leaves the block overwritten and which columns
                                // were healthy unknown, so no probe can be read and refilling
                                // would fabricate: retract the source X block's refills and stop.
                                x_cols -= pending_refills;
                                pending_refills = 0;
                                this->refills_exhausted = true;
                                this->final_block_width = 0;
                                this->termination_reason = BKTermination::rank_deficient;
                                break;
                            }
                            // Copy R_ii over to R's (in transposed format).

                            util::transposition(0, w, R_11_trans, k, R_ii, n, 1);
                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                            }
                        } else {
                            // [Y_i, R_ii] = qr(Y_i, 0)
                            if(this -> timing)
                                qr_t_start = steady_clock::now();
                            lapack::geqrf(n, w, Y_i, n, tau);

                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                                r_cpy_t_start = steady_clock::now();
                            }

                            // Copy R_ii over to R's (in transposed format).
                            util::transposition(0, w, Y_i, n, R_ii, n, 1);

                            if(this -> timing) {
                                r_cpy_t_stop  = steady_clock::now();
                                r_cpy_t_dur  += duration_cast<microseconds>(r_cpy_t_stop - r_cpy_t_start).count();
                                ungqr_t_start = steady_clock::now();
                            }

                            // Convert Y_i into an explicit form. It is now stored in Y_odd as it should be.
                            lapack::ungqr(n, w, w, Y_i, n, tau);

                            if(this -> timing) {
                                ungqr_t_stop  = steady_clock::now();
                                ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
                            }
                        }

                        // Rank criterion, right basis. Relative to ||A||, reads the whole
                        // trailing block, and yields a width so the healthy prefix survives.
                        {
                            int64_t r_blk = util::block_numerical_rank<T>(w, R_ii, n, norm_A, tau_eff);

                            // (1) Probe, measured BEFORE the zeroing below. The source X block's
                            // last pending_refills columns are refills, and R_ii holds the factor
                            // transposed, so their images are its trailing ROWS (all w columns).
                            // A full-width block has already confirmed them; only a deficient one
                            // needs the measurement.
                            const bool probing = (pending_refills > 0 && r_blk < w);
                            bool null_refills  = false;
                            if (probing) {
                                T acc = 0;
                                for (int64_t j = 0; j < w; ++j)
                                    for (int64_t i = w - pending_refills; i < w; ++i)
                                        acc += R_ii[i + j * n] * R_ii[i + j * n];
                                null_refills = (r_blk == 0) || (std::sqrt(acc) <= tau_eff * norm_A);
                            }

                            if (r_blk < w) {
                                // (2) Zero the REJECTED COLUMNS of the diagonal block. Their true
                                // value is zero by the band's block-bidiagonal structure, and
                                // the error this introduces is bounded by tau*||A||, which is
                                // exactly the order the criterion just declared negligible.
                                // That bound is the mathematical content of prune-and-narrow,
                                // and it holds unchanged for a refill placed in those slots.
                                //
                                // A no-op while the loop still breaks here, because those
                                // positions lie beyond end_cols. It stops being a no-op the
                                // moment a narrowed block continues: end_cols then grows past
                                // them and they become interior to the reported band, with
                                // nothing else ever overwriting them.
                                //
                                // Note the copy above cannot simply be narrowed to r_blk
                                // instead: util::transposition writes column i of the factor
                                // into ROW i of R_ii, so restricting it would drop genuine
                                // subdiagonal entries belonging to fully accepted columns.
                                for (int64_t j = r_blk; j < w; ++j)
                                    for (int64_t i = 0; i < w; ++i)
                                        R_ii[i + j * n] = T(0);
                            }

                            // (3) Probe outcome. Dead images put the refills in the numerical
                            // null space of A^T, so retract them; they are the last X columns,
                            // which makes that a counter change. Dead or not, a deficient probe
                            // switches refilling off for the rest of the run.
                            if (probing) {
                                if (null_refills)
                                    x_cols -= pending_refills;
                                this->refills_exhausted = true;
                            }
                            pending_refills = 0;

                            // (4) REFILL the rejected slots with random directions orthogonal to
                            // every accepted Y column and the kept prefix, so the run continues
                            // where the Krylov space closed early. Y lives in R^n, hence the room.
                            // The band entries for the refills stay the zeros written in (2).
                            int64_t n_ref = 0;
                            const int64_t room = n - (y_cols + r_blk);
                            if (r_blk < w && this->refill_dead_columns && !this->refills_exhausted) {
                                n_ref = std::max<int64_t>(0, std::min(w - r_blk, room));
                                if (n_ref > 0) {
                                    refill(n, Y_od, y_cols + r_blk, &Y_i[n * r_blk], n_ref, Y_orth_buf, n);
                                    ++this->refilled_blocks;
                                }
                            }

                            // (5) A dead block that nothing refills is terminal: saturated when
                            // refilling is on and only the room ran out, rank_deficient otherwise.
                            if (r_blk == 0 && n_ref == 0) {
                                this->final_block_width = 0;
                                this->termination_reason =
                                    (this->refill_dead_columns && !this->refills_exhausted && room <= 0)
                                        ? BKTermination::saturated : BKTermination::rank_deficient;
                                break;
                            }

                            // (6) PRUNE AND NARROW, then refill: keep the healthy prefix, append
                            // the refills after it, and carry on rather than stopping with the
                            // right basis short. w_last carries the width to the next iteration,
                            // which is what makes the block widths variable.
                            if (r_blk < w)
                                ++this->narrowed_blocks;
                            y_cols += r_blk + n_ref;
                            w_last  = r_blk + n_ref;
                            this->final_block_width = r_blk;
                            pending_refills = n_ref;
                        }

                        // Buffer growth and pointer derivation both happen at the top of the
                        // branch now, so there is nothing to advance here.
                    }
                    else {
                        // EVEN: build an X block from the last accepted Y block, write the S band.
                        //
                        // S is indexed the same way as R, (row = X column, col = Y column), and
                        // its leading dimension is n + k because the diagonal block sits one
                        // block BELOW the diagonal. Narrowing only ever reduces w, so it can
                        // only increase the slack in that extra k rows: the deepest write is
                        // row x_cols + w - 1, and the saturation guard holds x_cols <= n.
                        const int64_t w     = w_last;
                        const int64_t y_col = y_cols - w;

                        // Growth FIRST; see the odd branch. S is sized by the X-column count
                        // because that bounds its ROW extent (S_ii sits at row x_cols), and
                        // x_cols >= y_cols always, so it also covers S's y_col column offset.
                        if (!prealloc && x_cols + w > alloc_X_cols) {
                            int64_t want = x_cols + w;
                            T* X_new = ( T * ) realloc(X_ev, m * want * sizeof( T ));
                            T* S_new = ( T * ) realloc(S,    (n + k) * want * sizeof( T ));
                            if (X_new) X_ev = X_new;
                            if (S_new) S    = S_new;
                            if (!X_new || !S_new) return cleanup_and_fail();
                            std::fill(&X_ev[m * alloc_X_cols], &X_ev[m * want], T(0));
                            std::fill(&S[(n + k) * alloc_X_cols], &S[(n + k) * want], T(0));
                            alloc_X_cols = want;
                        }

                        Y_i  = &Y_od[n * y_col];
                        X_i  = &X_ev[m * x_cols];
                        S_i  = &S[(n + k) * y_col];
                        S_ii = &S[(n + k) * y_col + x_cols];

                        if(this -> timing)
                            gemm_A_t_start = steady_clock::now();

                        // X_i = A * Y_i
                        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, w, n, 1.0, Y_i, n, 0.0, X_i, m);

                        if(this -> timing) {
                            gemm_A_t_stop = steady_clock::now();
                            gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                            allocation_t_start  = steady_clock::now();
                        }


                        if(this -> timing) {
                            allocation_t_stop  = steady_clock::now();
                            allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                            reorth_t_start  = steady_clock::now();
                        }

                        // S_i = X_ev' * X_i
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, x_cols, w, m, 1.0, X_ev, m, X_i, m, 0.0, S_i, n + k);

                        //X_i = X_i - X_ev * S_i;
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, w, x_cols, -1.0, X_ev, m, S_i, n + k, 1.0, X_i, m);

                        // Reorthogonalization
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, x_cols, w, m, 1.0, X_ev, m, X_i, m, 0.0, X_orth_buf, n + k);
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, w, x_cols, -1.0, X_ev, m, X_orth_buf, n + k, 1.0, X_i, m);

                        if(this -> timing) {
                            reorth_t_stop  = steady_clock::now();
                            reorth_t_dur   += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
                        }

                        // Perform explicit QR via a method of choice
                        if(this -> qr_exp == Subroutines::QR_explicit::cqrrt) {
                            if(this -> timing)
                                qr_t_start = steady_clock::now();

                            // Honour the status, matching the odd branch. Unlike that
                            // branch there is no stale-buffer hazard here: S_ii is a fresh
                            // region of S each iteration, zeroed by the initial calloc or
                            // by the fill after realloc. A failed call still leaves a partial
                            // factor there (CQRRT writes into S_ii before its checks), and the
                            // rank test below would then refill, so the status must decide.
                            //
                            // DEFENSIVE, not a fix for observed behaviour. A sweep of exact
                            // ranks 11 to 32 at block size 10 under cqrrt never produced a
                            // zero-width even-side block: non-multiples of the block size
                            // stop with final_block_width = r mod 10 in [1,9], and
                            // multiples stop one iteration earlier via norm_converged. See
                            // TestBK.BK_even_terminal_band_identity_cqrrt. The check earns
                            // its place because relying on the probe means relying on the
                            // buffer happening to be zeroed, and because once a narrowed
                            // block continues instead of terminating, a fully dead even-side
                            // block becomes reachable.
                            int cq_status_ev = CQRRT -> call(m, w, X_i, m, S_ii, n + k, d_factor, state);
                            if (cq_status_ev != 0) {
                                // As in the odd branch: which columns were healthy is unknown, so
                                // refilling would fabricate. Retract the source Y block's refills.
                                y_cols -= pending_refills;
                                pending_refills = 0;
                                this -> refills_exhausted = true;
                                this -> final_block_width = 0;
                                this -> termination_reason = BKTermination::rank_deficient;
                                break;
                            }

                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                            }

                        } else {
                            // [X_i, S_ii] = qr(X_i, 0);
                            if(this -> timing)
                                qr_t_start = steady_clock::now();

                            lapack::geqrf(m, w, X_i, m, tau);

                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                                s_cpy_t_start = steady_clock::now();
                            }

                            // Copy S_ii over to S's space under S_i (offset down by iter_od * k)
                            lapack::lacpy(MatrixType::Upper, w, w, X_i, m, S_ii, n + k);

                            if(this -> timing) {
                                s_cpy_t_stop  = steady_clock::now();
                                s_cpy_t_dur  += duration_cast<microseconds>(s_cpy_t_stop - s_cpy_t_start).count();
                                ungqr_t_start = steady_clock::now();
                            }

                            // Convert X_i into an explicit form. It is now stored in X_ev as it should be
                            lapack::ungqr(m, w, w, X_i, m, tau);

                            if(this -> timing) {
                                ungqr_t_stop  = steady_clock::now();
                                ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
                            }
                        }

                        // Early termination
                        // if (abs(S(end)) <= sqrt(eps('T')))
                        // Rank criterion, left basis. S_ii has leading dimension n + k and
                        // is written upper-triangular by lacpy, where R is lower; the shared
                        // helper reads the full square sub-block so both are handled.
                        {
                            int64_t r_blk = util::block_numerical_rank<T>(w, S_ii, n + k, norm_A, tau_eff);

                            // (1) Probe, measured BEFORE the zeroing below; the mirror of the odd
                            // branch. S_ii is upper triangular, so the images of the source Y
                            // block's trailing refills are its trailing COLUMNS (all w rows).
                            const bool probing = (pending_refills > 0 && r_blk < w);
                            bool null_refills  = false;
                            if (probing) {
                                T acc = 0;
                                for (int64_t j = w - pending_refills; j < w; ++j)
                                    for (int64_t i = 0; i < w; ++i)
                                        acc += S_ii[i + j * (n + k)] * S_ii[i + j * (n + k)];
                                null_refills = (r_blk == 0) || (std::sqrt(acc) <= tau_eff * norm_A);
                            }

                            if (r_blk < w) {
                                // (2) Zero the REJECTED ROWS of the diagonal block; see the odd
                                // branch for why the true value there is zero and why this is
                                // a no-op only until a narrowed block continues. Rows rather
                                // than columns because S_ii is written upper-triangular by
                                // lacpy, where R_ii is lower.
                                for (int64_t j = 0; j < w; ++j)
                                    for (int64_t i = r_blk; i < w; ++i)
                                        S_ii[i + j * (n + k)] = T(0);
                            }

                            // (3) Probe outcome; see the odd branch. Here the refills are the last
                            // Y columns, so retraction is y_cols -= pending_refills.
                            if (probing) {
                                if (null_refills)
                                    y_cols -= pending_refills;
                                this->refills_exhausted = true;
                            }
                            pending_refills = 0;

                            // (4) REFILL on the left side; see the odd branch. X lives in R^m and
                            // the band's row extent is bounded by n (S has n + k rows), so the
                            // room is min(m, n), never n alone.
                            int64_t n_ref = 0;
                            const int64_t room = std::min(m, n) - (x_cols + r_blk);
                            if (r_blk < w && this->refill_dead_columns && !this->refills_exhausted) {
                                n_ref = std::max<int64_t>(0, std::min(w - r_blk, room));
                                if (n_ref > 0) {
                                    refill(m, X_ev, x_cols + r_blk, &X_i[m * r_blk], n_ref, X_orth_buf, n + k);
                                    ++this->refilled_blocks;
                                }
                            }

                            // (5) Terminal, as in the odd branch.
                            if (r_blk == 0 && n_ref == 0) {
                                this->final_block_width = 0;
                                this->termination_reason =
                                    (this->refill_dead_columns && !this->refills_exhausted && room <= 0)
                                        ? BKTermination::saturated : BKTermination::rank_deficient;
                                break;
                            }

                            // (6) PRUNE AND NARROW on the left side, then refill; see the odd
                            // branch. THIS is the half that fixes the shortfall. The left basis
                            // runs a block ahead of the right, so it reaches the numerical rank
                            // first; stopping here is what stranded the right basis up to k
                            // columns short and left the leading triplets non-convergent.
                            if (r_blk < w)
                                ++this->narrowed_blocks;
                            x_cols += r_blk + n_ref;
                            w_last  = r_blk + n_ref;
                            this->final_block_width = r_blk;
                            pending_refills = n_ref;
                        }

                        // Buffer growth and pointer derivation both happen at the top of the
                        // branch now, so there is nothing to advance here.

                        if(this -> timing) {
                            allocation_t_stop  = steady_clock::now();
                            allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                        }
                    }

                    if(this -> timing)
                        norm_t_start = steady_clock::now();

                    // This is only changed on odd iters.
                    //
                    // Uplo::Lower, not Upper. R is written by
                    // util::transposition(0, k, Y_i, n, R_ii, n, /*copy_upper_triangle=*/1),
                    // which sets AT(i,j) = A(j,i) for j <= i (rl_util.hh:313-317), so the
                    // stored buffer is LOWER triangular. Asking lantr for the upper triangle
                    // returned the diagonal and the exact zeros above it, making norm_R equal
                    // ||diag(R)||_F -- a severe undercount against
                    // threshold = sqrt(1 - tol^2) * ||A||_F. The consequence was that
                    // norm_converged almost never fired and rank_deficient silently absorbed
                    // terminations belonging to this criterion: before this fix
                    // ABRIK_adaptive_norm_converged itself terminated as rank_deficient.
                    //
                    // Trapezoidal, not square. x_cols >= y_cols always, and after a narrowing
                    // the two differ, so a square y_cols by y_cols window would silently drop
                    // the R_i rows belonging to the last accepted X block. Legal because every
                    // written entry is on or below the diagonal: R_ii starts at row
                    // x_cols - w >= y_cols, which is its own column offset, and is itself
                    // lower triangular. Identical to the old square call whenever nothing has
                    // narrowed.
                    if (iter % 2 != 0)
                        norm_R = lapack::lantr(Norm::Fro, Uplo::Lower, Diag::NonUnit, x_cols, y_cols, R, n);

                    if(this -> timing) {
                        norm_t_stop       = steady_clock::now();
                        norm_t_dur        += duration_cast<microseconds>(norm_t_stop - norm_t_start).count();
                        main_loop_t_stop  = steady_clock::now();
                        main_loop_t_dur   += duration_cast<microseconds>(main_loop_t_stop - main_loop_t_start).count();
                    }

                    // Report terminal conditions before an exhausted budget so callers
                    // cannot resume an already finished factorization.
                    // Frobenius-content convergence (criterion 1 above): ||R||_F exceeding
                    // sqrt(1 - tol^2)||M||_F means ||M - hat(M)||_F <= tol * ||M||_F.
                    //
                    // This check must come BEFORE ++iter. `iter` is the count of COMPLETED
                    // iterations, and end_cols = ((iter + 1) / 2) * k below reads it that
                    // way; the max_iters_reached exit below likewise breaks before the
                    // increment. Breaking after it left iter one too high, so end_cols
                    // claimed a block that was never built and gesdd read uninitialized
                    // columns of Y_od/X_ev. That was latent until the Uplo fix above: with
                    // norm_R stuck at ||diag(R)|| this exit essentially never fired, so the
                    // miscount was unreachable. Fixing the norm alone turned six passing
                    // tests into residuals of order 1.
                    if(norm_R > threshold) {
                        this->termination_reason = BKTermination::norm_converged;
                        break;
                    }

                    // Saturation guard. The right basis lives in R^n, so it cannot exceed n
                    // columns; ((iter + 1) / 2) * k is the count already built (the same
                    // expression end_cols uses below), and another block needs k more.
                    //
                    // This is a termination criterion in its own right, independent of any
                    // rank test, and it is also a memory-safety bound. R_ii is placed at row
                    // k*(iter_ev+1) in a buffer with leading dimension n, and S_ii at row
                    // k*(iter_od+1) with leading dimension n+k. Once the basis passes n
                    // columns those writes run past the end of their column and land in the
                    // next one: silent corruption of the band, no segfault, and nothing for
                    // a sanitizer to catch because the memory is validly allocated.
                    //
                    // Until now this job was being done implicitly by the rank-deficiency
                    // exit, which is why removing that exit on 2026-07-29 broke nine tests
                    // that have nothing to do with rank deficiency: every non-adaptive test
                    // is budgeted to exactly the saturation count (test_abrik.cc:189) and
                    // relied on it to stop. Making the guard explicit is what allows the
                    // rank criterion to be changed safely.
                    //
                    // Ordered after norm_converged so that genuinely exhausting the
                    // Frobenius content still reports as such rather than as saturation.
                    //
                    // Only ODD iterations append to the right basis, so only they are
                    // constrained by n. An even iteration appends to the LEFT basis, which
                    // lives in R^m, and gives end_rows = end_cols + k; blocking it as well
                    // would cut the run one half-step short and force extraction through
                    // the narrower odd/R path for no reason. There can be at most one even
                    // iteration after the final odd one, so guarding the odd side alone
                    // bounds both.
                    //
                    // Stated as the memory bound it always was: x_cols > n. Under fixed widths
                    // this is exactly the old expression, since ((iter + 1) / 2) * k was y_cols
                    // and x_cols was y_cols + k at the end of an even iteration. x_cols changes
                    // nowhere but the even branch, so one check per even iteration covers every
                    // change to it, and it is sufficient for the whole next iteration pair: the
                    // next odd writes R_ii at rows [x_cols - w, x_cols) with ld n, and the next
                    // even writes S_ii at rows [x_cols, x_cols + w) with ld n + k.
                    if ((iter % 2 == 0) && (x_cols > n)) {
                        this->termination_reason = BKTermination::saturated;
                        break;
                    }
                    if (iter >= max_iters) {
                        this->termination_reason = BKTermination::max_iters_reached;
                        break;
                    }
                    ++iter;
                }

                // Set output state
                this->norm_R_end = norm_R;
                this->num_krylov_iters = iter;
                final_iter_is_odd = (iter % 2 != 0);
                // end_rows and end_cols are read off the accepted-column counters, less the
                // refills of the newest block that no iteration has probed: X's after an even
                // final iteration, Y's after an odd one. Those stay in the basis and in the
                // saved counters, so resume() restores and probes them; they are never reported.
                //
                // This replaces a reconstruction from `iter` that needed a separate truncation
                // adjustment per parity, because x_cols and y_cols already ARE those two
                // quantities up to pending refills. With refilling off pending_refills is always
                // zero, and the old formula agrees in every case it handled:
                //
                //   odd terminal, full width   old end_cols = ((iter+1)/2)*k = y_cols
                //   odd terminal, truncated    old end_cols = full_cols - (k - width),
                //                              and y_cols = prev_y_cols + width, which is
                //                              the same number since full_cols = prev_y_cols + k
                //   even terminal, full width  old end_rows = full_cols + k = x_cols
                //   even terminal, truncated   old end_rows = full_cols + width = x_cols
                //
                // The parity-dependent adjustment disappears because the counters are already
                // per-side; final_block_width survives as a diagnostic only.
                end_rows = final_iter_is_odd ? x_cols : x_cols - pending_refills;
                end_cols = final_iter_is_odd ? y_cols - pending_refills : y_cols;

                // Save resume state, so resume() restores rather than reconstructs.
                this->saved_x_cols          = x_cols;
                this->saved_y_cols          = y_cols;
                this->saved_w_last          = w_last;
                this->saved_alloc_X_cols    = alloc_X_cols;
                this->saved_alloc_Y_cols    = alloc_Y_cols;
                this->saved_pending_refills = pending_refills;

                // Free internal temporaries (NOT X_ev, Y_od, R, S; those are returned to caller)
                free(tau);
                free(Y_orth_buf);
                free(X_orth_buf);
                if(R_11_trans != nullptr) {
                    free(R_11_trans);
                }

                if(this -> timing) {
                    bk_total_t_stop = steady_clock::now();
                    bk_total_t_dur  = duration_cast<microseconds>(bk_total_t_stop - bk_total_t_start).count();

                    this -> times.resize(10);
                    this -> times = {allocation_t_dur, ungqr_t_dur, reorth_t_dur, qr_t_dur,
                                     gemm_A_t_dur, main_loop_t_dur, sketching_t_dur,
                                     r_cpy_t_dur, s_cpy_t_dur, norm_t_dur};
                }
                return 0;
            }
    };
}
