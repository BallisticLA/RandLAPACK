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

/// How many leading columns of a k-wide block carry operator content.
///
/// Returns the smallest r for which the trailing (k-r)-by-(k-r) sub-block of the block's
/// triangular factor has Frobenius norm at most tau*||A||_F; r = k means the whole block is
/// healthy, r = 0 means none of it is.
///
/// Three things distinguish this from the test it replaces, which was
///     std::abs(R_ii[(n + 1) * (k - 1)]) < std::sqrt(eps)
///
/// 1. It reads the whole trailing block, not the single trailing diagonal entry. One
///    collapsed entry at the end said nothing about the k-1 columns before it.
/// 2. The threshold is RELATIVE to ||A||_F. The old constant was absolute, so the same
///    matrix scaled by 1e8 got a different answer -- measured: 50 triplets claimed on a
///    rank-5 input instead of 10, and a different termination reason.
/// 3. It returns a width rather than a boolean, so the caller can keep the healthy part.
///
/// Follows Balabanov, "Randomized Cholesky QR factorizations", arXiv:2210.09953, Alg. 7
/// step 3, with the anchor carried across to the blocked setting: Alg. 7 measures against
/// ||R||_2 of the factorization it is revealing, which for a whole-matrix factorization is
/// the matrix scale. Anchoring to a *block's* own scale does not survive blocking -- a
/// wholly dead block has no healthy reference, and a block-relative rule then flags nothing
/// (measured: 0 of 5 seeds on an exactly-rank-40 input, residual 3.0e+00). Theorem 5.6 of
/// that paper is the contract this buys: cond(retained) <= 10 n^1.5 r / tau.
///
/// Reads the full square sub-block rather than one triangle, so it is correct for both
/// bands: R is stored lower-triangular by util::transposition, while S_ii is written upper
/// by lacpy(MatrixType::Upper, ...). The untouched triangle is exactly zero either way.
///
/// Degenerate ||A||: a zero or denormal matrix gives a threshold of zero, so only an
/// exactly-zero trailing block satisfies it and the function returns 0 rather than k. A
/// non-finite ||A|| makes every comparison false and returns k. Neither can hang, because
/// termination no longer depends on this function -- that is what the saturation guard is
/// for.
template <typename T>
int64_t block_numerical_rank(int64_t k, const T* Rii, int64_t ldr, T norm_A, T tau) {
    const T thresh = tau * norm_A;
    for (int64_t r = 0; r < k; ++r) {
        T acc = 0;
        for (int64_t j = r; j < k; ++j) {
            for (int64_t i = r; i < k; ++i) {
                const T v = Rii[i + j * ldr];
                acc += v * v;
            }
        }
        if (std::sqrt(acc) <= thresh)
            return r;
    }
    return k;
}

/// Reason BK terminated its main loop.
enum class BKTermination {
    max_iters_reached, ///< Reached max_krylov_iters without convergence (resumable).
    norm_converged,    ///< norm_R exceeded threshold (A's spectral content exhausted).
    rank_deficient,    ///< Near-zero diagonal entry in R or S (subspace can't grow).
    saturated          ///< The basis has as many columns as the ambient dimension allows.
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
        /// Conditioning contract for the rank criterion: the trailing block is judged
        /// against tau*||A||_F. Larger tau retains fewer columns and bounds their
        /// conditioning more tightly (Balabanov Thm 5.6: cond <= 10 n^1.5 r / tau).
        /// Mirrors CQRRPT's user-facing `eps` (rl_cqrrpt.hh:120). Zero means "derive a
        /// default from the problem size", which is done inside call_impl where n is known.
        T tau;
        /// Width of the terminal block after truncation; equals k unless the rank criterion
        /// rejected part of it. Needed because end_cols/end_rows can no longer be
        /// reconstructed from `iter` alone.
        int64_t final_block_width;

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
        ///     Caller must free().
        ///
        /// @param[out] Y_od
        ///     Right Krylov basis (n x end_cols), allocated internally with calloc.
        ///     Caller must free().
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
        ///
        /// @param[out] S
        ///     Band matrix for an even final iteration: end_rows by end_cols with leading
        ///     dimension n + k, allocated internally with calloc. Caller must free().
        ///     The extra k rows exist because the diagonal block sits one block below the
        ///     diagonal (lower Hessenberg). Same orientation convention as R: the buffer as
        ///     stored is the band.
        ///
        /// @param[out] end_rows
        ///     Number of rows in the band matrix for SVD.
        ///
        /// @param[out] end_cols
        ///     Number of columns in the band matrix for SVD.
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
                int64_t iter, iter_od, iter_ev;
                int64_t curr_X_cols, curr_Y_cols;
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

                    iter = 0; iter_od = 0; iter_ev = 0;
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
                    curr_Y_cols = k;
                    curr_X_cols = k;

                    // Initialize pointers.
                    Y_i  = Y_od;
                    X_i  = X_ev;
                    R_i  = NULL;
                    R_ii = R;
                    S_i  = S;
                    S_ii = &S[k];

                    if(this -> timing) {
                        allocation_t_stop  = steady_clock::now();
                        allocation_t_dur   = duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                    }
                } else {
                    // Resume: reconstruct loop state from stored members
                    // Only valid after a prior call() that terminated with max_iters_reached.
                    iter     = this->num_krylov_iters;
                    norm_R   = this->norm_R_end;
                    iter_od  = 1 + iter / 2;
                    iter_ev  = (iter + 1) / 2;
                    curr_X_cols = (1 + iter_ev) * k;
                    curr_Y_cols = iter_od * k;

                    // Grow buffers if the new max_krylov_iters requires more space
                    // than was allocated in the previous call.
                    if (prealloc) {
                        if (max_X_cols > curr_X_cols) {
                            X_ev = ( T * ) realloc(X_ev, m * max_X_cols * sizeof( T ));
                            R    = ( T * ) realloc(R,    n * max_X_cols * sizeof( T ));
                            if (!X_ev || !R) {
                                free(X_ev); free(Y_od); free(R); free(S);
                                X_ev = nullptr; Y_od = nullptr; R = nullptr; S = nullptr;
                                return -1;
                            }
                            std::fill(&X_ev[m * curr_X_cols], &X_ev[m * max_X_cols], T(0));
                            std::fill(&R[n * curr_X_cols],    &R[n * max_X_cols],    T(0));
                        }
                        if (max_Y_cols > curr_Y_cols) {
                            Y_od = ( T * ) realloc(Y_od, n * max_Y_cols * sizeof( T ));
                            S    = ( T * ) realloc(S,    (n + k) * max_Y_cols * sizeof( T ));
                            if (!Y_od || !S) {
                                free(X_ev); free(Y_od); free(R); free(S);
                                X_ev = nullptr; Y_od = nullptr; R = nullptr; S = nullptr;
                                return -1;
                            }
                            std::fill(&Y_od[n * curr_Y_cols],       &Y_od[n * max_Y_cols],       T(0));
                            std::fill(&S[(n + k) * curr_Y_cols], &S[(n + k) * max_Y_cols], T(0));
                        }
                    }

                    // Reconstruct pointers into (potentially reallocated) buffers.
                    X_i  = &X_ev[m * (curr_X_cols - k)];
                    Y_i  = &Y_od[n * (curr_Y_cols - k)];
                    R_i  = &R[iter_ev * k];
                    R_ii = &R[(n * k * iter_ev) + k + (k * (iter_ev - 1))];
                    S_i  = &S[(n + k) * k * (iter_od - 1)];
                    S_ii = &S[(n + k) * k * (iter_od - 1) + k + ((iter_od - 1) * k)];

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
                // 2. Rank deficiency: stop if the trailing diagonal entry of the band just updated
                //    (R on odd iters, S on even) falls below sqrt(eps), i.e. the Krylov subspace
                //    can no longer grow. See the per-branch checks below.
                // The bounded loop also stops at max_krylov_iters (termination_reason set per case;
                // the ABRIK driver only resumes when that was the reason).
                T norm_A = A.fro_nrm();
                // tau = 0 means "derive". n*eps clears Theorem 5.6's floor of 4 n^1.5 r u
                // and is the default until measurement says otherwise; it is user-settable
                // for exactly that reason.
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
                    ++iter_od;
                    // Advance iteration count.
                    ++iter;
                }

                // Main loop: shared for both fresh start and resume.
                while(1) {
                    if(this -> timing)
                        main_loop_t_start = steady_clock::now();

                    if (iter % 2 != 0) {
                        if(this -> timing)
                            gemm_A_t_start = steady_clock::now();
                        // Y_i = A' * X_i
                        A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, X_i, m, 0.0, Y_i, n);

                        if(this -> timing) {
                            gemm_A_t_stop = steady_clock::now();
                            gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                            allocation_t_start  = steady_clock::now();
                        }

                        // Grow X_ev buffer
                        curr_X_cols += k;
                        if (!prealloc) {
                            X_ev = ( T * ) realloc(X_ev, m * curr_X_cols * sizeof( T ));
                        }
                        // Move the X_i pointer
                        X_i = &X_ev[m * (curr_X_cols - k)];


                        if(this -> timing) {
                            allocation_t_stop  = steady_clock::now();
                            allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                            reorth_t_start  = steady_clock::now();
                        }

                        if (iter != 1) {
                            // R_i' = Y_i' * Y_od
                            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, iter_ev * k, n, 1.0, Y_i, n, Y_od, n, 0.0, R_i, n);

                            // Y_i = Y_i - Y_od * R_i
                            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, k, iter_ev * k, -1.0, Y_od, n, R_i, n, 1.0, Y_i, n);

                            // Reorthogonalization
                            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, iter_ev * k, n, 1.0, Y_i, n, Y_od, n, 0.0, Y_orth_buf, k);
                            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, k, iter_ev * k, -1.0, Y_od, n, Y_orth_buf, k, 1.0, Y_i, n);
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
                            int cq_status = CQRRT -> call(n, k, Y_i, n, R_11_trans, k, d_factor, state);
                            if (cq_status != 0) {
                                this->final_block_width = 0;
                                this->termination_reason = BKTermination::rank_deficient;
                                break;
                            }
                            // Copy R_ii over to R's (in transposed format).

                            util::transposition(0, k, R_11_trans, k, R_ii, n, 1);
                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                            }
                        } else {
                            // [Y_i, R_ii] = qr(Y_i, 0)
                            if(this -> timing)
                                qr_t_start = steady_clock::now();
                            lapack::geqrf(n, k, Y_i, n, tau);

                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                                r_cpy_t_start = steady_clock::now();
                            }

                            // Copy R_ii over to R's (in transposed format).
                            util::transposition(0, k, Y_i, n, R_ii, n, 1);

                            if(this -> timing) {
                                r_cpy_t_stop  = steady_clock::now();
                                r_cpy_t_dur  += duration_cast<microseconds>(r_cpy_t_stop - r_cpy_t_start).count();
                                ungqr_t_start = steady_clock::now();
                            }

                            // Convert Y_i into an explicit form. It is now stored in Y_odd as it should be.
                            lapack::ungqr(n, k, k, Y_i, n, tau);

                            if(this -> timing) {
                                ungqr_t_stop  = steady_clock::now();
                                ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
                            }
                        }

                        // Rank criterion, right basis. Relative to ||A||, reads the whole
                        // trailing block, and yields a width so the healthy prefix survives.
                        {
                            int64_t r_blk = block_numerical_rank<T>(k, R_ii, n, norm_A, tau_eff);
                            if (r_blk < k) {
                                this->final_block_width = r_blk;
                                this->termination_reason = BKTermination::rank_deficient;
                                break;
                            }
                        }

                        // Grow R buffer
                        if (!prealloc) {
                            T* R_new = ( T * ) realloc(R, n * curr_X_cols * sizeof( T ));
                            if (!R_new) return cleanup_and_fail();
                            R = R_new;
                            T* temp_r = &R[n * (curr_X_cols - k)];
                            std::fill(temp_r, temp_r + n*k, 0.0);
                        }

                        // Advance R pointers
                        R_i = &R[(iter_ev + 1) * k];
                        R_ii = &R[(n * k * (iter_ev + 1)) + k + (k * (iter_ev))];

                        // Advance even iteration count;
                        ++iter_ev;
                    }
                    else {
                        if(this -> timing)
                            gemm_A_t_start = steady_clock::now();

                        // X_i = A * Y_i
                        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, Y_i, n, 0.0, X_i, m);

                        if(this -> timing) {
                            gemm_A_t_stop = steady_clock::now();
                            gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                            allocation_t_start  = steady_clock::now();
                        }

                        // Grow Y_od buffer
                        curr_Y_cols += k;
                        if (!prealloc) {
                            Y_od = ( T * ) realloc(Y_od, n * curr_Y_cols * sizeof( T ));
                        }
                        // Move the Y_i pointer
                        Y_i = &Y_od[n * (curr_Y_cols - k)];

                        if(this -> timing) {
                            allocation_t_stop  = steady_clock::now();
                            allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                            reorth_t_start  = steady_clock::now();
                        }

                        // S_i = X_ev' * X_i
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, S_i, n + k);

                        //X_i = X_i - X_ev * S_i;
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, S_i, n + k, 1.0, X_i, m);

                        // Reorthogonalization
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, X_orth_buf, n + k);
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, X_orth_buf, n + k, 1.0, X_i, m);

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
                            // by the fill after realloc, so a failed factorisation leaves
                            // zeros rather than a previous block's diagonal, and
                            // block_numerical_rank below would return 0 and exit anyway.
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
                            int cq_status_ev = CQRRT -> call(m, k, X_i, m, S_ii, n + k, d_factor, state);
                            if (cq_status_ev != 0) {
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

                            lapack::geqrf(m, k, X_i, m, tau);

                            if(this -> timing) {
                                qr_t_stop = steady_clock::now();
                                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                                s_cpy_t_start = steady_clock::now();
                            }

                            // Copy S_ii over to S's space under S_i (offset down by iter_od * k)
                            lapack::lacpy(MatrixType::Upper, k, k, X_i, m, S_ii, n + k);

                            if(this -> timing) {
                                s_cpy_t_stop  = steady_clock::now();
                                s_cpy_t_dur  += duration_cast<microseconds>(s_cpy_t_stop - s_cpy_t_start).count();
                                ungqr_t_start = steady_clock::now();
                            }

                            // Convert X_i into an explicit form. It is now stored in X_ev as it should be
                            lapack::ungqr(m, k, k, X_i, m, tau);

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
                            int64_t r_blk = block_numerical_rank<T>(k, S_ii, n + k, norm_A, tau_eff);
                            if (r_blk < k) {
                                this->final_block_width = r_blk;
                                this->termination_reason = BKTermination::rank_deficient;
                                break;
                            }
                        }

                        if(this -> timing) {
                            allocation_t_start  = steady_clock::now();
                        }

                        // Grow S buffer
                        if (!prealloc) {
                            T* S_new = ( T * ) realloc(S, (n + k) * curr_Y_cols * sizeof( T ));
                            if (!S_new) return cleanup_and_fail();
                            S = S_new;
                            T* temp_s = &S[(n + k)* (curr_Y_cols - k)];
                            std::fill(temp_s, temp_s + (n + k) * k, 0.0);
                        }

                        // Advance S pointers
                        S_i  = &S[(n + k) * k * iter_od];
                        S_ii = &S[(n + k) * k * iter_od + k + (iter_od * k)];

                        // Advance odd iteration count;
                        ++iter_od;

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
                    if (iter % 2 != 0)
                        norm_R = lapack::lantr(Norm::Fro, Uplo::Lower, Diag::NonUnit, iter_ev * k, iter_ev * k, R, n);

                    if(this -> timing) {
                        norm_t_stop       = steady_clock::now();
                        norm_t_dur        += duration_cast<microseconds>(norm_t_stop - norm_t_start).count();
                        main_loop_t_stop  = steady_clock::now();
                        main_loop_t_dur   += duration_cast<microseconds>(main_loop_t_stop - main_loop_t_start).count();
                    }

                    if (iter >= max_iters) {
                        this->termination_reason = BKTermination::max_iters_reached;
                        break;
                    }

                    // Frobenius-content convergence (criterion 1 above): ||R||_F exceeding
                    // sqrt(1 - tol^2)||M||_F means ||M - hat(M)||_F <= tol * ||M||_F.
                    //
                    // This check must come BEFORE ++iter. `iter` is the count of COMPLETED
                    // iterations, and end_cols = ((iter + 1) / 2) * k below reads it that
                    // way; the max_iters_reached exit above likewise breaks before the
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
                    if ((((iter + 1) % 2) != 0) && (((iter + 1) / 2) * k + k > n)) {
                        this->termination_reason = BKTermination::saturated;
                        break;
                    }
                    ++iter;
                }

                // Set output state
                this->norm_R_end = norm_R;
                this->num_krylov_iters = iter;
                // end_cols is the number of orthonormal Y_od columns actually built (the V-basis
                // width): one k-column block is appended per odd-branch completion. After `iter`
                // half-steps that count is ceil(iter/2) blocks of k columns.
                //
                // The scalar form ((iter + 1) / 2) * k does the integer division before the
                // multiply. This matters: the old (iter * k + 1) / 2 divided after multiplying,
                // which is identical only when the final iteration is even or k == 1, but on an
                // odd final iteration with even k it undercounts by k/2 and gesdd then silently
                // drops k/2 singular triplets. An odd final iteration is reachable via a
                // rank-deficient break in the odd branch, an odd max_krylov_iters, or an odd
                // checkpoint value.
                //
                // Truncation. When the rank criterion rejected part of the terminal block,
                // only final_block_width of its k columns are real and the rest must not be
                // reported. The two sides need DIFFERENT adjustments, which is why a single
                // formula could not express it and why final_block_width has to be carried
                // out of the loop rather than recovered from `iter`:
                //
                //   odd terminal  -> the truncated block is a Y (right) block, so end_cols
                //                    loses the rejected columns while end_rows keeps the
                //                    full X width. end_rows == end_cols therefore stops
                //                    holding, and forcing it would silently re-admit the
                //                    junk columns as basis width.
                //   even terminal -> the truncated block is an X (left) block, so end_cols
                //                    is untouched and only end_rows shrinks.
                //
                // Getting this backwards is invisible to a test that asserts on end_cols
                // alone: on the even side the count is already correct and it is the LEFT
                // basis that is contaminated.
                const int64_t full_cols = ((iter + 1) / 2) * k;
                const bool truncated = (this->termination_reason == BKTermination::rank_deficient)
                                       && (this->final_block_width < k);
                if (iter % 2 != 0) {
                    end_cols = truncated ? full_cols - (k - this->final_block_width) : full_cols;
                    end_rows = full_cols;
                } else {
                    end_cols = full_cols;
                    end_rows = full_cols + (truncated ? this->final_block_width : k);
                }
                final_iter_is_odd = (iter % 2 != 0);

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
