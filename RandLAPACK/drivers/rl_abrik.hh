#pragma once

#include "rl_bk.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_svd_residual.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <climits>

using namespace std::chrono;

namespace RandLAPACK {

    /// ABRIK algorithm is a method for finding truncated SVD based on block Krylov iterations.
    /// This algorithm is a version of Algorithm A.1 from https://arxiv.org/pdf/2306.12418.pdf
    ///
    /// The main difference is in the fact that an economy SVD is performed only once at the very end
    /// of the algorithm run and that the termination criterion is not based on singular vector residual evaluation.
    /// Instead, the scheme terminates if:
    ///     1. ||R||_F > sqrt(1 - eps^2) ||A||_F, which ensures that we've exhausted all vectors and doing more
    ///        iterations would bring no benefit or that ||A - hat(A)||_F < eps * ||A||_F.
    ///     2. Stop if the bottom right entry of R or S is numerically close to zero (up to square root of machine eps).
    ///
    /// The main cost of this algorithm comes from large GEMMs with the input matrix A.
    ///
    /// The algorithm optionally times all of its subcomponents through a user-defined 'timing' parameter.
    ///
    /// ABRIK is a driver that delegates the block Krylov iteration to the BK computational routine,
    /// then performs SVD on the resulting band matrix and reconstructs the final U, Sigma, V factors.
    /// This follows the same pattern as RSVD (driver) + QB (comp).

// Backward compatibility alias
using ABRIKSubroutines = BKSubroutines;

/// Why the adaptive loop stopped. Reported through ABRIK::termination_reason so
/// callers do not have to infer it from the residual, which cannot distinguish an
/// exhausted retry budget from a saturated Krylov subspace.
enum class ABRIKTermination {
    not_adaptive,    ///< Adaptive mode was off; a single pass was run.
    converged,       ///< Assessed error fell to or below tol over the full assessed rank.
    max_retries,     ///< Retry budget exhausted with the error still above tol.
    norm_converged,  ///< BK exhausted the Frobenius content of the input.
    rank_deficient,  ///< BK could not grow the Krylov subspace any further.
    under_delivered, ///< Fewer triplets exist than were asked for; see below.
    saturated        ///< The basis reached the ambient dimension; no room to grow.
};

template <typename T, typename RNG>
class ABRIK {
    public:
        // Subroutine used for explicit orthogonalization process.
        using Subroutines = ABRIKSubroutines;
        Subroutines::QR_explicit qr_exp;

        bool verbose;
        bool timing;
        T tol;
        int num_krylov_iters;
        int max_krylov_iters;
        std::vector<long> times;
        T norm_R_end;

        int64_t singular_triplets_found;

        // Adaptive mode: assess the error after BK and resume if needed.
        //
        // The number of leading triplets the error is assessed over is derived, by
        // default, from the initial iteration budget:
        //
        //     assessed_rank = ceil(max_krylov_iters / 2) * b
        //
        // which is exactly the number of triplets that budget produces. So the initial
        // budget states how many triplets you are asking to be accurate, and the growth
        // below states how hard the driver may work to make them so. Deriving it this way
        // makes the request unsatisfiable-by-construction impossible: you cannot ask for
        // more triplets than your own starting budget yields.
        //
        // Set `assessed_rank` explicitly to override that. The derived value is a multiple
        // of the block size, so a specific count such as ten cannot be expressed at b = 4;
        // an evaluation protocol that holds the assessed rank fixed while sweeping the
        // block size needs the override, since block size is a performance knob and the
        // assessed rank is a problem specification. When set, the initial budget must be
        // large enough to produce that many triplets, which is checked at entry.
        //
        // Assessing over ALL computed triplets instead (the behavior before 2026-07-28)
        // cannot work on a decaying spectrum: every restart appends trailing triplets
        // whose relative error is order one, so the assessment is dominated by exactly the
        // terms the restart just introduced and only passes once the Krylov subspace
        // saturates. Measured on a spectrum decaying over six decades, the leading-10
        // error fell from 3.5e-1 to 6.6e-15 while the all-triplets figure stayed near 1.7.
        bool adaptive;             // Enable adaptive error assessment (default: false).
        double adaptive_growth;    // Budget multiplier per retry (default: 2.0).
        int adaptive_max_retries;  // Hard limit on resume attempts (default: 10).

        // Smallest initial budget that yields a triplet; also the adaptive default.
        // Algorithm 1 requires p > 1, so 2 rather than 1.
        static constexpr int adaptive_default_iters = 2;

        // Leading triplets the error is assessed over. Set to 0 (the default) to derive it
        // from the initial budget as above; set > 0 to request a specific count. On exit
        // this always holds the value actually used.
        int64_t assessed_rank;
        ABRIKTermination termination_reason;
        /// Conditioning contract for BK's rank criterion; forwarded to BK::tau. Zero means
        /// derive a default. Mirrors CQRRPT's user-facing `eps`. Kept distinct from `tol`
        /// on purpose: `tol` is the Frobenius-convergence threshold and is ALSO handed to
        /// CQRRT as its eps (rl_bk.hh, CQRRT.emplace(false, tol)), so overloading it again
        /// would make the rank decision move whenever convergence was retuned.
        T tau;

        ABRIK(
            bool verb,
            bool time_subroutines,
            T ep
        ) : bk_obj(verb, time_subroutines, ep) {
            qr_exp = Subroutines::QR_explicit::geqrf_ungqr;
            verbose = verb;
            timing = time_subroutines;
            tol = ep;
            max_krylov_iters = INT_MAX;
            singular_triplets_found = 0;
            adaptive = false;
            adaptive_growth = 2.0;
            adaptive_max_retries = 10;
            assessed_rank = 0;
            termination_reason = ABRIKTermination::not_adaptive;
            tau = 0;
        }

        /// Computes an SVD of the form:
        ///     A = U diag(Sigma) VT.
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
        ///     Sampling dimension of a sketching operator, m >= (k * n) >= n.
        ///
        /// @param[in] U
        ///     On input, a nullptr
        ///
        /// @param[in] V
        ///     On input, a nullptr
        ///
        /// @param[in] Sigma
        ///     On input, a nullptr
        ///
        /// @param[in] state
        ///     RNG state parameter, required for sketching operator generation.
        ///
        /// @param[out] U
        ///     Stores m by ((num_iters / 2) * k) orthonormal matrix of left singular vectors.
        ///
        /// @param[out] V
        ///     Stores n by ((num_iters / 2) * k) orthonormal matrix of right singular vectors.
        ///
        /// @param[out] Sigma
        ///     Stores ((num_iters / 2) * k) singular values.
        ///
        /// @return = 0: successful exit
        ///

        // ABRIK call that accepts a general dense matrix.
        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            int64_t k,
            T* &U,
            T* &V,
            T* &Sigma,
            RandBLAS::RNGState<RNG> &state
        ) {
            // Input parameter validation. Bad inputs would otherwise propagate to a
            // downstream BLAS/LAPACK failure or a segfault, the latter fatal when
            // ABRIK is called through a binding layer (e.g. MEX/MATLAB).
            randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
            randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
            randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (lda must be >= m for ColMajor)";
            randlapack_require(k > 0) << "target rank k=" << k << " must be > 0";
            randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
            linops::DenseLinOp<T> A_linop(m, n, A, lda, Layout::ColMajor);
            return this->call(A_linop, k, U, V, Sigma, state);
        }

        // ABRIK call that accepts sparse matrix.
        template <RandBLAS::sparse_data::SparseMatrix SpMat>
        int call(
            int64_t m,
            int64_t n,
            SpMat &A,
            int64_t k,
            T* &U,
            T* &V,
            T* &Sigma,
            RandBLAS::RNGState<RNG> &state
        ) {
            // Input parameter validation; same MEX-safety motivation as above.
            randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
            randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
            randlapack_require(k > 0) << "target rank k=" << k << " must be > 0";
            linops::SparseLinOp<SpMat> A_linop(m, n, A);
            return this->call(A_linop, k, U, V, Sigma, state);
        }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            int64_t k,
            T* &U,
            T* &V,
            T* &Sigma,
            RandBLAS::RNGState<RNG> &state
        ){
                steady_clock::time_point total_t_start;
                steady_clock::time_point total_t_stop;
                steady_clock::time_point get_factors_t_start;
                steady_clock::time_point get_factors_t_stop;
                steady_clock::time_point allocation_t_start;
                steady_clock::time_point allocation_t_stop;
                long get_factors_t_dur = 0;
                long driver_alloc_t_dur = 0;
                long total_t_dur = 0;

                if(this -> timing)
                    total_t_start = steady_clock::now();

                // Forward config to BK
                bk_obj.qr_exp            = this->qr_exp;
                bk_obj.tol               = this->tol;
                bk_obj.max_krylov_iters  = this->max_krylov_iters;
                bk_obj.verbose           = this->verbose;
                bk_obj.timing            = this->timing;
                bk_obj.tau               = this->tau;

                // Call BK to build Krylov subspaces and band matrices
                T* X_ev = nullptr;
                T* Y_od = nullptr;
                T* R    = nullptr;
                T* S    = nullptr;
                int64_t end_rows = 0, end_cols = 0;
                bool final_iter_is_odd = false;

                // Adaptive setup, before BK runs and before any growth.
                //
                // The assessed rank is fixed here and never tracks end_cols. That is the
                // whole point: deepening the subspace must improve a FIXED set of leading
                // triplets, otherwise each restart manufactures the very error terms that
                // keep the loop from terminating.
                this->termination_reason = ABRIKTermination::not_adaptive;
                int64_t requested_rank = this->assessed_rank;   // 0 = derive
                if (this->adaptive) {
                    if (this->max_krylov_iters == INT_MAX)
                        this->max_krylov_iters = adaptive_default_iters;
                    randlapack_require(this->max_krylov_iters >= 1)
                        << "adaptive mode needs max_krylov_iters >= 1 (got "
                        << this->max_krylov_iters << ")";
                    randlapack_require(this->adaptive_growth > 1.0)
                        << "adaptive_growth=" << this->adaptive_growth
                        << " must be > 1 for the budget to make progress";

                    int64_t derived = ((this->max_krylov_iters + 1) / 2) * k;
                    if (requested_rank <= 0) {
                        this->assessed_rank = derived;
                    } else {
                        // An explicit request must be reachable from the initial budget,
                        // otherwise the first assessment would be taken over fewer triplets
                        // than asked for and the growth would chase a moving target.
                        randlapack_require(derived >= requested_rank)
                            << "assessed_rank=" << requested_rank << " exceeds the "
                            << derived << " triplets that the initial budget produces; "
                            << "raise max_krylov_iters to at least "
                            << (2 * ((requested_rank + k - 1) / k) - 1);
                        this->assessed_rank = requested_rank;
                    }
                } else {
                    this->assessed_rank = 0;
                }

                int status = bk_obj.call(A, k, X_ev, Y_od, R, S,
                                         end_rows, end_cols, final_iter_is_odd, state);

                // Read back BK outputs
                this->num_krylov_iters = bk_obj.num_krylov_iters;
                this->norm_R_end       = bk_obj.norm_R_end;

                if (status != 0) return status;

                int64_t m = A.n_rows;
                int64_t n = A.n_cols;

                T* U_hat  = nullptr;
                T* VT_hat = nullptr;
                int retries = 0;

                // SVD + reconstruction loop (runs once in non-adaptive mode).
                while (true) {
                    // Phase: SVD on band matrix + factor reconstruction
                    if(this -> timing)
                        allocation_t_start = steady_clock::now();

                    // Zero-width guard. The rank criterion can now reject an entire
                    // terminal block, giving end_cols == 0; call_with_checkpoints already
                    // guards this case (:585) but call() did not, and would reach
                    // malloc(0), new T[0] and gesdd with a zero dimension.
                    if (end_cols == 0 || end_rows == 0) {
                        this->singular_triplets_found = 0;
                        if (this->termination_reason == ABRIKTermination::not_adaptive)
                            this->termination_reason = ABRIKTermination::rank_deficient;
                        break;
                    }

                    // Internal SVD workspace: freed in this function.
                    U_hat  = ( T * ) malloc( end_rows * end_cols * sizeof( T ) );
                    VT_hat = ( T * ) malloc( end_cols * end_cols * sizeof( T ) );

                    // Output arrays: ownership transfers to caller (use delete[]).
                    // No value-initialization: Sigma is fully written by gesdd and U, V are
                    // fully written by the beta=0 reconstruction GEMMs below.
                    Sigma = new T[std::min(end_cols, end_rows)];
                    U     = new T[m * end_cols];
                    V     = new T[n * end_cols];

                    if(this -> timing) {
                        allocation_t_stop = steady_clock::now();
                        driver_alloc_t_dur += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                        get_factors_t_start = steady_clock::now();
                    }

                    if (this->adaptive) {
                        // Adaptive: run gesdd on a copy to preserve R/S for potential resume.
                        T* svd_input = ( T * ) malloc( end_rows * end_cols * sizeof( T ) );
                        if (final_iter_is_odd) {
                            lapack::lacpy(MatrixType::General, end_rows, end_cols, R, n, svd_input, end_rows);
                        } else {
                            lapack::lacpy(MatrixType::General, end_rows, end_cols, S, n + k, svd_input, end_rows);
                        }
                        lapack::gesdd(Job::SomeVec, end_rows, end_cols, svd_input, end_rows,
                                      Sigma, U_hat, end_rows, VT_hat, end_cols);
                        free(svd_input);
                    } else {
                        // Non-adaptive: gesdd overwrites R or S directly (they're freed below).
                        if (final_iter_is_odd) {
                            lapack::gesdd(Job::SomeVec, end_rows, end_cols, R, n,
                                          Sigma, U_hat, end_rows, VT_hat, end_cols);
                        } else {
                            lapack::gesdd(Job::SomeVec, end_rows, end_cols, S, n + k,
                                          Sigma, U_hat, end_rows, VT_hat, end_cols);
                        }
                    }

                    // U = X_ev * U_hat
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, end_cols, end_rows,
                               1.0, X_ev, m, U_hat, end_rows, 0.0, U, m);
                    // V = Y_od * V_hat
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, end_cols, end_cols,
                               1.0, Y_od, n, VT_hat, end_cols, 0.0, V, n);

                    this->singular_triplets_found = end_cols;

                    if(this -> timing) {
                        get_factors_t_stop = steady_clock::now();
                        get_factors_t_dur  += duration_cast<microseconds>(get_factors_t_stop - get_factors_t_start).count();
                    }

                    if (!this->adaptive) break;

                    // Assess the error over the leading assessed_rank triplets only.
                    // Early on, before the budget has produced that many, assess over what
                    // exists; end_cols reaches assessed_rank by the end of the first pass.
                    int64_t k_assess = std::min(this->assessed_rank, end_cols);
                    T residual = linops::svd_residual<T>(A, U, V, Sigma, k_assess);

                    // A small residual over FEWER triplets than were asked for is not
                    // convergence. The two cases look identical here but are not: the
                    // subspace may simply not have grown yet (benign, keep going), or it
                    // may be unable to grow at all, in which case the request can never be
                    // met and reporting success would be a silent under-delivery. The
                    // identity matrix is the extreme case: its Krylov space is span(Omega)
                    // and never grows, so a request for any rank above b is unsatisfiable.
                    bool short_of_request = (k_assess < this->assessed_rank);
                    bool cannot_grow =
                        bk_obj.termination_reason == BKTermination::norm_converged ||
                        bk_obj.termination_reason == BKTermination::rank_deficient ||
                        bk_obj.termination_reason == BKTermination::saturated;

                    if (residual <= this->tol && !short_of_request) {
                        this->termination_reason = ABRIKTermination::converged;
                        if (this->verbose)
                            printf("ABRIK adaptive: converged, residual %e <= tol %e over %ld triplets after %d retries.\n",
                                   residual, this->tol, (long)k_assess, retries);
                        break;
                    }

                    if (short_of_request && cannot_grow) {
                        this->termination_reason = ABRIKTermination::under_delivered;
                        if (this->verbose)
                            std::cerr << "ABRIK adaptive: only " << k_assess << " of the "
                                      << this->assessed_rank << " requested triplets exist and the "
                                      << "Krylov subspace cannot grow further. Residual over the "
                                      << "available triplets = " << residual << "." << std::endl;
                        break;
                    }

                    if (bk_obj.termination_reason == BKTermination::norm_converged) {
                        this->termination_reason = ABRIKTermination::norm_converged;
                        if (this->verbose)
                            std::cerr << "ABRIK adaptive: BK exhausted the Frobenius content of the input. "
                                      << "Cannot improve further. Residual = " << residual
                                      << ", tol = " << this->tol << std::endl;
                        break;
                    }
                    if (bk_obj.termination_reason == BKTermination::rank_deficient) {
                        this->termination_reason = ABRIKTermination::rank_deficient;
                        if (this->verbose)
                            std::cerr << "ABRIK adaptive: BK could not grow the Krylov subspace further. "
                                      << "Residual = " << residual << ", tol = " << this->tol << std::endl;
                        break;
                    }
                    // Terminal, like the two above: the basis has reached the ambient
                    // dimension, so retrying with a larger budget cannot produce anything.
                    // Falling through to the retry branch would spend the whole retry
                    // allowance re-deriving the same answer.
                    if (bk_obj.termination_reason == BKTermination::saturated) {
                        this->termination_reason = ABRIKTermination::saturated;
                        if (this->verbose)
                            std::cerr << "ABRIK adaptive: the Krylov basis reached the ambient dimension. "
                                      << "Residual = " << residual << ", tol = " << this->tol << std::endl;
                        break;
                    }
                    if (retries >= this->adaptive_max_retries) {
                        this->termination_reason = ABRIKTermination::max_retries;
                        if (this->verbose)
                            std::cerr << "ABRIK adaptive: reached max retries (" << this->adaptive_max_retries
                                      << "). Residual = " << residual << ", tol = " << this->tol << std::endl;
                        break;
                    }

                    // Not satisfied, BK stopped at max_iters: discard current factors, resume BK.
                    delete[] U;     U     = nullptr;
                    delete[] V;     V     = nullptr;
                    delete[] Sigma; Sigma = nullptr;
                    free(U_hat);    U_hat  = nullptr;
                    free(VT_hat);   VT_hat = nullptr;

                    // Grow the budget multiplicatively. Doubling bounds the overshoot past
                    // the true convergence point at 2x in iterations, hence about 4x in work
                    // since reorthogonalization is quadratic in the iteration count. A larger
                    // ratio buys almost nothing: the per-check costs telescope to
                    // r^2/(r^2-1) of a single check, which is 1.33 at r=2 against 1.01 at
                    // r=10, while the overshoot penalty grows as r^2. The +1 floor keeps the
                    // budget strictly increasing for growth factors close to 1.
                    bk_obj.max_krylov_iters = std::max(
                        (int)std::ceil(this->adaptive_growth * bk_obj.max_krylov_iters),
                        bk_obj.max_krylov_iters + 1);
                    status = bk_obj.resume(A, k, X_ev, Y_od, R, S,
                                           end_rows, end_cols, final_iter_is_odd, state);

                    this->num_krylov_iters = bk_obj.num_krylov_iters;
                    this->norm_R_end       = bk_obj.norm_R_end;

                    if (status != 0) {
                        // BK resume failed (realloc failure); BK already cleaned up its buffers.
                        return status;
                    }

                    ++retries;
                }

                if(this -> timing)
                    allocation_t_start = steady_clock::now();

                // Free BK-allocated buffers and SVD workspace
                free(Y_od);
                free(X_ev);
                free(R);
                free(S);
                free(U_hat);
                free(VT_hat);

                if(this -> timing) {
                    allocation_t_stop = steady_clock::now();
                    driver_alloc_t_dur += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                }

                // Assemble the 13-entry timing vector (same layout as before)
                if(this -> timing) {
                    total_t_stop = steady_clock::now();
                    total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();

                    // BK times: [0]=alloc, [1]=ungqr, [2]=reorth, [3]=qr, [4]=gemm_A,
                    //           [5]=main_loop, [6]=sketching, [7]=r_cpy, [8]=s_cpy, [9]=norm
                    auto& bt = bk_obj.times;
                    long allocation_t_dur = bt[0] + driver_alloc_t_dur;
                    long ungqr_t_dur      = bt[1];
                    long reorth_t_dur     = bt[2];
                    long qr_t_dur         = bt[3];
                    long gemm_A_t_dur     = bt[4];
                    long main_loop_t_dur  = bt[5];
                    long sketching_t_dur  = bt[6];
                    long r_cpy_t_dur      = bt[7];
                    long s_cpy_t_dur      = bt[8];
                    long norm_t_dur       = bt[9];

                    long t_rest = total_t_dur - (allocation_t_dur + get_factors_t_dur + ungqr_t_dur + reorth_t_dur
                                  + qr_t_dur + gemm_A_t_dur + sketching_t_dur + r_cpy_t_dur + s_cpy_t_dur + norm_t_dur);

                    this -> times = {allocation_t_dur, get_factors_t_dur, ungqr_t_dur, reorth_t_dur, qr_t_dur,
                                     gemm_A_t_dur, main_loop_t_dur, sketching_t_dur, r_cpy_t_dur, s_cpy_t_dur,
                                     norm_t_dur, t_rest, total_t_dur};

                    if (this -> verbose) {
                        printf("\n\n/------------ABRIK TIMING RESULTS BEGIN------------/\n");
                        printf("Basic info: b_sz=%ld krylov_iters=%d\n",      k, num_krylov_iters);

                        printf("Allocate and free time:          %25ld μs,\n", allocation_t_dur);
                        printf("Time to acquire the SVD factors: %25ld μs,\n", get_factors_t_dur);
                        printf("UNGQR time:                      %25ld μs,\n", ungqr_t_dur);
                        printf("Reorthogonalization time:        %25ld μs,\n", reorth_t_dur);
                        printf("QR time:                         %25ld μs,\n", qr_t_dur);
                        printf("GEMM A time:                     %25ld μs,\n", gemm_A_t_dur);
                        printf("Sketching time:                  %25ld μs,\n", sketching_t_dur);
                        printf("R_ii cpy time:                   %25ld μs,\n", r_cpy_t_dur);
                        printf("S_ii cpy time:                   %25ld μs,\n", s_cpy_t_dur);
                        printf("Norm R time:                     %25ld μs,\n", norm_t_dur);

                        printf("\nAllocation takes %22.2f%% of runtime.\n",                100 * ((T) allocation_t_dur  / (T) total_t_dur));
                        printf("Factors takes    %22.2f%% of runtime.\n",                  100 * ((T) get_factors_t_dur / (T) total_t_dur));
                        printf("Ungqr takes      %22.2f%% of runtime.\n",                  100 * ((T) ungqr_t_dur       / (T) total_t_dur));
                        printf("Reorth takes     %22.2f%% of runtime.\n",                  100 * ((T) reorth_t_dur      / (T) total_t_dur));
                        printf("QR takes         %22.2f%% of runtime.\n",                  100 * ((T) qr_t_dur          / (T) total_t_dur));
                        printf("GEMM A takes     %22.2f%% of runtime.\n",                  100 * ((T) gemm_A_t_dur      / (T) total_t_dur));
                        printf("Sketching takes  %22.2f%% of runtime.\n",                  100 * ((T) sketching_t_dur   / (T) total_t_dur));
                        printf("R_ii cpy takes   %22.2f%% of runtime.\n",                  100 * ((T) r_cpy_t_dur       / (T) total_t_dur));
                        printf("S_ii cpy takes   %22.2f%% of runtime.\n",                  100 * ((T) s_cpy_t_dur       / (T) total_t_dur));
                        printf("Norm R takes     %22.2f%% of runtime.\n",                  100 * ((T) norm_t_dur        / (T) total_t_dur));
                        printf("Rest takes       %22.2f%% of runtime.\n",                  100 * ((T) t_rest            / (T) total_t_dur));

                        printf("\nMain loop takes  %22.2f%% of runtime.\n",                  100 * ((T) main_loop_t_dur   / (T) total_t_dur));
                        printf("/-------------ABRIK TIMING RESULTS END-------------/\n\n");
                    }
                }
                return 0;
            }

        /// Runs BK iteratively with checkpoints. At each checkpoint, extracts SVD
        /// factors and invokes on_checkpoint(total_matvecs, elapsed_us, residual).
        ///
        /// Elapsed time covers BK iterations + SVD extraction but NOT residual eval.
        /// BK internally uses call()/resume() so no work is repeated between checkpoints.
        ///
        /// @param k                 Block size (matvecs per BK iteration = k).
        /// @param target_rank       How many singular triplets to include in the residual.
        /// @param checkpoint_iters  Sorted list of Krylov iteration counts at which to stop.
        ///                          Must be strictly increasing; last entry is the full budget.
        /// @param on_checkpoint     Called after each checkpoint.
        ///                          Signature: void(int64_t total_matvecs, long elapsed_us, T residual).
        ///                          total_matvecs = k * actual_iters_done (may differ from
        ///                          k * checkpoint_iters[i] if BK terminated early).
        template <RandLAPACK::linops::LinearOperator GLO, typename CheckpointFn>
        int call_with_checkpoints(
            GLO& A,
            int64_t k,
            int64_t target_rank,
            std::vector<int64_t> checkpoint_iters,
            CheckpointFn on_checkpoint,
            RandBLAS::RNGState<RNG>& state
        ) {
            int64_t m = A.n_rows;
            int64_t n = A.n_cols;

            bk_obj.qr_exp          = this->qr_exp;
            bk_obj.tol             = this->tol;
            bk_obj.verbose         = this->verbose;
            bk_obj.timing          = false;
            bk_obj.tau             = this->tau;

            T* X_ev = nullptr, *Y_od = nullptr, *R = nullptr, *S = nullptr;
            int64_t end_rows = 0, end_cols = 0;
            bool final_iter_is_odd = false;
            long elapsed_us = 0;

            for (int ci = 0; ci < (int)checkpoint_iters.size(); ++ci) {
                bk_obj.max_krylov_iters = (int)checkpoint_iters[ci];

                // BK step: call on first, resume on subsequent.
                auto t0 = steady_clock::now();
                int status;
                if (ci == 0)
                    status = bk_obj.call(A, k, X_ev, Y_od, R, S,
                                         end_rows, end_cols, final_iter_is_odd, state);
                else
                    status = bk_obj.resume(A, k, X_ev, Y_od, R, S,
                                           end_rows, end_cols, final_iter_is_odd, state);
                elapsed_us += duration_cast<microseconds>(steady_clock::now() - t0).count();

                if (status != 0) {
                    free(X_ev); free(Y_od); free(R); free(S);
                    return status;
                }

                if (end_cols == 0) {
                    on_checkpoint(0, elapsed_us, (T)1);
                    break;
                }

                // SVD extraction: copy band matrix (preserves BK buffers for resume),
                // then gesdd + two GEMMs. Timed as part of the "total ABRIK cost."
                auto t1 = steady_clock::now();

                T* band = (T*) malloc(end_rows * end_cols * sizeof(T));
                if (final_iter_is_odd)
                    lapack::lacpy(MatrixType::General, end_rows, end_cols, R, n, band, end_rows);
                else
                    lapack::lacpy(MatrixType::General, end_rows, end_cols, S, n + k, band, end_rows);

                T* U_hat  = (T*) malloc(end_rows * end_cols * sizeof(T));
                T* VT_hat = (T*) malloc(end_cols * end_cols * sizeof(T));
                // Fully overwritten below (gesdd + beta=0 GEMMs), so no value-init.
                T* Sigma  = new T[std::min(end_rows, end_cols)];
                T* U      = new T[m * end_cols];
                T* V      = new T[n * end_cols];

                lapack::gesdd(Job::SomeVec, end_rows, end_cols, band, end_rows,
                              Sigma, U_hat, end_rows, VT_hat, end_cols);
                free(band);

                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                           m, end_cols, end_rows, (T)1, X_ev, m, U_hat, end_rows, (T)0, U, m);
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                           n, end_cols, end_cols, (T)1, Y_od, n, VT_hat, end_cols, (T)0, V, n);
                free(U_hat); free(VT_hat);

                elapsed_us += duration_cast<microseconds>(steady_clock::now() - t1).count();

                // Residual check, NOT included in elapsed_us.
                int64_t k_out = std::min(target_rank, end_cols);
                T residual = linops::svd_residual<T>(A, U, V, Sigma, k_out);

                delete[] U; delete[] V; delete[] Sigma;

                on_checkpoint(k * (int64_t)bk_obj.num_krylov_iters, elapsed_us, residual);

                if (bk_obj.termination_reason != BKTermination::max_iters_reached)
                    break;
            }

            free(X_ev); free(Y_od); free(R); free(S);
            return 0;
        }

    private:
        BK<T, RNG> bk_obj;
    };
}
