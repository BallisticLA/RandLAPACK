#pragma once

#include "rl_bk.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_util_linop.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <climits>

using namespace std::chrono;

namespace RandLAPACK {

    /// ABRIK algorithm is a method for finding truncated SVD based on block Krylov iterations.
    /// This algorithm is a version of Algroithm A.1 from https://arxiv.org/pdf/2306.12418.pdf
    ///
    /// The main difference is in the fact that an economy SVD is performed only once at the very end
    /// of the algorithm run and that the termination criteria is not based on singular vectir residual evaluation.
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

        // Adaptive mode: check SVD residual after BK and resume if needed.
        bool adaptive;             // Enable adaptive residual checking (default: false).
        int adaptive_increment;    // Extra BK iterations per retry (0 = use max_krylov_iters).
        int adaptive_max_retries;  // Hard limit on resume attempts (default: 10).

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
            adaptive_increment = 0;
            adaptive_max_retries = 10;
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
            linops::GenLinOp<T> A_linop(m, n, A, lda, Layout::ColMajor);
            return this->call(A_linop, k, U, V, Sigma, state);
        }

        // ABRIK call that accepts sparse matrix.
        template <RandBLAS::sparse_data::SparseMatrix SpMat>
        int call(
            int64_t m,
            int64_t n,
            SpMat &A,
            int64_t lda,
            int64_t k,
            T* &U,
            T* &V,
            T* &Sigma,
            RandBLAS::RNGState<RNG> &state
        ) {
            linops::SpLinOp<SpMat> A_linop(m, n, A);
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

                // Call BK to build Krylov subspaces and band matrices
                T* X_ev = nullptr;
                T* Y_od = nullptr;
                T* R    = nullptr;
                T* S    = nullptr;
                int64_t end_rows = 0, end_cols = 0;
                bool final_iter_is_odd = false;

                int status = bk_obj.call(A, k, X_ev, Y_od, R, S,
                                         end_rows, end_cols, final_iter_is_odd, state);

                // Read back BK outputs
                this->num_krylov_iters = bk_obj.num_krylov_iters;
                this->norm_R_end       = bk_obj.norm_R_end;

                if (status != 0) return status;

                int64_t m = A.n_rows;
                int64_t n = A.n_cols;
                int increment = (this->adaptive_increment > 0)
                              ? this->adaptive_increment : this->max_krylov_iters;

                T* U_hat  = nullptr;
                T* VT_hat = nullptr;
                int retries = 0;

                // SVD + reconstruction loop (runs once in non-adaptive mode).
                while (true) {
                    // Phase: SVD on band matrix + factor reconstruction
                    if(this -> timing)
                        allocation_t_start = steady_clock::now();

                    // Internal SVD workspace — freed in this function.
                    U_hat  = ( T * ) malloc( end_rows * end_cols * sizeof( T ) );
                    VT_hat = ( T * ) malloc( end_cols * end_cols * sizeof( T ) );

                    // Output arrays — ownership transfers to caller (use delete[]).
                    Sigma = new T[std::min(end_cols, end_rows)]();
                    U     = new T[m * end_cols]();
                    V     = new T[n * end_cols]();

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

                    // --- Adaptive residual check ---
                    T residual = linops::svd_residual<T>(A, U, V, Sigma, end_cols);

                    if (residual <= this->tol) {
                        if (this->verbose)
                            printf("ABRIK adaptive: converged, residual %e <= tol %e after %d retries.\n",
                                   residual, this->tol, retries);
                        break;
                    }

                    if (bk_obj.termination_reason == BKTermination::norm_converged) {
                        std::cerr << "ABRIK adaptive: BK terminated via norm convergence. "
                                  << "Cannot improve further. Residual = " << residual
                                  << ", tol = " << this->tol << std::endl;
                        break;
                    }
                    if (bk_obj.termination_reason == BKTermination::rank_deficient) {
                        std::cerr << "ABRIK adaptive: BK terminated due to rank deficiency. "
                                  << "Cannot improve further. Residual = " << residual
                                  << ", tol = " << this->tol << std::endl;
                        break;
                    }
                    if (retries >= this->adaptive_max_retries) {
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

                    bk_obj.max_krylov_iters += increment;
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

    private:
        BK<T, RNG> bk_obj;
    };
}
