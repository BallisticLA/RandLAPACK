#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"
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

        /// RBKI algorithm is a method for finding truncated SVD based on block Krylov iterations.
        /// This algorithm is a version of Algroithm A.1 from https://arxiv.org/pdf/2306.12418.pdf
        /// 
        /// The main difference is in the fact that an economy SVD is performed only once at the very end 
        /// of the algorithm run and that the termination criteria is not based on singular vectir residual evaluation.
        /// Instead, the scheme terminates if:
        ///     1. ||R||_F > sqrt(1 - eps^2) ||A||_F, which ensures that we've exhausted all vectors and doing more 
        ///        iterations would bring no benefit or that ||A - hat(A)||_F < eps * ||A||_F.
        ///     2. Stop if the bottom right entry of R or S is numerically close to zero (up to square root of machine eps).
        /// 
        /// The main cos of this algorithm comes from large GEMMs with the input matrix A.
        ///
        /// The algorithm optionally times all of its subcomponents through a user-defined 'timing' parameter.

template <typename T, typename RNG>
class RBKI {
    public:
        bool verbose;
        bool timing;
        T tol;
        int num_krylov_iters;
        int max_krylov_iters;
        std::vector<long> times;
        T norm_R_end;

        int num_threads_some;
        int num_threads_rest;

        RBKI(
            bool verb,
            bool time_subroutines,
            T ep
        ) {
            verbose = verb;
            timing = time_subroutines;
            tol = ep;
            max_krylov_iters = INT_MAX;
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
        ///     On output, an empty matrix.
        ///
        /// @param[in] VT
        ///     On output, an empty matrix.
        ///
        /// @param[in] Sigma
        ///     On output, an empty matrix.
        ///
        /// @param[in] state
        ///     RNG state parameter, required for sketching operator generation.
        ///
        /// @param[out] U
        ///     Stores m by ((num_iters / 2) * k) orthonormal matrix of left singular vectors.
        ///
        /// @param[out] VT
        ///     Stores ((num_iters / 2) * k) * n orthonormal matrix of right singular vectors.
        ///
        /// @param[out] Sigma
        ///     Stores ((num_iters / 2) * k) singular values. 
        ///
        /// @return = 0: successful exit
        ///

        int call(
            const int64_t n_rows,
            const int64_t n_cols,
            const T* A,
            const int64_t lda,
            int64_t m,
            int64_t n,
            int64_t k,
            T* U,
            T* VT,
            T* Sigma,
            RandBLAS::RNGState<RNG> &state
        ) {
            linops::GemLinOp<T> A_linop(n_rows, n_cols, A, lda, Layout::ColMajor);
            return this->call(m, n, A_linop, lda, k, U, VT, Sigma, state);
        }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            int64_t m,
            int64_t n,
            GLO& A,
            int64_t lda,
            int64_t k,
            T* U,
            T* VT,
            T* Sigma,
            RandBLAS::RNGState<RNG> &state
        ){
                steady_clock::time_point allocation_t_start;
                steady_clock::time_point allocation_t_stop;
                steady_clock::time_point get_factors_t_start;
                steady_clock::time_point get_factors_t_stop;
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
                steady_clock::time_point total_t_start;
                steady_clock::time_point total_t_stop;

                long allocation_t_dur  = 0;
                long get_factors_t_dur = 0;
                long ungqr_t_dur       = 0;
                long reorth_t_dur      = 0;
                long qr_t_dur          = 0;
                long gemm_A_t_dur      = 0;
                long main_loop_t_dur   = 0;
                long sketching_t_dur   = 0;
                long r_cpy_t_dur       = 0;
                long s_cpy_t_dur       = 0;
                long norm_t_dur        = 0;
                long total_t_dur       = 0;

                if(this -> timing) {
                    total_t_start = steady_clock::now();
                    allocation_t_start  = steady_clock::now();
                }

                int64_t iter = 0, iter_od = 0, iter_ev = 0, end_rows = 0, end_cols = 0;
                T norm_R = 0;
                int max_iters = this->max_krylov_iters;//std::min(this->max_krylov_iters, (int) (n / (T) k));

                // We need a full copy of X and Y all the way through the algorithm
                // due to an operation with X_odd and Y_odd happening at the end.
                // Below pointers stay the same throughout the alg; the space will be alloacted iteratively
                // Space for Y_i and Y_odd.
                T* Y_od  = ( T * ) calloc( n * k, sizeof( T ) );
                int64_t curr_Y_cols = k;
                // Space for X_i and X_ev. 
                T* X_ev  = ( T * ) calloc( m * k, sizeof( T ) );
                int64_t curr_X_cols = k;

                // While R and S matrices are structured (both band), we cannot make use of this structure through
                // BLAS-level functions.
                // Note also that we store a transposed version of R.
                // 
                // At each iterations, matrices R and S grow by b_sz.
                // At the end, size of R would by d x d and size of S would
                // be (d + 1) x d, where d = numiters_complete * b_sz, d <= n.
                // Note that the total amount of iterations will always be numiters <= n * 2 / block_size
                T* R   = ( T * ) calloc( n * k, sizeof( T ) );
                T* S   = ( T * ) calloc( (n + k) * k, sizeof( T ) );

                // These buffers are of constant size
                T* Y_orth_buf = ( T * ) calloc( k * n, sizeof( T ) );
                T* X_orth_buf = ( T * ) calloc( k * (n + k), sizeof( T ) );

                // Pointers allocation
                // Below pointers will be offset by (n or m) * k at every even iteration.
                T* Y_i  = Y_od;
                T* X_i  = X_ev;
                // S and S pointers are offset at every step.
                T* R_i  = NULL;
                T* R_ii = R;
                T* S_i  = S;
                T* S_ii = &S[k];
                // Pre-decloration of SVD-related buffers.
                T* U_hat = NULL;
                T* VT_hat = NULL;
                // tau space for QR
                T* tau = ( T * ) calloc( k, sizeof( T ) );

                if(this -> timing) {
                    allocation_t_stop  = steady_clock::now();
                    allocation_t_dur   = duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                }

                // Pre-conpute Fro norm of an input matrix.
                T norm_A = lapack::lange(Norm::Fro, m, n, A.A_buff, lda);
                T sq_tol = std::pow(this->tol, 2);
                T threshold =  std::sqrt(1 - sq_tol) * norm_A;

                if(this -> timing)
                    sketching_t_start  = steady_clock::now();

                // Generate a dense Gaussian random matrx.
                // OMP_NUM_THREADS=4 seems to be the best option for dense sketch generation.
            #if RandLAPACK_HAS_OpenMP
                omp_set_num_threads(this->num_threads_some);
            #endif
                RandBLAS::DenseDist D(n, k);
                state = RandBLAS::fill_dense(D, Y_i, state);
            #if RandLAPACK_HAS_OpenMP
                omp_set_num_threads(this->num_threads_rest);
            #endif

                if(this -> timing) {
                    sketching_t_stop  = steady_clock::now();
                    sketching_t_dur   = duration_cast<microseconds>(sketching_t_stop - sketching_t_start).count();
                    gemm_A_t_start = steady_clock::now();
                }

                // [X_ev, ~] = qr(A * Y_i, 0)
                //blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);
                A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, m, Y_i, n, 0.0, X_i, m);

                if(this -> timing) {
                    gemm_A_t_stop = steady_clock::now();
                    gemm_A_t_dur  = duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                }

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

                // Advance odd iteration count.
                ++iter_od;
                // Advance iteration count.
                ++iter;

                // Iterate until in-loop termination criteria is met.
                while(1) {
                    if(this -> timing)
                        main_loop_t_start = steady_clock::now();

                    if (iter % 2 != 0) {
                        if(this -> timing)
                            gemm_A_t_start = steady_clock::now();
                        // Y_i = A' * X_i 
                        //blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, X_i, m, 0.0, Y_i, n);
                        A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, m, X_i, m, 0.0, Y_i, n);

                        if(this -> timing) {
                            gemm_A_t_stop = steady_clock::now();
                            gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                        }

                        // Allocate more spece for Y_od
                        curr_X_cols += k;
                        X_ev = ( T * ) realloc(X_ev, m * curr_X_cols * sizeof( T ));
                        // Move the X_i pointer;
                        X_i = &X_ev[m * (curr_X_cols - k)];

                        //X_i = &X_i[m * k];

                        if (iter != 1) {
                            // R_i' = Y_i' * Y_od
                            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, iter_ev * k, n, 1.0, Y_i, n, Y_od, n, 0.0, R_i, n);
                            
                            if(this -> timing)
                                reorth_t_start  = steady_clock::now();                
                            
                            // Y_i = Y_i - Y_od * R_i
                            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, k, iter_ev * k, -1.0, Y_od, n, R_i, n, 1.0, Y_i, n);

                            // Reorthogonalization
                            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, iter_ev * k, n, 1.0, Y_i, n, Y_od, n, 0.0, Y_orth_buf, k);
                            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, k, iter_ev * k, -1.0, Y_od, n, Y_orth_buf, k, 1.0, Y_i, n);

                            if(this -> timing) {
                                reorth_t_stop  = steady_clock::now();
                                reorth_t_dur   += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
                            }
                        }

                        // [Y_i, R_ii] = qr(Y_i, 0)
                        std::fill(&tau[0], &tau[k], 0.0);

                        if(this -> timing)
                            qr_t_start = steady_clock::now();
                        lapack::geqrf(n, k, Y_i, n, tau);

                        if(this -> timing) {
                            qr_t_stop = steady_clock::now();
                            qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                            r_cpy_t_start = steady_clock::now();
                        }

                        // Copy R_ii over to R's (in transposed format).
            #if RandLAPACK_HAS_OpenMP
                        omp_set_num_threads(this->num_threads_some);
            #endif
                        util::transposition(0, k, Y_i, n, R_ii, n, 1);
            #if RandLAPACK_HAS_OpenMP
                        omp_set_num_threads(this->num_threads_rest);
            #endif

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

                        // Early termination
                        // if (abs(R(end)) <= sqrt(eps('T')))
                        if(std::abs(R_ii[(n + 1) * (k - 1)]) < std::sqrt(std::numeric_limits<T>::epsilon())) {
                            //printf("TERMINATION 1 at iteration %ld\n", iter);
                            break;
                        }

                        // Allocate more space for R
                        R = ( T * ) realloc(R, n * curr_X_cols * sizeof( T ));
                        // Need to make sure the newly-allocated space is empty
                        memset(&R[n * (curr_X_cols - k)], 0.0, n * k * sizeof( T ));

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
                        //blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);
                        A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, m, Y_i, n, 0.0, X_i, m);

                        if(this -> timing) {
                            gemm_A_t_stop = steady_clock::now();
                            gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
                        }

                        // Allocate more spece for Y_od
                        curr_Y_cols += k;
                        Y_od = ( T * ) realloc(Y_od, n * curr_Y_cols * sizeof( T ));
                        // Move the X_i pointer;
                        Y_i = &Y_od[n * (curr_Y_cols - k)];

                        //Y_i = &Y_i[n * k];

                        // S_i = X_ev' * X_i 
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, S_i, n + k);
                        
                        if(this -> timing)
                            reorth_t_start  = steady_clock::now();
                        
                        //X_i = X_i - X_ev * S_i;
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, S_i, n + k, 1.0, X_i, m);

                        // Reorthogonalization
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, X_orth_buf, n + k);
                        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, X_orth_buf, n + k, 1.0, X_i, m);
                        
                        if(this -> timing) {
                            reorth_t_stop  = steady_clock::now();
                            reorth_t_dur   += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
                        }

                        // [X_i, S_ii] = qr(X_i, 0);
                        std::fill(&tau[0], &tau[k], 0.0);

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

                        // Early termination
                        // if (abs(S(end)) <= sqrt(eps('double')))
                        if(std::abs(S_ii[((n + k) + 1) * (k - 1)]) < std::sqrt(std::numeric_limits<T>::epsilon())) {
                            //printf("TERMINATION 2 at iteration %ld\n", iter);
                            break;
                        }

                        // Allocate more space for S
                        S = ( T * ) realloc(S, (n + k) * curr_Y_cols * sizeof( T ));
                        // Need to make sure the newly-allocated space is empty
                        memset(&S[(n + k)* (curr_Y_cols - k)], 0.0, (n + k) * k * sizeof( T ));

                        // Advance S pointers
                        S_i  = &S[(n + k) * k * iter_od];
                        S_ii = &S[(n + k) * k * iter_od + k + (iter_od * k)];

                        // Advance odd iteration count;
                        ++iter_od;

                    }

                    if(this -> timing)
                        norm_t_start = steady_clock::now();

                    // This is only changed on odd iters
                    if (iter % 2 != 0)
                        norm_R = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, iter_ev * k, iter_ev * k, R, n);

                    if(this -> timing) {
                        norm_t_stop       = steady_clock::now();
                        norm_t_dur        += duration_cast<microseconds>(norm_t_stop - norm_t_start).count();
                        main_loop_t_stop  = steady_clock::now();
                        main_loop_t_dur   += duration_cast<microseconds>(main_loop_t_stop - main_loop_t_start).count();
                    }

                    if (iter >= max_iters) {
                        break;
                    }

                    ++iter;
                    //norm(R, 'fro') > sqrt(1 - sq_tol) * norm_A
                    if(norm_R > threshold) {
                        //printf("Threshold termination\n");
                        break;
                    }
                }

                this->norm_R_end = norm_R;
                this->num_krylov_iters = iter;
                end_cols = num_krylov_iters * k / 2;
                iter % 2 == 0 ? end_rows = end_cols + k : end_rows = end_cols;
                

                if(this -> timing) {
                    allocation_t_start  = steady_clock::now();
                }

                U_hat  = ( T * ) calloc( end_rows * end_cols, sizeof( T ) );
                VT_hat = ( T * ) calloc( end_cols * end_cols, sizeof( T ) );

                if(this -> timing) {
                    allocation_t_stop  = steady_clock::now();
                    allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                    get_factors_t_start  = steady_clock::now();
                }

                if (iter % 2 != 0) {
                    // [U_hat, Sigma, V_hat] = svd(R')
                    lapack::gesdd(Job::SomeVec, end_rows, end_cols, R, n, Sigma, U_hat, end_rows, VT_hat, end_cols);
                } else { 
                    // [U_hat, Sigma, V_hat] = svd(S)
                    lapack::gesdd(Job::SomeVec, end_rows, end_cols, S, n + k, Sigma, U_hat, end_rows, VT_hat, end_cols);
                }

                // U = X_ev * U_hat
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, end_cols, end_rows, 1.0, X_ev, m, U_hat, end_rows, 0.0, U, m);
                // V = Y_od * V_hat
                // We actually perform VT = V_hat' * Y_odd'
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, end_cols, n, end_cols, 1.0, VT_hat, end_cols, Y_od, n, 0.0, VT, n);

                if(this -> timing) {
                    get_factors_t_stop  = steady_clock::now();
                    get_factors_t_dur   = duration_cast<microseconds>(get_factors_t_stop - get_factors_t_start).count();
                    allocation_t_start  = steady_clock::now();
                }
                free(Y_od);
                free(X_ev);
                free(tau);
                free(R);
                free(S);
                free(U_hat);
                free(VT_hat);
                free(Y_orth_buf);
                free(X_orth_buf);

                if(this -> timing) {
                    allocation_t_stop  = steady_clock::now();
                    allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                }

                if(this -> timing) {
                    total_t_stop = steady_clock::now();
                    total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                    long t_rest  = total_t_dur - (allocation_t_dur + get_factors_t_dur + ungqr_t_dur + reorth_t_dur + qr_t_dur + gemm_A_t_dur + sketching_t_dur + r_cpy_t_dur + s_cpy_t_dur + norm_t_dur);
                    this -> times.resize(13);
                    this -> times = {allocation_t_dur, get_factors_t_dur, ungqr_t_dur, reorth_t_dur, qr_t_dur, gemm_A_t_dur, main_loop_t_dur, sketching_t_dur, r_cpy_t_dur, s_cpy_t_dur, norm_t_dur, t_rest, total_t_dur};

                    if (this -> verbose) {
                        printf("\n\n/------------RBKI TIMING RESULTS BEGIN------------/\n");
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
                        printf("/-------------RBKI TIMING RESULTS END-------------/\n\n");
                    }
                }
                return 0;
            }
};
}