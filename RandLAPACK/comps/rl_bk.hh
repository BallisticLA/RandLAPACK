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
        ///     Upper band matrix (stored transposed), allocated internally with calloc.
        ///     Caller must free().
        ///
        /// @param[out] S
        ///     Lower Hessenberg band matrix, allocated internally with calloc.
        ///     Caller must free().
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
            linops::GenLinOp<T> A_linop(m, n, A, lda, Layout::ColMajor);
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
            linops::SpLinOp<SpMat> A_linop(m, n, A);
            return this->call(A_linop, k, X_ev, Y_od, R, S, end_rows, end_cols, final_iter_is_odd, state);
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

                if(this -> timing) {
                    bk_total_t_start = steady_clock::now();
                    allocation_t_start  = steady_clock::now();
                }

                int64_t m = A.n_rows;
                int64_t n = A.n_cols;
                int64_t iter = 0, iter_od = 0, iter_ev = 0;
                end_rows = 0;
                end_cols = 0;
                T norm_R = 0;
                int max_iters = this->max_krylov_iters;

                // We need a full copy of X and Y all the way through the algorithm
                // due to an operation with X_odd and Y_odd happening at the end.
                // Below pointers stay the same throughout the alg; the space will be alloacted iteratively
                // Space for Y_i and Y_odd.
                Y_od  = ( T * ) calloc( n * k, sizeof( T ) );
                int64_t curr_Y_cols = k;
                // Space for X_i and X_ev.
                X_ev  = ( T * ) calloc( m * k, sizeof( T ) );
                int64_t curr_X_cols = k;

                // While R and S matrices are structured (both band), we cannot make use of this structure through
                // BLAS-level functions.
                // Note also that we store a transposed version of R.
                //
                // At each iterations, matrices R and S grow by b_sz.
                // At the end, size of R would by d x d and size of S would
                // be (d + 1) x d, where d = numiters_complete * b_sz, d <= n.
                // Note that the total amount of iterations will always be numiters <= n * 2 / block_size
                R   = ( T * ) calloc( n * k, sizeof( T ) );
                S   = ( T * ) calloc( (n + k) * k, sizeof( T ) );

                // These buffers are pure GEMM outputs (beta=0.0), no need to zero-initialize.
                T* Y_orth_buf = ( T * ) malloc( k * n * sizeof( T ) );
                T* X_orth_buf = ( T * ) malloc( k * (n + k) * sizeof( T ) );

                // Pointers allocation
                // Below pointers will be offset by (n or m) * k at every even iteration.
                T* Y_i  = Y_od;
                T* X_i  = X_ev;
                // S and S pointers are offset at every step.
                T* R_i  = NULL;
                T* R_ii = R;
                T* S_i  = S;
                T* S_ii = &S[k];
                // tau space for QR (geqrf fully overwrites it)
                T* tau = ( T * ) malloc( k * sizeof( T ) );
                // Declared here (before cleanup lambda) so cleanup can free it.
                // Conditionally allocated below only when CQRRT is used.
                T* R_11_trans = nullptr;

                // Cleanup lambda for realloc failure — frees all buffers and nulls output pointers.
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

                if(this -> timing) {
                    allocation_t_stop  = steady_clock::now();
                    allocation_t_dur   = duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
                }

                // Pre-compute Fro norm of an input matrix.
                T norm_A = A.fro_nrm();
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

                if(this -> timing)
                    sketching_t_start  = steady_clock::now();

                // Generate a dense Gaussian random matrix.
                // We are using the plain dense operator instead of DenseSkOp here since
                // the space in which the dense operator is stored will be reused later, and
                // also needs to be used together with the input's abstract linear operator form.
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

                    CQRRT -> call(m, k, X_i, m, R_11_trans, k, d_factor, state);

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

                // Iterate until in-loop termination criteria is met.
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

                        // Allocate more space for X_ev
                        curr_X_cols += k;
                        X_ev = ( T * ) realloc(X_ev, m * curr_X_cols * sizeof( T ));
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

                            CQRRT -> call(n, k, Y_i, n, R_11_trans, k, d_factor, state);
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

                        // Early termination
                        // if (abs(R(end)) <= sqrt(eps('T')))
                        if(std::abs(R_ii[(n + 1) * (k - 1)]) < std::sqrt(std::numeric_limits<T>::epsilon())) {
                            break;
                        }

                        // Allocate more space for R
                        T* R_new = ( T * ) realloc(R, n * curr_X_cols * sizeof( T ));
                        if (!R_new) return cleanup_and_fail();
                        // Need to make sure the newly-allocated space is empty
                        R = R_new;
                        T* temp_r = &R[n * (curr_X_cols - k)];
                        std::fill(temp_r, temp_r + n*k, 0.0);

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

                        // Allocate more space for Y_od
                        curr_Y_cols += k;
                        Y_od = ( T * ) realloc(Y_od, n * curr_Y_cols * sizeof( T ));
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

                            CQRRT -> call(m, k, X_i, m, S_ii, n + k, d_factor, state);

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
                        if(std::abs(S_ii[((n + k) + 1) * (k - 1)]) < std::sqrt(std::numeric_limits<T>::epsilon())) {
                            break;
                        }

                        if(this -> timing) {
                            allocation_t_start  = steady_clock::now();
                        }

                        // Allocate more space for S
                        T* S_new = ( T * ) realloc(S, (n + k) * curr_Y_cols * sizeof( T ));
                        if (!S_new) return cleanup_and_fail();
                        // Need to make sure the newly-allocated space is empty
                        S = S_new;
                        T* temp_s = &S[(n + k)* (curr_Y_cols - k)];
                        std::fill(temp_s, temp_s + (n + k) * k, 0.0);

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
                        // Threshold termination.
                        break;
                    }
                }

                // Set output state
                this->norm_R_end = norm_R;
                this->num_krylov_iters = iter;
                end_cols = iter * k / 2;
                iter % 2 == 0 ? end_rows = end_cols + k : end_rows = end_cols;
                final_iter_is_odd = (iter % 2 != 0);

                // Free internal temporaries (NOT X_ev, Y_od, R, S — those are returned to caller)
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
