#ifndef randlapack_rbki_h
#define randlapack_rbki_h

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>
#include <climits>

using namespace std::chrono;

namespace RandLAPACK {

template <typename T, typename RNG>
class RBKIalg {
    public:
        virtual ~RBKIalg() {}
        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            int64_t k,
            T* U,
            T* VT,
            T* Sigma,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class RBKI : public RBKIalg<T, RNG> {
    public:
        RBKI(
            bool verb,
            bool time_subroutines,
            T ep
        ) {
            verbosity = verb;
            timing = time_subroutines;
            tol = ep;
            max_krylov_iters = INT_MAX;
        }
        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            int64_t k,
            T* U,
            T* VT,
            T* Sigma,
            RandBLAS::RNGState<RNG> &state
        ) override;
    public:
        bool verbosity;
        bool timing;
        T tol;
        int num_krylov_iters;
        int max_krylov_iters;
        std::vector<long> times;
        T norm_R_end;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int RBKI<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    int64_t k,
    T* U,
    T* VT,
    T* Sigma,
    RandBLAS::RNGState<RNG> &state
){
    high_resolution_clock::time_point allocation_t_start;
    high_resolution_clock::time_point allocation_t_stop;
    high_resolution_clock::time_point get_factors_t_start;
    high_resolution_clock::time_point get_factors_t_stop;
    high_resolution_clock::time_point ungqr_t_start;
    high_resolution_clock::time_point ungqr_t_stop;
    high_resolution_clock::time_point reorth_t_start;
    high_resolution_clock::time_point reorth_t_stop;
    high_resolution_clock::time_point qr_t_start;
    high_resolution_clock::time_point qr_t_stop;
    high_resolution_clock::time_point gemm_A_t_start;
    high_resolution_clock::time_point gemm_A_t_stop;
    high_resolution_clock::time_point main_loop_t_start;
    high_resolution_clock::time_point main_loop_t_stop;
    high_resolution_clock::time_point sketching_t_start;
    high_resolution_clock::time_point sketching_t_stop;
    high_resolution_clock::time_point r_cpy_t_start;
    high_resolution_clock::time_point r_cpy_t_stop;
    high_resolution_clock::time_point s_cpy_t_start;
    high_resolution_clock::time_point s_cpy_t_stop;
    high_resolution_clock::time_point norm_t_start;
    high_resolution_clock::time_point norm_t_stop;
    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;

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
        total_t_start = high_resolution_clock::now();
        allocation_t_start  = high_resolution_clock::now();
    }

    int64_t iter = 0, iter_od = 0, iter_ev = 0, i = 0, end_rows = 0, end_cols = 0;
    T norm_R = 0;
    int64_t space_rows = k * std::ceil(m / (T) k);
    int max_iters = std::min(this->max_krylov_iters, (int) (n / (T) k));

    // We need a full copy of X and Y all the way through the algorithm
    // due to an operation with X_odd and Y_odd happening at the end.
    // Space for Y_i and Y_odd.
    T* Y   = ( T * ) calloc( n * m,       sizeof( T ) );
    // Space for X_i and X_ev. (maybe needs to be m by m + k)
    T* X   = ( T * ) calloc( m * (m + k), sizeof( T ) );
    // tau space for QR
    T* tau = ( T * ) calloc( k,           sizeof( T ) );
    // While R and S matrices are structured (both band), we cannot make use of this structure through
    // BLAS-level functions.
    // Note also that we store a transposed version of R.
    T* R   = ( T * ) calloc( n * n,       sizeof( T ) );
    T* S   = ( T * ) calloc( (n + k) * n, sizeof( T ) );

    T* Y_orth_buf = ( T * ) calloc( k * n, sizeof( T ) );
    T* X_orth_buf = ( T * ) calloc( k * (n + k), sizeof( T ) );

    // Pointers allocation
    // Below pointers will be offset by (n or m) * k at every even iteration.
    T* Y_i  = Y;
    T* X_i  = X;
    // Below pointers stay the same throughout the alg.
    T* Y_od = Y;
    T* X_ev = X;
    // S and S pointers are offset at every step.
    T* R_i  = NULL;
    T* R_ii = R;
    T* S_i  = S;
    T* S_ii = &S[k];
    // Pre-decloration of SVD-related buffers.
    T* U_hat = NULL;
    T* VT_hat = NULL;

    if(this -> timing) {
        allocation_t_stop  = high_resolution_clock::now();
        allocation_t_dur   = duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
    }

    // Pre-conpute Fro norm of an input matrix.
    T norm_A = lapack::lange(Norm::Fro, m, n, A, lda);
    T sq_tol = std::pow(this->tol, 2);
    T threshold =  std::sqrt(1 - sq_tol) * norm_A;

    if(this -> timing)
        sketching_t_start  = high_resolution_clock::now();

    // Generate a dense Gaussian random matrx.
    RandBLAS::DenseDist D(n, k);
    state = RandBLAS::fill_dense(D, Y_i, state).second;

    if(this -> timing) {
        sketching_t_stop  = high_resolution_clock::now();
        sketching_t_dur   = duration_cast<microseconds>(sketching_t_stop - sketching_t_start).count();
        gemm_A_t_start = high_resolution_clock::now();
    }

    // [X_ev, ~] = qr(A * Y_i, 0)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);

    if(this -> timing) {
        gemm_A_t_stop = high_resolution_clock::now();
        gemm_A_t_dur  = duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
    }

    if(this -> timing)
        qr_t_start = high_resolution_clock::now();

    lapack::geqrf(m, k, X_i, m, tau);

    if(this -> timing) {
        qr_t_stop = high_resolution_clock::now();
        qr_t_dur  = duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
        ungqr_t_start  = high_resolution_clock::now();
    }

    // Convert X_i into an explicit form. It is now stored in X_ev as it should be.
    lapack::ungqr(m, k, k, X_i, m, tau);

    if(this -> timing) {
        ungqr_t_stop  = high_resolution_clock::now();
        ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
    }

    // Advance odd iteration count.
    ++iter_od;
    // Advance iteration count.
    ++iter;

    // Iterate until in-loop termination criteria is met.

    while((iter_ev + iter_od) < max_iters) {
        if(this -> timing)
            main_loop_t_start = high_resolution_clock::now();

        if (iter % 2 != 0) {
            
            if(this -> timing)
                gemm_A_t_start = high_resolution_clock::now();
            
            // Y_i = A' * X_i 
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, X_i, m, 0.0, Y_i, n);

            if(this -> timing) {
                gemm_A_t_stop = high_resolution_clock::now();
                gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
            }

            // Move the X_i pointer;
            X_i = &X_i[m * k];

            if (iter != 1) {
                // R_i' = Y_i' * Y_od
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, iter_ev * k, n, 1.0, Y_i, n, Y_od, n, 0.0, R_i, n);
                
                if(this -> timing)
                    reorth_t_start  = high_resolution_clock::now();                
                
                // Y_i = Y_i - Y_od * R_i
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, k, iter_ev * k, -1.0, Y_od, n, R_i, n, 1.0, Y_i, n);

                // Reorthogonalization
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, iter_ev * k, n, 1.0, Y_i, n, Y_od, n, 0.0, Y_orth_buf, k);
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, k, iter_ev * k, -1.0, Y_od, n, Y_orth_buf, k, 1.0, Y_i, n);

                if(this -> timing) {
                    reorth_t_stop  = high_resolution_clock::now();
                    reorth_t_dur   += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
                }
            }

            // [Y_i, R_ii] = qr(Y_i, 0)
            std::fill(&tau[0], &tau[k], 0.0);

            if(this -> timing)
                qr_t_start = high_resolution_clock::now();

            lapack::geqrf(n, k, Y_i, n, tau);

            if(this -> timing) {
                qr_t_stop = high_resolution_clock::now();
                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                r_cpy_t_start = high_resolution_clock::now();
            }

            // Copy R_ii over to R's (in transposed format).
            for(i = 0; i < k; ++i)
                blas::copy(i + 1, &Y_i[i * n], 1, &R_ii[i], n);

            if(this -> timing) {
                r_cpy_t_stop  = high_resolution_clock::now();
                r_cpy_t_dur  += duration_cast<microseconds>(r_cpy_t_stop - r_cpy_t_start).count();
                ungqr_t_start = high_resolution_clock::now();
            }

            // Convert Y_i into an explicit form. It is now stored in Y_odd as it should be.
            lapack::ungqr(n, k, k, Y_i, n, tau);

            if(this -> timing) {
                ungqr_t_stop  = high_resolution_clock::now();
                ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
            }

            // Early termination
            // if (abs(R(end)) <= sqrt(eps('double')))
            if(std::abs(R_ii[(n + 1) * (k - 1)]) < std::sqrt(std::numeric_limits<double>::epsilon())) {
                //printf("TERMINATION 1 at iteration %ld\n", iter_ev);
                break;
            }

            // Advance R pointers
            iter == 1 ? R_i = &R_ii[k] : R_i = &R_i[k];
            R_ii = &R_ii[(n + 1) * k];

            // Advance even iteration count;
            ++iter_ev;
        }
        else {

            if(this -> timing)
                gemm_A_t_start = high_resolution_clock::now();

            // X_i = A * Y_i
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);

            if(this -> timing) {
                gemm_A_t_stop = high_resolution_clock::now();
                gemm_A_t_dur  += duration_cast<microseconds>(gemm_A_t_stop - gemm_A_t_start).count();
            }

            // Move the X_i pointer;
            Y_i = &Y_i[n * k];
            
            // S_i = X_ev' * X_i 
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, S_i, n + k);
            
            if(this -> timing)
                reorth_t_start  = high_resolution_clock::now();
            
            //X_i = X_i - X_ev * S_i;
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, S_i, n + k, 1.0, X_i, m);

            // Reorthogonalization
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, X_orth_buf, n + k);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, X_orth_buf, n + k, 1.0, X_i, m);
            
            if(this -> timing) {
                reorth_t_stop  = high_resolution_clock::now();
                reorth_t_dur   += duration_cast<microseconds>(reorth_t_stop - reorth_t_start).count();
            }

            // [X_i, S_ii] = qr(X_i, 0);
            std::fill(&tau[0], &tau[k], 0.0);

            if(this -> timing)
                qr_t_start = high_resolution_clock::now();

            lapack::geqrf(m, k, X_i, m, tau);

            if(this -> timing) {
                qr_t_stop = high_resolution_clock::now();
                qr_t_dur  += duration_cast<microseconds>(qr_t_stop - qr_t_start).count();
                s_cpy_t_start = high_resolution_clock::now();
            }

            // Copy S_ii over to S's space under S_i (offset down by iter_od * k)
            lapack::lacpy(MatrixType::Upper, k, k, X_i, m, S_ii, n + k);

            if(this -> timing) {
                s_cpy_t_stop  = high_resolution_clock::now();
                s_cpy_t_dur  += duration_cast<microseconds>(s_cpy_t_stop - s_cpy_t_start).count();
                ungqr_t_start = high_resolution_clock::now();
            }

            // Convert X_i into an explicit form. It is now stored in X_ev as it should be
            lapack::ungqr(m, k, k, X_i, m, tau);

            if(this -> timing) {
                ungqr_t_stop  = high_resolution_clock::now();
                ungqr_t_dur   += duration_cast<microseconds>(ungqr_t_stop - ungqr_t_start).count();
            }

            // Early termination
            // if (abs(S(end)) <= sqrt(eps('double')))
            if(std::abs(S_ii[((n + k) + 1) * (k - 1)]) < std::sqrt(std::numeric_limits<double>::epsilon())) {
                //printf("TERMINATION 2 at iteration %ld\n", iter_od);
                break;
            }

            // Advance R pointers
            S_i = &S_i[(n + k) * k];
            S_ii = &S_ii[((n + k)  + 1) * k];
            // Advance odd iteration count;
            ++iter_od;
        }

        if(this -> timing)
            norm_t_start = high_resolution_clock::now();

        // This is only changed on odd iters
        if (iter % 2 != 0)
            norm_R = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, iter_ev * k, iter_ev * k, R, n);

        if(this -> timing) {
            norm_t_stop       = high_resolution_clock::now();
            norm_t_dur        += duration_cast<microseconds>(norm_t_stop - norm_t_start).count();
            main_loop_t_stop  = high_resolution_clock::now();
            main_loop_t_dur   += duration_cast<microseconds>(main_loop_t_stop - main_loop_t_start).count();
        }

        ++iter;
        //norm(R, 'fro') > sqrt(1 - sq_tol) * norm_A
        if(norm_R > threshold) {
            break;
        }
    }

    this -> norm_R_end = norm_R;
    this->num_krylov_iters = iter;
    iter % 2 == 0 ? end_rows = k * (iter_ev + 1), end_cols = k * iter_ev : end_rows = k * (iter_od + 1), end_cols = k * iter_od;

    if(this -> timing) {
        allocation_t_start  = high_resolution_clock::now();
    }

    U_hat  = ( T * ) calloc( end_rows * end_cols, sizeof( T ) );
    VT_hat = ( T * ) calloc( end_cols * end_cols, sizeof( T ) );

    if(this -> timing) {
        allocation_t_stop  = high_resolution_clock::now();
        allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
        get_factors_t_start  = high_resolution_clock::now();
    }

    if (iter % 2 == 0) {
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
        get_factors_t_stop  = high_resolution_clock::now();
        get_factors_t_dur   = duration_cast<microseconds>(get_factors_t_stop - get_factors_t_start).count();
        allocation_t_start  = high_resolution_clock::now();
    }

    free(Y);
    free(X);
    free(tau);
    free(R);
    free(S);
    free(U_hat);
    free(VT_hat);
    free(Y_orth_buf);
    free(X_orth_buf);

    if(this -> timing) {
        allocation_t_stop  = high_resolution_clock::now();
        allocation_t_dur   += duration_cast<microseconds>(allocation_t_stop - allocation_t_start).count();
    }

        if(this -> timing) {
            total_t_stop = high_resolution_clock::now();
            total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
            long t_rest  = total_t_dur - (allocation_t_dur + get_factors_t_dur + ungqr_t_dur + reorth_t_dur + qr_t_dur + gemm_A_t_dur + sketching_t_dur + r_cpy_t_dur + s_cpy_t_dur + norm_t_dur);
            this -> times.resize(11);
            this -> times = {allocation_t_dur, get_factors_t_dur, ungqr_t_dur, reorth_t_dur, qr_t_dur, gemm_A_t_dur, main_loop_t_dur, sketching_t_dur, r_cpy_t_dur, s_cpy_t_dur, norm_t_dur, t_rest, total_t_dur};

            if (this -> verbosity) {
                printf("\n\n/------------RBKI TIMING RESULTS BEGIN------------/\n");
                printf("Basic info: b_sz=%ld krylov_iters=%ld\n",      k, num_krylov_iters);

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
} // end namespace RandLAPACK
#endif