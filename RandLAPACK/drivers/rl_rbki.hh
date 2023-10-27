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
            T* V,
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
        }

        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            int64_t k,
            T* U,
            T* V,
            T* Sigma,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        bool verbosity;
        bool timing;
        T tol;
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
    T* V,
    T* Sigma,
    RandBLAS::RNGState<RNG> &state
){
    int64_t iter, iter_od, iter_ev = 0;
    T norm_R = 0;

    // Space for Y_i and Y_odd. (maybe needs to be n by m + k)
    T* Y = ( T * ) calloc( n * m, sizeof( T ) );
    // Space for X_i and X_ev. (maybe needs to be m by m + k)
    T* X = ( T * ) calloc( m * m, sizeof( T ) );
    // tau space for QR
    T* tau = ( T * ) calloc( k, sizeof( T ) );
    //
    T* R = ( T * ) calloc( n * n, sizeof( T ) );
    T* S = ( T * ) calloc( n * n, sizeof( T ) );

    // Pointers allocation
    // This will be offset by n * k at every even iteration.
    T* Y_i  = Y;
    // This stays the same throughout execution.
    T* Y_od = Y;
    T* R_i  = NULL;
    T* R_ii = R;

    T* X_i  = X; //&X_ev[m * k];
    T* X_ev = X;
    T* S_i  = S;
    T* S_ii = &S[k];

    // Pre-conpute Fro norm of an input matrix.
    T norm_A = lapack::lange(Norm::Fro, m, n, A, lda);
    T sq_tol = std::pow(this->tol, 2);

    // Generate a dense Gaussian random matrx.
    RandBLAS::DenseDist D(n, k);
    state = RandBLAS::fill_dense(D, Y_i, state).second;

    char name [] = "A input";
    RandBLAS::util::print_colmaj(m, n, A, name);

    char name1 [] = "Y sketching";
    RandBLAS::util::print_colmaj(n, k, Y_i, name1);
    char name2 [] = "Y_od";
    char name3 [] = "R";

    char name4 [] = "X_ev";
    char name5 [] = "S";

    char name6 [] = "Y_i";
    char name7 [] = "X_i";

    // [X_ev, ~] = qr(A * Y_i, 0)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);
    lapack::geqrf(m, k, X_i, m, tau);
    // Convert X_i into an explicit form. It is now stored in X_ev as it should be
    lapack::ungqr(m, k, k, X_i, m, tau);

    // Advance odd iteration count;
    ++iter_od;

    // Iterate until in-loop termination criteria is met.
    while(1) {
        if (iter % 2 == 0) {
            // Y_i = A' * X_i 
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, m, 1.0, A, m, X_i, m, 0.0, Y_i, n);

            if (iter == 2) {
                printf("PROBLEM CASE START\n");

                //RandBLAS::util::print_colmaj(m, n, A, name);
                RandBLAS::util::print_colmaj(m, k, X_i, name7);

                char name8 [] = "Y_i = A' * X_i";
                RandBLAS::util::print_colmaj(n, k, Y_i, name8);
            }


            // Move the X_i pointer;
            X_i = &X_i[m * k];

            if (iter != 0) {
                // R_i = Y_od' * Y_i
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_ev * k, k, n, 1.0, Y_od, n, Y_i, n, 0.0, R_i, n);
                
                if (iter == 2) {
                    char name9 [] = "R_i = Y_od' * Y_i";
                    RandBLAS::util::print_colmaj(n, n, R, name9);
                }
                
                // Y_i = Y_i - Y_od * R_i 
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, k, iter_ev * k, -1.0, Y_od, n, R_i, n, 1.0, Y_i, n);
            }


            RandBLAS::util::print_colmaj(n, k, Y_i, name6); 

            // [Y_i, R_ii] = qr(Y_i, 0)
            std::fill(&tau[0], &tau[k], 0.0);
            lapack::geqrf(n, k, Y_i, n, tau);

            // Early termination
            // if (abs(R(end)) <= sqrt(eps('double')))
            if(std::abs(Y_i[n * k - 1]) < std::sqrt(std::numeric_limits<double>::epsilon()))
                break;

            // Copy R_ii over to R's space under R_i (offset down by iter * k)
            lapack::lacpy(MatrixType::Upper, k, k, Y_i, n, R_ii, n);
            // Convert Y_i into an explicit form. It is now stored in Y_odd as it should be
            lapack::ungqr(n, k, k, Y_i, n, tau);
 
            //RandBLAS::util::print_colmaj(n, m, Y_od, name2);     
            RandBLAS::util::print_colmaj(n, n, R, name3);

            // Advance R pointers
            iter == 0 ? R_i = &R_ii[n * k] : R_i = &R_i[n * k];
            R_ii = &R_ii[(n + 1) * k];

            // Advance even iteration count;
            ++iter_ev;

            if(iter == 4)
                return 0;
        }
        else {
            // X_i = A * Y_i
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Y_i, n, 0.0, X_i, m);

            // Move the X_i pointer;
            Y_i = &Y_i[n * k];

            // S_i = X_ev' * X_i 
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter_od * k, k, m, 1.0, X_ev, m, X_i, m, 0.0, S_i, n);
  
            //X_i = X_i - X_ev * S_i;
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, iter_od * k, -1.0, X_ev, m, S_i, n, 1.0, X_i, m);
            
            RandBLAS::util::print_colmaj(m, k, X_i, name7); 
            
            // [X_i, S_ii] = qr(X_i, 0);
            std::fill(&tau[0], &tau[k], 0.0);
            lapack::geqrf(m, k, X_i, m, tau);

            // Early termination
            // if (abs(S(end)) <= sqrt(eps('double')))
            if(std::abs(X_i[m * k - 1]) < std::sqrt(std::numeric_limits<double>::epsilon()))
                break;

            // Copy S_ii over to S's space under S_i (offset down by iter * k)
            lapack::lacpy(MatrixType::Upper, k, k, X_i, m, S_ii, n);
            // Convert X_i into an explicit form. It is now stored in X_ev as it should be
            lapack::ungqr(m, k, k, X_i, m, tau);

            RandBLAS::util::print_colmaj(m, m, X_ev, name4);     
            RandBLAS::util::print_colmaj(n, n, S, name5);

            // Advance R pointers
            S_i = &S_i[n * k];
            S_ii = &S_ii[(n + 1) * k];

            // Advance odd iteration count;
            ++iter_od;
        }
        ++iter;
        norm_R = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, n, n, R, n);
        printf("norm_R: %e\n", norm_R);
    }
    /*
    if (iter % 2 == 0) {
        lapack::gesdd(Job::SomeVec, n, iter * k, R, n, Sigma, U_hat, n, VT_hat, n);

    } else {
        lapack::gesdd(Job::SomeVec, iter * k, iter * k, S, n, Sigma, U_hat, n, VT_hat, n);
        //blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter * k, k, m, 1.0, X_ev, m, U_hat, m, 0.0, &S[n * iter * k], n);
        //blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, iter * k, k, m, 1.0, Y_od, m, V_hat, m, 0.0, &S[n * iter * k], n);
    }
    */
    return 0;
}
} // end namespace RandLAPACK
#endif