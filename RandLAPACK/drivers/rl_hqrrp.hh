/*
===============================================================================
Authors
===============================================================================
Per-Gunnar Martinsson
  Dept. of Applied Mathematics, 
  University of Colorado at Boulder, 
  526 UCB, Boulder, CO 80309-0526, USA
Gregorio Quint64_tana-Ortí
  Depto. de Ingenieria y Ciencia de Computadores, 
  Universitat Jaume I, 
  12.071 Castellon, Spain
Nathan Heavner
  Dept. of Applied Mathematics, 
  University of Colorado at Boulder, 
  526 UCB, Boulder, CO 80309-0526, USA
Robert van de Geijn
  Dept. of Computer Science and Institute for Computational Engineering and 
  Sciences, 
  The University of Texas at Austin
  Austin, TX.
===============================================================================
Copyright
===============================================================================
Copyright (C) 2016, 
  Universitat Jaume I,
  University of Colorado at Boulder,
  The University of Texas at Austin.
/home/riley/Documents/Research/software/hqrrp/lapack_compatible_sources/simple_test.cpp:87:47
===============================================================================
Disclaimer
===============================================================================
This code is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY EXPRESSED OR IMPLIED.
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <typeinfo>

#ifndef randlapack_hqrrp_h
#define randlapack_hqrrp_h

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <lapack/fortran.h>
#include <lapack/config.h>
#include <chrono>

// Matrices with dimensions larger than THRESHOLD_FOR_DGEQP3 are processed 
// with the new HQRRP code.
#define THRESHOLD_FOR_DGEQP3  2

// ============================================================================
// Definition of macros.
#define dabs( a )    ( (a) >= 0.0 ? (a) : -(a) )

// ============================================================================
// Compilation declarations.

#undef CHECK_DOWNDATING_OF_Y

using namespace std::chrono;

namespace RandLAPACK {

// ============================================================================
// Functions
template <typename T>
void _LAPACK_lafrb(
    lapack::Side side,
    lapack::Op op,
    lapack::Direction dir,
    lapack::StoreV  storev,
    int64_t m, int64_t n, int64_t k,
    T *buff_U, int64_t ldu,
    T *buff_T, int64_t ldt,
    T *buff_B, int64_t ldb,
    T *buff_W, int64_t ldw
){
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int k_ = (lapack_int) k;
    lapack_int ldim_U = (lapack_int) ldu;
    lapack_int ldim_T = (lapack_int) ldt;
    lapack_int ldim_B = (lapack_int) ldb;
    lapack_int ldim_W = (lapack_int) ldw;
    char side_ = blas::side2char( side );
    char trans_ = blas::op2char( op );
    char direction_ = lapack::direction2char( dir );
    char storev_ = lapack::storev2char( storev );
    if (typeid(T) == typeid(double)) {
        LAPACK_dlarfb( & side_, & trans_, & direction_, & storev_,  
                    & m_, & n_, & k_, (double *) buff_U, & ldim_U, (double *) buff_T, & ldim_T, 
                    (double *) buff_B, & ldim_B, (double *) buff_W, & ldim_W
                    #ifdef LAPACK_FORTRAN_STRLEN_END
                    , 1, 1, 1, 1
                    #endif
                    );
    } else if (typeid(T) == typeid(float)) {
        LAPACK_slarfb( & side_, & trans_, & direction_, & storev_,  
                    & m_, & n_, & k_, (float *) buff_U, & ldim_U, (float *) buff_T, & ldim_T, 
                    (float *) buff_B, & ldim_B, (float *) buff_W, & ldim_W
                    #ifdef LAPACK_FORTRAN_STRLEN_END
                    , 1, 1, 1, 1
                    #endif
                    );
    } else {
        // Unsupported type
    }
    return;
}

// ============================================================================
template <typename T>
void _LAPACK_larf(
    lapack::Side side,
    int64_t m, int64_t n,
    T *v, int64_t inc_v, T* tau,
    T *C, int64_t ldc,
    T *work
){
    char side_ = blas::side2char( side );
    lapack_int inc_v_ = (lapack_int) inc_v;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldc_ = (lapack_int) ldc;
    if (typeid(T) == typeid(double)) {
        LAPACK_dlarf( & side_, & m_, & n_, 
            (double *) v, & inc_v_,
            (double *) tau,
            (double *) C, & ldc_,
            (double *) work
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
            );
    } else if (typeid(T) == typeid(float)) {
        LAPACK_slarf( & side_, & m_, & n_, 
            (float *) v, & inc_v_,
            (float *) tau,
            (float *) C, & ldc_,
            (float *) work
            #ifdef LAPACK_FORTRAN_STRLEN_END
            , 1
            #endif
            );
    } else {
        // Unsupported type
    }
    return;
}
// ============================================================================
template <typename T>
void _LAPACK_geqrf(
    int64_t m, int64_t n, T *A, int64_t lda,
    T *tau, T *work,
    int64_t * lwork, int64_t * info) {
    lapack_int m_      = (lapack_int) m;
    lapack_int n_      = (lapack_int) n;
    lapack_int lda_    = (lapack_int) lda;
    lapack_int *lwork_ = (lapack_int *) lwork;
    lapack_int *info_  = (lapack_int *) info;

    if (typeid(T) == typeid(double)) {
        LAPACK_dgeqrf(&m_, &n_, (double *) A, &lda_, (double *) tau, (double *) work, lwork_, info_);
    } else if (typeid(T) == typeid(float)) {
        LAPACK_sgeqrf(&m_, &n_, (float *) A, &lda_, (float *) tau, (float *) work, lwork_, info_);
    } else {
        // Unsupported type
    }
  
    *info = (int64_t) *info_;
    *lwork = (int64_t) *lwork_;
    return;
}

// ============================================================================
template <typename T>
int64_t NoFLA_Apply_Q_WY_rnfc_blk_var4( 
    int64_t n_U, T * buff_U, int64_t ldim_U,
    T * buff_T, int64_t ldim_T, int64_t m_B, 
    int64_t n_B, T * buff_B, int64_t ldim_B ) {
//
// It applies a block transformation Q to a matrix B from the right:
//   B = B * Q
// where:
//   Q = I - U * T' * U'.
//
    T  * buff_W;
    int64_t   ldim_W;

    // Create auxiliary object.
    //// FLA_Obj_create_conf_to( FLA_TRANSPOSE, B1, & W );
    buff_W = ( T * ) malloc( m_B * n_U * sizeof( T ) );
    ldim_W = std::max<int64_t>( 1, m_B );

    // Apply the block transformation. 
    _LAPACK_lafrb(lapack::Side::Right, lapack::Op::NoTrans,
        lapack::Direction::Forward, lapack::StoreV::Columnwise,
        m_B, n_B, n_U, buff_U, ldim_U, buff_T, ldim_T,
        buff_B, ldim_B, buff_W, ldim_W
    );

    // Remove auxiliary object.
    free( buff_W );

    return 0;
}


// ============================================================================
template <typename T>
int64_t NoFLA_Downdate_Y( 
    int64_t n_U11, T * buff_U11, int64_t ldim_U11,
    int64_t m_U21, T * buff_U21, int64_t ldim_U21,
    int64_t m_A12, T * buff_A12, int64_t ldim_A12,
    T * buff_T, int64_t ldim_T,
    int64_t m_Y2, int64_t n_Y2, T * buff_Y2, int64_t ldim_Y2,
    int64_t m_G1, int64_t n_G1, T * buff_G1, int64_t ldim_G1,
    int64_t n_G2, T * buff_G2, int64_t ldim_G2 ) {
//
// It downdates matrix Y, and updates matrix G.
// Only Y2 of Y is updated.
// Only G1 and G2 of G are updated.
//
// Y2 = Y2 - ( G1 - ( G1*U11 + G2*U21 ) * T11 * U11' ) * R12.
//
    int64_t    i, j;
    T * buff_B;
    T d_one       = 1.0;
    T d_minus_one = -1.0;
    int64_t    m_B         = m_G1;
    int64_t    n_B         = n_G1;
    int64_t    ldim_B      = m_G1;

    // Create object B.
    buff_B = ( T * ) malloc( m_B * n_B * sizeof( T ) );

    // B = G1.
    lapack::lacpy( MatrixType::General,
                    m_G1, n_G1,
                    buff_G1, ldim_G1,
                    buff_B, ldim_B )  ;

    // B = B * U11.
    blas::trmm( Layout::ColMajor,
                Side::Right, 
                Uplo::Lower,
                Op::NoTrans,
                Diag::Unit, m_B, n_B,
                d_one, buff_U11, ldim_U11, buff_B, ldim_B );

    // B = B + G2 * U21.
    blas::gemm( Layout::ColMajor,
                Op::NoTrans, Op::NoTrans, m_B, n_B, m_U21,
                d_one, buff_G2, ldim_G2, buff_U21, ldim_U21,
                d_one, buff_B,  ldim_B );

    // B = B * T11.
    blas::trmm( Layout::ColMajor,
                Side::Right,
                Uplo::Upper,
                Op::NoTrans,
                Diag::NonUnit, m_B, n_B,
                d_one, buff_T, ldim_T, buff_B, ldim_B );

    // B = - B * U11^H.
    blas::trmm( Layout::ColMajor,
                Side::Right,
                Uplo::Lower,
                Op::ConjTrans,
                Diag::Unit, m_B, n_B,
                d_minus_one, buff_U11, ldim_U11, buff_B, ldim_B );

    // B = G1 + B.
    //#pragma omp parallel for
    for( j = 0; j < n_B; j++ ) {
        for( i = 0; i < m_B; i++ ) {
            buff_B[ i + j * ldim_B ] += buff_G1[ i + j * ldim_G1 ];
        }
    }

    // Y2 = Y2 - B * R12.
    blas::gemm( Layout::ColMajor,
                Op::NoTrans,
                Op::NoTrans, m_Y2, n_Y2, m_A12,
                d_minus_one, buff_B, ldim_B, buff_A12, ldim_A12,
                d_one, buff_Y2, ldim_Y2 );

    //
    // GR = GR * Q
    //
    NoFLA_Apply_Q_WY_rnfc_blk_var4( 
        n_U11, buff_U11, ldim_U11,
        buff_T, ldim_T, m_G1, 
        n_G1 + n_G2, buff_G1, ldim_G1 );

    // Remove object B.
    free( buff_B );

    return 0;
}

// ============================================================================
template <typename T>
int64_t NoFLA_Apply_Q_WY_lhfc_blk_var4( 
    int64_t n_U, T * buff_U, int64_t ldim_U,
    T * buff_T, int64_t ldim_T, int64_t m_B, 
    int64_t n_B, T * buff_B, int64_t ldim_B ) {
//
// It applies the transpose of a block transformation Q to a matrix B from 
// the left:
//   B := Q' * B
// where:
//   Q = I - U * T' * U'.
//
    T  * buff_W;
    int64_t     ldim_W;

    // Create auxiliary object.
    //// FLA_Obj_create_conf_to( FLA_NO_TRANSPOSE, B1, & W );
    buff_W = ( T * ) malloc( n_B * n_U * sizeof( T ) );
    ldim_W = std::max<int64_t>( 1, n_B );

    // Apply the block transformation.
    _LAPACK_lafrb(lapack::Side::Left, lapack::Op::Trans,
    lapack::Direction::Forward, lapack::StoreV::Columnwise,
    m_B, n_B, n_U,
    buff_U, ldim_U,
    buff_T, ldim_T,
    buff_B, ldim_B,
    buff_W, ldim_W
    );

    // Remove auxiliary object.
    free( buff_W );

    return 0;
}

// ============================================================================
template <typename T>
int64_t NoFLA_QRP_compute_norms(
    int64_t m_A, int64_t n_A, T * buff_A, int64_t ldim_A,
    T * buff_d, T * buff_e ) {
//
// It computes the column norms of matrix A. The norms are stored int64_to 
// vectors d and e.
//
    int64_t     j, i_one = 1;

    // Main loop.
    //#pragma omp parallel for
    for( j = 0; j < n_A; j++ ) {
        * buff_d = blas::nrm2(m_A, buff_A, i_one);
        * buff_e = * buff_d;
        buff_A += ldim_A;
        buff_d++;
        buff_e++;
    }

    return 0;
}

// ============================================================================
template <typename T>
static int64_t NoFLA_QRP_downdate_partial_norms( 
    int64_t m_A, int64_t n_A,
    T * buff_d,  int64_t st_d,
    T * buff_e,  int64_t st_e,
    T * buff_wt, int64_t st_wt,
    T * buff_A,  int64_t ldim_A ) {
//
// It updates (downdates) the column norms of matrix A. It uses Drmac's method.
//
    int64_t     j, i_one = 1;
    T  * ptr_d, * ptr_e, * ptr_wt, * ptr_A;
    T  temp, temp2, temp5, tol3z;
    // T dnrm2_(), dlamch_();

    // Some initializations.
    char dlmach_param = 'E';
    tol3z = sqrt( LAPACK_dlamch( & dlmach_param
    #ifdef LAPACK_FORTRAN_STRLEN_END
    , 1
    #endif
    ) );
    ptr_d  = buff_d;
    ptr_e  = buff_e;
    ptr_wt = buff_wt;
    ptr_A  = buff_A;

    // Main loop.
    //#pragma omp parallel for
    for( j = 0; j < n_A; j++ ) {
    if( * ptr_d != 0.0 ) {
        temp = dabs( * ptr_wt ) / * ptr_d;
        temp = std::max( 0.0, ( 1.0 + temp ) * ( 1 - temp ) );
        temp5 = * ptr_d / * ptr_e;
        temp2 = temp * temp5 * temp5;
        if( temp2 <= tol3z ) {
            if( m_A > 0 ) {
                * ptr_d = blas::nrm2( m_A, ptr_A, i_one );
                * ptr_e = *ptr_d;
            } else {
                * ptr_d = 0.0;
                * ptr_e = 0.0;
            }
        } else {
            * ptr_d = * ptr_d * sqrt( temp );
        }
    } 
    ptr_A  += ldim_A;
    ptr_d  += st_d;
    ptr_e  += st_e;
    ptr_wt += st_wt;
    }

    return 0;
}

// ============================================================================
template <typename T>
static int64_t NoFLA_QRP_pivot_G_B_C( 
    int64_t j_max_col,
    int64_t m_G, T * buff_G, int64_t ldim_G, 
    int64_t pivot_B, int64_t m_B, T * buff_B, int64_t ldim_B, 
    int64_t pivot_C, int64_t m_C, T * buff_C, int64_t ldim_C, 
    int64_t * buff_p,
    T * buff_d, T * buff_e ) {
//
// It pivots matrix G, pivot vector p, and norms vectors d and e.
// Matrices B and C are optionally pivoted.
//
    int64_t     ival = 1; 
    int64_t i_one = 1;
    T  * ptr_g1, * ptr_g2, * ptr_b1, * ptr_b2, * ptr_c1, * ptr_c2;

    // Swap columns of G, pivots, and norms.
    if( j_max_col != 0 ) {
        // Swap full column 0 and column "j_max_col" of G.
        ptr_g1 = & buff_G[ 0 + 0         * ldim_G ];
        ptr_g2 = & buff_G[ 0 + j_max_col * ldim_G ];
        blas::swap( m_G, ptr_g1, i_one, ptr_g2, i_one );

        // Swap full column 0 and column "j_max_col" of B.
        if( pivot_B ) {
            ptr_b1 = & buff_B[ 0 + 0         * ldim_B ];
            ptr_b2 = & buff_B[ 0 + j_max_col * ldim_B ];
            blas::swap( m_B, ptr_b1, i_one, ptr_b2, i_one );
        }

        // Swap full column 0 and column "j_max_col" of C.
        if( pivot_C ) {
            ptr_c1 = & buff_C[ 0 + 0         * ldim_C ];
            ptr_c2 = & buff_C[ 0 + j_max_col * ldim_C ];
            blas::swap( m_C, ptr_c1, i_one, ptr_c2, i_one );
        }

        // Swap element 0 and element "j_max_col" of pivot vector "p".
        ival = buff_p[ j_max_col ];
        buff_p[ j_max_col ] = buff_p[ 0 ];
        buff_p[ 0 ] = ival;

        // Copy norms of column 0 to column "j_max_col".
        buff_d[ j_max_col ] = buff_d[ 0 ];
        buff_e[ j_max_col ] = buff_e[ 0 ];
    }
    return 0;
}

// ==========================================================================
template <typename T>
static int64_t GEQRF_mod_WY(
        int64_t num_stages,
        int64_t m_A, int64_t n_A, T * buff_A, int64_t ldim_A,
        T * buff_t,
        T * buff_T, int64_t ldim_T
) {
    high_resolution_clock::time_point timing_t_start = high_resolution_clock::now();
    //
    // Simplification of NoFLA_QRPmod_WY_unb_var4 for the case when pivoting=0.
    //

    // Some initializations.
    if( num_stages < 0 )
        num_stages = std::min( m_A, n_A );;

    // run unpivoted Householder QR on buff_A.
    int64_t info[1];
    T work_query[1];
    int64_t lwork[1];
    lwork[0] = -1;
    _LAPACK_geqrf(m_A, n_A, buff_A, ldim_A, buff_t, work_query, lwork, info);
    lwork[0] = std::max((int64_t) blas::real(work_query[0]), n_A);
    T *buff_workspace = ( T * ) malloc( lwork[0] * sizeof( T ) );
    _LAPACK_geqrf(m_A, n_A, buff_A, ldim_A, buff_t, buff_workspace, lwork, info);

    // Build T.
    lapack::larft( lapack::Direction::Forward,
                    lapack::StoreV::Columnwise,
                    m_A, num_stages, buff_A, ldim_A, 
                    buff_t, buff_T, ldim_T
    );

    // Remove auxiliary vectors.
    free( buff_workspace );
    high_resolution_clock::time_point timing_t_stop = high_resolution_clock::now();
    printf("+%ld\n", duration_cast<microseconds>(timing_t_stop - timing_t_start).count());
    return 0;
}

// ==========================================================================
// TODO: pre-allocate workspace

template <typename T>
static int64_t CHOLQR_mod_WY(
        int64_t num_stages,
        int64_t m_A, int64_t n_A, T * buff_A, int64_t ldim_A,
        T * buff_t,
        T * buff_T, int64_t ldim_T
) {
    high_resolution_clock::time_point timing_t_start = high_resolution_clock::now();
    //
    // Simplification of NoFLA_QRPmod_WY_unb_var4 for the case when pivoting=0.
    //

    // Some initializations.
    if( num_stages < 0 )
        num_stages = std::min( m_A, n_A );

    // run unpivoted Cholesky QR on buff_A.
    // Allocate space for the R-factor.
    // This should ONLY require n_A by n_A space.
    T *buff_R = ( T*) malloc(n_A * n_A * sizeof( T ));

    // Find R = A^TA.
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n_A, m_A, 1.0, buff_A, ldim_A, 0.0, buff_R, n_A);

    // Perform Cholesky factorization on A.
    lapack::potrf(Uplo::Upper, n_A, buff_R, n_A);
    // Find Q = A * inv(R)

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m_A, n_A, 1.0, buff_R, n_A, buff_A, ldim_A);

    // Perform Householder reconstruction
    // Allocate space for the sign vector D
    T *buff_D = ( T*) malloc(m_A * sizeof( T ));
    lapack::orhr_col(m_A, n_A, n_A, buff_A, ldim_A, buff_T, ldim_T, buff_D);

    // Update the signs in the R-factor
    int i, j;
    for(i = 0; i < n_A; ++i)
        for(j = 0; j < (i + 1); ++j)
            buff_R[(n_A * i) + j] *= buff_D[j];

    // Copy the R-factor into the upper-trianular portion of A
    lapack::lacpy(MatrixType::Upper, n_A, n_A, buff_R, n_A, buff_A, ldim_A);

    // Entries of tau will be placed on the main diagonal of matrix T from orhr_col().
    for(i = 0; i < n_A; ++i)
        buff_t[i] = buff_T[(ldim_T + 1) * i];

    // Remove auxiliary vectors.
    free( buff_D );
    free( buff_R );

    high_resolution_clock::time_point timing_t_stop = high_resolution_clock::now();
    printf("+%ld\n", duration_cast<microseconds>(timing_t_stop - timing_t_start).count());
    return 0;
}

// ============================================================================
template <typename T>
int64_t NoFLA_QRPmod_WY_unb_var4( 
    int64_t use_cholqr, int64_t pivoting, int64_t num_stages, 
    int64_t m_A, int64_t n_A, T * buff_A, int64_t ldim_A,
    int64_t * buff_p, T * buff_t, 
    int64_t pivot_B, int64_t m_B, T * buff_B, int64_t ldim_B,
    int64_t pivot_C, int64_t m_C, T * buff_C, int64_t ldim_C,
    int64_t build_T, T * buff_T, int64_t ldim_T ) {
//
// "pivoting": If pivoting==1, then QR factorization with pivoting is used.
//
// "numstages": It tells the number of columns that are factorized.
//   If "num_stages" is negative, the whole matrix A is factorized.
//   If "num_stages" is positive, only the first "num_stages" are factorized.
//   The typical use-case for this function is to call with num_stages=-1.
//   Calling with num_stages > 0 only happens at HQRRP's last iteration.
//
// "pivot_B": if "pivot_B" is true, matrix "B" is pivoted too.
//
// "pivot_C": if "pivot_C" is true, matrix "C" is pivoted too.
//
// "build_T": if "build_T" is true, matrix "T" is built.
//    The typical use-case for this function is to call with build_T=true.
//    Calling with build_T=false is only done at HQRRP's last iteration.
//

    if (!pivoting && !use_cholqr) {
        return GEQRF_mod_WY(num_stages, m_A, n_A, buff_A, ldim_A, buff_t, buff_T, ldim_T);
    } else if (!pivoting && use_cholqr) {
        return CHOLQR_mod_WY(num_stages, m_A, n_A, buff_A, ldim_A, buff_t, buff_T, ldim_T);
    }

    int64_t j, mn_A, m_a21, m_A22, n_A22, n_dB, idx_max_col, 
            i_one = 1, n_house_vector, m_rest;
    T  * buff_d, * buff_e, * buff_workspace, diag;

    //// printf( "NoFLA_QRPmod_WY_unb_var4. pivoting: %d \n", pivoting );

    // Some initializations.
    mn_A    = std::min( m_A, n_A );

    // Set the number of stages, if needed.
    if( num_stages < 0 ) {
    num_stages = mn_A;
    }

    // Create auxiliary vectors.
    buff_d         = ( T * ) malloc( n_A * sizeof( T ) );
    buff_e         = ( T * ) malloc( n_A * sizeof( T ) );
    buff_workspace = ( T * ) malloc( n_A * sizeof( T ) );

    // Compute initial norms of A int64_to d and e.
    NoFLA_QRP_compute_norms( m_A, n_A, buff_A, ldim_A, buff_d, buff_e );

    // Main Loop.
    for( j = 0; j < num_stages; j++ ) {
        n_dB  = n_A - j;
        m_a21 = m_A - j - 1;
        m_A22 = m_A - j - 1;
        n_A22 = n_A - j - 1;

        // Obtain the index of the column with largest 2-norm.
        idx_max_col = blas::iamax( n_dB, & buff_d[ j ], i_one ); // - 1;

        // Swap columns of A, B, C, pivots, and norms vectors.
        NoFLA_QRP_pivot_G_B_C( idx_max_col,
            m_A, & buff_A[ 0 + j * ldim_A ], ldim_A,
            pivot_B, m_B, & buff_B[ 0 + j * ldim_B ], ldim_B,
            pivot_C, m_C, & buff_C[ 0 + j * ldim_C ], ldim_C,
            & buff_p[ j ],
            & buff_d[ j ],
            & buff_e[ j ] );

        // Compute tau1 and u21 from alpha11 and a21 such that tau1 and u21
        // determine a Householder transform H such that applying H from the
        // left to the column vector consisting of alpha11 and a21 annihilates
        // the entries in a21 (and updates alpha11).
        n_house_vector = m_a21 + 1;
        lapack::larfg(n_house_vector,
            & buff_A[ j + j * ldim_A ],
            & buff_A[ std::min( m_A-1, j+1 ) + j * ldim_A ],
            i_one,
            & buff_t[j]
        );

        // | a12t | =  H | a12t |
        // | A22  |      | A22  |
        //
        // where H is formed from tau1 and u21.
        diag = buff_A[ j + j * ldim_A ];
        buff_A[ j + j * ldim_A ] = 1.0;
        m_rest = m_A22 + 1;
        _LAPACK_larf( lapack::Side::Left, m_rest, n_A22, 
            & buff_A[ j + j * ldim_A ], 1,
            & buff_t[ j ],
            & buff_A[ j + ( j+1 ) * ldim_A ], ldim_A,
            buff_workspace
        );
        buff_A[ j + j * ldim_A ] = diag;

        // Update partial column norms.
        NoFLA_QRP_downdate_partial_norms( m_A22, n_A22, 
            & buff_d[ j+1 ], 1,
            & buff_e[ j+1 ], 1,
            & buff_A[ j + ( j+1 ) * ldim_A ], ldim_A,
            & buff_A[ ( j+1 ) + std::min( n_A-1, ( j+1 ) ) * ldim_A ], ldim_A );
    }

    // Build T.
    if( build_T ) {
    lapack::larft( lapack::Direction::Forward,
                    lapack::StoreV::Columnwise,
                    m_A, num_stages, buff_A, ldim_A, 
                    buff_t, buff_T, ldim_T);
    }

    // Remove auxiliary vectors.
    free( buff_d );
    free( buff_e );
    free( buff_workspace );

    return 0;
}

// HQRRP: It computes the Householder QR with Randomized Pivoting of matrix A.
// This routine is almost compatible with LAPACK's dgeqp3.
// The main difference is that this routine does not manage fixed columns.
//
// Main features:
//   * BLAS-3 based.
//   * Norm downdating method by Drmac.
//   * Downdating for computing Y.
//   * No use of libflame.
//   * Compact WY transformations are used instead of UT transformations.
//   * LAPACK's routine dlarfb is used to apply block transformations.
//
// Arguments:
// ----------
// m_A:            Number of rows of matrix A.
// n_A:            Number of columns of matrix A.
// buff_A:         Address/pointer of/to data in matrix A. Matrix A must be 
//                 stored in column-order.
// ldim_A:         Leading dimension of matrix A.
// buff_jpvt:      Input/output vector with the pivots.
// buff_tau:       Output vector with the tau values of the Householder factors.
// nb_alg:         Block size. 
//                 Usual values for nb_alg are 32, 64, etc.
// pp:             Oversampling size.
//                 Usual values for pp are 5, 10, etc.
// panel_pivoting: If panel_pivoting==1, QR with pivoting is applied to 
//                 factorize the panels of matrix A. Otherwise, QR without 
//                 pivoting is used. Usual value for panel_pivoting is 1.
// Final comments:
// ---------------
// This code has been created from a libflame code. Hence, you can find some
// commented calls to libflame routines. We have left them to make it easier
// to interpret the meaning of the C code.
template <typename T, typename RNG>
int64_t hqrrp( 
    int64_t m_A, int64_t n_A, T * buff_A, int64_t ldim_A,
    int64_t * buff_jpvt, T * buff_tau,
    int64_t nb_alg, int64_t pp, int64_t panel_pivoting, int64_t use_cholqr, RandBLAS::RNGState<RNG> &state, T* block_per_time) {

    int64_t b, j, last_iter, mn_A, m_Y, n_Y, ldim_Y, m_V, n_V, ldim_V, 
            m_W, n_W, ldim_W, n_VR, m_AB1, n_AB1, ldim_T1_T,
            n_A11, m_A12, n_A12, m_A21, m_A22,
            m_G, n_G, ldim_G;
    T  * buff_Y, * buff_V, * buff_W, * buff_VR, * buff_YR, 
            * buff_s, * buff_sB, * buff_s1, 
            * buff_AR, * buff_AB1, * buff_A01, * buff_Y1, * buff_T1_T,
            * buff_A11, * buff_A21, * buff_A12,
            * buff_Y2, * buff_G, * buff_G1, * buff_G2;
    int64_t * buff_p, * buff_pB, * buff_p1;
    T  d_zero = 0.0;
    T  d_one  = 1.0;

    // Executable Statements.

    // Check arguments.
    if( m_A < 0 ) {
        fprintf( stderr, "ERROR in hqrrp: m_A is < 0.\n" );
    } if( n_A < 0 ) {
        fprintf( stderr, "ERROR in hqrrp: n_A is < 0.\n" );
    } if( ldim_A < std::max<int64_t>( 1, m_A ) ) {
        fprintf( stderr, "ERROR in hqrrp: ldim_A is < std::max<int64_t>( 1, m_A ).\n" );
    }

    // Some initializations.
    mn_A   = std::min( m_A, n_A );
    buff_p = buff_jpvt;
    buff_s = buff_tau;

    // Quick return.
    if( mn_A == 0 ) {
        return 0;
    }

    // Create auxiliary objects.
    m_Y     = nb_alg + pp;
    n_Y     = n_A;
    buff_Y  = ( T * ) malloc( m_Y * n_Y * sizeof( T ) );
    ldim_Y  = m_Y;

    m_V     = nb_alg + pp;
    n_V     = n_A;
    buff_V  = ( T * ) malloc( m_V * n_V * sizeof( T ) );
    ldim_V  = m_V;

    m_W     = nb_alg;
    n_W     = n_A;
    buff_W  = ( T * ) malloc( m_W * n_W * sizeof( T ) );
    ldim_W  = m_W;

    m_G     = nb_alg + pp;
    n_G     = m_A;
    buff_G  = ( T * ) malloc( m_G * n_G * sizeof( T ) );
    ldim_G  = m_G;

    // Initialize matrices G and Y.
    RandBLAS::DenseDist D{.n_rows = nb_alg + pp, .n_cols = m_A, .family=RandBLAS::DenseDistName::Uniform};
    state = RandBLAS::fill_dense(D, buff_G, state);
    
    blas::gemm(Layout::ColMajor,
                Op::NoTrans, Op::NoTrans, m_Y, n_Y, m_A, 
                d_one, buff_G,  ldim_G, buff_A, ldim_A, 
                d_zero, buff_Y, ldim_Y );

    //**********************************
    // This is for the advanced timing
    high_resolution_clock::time_point iter_t_start;
    high_resolution_clock::time_point iter_t_stop;
    if (block_per_time != nullptr) {
        // The space required has already been preallocated
        iter_t_start  = high_resolution_clock::now();
    }

    //**********************************

    // Main Loop.
    for( j = 0; j < mn_A; j += nb_alg ) {
        b = std::min( nb_alg, std::min( n_A - j, m_A - j ) );

        // Check whether it is the last iteration.
        last_iter = ( ( ( j + nb_alg >= m_A )||( j + nb_alg >= n_A ) ) ? 1 : 0 );

        // Some initializations for the iteration of this loop.
        n_VR = n_V - j;
        buff_VR = & buff_V[ 0 + j * ldim_V ];
        buff_YR = & buff_Y[ 0 + j * ldim_Y ];
        buff_pB = & buff_p[ j ];
        buff_sB = & buff_s[ j ];
        buff_AR = & buff_A[ 0 + j * ldim_A ];

        m_AB1     = m_A - j;
        n_AB1     = b;
        buff_AB1  = & buff_A[ j + j * ldim_A ];
        buff_p1   = & buff_p[ j ];
        buff_s1   = & buff_s[ j ];
        buff_A01  = & buff_A[ 0 + j * ldim_A ];
        buff_Y1   = & buff_Y[ 0 + j * ldim_Y ];
        buff_T1_T = & buff_W[ 0 + j * ldim_W ];
        ldim_T1_T = ldim_W;

        buff_A11 = & buff_A[ j + j * ldim_A ];
        n_A11 = b;

        buff_A21 = & buff_A[ std::min( m_A - 1, j + nb_alg ) + j * ldim_A ];
        m_A21 = std::max<int64_t>( 0, m_A - j - b );

        buff_A12 = & buff_A[ j + std::min( n_A - 1, j + b ) * ldim_A ];
        m_A12 = b;
        n_A12 = std::max<int64_t>( 0, n_A - j - b );

        //// buff_A22 = & buff_A[ std::min( m_A - 1, j + b ) + 
        ////                      std::min( n_A - 1, j + b ) * ldim_A ];
        m_A22 = std::max<int64_t>( 0, m_A - j - b );
        //// n_A22 = std::max<int64_t>( 0, n_A - j - b );

        buff_Y2 = & buff_Y[ 0 + std::min( n_Y - 1, j + b ) * ldim_Y ];
        buff_G1 = & buff_G[ 0 + j * ldim_G ];
        buff_G2 = & buff_G[ 0 + std::min( n_G - 1, j + b ) * ldim_G ];
            
#ifdef CHECK_DOWNDATING_OF_Y
        // Check downdating of matrix Y: Compare downdated matrix Y with 
        // matrix Y computed from scratch.
        int64_t     m_cyr, n_cyr, ldim_cyr, m_ABR, ii, jj;
        T  * buff_cyr, aux, sum;

        m_cyr    = m_Y;
        n_cyr    = n_Y - j;
        ldim_cyr = m_cyr;
        m_ABR    = m_A - j;
        buff_cyr = ( T * ) malloc( m_cyr * n_cyr * sizeof( T ) );

        //// FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE, 
        ////           FLA_ONE, GR, ABR, FLA_ZERO, CYR ); 
        blas::gemm(Layout::ColMajor,
                    Op::NoTrans, Op::NoTrans, m_cyr, n_cyr, m_ABR,
                    d_one, & buff_G[ 0 + j * ldim_G ], ldim_G,
                            & buff_A[ j + j * ldim_A ], ldim_A,
                    d_zero, & buff_cyr[ 0 + 0 * ldim_cyr ], ldim_cyr );

        //// print_double_matrix( "cyr", m_cyr, n_cyr, buff_cyr, ldim_cyr );
        //// print_double_matrix( "y", m_Y, n_Y, buff_Y, ldim_Y );
        sum = 0.0;
        //#pragma omp parallel for
        for( jj = 0; jj < n_cyr; jj++ ) {
            for( ii = 0; ii < m_cyr; ii++ ) {
            aux = buff_Y[ ii + ( j + jj ) * ldim_Y ] -
                    buff_cyr[ ii + jj * ldim_cyr ];
            sum += aux * aux;
            }
        }
        sum = sqrt( sum );
        printf( "%%  diff between Y and downdated Y: %le\n", sum );

        free( buff_cyr );
#endif

        if( !last_iter ) {
            // Compute QRP of YR, and apply permutations to matrix AR.
            // A copy of YR is made into VR, and permutations are applied to YR.
            //
            //    Notes
            //    -----
            //    The "NoFLA" function below is basically running GEQP3 on the updated sketch.
            //
            //    I only see one reason for doing this with a custom function instead of GEQP3 itself.
            //    Specifically, this custom function operates on three matrices (VR, AR, and YR) in
            //    sync with one another, while GEQP3 only operates on one matrix.
            //
            lapack::lacpy( MatrixType::General,
                            m_V, n_VR,
                            buff_YR, ldim_Y,
                            buff_VR, ldim_V);
            NoFLA_QRPmod_WY_unb_var4(0, 1, b,
                m_V, n_VR,
                buff_VR, ldim_V,
                buff_pB, buff_sB,
                1, m_A, buff_AR, ldim_A,
                1, m_Y, buff_YR, ldim_Y,
                0, (T*) nullptr, 0 
            );
        }

        //
        // Compute QRP of panel AB1 = [ A11; A21 ].
        // Apply same permutations to A01 and Y1, and build T1_T.
        //
        //    Notes
        //    -----
        //    The function below basically runs GEQP3 *or* GEQRF on
        //    the updated sketch *and then* changes the representation of the
        //    composition of Householder reflectors.
        //
        //    In the code path where we hit a GEQP3-like function we can't use
        //    GEQP3 directly because we actually need to modify three matrices
        //    (AB1, A01, and Y1) alongside one another.
        //    
        //    The code path where we hit a GEQRF-like function is very different;
        //    it only operates on AB1!
        //
        NoFLA_QRPmod_WY_unb_var4(use_cholqr, panel_pivoting, -1,
            m_AB1, n_AB1, buff_AB1, ldim_A, buff_p1, buff_s1,
            1, j, buff_A01, ldim_A,
            1, m_Y, buff_Y1, ldim_Y,
            1, buff_T1_T, ldim_W );

        //
        // Update the rest of the matrix.
        //
        if ( ( j + b ) < n_A ) {
            // Apply the Householder transforms associated with AB1 = [ A11; A21 ] 
            // and T1_T to [ A12; A22 ]:
            //   | A12 | := QB1' | A12 |
            //   | A22 |         | A22 |
            // where QB1 is formed from AB1 and T1_T.
            NoFLA_Apply_Q_WY_lhfc_blk_var4( 
            n_A11, buff_A11, ldim_A,
            buff_T1_T, ldim_W, m_A12 + m_A22, 
            n_A12, buff_A12, ldim_A );
        }

        //
        // Downdate matrix Y.
        //
        if ( ! last_iter ) {
            NoFLA_Downdate_Y<T>(
                n_A11, buff_A11, ldim_A,
                m_A21, buff_A21, ldim_A,
                m_A12, buff_A12, ldim_A,
                buff_T1_T, ldim_T1_T,
                m_Y, std::max<int64_t>( 0, n_Y - j - b ), buff_Y2, ldim_Y,
                m_G, b, buff_G1, ldim_G,
                std::max<int64_t>( 0, n_G - j - b ), buff_G2, ldim_G );
        }
        if (block_per_time != nullptr) {
            // The space required has already been preallocated
            iter_t_stop  = high_resolution_clock::now();
            T* nextval = &(block_per_time[j / nb_alg]);
            *nextval = ((mn_A - j) * (T) nb_alg) / duration_cast<microseconds>(iter_t_stop - iter_t_start).count();
        }
    }

    // Remove auxiliary objects.
    free( buff_G );
    free( buff_Y );
    free( buff_V );
    free( buff_W );

    return 0;
}

} // end namespace RandLAPACK
#endif