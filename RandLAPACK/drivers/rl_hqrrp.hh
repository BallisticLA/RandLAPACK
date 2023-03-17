#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <typeinfo>

#ifndef randlapack_hqrrp_h
#define randlapack_hqrrp_h

#include "rl_hqrrp.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <lapack/fortran.h>
#include <lapack/config.h>

// Matrices with dimensions larger than THRESHOLD_FOR_DGEQP3 are processed
// with the new HQRRP code.
#define THRESHOLD_FOR_DGEQP3  3

// ============================================================================
// Definition of macros.
#define dabs( a )    ( (a) >= 0.0 ? (a) : -(a) )

// ============================================================================
// Compilation declarations.

#undef CHECK_DOWNDATING_OF_Y

namespace RandLAPACK {

// ============================================================================
// Definition of macros.

#define max_untyped( a, b )  ( (a) > (b) ? (a) : (b) )
#define min_untyped( a, b )  ( (a) > (b) ? (b) : (a) )
#define dabs( a )    ( (a) >= 0.0 ? (a) : -(a) )

// ============================================================================
// Compilation declarations.

#undef CHECK_DOWNDATING_OF_Y


// ============================================================================
// Declaration of local prototypes.

static int64_t NoFLA_Normal_random_matrix( int64_t m_A, int64_t n_A, 
               double * buff_A, int64_t ldim_A );

static double NoFLA_Normal_random_number( double mu, double sigma );

static int64_t NoFLA_Downdate_Y( 
               int64_t m_U11, int64_t n_U11, double * buff_U11, int64_t ldim_U11,
               int64_t m_U21, int64_t n_U21, double * buff_U21, int64_t ldim_U21,
               int64_t m_A12, int64_t n_A12, double * buff_A12, int64_t ldim_A12,
               int64_t m_T, int64_t n_T, double * buff_T, int64_t ldim_T,
               int64_t m_Y2, int64_t n_Y2, double * buff_Y2, int64_t ldim_Y2,
               int64_t m_G1, int64_t n_G1, double * buff_G1, int64_t ldim_G1,
               int64_t m_G2, int64_t n_G2, double * buff_G2, int64_t ldim_G2 );

static int64_t NoFLA_Apply_Q_WY_lhfc_blk_var4( 
               int64_t m_U, int64_t n_U, double * buff_U, int64_t ldim_U,
               int64_t m_T, int64_t n_T, double * buff_T, int64_t ldim_T,
               int64_t m_B, int64_t n_B, double * buff_B, int64_t ldim_B );

static int64_t NoFLA_Apply_Q_WY_rnfc_blk_var4( 
               int64_t m_U, int64_t n_U, double * buff_U, int64_t ldim_U,
               int64_t m_T, int64_t n_T, double * buff_T, int64_t ldim_T,
               int64_t m_B, int64_t n_B, double * buff_B, int64_t ldim_B );

static int64_t NoFLA_QRPmod_WY_unb_var4( int64_t pivoting, int64_t num_stages, 
               int64_t m_A, int64_t n_A, double * buff_A, int64_t ldim_A,
               int64_t * buff_p, double * buff_t, 
               int64_t pivot_B, int64_t m_B, double * buff_B, int64_t ldim_B,
               int64_t pivot_C, int64_t m_C, double * buff_C, int64_t ldim_C,
               int64_t build_T, double * buff_T, int64_t ldim_T );

static int64_t NoFLA_QRP_compute_norms(
               int64_t m_A, int64_t n_A, double * buff_A, int64_t ldim_A,
               double * buff_d, double * buff_e );

static int64_t NoFLA_QRP_downdate_partial_norms( int64_t m_A, int64_t n_A,
               double * buff_d,  int64_t st_d,
               double * buff_e,  int64_t st_e,
               double * buff_wt, int64_t st_wt,
               double * buff_A,  int64_t ldim_A );

static int64_t NoFLA_QRP_pivot_G_B_C( int64_t j_max_col,
               int64_t m_G, double * buff_G, int64_t ldim_G, 
               int64_t pivot_B, int64_t m_B, double * buff_B, int64_t ldim_B, 
               int64_t pivot_C, int64_t m_C, double * buff_C, int64_t ldim_C, 
               int64_t * buff_p,
               double * buff_d, double * buff_e );

/*
void _LAPACK_dgeqp3(
  int64_t m, int64_t n, double *A, int64_t lda,
  int64_t * jpvt, double *tau, double *work,
  int64_t * lwork, int64_t * info)
{
  lapack_int m_ = (lapack_int) m;
  lapack_int n_ = (lapack_int) n;
  lapack_int lda_ = (lapack_int) lda;
  lapack_int *jpvt_ = (lapack_int *) malloc(n_ * sizeof(lapack_int));
  for (int64_t i = 0; i < n; ++i)
  {
    jpvt_[i] = (lapack_int) jpvt[i];
  }
  lapack_int *lwork_ = (lapack_int *) lwork;
  lapack_int *info_ = (lapack_int *) info;
  LAPACK_dgeqrf(&m_, &n_, A, &lda_, tau, work, lwork_, info_);
  for (int64_t i = 0; i < n; ++i)
  {
    jpvt[i] = (int64_t) jpvt[i];
  }
  *info = (int64_t) *info_;
  free(jpvt_);
  return;
}

void _LAPACK_dgeqrf(
  int64_t m, int64_t n, double *A, int64_t lda,
  double *tau, double *work,
  int64_t * lwork, int64_t * info)
{
  lapack_int m_ = (lapack_int) m;
  lapack_int n_ = (lapack_int) n;
  lapack_int lda_ = (lapack_int) lda;
  lapack_int *lwork_ = (lapack_int *) lwork;
  lapack_int *info_ = (lapack_int *) info;
  LAPACK_dgeqrf(&m_, &n_, A, &lda_, tau, work, lwork_, info_);
  *info = (int64_t) *info_;
  return;
}

void _LAPACK_dormqr(
  blas::Side side,
  lapack::Op op,
  int64_t m, int64_t n, int64_t k, double *A, int64_t lda,
  double * tau, double *b, int64_t ldb,
  double * work, int64_t * lwork, int64_t * info)
{
  char side_ = blas::side2char(side);
  char trans_ = blas::op2char(op);
  lapack_int m_ = (lapack_int) m;
  lapack_int n_ = (lapack_int) n;
  lapack_int k_ = (lapack_int) k;
  lapack_int lda_ = (lapack_int) lda;
  lapack_int *lwork_ = (lapack_int *) lwork;
  lapack_int *info_ = (lapack_int *) info;
  lapack_int ldb_ = (lapack_int) ldb;
  LAPACK_dormqr(& side_, & trans_,
            & m_, & n_, & k_,
            A, & lda_, tau,
            b, & ldb_, 
            work, lwork_, info_
          #ifdef LAPACK_FORTRAN_STRLEN_END
          , 1, 1
          #endif 
            );
  *info = (int64_t) *info_;
  return;
}

void _LAPACK_dlacpy(
  lapack::MatrixType atype,
  int64_t m, int64_t n, double *A, int64_t lda,
  double *b, int64_t ldb)
{
  lapack_int m_ = (lapack_int) m;
  lapack_int n_ = (lapack_int) n;
  lapack_int lda_ = (lapack_int) lda;
  lapack_int ldb_ = (lapack_int) ldb;
  char matrixtype_ = matrixtype2char( atype );
  LAPACK_dlacpy( &matrixtype_, & m_, & n_, A, & lda_,
                                b, & ldb_
              #ifdef LAPACK_FORTRAN_STRLEN_END
              , 1
              #endif
              );
}
*/
void _LAPACK_dlafrb(
  lapack::Side side,
  lapack::Op op,
  lapack::Direction dir,
  lapack::StoreV  storev,
  int64_t m, int64_t n, int64_t k,
  double *buff_U, int64_t ldu,
  double *buff_T, int64_t ldt,
  double *buff_B, int64_t ldb,
  double *buff_W, int64_t ldw
)
{
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
  LAPACK_dlarfb( & side_, & trans_, & direction_, & storev_,  
                & m_, & n_, & k_, buff_U, & ldim_U, buff_T, & ldim_T, 
                buff_B, & ldim_B, buff_W, & ldim_W
                #ifdef LAPACK_FORTRAN_STRLEN_END
                , 1, 1, 1, 1
                #endif
                );
  return;
}


void _LAPACK_dlarf(
  lapack::Side side,
  int64_t m, int64_t n,
  double *v, int64_t inc_v, double tau,
  double *C, int64_t ldc,
  double *work
)
{
    char side_ = blas::side2char( side );
    lapack_int inc_v_ = (lapack_int) inc_v;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int ldc_ = (lapack_int) ldc;
    LAPACK_dlarf( & side_, & m_, & n_, 
        v, & inc_v_,
        & tau,
        C, & ldc_,
        work
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1
        #endif
        );
  return;
}

/*
// ============================================================================
void dgeqpr( int64_t * m, int64_t * n, double * A, int64_t * lda, int64_t * jpvt, double * tau,
        double * work, int64_t * lwork, int64_t * info ) {
// 
// This routine is plug compatible with LAPACK's routine dgeqp3.
// It computes the new HQRRP while keeping the same header as LAPACK's dgeqp3.
// It uses dgeqp3 for small matrices. The threshold is defined in the
// constant THRESHOLD_FOR_DGEQP3.
//
// The work parameters (work and lwork) are only used when this routine calls
//    LAPACK_dgeqp3 (for sufficiently small matrices), and
//    LAPACK_dgeqrf and LAPACK_dormqr (when jpvt specifies fixed cols on entry).
// The work parameters do not affect the main randomized algorithm, which is
// implemented by hqrrp.
//
// In particular, the work parameters do not effect algorithm performance when
// no fixed columns are specified and the smaller dimension of the input matrix
// is larger than THRESHOLD_FOR_DGEQP3. 
//   
  int64_t     INB = 1;
  int64_t     i_one = 1, i_minus_one = -1, 
          m_A, n_A, mn_A, ldim_A, lquery, nb, num_factorized_fixed_cols, 
          minus_info, iws, lwkopt, j, k, num_fixed_cols, n_rest, itmp;
  int64_t     * previous_jpvt;

  using blas::real;

  // Some initializations.
  m_A    = * m;
  n_A    = * n;
  mn_A   = min_untyped( m_A, n_A );
  ldim_A = * lda;

  int64_t ineg_one = -1;

  // Check input arguments.
  * info = 0;
  lquery = ( * lwork == -1 );
  if( m_A < 0 ) {
     * info = -1;
  } else if ( n_A < 0 ) {
     * info = -2;
  } else if ( ldim_A < max_untyped( 1, m_A ) ) {
     * info = -4;
  }

  if( *info == 0 ) {
    if( mn_A == 0 ) {
      iws    = 1;
      lwkopt = 1;
    } else {
      
      // This code block originally called ilaenv_ in order to find the optimal "nb",
      // then set lwkopt = 2 * n_A + (n_A + 1) * nb.
      // We can't do that because LAPACK++ doesn't expose ILAENV, but we can look at what
      // LAPACK++ does and back out a valid value for "nb".
      
          // LAPACK++ gets the workspace by calling DGEQRF
          // https://bitbucket.org/icl/lapackpp/src/7f1feb420fd94332200ac0636bab451157cbee6d/src/geqrf.cc#lines-81:92
          
          // DGEQRF calls ILAENV to get NB, and then sets LWKOPT to N*NB, where N is the number of cols in A
          // https://github.com/Reference-LAPACK/lapack/blob/a066b6a08f23186f2f38f1d9167f6616528ad89f/SRC/dgeqrf.f#L200
          // So we'll call DGEQRF to get LWKOPT_DETERM, assume LWKOPT_DETERM = NB*N, then set NB = LWKOPT_DETERM / N.

      iws    = 3 * n_A + 1;
      double qry_work[1];
      _LAPACK_dgeqrf(
          m_A, n_A,
          A, ldim_A,
          tau,
          qry_work, &ineg_one, info );
      if (*info < 0) {
          throw blas::Error();
      }
      int64_t lwork_determ = real(qry_work[0]);
      int64_t nb = ((int64_t) lwork_determ) / n_A;
      lwkopt = 2 * n_A + (n_A + 1) * nb;
    }
    work[ 0 ] = ( double ) lwkopt;

    if ( ( * lwork < iws )&&( ! lquery ) ) {
      * info = -8;
    }
  }

  if( * info != 0 ) {
    throw blas::Error();
  } else if( lquery ) {
    return;
  }

  // Quick return if possible.
  if( mn_A == 0 ) {
    return;
  }

  // Use LAPACK's DGEQP3 for small matrices.
  if( mn_A < THRESHOLD_FOR_DGEQP3 ) {
    //// printf( "Calling dgeqp3\n" );
    _LAPACK_dgeqp3( m_A,
                    n_A, A,
                    ldim_A, jpvt, tau, work, lwork, info );
    return;
  }

  // Move initial columns up front.
  num_fixed_cols = 0;
  for( j = 0; j < n_A; j++ ) {
    if( jpvt[ j ] != 0 ) {
      if( j != num_fixed_cols ) {
        /// printf( "Swapping columns: %d %d \n", (int) j, (int) num_fixed_cols );
        blas::swap( m_A, & A[ 0 + j              * ldim_A ], i_one, 
                         & A[ 0 + num_fixed_cols * ldim_A ], i_one );
        jpvt[ j ] = jpvt[ num_fixed_cols ];
        jpvt[ num_fixed_cols ] = j + 1;
      } else {
        jpvt[ j ] = j + 1 ;
      }
      num_fixed_cols++;
    } else {
      jpvt[ j ] = j + 1 ;
    }
  }

  // Factorize fixed columns at the front.
  num_factorized_fixed_cols = min_untyped( m_A, num_fixed_cols );
  if( num_factorized_fixed_cols > 0 ) {
    _LAPACK_dgeqrf( m_A, num_factorized_fixed_cols, A, ldim_A, tau, work, lwork,
                    info );
    if( * info != 0 ) {
      fprintf( stderr, "ERROR in dgeqrf: Info: %d \n", (int) (*info) );
    }
    iws = max_untyped( iws, ( int64_t ) work[ 0 ] );
    if( num_factorized_fixed_cols < n_A ) {
      n_rest = n_A - num_factorized_fixed_cols;
      _LAPACK_dormqr( blas::Side::Left, lapack::Op::Trans,
                      m_A, n_rest, num_factorized_fixed_cols,
                      A, ldim_A, tau,
                      & A[ 0 + num_factorized_fixed_cols * ldim_A ], ldim_A, 
                      work, lwork, info);
      if( * info != 0 ) {
        fprintf( stderr, "ERROR in dormqr: Info: %d \n", (int) (*info ));
      }

      iws = max_untyped( iws, ( int64_t ) work[ 0 ] );
    }
  }

  // Create intermediate jpvt vector.
  previous_jpvt = ( int64_t * ) malloc( n_A * sizeof( int64_t ) );

  // Save a copy of jpvt vector.
  if( num_factorized_fixed_cols > 0 ) {
    // Copy vector.
    for( j = 0; j < n_A; j++ ) {
      previous_jpvt[ j ] = jpvt[ j ];
    }
  }

  // Factorize free columns at the bottom with default values:
  // nb_alg = 64, pp = 10, panel_pivoting = 1.
  if( num_factorized_fixed_cols < mn_A ) {
    //* info = hqrrp( 
    //    m_A - num_factorized_fixed_cols, n_A - num_factorized_fixed_cols, 
    //    & A[ num_factorized_fixed_cols + num_factorized_fixed_cols * ldim_A ], 
    //        ldim_A,
    //    & jpvt[ num_factorized_fixed_cols ], 
    //    & tau[ num_factorized_fixed_cols ],
    //    64, 10, 1 );
  }

  // Pivot block above factorized block by NoFLA_HQRRP.
  if( num_factorized_fixed_cols > 0 ) {
    // Pivot block above factorized block.
    for( j = num_factorized_fixed_cols; j < n_A; j++ ) {
      //// printf( "%% Processing j: %d \n", j );
      for( k = j; k < n_A; k++ ) {
        if( jpvt[ j ] == previous_jpvt[ k ] ) {
          //// printf( "%%   Found j: %d  k: %d \n", j, k );
          break;
        }
      }
      // Swap vector previous_jpvt and block above factorized block.
      if( k != j ) { 
        // Swap elements in previous_jpvt.
        //// printf( "%%   Swapping  j: %d  k: %d \n", j, k );
        itmp = previous_jpvt[ j ];
        previous_jpvt[ j ] = previous_jpvt[ k ];
        previous_jpvt[ k ] = itmp;

        // Swap columns in block above factorized block.
        blas::swap( num_factorized_fixed_cols,
                & A[ 0 + j * ldim_A ], i_one,
                & A[ 0 + k * ldim_A ], i_one );
      }
    }
  }

  // Remove intermediate jpvt vector.
  free( previous_jpvt );

  // Return workspace length required.
  work[ 0 ] = iws;
  return;
}


void dgeqpr(int64_t m, int64_t n, double *A, int64_t lda, int64_t *jpvt, double *tau)
{
  // This function is compatible with LAPACK++'s geqp3, provided the matrix A is 
  // double precision.
  int64_t info = 0;
  int64_t lwork = -1;
  double *buff_wk_qp4 = (double *) malloc( sizeof( double ) );
  dgeqpr( & m, & n, A, & lda, jpvt, tau, 
          buff_wk_qp4, & lwork, &info );
  if (info != 0) throw lapack::Error();

  lwork = (int64_t) *buff_wk_qp4;
  free(buff_wk_qp4);
  buff_wk_qp4 = ( double * ) malloc( lwork * sizeof( double ) );
  dgeqpr( & m, & n, A, & lda, jpvt, tau, 
          buff_wk_qp4, & lwork, &info );
  if (info != 0) throw lapack::Error();
  
  free(buff_wk_qp4);
  return;
}
*/

// ============================================================================
int64_t hqrrp( int64_t m_A, int64_t n_A, double * buff_A, int64_t ldim_A,
        int64_t * buff_jpvt, double * buff_tau,
        int64_t nb_alg, int64_t pp, int64_t panel_pivoting ) {
//
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
//
  int64_t     b, j, last_iter, mn_A, m_Y, n_Y, ldim_Y, m_V, n_V, ldim_V, 
          m_W, n_W, ldim_W, n_VR, m_AB1, n_AB1, ldim_T1_T,
          m_A11, n_A11, m_A12, n_A12, m_A21, n_A21, m_A22,
          m_G, n_G, ldim_G;
  double  * buff_Y, * buff_V, * buff_W, * buff_VR, * buff_YR, 
          * buff_s, * buff_sB, * buff_s1, 
          * buff_AR, * buff_AB1, * buff_A01, * buff_Y1, * buff_T1_T,
          * buff_A11, * buff_A21, * buff_A12,
          * buff_Y2, * buff_G, * buff_G1, * buff_G2;
  int64_t     * buff_p, * buff_pB, * buff_p1;
  double  d_zero = 0.0;
  double  d_one  = 1.0;

  // Executable Statements.

  // Check arguments.
  if( m_A < 0 ) {
    fprintf( stderr, 
             "ERROR in hqrrp: m_A is < 0.\n" );
  } if( n_A < 0 ) {
    fprintf( stderr, 
             "ERROR in hqrrp: n_A is < 0.\n" );
  } if( ldim_A < max_untyped( 1, m_A ) ) {
    fprintf( stderr, 
             "ERROR in hqrrp: ldim_A is < max_untyped( 1, m_A ).\n" );
  }

  // Some initializations.
  mn_A   = min_untyped( m_A, n_A );
  buff_p = buff_jpvt;
  buff_s = buff_tau;

  // Quick return.
  if( mn_A == 0 ) {
    return 0;
  }

  // Initialize the seed for the generator of random numbers.
  srand( 12 );

  // Create auxiliary objects.
  m_Y     = nb_alg + pp;
  n_Y     = n_A;
  buff_Y  = ( double * ) malloc( m_Y * n_Y * sizeof( double ) );
  ldim_Y  = m_Y;

  m_V     = nb_alg + pp;
  n_V     = n_A;
  buff_V  = ( double * ) malloc( m_V * n_V * sizeof( double ) );
  ldim_V  = m_V;

  m_W     = nb_alg;
  n_W     = n_A;
  buff_W  = ( double * ) malloc( m_W * n_W * sizeof( double ) );
  ldim_W  = m_W;

  m_G     = nb_alg + pp;
  n_G     = m_A;
  buff_G  = ( double * ) malloc( m_G * n_G * sizeof( double ) );
  ldim_G  = m_G;

  // Initialize matrices G and Y.
  NoFLA_Normal_random_matrix( nb_alg + pp, m_A, buff_G, ldim_G );
  blas::gemm(blas::Layout::ColMajor,
             blas::Op::NoTrans, blas::Op::NoTrans, m_Y, n_Y, m_A, 
             d_one, buff_G,  ldim_G, buff_A, ldim_A, 
             d_zero, buff_Y, ldim_Y );

  // Main Loop.
  for( j = 0; j < mn_A; j += nb_alg ) {
    b = min_untyped( nb_alg, min_untyped( n_A - j, m_A - j ) );

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
    m_A11 = b;
    n_A11 = b;

    buff_A21 = & buff_A[ min_untyped( m_A - 1, j + nb_alg ) + j * ldim_A ];
    m_A21 = max_untyped( 0, m_A - j - b );
    n_A21 = b;

    buff_A12 = & buff_A[ j + min_untyped( n_A - 1, j + b ) * ldim_A ];
    m_A12 = b;
    n_A12 = max_untyped( 0, n_A - j - b );

    //// buff_A22 = & buff_A[ min_untyped( m_A - 1, j + b ) + 
    ////                      min_untyped( n_A - 1, j + b ) * ldim_A ];
    m_A22 = max_untyped( 0, m_A - j - b );
    //// n_A22 = max_untyped( 0, n_A - j - b );

    buff_Y2 = & buff_Y[ 0 + min_untyped( n_Y - 1, j + b ) * ldim_Y ];
    buff_G1 = & buff_G[ 0 + j * ldim_G ];
    buff_G2 = & buff_G[ 0 + min_untyped( n_G - 1, j + b ) * ldim_G ];
      
#ifdef CHECK_DOWNDATING_OF_Y
    // Check downdating of matrix Y: Compare downdated matrix Y with 
    // matrix Y computed from scratch.
    int64_t     m_cyr, n_cyr, ldim_cyr, m_ABR, ii, jj;
    double  * buff_cyr, aux, sum;

    m_cyr    = m_Y;
    n_cyr    = n_Y - j;
    ldim_cyr = m_cyr;
    m_ABR    = m_A - j;
    buff_cyr = ( double * ) malloc( m_cyr * n_cyr * sizeof( double ) );
 
    //// FLA_Gemm( FLA_NO_TRANSPOSE, FLA_NO_TRANSPOSE, 
    ////           FLA_ONE, GR, ABR, FLA_ZERO, CYR ); 
    blas::gemm(blas::Layout::ColMajor,
               blas::Op::NoTrans, blas::Op::NoTrans, m_cyr, n_cyr, m_ABR,
               d_one, & buff_G[ 0 + j * ldim_G ], ldim_G,
                      & buff_A[ j + j * ldim_A ], ldim_A,
               d_zero, & buff_cyr[ 0 + 0 * ldim_cyr ], ldim_cyr );

    //// print_double_matrix( "cyr", m_cyr, n_cyr, buff_cyr, ldim_cyr );
    //// print_double_matrix( "y", m_Y, n_Y, buff_Y, ldim_Y );
    sum = 0.0;
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

    if( last_iter == 0 ) {
      // Compute QRP of YR, and apply permutations to matrix AR.
      // A copy of YR is made into VR, and permutations are applied to YR.
      lapack::lacpy( lapack::MatrixType::General,
                     m_V, n_VR,
                     buff_YR, ldim_Y,
                     buff_VR, ldim_V);
      NoFLA_QRPmod_WY_unb_var4( 1, b,
          m_V, n_VR,
          buff_VR, ldim_V,
          buff_pB, buff_sB,
          1, m_A, buff_AR, ldim_A,
          1, m_Y, buff_YR, ldim_Y,
          0, buff_Y, ldim_Y );
    }

    //
    // Compute QRP of panel AB1 = [ A11; A21 ].
    // Apply same permutations to A01 and Y1, and build T1_T.
    //

    NoFLA_QRPmod_WY_unb_var4( panel_pivoting, -1,
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
      //   / A12 \ := QB1' / A12 \
      //   \ A22 /         \ A22 /
      // where QB1 is formed from AB1 and T1_T.
      NoFLA_Apply_Q_WY_lhfc_blk_var4( 
          m_A11 + m_A21, n_A11, buff_A11, ldim_A,
          b, b, buff_T1_T, ldim_W,
          m_A12 + m_A22, n_A12, buff_A12, ldim_A );
    }

    //
    // Downdate matrix Y.
    //
    if ( ! last_iter ) {
      NoFLA_Downdate_Y(
          m_A11, n_A11, buff_A11, ldim_A,
          m_A21, n_A21, buff_A21, ldim_A,
          m_A12, n_A12, buff_A12, ldim_A,
          b, b, buff_T1_T, ldim_T1_T,
          m_Y, max_untyped( 0, n_Y - j - b ), buff_Y2, ldim_Y,
          m_G, b, buff_G1, ldim_G,
          m_G, max_untyped( 0, n_G - j - b ), buff_G2, ldim_G );
    }
  }

  // Remove auxiliary objects.
  free( buff_G );
  free( buff_Y );
  free( buff_V );
  free( buff_W );

  return 0;
}


// ============================================================================
static int64_t NoFLA_Normal_random_matrix( int64_t m_A, int64_t n_A, 
               double * buff_A, int64_t ldim_A ) {
//
// It generates a random matrix with normal distribution.
//
  int64_t  i, j;

  // Main loop.
  for ( j = 0; j < n_A; j++ ) {
    for ( i = 0; i < m_A; i++ ) {
      buff_A[ i + j * ldim_A ] = NoFLA_Normal_random_number( 0.0, 1.0 );
    }
  }

  return 0;
}

/* ========================================================================= */
static double NoFLA_Normal_random_number( double mu, double sigma ) {
  static int64_t     alternate_calls = 0;
  static double  b1, b2;
  double         c1, c2, a, factor;

  // Quick return.
  if( alternate_calls == 1 ) {
    alternate_calls = ! alternate_calls;
    return( mu + sigma * b2 );
  }
  // Main loop.
  do {
    c1 = -1.0 + 2.0 * ( (double) rand() / RAND_MAX );
    c2 = -1.0 + 2.0 * ( (double) rand() / RAND_MAX );
    a = c1 * c1 + c2 * c2;
  } while ( ( a == 0 )||( a >= 1 ) );
  factor = sqrt( ( -2 * log( a ) ) / a );
  b1 = c1 * factor;
  b2 = c2 * factor;
  alternate_calls = ! alternate_calls;
  return( mu + sigma * b1 );
}

// ============================================================================
static int64_t NoFLA_Downdate_Y( 
               int64_t m_U11, int64_t n_U11, double * buff_U11, int64_t ldim_U11,
               int64_t m_U21, int64_t n_U21, double * buff_U21, int64_t ldim_U21,
               int64_t m_A12, int64_t n_A12, double * buff_A12, int64_t ldim_A12,
               int64_t m_T, int64_t n_T, double * buff_T, int64_t ldim_T,
               int64_t m_Y2, int64_t n_Y2, double * buff_Y2, int64_t ldim_Y2,
               int64_t m_G1, int64_t n_G1, double * buff_G1, int64_t ldim_G1,
               int64_t m_G2, int64_t n_G2, double * buff_G2, int64_t ldim_G2 ) {
//
// It downdates matrix Y, and updates matrix G.
// Only Y2 of Y is updated.
// Only G1 and G2 of G are updated.
//
// Y2 = Y2 - ( G1 - ( G1*U11 + G2*U21 ) * T11 * U11' ) * R12.
//
  int64_t    i, j;
  double * buff_B;
  double d_one       = 1.0;
  double d_minus_one = -1.0;
  int64_t    m_B         = m_G1;
  int64_t    n_B         = n_G1;
  int64_t    ldim_B      = m_G1;

  // Create object B.
  buff_B = ( double * ) malloc( m_B * n_B * sizeof( double ) );

  // B = G1.
  lapack::lacpy( lapack::MatrixType::General,
                 m_G1, n_G1,
                 buff_G1, ldim_G1,
                 buff_B, ldim_B )  ;

  // B = B * U11.
  blas::trmm( blas::Layout::ColMajor,
              blas::Side::Right, 
              blas::Uplo::Lower,
              blas::Op::NoTrans,
              blas::Diag::Unit, m_B, n_B,
              d_one, buff_U11, ldim_U11, buff_B, ldim_B );

  // B = B + G2 * U21.
  blas::gemm( blas::Layout::ColMajor,
              blas::Op::NoTrans, blas::Op::NoTrans, m_B, n_B, m_U21,
              d_one, buff_G2, ldim_G2, buff_U21, ldim_U21,
              d_one, buff_B,  ldim_B );

  // B = B * T11.
  blas::trmm( blas::Layout::ColMajor,
              blas::Side::Right,
              blas::Uplo::Upper,
              blas::Op::NoTrans,
              blas::Diag::NonUnit, m_B, n_B,
              d_one, buff_T, ldim_T, buff_B, ldim_B );

  // B = - B * U11^H.
  blas::trmm( blas::Layout::ColMajor,
              blas::Side::Right,
              blas::Uplo::Lower,
              blas::Op::ConjTrans,
              blas::Diag::Unit, m_B, n_B,
              d_minus_one, buff_U11, ldim_U11, buff_B, ldim_B );

  // B = G1 + B.
  for( j = 0; j < n_B; j++ ) {
    for( i = 0; i < m_B; i++ ) {
      buff_B[ i + j * ldim_B ] += buff_G1[ i + j * ldim_G1 ];
    }
  }

  // Y2 = Y2 - B * R12.
  blas::gemm( blas::Layout::ColMajor,
              blas::Op::NoTrans,
              blas::Op::NoTrans, m_Y2, n_Y2, m_A12,
              d_minus_one, buff_B, ldim_B, buff_A12, ldim_A12,
              d_one, buff_Y2, ldim_Y2 );

  //
  // GR = GR * Q
  //
  NoFLA_Apply_Q_WY_rnfc_blk_var4( 
          m_U11 + m_U21, n_U11, buff_U11, ldim_U11,
          m_T, n_T, buff_T, ldim_T,
          m_G1, n_G1 + n_G2, buff_G1, ldim_G1 );

  // Remove object B.
  free( buff_B );

  return 0;
}

// ============================================================================
static int64_t NoFLA_Apply_Q_WY_lhfc_blk_var4( 
               int64_t m_U, int64_t n_U, double * buff_U, int64_t ldim_U,
               int64_t m_T, int64_t n_T, double * buff_T, int64_t ldim_T,
               int64_t m_B, int64_t n_B, double * buff_B, int64_t ldim_B ) {
//
// It applies the transpose of a block transformation Q to a matrix B from 
// the left:
//   B := Q' * B
// where:
//   Q = I - U * T' * U'.
//
  double  * buff_W;
  int64_t     ldim_W;

  // Create auxiliary object.
  //// FLA_Obj_create_conf_to( FLA_NO_TRANSPOSE, B1, & W );
  buff_W = ( double * ) malloc( n_B * n_U * sizeof( double ) );
  ldim_W = max_untyped( 1, n_B );
 
  // Apply the block transformation.
  _LAPACK_dlafrb(lapack::Side::Left, lapack::Op::Trans,
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
static int64_t NoFLA_Apply_Q_WY_rnfc_blk_var4( 
               int64_t m_U, int64_t n_U, double * buff_U, int64_t ldim_U,
               int64_t m_T, int64_t n_T, double * buff_T, int64_t ldim_T,
               int64_t m_B, int64_t n_B, double * buff_B, int64_t ldim_B ) {
//
// It applies a block transformation Q to a matrix B from the right:
//   B = B * Q
// where:
//   Q = I - U * T' * U'.
//
  double  * buff_W;
  int64_t   ldim_W;

  // Create auxiliary object.
  //// FLA_Obj_create_conf_to( FLA_TRANSPOSE, B1, & W );
  buff_W = ( double * ) malloc( m_B * n_U * sizeof( double ) );
  ldim_W = max_untyped( 1, m_B );
  
  // Apply the block transformation. 
  _LAPACK_dlafrb(lapack::Side::Right, lapack::Op::NoTrans,
        lapack::Direction::Forward, lapack::StoreV::Columnwise,
        m_B, n_B, n_U, buff_U, ldim_U, buff_T, ldim_T,
        buff_B, ldim_B, buff_W, ldim_W
  );

  // Remove auxiliary object.
  free( buff_W );

  return 0;
}

// ============================================================================
static int64_t NoFLA_QRPmod_WY_unb_var4( int64_t pivoting, int64_t num_stages, 
               int64_t m_A, int64_t n_A, double * buff_A, int64_t ldim_A,
               int64_t * buff_p, double * buff_t, 
               int64_t pivot_B, int64_t m_B, double * buff_B, int64_t ldim_B,
               int64_t pivot_C, int64_t m_C, double * buff_C, int64_t ldim_C,
               int64_t build_T, double * buff_T, int64_t ldim_T ) {
//
// It computes an unblocked QR factorization of matrix A with or without 
// pivoting. Matrices B and C are optionally pivoted, and matrix T is
// optionally built.
//
// Arguments:
// "pivoting": If pivoting==1, then QR factorization with pivoting is used.
// "numstages": It tells the number of columns that are factorized.
//   If "num_stages" is negative, the whole matrix A is factorized.
//   If "num_stages" is positive, only the first "num_stages" are factorized.
// "pivot_B": if "pivot_B" is true, matrix "B" is pivoted too.
// "pivot_C": if "pivot_C" is true, matrix "C" is pivoted too.
// "build_T": if "build_T" is true, matrix "T" is built.
//
  int64_t     j, mn_A, m_a21, m_A22, n_A22, n_dB, idx_max_col, 
          i_one = 1, n_house_vector, m_rest;
  double  * buff_d, * buff_e, * buff_workspace, diag;

  //// printf( "NoFLA_QRPmod_WY_unb_var4. pivoting: %d \n", pivoting );

  // Some initializations.
  mn_A    = min_untyped( m_A, n_A );

  // Set the number of stages, if needed.
  if( num_stages < 0 ) {
    num_stages = mn_A;
  }

  // Create auxiliary vectors.
  buff_d         = ( double * ) malloc( n_A * sizeof( double ) );
  buff_e         = ( double * ) malloc( n_A * sizeof( double ) );
  buff_workspace = ( double * ) malloc( n_A * sizeof( double ) );

  if( pivoting == 1 ) {
    // Compute initial norms of A int64_to d and e.
    NoFLA_QRP_compute_norms( m_A, n_A, buff_A, ldim_A, buff_d, buff_e );
  }

  // Main Loop.
  for( j = 0; j < num_stages; j++ ) {
    n_dB  = n_A - j;
    m_a21 = m_A - j - 1;
    m_A22 = m_A - j - 1;
    n_A22 = n_A - j - 1;

    if( pivoting == 1 ) {
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
    }

    // Compute tau1 and u21 from alpha11 and a21 such that tau1 and u21
    // determine a Householder transform H such that applying H from the
    // left to the column vector consisting of alpha11 and a21 annihilates
    // the entries in a21 (and updates alpha11).
    n_house_vector = m_a21 + 1;
    lapack::larfg(n_house_vector,
        & buff_A[ j + j * ldim_A ],
        & buff_A[ min_untyped( m_A-1, j+1 ) + j * ldim_A ],
        i_one,
        & buff_t[j]
    );

    // / a12t \ =  H / a12t \
    // \ A22  /      \ A22  /
    //
    // where H is formed from tau1 and u21.
    diag = buff_A[ j + j * ldim_A ];
    buff_A[ j + j * ldim_A ] = 1.0;
    m_rest = m_A22 + 1;
    _LAPACK_dlarf( lapack::Side::Left, m_rest, n_A22, 
        & buff_A[ j + j * ldim_A ], 1,
        buff_t[ j ],
        & buff_A[ j + ( j+1 ) * ldim_A ], ldim_A,
        buff_workspace
    );
    buff_A[ j + j * ldim_A ] = diag;

    if( pivoting == 1 ) {
      // Update partial column norms.
      NoFLA_QRP_downdate_partial_norms( m_A22, n_A22, 
          & buff_d[ j+1 ], 1,
          & buff_e[ j+1 ], 1,
          & buff_A[ j + ( j+1 ) * ldim_A ], ldim_A,
          & buff_A[ ( j+1 ) + min_untyped( n_A-1, ( j+1 ) ) * ldim_A ], ldim_A );
    }
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

// ============================================================================
static int64_t NoFLA_QRP_compute_norms(
               int64_t m_A, int64_t n_A, double * buff_A, int64_t ldim_A,
               double * buff_d, double * buff_e ) {
//
// It computes the column norms of matrix A. The norms are stored int64_to 
// vectors d and e.
//
  int64_t     j, i_one = 1;

  // Main loop.
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
static int64_t NoFLA_QRP_downdate_partial_norms( int64_t m_A, int64_t n_A,
               double * buff_d,  int64_t st_d,
               double * buff_e,  int64_t st_e,
               double * buff_wt, int64_t st_wt,
               double * buff_A,  int64_t ldim_A ) {
//
// It updates (downdates) the column norms of matrix A. It uses Drmac's method.
//
  int64_t     j, i_one = 1;
  double  * ptr_d, * ptr_e, * ptr_wt, * ptr_A;
  double  temp, temp2, temp5, tol3z;
  // double  dnrm2_(), dlamch_();

  /*
*
*           Update partial column norms
*
          DO 30 J = I + 1, N
             IF( WORK( J ).NE.ZERO ) THEN
*
*                 NOTE: The following 4 lines follow from the analysis in
*                 Lapack Working Note 176.
*                 
                TEMP = ABS( A( I, J ) ) / WORK( J )
                TEMP = MAX( ZERO, ( ONE+TEMP )*( ONE-TEMP ) )
                TEMP2 = TEMP*( WORK( J ) / WORK( N+J ) )**2
                IF( TEMP2 .LE. TOL3Z ) THEN 
                   IF( M-I.GT.0 ) THEN
                      WORK( J ) = DNRM2( M-I, A( I+1, J ), 1 )
                      WORK( N+J ) = WORK( J )
                   ELSE
                      WORK( J ) = ZERO
                      WORK( N+J ) = ZERO
                   END IF
                ELSE
                   WORK( J ) = WORK( J )*SQRT( TEMP )
                END IF
             END IF
 30       CONTINUE
  */

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
  for( j = 0; j < n_A; j++ ) {
    if( * ptr_d != 0.0 ) {
      temp = dabs( * ptr_wt ) / * ptr_d;
      temp = max_untyped( 0.0, ( 1.0 + temp ) * ( 1 - temp ) );
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
static int64_t NoFLA_QRP_pivot_G_B_C( int64_t j_max_col,
               int64_t m_G, double * buff_G, int64_t ldim_G, 
               int64_t pivot_B, int64_t m_B, double * buff_B, int64_t ldim_B, 
               int64_t pivot_C, int64_t m_C, double * buff_C, int64_t ldim_C, 
               int64_t * buff_p,
               double * buff_d, double * buff_e ) {
//
// It pivots matrix G, pivot vector p, and norms vectors d and e.
// Matrices B and C are optionally pivoted.
//
  int64_t     ival, i_one = 1;
  double  * ptr_g1, * ptr_g2, * ptr_b1, * ptr_b2, * ptr_c1, * ptr_c2;

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

} // end namespace RandLAPACK
#endif
