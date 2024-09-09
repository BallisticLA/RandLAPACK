#pragma once

#ifndef randlapack_cqrrp_h
#define randlapack_cqrrp_h

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
class CQRRPalg {
    public:

        virtual ~CQRRPalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T d_factor,
            T* tau,
            int64_t* J,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class CQRRP_blocked : public CQRRPalg<T, RNG> {
    public:

        /// This algorithm serves as an extension of CQRRPT's idea - CQRRP presents a blocked version of
        /// randomized QR with column pivoting that utilizes Cholesky QR.
        ///
        /// The base structure of CQRRP resembles that of Algorithm 4 from https://arxiv.org/abs/1509.06820. 
        /// CQRRP allows for error-tolerance based adaptive stopping criteria, taken from Section 3 of 
        /// https://arxiv.org/abs/1606.09402.
        ///
        /// The main computational bottlenecks of CQRRP are in its following two components:
        ///     1. Performing QRCP on a sketch - in our case, is implemented via pivoted LU (see below for details).
        ///     2. Applying Q-factor from Cholesky QR to the working area of matrix A (done via gemqrt).
        ///
        /// The algorithm optionally times all of its subcomponents through a user-defined 'timing' parameter.


        CQRRP_blocked(
            bool time_subroutines,
            T ep,
            int64_t b_sz
        ) {
            timing     = time_subroutines;
            eps        = ep;
            timing     = time_subroutines;
            eps        = ep;
            block_size = b_sz;
            use_qp3      = false;
            use_gemqrt   = false;
            internal_nb  = b_sz;
            tol = std::numeric_limits<T>::epsilon();
            use_qp3      = false;
            use_gemqrt   = false;
            internal_nb  = b_sz;
            tol = std::numeric_limits<T>::epsilon();
        }

        /// Computes a QR factorization with column pivots of the form:
        ///     A[:, J] = QR,
        /// where Q and R are of size m-by-k and k-by-n, with rank(A) = k.
        /// Stores implict Q factor and explicit R factor in A's space (output formatted exactly like GEQP3).
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
        /// @param[in] d_factor
        ///     Embedding dimension of a sketch factor, m >= (d_factor * n) >= n.
        ///
        /// @param[in] tau
        ///     Pointer to a vector of size n. On entry, is empty.
        ///
        /// @param[in] state
        ///     RNG state parameter, required for sketching operator generation.
        ///
        /// @param[out] A
        ///     Overwritten by Implicit Q and explicit R factors.
        ///
        /// @param[out] tau
        ///     On output, similar in format to that in GEQP3.
        ///
        /// @param[out] J
        ///     Stores k integer type pivot index extries.
        ///
        /// @return = 0: successful exit
        ///

        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T d_factor,
            T* tau,
            int64_t* J,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        bool timing;
        bool cond_check;
        RandBLAS::RNGState<RNG> state;
        T eps;
        int64_t rank;
        int64_t block_size;

        // 12 entries - logs time for different portions of the algorithm
        std::vector<long> times;

        // QRCP option
        bool use_qp3;

        // Option for updating A
        bool use_gemqrt;

        // NB for orhr_col and gemqrt
        int64_t internal_nb;

        // Naive rank estimation parameter;
        T tol;
};

// We are assuming that tau and J have been pre-allocated
// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRP_blocked<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T d_factor,
    T* tau,
    int64_t* J,
    RandBLAS::RNGState<RNG> &state
){
    #ifdef __APPLE__
    UNUSED(m); UNUSED(n); UNUSED(A); UNUSED(lda); UNUSED(d_factor); UNUSED(tau); UNUSED(J); UNUSED(state);
    throw std::runtime_error("CQRRP is not supported when BLAS is linked against Apple Accelerate.");
    #else
    //-------TIMING VARS--------/
    high_resolution_clock::time_point preallocation_t_stop;
    high_resolution_clock::time_point preallocation_t_start;
    high_resolution_clock::time_point skop_t_stop;
    high_resolution_clock::time_point skop_t_start;
    high_resolution_clock::time_point qrcp_t_start;
    high_resolution_clock::time_point qrcp_t_stop;
    high_resolution_clock::time_point cholqr_t_start;
    high_resolution_clock::time_point cholqr_t_stop;
    high_resolution_clock::time_point reconstruction_t_start;
    high_resolution_clock::time_point reconstruction_t_stop;
    high_resolution_clock::time_point preconditioning_t_start;
    high_resolution_clock::time_point preconditioning_t_stop;
    high_resolution_clock::time_point r_piv_t_start;
    high_resolution_clock::time_point r_piv_t_stop;
    high_resolution_clock::time_point updating1_t_start;
    high_resolution_clock::time_point updating1_t_stop;
    high_resolution_clock::time_point updating2_t_start;
    high_resolution_clock::time_point updating2_t_stop;
    high_resolution_clock::time_point updating3_t_start;
    high_resolution_clock::time_point updating3_t_stop;
    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;
    long preallocation_t_dur   = 0;
    long skop_t_dur            = 0;
    long qrcp_t_dur            = 0;
    long cholqr_t_dur          = 0;
    long reconstruction_t_dur  = 0;
    long preconditioning_t_dur = 0;
    long r_piv_t_dur           = 0;
    long updating1_t_dur       = 0;
    long updating2_t_dur       = 0;
    long updating3_t_dur       = 0;
    long total_t_dur           = 0;

    if(this -> timing) {
        total_t_start = high_resolution_clock::now();
        preallocation_t_start = high_resolution_clock::now();
    }
    int iter, i, j;
    int64_t tmp;
    int64_t rows       = m;
    int64_t cols       = n;
    // Describes sizes of full Q and R factors at a given iteration.
    int64_t curr_sz    = 0;
    int64_t b_sz       = this->block_size;
    int64_t maxiter    = (int64_t) std::ceil(std::min(m, n) / (T) b_sz);
    // Using this variable to work with matrices with leading dimension = b_sz.
    int64_t b_sz_const = b_sz;
    // This will serve as lda of a sketch
    int64_t d          = d_factor * b_sz;
    // We will be using this parameter when performing QRCP on a sketch.
    // After the first iteration of the algorithm, this will change its value to min(d, cols) 
    // before "cols" is updated.
    int64_t sampling_dimension = d;
    // An indicator for whether all entries in a given block are zero.
    bool block_zero = true;
    // Rank of a block at a given iteration. If it changes, algorithm would iterate at the given iteration, 
    // since the rest of the matrx must be zero.
    // Is equal to block size by default, needs to be upated if the block size has changed.
    int64_t block_rank = b_sz;
    // Parameter to control number of blocks in orhr_col and gemqrt;
    int64_t internal_nb = this -> internal_nb;

    //*********************************POINTERS TO A BEGIN*********************************
    // LDA for all of the below is m

    // Pointer to the beginning of the original space of A.
    // Pointer to the beginning of A's "work zone," 
    // will shift at every iteration of an algorithm by (lda * b_sz) + b_sz.
    T* A_work = A;
    // Workspace 1 pointer - will serve as a buffer for computing R12 and updated matrix A.
    // Points to a location, offset by lda * b_sz from the current "A_work."
    T* Work1  = NULL;
    // Points to R11 factor, right above the compact Q, of size b_sz by b_sz.
    T* R11    = NULL;
    // Points to R12 factor, to the right of R11 and above Work1 of size b_sz by n - curr_sz - b_sz.
    T* R12    = NULL;
    //**********************************POINTERS TO A END**********************************

    //*********************************POINTERS TO OTHER BEGIN*********************************
    // Pointer to the portion of vector tau at current iteration.
    T* tau_sub = NULL;
    //**********************************POINTERS TO OTHER END**********************************

    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE BEGIN*******************
    // BELOW ARE MATRICES THAT WE CANNOT PUT INTO COMMON BUFFERS

    // J_buffer serves as a buffer for the pivots found at every iteration, of size n.
    // At every iteration, it would only hold "cols" entries.
    // Cannot really fully switch this to pointers bc we do not want data to be modified in "col_swap."
    std::vector<int64_t> J_buf (n, 0);
    int64_t* J_buffer = J_buf.data();
    // Special pivoting buffer for LU factorization, capturing the swaps on A_sk'.
    // Needs to be converted in a proper format of length rows(A_sk')
    int64_t* J_buffer_lu = ( int64_t * ) calloc( std::min(d, n), sizeof( int64_t ) );

    // A_sk serves as a skething matrix, of size d by n, lda d
    // Below algorithm does not perform repeated sampling, hence A_sk
    // is updated at the end of every iteration.
    // Should remain unchanged throughout the algorithm,
    // As the algorithm needs to have access to the upper-triangular factor R
    // (stored in this matrix after geqp3) at all times. 
    T* A_sk = ( T * ) calloc( d * n, sizeof( T ) );
    // Create a separate pointer to free when function terminates
    T* A_sk_const = A_sk;
    // Pointer to the b_sz by b_sz upper-triangular facor R stored in A_sk after GEQP3.
    T* R_sk = NULL;
    // View to the transpose of A_sk.
    // Is of size n * d, with an lda n.
    T* A_sk_trans = ( T * ) calloc( n * d, sizeof( T ) );

    // Buffer for the R-factor in Cholesky QR, of size b_sz by b_sz, lda b_sz.
    // Also used to store the proper R11_full-factor after the 
    // full Q has been restored form economy Q (the has been found via Cholesky QR);
    // That is done by applying the sign vector D from orhr_col().
    // Eventually, will be used to store R11 (computed via trmm)
    // which is then copied into its appropriate space in the matrix A.
    T* R_cholqr = ( T * ) calloc( b_sz_const * b_sz_const, sizeof( T ) );
    // Pointer to matrix T from orhr_col at currect iteration, will point to Work2 space.
    T* T_dat    = ( T * ) calloc( b_sz_const * b_sz_const, sizeof( T ) );

    // Buffer for Tau in GEQP3 and D in orhr_col, of size n.
    T* Work2    = ( T * ) calloc( n, sizeof( T ) );
    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE END*******************

    if(this -> timing) {
        preallocation_t_stop  = high_resolution_clock::now();
        preallocation_t_dur   = duration_cast<microseconds>(preallocation_t_stop - preallocation_t_start).count();
        skop_t_start = high_resolution_clock::now();
    }

    // Using Gaussian matrix as a sketching operator.
    // Using a sparse sketching operator may be dangerous if LU-based QRCP is in use,
    // as LU is not intended to be used with rank-deficient matrices.
    T* S  = ( T * ) calloc( d * m, sizeof( T ) );
    RandBLAS::DenseDist D(d, m);
    state = RandBLAS::fill_dense(D, S, state).second;
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, S, d, A, m, 0.0, A_sk, d);
    free(S);

    if(this -> timing) {
        skop_t_stop  = high_resolution_clock::now();
        skop_t_dur   = duration_cast<microseconds>(skop_t_stop - skop_t_start).count();
    }

    for(iter = 0; iter < maxiter; ++iter) {
        // Make sure we fit into the available space
        b_sz = std::min(b_sz, std::min(m, n) - curr_sz);
        internal_nb = std::min(internal_nb, b_sz);
        block_rank = b_sz;

        // Zero-out data - may not be necessary
        std::fill(&J_buffer[0], &J_buffer[n], 0);
        std::fill(&J_buffer_lu[0], &J_buffer_lu[std::min(d, n)], 0);
        std::fill(&Work2[0], &Work2[n], (T) 0.0);

        if(this -> timing)
            qrcp_t_start = high_resolution_clock::now();
            
        if (this -> use_qp3) {
            lapack::geqp3(sampling_dimension, cols, A_sk, d, J_buffer, Work2);
        } else {
            // Perform pivoted LU on A_sk', follow it up by unpivoted QR on a permuted A_sk.
            // Get a transpose of A_sk 
            util::transposition(sampling_dimension, cols, A_sk, d, A_sk_trans, n, 0);
            
            // Perform a row-pivoted LU on a transpose of A_sk
            lapack::getrf(cols, sampling_dimension, A_sk_trans, n, J_buffer_lu);
            // Fill the pivot vector, apply swaps found via lu on A_sk'.
            std::iota(&J_buffer[0], &J_buffer[cols], 1);

            for (i = 0; i < std::min(sampling_dimension, cols); ++i) {
                tmp = J_buffer[J_buffer_lu[i] - 1];
                J_buffer[J_buffer_lu[i] - 1] = J_buffer[i];
                J_buffer[i] = tmp;
            }
            // Apply pivots to A_sk
            util::col_swap(sampling_dimension, cols, cols, A_sk, d, J_buf);
            // Perform an unpivoted QR on A_sk
            lapack::geqrf(sampling_dimension, cols, A_sk, d, Work2);
        }

        if(this -> timing) {
            qrcp_t_stop = high_resolution_clock::now();
            qrcp_t_dur += duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
            r_piv_t_start = high_resolution_clock::now();
        }

        // Need to premute trailing columns of the full R-factor.
        // Remember that the R-factor is stored the upper-triangular portion of A.
        if(iter != 0)
            util::col_swap(curr_sz, cols, cols, &A[lda * curr_sz], m, J_buf);

        if(this -> timing) {
            r_piv_t_stop  = high_resolution_clock::now();
            r_piv_t_dur  += duration_cast<microseconds>(r_piv_t_stop - r_piv_t_start).count();
            preconditioning_t_start = high_resolution_clock::now();
        }

        // Pivoting the current matrix A.
        util::col_swap(rows, cols, cols, A_work, lda, J_buf);

        // Checking for the zero matrix post-pivoting is the best idea, 
        // as we would only need to check one column (pivoting moves the column with the largest norm upfront)
        block_zero = true;
        for (i = 0; i < rows; ++i) {
            if (std::abs(A_work[i]) > std::numeric_limits<T>::epsilon()) {
                block_zero = false;
                break;
            }
        }
        if(block_zero){
            // Zero leftover matrix, early termination
            this -> rank = curr_sz;

            // Updating pivots
            if(iter == 0) {
                blas::copy(cols, J_buffer, 1, J, 1);
            } else {
                RandLAPACK::util::col_swap<T>(cols, cols, &J[curr_sz], J_buf);
            }

            free(J_buffer_lu);
            free(A_sk_const);
            free(A_sk_trans);
            free(R_cholqr);
            free(T_dat);
            free(Work2);
            return 0;
        }

        // Defining the new "working subportion" of matrix A.
        // In a global sense, below is identical to:
        // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz];
        Work1 = &A_work[lda * b_sz];

        // Define the space representing R_sk (stored in A_sk)
        R_sk = A_sk;

        // Naive rank estimation to perform preconditioning safely.
        // Variable block_rank is altered if the rank is not full.
        // If this happens, we will terminate at the end of the current iteration.
        // If the internal_nb, used in gemqrt and orhr_col is larger than the updated block_rank, it would need to be updated as well.
        // Updating block_rank affects the way the preconditioning is done, which, in its turn, affects CholQR, ORHR_COL, updating A and updating R.
        for(i = 0; i < b_sz; ++i) {
            if(std::abs(R_sk[i * d + i]) / std::abs(R_sk[0]) < this -> tol) {
                block_rank = i;
                internal_nb = std::min(internal_nb, block_rank);
                break;
            }
        }
        
        // A_pre = AJ(:, 1:rank_b_sz) * inv(R_sk)
        // Performing preconditioning of the current matrix A.
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, block_rank, (T) 1.0, R_sk, d, A_work, lda);

        if(this -> timing) {
            preconditioning_t_stop  = high_resolution_clock::now();
            preconditioning_t_dur  += duration_cast<microseconds>(preconditioning_t_stop - preconditioning_t_start).count();
        }

        if(this -> timing)
            cholqr_t_start = high_resolution_clock::now();

        // Performing Cholesky QR
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, block_rank, rows, (T) 1.0, A_work, lda, (T) 0.0, R_cholqr, b_sz_const);
        lapack::potrf(Uplo::Upper, block_rank, R_cholqr, b_sz_const);
        // Compute Q_econ from Cholesky QR
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, block_rank, (T) 1.0, R_cholqr, b_sz_const, A_work, lda);

        if(this -> timing) {
            cholqr_t_stop  = high_resolution_clock::now();
            cholqr_t_dur  += duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();
            reconstruction_t_start = high_resolution_clock::now();
        }

        // Find Q (stored in A) using Householder reconstruction. 
        // This will represent the full (rows by rows) Q factor form Cholesky QR
        // It would have been really nice to store T right above Q, but without using extra space,
        // it would result in us loosing the first lower-triangular b_sz by b_sz portion of implicitly-stored Q.
        // Filling T without ever touching its lower-triangular space would be a nice optimization for orhr_col routine.
        // Q is defined with block_rank elementary reflectors. 
        // NOTE:
        ///     This routine is defined in LAPACK 3.9.0.
        lapack::orhr_col(rows, block_rank, internal_nb, A_work, lda, T_dat, b_sz_const, Work2);

        // Need to change signs in the R-factor from Cholesky QR.
        // Signs correspond to matrix D from orhr_col().
        // This allows us to not explicitoly compute R11_full = (Q[:, 1:block_rank])' * A_pre.
        for(i = 0; i < block_rank; ++i)
            for(j = 0; j < (i + 1); ++j)
               R_cholqr[(b_sz_const * i) + j] *= Work2[j];

        // Define a pointer to the current subportion of tau vector.
        tau_sub = &tau[curr_sz];
        // Entries of tau will be placed on the main diagonal of the block matrix T from orhr_col().
        for(i = 0; i < block_rank; ++i)
            tau_sub[i] = T_dat[(b_sz_const * i) + (i % internal_nb)];

        if(this -> timing) {
            reconstruction_t_stop  = high_resolution_clock::now();
            reconstruction_t_dur  += duration_cast<microseconds>(reconstruction_t_stop - reconstruction_t_start).count();
            updating1_t_start = high_resolution_clock::now();
        }

        // Perform Q_full' * A_piv(:, b_sz:end) to find R12 and the new "current A."
        // A_piv (Work1) is a rows by cols - b_sz matrix, stored in space of the original A.
        // The first b_sz rows will represent R12.
        // The last rows-b_sz rows will represent the new A.
        // With that, everything is placed where it should be, no copies required.
        // Q is defined with block_rank elementary reflectors. 
        // GEMQRT is a faster alternative to ORMQR, takes in the matrix T instead of vector tau.
        if(use_gemqrt) {
            lapack::gemqrt(Side::Left, Op::Trans, rows, cols - b_sz, block_rank, internal_nb, A_work, lda, T_dat, b_sz_const, Work1, lda);
        } else {
            lapack::ormqr(Side::Left, Op::Trans, rows, cols - b_sz, block_rank, A_work, lda, tau_sub, Work1, lda);
        }

        if(this -> timing) {
            updating1_t_stop  = high_resolution_clock::now();
            updating1_t_dur  += duration_cast<microseconds>(updating1_t_stop - updating1_t_start).count();
            updating3_t_start = high_resolution_clock::now();
        }

        // Updating pivots
        if(iter == 0) {
            blas::copy(cols, J_buffer, 1, J, 1);
        } else {
            RandLAPACK::util::col_swap<T>(cols, cols, &J[curr_sz], J_buf);
        }

        // Alternatively, instead of trmm + copy, we could perform a single gemm.
        // Compute R11 = R11_full(1:block_rank, :) * R_sk
        // R11_full is stored in R_cholqr space, R_sk is stored in A_sk space.
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, block_rank, block_rank, (T) 1.0, R_sk, d, R_cholqr, b_sz_const);

        // Need to copy R11 over form R_cholqr into the appropriate space in A.
        // We cannot avoid this copy, since trmm() assumes R_cholqr is a square matrix.
        // In a global sense, this is identical to:
        // R11 =  &A[(m + 1) * curr_sz];
        R11 = A_work;
        lapack::lacpy(MatrixType::Upper, block_rank, block_rank, R_cholqr, b_sz_const, A_work, lda);

        // Updating the pointer to R12
        // In a global sense, this is identical to:
        // R12 =  &A[(m * (curr_sz + b_sz)) + curr_sz];
        R12 = &R11[lda * b_sz];

        if(this -> timing) {
            updating3_t_stop  = high_resolution_clock::now();
            updating3_t_dur  += duration_cast<microseconds>(updating3_t_stop - updating3_t_start).count();
        }

        // Size of the factors is updated;
        curr_sz += b_sz;

        // Termination criteria is reached when:
        // 1. All iterations are exhausted.
        // 2. block_rank has been altered, which happens
        // when the estimated rank of the R-factor 
        // from QRCP at this iteration is not full,
        // meaning that the rest of the matrix is zero.
        if((curr_sz >= n) || (block_rank != b_sz_const)) {
            this -> rank = curr_sz;

            if(this -> timing) {
                total_t_stop = high_resolution_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest  = total_t_dur - (preallocation_t_dur + skop_t_dur + qrcp_t_dur + reconstruction_t_dur + preconditioning_t_dur + updating1_t_dur + updating2_t_dur + updating3_t_dur + r_piv_t_dur);
                this -> times.resize(12);
                this -> times = {skop_t_dur, preallocation_t_dur, qrcp_t_dur, preconditioning_t_dur, cholqr_t_dur, reconstruction_t_dur, updating1_t_dur, updating2_t_dur, updating3_t_dur, r_piv_t_dur, t_rest, total_t_dur};

                printf("\n\n/------------CQRRP TIMING RESULTS BEGIN------------/\n");
                printf("Preallocation time: %25ld μs,\n",                  preallocation_t_dur);
                printf("skop time: %34ld μs,\n",                           skop_t_dur);
                printf("QRCP time: %36ld μs,\n",                           qrcp_t_dur);
                printf("Preconditioning time: %24ld μs,\n",                preconditioning_t_dur);
                printf("CholQR time: %32ld μs,\n",                         cholqr_t_dur);
                printf("Householder vector restoration time: %7ld μs,\n",  reconstruction_t_dur);
                printf("Computing A_new, R12 time: %23ld μs,\n",           updating1_t_dur);
                printf("Factors updating time: %23ld μs,\n",               updating3_t_dur);
                printf("Sketch updating time: %24ld μs,\n",                updating2_t_dur);
                printf("Trailing cols(R) pivoting time: %10ld μs,\n",      r_piv_t_dur);
                printf("Other routines time: %24ld μs,\n",                 t_rest);
                printf("Total time: %35ld μs.\n",                          total_t_dur);

                printf("\nPreallocation takes %22.2f%% of runtime.\n",                  100 * ((T) preallocation_t_dur   / (T) total_t_dur));
                printf("skop generation and application takes %2.2f%% of runtime.\n",   100 * ((T) skop_t_dur            / (T) total_t_dur));
                printf("QRCP takes %32.2f%% of runtime.\n",                             100 * ((T) qrcp_t_dur            / (T) total_t_dur));
                printf("Preconditioning takes %20.2f%% of runtime.\n",                  100 * ((T) preconditioning_t_dur / (T) total_t_dur));
                printf("Cholqr takes %29.2f%% of runtime.\n",                           100 * ((T) cholqr_t_dur          / (T) total_t_dur));
                printf("Householder restoration takes %12.2f%% of runtime.\n",          100 * ((T) reconstruction_t_dur  / (T) total_t_dur));
                printf("Computing A_new, R12 takes %14.2f%% of runtime.\n",             100 * ((T) updating1_t_dur       / (T) total_t_dur));
                printf("Factors updating time takes %14.2f%% of runtime.\n",            100 * ((T) updating3_t_dur       / (T) total_t_dur));
                printf("Sketch updating time takes %15.2f%% of runtime.\n",             100 * ((T) updating2_t_dur       / (T) total_t_dur));
                printf("Trailing cols(R) pivoting takes %10.2f%% of runtime.\n",        100 * ((T) r_piv_t_dur           / (T) total_t_dur));
                printf("Everything else takes %20.2f%% of runtime.\n",                  100 * ((T) t_rest                / (T) total_t_dur));
                printf("/-------------CQRRP TIMING RESULTS END-------------/\n\n");
            }

            free(J_buffer_lu);
            free(A_sk_const);
            free(A_sk_trans);
            free(R_cholqr);
            free(T_dat);
            free(Work2);

            return 0;
        }

        if(this -> timing)
            updating2_t_start = high_resolution_clock::now();

        // Updating the pointer to "Current A."
        // In a global sense, below is identical to:
        // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz + b_sz];
        // Also, Below is identical to:
        // A_work = &A_work[(lda + 1) * b_sz];
        A_work = &Work1[b_sz];
        
        // Updating the skethcing buffer
        // trsm (R_sk, R11) -> R_sk
        // Clearing the lower-triangular portion here is necessary, if there is a more elegant way, need to use that.
        RandLAPACK::util::get_U(b_sz, b_sz, R_sk, d);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, b_sz, b_sz, (T) 1.0, R11, lda, R_sk, d);
        // R_sk_12 - R_sk_11 * inv(R_11) * R_12
        // Side note: might need to be careful when d = b_sz.
        // Cannot perform trmm here as an alternative, since matrix difference is involved.
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, cols - b_sz, b_sz, (T) -1.0, R_sk, d, R12, lda, (T) 1.0, &R_sk[d * b_sz], d);
        
        // Changing the sampling dimension parameter
        sampling_dimension = std::min(sampling_dimension, cols);

        // Need to zero out the lower triangular portion of R_sk_22
        // Make sure R_sk_22 exists.
        if (sampling_dimension - b_sz > 0)
            RandLAPACK::util::get_U(sampling_dimension - b_sz, sampling_dimension - b_sz, &R_sk[(d + 1) * b_sz], d);

        // Changing the pointer to relevant data in A_sk - this is equaivalent to copying data over to the beginning of A_sk.
        // Remember that the only "active" portion of A_sk remaining would be of size sampling_dimension by cols;
        // if any rows beyond that would be accessed, we would have issues. 
        A_sk = &A_sk[d * b_sz];

        if(this -> timing) {
            updating2_t_stop  = high_resolution_clock::now();
            updating2_t_dur  += duration_cast<microseconds>(updating2_t_stop - updating2_t_start).count();
        }

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;
    }
    #endif
    return 0;
}

} // end namespace RandLAPACK

#endif
