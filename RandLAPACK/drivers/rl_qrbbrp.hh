#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>


namespace RandLAPACK {

/**
 * This is an adaptation of BQRRP that accepts a callable function for qrcp_wide.
 * I removed support for Cholesky QR and GEQRT for the panel QR.
 * The callable_qrcp_wide_t needs to have reserve and free methods, where
 *      qrcp_wide.reserve(_a, _b) allocates space needed for QRCP of an (_a)-by-(_b) matrix, and
 *      qrcp_wide.free() deallocates space allocated in reserve.
 * We also need 
*/
template <typename T, typename callable_qrcp_wide_t, typename RNG = RandBLAS::DefaultRNG>
struct QRBBRP {

    using qrcp_wide_t = callable_qrcp_wide_t;
    using state_t = RandBLAS::RNGState<RNG>;

    qrcp_wide_t &qrcp_wide;
    bool timing;
    int64_t block_size;
    T d_factor;
    std::vector<long> times;


    QRBBRP(
        qrcp_wide_t &qrcp_wide,
        bool time_subroutines,
        int64_t b_sz,
        T d_factor
    ) : qrcp_wide(qrcp_wide), timing(time_subroutines), block_size(b_sz), d_factor(d_factor) { }

    /// Computes a pivoted QR decomposition in a format compatible with GEQP3.
    /// The pivot decisions are made by applying this object's qrcp_wide function
    /// to a sketch of the input matrix.
    ///
    /// @param[in] m
    ///     The number of rows in the matrix A.
    ///
    /// @param[in] n
    ///     The number of columns in the matrix A.
    ///
    /// @param[in, out] A
    ///     On entry, pointer to the m-by-n matrix A, stored in a column-major format.
    ///     On exit, overwritten by implicit Q and explicit R factors.
    ///
    /// @param[in] lda
    ///     Leading dimension of A.
    ///
    /// @param[out] J
    ///     Stores GEQP3-compatible pivot vector indices.
    ///
    /// @param[out] tau
    ///     On output, similar in format to that in GEQP3.
    ///
    /// @param[in] state
    ///     RNGState used in sketching operator generation.
    ///
    /// @return next_state, value of the generated sketching operator's next_state field.
    ///
    state_t call(
        int64_t m,
        int64_t n,
        T* A,
        int64_t lda,
        int64_t* J,
        T* tau,
        state_t &state
    );

};

// We are assuming that tau and J have been pre-allocated
// -----------------------------------------------------------------------------
template <typename T, typename qrcp_wide_t, typename RNG>
RandBLAS::RNGState<RNG> QRBBRP<T, qrcp_wide_t, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    int64_t* J,
    T* tau,
    RandBLAS::RNGState<RNG> &state
){
    using namespace std::chrono;
    //-------TIMING VARS--------/
    steady_clock::time_point preallocation_t_start;
    steady_clock::time_point preallocation_t_stop;
    steady_clock::time_point skop_t_start;
    steady_clock::time_point skop_t_stop;
    steady_clock::time_point qrcp_wide_t_start;
    steady_clock::time_point qrcp_wide_t_stop;
    steady_clock::time_point qr_tall_t_start;
    steady_clock::time_point qr_tall_t_stop;
    steady_clock::time_point apply_transq_t_start;
    steady_clock::time_point apply_transq_t_stop;
    steady_clock::time_point sample_update_t_start;
    steady_clock::time_point sample_update_t_stop;
    steady_clock::time_point total_t_start;
    steady_clock::time_point total_t_stop;
    long preallocation_t_dur       = 0;
    long skop_t_dur                = 0;
    long qrcp_wide_t_dur           = 0;
    long qr_tall_t_dur             = 0;
    long apply_transq_t_dur        = 0;
    long sample_update_t_dur       = 0;
    long total_t_dur               = 0;

    if (this -> timing) {
        total_t_start = steady_clock::now();
        preallocation_t_start = steady_clock::now();
    }
    int64_t rows              = m;
    int64_t cols              = n;
    // Describes sizes of full Q and R factors at a given iteration.
    int64_t curr_sz           = 0;
    int64_t b_sz              = this->block_size;
    int64_t maxiter           = (int64_t) std::ceil(std::min(m, n) / (T) b_sz);
    // Using this variable to work with matrices with leading dimension = b_sz.
    int64_t b_sz_const        = b_sz;
    // This will serve as lda of a sketch
    int64_t d                 = this->d_factor * b_sz;
    // We will be using this parameter when performing QRCP on a sketch.
    // After the first iteration of the algorithm, this will change its value to min(d, cols) 
    // before "cols" is updated.
    int64_t sampling_dimension = d;

    //********************************* POINTERS TO A BEGIN *********************************
    // LDA for all of the below is m

    // Pointer to the beginning of the original space of A.
    // Pointer to the beginning of A's "work zone," 
    // will shift at every iteration of an algorithm by (lda * b_sz) + b_sz.
    T* A_work = A;
    // Workspace 1 pointer - will serve as a buffer for computing R12 and updated matrix A.
    // Points to a location, offset by lda * b_sz from the current "A_work."
    T* Work1  = nullptr;
    // Points to R11 factor, right above the compact Q, of size b_sz by b_sz.
    T* R11    = nullptr;
    // Points to R12 factor, to the right of R11 and above Work1 of size b_sz by n - curr_sz - b_sz.
    T* R12    = nullptr;
    //********************************** POINTERS TO A END **********************************

    //********************************* POINTERS TO OTHER BEGIN *********************************
    // Pointer to the portion of vector tau at current iteration.
    T* tau_sub = nullptr;
    //********************************** POINTERS TO OTHER END **********************************

    //******************* POINTERS TO DATA REQUIRING ADDITIONAL STORAGE BEGIN *******************

    // J_buffer serves as a buffer for the pivots found at every iteration, of size n.
    // At every iteration, it would only hold "cols" entries.
    // Cannot really fully switch this to pointers bc we do not want data to be modified in "col_swap."
    std::vector<int64_t> J_buf (n, 0);
    int64_t* J_buffer = J_buf.data();

    // A_sk serves as a skething matrix, of size d by n, lda=d.
    // It's updated at the end of every iteration.
    // The algorithm needs to have access to the upper-triangular factor R
    // (stored in this matrix after geqp3) at all times. 
    T* A_sk = new T[d * n]();

    // Create a separate pointer to free when function terminates
    T* A_sk_const = A_sk;
    // Pointer to the b_sz by b_sz upper-triangular factor R stored in A_sk after GEQP3.
    T* R_sk = nullptr;
    //******************* POINTERS TO DATA REQUIRING ADDITIONAL STORAGE END *******************

    this->qrcp_wide.reserve(d, n);

    if (this -> timing) {
        preallocation_t_stop  = steady_clock::now();
        preallocation_t_dur   = duration_cast<microseconds>(preallocation_t_stop - preallocation_t_start).count();
        skop_t_start = steady_clock::now();
    }

    // Using Gaussian matrix as a sketching operator.
    RandBLAS::DenseSkOp<T> S({d, m}, state);
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, (T)S.dist.isometry_scale, S, A, m, (T)0.0, A_sk, d);
    auto out_state = S.next_state;

    if (this -> timing) {
        skop_t_stop  = steady_clock::now();
        skop_t_dur   = duration_cast<microseconds>(skop_t_stop - skop_t_start).count();
    }


    for (int64_t iter = 0; iter < maxiter; ++iter) {
        // Make sure we fit into the available space
        b_sz = std::min(b_sz, std::min(m, n) - curr_sz);

        // Zero-out data - may not be necessary
        std::fill(&J_buffer[0], &J_buffer[n], 0);

        if (this -> timing) { qrcp_wide_t_start = steady_clock::now(); }
            
        // Performing qrcp_wide below
        this->qrcp_wide(sampling_dimension, cols, A_sk, d, J_buffer);

        if (this -> timing) {
            qrcp_wide_t_stop = steady_clock::now();
            qrcp_wide_t_dur += duration_cast<microseconds>(qrcp_wide_t_stop - qrcp_wide_t_start).count();
        }

        // Need to premute trailing columns of the full R-factor.
        // Remember that the R-factor is stored the upper-triangular portion of A.
        // Pivoting the trailing R and the ``current'' A.      
        // The copy of A operation is done on a separete stream. If it was not, it would have been done here.  
        util::col_swap(m, cols, cols, &A[lda * curr_sz], lda, J_buf);

        // Updating pivots
        if (iter == 0) {
            blas::copy(cols, J_buffer, 1, J, 1);
        } else {
            RandLAPACK::util::col_swap<T>(cols, cols, &J[curr_sz], J_buf);
        }

        // Defining the new "working subportion" of matrix A.
        // In a global sense, below is identical to:
        // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz];
        Work1 = &A_work[lda * b_sz];
        // Define the space representing R_sk (stored in A_sk)
        R_sk = A_sk;
        
        if (this -> timing) { qr_tall_t_start = steady_clock::now(); }

        // Define a pointer to the current subportion of tau vector.
        tau_sub = &tau[curr_sz];

        // Panel QR
        lapack::geqrf(rows, b_sz, A_work, lda, tau_sub);
        // R11 is computed and placed in the appropriate space
        R11 = A_work;
        if (this -> timing) {
            qr_tall_t_stop  = steady_clock::now();
            qr_tall_t_dur  += duration_cast<microseconds>(qr_tall_t_stop - qr_tall_t_start).count();
            apply_transq_t_start = steady_clock::now();
        }

        // Performing apply_trans_q below.
        // Perform Q_full' * A_piv(:, b_sz:end) to find R12 and the new "current A."
        // If b_sz != b_sz_const -> last iteration, no need to find the new "current A." 
        // A_piv (Work1) is a rows by cols - b_sz matrix, stored in space of the original A.
        // The first b_sz rows will represent R12.
        // The last rows-b_sz rows will represent the new A.
        // With that, everything is placed where it should be, no copies required.
        // Q is defined with b_sz elementary reflectors. 
        if (b_sz != b_sz_const) {
            lapack::ormqr(Side::Left, Op::Trans, b_sz, cols - b_sz, b_sz, A_work, lda, tau_sub, Work1, lda);
        } else {
            lapack::ormqr(Side::Left, Op::Trans, rows, cols - b_sz, b_sz, A_work, lda, tau_sub, Work1, lda);
        }
        
        if (this -> timing) {
            apply_transq_t_stop  = steady_clock::now();
            apply_transq_t_dur  += duration_cast<microseconds>(apply_transq_t_stop - apply_transq_t_start).count();
        }

        // Updating the pointer to R12
        // In a global sense, this is identical to:
        // R12 =  &A[(m * (curr_sz + b_sz)) + curr_sz];
        R12 = &R11[lda * b_sz];

        // Size of the factors is updated;
        curr_sz += b_sz;

        // Termination criteria is reached when:
        // 1. All iterations are exhausted.
        // 2. b_sz has been altered, which happens
        // when the estimated rank of the R-factor 
        // from QRCP at this iteration is not full,
        // meaning that the rest of the matrix is zero.
        if ((curr_sz >= n) || (b_sz != b_sz_const)) {

            if(this -> timing) {
                total_t_stop = steady_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_other  = total_t_dur - (skop_t_dur + preallocation_t_dur + qrcp_wide_t_dur + qr_tall_t_dur + apply_transq_t_dur + sample_update_t_dur);
                this -> times.resize(8);
                this -> times = {skop_t_dur, preallocation_t_dur, qrcp_wide_t_dur, qr_tall_t_dur, apply_transq_t_dur, sample_update_t_dur, t_other, total_t_dur};

                printf("\n\n/------------QRBBRP TIMING RESULTS BEGIN------------/\n");
                printf("Preallocation time:                 %ld μs,\n", preallocation_t_dur);
                printf("SKOP time:                          %ld μs,\n", skop_t_dur);
                printf("QRCP_wide time:                     %ld μs,\n", qrcp_wide_t_dur);
                printf("QR_tall time:                       %ld μs,\n", qr_tall_t_dur);
                printf("Apply QT time:                      %ld μs,\n", apply_transq_t_dur);
                printf("Sample updating time:               %ld μs,\n", sample_update_t_dur);
                printf("Other routines time:                %ld μs,\n", t_other);
                printf("Total time:                         %ld μs.\n", total_t_dur);

                printf("\nPreallocation takes                     %6.2f%% of runtime.\n",  100 * ((T) preallocation_t_dur       / (T) total_t_dur));
                printf("SKOP generation and application takes     %6.2f%% of runtime.\n",  100 * ((T) skop_t_dur                / (T) total_t_dur));
                printf("QRCP_wide takes                           %6.2f%% of runtime.\n",  100 * ((T) qrcp_wide_t_dur           / (T) total_t_dur));
                printf("QR_tall takes                             %6.2f%% of runtime.\n",  100 * ((T) qr_tall_t_dur             / (T) total_t_dur));
                printf("Apply QT takes                            %6.2f%% of runtime.\n",  100 * ((T) apply_transq_t_dur        / (T) total_t_dur));
                printf("Sample updating time takes                %6.2f%% of runtime.\n",  100 * ((T) sample_update_t_dur       / (T) total_t_dur));
                printf("Everything else takes                     %6.2f%% of runtime.\n",  100 * ((T) t_other                   / (T) total_t_dur));
                printf("/-------------QRBBRP TIMING RESULTS END-------------/\n\n");
            }
            delete [] A_sk_const;
            this->qrcp_wide.free();
            return out_state;
        }

        if (this -> timing) { sample_update_t_start = steady_clock::now(); }

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
        if (sampling_dimension - b_sz > 0) {
            RandLAPACK::util::get_U(sampling_dimension - b_sz, sampling_dimension - b_sz, &R_sk[(d + 1) * b_sz], d);
        }

        // Changing the pointer to relevant data in A_sk - this is equaivalent to copying data over to the beginning of A_sk.
        // Remember that the only "active" portion of A_sk remaining would be of size sampling_dimension by cols;
        // if any rows beyond that would be accessed, we would have issues. 
        A_sk = &A_sk[d * b_sz];

        if (this -> timing) {
            sample_update_t_stop  = steady_clock::now();
            sample_update_t_dur  += duration_cast<microseconds>(sample_update_t_stop - sample_update_t_start).count();
        }

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;
    }
    throw RandBLAS::Error("This function should only return from within its main loop.");
}

} // end namespace RandLAPACK
