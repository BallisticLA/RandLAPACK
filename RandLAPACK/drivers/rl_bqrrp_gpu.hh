#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "lapack/device.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>

#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"

using namespace std::chrono;

namespace RandLAPACK {

template <typename T, typename RNG>
class BQRRP_GPU_alg {
    public:

        virtual ~BQRRP_GPU_alg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* A_sk,
            int64_t d,
            T* tau,
            int64_t* J
        ) = 0;
};

// Struct outside of BQRRP class to make symbols shorter
struct BQRRPGPUSubroutines {
    enum QRTall {cholqr, geqrf};
};

template <typename T, typename RNG>
class BQRRP_GPU : public BQRRP_GPU_alg<T, RNG> {
    public:

        using GPUSubroutine = BQRRPGPUSubroutines;

        /// This is a device version of the BQRRP algorithm - ALL INPUT DATA LIVES ON A GPU.
        /// By contrast to the CPU version of BQRRP scheme, the GPU version takes the sketch as an input;
        /// that is because at the moment RandBLAS does not have GPU support. 
        ///
        /// This file presents the BQRRP algorithmic framework for a blocked version of
        /// randomized QR with column pivoting, applicable to matrices with any aspect ratio.
        /// Depending on the user's choice for the subroutines, this framework can define versions of the practical 
        ///
        /// The core subroutines in question are:
        ///     1. qrcp_wide     - epresents a column-pivoted QR factorization method, suitable for wide matrices -- for now, no options other than the default LUQR is offered;
        ///     2. rank_est      - aims to estimate the exact rank of the given matrix -- for now, no options other than the default naive is offered;
        ///     3. col_perm      - responsible for permuting the columns of a given matrix in accordance with the indices stored in a given vector;
        ///     4. qr_tall       - performs a tall unpivoted QR factorization;
        ///     5. apply_trans_q - applies the transpose Q-factor output by qr_tall to a given matrix -- for now, no options other than the default ORMQR is offered.
        ///    
        /// The base structure of BQRRP resembles that of Algorithm 4 from https://arxiv.org/abs/1509.06820. 
        ///
        /// The algorithm optionally times all of its subcomponents through a user-defined 'timing' parameter.


        BQRRP_GPU(
            bool time_subroutines,
            int64_t b_sz
        ) {
            timing     = time_subroutines;
            tol        = std::numeric_limits<T>::epsilon();
            block_size = b_sz;
            qr_tall    = GPUSubroutine::QRTall::geqrf;
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
        /// @param[in] A_sk
        ///     d by n sketch of A, computed via RandBLAS
        ///
        /// @param[in] d
        ///     Sampling dimension of a sketch, n >= d_factor >= block_size.
        ///
        /// @param[in] tau
        ///     Pointer to a vector of size n. On entry, is empty.
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
            T* A_sk,
            int64_t d,
            T* tau,
            int64_t* J
        ) override;

    public:
        bool timing;
        RandBLAS::RNGState<RNG> state;
        int64_t rank;
        int64_t block_size;

        // 15 entries - logs time for different portions of the algorithm
        std::vector<long> times;

        // Naive rank estimation parameter;
        T tol;

        // Core subroutines options, controlled by user
        GPUSubroutine::QRTall qr_tall;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int BQRRP_GPU<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* A_sk,
    int64_t d,
    T* tau,
    int64_t* J
){
    steady_clock::time_point preallocation_t_start;
    steady_clock::time_point preallocation_t_stop;
    steady_clock::time_point qrcp_wide_t_start;
    steady_clock::time_point qrcp_wide_t_stop;
    steady_clock::time_point copy_A_sk_t_start;
    steady_clock::time_point copy_A_sk_t_stop;
    steady_clock::time_point qrcp_piv_t_start;
    steady_clock::time_point qrcp_piv_t_stop;
    steady_clock::time_point copy_A_t_start;
    steady_clock::time_point copy_A_t_stop;
    steady_clock::time_point piv_A_t_start;
    steady_clock::time_point piv_A_t_stop;
    steady_clock::time_point updating_J_t_start;
    steady_clock::time_point updating_J_t_stop;
    steady_clock::time_point copy_J_t_start;
    steady_clock::time_point copy_J_t_stop;
    steady_clock::time_point preconditioning_t_start;
    steady_clock::time_point preconditioning_t_stop;
    steady_clock::time_point qr_tall_t_start;
    steady_clock::time_point qr_tall_t_stop;
    steady_clock::time_point q_reconstruction_t_start;
    steady_clock::time_point q_reconstruction_t_stop;
    steady_clock::time_point apply_transq_t_start;
    steady_clock::time_point apply_transq_t_stop;
    steady_clock::time_point sample_update_t_start;
    steady_clock::time_point sample_update_t_stop;
    steady_clock::time_point total_t_start;
    steady_clock::time_point total_t_stop;
    long preallocation_t_dur    = 0;
    long qrcp_wide_t_dur        = 0;
    long copy_A_sk_t_dur        = 0;
    long qrcp_piv_t_dur         = 0;
    long copy_A_t_dur           = 0;
    long piv_A_t_dur            = 0;
    long updating_J_t_dur       = 0;
    long copy_J_t_dur           = 0;
    long preconditioning_t_dur  = 0;
    long qr_tall_t_dur          = 0;
    long q_reconstruction_t_dur = 0;
    long apply_transq_t_dur     = 0;
    long sample_update_t_dur    = 0;
    long total_t_dur            = 0;

    if(this -> timing) {
        total_t_start = steady_clock::now();
        preallocation_t_start = steady_clock::now();
    }

    int iter;
    int64_t rows       = m;
    int64_t cols       = n;
    // Describes sizes of full Q and R factors at a given iteration.
    int64_t curr_sz    = 0;
    int64_t b_sz       = this->block_size;
    int64_t maxiter    = (int64_t) std::ceil(std::min(m, n) / (T) b_sz);
    // Using this variable to work with matrices with leading dimension = b_sz.
    int64_t b_sz_const = b_sz;
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

    /******************************STREAM/QUEUE/HANDLE*********************************/
    lapack::Queue lapack_queue(0);
    cudaStream_t strm = lapack_queue.stream();
    using lapack::device_info_int;
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
    int *d_info_cusolver = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_info_cusolver), sizeof(int));
    // Create cusolver handle - used for the ORMQR call
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnCreate(&cusolverH);
    cusolverDnSetStream(cusolverH, strm);

    /******************************WORKSPACE PARAMETERS*********************************/
    char* d_work_getrf, * d_work_geqrf;
    char* h_work_getrf = nullptr;
    char* h_work_geqrf = nullptr;
    int lwork_ormqr = 0;
    T *d_work_ormqr = nullptr;
    size_t d_size_getrf, h_size_getrf, d_size_geqrf, h_size_geqrf;

    char* d_work_geqrf_opt = nullptr;
    char* h_work_geqrf_opt = nullptr;
    size_t d_size_geqrf_opt, h_size_geqrf_opt;

    //*********************************POINTERS TO INPUT DATA BEGIN*********************************
    // will shift at every iteration of an algorithm by (lda * b_sz) + b_sz.
    T* A_work = A;    
    // Workspace 1 pointer - will serve as a buffer for computing R12 and updated matrix A.
    // Points to a location, offset by lda * b_sz from the current "A_work."
    T* Work1 = NULL;
    // Points to R11 factor, right above the compact Q, of size b_sz by b_sz.
    T* R11 = NULL;
    // Points to R12 factor, to the right of R11 and above Work1 of size b_sz by n - curr_sz - b_sz.
    T* R12 = NULL;
    // A_sk is a sketch of the orifinal data, of size d by n, lda d
    // Below algorithm does not perform repeated sampling, hence A_sk
    // is updated at the end of every iteration.
    // Should remain unchanged throughout the algorithm,
    // As the algorithm needs to have access to the upper-triangular factor R
    // (stored in this matrix after geqp3) at all times. 
    T* A_sk_work = A_sk;
    //**********************************POINTERS TO INPUT DATA END**********************************

    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE BEGIN*******************
    // J_buffer serves as a buffer for the pivots found at every iteration, of size n.
    // At every iteration, it would only hold "cols" entries.
    int64_t* J_buffer;
    cudaMallocAsync(&J_buffer, n * sizeof(int64_t), strm);

    // Special pivoting buffer for LU factorization, capturing the swaps on A_sk'.
    // Needs to be converted in a proper format of length rows(A_sk')
    int64_t* J_buffer_lu;
    cudaMallocAsync(&J_buffer_lu, std::min(d, n) * sizeof(int64_t), strm);

    // Pointer to the b_sz by b_sz upper-triangular facor R stored in A_sk after GEQP3.
    T* R_sk = NULL;

    // View to the transpose of A_sk.
    // Is of size n * d, with an lda n.   
    T* A_sk_trans;
    cudaMallocAsync(&A_sk_trans, n * d * sizeof(T), strm);

    // Buffer for the R-factor in tall QR, of size b_sz by b_sz, lda b_sz.
    // Also used to store the proper R11_full-factor after the 
    // full Q has been restored form economy Q (the has been found via tall QR);
    // That is done by applying the sign vector D from orhr_col().
    // Eventually, will be used to store R11 (computed via trmm)
    T* R_tall_qr;
    cudaMallocAsync(&R_tall_qr, b_sz_const * b_sz_const * sizeof(T), strm);

    // Buffer for Tau in GEQP3 and D in orhr_col, of size n.
    T* Work2;
    cudaMallocAsync(&Work2, n * sizeof(T), strm);

    // Pointer to the working subportion of the vetcor J
    int64_t* J_work = J;

    // Additional buffers required to perform parallel column swapping.
    // Parallel column swapping is done by moving the columns from
    // a copy of the input matrix into the input matrix in accordance with
    // the entries in the input index vector.
    // As you will see below, we use a special strategy to avoid performing explicit copies of 
    // A_sk and J.
    // This strategy would still require using buffers of size of the original data.
    T* A_copy_col_swap;
    cudaMallocAsync(&A_copy_col_swap, sizeof(T) * m * n, strm);

    T* A_sk_copy_col_swap;
    cudaMallocAsync(&A_sk_copy_col_swap, sizeof(T) * d * n, strm);
    T* A_sk_copy_col_swap_work = A_sk_copy_col_swap;
    
    int64_t* J_copy_col_swap;
    cudaMallocAsync(&J_copy_col_swap, sizeof(int64_t) * n, strm);
    int64_t* J_copy_col_swap_work = J_copy_col_swap;
    
    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE END*******************
    cudaStreamSynchronize(strm);
    if(this -> timing) {
        lapack_queue.sync();
        preallocation_t_stop  = steady_clock::now();
        preallocation_t_dur   = duration_cast<microseconds>(preallocation_t_stop - preallocation_t_start).count();
    }

    for(iter = 0; iter < maxiter; ++iter) {
        nvtxRangePushA("Iteration");

        // Make sure we fit into the available space
        b_sz = std::min(this->block_size, std::min(m, n) - curr_sz);
        block_rank = b_sz;

        // Zero-out data - may not be necessary
        cudaMemsetAsync(J_buffer_lu, (T) 0.0, std::min(d, n), strm);
        cudaMemsetAsync(J_buffer,    (T) 0.0, n, strm);
        cudaMemsetAsync(Work2,       (T) 0.0, n, strm);


        if(this -> timing) {
            nvtxRangePushA("qrcp_wide");
            qrcp_wide_t_start = steady_clock::now();
        }
        // qrcp_wide through LUQR below
        // Perform pivoted LU on A_sk', follow it up by unpivoted QR on a permuted A_sk.
        // Get a transpose of A_sk
        RandLAPACK::cuda_kernels::transposition_gpu(strm, sampling_dimension, cols, A_sk_work, d, A_sk_trans, n, 0);
    
        // Perform a row-pivoted LU on a transpose of A_sk
        // Probing workspace size - performed only once.
        if(iter == 0) {
            lapack::getrf_work_size_bytes(cols, sampling_dimension, A_sk_trans, n, &d_size_getrf, &h_size_getrf, lapack_queue);
            d_work_getrf = blas::device_malloc< char >( d_size_getrf, lapack_queue );
        }
        lapack::getrf(cols, sampling_dimension, A_sk_trans, n, J_buffer_lu, d_work_getrf, d_size_getrf, h_work_getrf, h_size_getrf, d_info, lapack_queue);
        // Fill the pivot vector, apply swaps found via lu on A_sk'.
        RandLAPACK::cuda_kernels::LUQRCP_piv_process_gpu(strm, sampling_dimension, cols, J_buffer, J_buffer_lu);
        
        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePushA("copy_A_sk");
            copy_A_sk_t_start = steady_clock::now();
        }
        // Instead of copying A_sk_work into A_sk_copy_col_swap, we ``swap'' the pointers.
        // This is safe, as A_sk is not needed outside of BQRRP.
        std::swap(A_sk_copy_col_swap_work, A_sk_work);
        
        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            copy_A_sk_t_stop = steady_clock::now();
            copy_A_sk_t_dur += duration_cast<microseconds>(copy_A_sk_t_stop - copy_A_sk_t_start).count();
            nvtxRangePushA("piv_A_sk");
            qrcp_piv_t_start = steady_clock::now();
        }

        // Apply pivots to A_sk
        RandLAPACK::cuda_kernels::col_swap_gpu(strm, sampling_dimension, cols, cols, A_sk_work, d, A_sk_copy_col_swap_work, d, J_buffer);

        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            qrcp_piv_t_stop = steady_clock::now();
            qrcp_piv_t_dur += duration_cast<microseconds>(qrcp_piv_t_stop - qrcp_piv_t_start).count();
        }

        // Perform an unpivoted QR on A_sk
        if(iter == 0) {
            lapack::geqrf_work_size_bytes(sampling_dimension, cols, A_sk_work, d, &d_size_geqrf, &h_size_geqrf, lapack_queue);
            d_work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
        }
        lapack::geqrf(sampling_dimension, cols, A_sk_work, d, Work2, d_work_geqrf, d_size_geqrf, h_work_geqrf, h_size_geqrf, d_info, lapack_queue);
        
        if(this -> timing) {
            lapack_queue.sync();
            nvtxRangePop();
            qrcp_wide_t_stop = steady_clock::now();
            qrcp_wide_t_dur += duration_cast<microseconds>(qrcp_wide_t_stop - qrcp_wide_t_start).count();
            nvtxRangePushA("copy_A");
            copy_A_t_start = steady_clock::now();
        }
        // Need to premute trailing columns of the full R-factor.
        // Remember that the R-factor is stored the upper-triangular portion of A.
        // Pivoting the trailing R and the ``current'' A.      
        // The copy of A operation is done on a separete stream. If it was not, it would have been done here.  
        
        if(this -> timing) {
            nvtxRangePop();
            copy_A_t_stop = steady_clock::now();
            copy_A_t_dur += duration_cast<microseconds>(copy_A_t_stop - copy_A_t_start).count();
            nvtxRangePushA("piv_A");
            piv_A_t_start = steady_clock::now();
        }

        // Instead of copying A into A_copy_col_swap, we ``swap'' the pointers.
        // We have to take some precautions when BQRRP main loop terminates.
        // Since we want A to be accessible and valid outside of BQRRP, we need to make sure that 
        // its entries were, in fact, computed correctly. 
        //
        // The original memory space that the matrix A points to would only contain the correct entry ranges, computed at ODD
        // iterations of BQRRP's main loop.
        // The correct entries from the even iterations would be contained in the memory space that was originally pointed to
        // by A_copy_col_swap.
        // Hence, when BQRRP terminates, we would need to copy the results from the even iterations form A_copy_col_swap to A.
        //
        // Remember that since the pointers A and A_copy_col_swap are swapped at every even iteration of the main BQRRP loop,
        // if the BQRRP terminates with iter being even, we would need to swap these pointers back around.
        // Recall also that if A and A_cpy needed to be swapped at termination and iter != maxiters, A_cpy would contain the "correct"
        // entries in column range ((iter + 1) * b_sz : end), so we need to not forget to copy those over into A.
        //
        // Additional thing to remember is that the final copy needs to be performed in terms of b_sz_const, not b_sz.
        std::swap(A_copy_col_swap, A);
        A_work = &A[lda * curr_sz + curr_sz];
        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, cols, cols, &A[lda * curr_sz], lda, &A_copy_col_swap[lda * curr_sz], lda, J_buffer);
        
        // Checking for the zero matrix post-pivoting is the best idea, 
        // as we would only need to check one column (pivoting moves the column with the largest norm upfront)
        block_zero = true;
        RandLAPACK::cuda_kernels::all_of(strm, rows, std::numeric_limits<T>::epsilon(), A_work, block_zero);
        if(block_zero){
            // Need to update the pivot vector, same as below.
            if(iter == 0) {
                RandLAPACK::cuda_kernels::copy_gpu(strm, n, J_buffer, 1, J, 1);
            } else {
                std::swap(J_copy_col_swap, J);
                J_work = &J[curr_sz];
                J_copy_col_swap_work = &J_copy_col_swap[curr_sz];
                RandLAPACK::cuda_kernels::col_swap_gpu<T>(strm, cols, cols, J_work, J_copy_col_swap_work, J_buffer);
            }

            this -> rank = curr_sz;
            // Measures taken to ensure J holds correct data, explained above.
            if(iter % 2) {
                // Total number of iterations is odd (iter starts at 0)
                std::swap(J_copy_col_swap, J);
            } else {
                // Total number of iterations is even
                std::swap(A_copy_col_swap, A);
                if(iter != (maxiter - 1)){
                    // Copy trailing portion of A_cpy into A
                    blas::device_copy_matrix(m, n - (iter + 1) * b_sz_const, &A_copy_col_swap[lda * (iter + 1) * b_sz_const], lda, &A[lda * (iter + 1) * b_sz_const], lda, lapack_queue);
                }
            }
            for (int idx = 0; idx <= iter; ++idx) {
                if (idx % 2) {  // Odd index - copy portions of J
                    if (idx == iter) {
                        // Avoid copying extra entries if b_sz has changed
                        RandLAPACK::cuda_kernels::copy_gpu(strm, b_sz, &J_copy_col_swap[idx * b_sz_const], 1, &J[idx * b_sz_const], 1);
                    } else {
                        RandLAPACK::cuda_kernels::copy_gpu(strm, b_sz_const, &J_copy_col_swap[idx * b_sz_const], 1, &J[idx * b_sz_const], 1);
                    }
                } else {  // Even index - copy portions of A
                    if (idx == iter) {
                        // Avoid copying extra entries if b_sz has changed
                        blas::device_copy_matrix(m, b_sz, &A_copy_col_swap[lda * idx * b_sz_const], lda, &A[lda * idx * b_sz_const], lda, lapack_queue);
                    } else {
                        blas::device_copy_matrix(m, b_sz_const, &A_copy_col_swap[lda * idx * b_sz_const], lda, &A[lda * idx * b_sz_const], lda, lapack_queue);
                    }
                }
            }
            lapack_queue.sync();

            cudaFree(A_sk_trans);
            cudaFree(J_buffer);
            cudaFree(J_buffer_lu);
            cudaFree(Work2);
            cudaFree(R_tall_qr);
            cudaFree(A_copy_col_swap);
            cudaFree(A_sk_copy_col_swap);
            cudaFree(J_copy_col_swap);

            // Freeing workspace info variables
            blas::device_free(d_info, lapack_queue);
            cudaFree(d_info_cusolver);
            blas::device_free(d_work_getrf, lapack_queue);
            blas::device_free(d_work_geqrf, lapack_queue);

            // At iteration 0, below allocations have not yet taken place
            if (iter > 0) {
                cudaFree(d_work_ormqr);
                if(this -> qr_tall != GPUSubroutine::QRTall::cholqr){
                    blas::device_free(d_work_geqrf_opt, lapack_queue);
                }
            }
            return 0;
        }
        
        // Defining the new "working subportion" of matrix A.
        // In a global sense, below is identical to:
        // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz];
        Work1 = &A_work[lda * b_sz];
        // Define the space representing R_sk (stored in A_sk)
        R_sk = A_sk_work;

        // Naive rank estimation to perform preconditioning safely.
        // Variable block_rank is altered if the rank is not full.
        // If this happens, we will terminate at the end of the current iteration.
        // If the internal_nb, used in gemqrt and orhr_col is larger than the updated block_rank, it would need to be updated as well.
        // Updating block_rank affects the way the preconditioning is done, which, in its turn, affects CholQR, ORHR_COL, updating A and updating R.
        RandLAPACK::cuda_kernels::naive_rank_est(strm, b_sz, this -> tol, R_sk, d, block_rank);

        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            piv_A_t_stop  = steady_clock::now();
            piv_A_t_dur  += duration_cast<microseconds>(piv_A_t_stop - piv_A_t_start).count();
        }

        // Updating pivots
        if(iter == 0) {
            if(this -> timing) {
                nvtxRangePushA("update_J");
                updating_J_t_start = steady_clock::now();
            }
            RandLAPACK::cuda_kernels::copy_gpu(strm, n, J_buffer, 1, J, 1);
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                updating_J_t_stop  = steady_clock::now();
                updating_J_t_dur  += duration_cast<microseconds>(updating_J_t_stop - updating_J_t_start).count();
                nvtxRangePushA("update_R");
            }
        } else {
            if(this -> timing) {
                nvtxRangePushA("copy_J");
                copy_J_t_start = steady_clock::now();
            }
            // Instead of copying J into J_copy_col_swap, we ``swap'' the pointers.
            // We have to take some precautions when BQRRP main loop terminates.
            // Since we want J to be accessible and valid outside of BQRRP, we need to make sure that 
            // its entries were, in fact, computed correctly. 
            //
            // The original memory space that the vector J points to would only contain the correct pivot ranges, computed at EVEN
            // iterations of BQRRP's main loop (by contrast to the situation with matrix A, since the pointers J and J_cpy do not get swapped at iteration 0).
            // The correct entries from the odd iterations would be contained in the memory space that was originbally pointed to
            // by J_copy_col_swap.
            // Hence, when BQRRP terminates, we would need to copy the results from the odd iterations form J_copy_col_swap to J.
            //
            // Remember that since the pointers J and J_copy_col_swap are swapped at every odd iteration of the main BQRRP loop,
            // if the BQRRP terminates with iter being odd, we would need to swap these pointers back around.
            //
            // Additional thing to remember is that the final copy needs to be performed in terms of b_sz_const, not b_sz.
            std::swap(J_copy_col_swap, J);
            J_work = &J[curr_sz];
            J_copy_col_swap_work = &J_copy_col_swap[curr_sz];

            if(this -> timing) {
                nvtxRangePop();
                copy_J_t_stop  = steady_clock::now();
                copy_J_t_dur  += duration_cast<microseconds>(copy_J_t_stop - copy_J_t_start).count();
                nvtxRangePushA("update_J");
                updating_J_t_start = steady_clock::now();
            }

            RandLAPACK::cuda_kernels::col_swap_gpu<T>(strm, cols, cols, J_work, J_copy_col_swap_work, J_buffer);
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                updating_J_t_stop  = steady_clock::now();
                updating_J_t_dur  += duration_cast<microseconds>(updating_J_t_stop - updating_J_t_start).count();
            }
        }

        // qr_tall through either cholqr or geqrf below
        if(this -> qr_tall == GPUSubroutine::QRTall::cholqr) {
            if(this -> timing) {
                nvtxRangePushA("precond_A");
                preconditioning_t_start = steady_clock::now();
            }
            
            // A_pre = AJ(:, 1:b_sz) * inv(R_sk)
            // Performing preconditioning of the current matrix A.
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, block_rank, (T) 1.0, R_sk, d, A_work, lda, lapack_queue);

            if(this -> timing) {
                lapack_queue.sync();
                nvtxRangePop();
                preconditioning_t_stop  = steady_clock::now();
                preconditioning_t_dur  += duration_cast<microseconds>(preconditioning_t_stop - preconditioning_t_start).count();
                nvtxRangePushA("qr_tall");
                qr_tall_t_start = steady_clock::now();
            }
            
            // Performing tall QR
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, block_rank, rows, (T) 1.0, A_work, lda, (T) 0.0, R_tall_qr, b_sz_const, lapack_queue);
            lapack::potrf(Uplo::Upper,  block_rank, R_tall_qr, b_sz_const, d_info, lapack_queue);
            // Compute Q_econ from tall QR
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, block_rank, (T) 1.0, R_tall_qr, b_sz_const, A_work, lda, lapack_queue);
            
            if(this -> timing) {
                lapack_queue.sync();
                nvtxRangePop();
                qr_tall_t_stop    = steady_clock::now();
                qr_tall_t_dur     += duration_cast<microseconds>(qr_tall_t_stop - qr_tall_t_start).count();
                nvtxRangePushA("orhr_col");
                q_reconstruction_t_start = steady_clock::now();
            }

            // Find Q (stored in A) using Householder reconstruction. 
            // This will represent the full (rows by rows) Q factor form tall QR
            // It would have been really nice to store T right above Q, but without using extra space,
            // it would result in us loosing the first lower-triangular b_sz by b_sz portion of implicitly-stored Q.
            // Filling T without ever touching its lower-triangular space would be a nice optimization for orhr_col routine.
            // This routine is defined in LAPACK 3.9.0. At the moment, LAPACK++ fails to invoke the newest Accelerate library.
            // Q is defined with block_rank elementary reflectors.
            RandLAPACK::cuda_kernels::orhr_col_gpu(strm, rows, block_rank, A_work, lda, &tau[curr_sz], Work2);  

            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                q_reconstruction_t_stop  = steady_clock::now();
                q_reconstruction_t_dur  += duration_cast<microseconds>(q_reconstruction_t_stop - q_reconstruction_t_start).count();
            }

            // Need to change signs in the R-factor from tall QR.
            // Signs correspond to matrix D from orhr_col().
            // This allows us to not explicitly compute R11_full = (Q[:, 1:block_rank])' * A_pre.
            RandLAPACK::cuda_kernels::R_cholqr_signs_gpu(strm, block_rank, b_sz_const, R_tall_qr, Work2);

            // Undoing the preconditioning below

            // Alternatively, instead of trmm + copy, we could perform a single gemm.
            // Compute R11 = R11_full(1:block_rank, :) * R_sk
            // R11_full is stored in R_tall_qr space, R_sk is stored in A_sk space.
            blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, block_rank, b_sz, (T) 1.0, R_sk, d, R_tall_qr, b_sz_const, lapack_queue);
            // Need to copy R11 over form R_tall_qr into the appropriate space in A.
            // We cannot avoid this copy, since trmm() assumes R_tall_qr is a square matrix.
            // In a global sense, this is identical to:
            // R11 =  &A[(m + 1) * curr_sz];
            R11 = A_work;
            RandLAPACK::cuda_kernels::copy_mat_gpu(strm, block_rank, b_sz, R_tall_qr, b_sz_const, A_work, lda, true);

        } else {
            // Default case - performing QR_tall via QRF
            if(this -> timing) {
                nvtxRangePushA("qr_tall");
                qr_tall_t_start = steady_clock::now();
            }
            // Perform an unpivoted QR instead of CholQR
            // Uncommenting the conditional below for the following reason: in contrary to my assumption that the larger 
            // problem size (the largest problem occurs at iter==0) would require the most amount of device workspace,
            // on an NVIDIA H100, the most workspace is required at iter==1 (the amount of workspace stays constent afterward).
            // For a skeptical reviewer, note: this is NOT related to a synch barrier.
            //if(iter == 0) {
                lapack::geqrf_work_size_bytes(rows, b_sz, A_work, lda, &d_size_geqrf_opt, &h_size_geqrf_opt, lapack_queue);
                d_work_geqrf_opt = blas::device_malloc< char >( d_size_geqrf_opt, lapack_queue );
                // Below shoudl not be necessary
                //std::vector<char> h_work_geqrf_vector_opt( h_size_geqrf_opt );
                //h_work_geqrf_opt = h_work_geqrf_vector_opt.data();
            //}
            lapack::geqrf(rows, b_sz, A_work, lda, &tau[curr_sz], d_work_geqrf_opt, d_size_geqrf_opt, h_work_geqrf_opt, h_size_geqrf_opt, d_info, lapack_queue);
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                qr_tall_t_stop    = steady_clock::now();
                qr_tall_t_dur     += duration_cast<microseconds>(qr_tall_t_stop - qr_tall_t_start).count();
            }
            // R11 is computed and placed in the appropriate space
            R11 = A_work;
        }
        // apply_trans_q through CuSOLVER's ormqr below
        //
        // Perform Q_full' * A_piv(:, block_rank:end) to find R12 and the new "current A."
        // A_piv (Work1) is a rows by cols - b_sz matrix, stored in space of the original A.
        // The first b_sz rows will represent R12.
        // The last rows-b_sz rows will represent the new A.
        // With that, everything is placed where it should be, no copies required.
        // ORMQR proves to be much faster than GEMQRT with MKL.
        // Q is defined with block_rank elementary reflectors. 
        if(this -> timing) {
            nvtxRangePushA("update_A");
            apply_transq_t_start = steady_clock::now();
        }

        if (block_rank != b_sz_const) {
            if (iter == 0) {
                // Compute optimal workspace size
                cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, block_rank, cols - b_sz, block_rank, A_work, lda, &tau[iter * b_sz], Work1, lda, &lwork_ormqr);
                // Allocate workspace
                cudaMalloc(reinterpret_cast<void **>(&d_work_ormqr), sizeof(double) * lwork_ormqr);
            }
            cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, block_rank, cols - b_sz, block_rank, A_work, lda, &tau[iter * b_sz], Work1, lda, d_work_ormqr, lwork_ormqr, d_info_cusolver);
            // Synchronization required after using cusolver
            cudaStreamSynchronize(strm);
        } else {
            if (iter == 0) {
                // Compute optimal workspace size
                cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rows, cols - b_sz, block_rank, A_work, lda, &tau[iter * b_sz], Work1, lda, &lwork_ormqr);
                // Allocate workspace
                cudaMalloc(reinterpret_cast<void **>(&d_work_ormqr), sizeof(double) * lwork_ormqr);
            }
            cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rows, cols - b_sz, block_rank, A_work, lda, &tau[iter * b_sz], Work1, lda, d_work_ormqr, lwork_ormqr, d_info_cusolver);
            // Synchronization required after using cusolver
            cudaStreamSynchronize(strm);
        }

        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
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
        // 2. block_rank has been altered, which happens
        // when the estimated rank of the R-factor 
        // from QRCP at this iteration is not full,
        // meaning that the rest of the matrix is zero.
        if((curr_sz >= n) || (block_rank != b_sz_const)) {
            this -> rank = curr_sz;
            // Measures taken to insure J holds correct data, explained above.
            if(iter % 2) {
                // Total number of iterations is odd (iter starts at 0)
                std::swap(J_copy_col_swap, J);
            } else {
                // Total number of iterations is even
                std::swap(A_copy_col_swap, A);
                // In addition to the copy from A_cpy to A space below, we also need to account for the cases when early termination has occured (iter != maxiters - 1), and pointers A and A_cpy need to switch places,
                // Aka when A_cpy has the "correct" trailing entries.
                // This means that the all entries from (iter + 1) * b_sz to end need to be copied over from A_cpy to A.
                // It is most likely the case that these trailing entries are all 0, but in order to be extra safe, we shall perform a full copy.
                if(iter != (maxiter - 1)){
                    blas::device_copy_matrix(m, n - (iter + 1) * b_sz_const, &A_copy_col_swap[lda * (iter + 1) * b_sz_const], lda, &A[lda * (iter + 1) * b_sz_const], lda, lapack_queue);
                }
            }
            for (int idx = 0; idx <= iter; ++idx) {
                if (idx % 2) {  // Odd index - copy portions of J
                    if (idx == iter) {
                        // Avoid copying extra entries if b_sz has changed
                        RandLAPACK::cuda_kernels::copy_gpu(strm, b_sz, &J_copy_col_swap[idx * b_sz_const], 1, &J[idx * b_sz_const], 1);
                    } else {
                        RandLAPACK::cuda_kernels::copy_gpu(strm, b_sz_const, &J_copy_col_swap[idx * b_sz_const], 1, &J[idx * b_sz_const], 1);
                    }
                } else {  // Even index - copy portions of A
                    if (idx == iter) {
                        // Avoid copying extra entries if b_sz has changed
                        blas::device_copy_matrix(m, b_sz, &A_copy_col_swap[lda * idx * b_sz_const], lda, &A[lda * idx * b_sz_const], lda, lapack_queue);
                    } else {
                        blas::device_copy_matrix(m, b_sz_const, &A_copy_col_swap[lda * idx * b_sz_const], lda, &A[lda * idx * b_sz_const], lda, lapack_queue);
                    }
                }
            }
            lapack_queue.sync();

            if(this -> timing) {
                total_t_stop = steady_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest  = total_t_dur - (preallocation_t_dur + qrcp_wide_t_dur + copy_A_t_dur + piv_A_t_dur + copy_J_t_dur + updating_J_t_dur + preconditioning_t_dur + qr_tall_t_dur + q_reconstruction_t_dur + apply_transq_t_dur + sample_update_t_dur);
                this -> times.resize(15);
                auto qrcp_main_t_dur = qrcp_wide_t_dur - qrcp_piv_t_dur - copy_A_sk_t_dur;
                this -> times = {preallocation_t_dur, qrcp_main_t_dur, copy_A_sk_t_dur, qrcp_piv_t_dur, copy_A_t_dur, piv_A_t_dur, copy_J_t_dur, updating_J_t_dur, preconditioning_t_dur, qr_tall_t_dur, q_reconstruction_t_dur, apply_transq_t_dur, sample_update_t_dur, t_rest, total_t_dur};

                printf("\n\n/------------BQRRP TIMING RESULTS BEGIN------------/\n");
                printf("Preallocation time:                    %ld μs,\n", preallocation_t_dur);
                printf("QRCP_wide main time:                  %ld μs,\n", qrcp_main_t_dur);
                printf("Copy(A_sk) time:                      %ld μs,\n", copy_A_sk_t_dur);
                printf("QRCP_wide piv time:                   %ld μs,\n", qrcp_piv_t_dur);
                printf("Copy(A) time:                         %ld μs,\n", copy_A_t_dur);
                printf("Piv(A) time:                          %ld μs,\n", piv_A_t_dur);
                printf("Copy(J) time:                         %ld μs,\n", copy_J_t_dur);
                printf("J updating time:                      %ld μs,\n", updating_J_t_dur);
                printf("Preconditioning time:                 %ld μs,\n", preconditioning_t_dur);
                printf("QR_tall time:                         %ld μs,\n", qr_tall_t_dur);
                printf("Householder reconstruction time:      %ld μs,\n", q_reconstruction_t_dur);
                printf("Apply QT time:                        %ld μs,\n", apply_transq_t_dur);
                printf("Sample updating time:                 %ld μs,\n", sample_update_t_dur);
                printf("Other routines time:                  %ld μs,\n", t_rest);
                printf("Total time:                           %ld μs.\n", total_t_dur);

                printf("\nPreallocation takes                 %6.2f%% of runtime.\n", 100 * ((T) preallocation_t_dur    / (T) total_t_dur));
                printf("QRCP_wide main takes                  %6.2f%% of runtime.\n", 100 * ((T) qrcp_main_t_dur        / (T) total_t_dur));
                printf("Copy(A_sk) takes                      %6.2f%% of runtime.\n", 100 * ((T) copy_A_sk_t_dur        / (T) total_t_dur));
                printf("QRCP_wide piv takes                   %6.2f%% of runtime.\n", 100 * ((T) qrcp_piv_t_dur         / (T) total_t_dur));
                printf("Copy(A) takes                         %6.2f%% of runtime.\n", 100 * ((T) copy_A_t_dur           / (T) total_t_dur));
                printf("Piv(A) takes                          %6.2f%% of runtime.\n", 100 * ((T) piv_A_t_dur            / (T) total_t_dur));
                printf("Copy(J) takes                         %6.2f%% of runtime.\n", 100 * ((T) copy_J_t_dur           / (T) total_t_dur));
                printf("J updating time takes                 %6.2f%% of runtime.\n", 100 * ((T) updating_J_t_dur       / (T) total_t_dur));
                printf("Preconditioning takes                 %6.2f%% of runtime.\n", 100 * ((T) preconditioning_t_dur  / (T) total_t_dur));
                printf("QR_tall takes                         %6.2f%% of runtime.\n", 100 * ((T) qr_tall_t_dur          / (T) total_t_dur));
                printf("Householder reconstruction takes      %6.2f%% of runtime.\n", 100 * ((T) q_reconstruction_t_dur / (T) total_t_dur));
                printf("Apply QT takes                        %6.2f%% of runtime.\n", 100 * ((T) apply_transq_t_dur     / (T) total_t_dur));
                printf("Sample updating time takes            %6.2f%% of runtime.\n", 100 * ((T) sample_update_t_dur    / (T) total_t_dur));
                printf("Everything else takes                 %6.2f%% of runtime.\n", 100 * ((T) t_rest                 / (T) total_t_dur));
                printf("/-------------BQRRP TIMING RESULTS END-------------/\n\n");
            }

            cudaFree(A_sk_trans);        
            cudaFree(J_buffer); 
            cudaFree(J_buffer_lu);
            cudaFree(Work2);
            cudaFree(R_tall_qr);    
            cudaFree(A_copy_col_swap);
            cudaFree(A_sk_copy_col_swap); 
            cudaFree(J_copy_col_swap);

            // Freeing workspace info variables
            blas::device_free(d_info, lapack_queue);
            cudaFree(d_info_cusolver);
            blas::device_free(d_work_getrf, lapack_queue);
            blas::device_free(d_work_geqrf, lapack_queue);

            cudaFree(d_work_ormqr);
            if(this -> qr_tall != GPUSubroutine::QRTall::cholqr){
                blas::device_free(d_work_geqrf_opt, lapack_queue);
            }
            return 0;
        }
        if(this -> timing) {
            nvtxRangePushA("update_Sk");
            sample_update_t_start = steady_clock::now();
        }
        // Updating the pointer to "Current A."
        // In a global sense, below is identical to:
        // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz + b_sz];
        // Also, Below is identical to:
        // A_work = &A_work[(lda + 1) * b_sz];
        A_work = &Work1[b_sz];

        // Updating the skethcing buffer
        // trsm (R_sk, R11) -> R_sk
        // Clearing the lower-triangular portion here is necessary, if there is a more elegant way, need to use that.
        RandLAPACK::cuda_kernels::get_U_gpu(strm, b_sz, b_sz, R_sk, d);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, b_sz, b_sz, (T) 1.0, R11, lda, R_sk, d, lapack_queue);

        // R_sk_12 - R_sk_11 * inv(R_11) * R_12
        // Side note: might need to be careful when d = b_sz.
        // Cannot perform trmm here as an alternative, since matrix difference is involved.
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, cols - b_sz, b_sz, (T) -1.0, R_sk, d, R12, lda, (T) 1.0, &R_sk[d * b_sz], d, lapack_queue);

        // Changing the sampling dimension parameter
        sampling_dimension = std::min(sampling_dimension, cols);

        // Need to zero out the lower triangular portion of R_sk_22
        // Make sure R_sk_22 exists.
        if (sampling_dimension - b_sz > 0) {
            RandLAPACK::cuda_kernels::get_U_gpu(strm, sampling_dimension - b_sz, sampling_dimension - b_sz, &R_sk[(d + 1) * b_sz], d);
        }

        // Changing the pointer to relevant data in A_sk - this is equaivalent to copying data over to the beginning of A_sk.
        // Remember that the only "active" portion of A_sk remaining would be of size sampling_dimension by cols;
        // if any rows beyond that would be accessed, we would have issues. 

        A_sk_work = &A_sk_work[d * b_sz];

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;

        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            sample_update_t_stop  = steady_clock::now();
            sample_update_t_dur  += duration_cast<microseconds>(sample_update_t_stop - sample_update_t_start).count();
        }
        nvtxRangePop();
    }
    return 0;
}

} // end namespace RandLAPACK
