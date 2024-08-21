#ifndef randlapack_cqrrp_gpu_h
#define randlapack_cqrrp_gpu_h

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
class CQRRP_GPU_alg {
    public:

        virtual ~CQRRP_GPU_alg() {}

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

template <typename T, typename RNG>
class CQRRP_blocked_GPU : public CQRRP_GPU_alg<T, RNG> {
    public:
        /// This is a device version of the ICQRRP algorithm - ALL INPUT DATA LIVES ON A GPU.
        ///
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


        CQRRP_blocked_GPU(
            bool time_subroutines,
            T ep,
            int64_t b_sz
        ) {
            timing = time_subroutines;
            eps = ep;
            block_size = b_sz;
            use_qrf = false;
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
        bool use_qrf;
        RandBLAS::RNGState<RNG> state;
        T eps;
        int64_t rank;
        int64_t block_size;
        std::vector<long> times;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRP_blocked_GPU<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* A_sk,
    int64_t d,
    T* tau,
    int64_t* J
){
    high_resolution_clock::time_point preallocation_t_stop;
    high_resolution_clock::time_point preallocation_t_start;
    high_resolution_clock::time_point qrcp_t_start;
    high_resolution_clock::time_point qrcp_t_stop;
    high_resolution_clock::time_point copy_A_sk_t_start;
    high_resolution_clock::time_point copy_A_sk_t_stop;
    high_resolution_clock::time_point qrcp_piv_t_start;
    high_resolution_clock::time_point qrcp_piv_t_stop;
    high_resolution_clock::time_point copy_A_t_start;
    high_resolution_clock::time_point copy_A_t_stop;
    high_resolution_clock::time_point piv_A_t_start;
    high_resolution_clock::time_point piv_A_t_stop;
    high_resolution_clock::time_point preconditioning_t_start;
    high_resolution_clock::time_point preconditioning_t_stop;
    high_resolution_clock::time_point cholqr_t_start;
    high_resolution_clock::time_point cholqr_t_stop;
    high_resolution_clock::time_point orhr_col_t_start;
    high_resolution_clock::time_point orhr_col_t_stop;
    high_resolution_clock::time_point updating_A_t_start;
    high_resolution_clock::time_point updating_A_t_stop;
    high_resolution_clock::time_point updating_J_t_start;
    high_resolution_clock::time_point updating_J_t_stop;
    high_resolution_clock::time_point copy_J_t_start;
    high_resolution_clock::time_point copy_J_t_stop;
    high_resolution_clock::time_point updating_R_t_start;
    high_resolution_clock::time_point updating_R_t_stop;
    high_resolution_clock::time_point updating_Sk_t_start;
    high_resolution_clock::time_point updating_Sk_t_stop;
    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;
    long preallocation_t_dur   = 0;
    long qrcp_t_dur            = 0;
    long copy_A_sk_t_dur       = 0;
    long qrcp_piv_t_dur        = 0;
    long copy_A_t_dur          = 0;
    long piv_A_t_dur           = 0;
    long preconditioning_t_dur = 0;
    long cholqr_t_dur          = 0;
    long orhr_col_t_dur        = 0;
    long updating_A_t_dur      = 0;
    long updating_J_t_dur      = 0;
    long copy_J_t_dur          = 0;
    long updating_R_t_dur      = 0;
    long updating_Sk_t_dur     = 0;
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
    // We will be using this parameter when performing QRCP on a sketch.
    // After the first iteration of the algorithm, this will change its value to min(d, cols) 
    // before "cols" is updated.
    int64_t sampling_dimension = d;

    /******************************STREAM/QUEUE/HANDLE*********************************/
    lapack::Queue lapack_queue(0);
    cudaStream_t strm = lapack_queue.stream();
    cudaStream_t stream_cusolver;
    using lapack::device_info_int;
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
    int *d_info_cusolver = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&d_info_cusolver), sizeof(int));
    // Create cusolver handle - used for the ORMQR call
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnCreate(&cusolverH);

    /******************************WORKSPACE PARAMETERS*********************************/
    char* d_work_getrf, * d_work_geqrf;
    char* h_work_getrf, * h_work_geqrf;
    int lwork_ormqr = 0;
    T *d_work_ormqr = nullptr;
    size_t d_size_getrf, h_size_getrf, d_size_geqrf, h_size_geqrf;

    char* d_work_geqrf_opt;
    char* h_work_geqrf_opt;
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

    // Buffer for the R-factor in Cholesky QR, of size b_sz by b_sz, lda b_sz.
    // Also used to store the proper R11_full-factor after the 
    // full Q has been restored form economy Q (the has been found via Cholesky QR);
    // That is done by applying the sign vector D from orhr_col().
    // Eventually, will be used to store R11 (computed via trmm)
    T* R_cholqr;
    cudaMallocAsync(&R_cholqr, b_sz_const * b_sz_const * sizeof(T), strm);

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
    int64_t* J_copy_col_swap;
    cudaMallocAsync(&J_copy_col_swap, sizeof(int64_t) * n, strm);
    int64_t* J_copy_col_swap_work = J_copy_col_swap;
    // Pointer buffers required for our special data movement-avoiding strategy.
    T* A_sk_buf;
    int64_t* J_cpy_buf;
    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE END*******************

    if(this -> timing) {
        cudaStreamSynchronize(strm);
        lapack_queue.sync();
        preallocation_t_stop  = high_resolution_clock::now();
        preallocation_t_dur   = duration_cast<microseconds>(preallocation_t_stop - preallocation_t_start).count();
    }

    for(iter = 0; iter < maxiter; ++iter) {
        nvtxRangePushA("Iteration");
        // Make sure we fit into the available space
        b_sz = std::min(this->block_size, std::min(m, n) - curr_sz);

        // Zero-out data - may not be necessary
        cudaMemsetAsync(J_buffer_lu, (T) 0.0, std::min(d, n), strm);
        cudaMemsetAsync(J_buffer,    (T) 0.0, n, strm);
        cudaMemsetAsync(Work2,       (T) 0.0, n, strm);

        // Perform pivoted LU on A_sk', follow it up by unpivoted QR on a permuted A_sk.
        // Get a transpose of A_sk
        if(this -> timing) {
            nvtxRangePushA("qrcp");
            qrcp_t_start = high_resolution_clock::now();
        }
        RandLAPACK::cuda_kernels::transposition_gpu(strm, sampling_dimension, cols, A_sk_work, d, A_sk_trans, n, 0);
    
        // Perform a row-pivoted LU on a transpose of A_sk
        // Probing workspace size - performed only once.
        if(iter == 0) {
            lapack::getrf_work_size_bytes(cols, sampling_dimension, A_sk_trans, n, &d_size_getrf, &h_size_getrf, lapack_queue);

            d_work_getrf = blas::device_malloc< char >( d_size_getrf, lapack_queue );
            std::vector<char> h_work_getrf_vector( h_size_getrf );
            h_work_getrf = h_work_getrf_vector.data();
        }
        lapack::getrf(cols, sampling_dimension, A_sk_trans, n, J_buffer_lu, d_work_getrf, d_size_getrf, h_work_getrf, h_size_getrf, d_info, lapack_queue);
        // Fill the pivot vector, apply swaps found via lu on A_sk'.
        RandLAPACK::cuda_kernels::LUQRCP_piv_porcess_gpu(strm, sampling_dimension, cols, J_buffer, J_buffer_lu);
        
        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePushA("copy_A_sk");
            copy_A_sk_t_start = high_resolution_clock::now();
        }
        // Instead of copying A_sk_work into A_sk_copy_col_swap, we ``swap'' the pointers.
        // This is safe, as A_sk is not needed outside of ICQRRP.
        std::swap(A_sk_copy_col_swap, A_sk_work);

        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            copy_A_sk_t_stop = high_resolution_clock::now();
            copy_A_sk_t_dur += duration_cast<microseconds>(copy_A_sk_t_stop - copy_A_sk_t_start).count();
            nvtxRangePushA("piv_A_sk");
            qrcp_piv_t_start = high_resolution_clock::now();
        }
        // Apply pivots to A_sk
        RandLAPACK::cuda_kernels::col_swap_gpu(strm, sampling_dimension, cols, cols, A_sk_work, d, A_sk_copy_col_swap, d, J_buffer);
        
        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            qrcp_piv_t_stop = high_resolution_clock::now();
            qrcp_piv_t_dur += duration_cast<microseconds>(qrcp_piv_t_stop - qrcp_piv_t_start).count();
        }

        // Perform an unpivoted QR on A_sk
        if(iter == 0) {
            lapack::geqrf_work_size_bytes(sampling_dimension, cols, A_sk_work, d, &d_size_geqrf, &h_size_geqrf, lapack_queue);
            d_work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
            std::vector<char> h_work_geqrf_vector( h_size_geqrf );
            h_work_geqrf = h_work_geqrf_vector.data();
        }
        lapack::geqrf(sampling_dimension, cols, A_sk_work, d, Work2, d_work_geqrf, d_size_geqrf, h_work_geqrf, h_size_geqrf, d_info, lapack_queue);

        if(this -> timing) {
            lapack_queue.sync();
            nvtxRangePop();
            qrcp_t_stop = high_resolution_clock::now();
            qrcp_t_dur += duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
            nvtxRangePushA("copy_A");
            copy_A_t_start = high_resolution_clock::now();
        }
        // Need to premute trailing columns of the full R-factor.
        // Remember that the R-factor is stored the upper-triangular portion of A.
        // Pivoting the trailing R and the ``current'' A.      
        RandLAPACK::cuda_kernels::copy_mat_gpu(strm, m, cols, &A[lda * curr_sz], lda, A_copy_col_swap, lda, false);    
        
        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            copy_A_t_stop = high_resolution_clock::now();
            copy_A_t_dur += duration_cast<microseconds>(copy_A_t_stop - copy_A_t_start).count();
            nvtxRangePushA("piv_A");
            piv_A_t_start = high_resolution_clock::now();
        }

        RandLAPACK::cuda_kernels::col_swap_gpu(strm, m, cols, cols, &A[lda * curr_sz], lda, A_copy_col_swap, lda, J_buffer);
        // Defining the new "working subportion" of matrix A.
        // In a global sense, below is identical to:
        // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz];
        Work1 = &A_work[lda * b_sz];
        // Define the space representing R_sk (stored in A_sk)
        R_sk = A_sk_work;

        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            piv_A_t_stop  = high_resolution_clock::now();
            piv_A_t_dur  += duration_cast<microseconds>(piv_A_t_stop - piv_A_t_start).count();
            nvtxRangePushA("precond_A");
            preconditioning_t_start = high_resolution_clock::now();
        }
        
        // A_pre = AJ(:, 1:b_sz) * inv(R_sk)
        // Performing preconditioning of the current matrix A.
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, (T) 1.0, R_sk, d, A_work, lda, lapack_queue);
        if(this -> timing) {
            lapack_queue.sync();
            nvtxRangePop();
            preconditioning_t_stop  = high_resolution_clock::now();
            preconditioning_t_dur  += duration_cast<microseconds>(preconditioning_t_stop - preconditioning_t_start).count();
        }

        if(this -> use_qrf) {
            if(this -> timing) {
                nvtxRangePushA("cholqr");
                cholqr_t_start = high_resolution_clock::now();
            }
            // Perform an unpivoted QR on A_sk
            if(iter == 0) {
                lapack::geqrf_work_size_bytes(sampling_dimension, cols, A_sk_work, d, &d_size_geqrf_opt, &h_size_geqrf_opt, lapack_queue);
                d_work_geqrf_opt = blas::device_malloc< char >( d_size_geqrf_opt, lapack_queue );
                std::vector<char> h_work_geqrf_vector_opt( h_size_geqrf_opt );
                h_work_geqrf_opt = h_work_geqrf_vector_opt.data();
            }
            lapack::geqrf(rows, b_sz, A_work, lda, &tau[curr_sz], d_work_geqrf_opt, d_size_geqrf_opt, h_work_geqrf_opt, h_size_geqrf_opt, d_info, lapack_queue);
            //R_cholqr = A_work;
            RandLAPACK::cuda_kernels::copy_mat_gpu(strm, b_sz, b_sz, A_work, lda, R_cholqr, b_sz_const, true);
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                cholqr_t_stop    = high_resolution_clock::now();
                cholqr_t_dur     += duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();
            }
        } else {
            if(this -> timing) {
                nvtxRangePushA("cholqr");
                cholqr_t_start = high_resolution_clock::now();
            }
            // Performing Cholesky QR
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, b_sz, rows, (T) 1.0, A_work, lda, (T) 0.0, R_cholqr, b_sz_const, lapack_queue);
            lapack::potrf(Uplo::Upper,  b_sz, R_cholqr, b_sz_const, d_info, lapack_queue);
            // Compute Q_econ from Cholesky QR
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, (T) 1.0, R_cholqr, b_sz_const, A_work, lda, lapack_queue);
            if(this -> timing) {
                lapack_queue.sync();
                nvtxRangePop();
                cholqr_t_stop    = high_resolution_clock::now();
                cholqr_t_dur     += duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();
                nvtxRangePushA("orhr_col");
                orhr_col_t_start = high_resolution_clock::now();
            }

            // Find Q (stored in A) using Householder reconstruction. 
            // This will represent the full (rows by rows) Q factor form Cholesky QR
            // It would have been really nice to store T right above Q, but without using extra space,
            // it would result in us loosing the first lower-triangular b_sz by b_sz portion of implicitly-stored Q.
            // Filling T without ever touching its lower-triangular space would be a nice optimization for orhr_col routine.
            // This routine is defined in LAPACK 3.9.0. At the moment, LAPACK++ fails to invoke the newest Accelerate library.
            RandLAPACK::cuda_kernels::orhr_col_gpu(strm, rows, b_sz, A_work, lda, &tau[curr_sz], Work2);  
            
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                orhr_col_t_stop  = high_resolution_clock::now();
                orhr_col_t_dur  += duration_cast<microseconds>(orhr_col_t_stop - orhr_col_t_start).count();
            }

            // Need to change signs in the R-factor from Cholesky QR.
            // Signs correspond to matrix D from orhr_col().
            // This allows us to not explicitly compute R11_full = (Q[:, 1:b_sz])' * A_pre.
            RandLAPACK::cuda_kernels::R_cholqr_signs_gpu(strm, b_sz, b_sz_const, R_cholqr, Work2);
        }
        // Perform Q_full' * A_piv(:, b_sz:end) to find R12 and the new "current A."
        // A_piv (Work1) is a rows by cols - b_sz matrix, stored in space of the original A.
        // The first b_sz rows will represent R12.
        // The last rows-b_sz rows will represent the new A.
        // With that, everything is placed where it should be, no copies required.
        // ORMQR proves to be much faster than GEMQRT with MKL.
        if(this -> timing) {
            nvtxRangePushA("update_A");
            updating_A_t_start = high_resolution_clock::now();
        }
        if (iter == 0) {
            // Compute optimal workspace size
            cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rows, cols - b_sz, b_sz, A_work, lda, &tau[iter * b_sz], Work1, lda, &lwork_ormqr);
            // Allocate workspace
            cudaMalloc(reinterpret_cast<void **>(&d_work_ormqr), sizeof(double) * lwork_ormqr);
        }
        cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rows, cols - b_sz, b_sz, A_work, lda, &tau[iter * b_sz], Work1, lda, d_work_ormqr, lwork_ormqr, d_info_cusolver);
        cusolverDnGetStream(cusolverH, &stream_cusolver);
        cudaStreamSynchronize(stream_cusolver);
        if(this -> timing) {
            nvtxRangePop();
            updating_A_t_stop  = high_resolution_clock::now();
            updating_A_t_dur  += duration_cast<microseconds>(updating_A_t_stop - updating_A_t_start).count();
        }
        // Updating pivots
        if(iter == 0) {
            if(this -> timing) {
                nvtxRangePushA("update_J");
                updating_J_t_start = high_resolution_clock::now();
            }
            RandLAPACK::cuda_kernels::copy_gpu(strm, n, J_buffer, 1, J, 1);
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                updating_J_t_stop  = high_resolution_clock::now();
                updating_J_t_dur  += duration_cast<microseconds>(updating_J_t_stop - updating_J_t_start).count();
                nvtxRangePushA("update_R");
                updating_R_t_start = high_resolution_clock::now();
            }
        } else {
            if(this -> timing) {
                nvtxRangePushA("copy_J");
                copy_J_t_start = high_resolution_clock::now();
            }
            // Instead of copying J into J_copy_col_swap, we ``swap'' the pointers.
            // We have to take some precautions when ICQRRP main loop terminates.
            // Since we want J to be accessible and valid outside of ICQRRP, we need to make sure that 
            // its entries were, in fact, computed correctly. 
            //
            // The original memory space that the vector J points to would only contain the correct pivot ranges, computed at EVEN
            // iterations of ICQRRP's main loop.
            // The correct entries from the odd iterations would be contained in the memory space that was originbally pointed to
            // by J_copy_col_swap.
            // Hence, when ICQRRP terminates, we would need to copy the results from the odd iterations form J_copy_col_swap to J.
            //
            // Remember that since the pointers J and J_copy_col_swap are swapped at every even iteration of the main ICQRRP loop,
            // if the ICQRRP terminates with iter being even, we would need to swap these pointers back around.
            //
            // Additional thing to remember is that the final copy needs to be performed in terms of b_sz_const, not b_sz.
            // No need to worry about the altered b_sz when performing a copy, because it is always placed where it should be in J.
            std::swap(J_copy_col_swap, J);
            J_work = &J[curr_sz];
            J_copy_col_swap_work = &J_copy_col_swap[curr_sz];

            if(this -> timing) {
                nvtxRangePop();
                copy_J_t_stop  = high_resolution_clock::now();
                copy_J_t_dur  += duration_cast<microseconds>(copy_J_t_stop - copy_J_t_start).count();
                nvtxRangePushA("update_J");
                updating_J_t_start = high_resolution_clock::now();
            }
            RandLAPACK::cuda_kernels::col_swap_gpu<T>(strm, cols, cols, J_work, J_copy_col_swap_work, J_buffer);
            if(this -> timing) {
                cudaStreamSynchronize(strm);
                nvtxRangePop();
                updating_J_t_stop  = high_resolution_clock::now();
                updating_J_t_dur  += duration_cast<microseconds>(updating_J_t_stop - updating_J_t_start).count();
                nvtxRangePushA("update_R");
                updating_R_t_start = high_resolution_clock::now();
            }
        }

        // Alternatively, instead of trmm + copy, we could perform a single gemm.
        // Compute R11 = R11_full(1:b_sz, :) * R_sk
        // R11_full is stored in R_cholqr space, R_sk is stored in A_sk space.
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, b_sz, b_sz, (T) 1.0, R_sk, d, R_cholqr, b_sz_const, lapack_queue);
        // Need to copy R11 over form R_cholqr into the appropriate space in A.
        // We cannot avoid this copy, since trmm() assumes R_cholqr is a square matrix.
        // In a global sense, this is identical to:
        // R11 =  &A[(m + 1) * curr_sz];
        R11 = A_work;
        RandLAPACK::cuda_kernels::copy_mat_gpu(strm, b_sz, b_sz, R_cholqr, b_sz_const, A_work, lda, true);

        // Updating the pointer to R12
        // In a global sense, this is identical to:
        // R12 =  &A[(m * (curr_sz + b_sz)) + curr_sz];
        R12 = &R11[lda * b_sz];
        if(this -> timing) {
            cudaStreamSynchronize(strm);
            nvtxRangePop();
            updating_R_t_stop  = high_resolution_clock::now();
            updating_R_t_dur  += duration_cast<microseconds>(updating_R_t_stop - updating_R_t_start).count();
        }

        // Size of the factors is updated;
        curr_sz += b_sz;

        if(curr_sz >= n) {
            // Termination criteria reached
            this -> rank = curr_sz;
            // Measures taken to insure J holds correct data, explained above.
            if(iter % 2) {
                // Total number of iterations is even (iter starts at 0)
                J_cpy_buf = J_copy_col_swap;
                J_copy_col_swap = J;
                J = J_cpy_buf;
            }
            for(int odd_idx = 1; odd_idx < iter; odd_idx += 2) {
                RandLAPACK::cuda_kernels::copy_gpu(strm, b_sz_const, &J_copy_col_swap[odd_idx * b_sz_const], 1, &J[odd_idx * b_sz_const], 1);
            }
            lapack_queue.sync();
            
            if(this -> timing) {
                total_t_stop = high_resolution_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest  = total_t_dur - (preallocation_t_dur + qrcp_t_dur + copy_A_t_dur + piv_A_t_dur + preconditioning_t_dur + cholqr_t_dur + orhr_col_t_dur + updating_A_t_dur + copy_J_t_dur + updating_J_t_dur + updating_R_t_dur + updating_Sk_t_dur);
                this -> times.resize(18);
                auto qrcp_main_t_dur = qrcp_t_dur - qrcp_piv_t_dur - copy_A_sk_t_dur;
                this -> times = {n, b_sz_const, preallocation_t_dur, qrcp_main_t_dur, copy_A_sk_t_dur, qrcp_piv_t_dur, copy_A_t_dur, piv_A_t_dur, preconditioning_t_dur, cholqr_t_dur, orhr_col_t_dur, updating_A_t_dur, copy_J_t_dur, updating_J_t_dur, updating_R_t_dur, updating_Sk_t_dur, t_rest, total_t_dur};

                printf("\n\n/------------ICQRRP TIMING RESULTS BEGIN------------/\n");
                printf("Preallocation time: %25ld μs,\n",                  preallocation_t_dur);
                printf("QRCP main time: %36ld μs,\n",                      qrcp_main_t_dur);
                printf("Copy(A_sk) time: %24ld μs,\n",                     copy_A_sk_t_dur);
                printf("QRCP piv time: %36ld μs,\n",                       qrcp_piv_t_dur);
                printf("Copy(A) time: %24ld μs,\n",                        copy_A_t_dur);
                printf("Piv(A) time: %24ld μs,\n",                         piv_A_t_dur);
                printf("Preconditioning time: %24ld μs,\n",                preconditioning_t_dur);
                printf("CholQR time: %32ld μs,\n",                         cholqr_t_dur);
                printf("ORHR_col time: %7ld μs,\n",                        orhr_col_t_dur);
                printf("Computing A_new, R12 time: %23ld μs,\n",           updating_A_t_dur);
                printf("Copy(J) time: %24ld μs,\n",                        copy_J_t_dur);
                printf("J updating time: %23ld μs,\n",                     updating_J_t_dur);
                printf("R updating time: %23ld μs,\n",                     updating_R_t_dur);
                printf("Sketch updating time: %24ld μs,\n",                updating_Sk_t_dur);
                printf("Other routines time: %24ld μs,\n",                 t_rest);
                printf("Total time: %35ld μs.\n",                          total_t_dur);

                printf("\nPreallocation takes %22.2f%% of runtime.\n",                  100 * ((T) preallocation_t_dur   / (T) total_t_dur));
                printf("QRCP main takes %32.2f%% of runtime.\n",                        100 * ((T) qrcp_main_t_dur       / (T) total_t_dur));
                printf("Cpy(A_sk) takes %20.2f%% of runtime.\n",                        100 * ((T) copy_A_sk_t_dur       / (T) total_t_dur));
                printf("QRCP piv takes %32.2f%% of runtime.\n",                         100 * ((T) qrcp_piv_t_dur        / (T) total_t_dur));
                printf("Cpy(A) takes %20.2f%% of runtime.\n",                           100 * ((T) copy_A_t_dur          / (T) total_t_dur));
                printf("Piv(A) takes %20.2f%% of runtime.\n",                           100 * ((T) piv_A_t_dur           / (T) total_t_dur));
                printf("Preconditioning takes %20.2f%% of runtime.\n",                  100 * ((T) preconditioning_t_dur / (T) total_t_dur));
                printf("Cholqr takes %29.2f%% of runtime.\n",                           100 * ((T) cholqr_t_dur          / (T) total_t_dur));
                printf("Orhr_col takes %22.2f%% of runtime.\n",                         100 * ((T) orhr_col_t_dur        / (T) total_t_dur));
                printf("Computing A_new, R12 takes %14.2f%% of runtime.\n",             100 * ((T) updating_A_t_dur      / (T) total_t_dur));
                printf("Cpy(J) takes %20.2f%% of runtime.\n",                           100 * ((T) copy_J_t_dur          / (T) total_t_dur));
                printf("J updating time takes %20.2f%% of runtime.\n",                  100 * ((T) updating_J_t_dur      / (T) total_t_dur));
                printf("R updating time takes %20.2f%% of runtime.\n",                  100 * ((T) updating_R_t_dur      / (T) total_t_dur));
                printf("Sketch updating time takes %15.2f%% of runtime.\n",             100 * ((T) updating_Sk_t_dur     / (T) total_t_dur));
                printf("Everything else takes %20.2f%% of runtime.\n",                  100 * ((T) t_rest                / (T) total_t_dur));
                printf("/-------------ICQRRP TIMING RESULTS END-------------/\n\n");
            }

            cudaFree(A_sk_trans);
            cudaFree(J_buffer);
            cudaFree(J_buffer_lu);
            cudaFree(Work2);
            cudaFree(R_cholqr);
            cudaFree(d_work_ormqr);
            cudaFree(A_copy_col_swap);
            cudaFree(A_sk_copy_col_swap);
            cudaFree(J_copy_col_swap);

            return 0;
        }
        if(this -> timing) {
            nvtxRangePushA("update_Sk");
            updating_Sk_t_start = high_resolution_clock::now();
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
            updating_Sk_t_stop  = high_resolution_clock::now();
            updating_Sk_t_dur  += duration_cast<microseconds>(updating_Sk_t_stop - updating_Sk_t_start).count();
        }
        nvtxRangePop();
    }
    return 0;
}

} // end namespace RandLAPACK
#endif