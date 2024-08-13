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
        bool cond_check;
        RandBLAS::RNGState<RNG> state;
        T eps;
        int64_t rank;
        int64_t block_size;
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
    cudaMalloc(&J_buffer, n * sizeof(int64_t));

    // Special pivoting buffer for LU factorization, capturing the swaps on A_sk'.
    // Needs to be converted in a proper format of length rows(A_sk')
    int64_t* J_buffer_lu;
    cudaMalloc(&J_buffer_lu, std::min(d, n) * sizeof(int64_t));

    // Pointer to the b_sz by b_sz upper-triangular facor R stored in A_sk after GEQP3.
    T* R_sk = NULL;

    // View to the transpose of A_sk.
    // Is of size n * d, with an lda n.   
    T* A_sk_trans;
    cudaMalloc(&A_sk_trans, n * d * sizeof(T));

    // Buffer for the R-factor in Cholesky QR, of size b_sz by b_sz, lda b_sz.
    // Also used to store the proper R11_full-factor after the 
    // full Q has been restored form economy Q (the has been found via Cholesky QR);
    // That is done by applying the sign vector D from orhr_col().
    // Eventually, will be used to store R11 (computed via trmm)
    T* R_cholqr;
    cudaMalloc(&R_cholqr, b_sz_const * b_sz_const * sizeof(T));

    // Buffer for Tau in GEQP3 and D in orhr_col, of size n.
    T* Work2;
    cudaMalloc(&Work2, n * sizeof(T));
    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE END*******************

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

    if(this -> timing) {
        cudaStreamSynchronize(strm);
        lapack_queue.sync();
        preallocation_t_stop  = high_resolution_clock::now();
        preallocation_t_dur   = duration_cast<microseconds>(preallocation_t_stop - preallocation_t_start).count();
    }

    for(iter = 0; iter < maxiter; ++iter) {
        nvtxRangePushA("iter");
        // Make sure we fit into the available space
        b_sz = std::min(this->block_size, std::min(m, n) - curr_sz);

        // Zero-out data - may not be necessary
        cudaMemsetAsync(J_buffer_lu, (T) 0.0, std::min(d, n), strm);
        cudaMemsetAsync(J_buffer,    (T) 0.0, n, strm);
        cudaMemsetAsync(Work2,       (T) 0.0, n, strm);

        // Perform pivoted LU on A_sk', follow it up by unpivoted QR on a permuted A_sk.
        // Get a transpose of A_sk
        {
            nvtxRangePushA("qrcp");
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
            // Apply pivots to A_sk
            RandLAPACK::cuda_kernels::col_swap_gpu(strm, sampling_dimension, cols, cols, A_sk_work, d, J_buffer);

            // Perform an unpivoted QR on A_sk
            if(iter == 0) {
                lapack::geqrf_work_size_bytes(sampling_dimension, cols, A_sk_work, d, &d_size_geqrf, &h_size_geqrf, lapack_queue);
                d_work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
                std::vector<char> h_work_geqrf_vector( h_size_geqrf );
                h_work_geqrf = h_work_geqrf_vector.data();
            }
            lapack::geqrf(sampling_dimension, cols, A_sk_work, d, Work2, d_work_geqrf, d_size_geqrf, h_work_geqrf, h_size_geqrf, d_info, lapack_queue);
            lapack_queue.sync();
            nvtxRangePop();
        }
           
        // Need to premute trailing columns of the full R-factor.
        // Remember that the R-factor is stored the upper-triangular portion of A.
        if(iter != 0) {
            nvtxRangePushA("piv_R");
            RandLAPACK::cuda_kernels::col_swap_gpu(strm, curr_sz, cols, cols, &A[lda * curr_sz], m, J_buffer);
            cudaStreamSynchronize(strm);
            nvtxRangePop();
        }
        {
            nvtxRangePushA("piv(A) + precond(A)");
            // Pivoting the current matrix A.
            RandLAPACK::cuda_kernels::col_swap_gpu(strm, rows, cols, cols, A_work, lda, J_buffer);
            // Defining the new "working subportion" of matrix A.
            // In a global sense, below is identical to:
            // Work1 = &A[(lda * (iter + 1) * b_sz) + curr_sz];
            Work1 = &A_work[lda * b_sz];

            // Define the space representing R_sk (stored in A_sk)
            R_sk = A_sk_work;

            // A_pre = AJ(:, 1:b_sz) * inv(R_sk)
            // Performing preconditioning of the current matrix A.
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, (T) 1.0, R_sk, d, A_work, lda, lapack_queue);
            cudaStreamSynchronize(strm);
            nvtxRangePop();
        }
        {
            nvtxRangePushA("CholQR");
            // Performing Cholesky QR
            blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, b_sz, rows, (T) 1.0, A_work, lda, (T) 0.0, R_cholqr, b_sz_const, lapack_queue);
            //lapack::potrf(Uplo::Upper, b_sz, R_cholqr, b_sz_const);
            lapack::potrf(Uplo::Upper,  b_sz, R_cholqr, b_sz_const, d_info, lapack_queue);

            // Compute Q_econ from Cholesky QR
            blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, (T) 1.0, R_cholqr, b_sz_const, A_work, lda, lapack_queue);

            // Find Q (stored in A) using Householder reconstruction. 
            // This will represent the full (rows by rows) Q factor form Cholesky QR
            // It would have been really nice to store T right above Q, but without using extra space,
            // it would result in us loosing the first lower-triangular b_sz by b_sz portion of implicitly-stored Q.
            // Filling T without ever touching its lower-triangular space would be a nice optimization for orhr_col routine.
            // This routine is defined in LAPACK 3.9.0. At the moment, LAPACK++ fails to invoke the newest Accelerate library.
            RandLAPACK::cuda_kernels::orhr_col_gpu(strm, rows, b_sz, A_work, lda, &tau[curr_sz], Work2);  
            
            // Need to change signs in the R-factor from Cholesky QR.
            // Signs correspond to matrix D from orhr_col().
            // This allows us to not explicitoly compute R11_full = (Q[:, 1:b_sz])' * A_pre.
            RandLAPACK::cuda_kernels::R_cholqr_signs_gpu(strm, b_sz, b_sz_const, R_cholqr, Work2);
            lapack_queue.sync();
            nvtxRangePop();
        }
        // Perform Q_full' * A_piv(:, b_sz:end) to find R12 and the new "current A."
        // A_piv (Work1) is a rows by cols - b_sz matrix, stored in space of the original A.
        // The first b_sz rows will represent R12.
        // The last rows-b_sz rows will represent the new A.
        // With that, everything is placed where it should be, no copies required.
        // ORMQR proves to be much faster than GEMQRT with MKL.
        {
            nvtxRangePushA("update_A");
            if (iter == 0) {
                // Compute optimal workspace size
                cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rows, cols - b_sz, b_sz, A_work, lda, &tau[iter * b_sz], Work1, lda, &lwork_ormqr);
                // Allocate workspace
                cudaMalloc(reinterpret_cast<void **>(&d_work_ormqr), sizeof(double) * lwork_ormqr);
            }
            cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, rows, cols - b_sz, b_sz, A_work, lda, &tau[iter * b_sz], Work1, lda, d_work_ormqr, lwork_ormqr, d_info_cusolver);
            cusolverDnGetStream(cusolverH, &stream_cusolver);
            cudaStreamSynchronize(stream_cusolver);
            nvtxRangePop();
        }
        {
            nvtxRangePushA("update_J");
            // Updating pivots
            if(iter == 0) {
                RandLAPACK::cuda_kernels::copy_gpu(strm, n, J_buffer, 1, J, 1);
            } else {
                RandLAPACK::cuda_kernels::col_swap_gpu<T>(strm, cols, cols, &J[curr_sz], J_buffer);
            }
            cudaStreamSynchronize(strm);
            nvtxRangePop();
        }
        {
            nvtxRangePushA("update_R");
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
            cudaStreamSynchronize(strm);
            nvtxRangePop();
        }

        // Size of the factors is updated;
        curr_sz += b_sz;

        if(curr_sz >= n) {
            // Termination criteria reached
            this -> rank = curr_sz;
            lapack_queue.sync();

            cudaFree(A_sk_trans);
            cudaFree(J_buffer);
            cudaFree(J_buffer_lu);
            cudaFree(Work2);
            cudaFree(R_cholqr);
            cudaFree(d_work_ormqr);

            return 0;
        }
        {
            nvtxRangePushA("update_sketch");
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
            //cudaStreamSynchronize(strm);
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
            nvtxRangePop();
        }
        nvtxRangePop();
    }
    return 0;
}

} // end namespace RandLAPACK
#endif