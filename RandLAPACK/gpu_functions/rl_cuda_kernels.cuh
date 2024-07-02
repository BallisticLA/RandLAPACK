// This consitional allows us to make sure that the cuda kernels are only compiled with nvcc.

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>

namespace RandLAPACK::cuda_kernels {

#ifdef USE_CUDA

/** Given the dimensions of a matrix decompose the work for CUDA.
 * @param[in] m number of rows
 * @param[in] n number of cols
 * @param[in] lda the number of elements between each row
 * @returns a tuple of 1. thread grid, 2. block grid, 3. number of blocks
 */

__global__ void __launch_bounds__(128) copy(int64_t* src, int64_t* dest, int64_t n) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        dest[id] = src[id];
    }
}
 
template <typename T>
__device__ void swap(T* a, T* b, int64_t n) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        T const v{a[id]};
        a[id] = b[id];
        b[id] = v;
    }
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array is to modified modified ONLY within the scope of this function.
template <typename T>
 __global__
void col_swap_gpu(    
    int64_t m,
    int64_t n,
    int64_t k,
    T* A,
    int64_t lda,
    int64_t* idx
    )
{
    A -= lda;
    int64_t* end = idx + k;
    for (int64_t i = 1; i <= k; ++i, ++idx) {
        // swap rows IFF mismatched
        if (int64_t const j = *idx; i != j) {
            // swap columns
            swap(A + i * lda, A + j * lda, m); 
            __syncthreads();
            // swap indices
            std::iter_swap(idx, std::find(idx, end, i));
        }
    }
}
/*
// Transposes the input matrix, copying the transposed version into the buffer.
// If an option is passes, stores only the upper-triangular portion of the transposed factor.
// This functioin would require a copy with an adjustible stride.
template <typename T>
 __global__
void transposition_gpu(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* AT,
    int64_t ldat,
    int copy_upper_triangle
) {
    if (copy_upper_triangle) {
        // Only transposing the upper-triangular portion of the original
        for(int i = 0; i < n; ++i)
            blas::copy(i + 1, &A[i * lda], 1, &AT[i], ldat);
    } else {
        for(int i = 0; i < n; ++i)
            blas::copy(m, &A[i * lda], 1, &AT[i], ldat);
    }
}
*/

#endif

/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T>
void col_swap_gpu(    
    int64_t m,
    int64_t n,
    int64_t k,
    T* A,
    int64_t lda,
    int64_t* idx,
    int64_t* temp_buf,
    cudaStream_t strm)
{
#ifdef USE_CUDA
    blas::Queue blas_queue(0);
    // threads per block
    int tpb = 128;
    // num blcoks to spawn
    int nb = (m + tpb - 1) / tpb;
    copy<<<nb, tpb, 0, strm>>>(idx, temp_buf, n);
    cudaStreamSynchronize(strm);
    col_swap_gpu<<<nb, tpb, 0, strm>>>(m, n, k, A, lda, idx);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}
/*
template <typename T>
void transposition_gpu(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* AT,
    int64_t ldat,
    int copy_upper_triangle,
    cudaStream_t strm
) {
#ifdef USE_CUDA
    blas::Queue blas_queue(0);
    // threads per block
    int tpb = 128;
    // num blcoks to spawn
    int nb = (m + tpb - 1) / tpb;
    transposition_gpu<<<nb, tpb, 0, strm>>>(m, n, A, lda, AT, ldat, copy_upper_triangle);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch transposition_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}
*/
} // end namespace cuda_kernels
