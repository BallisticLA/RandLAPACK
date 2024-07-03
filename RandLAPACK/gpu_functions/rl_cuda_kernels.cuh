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

template <typename T>
__global__ void __launch_bounds__(128) copy_gpu(int64_t n, T const* src, int64_t incr_src, T* dest, int64_t incr_dest) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        dest[id * incr_dest] = src[id * incr_src];
    }
}

template <typename T>
__global__ void __launch_bounds__(128) swap_gpu(T* a, int64_t incr_a, T* b, int64_t n, int64_t incr_b) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        T const v{a[id * incr_a]};
        a[id * incr_a] = b[id * incr_b];
        b[id * incr_b] = v;
    }
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array is to modified modified ONLY within the scope of this function.
template <typename T>
__global__ void __launch_bounds__(128) col_swap_gpu(
    int64_t m, int64_t n, int64_t k,
    T* A, int64_t lda,
    int64_t const* idx
) {
    extern __shared__ int64_t local_idx[];
    if (k > n) {
        return;
    }
    A -= lda;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        local_idx[i] = idx[i];
    }
    __syncthreads();
    int64_t* curr = local_idx;
    int64_t* end = local_idx + k;
    for (int64_t i = 1; i <= k; ++i, ++curr) {
        // swap rows IFF mismatched
        if (int64_t const j = *curr; i != j) {
            for (int64_t l = blockIdx.x * blockDim.x + threadIdx.x; l < m; l += blockDim.x * gridDim.x) {
                std::iter_swap(A + i * lda + l, A + j * lda + l);
            }
            if (threadIdx.x == 0) {
                std::iter_swap(curr, std::find(curr, end, i));
            }
            __syncthreads();
        }
    }
}

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
    int copy_upper_triangle,
    int64_t num_blocks,
    int64_t threadsPerBlock
) {
    if (copy_upper_triangle) {
        // Only transposing the upper-triangular portion of the original
        for(int64_t i = threadIdx.x; i < n; i += blockDim.x)
            copy_gpu<<<num_blocks, threadsPerBlock>>>(i + 1, &A[i * lda], 1, &AT[i], ldat);
            __syncthreads();
    } else {
        for(int64_t i = threadIdx.x; i < n; i += blockDim.x)
            copy_gpu<<<num_blocks, threadsPerBlock>>>(m, &A[i * lda], 1, &AT[i], ldat);
            __syncthreads();
    }
}

#endif


template <typename T>
void copy_gpu(cudaStream_t stream, int64_t n, T const* src, int64_t incr_src, T* dest, int64_t incr_dest) {
    
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{(n + threadsPerBlock - 1) / threadsPerBlock};
    copy_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(n, src, incr_src, dest, incr_dest);
    cudaStreamSynchronize(stream);
#endif
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T>
void col_swap_gpu(
    cudaStream_t stream, 
    int64_t m, 
    int64_t n, 
    int64_t k,
    T* A, 
    int64_t lda,
    int64_t const* idx
) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{(n + threadsPerBlock - 1) / threadsPerBlock};
    col_swap_gpu<<<num_blocks, threadsPerBlock, sizeof(int64_t) * n, stream>>>(m, n, k, A, lda, idx);
    cudaStreamSynchronize(stream);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}


template <typename T>
void transposition_gpu(
    cudaStream_t stream, 
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* AT,
    int64_t ldat,
    int copy_upper_triangle
) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks_copy{(m + threadsPerBlock - 1) / threadsPerBlock};
    int64_t num_blocks_transposition{(n + threadsPerBlock - 1) / threadsPerBlock};
    transposition_gpu<<<num_blocks_transposition, threadsPerBlock, 0, stream>>>(m, n, A, lda, AT, ldat, copy_upper_triangle, num_blocks_copy, threadsPerBlock);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch transposition_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}
} // end namespace cuda_kernels
