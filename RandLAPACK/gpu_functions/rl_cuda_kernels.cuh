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
__global__ void __launch_bounds__(128) copy_gpu(
    int64_t n, 
    T const* src, 
    int64_t incsrc, 
    T* dest, 
    int64_t incdest) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        dest[id * incdest] = src[id * incsrc];
    }
}

template <typename T>
__device__ void copy_gpu_device(
    int64_t n,
    T const* src,
    int64_t incsrc,
    T* dest,
    int64_t incdest) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        dest[id * incdest] = src[id * incsrc];
    }
}

template <typename T>
__global__ void __launch_bounds__(128) swap_gpu(
    T* a, 
    int64_t inca, 
    T* b, 
    int64_t n, 
    int64_t incb) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        T const v{a[id * inca]};
        a[id * inca] = b[id * incb];
        b[id * incb] = v;
    }
}

template <typename T>
__global__  void __launch_bounds__(128) axpy_gpu(
    int64_t n, 
    T alpha, 
    T *x, 
    int64_t incx, 
    T *y, 
    int64_t incy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i * incy] = alpha * x[i * incx] + y[i * incy];
    }
}

template <typename T>
__device__  void axpy_gpu_device(
    int64_t n, 
    T alpha, 
    T *x, 
    int64_t incx, 
    T *y, 
    int64_t incy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i * incy] = alpha * x[i * incx] + y[i * incy];
    }
}

template <typename T>
__global__  void __launch_bounds__(128) ger_gpu(
    int64_t m, 
    int64_t n, 
    T alpha, 
    T *x, 
    int64_t incx, 
    T *y, 
    int64_t incy, 
    T* A, 
    int64_t lda
) {
    #pragma unroll
    for(int i = 0; i < n; ++i) {
        axpy_gpu_device(m, alpha * y[i * incy], x, incx, &A[i * lda], 1);
    }
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array is to modified modified ONLY within the scope of this function.
template <typename T>
__global__ void __launch_bounds__(128) col_swap_gpu(
    int64_t m, 
    int64_t n, 
    int64_t k,
    T* A, 
    int64_t lda,
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
    int copy_upper_triangle
) {
    if (copy_upper_triangle) {
        // Only transposing the upper-triangular portion of the original
        #pragma unroll
        for(int64_t i = 0; i < n; ++i){
            copy_gpu_device(i + 1, &A[i * lda], 1, &AT[i], ldat);
        }
    } else {
        #pragma unroll
        for(int64_t i = 0; i < n; ++i){
            copy_gpu_device(m, &A[i * lda], 1, &AT[i], ldat);
        }
    }
}
#endif

template <typename T>
void ger_gpu(
    cudaStream_t stream, 
    int64_t m, 
    int64_t n, 
    T alpha, 
    T *x, 
    int64_t incx, 
    T *y, 
    int64_t incy, 
    T* A, 
    int64_t lda
) {

#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks_ger{(m + threadsPerBlock - 1) / threadsPerBlock};
    ger_gpu<<<num_blocks_ger, threadsPerBlock, 0, stream>>>(m, n, alpha, x, incx, y, incy, A, lda);
#endif
  
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch ger_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
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
    int64_t num_blocks{(m + threadsPerBlock - 1) / threadsPerBlock};
    transposition_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(m, n, A, lda, AT, ldat, copy_upper_triangle);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch transposition_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}
} // end namespace cuda_kernels
