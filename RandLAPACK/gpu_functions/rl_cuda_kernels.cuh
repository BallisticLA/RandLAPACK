// This consitional allows us to make sure that the cuda kernels are only compiled with nvcc.

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace RandLAPACK::cuda_kernels {

#ifdef USE_CUDA

/** Given the dimensions of a matrix decompose the work for CUDA.
 * @param[in] m number of rows
 * @param[in] n number of cols
 * @param[in] lda the number of elements between each row
 * @returns a tuple of 1. thread grid, 2. block grid, 3. number of blocks
 */

template <typename T>
__global__ void __launch_bounds__(128) elementwise_product(
int64_t n,
const T alpha,
const T* A,
int64_t lda,
const T* B,
int64_t ldb,
T* C,
int64_t ldc
){
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        C[id * ldc] = alpha * A[id * lda] * B[id * ldb];
    }
}


template <typename T>
__global__ void __launch_bounds__(1) diagonal_update(
    int i,
    T* A,
    int64_t lda,
    T* D,
    int64_t n
){
    if (threadIdx.x != 0) {
        return;
    }
    // S(i, i) = − sgn(Q(i, i)); = 1 if Q(i, i) == 0
    T buf = A[i * lda + i];
    T new_d = (buf == 0) ? 1 : -((buf > T{0}) - (buf < T{0}));
    A[i * lda + i] -= new_d;
    D[i] = new_d;
}

template <typename T>
__global__ void __launch_bounds__(128) copy_gpu(
    int64_t n, 
    T const* src, 
    int64_t incsrc, 
    T* dest, 
    int64_t incdest
) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        dest[id * incdest] = src[id * incsrc];
    }
}

template <typename T>
__global__ void __launch_bounds__(128) fill_gpu(
    int64_t n, 
    T* x, 
    int64_t incx, 
    T alpha
) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        x[id * incx] = alpha;
    }
}

template <typename T>
__global__ void __launch_bounds__(128) fill_gpu(
    int64_t n, 
    int64_t* x, 
    int64_t incx, 
    T alpha
) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        x[id * incx] = alpha;
    }
}

template <typename T>
__device__ void copy_gpu_device(
    int64_t n,
    T const* src,
    int64_t incsrc,
    T* dest,
    int64_t incdest
) {
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
    int64_t incb
) {
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        T const v{a[id * inca]};
        a[id * inca] = b[id * incb];
        b[id * incb] = v;
    }
}

// This operation is actually not parallelizable, as the vector b may contain repeated indices 
// that need to be accessed in a sepcific order. 
template <typename Idx>
__global__ void __launch_bounds__(128) LUQRCP_piv_porcess_gpu_global(
    Idx cols,
    Idx n, 
    Idx* a,  
    Idx* b
) {
    #pragma unroll
    for (Idx i = threadIdx.x; i < cols; i += blockDim.x) {
        a[i] = i + 1;
    }
    __syncthreads();
    if (Idx const id = (Idx)blockDim.x * blockIdx.x + threadIdx.x; id == 0) {
        for(int i = 0; i < n; ++ i) {
            std::swap(a[b[i] - 1], a[i]);
        }
    }
}

template <typename T>
__global__ void R_cholqr_signs_gpu(
    int64_t b_sz,
    int64_t b_sz_const, 
    T* R,  
    T* D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < b_sz && j <= i) {
        R[(b_sz_const * i) + j] *= D[j];
    }
}       

template <typename T>
__global__  void __launch_bounds__(128) axpy_gpu(
    int64_t n, 
    T alpha, 
    T *x, 
    int64_t incx, 
    T *y, 
    int64_t incy
) {
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
    int64_t incy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i * incy] = alpha * x[i * incx] + y[i * incy];
    }
}

template <typename T>
__global__  void __launch_bounds__(128) scal_gpu(
    int64_t n, 
    T* alpha, 
    bool inv_alpha,
    T *y, 
    int64_t incy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (inv_alpha) {
            y[i * incy] = (1 / *alpha) * y[i * incy];
        } else {
            y[i * incy] = *alpha * y[i * incy];
        }
    }
}

template <typename T>
__device__  void scal_gpu_device(
    int64_t n, 
    T alpha, 
    T *y, 
    int64_t incy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i * incy] = alpha * y[i * incy];
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

template <typename T>
__device__  void ger_gpu_device(
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

template <typename T>
__global__  void __launch_bounds__(128) get_U_gpu(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) {
        std::fill(&A[i * (lda + 1) + 1], &A[(i * lda) + m], 0.0);
    }
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array is to modified modified ONLY within the scope of this function.
template <typename T>
__global__ void __launch_bounds__(128) col_swap_gpu_kernel(
    int64_t m, 
    int64_t n, 
    int64_t k,
    T* A, 
    int64_t lda,
    int64_t const* idx,
    int64_t* idx_mut
) {
    if (k > n) {
        return;
    }
    cooperative_groups::grid_group g = cooperative_groups::this_grid();
    A -= lda;
    for (int64_t i = threadIdx.x + blockDim.x * blockIdx.x; i < k; i += blockDim.x * gridDim.x) {
        idx_mut[i] = idx[i];
    }
    g.sync();
    int64_t* curr = idx_mut;
    int64_t* end = idx_mut + k;
    for (int64_t i = 1; i <= k; ++i, ++curr) {
        // swap rows IFF mismatched
        if (int64_t const j = *curr; i != j) {
            for (int64_t l = threadIdx.x + blockDim.x * blockIdx.x; l < m; l += blockDim.x * gridDim.x) {
                std::iter_swap(A + i * lda + l, A + j * lda + l);
            }
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                std::iter_swap(curr, std::find(curr, end, i));
            }
            g.sync();
        }
    }
}


template <typename T>
__global__ void __launch_bounds__(512) vec_elem_swap_gpu(
    int64_t m, 
    int64_t k,
    int64_t* J, 
    int64_t const* idx,
    int64_t * idx_mut
) {
    J -= 1;
    for (int64_t i = threadIdx.x; i < k; i += blockDim.x) {
        idx_mut[i] = idx[i];
    }
    __syncthreads();
    int64_t* curr = idx_mut;
    int64_t* end = idx_mut + k;
    for (int64_t i = 1; i <= k; ++i, ++curr) {
        // swap rows IFF mismatched
        if (int64_t const j = *curr; i != j) {
            if (threadIdx.x == 0) {
                std::iter_swap(J + i, J + j);
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
__global__ void transposition_gpu(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* AT,
    int64_t ldat,
    bool copy_upper_triangle
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

template <typename T>
__global__ void copy_mat_gpu(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* A_cpy,
    int64_t ldac,
    bool copy_upper_triangle
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (copy_upper_triangle) {
        // Only copying the upper-triangular portion of the original
        #pragma unroll
        if (i < n && j <= i) {
            A_cpy[(ldac * i) + j] = A[(lda * i) + j];
        }
    } else {
        #pragma unroll
        if (i < n && j <= m) {
            A_cpy[(ldac * i) + j] = A[(lda * i) + j];
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
    //constexpr int threadsPerBlock{128};
    //int64_t num_blocks{(m + threadsPerBlock - 1) / threadsPerBlock};
    //col_swap_gpu<<<num_blocks, threadsPerBlock, sizeof(int64_t) * n, stream>>>(m, n, k, A, lda, idx);
    int64_t* idx_copy;
    cudaMallocAsync(&idx_copy, sizeof(int64_t) * n, stream);
    
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to allocate for col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
    int numBlocksPerSm = 0;
    int numThreads = 128;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, /*device_id=*/0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, col_swap_gpu_kernel<T>, numThreads, 0);
    void* kernelArgs[] = {(void*)&m, (void*)&n, (void*)&k, (void*)&A, (void*)&lda, (void*)&idx, (void*)&idx_copy};
    dim3 dimBlock(numThreads, 1, 1);
    int upper_bound = deviceProp.multiProcessorCount * numBlocksPerSm;
    int lower_bound = std::max(m, k);
    lower_bound = (lower_bound + numThreads - 1) / numThreads;
    dim3 dimGrid(std::min(upper_bound, lower_bound), 1, 1);
    cudaLaunchCooperativeKernel((void*)col_swap_gpu_kernel<T>, dimGrid, dimBlock, kernelArgs, 0, stream);
    
    ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
    cudaFreeAsync(idx_copy, stream);
    ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to deallocate for col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#endif
}

/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T>
void col_swap_gpu(
    cudaStream_t stream, 
    int64_t m,  
    int64_t k,
    int64_t* J, 
    int64_t const* idx
) {
#ifdef USE_CUDA
    //constexpr int threadsPerBlock{128};
    //int64_t num_blocks{(k + threadsPerBlock - 1) / threadsPerBlock};
    int64_t* idx_copy;
    cudaMallocAsync(&idx_copy, sizeof(int64_t) * k, stream);
    vec_elem_swap_gpu<T><<<1, 512, sizeof(int64_t) * k, stream>>>(m, k, J, idx, idx_copy);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch vector col_swap_gpu. " << cudaGetErrorString(ierr))
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
    bool copy_upper_triangle
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

template <typename T>
void copy_mat_gpu(
    cudaStream_t stream, 
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda,
    T* A_cpy,
    int64_t ldat,
    bool copy_upper_triangle
) {
#ifdef USE_CUDA
    dim3 threadsPerBlock(11, 11);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    copy_mat_gpu<<<numBlocks, threadsPerBlock, 0, stream>>>(m, n, A, lda, A_cpy, ldat, copy_upper_triangle);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch copy_mat_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}


// This function is the GPU equivalent of the operations performed on the pivot vector
// in the context of PLU-based QRCP.
// The implementation is very clumsy, but we needed to put it on a GPU.
// TODO: we get a multiple definition linker error if this function is not templated.
template <typename Idx>
inline void LUQRCP_piv_porcess_gpu(
    cudaStream_t stream, 
    Idx sampling_dim,
    Idx cols,
    Idx* J_buffer,
    Idx* J_buffer_lu
) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    Idx n{std::min(sampling_dim, cols)};
    LUQRCP_piv_porcess_gpu_global<<<1, threadsPerBlock, 0, stream>>>(cols, n, J_buffer, J_buffer_lu);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch piv_process_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

// Custom implementation of orhr_col.
// Outputs tau instead of T.
template <typename T>
void orhr_col_gpu(
    cudaStream_t stream, 
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* tau,
    T* D
) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    // We assume that the space for S, D has ben pre-allocated
    
    int i;
    for(i = 0; i < n; ++i) {
        int64_t num_blocks{std::max((m - (i + 1) + threadsPerBlock - 1) / threadsPerBlock, (int64_t) 1)};
        // S(i, i) = − sgn(Q(i, i)); = 1 if Q(i, i) == 0
        diagonal_update<<<1,1,0,stream>>>(i, A, lda, D, n);
        // Scale ith column if L by diagonal element
        scal_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(m - (i + 1), &A[i * (lda + 1)], 1, &A[(lda + 1) * i + 1], 1);
        // Perform Schur compliment update
        // A(i+1:m, i+1:n) = A(i+1:m, i+1:n) - (A(i+1:m, i) * A(i, i+1:n))
        ger_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(m - (i + 1), n - (i + 1), (T) -1.0, &A[(lda + 1) * i + 1], 1, &A[lda * (i + 1) + i], lda, &A[(lda + 1) * (i + 1)], lda);
    }
    int64_t num_blocks{(n + threadsPerBlock - 1) / threadsPerBlock};
    elementwise_product<<<num_blocks, threadsPerBlock, 0, stream>>>(n, T{-1}, A, lda + 1, D, 1, tau, 1);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch orhr_col_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

// Setting signs in the R-factor from CholQR after orhr_col outputs.
template <typename T>
void R_cholqr_signs_gpu(
    cudaStream_t stream, 
    int64_t b_sz,
    int64_t b_sz_const, 
    T* R,  
    T* D
) {
#ifdef USE_CUDA
    dim3 threadsPerBlock(11, 11);
    dim3 numBlocks((b_sz + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (b_sz + threadsPerBlock.y - 1) / threadsPerBlock.y);

    R_cholqr_signs_gpu<<<numBlocks, threadsPerBlock, 0, stream>>>(b_sz, b_sz_const, R, D);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch R_cholqr_signs_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

template <typename T>
void copy_gpu(
    cudaStream_t stream,
    int64_t n,  
    T const* src,
    int64_t incr_src,  
    T* dest, 
    int64_t incr_dest
    ) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{(n + threadsPerBlock - 1) / threadsPerBlock};
    copy_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(n, src, incr_src, dest, incr_dest);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch copy_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

template <typename T>
void fill_gpu(
    cudaStream_t stream,
    int64_t n,  
    T* x,
    int64_t incx,  
    T alpha
    ) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{(n + threadsPerBlock - 1) / threadsPerBlock};
    fill_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(n, x, incx, alpha);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch fill_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

template <typename T>
void fill_gpu(
    cudaStream_t stream,
    int64_t n,  
    int64_t* x,
    int64_t incx,  
    T alpha
    ) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{(n + threadsPerBlock - 1) / threadsPerBlock};
    fill_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(n, x, incx, alpha);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch fill_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

template <typename T>
void get_U_gpu(
    cudaStream_t stream,
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda
    ) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{std::max((n + threadsPerBlock - 1) / threadsPerBlock, (int64_t) 1)};
    get_U_gpu<<<num_blocks, threadsPerBlock, 0, stream>>>(m, n, A, lda);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch get_U_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}

} // end namespace cuda_kernels
