#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace RandLAPACK::cuda_kernels {

// This conditional allows us to make sure that the cuda kernels are only compiled with nvcc.
#ifdef USE_CUDA

/** Given the dimensions of a matrix decompose the work for CUDA.
 * @param[in] m number of rows
 * @param[in] n number of cols
 * @param[in] lda the number of elements between each row
 * @returns a tuple of 1. thread grid, 2. block grid, 3. number of blocks
 */

template <typename T>
__global__ void __launch_bounds__(128) all_of(
int64_t n,
const T alpha,
const T* A,
bool* all_greater
){
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ bool indicator;
    if (id == 0) {
        *all_greater = true;
        indicator = true;
    }
    __syncthreads();
    // No need to continue checking the elements if we have already encountered one greater than alpha
    if ((id < n) && (std::abs(A[id]) > alpha) && (*all_greater == true)) {
        indicator = false;
    }
    __syncthreads();
    
    if(id == 0 && !indicator) {
        *all_greater = false;
    }
}

template <typename T>
__global__ void __launch_bounds__(128) naive_rank_est(
int64_t n,
const T alpha,
const T* A,
int64_t lda,
int64_t* rank
){
    int64_t const id = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int64_t indicator;
    if (id == 0) {
        indicator = n;
        *rank = n;
    }
    __syncthreads();

    // As we are looking at the elements of the diagonal of A in parallel,
    // it is important to remember that the actual rank might be smaller 
    // than the currently-defined one, so this should not have an early termination.
    if((id < indicator) && (std::abs(A[id * lda + id]) / std::abs(A[0]) <= alpha)) {
        atomicMin((int*) &indicator, (int) id);
    }
    __syncthreads();
    if(id == 0 && indicator != n) {
        *rank = indicator;
    }
}

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
// that need to be accessed in a specific order. 
template <typename Idx>
__global__ void __launch_bounds__(128) LUQRCP_piv_process_gpu_global(
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
__global__ void __launch_bounds__(128) col_swap_gpu_kernel_sequential(
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

/// Positions columns of A in accordance with idx vector of length k.
/// idx array is to modified modified ONLY within the scope of this function.
template <typename T>
__global__ void col_swap_gpu_kernel(
    int64_t m, 
    int64_t n, 
    int64_t k,
    T* A, 
    int64_t lda,
    T* A_cpy, 
    int64_t ldac,
    int64_t const* idx
) {
    //The commented-out method should be faster, but apparently, something is off with it.
    /*
    for (int64_t colIdx = threadIdx.x + blockDim.x * blockIdx.x; colIdx < k; colIdx += blockDim.x * gridDim.x) {
        for (int64_t rowIdx = threadIdx.y + blockDim.y * blockIdx.y; rowIdx < m; rowIdx += blockDim.y * gridDim.y) {
                int64_t j = idx[colIdx] - 1; 
                A[colIdx * lda + rowIdx] = A_cpy[j * ldac + rowIdx];
            }
        }
    }
    */

    int colIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y;
    if (colIdx < k && rowIdx < m) {
        int64_t j = idx[colIdx] - 1; 
        A[colIdx * lda + rowIdx] = A_cpy[j * ldac + rowIdx];
    }
}

template <typename T>
__global__ void __launch_bounds__(512) vec_ell_swap_gpu(
    int64_t m, 
    int64_t k,
    int64_t* J, 
    int64_t const* idx,
    int64_t* idx_mut
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

template <typename T>
__global__ void vec_ell_swap_gpu(
    int64_t m, 
    int64_t k,
    int64_t* J, 
    int64_t* J_cpy, 
    int64_t const* idx
) {
    for (int64_t i = threadIdx.x + blockDim.x * blockIdx.x; i < k; i += blockDim.x * gridDim.x) {
        int64_t j = idx[i] - 1;
        J[i] = J_cpy[j];
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
    if (copy_upper_triangle) {
        // Only copying the upper-triangular portion of the original
        for (int64_t i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x) {
            for (int64_t j = threadIdx.y + blockDim.y * blockIdx.y; j <= i; j += blockDim.y * gridDim.y) {
                A_cpy[(ldac * i) + j] = A[(lda * i) + j];
            }
        }
    } else {
        for (int64_t i = threadIdx.x + blockDim.x * blockIdx.x; i < n; i += blockDim.x * gridDim.x) {
            for (int64_t j = threadIdx.y + blockDim.y * blockIdx.y; j < m; j += blockDim.y * gridDim.y) {
                A_cpy[(ldac * i) + j] = A[(lda * i) + j];
            }
        }
    }
}
#endif

template <typename T>
void naive_rank_est(
    cudaStream_t stream, 
    int64_t n,
    const T alpha,
    const T* A,
    int64_t lda,
    int64_t& rank
) {
#ifdef USE_CUDA
    int64_t* rank_device;
    cudaMalloc(&rank_device, sizeof(int64_t));

    constexpr int threadsPerBlock{128};
    int64_t num_blocks_ger{(n + threadsPerBlock - 1) / threadsPerBlock};
    naive_rank_est<<<num_blocks_ger, threadsPerBlock, sizeof(int64_t), stream>>>(n, alpha, A, lda, rank_device);

    // It seems that these synchs are required, after all
    cudaStreamSynchronize(stream);
    cudaMemcpy(&rank, rank_device, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaFree(rank_device);
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch naive_rank_est. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
}

template <typename T>
void all_of(
    cudaStream_t stream, 
    int64_t n,
    const T alpha,
    const T* A,
    bool& all_greater
) {
#ifdef USE_CUDA
    bool* all_greater_device;
    cudaMalloc(&all_greater_device, sizeof(bool));

    constexpr int threadsPerBlock{128};
    int64_t num_blocks_ger{(n + threadsPerBlock - 1) / threadsPerBlock};
    all_of<<<num_blocks_ger, threadsPerBlock, sizeof(bool), stream>>>(n, alpha, A, all_greater_device);
    cudaStreamSynchronize(stream);
    cudaMemcpy(&all_greater, all_greater_device, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    cudaFree(all_greater_device);
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch all_greater. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
}

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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch ger_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    copy_mat_gpu<<<numBlocks, threadsPerBlock, 0, stream>>>(m, n, A, lda, A_cpy, ldat, copy_upper_triangle);
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch copy_mat_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
}

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
    int64_t* idx_copy;
    cudaMallocAsync(&idx_copy, sizeof(int64_t) * n, stream);
    
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to allocate for col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
    constexpr int numThreads = 128;
    dim3 dimBlock(numThreads, 1, 1);
    static int upper_bound = [&] {
        int numBlocksPerSm = 0;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, col_swap_gpu_kernel<T>, numThreads, 0);
        return deviceProp.multiProcessorCount * numBlocksPerSm;
    }();
    int lower_bound = std::max(m, k);
    lower_bound = (lower_bound + numThreads - 1) / numThreads;
    dim3 dimGrid(std::min(upper_bound, lower_bound), 1, 1);
    void* kernelArgs[] = {(void*)&m, (void*)&n, (void*)&k, (void*)&A, (void*)&lda, (void*)&idx, (void*)&idx_copy};
    cudaLaunchCooperativeKernel((void*)col_swap_gpu_kernel_sequential<T>, dimGrid, dimBlock, kernelArgs, 0, stream);
    ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch col_swap_gpu sequential. " << cudaGetErrorString(ierr))
        abort();
    }
    cudaFreeAsync(idx_copy, stream);
    ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to deallocate for col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
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
    T* A_cpy,
    int64_t ldac,
    int64_t const* idx
) {
#ifdef USE_CUDA

    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                         (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    col_swap_gpu_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(m, n, k, A, lda, A_cpy, ldac, idx);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch col_swap_gpu with parallel pivots. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
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
    // Commented-out code offloads the data onto CPU.
/*
    std::vector<int64_t> idx_copy(k);
    auto j = std::make_unique<int64_t[]>(m);
    cudaMemcpyAsync(idx_copy.data(), idx, sizeof(int64_t) * k, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(j.get(), J, sizeof(int64_t) * m, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    RandLAPACK::util::col_swap<T>(m, k, j.get(), std::move(idx_copy));

    cudaMemcpyAsync(J, j.get(), sizeof(int64_t) * m, cudaMemcpyHostToDevice, stream);
    // must add this to avoid dangling reference during async copy
    cudaStreamSynchronize(stream);
*/
#ifdef USE_CUDA
    int64_t* idx_copy;
    cudaMallocAsync(&idx_copy, sizeof(int64_t) * k, stream);
    vec_ell_swap_gpu<T><<<1, 512, 0, stream>>>(m, k, J, idx, idx_copy);

    cudaFreeAsync(idx_copy, stream);
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
}

template <typename T>
void col_swap_gpu(
    cudaStream_t stream, 
    int64_t m,  
    int64_t k,
    int64_t* J, 
    int64_t* J_cpy, 
    int64_t const* idx
) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    int64_t num_blocks{(k + threadsPerBlock - 1) / threadsPerBlock};
    vec_ell_swap_gpu<T><<<num_blocks, threadsPerBlock, 0, stream>>>(m, k, J, J_cpy, idx);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch col_swap_gpu vector. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch transposition_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
}


// Transforms the LU pivot format into QR pivot format, used in the context of LU-based QRCP.
// TODO: we get a multiple definition linker error if this function is not templated.
template <typename Idx>
inline void LUQRCP_piv_process_gpu(
    cudaStream_t stream, 
    Idx sampling_dim,
    Idx cols,
    Idx* J_buffer,
    Idx* J_buffer_lu
) {
#ifdef USE_CUDA
    constexpr int threadsPerBlock{128};
    Idx n{std::min(sampling_dim, cols)};
    LUQRCP_piv_process_gpu_global<<<1, threadsPerBlock, 0, stream>>>(cols, n, J_buffer, J_buffer_lu);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch piv_process_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch orhr_col_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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
    dim3 threadsPerBlock(32, 8);
    dim3 numBlocks((b_sz + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (b_sz + threadsPerBlock.y - 1) / threadsPerBlock.y);

    R_cholqr_signs_gpu<<<numBlocks, threadsPerBlock, 0, stream>>>(b_sz, b_sz_const, R, D);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch R_cholqr_signs_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch copy_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch fill_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch fill_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
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

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        RandLAPACK_CUDA_ERROR("Failed to launch get_U_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
#else
    throw std::runtime_error("No CUDA support available.");
#endif
}

} // end namespace cuda_kernels
