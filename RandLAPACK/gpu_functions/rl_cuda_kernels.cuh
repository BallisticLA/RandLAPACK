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
inline
auto partition_1d(size_t n, size_t m, size_t lda, size_t threads_per_block = 128)
{
    (void) m;

    size_t n_elem = n*lda;

    dim3 thread_grid;
    thread_grid.x = threads_per_block;
    thread_grid.y = 1;
    thread_grid.z = 1;

    size_t n_blocks = n_elem / threads_per_block;

    if (n_elem % threads_per_block)
        ++n_blocks;

    dim3 block_grid;
    block_grid.x = n_blocks;
    block_grid.y = 1;
    block_grid.z = 1;

    return std::make_tuple(thread_grid, block_grid);
}

inline
__device__
size_t array_index()
{
    return threadIdx.x + blockDim.x*blockIdx.x;
}

inline
__device__
bool valid_index(size_t q, size_t m, size_t n, size_t lda)
{
    return ((q < m * lda) && ((q % lda) < n));
}

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


template <typename T>
 __global__
void hadamard_product(const T *A, const T *B, T *C, size_t n, size_t m, size_t lda)
{
    size_t q = array_index();

    if (!valid_index(q, m, n, lda))
        return;

    C[q] = A[q] * B[q];
}

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
    col_swap_gpu<<<nb, tpb, 0, strm>>>(m, n, k, A, lda, idx);
#endif

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}


template <typename T>
void hadamard_product(const T *A, const T *B, T *C, size_t n, size_t m, size_t lda, cudaStream_t strm)
{
    auto [tg, bg] = partition_1d(n, m, lda);
#ifdef USE_CUDA
    printf("Kernel Execution Begin.");
    hadamard_product<<<tg, bg, 0, strm>>>(A, B, C, n, m, lda);
#endif
    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch hadamard_product. " << cudaGetErrorString(ierr))
        abort();
    }
}


} // end namespace cuda_kernels