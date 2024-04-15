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
    return ((q < m*lda) && ((q % lda) < n));
}

inline
__device__
bool valid_index(size_t q, size_t m)
{
    return (q < m);
}

 
template <typename T>
inline
__device__
void find(const T* array, int64_t size, T target, T* result) {
    
    size_t q = array_index();
    if (!valid_index(q, size))
        return;
    
    if (q < size && array[q] == target) {
        *result = q;
    }
}


/// Positions columns of A in accordance with idx vector of length k.
/// idx array modified ONLY within the scope of this function.
template <typename T>
 __global__
void col_swap_gpu(    
    int64_t m,
    int64_t n,
    int64_t k,
    T* A,
    int64_t lda,
    T* idx)
{
    size_t q = array_index();

    if (!valid_index(q, m))
        return;

    if(k > n) 
        throw std::runtime_error("Invalid rank parameter.");

    int64_t i, j;
    T it;
    T buf;
    for (i = 0, j = 0; i < k; ++i) {
        j = idx[i] - 1;
        //blas::swap(m, &A[i * lda], 1, &A[j * lda], 1);
        buf = A[i * lda + q];
        A[q + i * lda] = A[q + j * lda];
        A[q + j * lda] = buf;

        // swap idx array elements
        // Find idx element with value i and assign it to j
        find(idx, k, i + 1, it);
        idx[it - idx] = j + 1;
    }
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
    T* idx, cudaStream_t strm)
{
#ifdef USE_CUDA
    auto [tg, bg] = partition_1d(n, m, lda);

    col_swap_gpu<<<tg, bg, 0, strm>>>(m, n, k, A, lda, idx);
#endif

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch col_swap_gpu. " << cudaGetErrorString(ierr))
        abort();
    }
}


} // end namespace cuda_kernels