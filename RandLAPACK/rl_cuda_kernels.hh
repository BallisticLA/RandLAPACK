#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#if USE_CUDA

#include "macros.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace RandLAPACK::cuda_kernels {

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



/** takes the product of two matrices element wise.
 * Currently hard wired for column major order.
 *
 * @param[in] A a matrix with n rows and m columns
 * @param[in] B a matrix with n rows and m columns
 * @param[out] C a matrix with n rows and m column
 * @param[in] m number of rows in A,B, and C
 * @param[in] n number of cols in A,B, and C
 * @param[in] lda the number of elements between each row
 */
template <typename T>
 __global__
void hadamard_product(const T *A, const T *B, T *C, size_t n, size_t m, size_t lda)
{
    size_t q = array_index();

    if (!valid_index(q, m, n, lda))
        return;

    C[q] = A[q] * B[q];
}

/** takes the product of a diagonal and a dense matrix.
 * Currently hard wired for column major order.
 *
 * @param[in] A the diagonal of a diagonal matrix with n rows and n columns
 * @param[in] B a matrix with n rows and m columns
 * @param[out] C a matrix with n rows and m column
 * @param[in] m number of rows in A,B, and C
 * @param[in] n number of cols in A,B, and C
 * @param[in] lda the number of elements between each row
 */
template <typename T>
 __global__
void diagonal_dense_matmul(const T *A, const T *B, T *C, size_t n, size_t m, size_t lda)
{
    size_t q = array_index(); // i + lda * j

    if (!valid_index(q, m, n, lda))
        return;

    C[q] = A[q % lda] * B[q];
}


/** 
 * @param[in] A: (n, m) matrix
 * @param[in] out
 */
template <typename T>
 __global__
void max_l2_column(const T *A, T* out, size_t n, size_t m, size_t lda)
{
    size_t q = array_index(); // i + lda * j

    if (!valid_index(q, m, n, lda))
        return;

    /*
     * Find y_i = \|a_i\|_2 for each column of A (use blaspp)
     * Find max y_i (use thrust)
     */
}

/** 
 * 
 *
 * @param[in] x  a vector of length n
 * @param[in] y  a vector of length n
 * @param[out] z  a vector of length n
 * @param[in] epsilon
 * @param[in] n number of cols in A,B, and C
 * @param[in] lda the number of elements between each row
 */
template <typename T>
 __global__
void elementwise_division(const T *A, const T *B, T *C, T epsilon, size_t n, size_t m, size_t lda)
{
    size_t q = array_index();

    if (!valid_index(q, m, n, lda))
        return;

    T denominator = B[q];

    if (denominator < epsilon)
        denominator = epsilon;
    C[q] = A[q] / denominator;
}

/** takes the product of two matrices element wise.
 * Currently hard wired for coluimn major order.
 *
 * @param[in] A a matrix with n rows and m columns
 * @param[in] B a matrix with n rows and m columns
 * @param[out] C a matrix with n rows and m column
 * @param[in] n number of cols in A,B, and C
 * @param[in] m number of rows in A,B, and C
 * @param[in] lda the number of elements between each row
 */
template <typename T>
void hadamard_product(const T *A, const T *B, T *C, size_t n, size_t m, size_t lda, cudaStream_t strm)
{
    auto [tg, bg] = partition_1d(n, m, lda);

    hadamard_product<<<tg, bg, 0, strm>>>(A, B, C, n, m, lda);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch hadamard_product. " << cudaGetErrorString(ierr))
        abort();
    }
}

/** Evaluates A / B elementwise; but if |B| < epsilon evaluates A / epsilon
 * Currently hard wired for coluimn major order.
 *
 * @param[in] A a matrix with n rows and m columns
 * @param[in] B a matrix with n rows and m columns
 * @param[out] C a matrix with n rows and m column
 * @param[in] n number of cols in A,B, and C
 * @param[in] m number of rows in A,B, and C
 * @param[in] lda the number of elements between each row
 */
template <typename T>
void elementwise_division(const T *A, const T *B, T *C, T epsilon, size_t n, size_t m, size_t lda, cudaStream_t strm)
{
    auto [tg, bg] = partition_1d(n, m, lda);

    elementwise_division<<<tg, bg, 0, strm>>>(A, B, C, epsilon, n, m, lda);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch elementwise_division. " << cudaGetErrorString(ierr))
        abort();
    }
}


/** Evaluates C = diag(A) @ B
 * Currently hard wired for coluimn major order.
 *
 * @param[in] A a vector with n entries
 * @param[in] B a matrix with n rows and m columns
 * @param[out] C a matrix with n rows and m column
 * @param[in] n number of cols in B, and C
 * @param[in] m number of rows in B, and C
 * @param[in] lda the number of elements between each row
 */
template <typename T>
void diagonal_dense_matmul(const T *A, const T *B, T *C, size_t n, size_t m, size_t lda, cudaStream_t strm)
{
    auto [tg, bg] = partition_1d(n, m, lda);

    diagonal_dense_matmul<<<tg, bg, 0, strm>>>(A, B, C, n, m, lda);

    cudaError_t ierr = cudaGetLastError();
    if (ierr != cudaSuccess)
    {
        BPCG_ERROR("Failed to launch elementwise_division. " << cudaGetErrorString(ierr))
        abort();
    }
}

void somefun(int i)
{
    i = i + 1;
}

} // end namespace cuda_kernels
#endif
#endif