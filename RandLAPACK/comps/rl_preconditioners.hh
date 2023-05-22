#ifndef randlapack_comps_preconditioners_h
#define randlapack_comps_preconditioners_h

#include "rl_orth.hh"
#include "rl_rf.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <cstdint>
#include <limits>
#include <vector>

#include <iostream>
#include <cstdint>

namespace RandLAPACK {

/**
 * Use sketching to produce data for a right-preconditioner of a
 * tall matrix A, or a regularized version thereof.
 * 
 * The data consists of column-orthonormal matrix matrix "V_sk" and
 * a vector "sigma_sk" of nonnegative nonincreasing values.
 * 
 * If no regularization is needed, then the preconditioner is the 
 * matrix "M" obtained by dividing column i of V_sk by sigma_sk[i];
 * this matrix M should make A_pre = A * M well-conditioned. This
 * makes it so that
 * 
 *      min{||A_pre * z - b ||} and min{ ||y|| s.t. A_pre' y = c }
 * 
 * can easily be solved by unpreconditioned iterative methods.
 * 
 * Warning: if A is in column-major format, then this function
 * will internally allocate (and subsequently deallocate) a 
 * workspace buffer of size n^2.
 * 
 * @param[in] layout_A
 *      Either blas::Layout::RowMajor or blas::Layout::ColMajor.
 * @param[in] m
 *      The number of rows in A; this must be at least as large
 *      as "n" and it should be much larger.
 * @param[in] n
 *      The number of columns in A (and rows in V_sk).
 * @param[in] d
 *      The number of rows in the sketched data matrix; must be >= n.
 * @param[in] k
 *      The number of nonzeros per column in the sketching operator
 *      A common value is k=8 when n is on the order of a couple thousand.
 * @param[in] A
 *      A buffer for an m \times n matrix A. Interpreted in 
 *      "layout_A" order.
 * @param[in] lda
 *      Leading dimension of A. This parameter, together with layout_A,
 *      m, n, and A itself define the m-by-n matrix mat(A).
 * @param[out] V_sk
 *      A buffer of size >= d*n.
 *      On exit, contains all n right singular vectors of a sketch of A,
 *      represented in column-major format with leading dimension n.
 * @param[out] sigma_sk
 *      A buffer of size >= n.
 *      On exit, contains all n singular values of a sketch of A.
 * @param[in] state
 *      The RNGState used to generate the sketching operator.
 * @returns
 *      An RNGState that the calling function should use the next
 *      time it needs an RNGState.
 */
template <typename T, typename RNG>
RandBLAS::base::RNGState<RNG> rpc_data_svd_saso(
    blas::Layout layout_A,
    int64_t m, // number of rows in A
    int64_t n, // number of columns in A
    int64_t d, // number of rows in sketch of A
    int64_t k, // number of nonzeros in each column of the sketching operator
    T *A, // buffer of size at least m*n.
    int64_t lda, // leading dimension for mat(A).
    T *V_sk, // buffer of size at least d*n.
    T *sigma_sk, //buffer of size at least n.
    RandBLAS::base::RNGState<RNG> state
) {
    T* buff_A = A;
    T* buff_A_sk = V_sk;
    if (m < n)
        throw std::invalid_argument("Input matrix A must have at least as many rows as columns.");
    
    // step 1.1: define the sketching operator
    RandBLAS::sparse::SparseDist D{
        .n_rows = d,
        .n_cols = m,
        .vec_nnz = k
    };
    RandBLAS::sparse::SparseSkOp<T> S(D, state);
    auto next_state = RandBLAS::sparse::fill_sparse(S);

    // step 1.2: sketch the data matrix
    int64_t lda_sk;
    if (layout_A == blas::Layout::RowMajor) {
        assert(lda >= n);
        lda_sk = n;
    } else {
        assert(lda >= m);
        lda_sk = d;
    }
    RandBLAS::util::safe_scal(d*n, 0.0, buff_A_sk, 1);
    RandBLAS::sparse::lskges<T>(
        layout_A,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        buff_A, lda,
        0.0, buff_A_sk, lda_sk
    );

    // step 2: compute SVD of sketch
    T* ignore = nullptr;
    if (layout_A == blas::Layout::RowMajor) {
        //
        //      buff_A_sk is stored in row-major format; we want its singular values
        //      and right singular vectors.
        //
        //      We get around LAPACK's column-major restriction by using the fact that
        //          memory(A_sk, RowMajor) == memory(transpose(A_sk), ColumnMajor).
        //      So we'll tell LAPACK that we want the singular values and left
        //      singular vectors of tranpose(A_sk).
        //
        lapack::Job jobu = lapack::Job::OverwriteVec;
        lapack::Job jobvt = lapack::Job::NoVec;
        lapack::gesvd(jobu, jobvt, n, d, buff_A_sk, n, sigma_sk, ignore, 1, ignore, 1);
        // interpret buff_A_sk in row-major
    } else {
        lapack::Job jobu = lapack::Job::NoVec;
        lapack::Job jobvt = lapack::Job::SomeVec;
        T *Vt = new T[n * n]{};
        lapack::gesvd(jobu, jobvt, d, n, buff_A_sk, d, sigma_sk, ignore, 1, Vt, n);
        for (int i = 0; i < n; ++i) {
            // buff_A_sk is now just scratch space that points to V_sk.
            // We just computed its SVD as a d-by-n matrix in column-major order.
            // Now we want to write to the first n^2 elements of buff_A_sk with
            // n "n-vectors" in column-major order.
            blas::copy(n, &Vt[i], n,  &buff_A_sk[i*n], 1);
        }
        delete[] Vt;
    }
    return next_state;
}


/**
 * Accepts the right singular vectors and singular values of some tall
 * matrix "T", along with a regularization parameter mu.
 * 
 * This function overwrites the the provided right singular vectors with
 * a matrix "M" so that if
 * 
 *          T_aug := [T; sqrt(mu)*I]
 * 
 * then T_aug * M is column-orthonormal. Such a matrix M is called an
 * _orthogonalizer_ of T_aug.
 * 
 * A thresholding scheme is applied to infer numerical rank of T_aug.
 * 
 * @param[in] n
 *      The number of rows and columns in V.
 * 
 * @param[in,out] V
 *      A buffer of size >= n*n, read in column-major order.
 * 
 *      On entry, the columns of V are the right singular vectors of some
 *      tall matrix T.
 * 
 *      On exit, the i-th column of V is scaled down by
 *              1/sqrt(sigma[i]^2 + mu)
 *      up until i = rank-1, where rank is the return value of this function.
 * 
 * @param[in] sigma
 *      A buffer of size >= n, containing the singular values of some tall
 *      matrix T.
 * 
 * @returns
 *      The number of columns in V that define the orthogonalizer.
 */
template <typename T>
int64_t make_right_orthogonalizer(
    int64_t n,
    T* V,
    T* sigma,
    T mu
) {
    double sqrtmu = std::sqrt((double) mu);
    auto regularized = [sqrtmu](T s) {
        return (sqrtmu == 0) ? s : (T) std::hypot((double) s, sqrtmu);
    };
    T curr_s = regularized(sigma[0]);
    T abstol = curr_s * n * std::numeric_limits<T>::epsilon();
    
    int64_t rank = 0;
    while (rank < n) {
        curr_s = regularized(sigma[rank]);
        if (curr_s < abstol)
            break;
        T scale = 1.0 / curr_s;
        blas::scal(n, scale, &V[rank*n], 1);
        rank = rank + 1;
    }
    return rank;
}

}  // end namespace RandLAPACK
#endif