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

// Note: This function is not intended for end-users at this time.
// We have it here to simplify unittests later on.
template <typename T, typename SKOP>
void rpc_data_svd(
    blas::Layout layout,
    int64_t m, // number of rows in A
    int64_t n, // number of columns in A
    T *A, // buffer of size at least m*n
    int64_t lda, // leading dimension for mat(A).
    SKOP &S, // d-by-m sketching operator.
    T *V_sk, // buffer of size at least d*n.
    T *sigma_sk //buffer of size at least n.
) {
    int64_t d = S.dist.n_rows;
    randblas_require(d >= n);
    T* A_sk = V_sk;
    if (m < n)
        throw std::invalid_argument("Input matrix A must have at least as many rows as columns.");

    // step 1
    int64_t lda_sk;
    if (layout == blas::Layout::RowMajor) {
        randblas_require(lda >= n);
        lda_sk = n;
    } else {
        randblas_require(lda >= m);
        lda_sk = d;
    }
    RandBLAS::util::safe_scal(d*n, 0.0, A_sk, 1);
    RandBLAS::sketch_general(
        layout,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        A, lda,
        0.0, A_sk, lda_sk
    );

    // step 2: apply an LAPACK SVD function to A_sk and process the output.
    if (layout == blas::Layout::ColMajor) {
        auto jobu = lapack::Job::NoVec;
        auto jobvt = lapack::Job::OverwriteVec;
        lapack::gesvd(jobu, jobvt, d, n, A_sk, d, sigma_sk, nullptr, 1, nullptr, 1);
        RandLAPACK::util::eat_lda_slack(A_sk, n, n, d);
    } else {
        auto jobu = lapack::Job::OverwriteVec;
        auto jobvt = lapack::Job::NoVec;
        lapack::gesvd(jobu, jobvt, n, d, A_sk, n, sigma_sk, nullptr, 1, nullptr, 1);
    }
    // memory(A_sk, 1,..., n^2) is the transposed right singular vectors, represented in
    // "layout" order. We need the _untransposed_ singular vectors in "layout" order.
    //
    // In-place transpose of a square matrix can (easily) be done in a way that's
    // layout-independent, as long as we don't care about cache efficiency. Since it's
    // an O(d n^2) operation to compute A_sk up to now and only O(n^2) to transpose,
    // the lack of cache efficiency shouldn't matter in this context.
    RandLAPACK::util::transpose_square(A_sk, n);
    return;
}

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
 * @param[in] layout
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
 *      "layout" order.
 * @param[in] lda
 *      Leading dimension of A. This parameter, together with layout,
 *      m, n, and A itself define the m-by-n matrix mat(A).
 * @param[out] V_sk
 *      A buffer of size >= d*n.
 *      On exit, contains all n right singular vectors of a sketch of A,
 *      represented in "layout" format with leading dimension n.
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
RandBLAS::RNGState<RNG> rpc_data_svd_saso(
    blas::Layout layout,
    int64_t m, // number of rows in A
    int64_t n, // number of columns in A
    int64_t d, // number of rows in sketch of A
    int64_t k, // number of nonzeros in each column of the sketching operator
    T *A, // buffer of size at least m*n.
    int64_t lda, // leading dimension for mat(A).
    T *V_sk, // buffer of size at least d*n.
    T *sigma_sk, //buffer of size at least n.
    RandBLAS::RNGState<RNG> state
) {
    RandBLAS::SparseDist D{
        .n_rows = d,
        .n_cols = m,
        .vec_nnz = k
    };
    RandBLAS::SparseSkOp<T> S(D, state);
    auto next_state = RandBLAS::fill_sparse(S);
    rpc_data_svd(layout, m, n, A, lda, S, V_sk, sigma_sk);
    return next_state;
}

/**
 * Accepts the right singular vectors and singular values of some tall
 * matrix "H", along with a regularization parameter mu.
 * 
 * This function overwrites the provided right singular vectors with
 * a matrix "M" so that if
 * 
 *          H_aug := [H; sqrt(mu)*I]
 * 
 * then H_aug * M is column-orthonormal. Such a matrix M is called an
 * _orthogonalizer_ of H_aug.
 * 
 * A thresholding scheme is applied to infer numerical rank of H_aug.
 * 
 * @param[in] layout
 *      blas::Layout::RowMajor or blas::Layout::ColMajor.
 *      The storage order for V.
 * 
 * @param[in] n
 *      The number of rows and columns in V.
 * 
 * @param[in,out] V
 *      A buffer of size >= n*n.
 * 
 *      On entry, the columns of V are the right singular vectors of some
 *      tall matrix H.
 * 
 *      On exit, the i-th column of V is scaled down by
 *              1/sqrt(sigma[i]^2 + mu)
 *      for i in {0, 1, ..., rank-1}, where rank is the return value of this function.
 * 
 * @param[in] sigma
 *      A buffer of size >= n, containing the singular values of some tall
 *      matrix H.
 * 
 * @returns
 *      The number of columns in V that define the orthogonalizer.
 */
template <typename T>
int64_t make_right_orthogonalizer(
    blas::Layout layout,
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
    int64_t inter_col_stride = (layout == blas::Layout::ColMajor) ? n : 1;
    int64_t intra_col_stride = (layout == blas::Layout::ColMajor) ? 1 : n;
    while (rank < n) {
        curr_s = regularized(sigma[rank]);
        if (curr_s < abstol)
            break;
        T scale = 1.0 / curr_s;
        blas::scal(n, scale, &V[rank * inter_col_stride], intra_col_stride);
        rank = rank + 1;
    }
    return rank;
}

}  // end namespace RandLAPACK
#endif
