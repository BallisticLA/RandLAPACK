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
 * @param[in] layout_A
 *      Either blas::Layout::RowMajor or blas::Layout::ColMajor.
 * @param[in] m
 *      The number of rows in A.
 * @param[in] n
 *      The number of columns in A (and rows in M).
 * @param[in] d
 *      The number of rows in the sketched data matrix; must be >= n.
 * @param[in] k
 *      The number of nonzeros per column in the sketching operator
 *      A common value is k=8 when n is on the order of a couple thousand.
 * @param[in] A
 *      A buffer for an m \times n matrix A. Interpreted in 
 *      "layout_A" order. This function does not allow a separate "lda"
 *      parameter. If layout_A=RowMajor, then we would say that lda=n.
 *      If layout_A=ColMajor, then we would say that lda=m.
 * @param[out] V_sk
 *      We require that V_sk.size() >= d*n. On exit, stores the right
 *      singular vectors of a sketch of A, represented in column-major
 *      format with leading dimension d.
 * @param[in] sigma_sk
 *      We require sigma_sk.size() >= n.
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
    std::vector<T>& A, // buffer of size m*n.
    std::vector<T>& V_sk, // length at least d*n 
    std::vector<T>& sigma_sk, // length at least n
    RandBLAS::base::RNGState<RNG> state
) {
    T* buff_A = A.data();
    if (V_sk.size() < d*n)
        throw std::invalid_argument("V_sk must be of size at least d*n.");
    T* buff_A_sk = V_sk.data();
    
    // step 1.1: define the sketching operator
    RandBLAS::sparse::SparseDist D{
        .n_rows = d,
        .n_cols = m,
        .vec_nnz = k
    };
    RandBLAS::sparse::SparseSkOp<T> S(D, state);
    auto next_state = RandBLAS::sparse::fill_sparse(S);

    // step 1.2: sketch the data matrix
    int64_t lda, lda_sk;
    if (layout_A == blas::Layout::RowMajor) {
        lda = n;
        lda_sk = n;
    } else {
        lda = m;
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
        //      TODO: check that rank(A_sk) == rank(A). This can be done by 
        //      taking a basis "B" for the kernel of A_sk and checking if
        //      if A*B == 0 (up to some tolerance).   
        //
        lapack::Job jobu = lapack::Job::OverwriteVec;
        lapack::Job jobvt = lapack::Job::NoVec;
        lapack::gesvd(jobu, jobvt, n, d, buff_A_sk, n, sigma_sk.data(), ignore, 1, ignore, 1);
        // interpret buff_A_sk in row-major
    } else {
        lapack::Job jobu = lapack::Job::NoVec;
        lapack::Job jobvt = lapack::Job::SomeVec;
        T *Vt = new T[n * n]{};
        lapack::gesvd(jobu, jobvt, d, n, buff_A_sk, d, sigma_sk.data(), ignore, 1, Vt, n);
        for (int i = 0; i < n; ++i) {
            // buff_A_sk is now just scratch space owned by the std::vector V_sk.
            // We just computed its SVD as a d-by-n matrix in column-major order.
            // Now we want to write to the first n^2 elements of buff_A_sk with
            // n "n-vectors" in column-major order.
            blas::copy(n, &Vt[i], n,  &buff_A_sk[i*n], 1);
        }
        delete[] Vt;
    }
    return next_state;
}

template <typename T, typename RNG>
int64_t rpc_svd_saso(
    blas::Layout layout_A,
    int64_t m,
    int64_t n,
    int64_t d,
    int64_t k,
    std::vector<T>& A,
    std::vector<T>& M,
    T mu,
    RandBLAS::base::RNGState<RNG> state
) {
    std::vector<T> sigma_sk(n, 0.0);
    rpc_data_svd_saso(layout_A, m, n, d, k, A, M, sigma_sk, state);
    // step 3: define preconditioner
    //
    //      3.1: Modify the singular values, in case mu > 0.
    //      3.2: Decide numerical rank.
    //      3.3: Scale columns of the matrix of right singular vectors.
    //
    // 3.1
    if (mu > 0) {
        double sqrtmu = std::sqrt((double) mu);
        for (int i = 0; i < n; ++i) {
            sigma_sk[i] = (T) std::hypot((double) sigma_sk[i], sqrtmu); // sqrt(s[i]^2 + mu)
        }
    }
    // 3.2 
    if (sigma_sk[0] == 0.0)
        throw std::runtime_error("The rank of the regularized sketch must be at least one.");
    int64_t rank = 0;
    while (rank < n) {
       ++rank;
       if (sigma_sk[rank - 1] < sigma_sk[0]*n*std::numeric_limits<T>::epsilon())
            break;
    }
    // 3.3
    T *buff_V = M.data(); // interpret as column-major
    for (int64_t i = 0; i < rank; ++i) {
        T scale = 1.0 / sigma_sk[i];
        blas::scal(n, scale, &buff_V[i*n], 1);
    }
    return rank;
}

}  // end namespace RandLAPACK
#endif