#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_pdkernels.hh"

#include "rl_orth.hh"
#include "rl_syps.hh"
#include "rl_syrf.hh"
#include "rl_revd2.hh"
#include "rl_rpchol.hh"

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
    Layout layout,
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
    if (layout == Layout::RowMajor) {
        randblas_require(lda >= n);
        lda_sk = n;
    } else {
        randblas_require(lda >= m);
        lda_sk = d;
    }
    RandBLAS::util::safe_scal(d*n, 0.0, A_sk, 1);
    RandBLAS::sketch_general(
        layout,
        Op::NoTrans,
        Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        A, lda,
        0.0, A_sk, lda_sk
    );

    // step 2: apply an LAPACK SVD function to A_sk and process the output.
    if (layout == Layout::ColMajor) {
        auto jobu = Job::NoVec;
        auto jobvt = Job::OverwriteVec;
        lapack::gesvd(jobu, jobvt, d, n, A_sk, d, sigma_sk, nullptr, 1, nullptr, 1);
        RandLAPACK::util::eat_lda_slack(A_sk, n, n, d);
    } else {
        auto jobu = Job::OverwriteVec;
        auto jobvt = Job::NoVec;
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
 *      Either Layout::RowMajor or Layout::ColMajor.
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
    Layout layout,
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
 *      Layout::RowMajor or Layout::ColMajor.
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
    Layout layout,
    int64_t n,
    T* V,
    T* sigma,
    T mu,
    int64_t cols_V = -1
) {
    if (cols_V < 0) {
        cols_V = n;
    }
    double sqrtmu = std::sqrt((double) mu);
    auto regularized = [sqrtmu](T s) {
        return (sqrtmu == 0) ? s : (T) std::hypot((double) s, sqrtmu);
    };
    T curr_s = regularized(sigma[0]);
    T abstol = curr_s * cols_V * std::numeric_limits<T>::epsilon();
    
    int64_t rank = 0;
    int64_t inter_col_stride = (layout == Layout::ColMajor) ? n : 1;
    int64_t intra_col_stride = (layout == Layout::ColMajor) ? 1 : cols_V;
    while (rank < cols_V) {
        curr_s = regularized(sigma[rank]);
        if (curr_s < abstol)
            break;
        T scale = 1.0 / curr_s;
        blas::scal(n, scale, &V[rank * inter_col_stride], intra_col_stride);
        rank = rank + 1;
    }
    return rank;
}

/**
 * Construct data (V, eigvals) for a Nystrom preconditioner of regularized linear
 * systems involving a PSD matrix A.
 * 
 *      If our regularized linear system is 
 *          (A + mu*I)x= b,
 *      then the preconditioner would be
 *          P^{-1} = V diag(eigvals + mu)^{-1} V' + (I - VV') / (min(eigvals) + mu).
 * 
 *      Such a preconditioners will lead to (A + mu*I)*P^{-1} being well-conditioned
 *      if the spectral norm error || A - V diag(eigvals) V' || is no larger than mu.
 * 
 * This function computes (V, eigvals) in an iterative process. The matrix V has "k_in"
 * columns at the first iteration, where "k_in" the value of k on entry. Each iteration
 * ends by computing an estimate for || A - V diag(eigvals) V' ||. If this estimate
 * falls below tol = mu_min/5, then we return. Otherwise, we double k and try again.
 * 
 * Optional arguments to this function can trade-off greater computational cost
 * with (1) better approximations A_hat = V diag(eigvals) V' for fixed k and (2)
 * better estimates of the spectral norm error ||A - A_hat||.
 * 
 * @param[in] A
 *      An object conforming to the SymmetricLinearOperator interface.
 *      It represents a matrix of order A.m. It is callable, and responsible
 *      for allocating any memory it might need when called.
 * @param[out] V
 *      A std::vector that gives a column-major representation of an m-by-k_out
 *      column-orthgonal matrix, where k_out is the value of k on exit.
  * @param[out] eigvals
 *      A std::vector of length k_out, where k_out is the value of k on exit.
 *      The entries of this vector are positive and they define the eigenvalues
 *      of A_hat = V diag(eigvals) V'.
 * @param[in,out] k
 *      On entry: the number of columns to be used in V for the first iteration.
 *      On exit: the number of columns in V.
 * @param[in] mu_min
 *      The smallest value of mu for which we want (V, eigvals) to define a useful
 *      preconditioner for regularized linear systems (A + mu*I)x = b.
 * @param[in] state
 *      The RNGState used for random sketching inside the algorithm.
 * 
 * @param[in] num_syps_passes
 *      A very small nonnegative integer. Optional; defaults to 3. 
 *      This controls the number of power iterations used in computing a sketch of A.
 *      Increasing the value would reduce ||A - V diag(eigvals) V'|| when the number
 *      of columns in V is fixed.
 * @param[in] num_steps_power_iter_error_est
 *      A small positive integer. Optional; defaults to 10.
 *      This controls the number of power iterations used to estimate the spectral
 *      norm of A - V diag(eigvals) V'. If this value is too small then (V, eigvals)
 *      might not lead to good preconditioners for A+mu*I even when mu is >= mu_min.
 * 
 * @returns
 *      An RNGState that the calling function should use the next
 *      time it needs an RNGState.
 */
template <typename T, typename RNG>
RandBLAS::RNGState<RNG> nystrom_pc_data(
    linops::SymmetricLinearOperator<T> &A,
    std::vector<T> &V,
    std::vector<T> &eigvals,
    int64_t &k,
    T mu_min,
    RandBLAS::RNGState<RNG> state,
    int64_t num_syps_passes = 3,
    int64_t num_steps_power_iter_error_est = 10
) {
    RandLAPACK::SYPS<T, RNG> SYPS(num_syps_passes, 1, false, false);
    // ^ Define a symmetric power sketch algorithm.
    //      (*) Stabilize power iteration with pivoted-LU after every
    //          mulitplication with A.
    //      (*) Do not check condition numbers or log to std::out.
    RandLAPACK::HQRQ<T> Orth(false, false); 
    // ^ Define an orthogonalizer for a symmetric rangefinder.
    //      (*) Get a dense representation of Q from Householder QR.
    //      (*) Do not check condition numbers or log to std::out.
    RandLAPACK::SYRF<T, RNG> SYRF(SYPS, Orth, false, false);
    // ^ Define the symmetric rangefinder algorithm.
    //      (*) Use power sketching followed by Householder orthogonalization.
    //      (*) Do not check condition numbers or log to std::out.
    RandLAPACK::REVD2<T, RNG> NystromAlg(SYRF, num_steps_power_iter_error_est, false);
    // ^ Define the algorithm for low-rank approximation via Nystrom.
    //      (*) Handle accuracy requests by estimating ||A - V diag(eigvals) V'||
    //          with "num_steps_power_iter_error_est" steps of power iteration.
    //      (*) Do not log to std::out.
    T tol = mu_min / 5;
    // ^ Set tolerance to something materially smaller than the smallest
    //   regularization parameter the user claims to need.
    return NystromAlg.call(A, k, tol, V, eigvals, state);
}

/**
 * This wraps a function of the same name that accepts a SymmetricLinearOperator object.
 * The purpose of this wrapper is just to define such an object from data (uplo, A, m).
 */
template <typename T, typename STATE>
STATE nystrom_pc_data(
    Uplo uplo,
    const T* A,
    int64_t m,
    std::vector<T> &V,
    std::vector<T> &eigvals,
    int64_t &k,
    T mu_min,
    STATE state,
    int64_t num_syps_passes = 3,
    int64_t num_steps_power_iter_error_est = 10
) {
    linops::ExplicitSymLinOp<T> A_linop(m, uplo, A, m, Layout::ColMajor);
    return nystrom_pc_data(A_linop, V, eigvals, k, mu_min, state, num_syps_passes, num_steps_power_iter_error_est);
}

/**
 * TODO: make an overload of rpchol_pc_data that omits "n" and assumes A implements
 * some linear operator interface.
 */

template <typename T, typename STATE, typename FUNC>
STATE rpchol_pc_data(
    int64_t n, FUNC &A_stateless, int64_t &k, int64_t b, T* V, T* eigvals, STATE state
) {
    std::vector<int64_t> selection(k, -1);
    state = RandLAPACK::rp_cholesky(n, A_stateless, k, selection.data(), V, b, state);
    selection.resize(k);
    // ^ A_stateless \approx VV'; need to convert VV' into its eigendecomposition.
    std::vector<T> work(k*k, 0.0);
    lapack::gesdd(lapack::Job::OverwriteVec, n, k, V, n, eigvals, nullptr, 1, work.data(), k);
    // V has been overwritten with its (nontrivial) left singular vectors
    for (int64_t i = 0; i < k; ++i) 
        eigvals[i] = std::pow(eigvals[i], 2);
    return state;
}

}  // end namespace RandLAPACK
