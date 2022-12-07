#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

#include <math.h>

#include <blas.hh>
#include <lapack.hh>

namespace RandLAPACK::comps::preconditioners {

/**
 * Use sketching to produce a right-preconditioner "M" for a tall
 * data matrix A_aug = [A; sqrt(mu) * I]. This is useful for ...
 *
 *      Overdetermined and underdetermined least squares
 *      ------------------------------------------------
 *      It makes A_pre = A_aug*M well-conditioned, so that
 *      min{||A_pre * z - b ||} and min{ ||y|| s.t. A_pre' y = c }
 *      can easily be solved by unpreconditioned iterative methods.
 *
 *      Solving linear systems with G = A'A + mu*I
 *      ------------------------------------------
 *      It makes M'GM well-conditioned, so that M'GM z = M'h
 *      can easily be solved by CG without preconditioning.
 *
 * The input matrix A must be contiguous and in ROW MAJOR order;
 * we denote it by "A_rm" to emphasize the row major restriction.
 *
 * On exit, the preconditioner is the dense matrix given reading the
 * entries of M_wk.data() in column-major order.
 *
 * @param[in] m
 *      The number of rows in A
 * @param[in] n
 *      The number of columns in A (and rows in M)
 * @param[in] d
 *      The number of rows in the sketched data matrix; must be >= n.
 * @param[in] k
 *      The number of nonzeros per column in the sketching operator
 *      A common value is k=8 when n is on the order of a couple thousand.
 * @param[in] A_rm
 *      A row-major representation of an m \times n matrix A
 * @param[out] M_wk
 *      We require that M_wk.data().size() >= d*n. Letting "r" denote the return
 *      value of this function, on exit the preconditioner M is identified
 *      with the first n*r entries in M_wk.data(), read in column-major format.
 * @param[in] mu
 *      A regularization parameter that affects the goal of preconditioning.
 * @param[in] threads
 *      The number of OpenMP threads used during sketching.
 * @param[in] seed_key
 *      The key provided to an underlying counter-based random number generator.
 * @param[inout] seed_ctr
 *      A 64-bit counter provided to the underlying counter-based random number
 *      generator. This value is incremented by m*k on-exit.
 * @returns
 *      The number of columns in M.
 */
template <typename T>
int64_t rpc_svd_sjlt(
    uint64_t m,
    uint64_t n,
    uint64_t d,
    uint64_t k,
    const T *A_rm,
    T *M_wk,
    T mu,
    int threads,
    uint64_t seed_key,
    uint64_t seed_ctr
){
    const T* buff_A = A_rm;
    T* buff_A_sk = M_wk; // interpret as row-major
    T *buff_V = M_wk; // interpret as column-major

    // step 1: sketch the data matrix
    RandBLAS::sjlts::SJLT<T> sjl(RandBLAS::sjlts::ColumnWise, d, m, k);
    RandBLAS::sjlts::fill_colwise(sjl, seed_key, seed_ctr);
    seed_ctr = sjl.vec_nnz * sjl.n_cols;
    blas::scal(d*n, 0.0, buff_A_sk, 1);
    RandBLAS::sjlts::sketch_cscrow(sjl, n, buff_A, buff_A_sk, threads);

    // step 2: compute SVD of sketch
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
    T* ignore = nullptr;
    std::vector<T> s(n);
    ::lapack::Job jobu = ::lapack::Job::OverwriteVec;
    ::lapack::Job jobvt = ::lapack::Job::NoVec;
    ::lapack::gesvd(jobu, jobvt, n, d, buff_A_sk, n, s.data(), ignore, 1, ignore, 1);

    // step 3: define preconditioner
    //
    //      3.1: Modify the singular values, in case mu > 0.
    //      3.2: Decide numerical rank.
    //      3.3: Scale columns of the matrix of right singular vectors.
    //
    if (mu > 0) {
        double sqrtmu = std::sqrt((double) mu);
        for (uint64_t i = 0; i < n; ++i) {
            s[i] = (T) std::hypot((double) s[i], sqrtmu); // sqrt(s[i]^2 + mu)
        }
    }
    uint64_t rank = 0;
    while (rank < n) {
       ++rank;
       if (s[rank - 1] < s[0]*n*std::numeric_limits<T>::epsilon())
            break;
    }
    if (s[rank - 1] == 0.0)
        throw std::runtime_error("The rank of the regularized sketch must be at least one.");

    for (uint64_t i = 0; i < rank; ++i) {
        T scale = 1.0 / s[i];
        blas::scal(n, scale, &buff_V[i*n], 1);
    }
    return rank;
}

template <typename T>
int64_t rpc_svd_sjlt(
    uint64_t m,
    uint64_t n,
    uint64_t d,
    uint64_t k,
    const std::vector<T>& A_rm,
    std::vector<T>& M_wk,
    T mu,
    int threads,
    uint64_t seed_key,
    uint64_t seed_ctr
){
    if (M_wk.size() < d*n)
        throw std::invalid_argument("M_wk must be of size at least d*n.");

    return rpc_svd_sjlt(m, n, d, k, A_rm.data(), M_wk.data(), mu, threads, seed_key, seed_ctr);
}
}
