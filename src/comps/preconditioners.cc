#include <math.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>


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
 * we denote it by "A" to emphasize the row major restriction.
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
 * @param[in] A
 *      A buffer for an m \times n matrix A. Interpreted in 
 *      "layout_A" order.
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
 *      A 32-bit counter provided to the underlying counter-based random number
 *      generator. This value is incremented by m*k on-exit.
 * @param[in] layout_A
 *      Either blas::Layout::RowMajor or blas::Layout::ColMajor.
 * @returns
 *      The number of columns in M.
 */
template <typename T>
int64_t rpc_svd_sjlt(
    int64_t m,
    int64_t n,
    int64_t d,
    int64_t k,
    std::vector<T>& A,
    std::vector<T>& M_wk,
    T mu, 
    int64_t threads,
    uint64_t seed_key, 
    uint32_t seed_ctr,
    blas::Layout layout_A
){
    T* buff_A = A.data();
    if (M_wk.size() < d*n)
        throw std::invalid_argument("M_wk must be of size at least d*n.");
    T* buff_A_sk = M_wk.data();
    
    // step 1.1: define the sketching operator
    RandBLAS::sparse::SparseDist D{
        .n_rows = d,
        .n_cols = m,
        .vec_nnz=k
    };
    auto state = RandBLAS::base::RNGState(seed_key, seed_ctr);
    auto S = RandBLAS::sparse::SparseSkOp<T>(D, state);
    auto next_state = RandBLAS::sparse::fill_saso<T>(S);

    // step 1.2: sketch the data matrix
    int64_t lda, lda_sk;
    if (layout_A == blas::Layout::RowMajor) {
        lda = n;
        lda_sk = n;
    } else {
        lda = m;
        lda_sk = d;
    }
    blas::scal(d*n, 0.0, buff_A_sk, 1);
    RandBLAS::sparse::lskges<T>(
        layout_A,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        d, n, m,
        1.0, S, 0, 0,
        buff_A, lda,
        0.0, buff_A_sk, lda_sk, threads
    );

    // step 2: compute SVD of sketch
    T* ignore;
    std::vector<T> s(n, 0.0);
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
        lapack::gesvd(jobu, jobvt, n, d, buff_A_sk, n, s.data(), ignore, 1, ignore, 1);
        // interpret buff_A_sk in row-major
    } else {
        lapack::Job jobu = lapack::Job::NoVec;
        lapack::Job jobvt = lapack::Job::SomeVec;
        T *Vt = new T[n * n];
        lapack::gesvd(jobu, jobvt, d, n, buff_A_sk, d, s.data(), ignore, 1, Vt, n);
        for (int i = 0; i < n; ++i) {
            // buff_A_sk is now just scratch space owned by the std::vector M_wk.
            // We just computed its SVD as a d-by-n matrix in column-major order.
            // Now we want to write to the first n^2 elements of buff_A_sk with
            // n "n-vectors" in column-major order.
            blas::copy(n, &Vt[i], n,  &buff_A_sk[i*n], 1);
        }
        delete[] Vt;
    }

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
            s[i] = (T) std::hypot((double) s[i], sqrtmu); // sqrt(s[i]^2 + mu)
        }
    }
    // 3.2
    int64_t rank = 0; 
    while (rank < n) {
       ++rank;
       if (s[rank - 1] < s[0]*n*std::numeric_limits<T>::epsilon())
            break;
    }
    if (s[rank - 1] == 0.0)
        throw std::runtime_error("The rank of the regularized sketch must be at least one.");
    // 3.3
    // if (layout_A == blas::Layout::RowMajor) {
    T *buff_V = M_wk.data(); // interpret as column-major
    for (int64_t i = 0; i < rank; ++i) {
        T scale = 1.0 / s[i];
        blas::scal(n, scale, &buff_V[i*n], 1);
    }
    // } else {
    //     throw std::runtime_error(std::string("Not implemented."));
    //     // Question: which of the following holds:
    //     //  (1) VT is stored as the upper n-by-n submatrix of the d-by-n matrix A_sk,
    //     //      we have to take length-n segments of the d*n buffer A_sk with stride d.
    //     //  (2) VT is stored as the first n^2 elements of the d*n buffer A_sk.
    //     //
    //     // One way to test is to compute the SVD of the block matrix [0; I], and see
    //     // how the layout of the buffer changes.
    // }

    return rank;
}


template int64_t rpc_svd_sjlt(
    int64_t m,
    int64_t n,
    int64_t d,
    int64_t k,
    std::vector<double>& A,
    std::vector<double>& M_wk,
    double mu,
    int64_t threads,
    uint64_t seed_key,
    uint32_t seed_ctr,
    blas::Layout layout_A
);

template int64_t rpc_svd_sjlt(
    int64_t m,
    int64_t n,
    int64_t d,
    int64_t k,
    std::vector<float>& A,
    std::vector<float> &M_wk,
    float mu,
    int64_t threads,
    uint64_t seed_key,
    uint32_t seed_ctr,
    blas::Layout layout_A
);

}