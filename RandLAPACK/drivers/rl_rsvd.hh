#pragma once

#include "rl_exceptions.hh"
#include "rl_qb.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <memory>
#include <vector>

namespace RandLAPACK {

template <typename T, typename RNG>
class RSVDalg {
    public:

        virtual ~RSVDalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t &k,
            T tol,
            T* &U,
            T* &S,
            T* &V,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class RSVD : public RSVDalg<T, RNG> {
    public:

        // Constructor
        RSVD(
            // Requires a QB algorithm object.
            RandLAPACK::QBalg<T, RNG> &qb_obj,
            int64_t b_sz
        ) : QB_Obj(qb_obj) {
            block_sz = b_sz;
        }

        /// Computes an economy Singular Value Decomposition:
        ///     A = U \Sigma \transpose{V},
        /// where U is m-by-k, V is n-by-k are orthogonal and \Sigma is k-by-k diagonal. 
        /// Relies on the randomized QB factorization algorithm
        ///
        /// Templated for `float` and `double` types.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-n matrix A, stored in a column-major format.
        ///
        /// @param[in] k
        ///     Expected rank of the matrix A. If unknown, set k=min(m,n).
        ///
        /// @param[in] block_sz
        ///     Block size parameter for randomized QB, block_sz <= k.
        ///
        /// @param[in] tol
        ///     Error tolerance parameter for ||A-QB||_Fro.
        ///
        /// @param[in] U
        ///     Buffer for the U-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[in] S
        ///     Buffer for the \Sigma-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[in] VT
        ///     Buffer for the V-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[out] U
        ///     Stores m-by-k factor U.
        ///
        /// @param[out] S
        ///     Stores k-by-k factor \Sigma.
        ///
        /// @param[out] V
        ///     Stores k-by-n factor V.
        ///
        /// @returns 0 if successful

        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t &k,
            T tol,
            T* &U,
            T* &S,
            T* &V,
            RandBLAS::RNGState<RNG> &state
        ) override;

        /// LinOp-based RSVD: accepts any LinearOperator.
        /// norm_A must be the positive Frobenius norm, supplied by the caller.
        /// Requires a QB object with RF/RS components and an operator supporting
        /// column-major applications. Other algorithm objects throw Error.
        /// The base operator is never modified (deflation is implicit).
        /// Returns 0 after factorization, a QB failure code (4, 5, or 6),
        /// or 7 if the SVD fails to converge. LAPACK exceptions propagate.
        /// Output pointers are left unchanged on failure or exception.
        template <linops::LinearOperator LinOp>
        int call(
            LinOp& A_op,
            T norm_A,
            int64_t &k,
            T tol,
            T* &U,
            T* &S,
            T* &V,
            RandBLAS::RNGState<RNG> &state
        );

    public:
        RandLAPACK::QBalg<T, RNG> &QB_Obj;
        int64_t block_sz;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int RSVD<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t &k,
    T tol,
    T* &U,
    T* &S,
    T* &V,
    RandBLAS::RNGState<RNG> &state
){
    // Input parameter validation. Bad inputs would otherwise propagate to a
    // downstream BLAS/LAPACK failure or a segfault, the latter fatal when
    // RSVD is called through a binding layer (e.g. MEX/MATLAB).
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
    randlapack_require(k > 0) << "target rank k=" << k << " must be > 0";
    randlapack_require(tol >= (T)0) << "tol=" << tol << " must be >= 0";
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";

    T* Q = nullptr;
    T* BT = nullptr; 
    // Q and B sizes will be adjusted automatically
    this->QB_Obj.call(m, n, A, k, this->block_sz, tol, Q, BT, state);

    T* UT_buf  = ( T * ) calloc(k * k, sizeof( T ) );
    // Making sure all vectors are large enough
    U  = ( T * ) calloc(m * k, sizeof( T ) );
    S  = ( T * ) calloc(k,     sizeof( T ) );
    V  = ( T * ) calloc(n * k, sizeof( T ) );

    // SVD of B
    lapack::gesdd(Job::SomeVec, n, k, BT, n, S, V, n, UT_buf, k);
    // Adjusting U
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, k, k, (T) 1.0, Q, m, UT_buf, k, (T) 0.0, U, m);

    free(Q);
    free(BT);
    free(UT_buf);
    return 0;
}

// -----------------------------------------------------------------------------
// LinOp-templated RSVD: accepts any LinearOperator.
// The base operator is never modified: deflation is handled implicitly
// by DowndatableLinOp inside QB.
template <typename T, typename RNG>
template <linops::LinearOperator LinOp>
int RSVD<T, RNG>::call(
    LinOp& A_op,
    T norm_A,
    int64_t &k,
    T tol,
    T* &U,
    T* &S,
    T* &V,
    RandBLAS::RNGState<RNG> &state
){
    T* Q = nullptr;
    T* BT = nullptr;

    auto* qb_concrete = dynamic_cast<QB<T, RNG>*>(&this->QB_Obj);
    randlapack_require(qb_concrete != nullptr) << "operator RSVD requires a QB factorization object";
    int status;
    try {
        status = qb_concrete->call(A_op, k, this->block_sz, tol, norm_A, Q, BT, state);
    } catch (...) {
        free(Q);
        free(BT);
        throw;
    }
    using Buffer = std::unique_ptr<T, decltype(&std::free)>;
    Buffer Q_owner(Q, &std::free);
    Buffer BT_owner(BT, &std::free);
    if (status >= 4 || k == 0) {
        return status;
    }

    std::vector<T> UT_buf(k * k);
    Buffer U_buffer(static_cast<T*>(calloc(A_op.n_rows * k, sizeof(T))), &std::free);
    Buffer S_buffer(static_cast<T*>(calloc(k, sizeof(T))), &std::free);
    Buffer V_buffer(static_cast<T*>(calloc(A_op.n_cols * k, sizeof(T))), &std::free);
    if (!U_buffer || !S_buffer || !V_buffer)
        throw std::bad_alloc();

    // SVD of B
    const int64_t info = lapack::gesdd(Job::SomeVec, A_op.n_cols, k, BT, A_op.n_cols,
                                      S_buffer.get(), V_buffer.get(), A_op.n_cols,
                                      UT_buf.data(), k);
    if (info != 0) return 7;
    // U = Q * UT_buf^T
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, A_op.n_rows, k, k, T(1),
               Q, A_op.n_rows, UT_buf.data(), k, T(0), U_buffer.get(), A_op.n_rows);

    U = U_buffer.release();
    S = S_buffer.release();
    V = V_buffer.release();
    return 0;
}

} // end namespace RandLAPACK
