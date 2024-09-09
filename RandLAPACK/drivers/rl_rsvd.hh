#pragma once

#ifndef randlapack_drivers_rsvd_h
#define randlapack_drivers_rsvd_h

#include "rl_qb.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
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
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, k, k, 1.0, Q, m, UT_buf, k, 0.0, U, m);

    free(Q);
    free(BT);
    free(UT_buf);
    return 0;
}

} // end namespace RandLAPACK
#endif
