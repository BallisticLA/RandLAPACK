#pragma once

#ifndef randlapack_comps_rf_h
#define randlapack_comps_rf_h

#include "rl_rs.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_orth.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <cstdio>

namespace RandLAPACK {

template <typename T, typename RNG>
class RangeFinder {
    public:
        virtual ~RangeFinder() {}

        virtual int call(
            int64_t m,
            int64_t n,
            const T* A,
            int64_t k,
            T* Q,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class RF : public RangeFinder<T, RNG> {
    public:

        RF(
            // Requires a RowSketcher scheme object.
            RandLAPACK::RowSketcher<T, RNG> &rs_obj,
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T> &orth_obj,
            bool verb,
            bool cond
        ) : RS_Obj(rs_obj), Orth_Obj(orth_obj) {
            verbose = verb;
            cond_check = cond;
        }

        /// RangeFinder - Return a matrix Q with k orthonormal columns, where range(Q) either subset of the range(A)
        /// if rank(A) >= k or
        /// range(A) is a subset of the range(Q) if rank(A) < k.
        /// Relies on a RowSketcher to do most of the work, then additionally reorthogonalizes RS's output.
        /// Optionally checks for whether the output of RS is ill-conditioned.
        /// This algorithm is shown in "the RandLAPACK book" book as Algorithm 9.
        ///
        ///    Conceptually, we compute Q by using [HMT:2011, Algorithm 4.3] and
        ///    [HMT:2011, Algorithm 4.4]. However, is a difference in how we perform
        ///    subspace iteration. Our subspace iteration still uses QR to stabilize
        ///    computations at each step, but computations are structured along the
        ///    lines of [ZM:2020, Algorithm 3.3] to allow for any number of passes over A.
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
        ///     Column size of the sketch.
        ///
        /// @param[in] Q
        ///     Buffer.
        ///
        /// @param[out] Q
        ///     Stores m-by-k matrix, range(Q) is
        ///     "reasonably" well aligned with A's leading left singular vectors.
        ///
        /// @return = 0: successful exit
        ///

        // Control of RF types calls.
        int call(
            int64_t m,
            int64_t n,
            const T* A,
            int64_t k,
            T* Q,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
       // Instantiated in the constructor
       RandLAPACK::RowSketcher<T, RNG> &RS_Obj;
       RandLAPACK::Stabilization<T> &Orth_Obj;
       bool verbose;
       bool cond_check;

       // Implementation-specific vars
       std::vector<T> cond_nums; // Condition nubers of sketches
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int RF<T, RNG>::call(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t k,
    T* Q,
    RandBLAS::RNGState<RNG> &state
){

    T* Omega  = ( T * ) calloc( n * k, sizeof( T ) );

    if(this->RS_Obj.call(m, n, A, k, Omega, state)) {
        free(Omega);
        return 1;
    }

    // Q = orth(A * Omega)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega, n, 0.0, Q, m);

    if(this->cond_check)
        // Writes into this->cond_nums
        this->cond_nums.push_back(util::cond_num_check(m, k, Q, this->verbose));

    if(this->Orth_Obj.call(m, k, Q))
        return 2; // Orthogonalization failed

    // Normal termination
    free(Omega);
    return 0;
}

} // end namespace RandLAPACK
#endif
