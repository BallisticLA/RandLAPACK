#ifndef randlapack_comps_qb_h
#define randlapack_comps_qb_h

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

namespace RandLAPACK {

template <typename T, typename RNG>
class QBalg {
    public:

        virtual ~QBalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t &k,
            int64_t b_sz,
            T tol,
            T* &Q,
            T* &B,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class QB : public QBalg<T, RNG> {
    public:

        // Constructor
        QB(
            // Requires a RangeFinder scheme object.
            RandLAPACK::RangeFinder<T, RNG> &rf_obj,
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T> &orth_obj,
            bool verb,
            bool orth
        ) : RF_Obj(rf_obj), Orth_Obj(orth_obj) {
            verbosity = verb;
            orth_check = orth;
        }

        /// Iteratively build an approximate QB factorization of A,
        /// which terminates once either of the following conditions
        /// is satisfied
        ///   (1)  || A - Q B ||_F <= tol * || A ||_F
        /// or
        ///   (2) Q has k columns.
        /// Each iteration involves sketching A from the right by a sketching
        /// matrix with "b_sz" columns.
        ///
        /// The number of columns in Q increase by "b_sz" at each iteration, unless
        /// that would bring #cols(Q) > k. In that case, the final iteration only
        /// adds enough columns to Q so that #cols(Q) == k.
        /// The implementation relies on RowSketcher and RangeFinder,
        ///
        /// This algorithm is shown in "the RandLAPACK book" book as Algorithm 11.
        ///
        /// This implements a variant of Algorithm 2 from YGL:2018. There are two
        /// main differences.
        ///     (1) We allow subspace iteration when building a new block
        ///         of the QB factorization.
        ///     (2) We have to explicitly update A.
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
        /// @param[in] b_sz
        ///     The block size in this blocked QB algorithm. Add this many columns
        ///     to Q at each iteration (except possibly the final iteration).
        ///
        /// @param[in] tol
        ///     Terminate if ||A - Q B||_F <= tol * || A ||_F.
        ///
        /// @param[in] Q
        ///     Buffer for the Q-factor.
        ///     Q is REQUIRED to be either nullptr or to point to >= m * b_sz * sizeof(T) bytes.
        ///
        /// @param[in] B
        ///     Buffer for the B-factor.
        ///     B is REQUIRED to be either nullptr or to point to >= n * b_sz * sizeof(T) bytes.
        ///
        /// @param[out] Q
        ///     Has the same number of rows of A, and orthonormal columns.
        ///
        /// @param[out] B
        ///     Number of rows in B is equal to number of columns in A (B is returned in a transposed format).
        ///
        /// @return = 0: successful exit
        ///

        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t &k,
            int64_t b_sz,
            T tol,
            T* &Q,
            T* &B,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        RandLAPACK::RangeFinder<T, RNG> &RF_Obj;
        RandLAPACK::Stabilization<T> &Orth_Obj;
        bool verbosity;
        bool orth_check;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int QB<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t &k,
    int64_t b_sz,
    T tol,
    T* &Q,
    T* &B,
    RandBLAS::RNGState<RNG> &state
){

    int ctr = 0;
    while(10 > ctr) {
        Q = ( double * ) realloc(Q, ctr * sizeof( double ));
        ++ctr;
    }

    return 1;
}

} // end namespace RandLAPACK
#endif
