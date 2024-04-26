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
            int64_t block_sz,
            T tol,
            T* Q,
            T* B,
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
            dim_growth_factor = 4;
        }

        /// Iteratively build an approximate QB factorization of A,
        /// which terminates once either of the following conditions
        /// is satisfied
        ///   (1)  || A - Q B ||_F <= tol * || A ||_F
        /// or
        ///   (2) Q has k columns.
        /// Each iteration involves sketching A from the right by a sketching
        /// matrix with "block_sz" columns.
        ///
        /// The number of columns in Q increase by "block_sz" at each iteration, unless
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
        /// @param[in] block_sz
        ///     The block size in this blocked QB algorithm. Add this many columns
        ///     to Q at each iteration (except possibly the final iteration).
        ///
        /// @param[in] tol
        ///     Terminate if ||A - Q B||_F <= tol * || A ||_F.
        ///
        /// @param[in] Q
        ///     Buffer for the Q-factor.
        ///     Initially, may not have any space allocated for it.
        ///
        /// @param[in] B
        ///     Buffer for the B-factor.
        ///     Initially, may not have any space allocated for it.
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
            int64_t block_sz,
            T tol,
            T* Q,
            T* B,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        RandLAPACK::RangeFinder<T, RNG> &RF_Obj;
        RandLAPACK::Stabilization<T> &Orth_Obj;
        bool verbosity;
        bool orth_check;

        //This represents how much space is currently allocated for cols of Q and rows of B.
        //This is <= k. We are assuming that the user may not have given "enough"
        //space when allocating Q, B initially.
        int64_t curr_lim;

        // By how much are we increasing the dimension when we've reached curr_lim
        int dim_growth_factor;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int QB<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t &k,
    int64_t block_sz,
    T tol,
    T* Q,
    T* B,
    RandBLAS::RNGState<RNG> &state
){

    int64_t curr_sz = 0;
    int64_t next_sz = 0;
    this->curr_lim = k;
    tol = std::max(tol, 100 * std::numeric_limits<T>::epsilon());
    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;

    T* A_cpy     = ( T * ) calloc( m * n,                     sizeof( T ) );
    T* QtQi      = ( T * ) calloc( this->curr_lim * block_sz, sizeof( T ) );
    T* Q_i       = ( T * ) calloc( m * block_sz,              sizeof( T ) );
    T* B_i_trans = ( T * ) calloc( block_sz * n,              sizeof( T ) );
    // Make sure Q, B have space for at least one iteration
    
    if(!Q) {
        Q = ( T * ) realloc(Q, m * n * sizeof( T ) );
    }
    if(!B) {
        B = ( T * ) realloc(B, n * n * sizeof( T ) );
    }
    

    // pre-compute nrom
    T norm_A = lapack::lange(Norm::Fro, m, n, A, m);
    // Immediate termination criteria
    if(norm_A == 0.0) {
        // Zero matrix termination
        k = curr_sz;
        free(A_cpy);
        free(QtQi);
        free(Q_i);
        free(B_i_trans);
        return 1;
    }

    // Copy the initial data to avoid unwanted modification TODO #1
    lapack::lacpy(MatrixType::General, m, n, A, m, A_cpy, m);

    while(k > curr_sz) {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        next_sz = curr_sz + block_sz;

        // Make sure we have enough space for everything
        if(next_sz > this->curr_lim) {
            this->curr_lim = std::min(2 * this->curr_lim, k);
            Q    = ( T * ) realloc(Q,    this->curr_lim * m * sizeof( T ));
            B    = ( T * ) realloc(B,    this->curr_lim * n * sizeof( T ));
            QtQi = ( T * ) realloc(QtQi, this->curr_lim * block_sz * sizeof( T ));
        }

        // Calling RangeFinder
        if(this->RF_Obj.call(m, n, A_cpy, block_sz, Q_i, state))
            return 6; // RF failed

        if(this->orth_check) {
            if (util::orthogonality_check(m, block_sz, block_sz, Q_i, this->verbosity)) {
                // Lost orthonormality of Q
                k = curr_sz;
                free(A_cpy);
                free(QtQi);
                free(Q_i);
                free(B_i_trans);
                return 4;
            }
        }

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0) {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, curr_sz, block_sz, m, 1.0, Q, m, Q_i, m, 0.0, QtQi, this->curr_lim);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, curr_sz, -1.0, Q, m, QtQi, this->curr_lim, 1.0, Q_i, m);
            this->Orth_Obj.call(m, block_sz, Q_i);
        }

        //B_i' = A' * Q_i'
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, block_sz, m, 1.0, A_cpy, m, Q_i, m, 0.0, B_i_trans, n);

        // Updating B norm estimation
        T norm_B_i = lapack::lange(Norm::Fro, n, block_sz, B_i_trans, n);
        norm_B = std::hypot(norm_B, norm_B_i);
        // Updating approximation error
        prev_err = approx_err;
        approx_err = std::sqrt(std::abs(norm_A - norm_B)) * (std::sqrt(norm_A + norm_B) / norm_A);

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err)) {
            // Early termination - error growth
            // Only need to move B's data, no resizing
            k = curr_sz;
            free(A_cpy);
            free(QtQi);
            free(Q_i);
            free(B_i_trans);
            return 2;
        }

        // Update the matrices Q and B
        lapack::lacpy(MatrixType::General, m, block_sz, Q_i, m, &Q[m * curr_sz], m);
        lapack::lacpy(MatrixType::General, n, block_sz, B_i_trans, n, &B[n * curr_sz], n);

        if(this->orth_check) {
            if (util::orthogonality_check(m, this->curr_lim, next_sz, Q, this->verbosity)) {
                // Lost orthonormality of Q
                k = curr_sz;
                free(A_cpy);
                free(QtQi);
                free(Q_i);
                free(B_i_trans);
                return 5;
            }
        }

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol) {
            // Reached the required error tol
            k = curr_sz;
            free(A_cpy);
            free(QtQi);
            free(Q_i);
            free(B_i_trans);
            return 0;
        }

        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, block_sz, -1.0, Q_i, m, B_i_trans, n, 1.0, A_cpy, m);
    }

    free(A_cpy);
    free(QtQi);
    free(Q_i);
    free(B_i_trans);

    // Reached expected rank without achieving the tolerance
    return 3;
}

} // end namespace RandLAPACK
#endif
