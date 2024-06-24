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
            T* &BT,
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
        /// @param[in] b_sz
        ///     The block size in this blocked QB algorithm. Add this many columns
        ///     to Q at each iteration (except possibly the final iteration).
        ///
        /// @param[in] tol
        ///     Terminate if ||A - Q B||_F <= tol * || A ||_F.
        ///
        /// @param[in] Q
        ///     Buffer for the Q-factor.
        ///     We expect Q to be nullptr.
        ///
        /// @param[in] BT
        ///     Buffer for the B-factor.
        ///     We expect BT to be nullptr.
        ///
        /// @param[out] Q
        ///     Has the same number of rows of A, and orthonormal columns.
        ///
        /// @param[out] BT
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
            T* &BT,
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
    T* &BT,
    RandBLAS::RNGState<RNG> &state
){
    // #cols(Q) & #cols(BT) that are filled at a given iteration.
    int64_t curr_sz = 0;
    // #cols(Q) & #cols(BT) that will be filled at the end of a given iteration.
    int64_t next_sz = 0;
    tol = std::max(tol, 100 * std::numeric_limits<T>::epsilon());
    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;

    // We require Q, B to be nullptr.
    if(Q) free(Q);
    if(BT) free(BT);
    // Make sure Q, B have space for one iteration
    Q  = ( T * ) calloc(m * b_sz, sizeof( T ) );
    BT = ( T * ) calloc(n * b_sz, sizeof( T ) );
    // Allocate buffers
    T* QtQi  = ( T * ) calloc( b_sz * b_sz, sizeof( T ) );
    T* A_cpy = ( T * ) calloc( m * n,       sizeof( T ) );
    // Declate pointers to the iteration buffers.
    T* Q_i;
    T* BT_i;

    // pre-compute nrom
    T norm_A = lapack::lange(Norm::Fro, m, n, A, m);

    // Copy the initial data to avoid unwanted modification
    lapack::lacpy(MatrixType::General, m, n, A, m, A_cpy, m);

    while(curr_sz < k) {
        // Dynamically changing block size.
        b_sz = std::min(b_sz, k - curr_sz);
        next_sz = curr_sz + b_sz;
        
        // Allocate more space in Q, B, QtQi buffer if needed.
        if (curr_sz != 0) {
            Q    = ( T * ) realloc(Q,    next_sz * m * sizeof( T ));
            BT   = ( T * ) realloc(BT,   next_sz * n * sizeof( T ));
            QtQi = ( T * ) realloc(QtQi, next_sz * b_sz * sizeof( T ));
        }

        // Avoid extra buffer allocation, but be careful about pointing to the
        // correct location.
        Q_i = &Q[m * curr_sz];
        BT_i = &BT[n * curr_sz];

        // Calling RangeFinder
        if(this->RF_Obj.call(m, n, A_cpy, b_sz, Q_i, state)) {
            // RF failed
            k = curr_sz;
            free(A_cpy);
            free(QtQi);
            return 6;
        }

        if(this->orth_check) {
            if (util::orthogonality_check(m, b_sz, b_sz, Q_i, this->verbosity)) {
                // Lost orthonormality of Q
                k = curr_sz;
                free(A_cpy);
                free(QtQi);
                return 4;
            }
        }

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0) {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, curr_sz, b_sz, m, 1.0, Q, m, Q_i, m, 0.0, QtQi, next_sz);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, b_sz, curr_sz, -1.0, Q, m, QtQi, next_sz, 1.0, Q_i, m);
            this->Orth_Obj.call(m, b_sz, Q_i);
        }

        //B_i' = A' * Q_i'
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_sz, m, 1.0, A_cpy, m, Q_i, m, 0.0, BT_i, n);

        // Updating B norm estimation
        T norm_B_i = lapack::lange(Norm::Fro, n, b_sz, BT_i, n);
        norm_B = std::hypot(norm_B, norm_B_i);
        // Updating approximation error
        prev_err = approx_err;
        approx_err = std::sqrt(std::abs(norm_A - norm_B)) * (std::sqrt(norm_A + norm_B) / norm_A);

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err)) {
            // Early termination - error has grown.
            k = curr_sz;
            free(A_cpy);
            free(QtQi);
            return 2;
        }

        if(this->orth_check) {
            if (util::orthogonality_check(m, next_sz, next_sz, Q, this->verbosity)) {
                // Lost orthonormality of Q
                k = curr_sz;
                free(A_cpy);
                free(QtQi);
                return 5;
            }
        }

        // Update #cols(Q) & #cols(B)
        curr_sz += b_sz;

        // Termination criteria
        if (approx_err < tol) {
            // Reached the required error tol
            k = curr_sz;
            free(A_cpy);
            free(QtQi);
            return 0;
        }

        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, n, b_sz, -1.0, Q_i, m, BT_i, n, 1.0, A_cpy, m);
    }

    free(A_cpy);
    free(QtQi);

    // Reached expected rank without achieving the tolerance
    return 3;
}

} // end namespace RandLAPACK
#endif
