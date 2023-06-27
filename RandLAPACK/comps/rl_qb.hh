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

template <typename T>
class QBalg {
    public:

        virtual ~QBalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            int64_t block_sz,
            T tol,
            std::vector<T>& Q,
            std::vector<T>& B
        ) = 0;
};

template <typename T>
class QB : public QBalg<T> {
    public:

        // Constructor
        QB(
            // Requires a RangeFinder scheme object.
            RandLAPACK::RangeFinder<T>& rf_obj,
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T>& orth_obj,
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
        ///     Has the same number of columns of A.
        ///
        /// @return = 0: successful exit
        ///

        int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t& k,
            int64_t block_sz,
            T tol,
            std::vector<T>& Q,
            std::vector<T>& B
        ) override;

    public:
        RandLAPACK::RangeFinder<T>& RF_Obj;
        RandLAPACK::Stabilization<T>& Orth_Obj;
        bool verbosity;
        bool orth_check;

        std::vector<T> Q_gram;
        std::vector<T> Q_i_gram;

        std::vector<T> QtQi;
        std::vector<T> Q_i;
        std::vector<T> B_i;

        //This represents how much space is currently allocated for cols of Q and rows of B.
        //This is <= k. We are assuming that the user may not have given "enough"
        //space when allocating Q, B initially.
        int64_t curr_lim;

        // By how much are we increasing the dimension when we've reached curr_lim
        int dim_growth_factor;
};

// -----------------------------------------------------------------------------
template <typename T>
int QB<T>::call(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& k,
    int64_t block_sz,
    T tol,
    std::vector<T>& Q,
    std::vector<T>& B
){

    int64_t curr_sz = 0;
    int64_t next_sz = 0;

    T* A_dat = A.data();
    // pre-compute nrom
    T norm_A = lapack::lange(Norm::Fro, m, n, A_dat, m);
    // Immediate termination criteria
    if(norm_A == 0.0) {
        // Zero matrix termination
        k = curr_sz;
        return 1;
    }

    tol = std::max(tol, 100 * std::numeric_limits<T>::epsilon());
    // If the space allocated for col in Q and row in B is insufficient for any iterations ...
    if(std::max( Q.size() / m, B.size() / n) < (uint64_t)k) {
        // ... allocate more!
        this->curr_lim = std::min(this->dim_growth_factor * block_sz, k);
        // No need for data movement in this case
        util::upsize(m * this->curr_lim, Q);
        util::upsize(this->curr_lim * n, B);
    } else {
        this->curr_lim = k;
    }

    // Copy the initial data to avoid unwanted modification TODO #1
    std::vector<T> A_cpy (m * n, 0.0);
    T* A_cpy_dat = A_cpy.data();
    lapack::lacpy(MatrixType::General, m, n, A_dat, m, A_cpy_dat, m);

    T norm_B = 0.0;
    T prev_err = 0.0;
    T approx_err = 0.0;

    if(this->orth_check) {
        util::upsize(this->curr_lim * this->curr_lim, this->Q_gram);
        util::upsize(block_sz * block_sz, this->Q_i_gram);
    }

    T* QtQi_dat = util::upsize(this->curr_lim * block_sz, this->QtQi);
    T* Q_i_dat = util::upsize(m * block_sz, this->Q_i);
    T* B_i_dat = util::upsize(block_sz * n, this->B_i);

    T* Q_dat = Q.data();
    T* B_dat = B.data();

    while(k > curr_sz) {
        // Dynamically changing block size
        block_sz = std::min(block_sz, k - curr_sz);
        next_sz = curr_sz + block_sz;

        // Make sure we have enough space for everything
        if(next_sz > this->curr_lim) {
            this->curr_lim = std::min(2 * this->curr_lim, k);
            Q_dat = util::upsize(this->curr_lim * m, Q);
            B_dat = util::row_resize(curr_sz, n, B, this->curr_lim);
            QtQi_dat = util::upsize(this->curr_lim * block_sz, QtQi);
            if(this->orth_check)
                util::upsize(this->curr_lim * this->curr_lim, Q_gram);
        }

        // Calling RangeFinder
        if(this->RF_Obj.call(m, n, A_cpy, block_sz, this->Q_i))
            throw std::runtime_error("RangeFinder failed.");

        if(this->orth_check) {
            if (util::orthogonality_check(m, block_sz, block_sz, Q_i, Q_i_gram, this->verbosity)) {
                // Lost orthonormality of Q
                util::row_resize(this->curr_lim, n, B, curr_sz);
                k = curr_sz;
            }
        }

        // No need to reorthogonalize on the 1st pass
        if(curr_sz != 0) {
            // Q_i = orth(Q_i - Q(Q'Q_i))
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, curr_sz, block_sz, m, 1.0, Q_dat, m, Q_i_dat, m, 0.0, QtQi_dat, this->curr_lim);
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, block_sz, curr_sz, -1.0, Q_dat, m, QtQi_dat, this->curr_lim, 1.0, Q_i_dat, m);
            this->Orth_Obj.call(m, block_sz, this->Q_i);
        }

        //B_i = Q_i' * A
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, block_sz, n, m, 1.0, Q_i_dat, m, A_cpy_dat, m, 0.0, B_i_dat, block_sz);

        // Updating B norm estimation
        T norm_B_i = lapack::lange(Norm::Fro, block_sz, n, B_i_dat, block_sz);
        norm_B = std::hypot(norm_B, norm_B_i);
        // Updating approximation error
        prev_err = approx_err;
        approx_err = std::sqrt(std::abs(norm_A - norm_B)) * (std::sqrt(norm_A + norm_B) / norm_A);

        // Early termination - handling round-off error accumulation
        if ((curr_sz > 0) && (approx_err > prev_err)) {
            // Early termination - error growth
            // Only need to move B's data, no resizing
            util::row_resize(this->curr_lim, n, B, curr_sz);
            k = curr_sz;
        }

        // Update the matrices Q and B
        lapack::lacpy(MatrixType::General, m, block_sz, &Q_i_dat[0], m, &Q_dat[m * curr_sz], m);
        lapack::lacpy(MatrixType::General, block_sz, n, &B_i_dat[0], block_sz, &B_dat[curr_sz], this->curr_lim);

        if(this->orth_check) {
            if (util::orthogonality_check(m, this->curr_lim, next_sz, Q, Q_gram, this->verbosity)) {
                // Lost orthonormality of Q
                util::row_resize(this->curr_lim, n, B, curr_sz);
                k = curr_sz;
            }
        }

        curr_sz += block_sz;
        // Termination criteria
        if (approx_err < tol) {
            // Reached the required error tol
            util::row_resize(this->curr_lim, n, B, curr_sz);
            k = curr_sz;
            return 0;
        }

        // This step is only necessary for the next iteration
        // A = A - Q_i * B_i
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, block_sz, -1.0, Q_i_dat, m, B_i_dat, block_sz, 1.0, A_cpy_dat, m);
    }
    // Reached expected rank without achieving the tolerance
    return 3;
}

} // end namespace RandLAPACK
#endif
