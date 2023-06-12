#ifndef randlapack_comps_rf_h
#define randlapack_comps_rf_h

#include "rl_rf.hh"
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

template <typename T>
class RangeFinder {
    public:
        virtual ~RangeFinder() {}

        virtual int call(
            int64_t m,
            int64_t n,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Q
        ) = 0;
};

template <typename T>
class RF : public RangeFinder<T> {
    public:

        RF(
            // Requires a RowSketcher scheme object.
            RandLAPACK::RowSketcher<T>& rs_obj,
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T>& orth_obj,
            bool verb,
            bool cond
        ) : RS_Obj(rs_obj), Orth_Obj(orth_obj) {
            verbosity = verb;
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
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Q
        ) override;

    public:
       // Instantiated in the constructor
       RandLAPACK::RowSketcher<T>& RS_Obj;
       RandLAPACK::Stabilization<T>& Orth_Obj;
       bool verbosity;
       bool cond_check;
       std::vector<T> Omega;

       std::vector<T> Q_cpy;
       std::vector<T> s;

       // Implementation-specific vars
       std::vector<T> cond_nums; // Condition nubers of sketches
};

// -----------------------------------------------------------------------------
template <typename T>
int RF<T>::call(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Q
){

    T* Omega_dat = util::upsize(n * k, this->Omega);
    T* Q_dat = Q.data();

    if(this->RS_Obj.call(m, n, A, k, this->Omega))
        return 1;

    // Q = orth(A * Omega)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega_dat, n, 0.0, Q_dat, m);

    if(this->cond_check)
        // Writes into this->cond_nums
        this->cond_nums.push_back(util::cond_num_check(m, k, Q, this->Q_cpy, this->s, this->verbosity));

    if(this->Orth_Obj.call(m, k, Q))
        return 2; // Orthogonalization failed

    // Normal termination
    return 0;
}


template <typename T, typename RNG>
class BK : public RangeFinder<T>
{
    public:

        BK(
            // Requires a stabilization algorithm object.
            RandLAPACK::Stabilization<T>& stab_obj,
            RandLAPACK::Stabilization<T>& orth_obj,
            RandBLAS::RNGState<RNG> s,
            int64_t p,
            int64_t q,
            bool verb,
            bool cond
        ) : Stab_Obj(stab_obj), Orth_Obj(orth_obj), st(s) {
            verbosity = verb;
            cond_check = cond;
            //st = s;
            passes_over_data = p;
            passes_per_stab = q;
        }

        int call(
            int64_t m,
            int64_t n,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Q
        ) override;

        RandLAPACK::Stabilization<T>& Stab_Obj;
        RandLAPACK::Stabilization<T>& Orth_Obj;
        RandBLAS::RNGState<RNG>& st;
        int64_t passes_over_data;
        int64_t passes_per_stab;
        bool verbosity;
        bool cond_check;

        std::vector<T> Work;
        std::vector<T> Work_2;
        std::vector<T> s;

        // Implementation-specific vars
       std::vector<T> cond_nums; // Condition nubers of sketches
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int BK<T, RNG>::call(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Q
){

    int64_t p = this->passes_over_data;
    int64_t q = this->passes_per_stab;
    int64_t p_done = 0;

    const T* A_dat = A.data();

    Q.clear();
    T* Q_dat       = RandLAPACK::util::upsize(m * k, Q);
    T* Work_dat    = RandLAPACK::util::upsize(n * k, this->Work);
    auto state     = this->st;

    // Number of columns in a sketching operator
    int64_t numcols = 0;
    // Number of Krylov iterations done
    int64_t iters_done = 0;
    if (p % 2 == 0) {
        // Compute the sketch size from the number of passes & block size.
        // In this case, we have an expression x = randn(m, numcols), K = [x, AA'x, ...].
        // Even number of passes over data, so numcols = ceil(k / ((p / 2) + 1).
        numcols = (int64_t) std::ceil((float) k / ((p / 2) + 1));

        // Place an n by numcols Sketching operator buffer into the full K matrix, m by numcols
        RandBLAS::DenseDist  D{.n_rows = m, .n_cols = numcols};
        state = RandBLAS::fill_dense(D, Q_dat, state);
        this->st = state;

        if ((p_done % q == 0) && (this->Stab_Obj.call(m, numcols, Q)))
            throw std::runtime_error("Stabilization failed.");

        char name [] = "S";
        RandBLAS::util::print_colmaj(m, k, Q_dat, name);

    } else {
        // Compute the sketch size from the number of passes & block size.
        // In this case, we have an expression x = randn(n, numcols), x = Ax, K = [x AA'x, ...].
        // Odd number of passes over data, so numcols = ceil(k / ((p - 1) / 2)).
        numcols = (int64_t) std::ceil((float) 2 * k / (p - 1));

        // Fill m by numcols Work buffer - we already have more space than we can ever need
        RandBLAS::DenseDist D{.n_rows = n, .n_cols = numcols};
        state = RandBLAS::fill_dense(D, Work_dat, state);
        this->st = state;

        // Write an m by k product of A Work into the full K matrix
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, numcols, n, 1.0, A_dat, m, Work_dat, n, 0.0, Q_dat, m);
        ++ p_done;

        // Need to take in a pointer
        if ((p_done % q == 0) && (this->Stab_Obj.call(m, numcols, Q)))
            throw std::runtime_error("Stabilization failed.");
    }
    // We have placed something into full Omega previously.
    ++ iters_done;

    char name2 [] = "Work";
    int64_t offset = m * numcols;
    while (p - p_done > 0) {

        // A' * prev, write into workspace buffer
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, numcols, m, 1.0, A_dat, m, Q_dat + offset * (iters_done - 1), m, 0.0, Work_dat, n);
        ++p_done;

        // Optional condition number check
        if(this->cond_check) {
            RandLAPACK::util::upsize(n * k, this->Work_2);
            this->cond_nums.push_back(util::cond_num_check(n, numcols, this->Work, this->Work_2, this->s, this->verbosity));
        }

        // Stabilizaton
        if ((p_done % q == 0) && (this->Stab_Obj.call(n, numcols, Work)))
            throw std::runtime_error("Stabilzation failed.");

        RandBLAS::util::print_colmaj(n, numcols, Work_dat, name2);

        // At the last iteration, we may not be able to fit numcols columns into block Krylov matrix.
        // We then need to alter numcols.
        // Computation above is still done for a full numcols.
        if ((iters_done + 1) * numcols > k) {
            numcols = k - (iters_done * numcols);
        }
        // A * A' * prev, write into K
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, numcols, n, 1.0, A_dat, m, Work_dat, n, 0.0, Q_dat + offset * iters_done, m);
        ++ p_done;
        ++ iters_done;
    }

    char name3 [] = "K before qr";
    RandBLAS::util::print_colmaj(m, k, Q_dat, name3);

    // Orthogonalization
    if (this->Stab_Obj.call(m, k, Q))
        throw std::runtime_error("Orthogonalization failed.");


    char name1 [] = "K";
    RandBLAS::util::print_colmaj(m, k, Q_dat, name1);


    //successful termination
    return 0;
}

} // end namespace RandLAPACK
#endif
