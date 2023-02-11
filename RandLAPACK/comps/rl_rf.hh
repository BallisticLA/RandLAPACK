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

        /// RangeFinder - Return a matrix Q with k orthonormal columns, where Range(Q) is
        /// an approximation for the span of A's top k left singular vectors.
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
        int rf1(
            int64_t m,
            int64_t n,
            const std::vector<T>& A,
            int64_t k,
            std::vector<T>& Q
        );

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
) {
    int termination = rf1(m, n, A, k, Q);

    if(this->verbosity) {
        switch(termination)
        {
        case 0:
            printf("\nRF TERMINATED VIA: Normal termination.\n");
            break;
        case 1:
            printf("\nRF TERMINATED VIA: RowSketcher failed.\n");
            break;
        case 2:
            printf("\nRF TERMINATED VIA: Orthogonalization failed.\n");
            break;
        }
    }
    return termination;
}

// -----------------------------------------------------------------------------
template <typename T>
int RF<T>::rf1(
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

} // end namespace RandLAPACK
#endif
