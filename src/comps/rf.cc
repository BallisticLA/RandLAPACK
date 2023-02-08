#include "RandLAPACK.hh"
#include "RandBLAS.hh"
#include "blaspp.h"
#include "lapackpp.h"

#include <cstdint>
#include <vector>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::rf {

// -----------------------------------------------------------------------------
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

template <typename T>
int RF<T>::rf1(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Q
){

    T* Omega_dat = upsize(n * k, this->Omega);
    T* Q_dat = Q.data();

    if(this->RS_Obj.call(m, n, A, k, this->Omega))
        return 1;

    // Q = orth(A * Omega)
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega_dat, n, 0.0, Q_dat, m);

    if(this->cond_check)
        // Writes into this->cond_nums
        this->cond_nums.push_back(cond_num_check(m, k, Q, this->Q_cpy, this->s, this->verbosity));
    
    if(this->Orth_Obj.call(m, k, Q))
        return 2; // Orthogonalization failed

    // Normal termination
    return 0;
}

template int RF<float>::rf1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q);
template int RF<double>::rf1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q);
} // end namespace RandLAPACK::comps::rf
