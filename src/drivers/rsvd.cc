#include <cstdint>
#include <vector>

#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;
using namespace blas;
using namespace lapack;

namespace RandLAPACK::drivers::rsvd {

// -----------------------------------------------------------------------------
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
///     Buffer for the \transpose{V}-factor.
///     Initially, may not have any space allocated for it.
///
/// @param[out] U
///     Stores m-by-k factor U.
///
/// @param[out] S
///     Stores k-by-k factor \Sigma.
///
/// @param[out] VT
///     Stores k-by-n factor \transpose{V}.
///
/// @return = 0: successful exit
///

template <typename T>
int RSVD<T>::RSVD1(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& k,
    T tol,
    std::vector<T>& U,
    std::vector<T>& S,
    std::vector<T>& VT
){
    // Q and B sizes will be adjusted automatically
    this->QB_Obj.call(m, n, A, k, this->block_sz, tol, this->Q, this->B);

    // Making sure all vectors are large enough
    upsize(m * k, U);
    upsize(k * k, this->U_buf);
    upsize(k * 1, S);
    upsize(k * n, VT);

    // SVD of B
    gesdd(Job::SomeVec, k, n, this->B.data(), k, S.data(), this->U_buf.data(), k, VT.data(), k);
    // Adjusting U
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, this->Q.data(), m, this->U_buf.data(), k, 0.0, U.data(), m);
    
    return 0;
}

template int RSVD<float>::RSVD1(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, float tol, std::vector<float>& U, std::vector<float>& S, std::vector<float>& VT);
template int RSVD<double>::RSVD1(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, double tol, std::vector<double>& U, std::vector<double>& S, std::vector<double>& VT);
}
