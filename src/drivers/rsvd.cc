#include <cstdint>
#include <vector>

#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;
using namespace blas;
using namespace lapack;

namespace RandLAPACK::drivers::rsvd {

template <typename T>
int RSVD<T>::RSVD1(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& k,
    int64_t block_sz,
    T tol,
    std::vector<T>& U,
    std::vector<T>& S,
    std::vector<T>& VT
){
    // Q and B sizes will be adjusted automatically
    this->QB_Obj.call(m, n, A, k, block_sz, tol, this->Q, this->B);

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

template int RSVD<float>::RSVD1(int64_t m, int64_t n, std::vector<float>& A, int64_t& k, int64_t block_sz, float tol, std::vector<float>& U, std::vector<float>& S, std::vector<float>& VT);
template int RSVD<double>::RSVD1(int64_t m, int64_t n, std::vector<double>& A, int64_t& k, int64_t block_sz, double tol, std::vector<double>& U, std::vector<double>& S, std::vector<double>& VT);
}
