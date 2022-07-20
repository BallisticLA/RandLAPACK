#include <cstdint>
#include <vector>

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::rf {

template <typename T>
int RF<T>::rf1(
    int64_t m,
    int64_t n,
    const std::vector<T>& A,
    int64_t k,
    std::vector<T>& Q
){
    using namespace blas;
    using namespace lapack;

    T* Omega_dat = upsize<T>(n * k, this->Omega);
    T* Q_dat = Q.data();

    if(this->RS_Obj.call(m, n, A, k, this->Omega))
        return 1;


    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega_dat, n, 0.0, Q_dat, m);

    if(this->cond_check)
        // Writes into this->cond_nums
        this->cond_nums.push_back(cond_num_check<T>(m, k, Q, this->Q_cpy, this->s, this->verbosity));
    
    if(this->Orth_Obj.call(m, k, Q))
        return 2; // Orthogonalization failed

    // Normal termination
    return 0;
}

template int RF<float>::rf1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q);
template int RF<double>::rf1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q);
} // end namespace RandLAPACK::comps::rf
