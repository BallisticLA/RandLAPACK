#include <RandLAPACK/comps/rf.hh>
#include <RandLAPACK/comps/rs.hh>
#include <iostream>

namespace RandLAPACK::comps::rf {

template <typename T>
void rf1(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        T* Q, // n by k
	bool use_lu,
	uint64_t seed
){
    using namespace blas;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);
    rs1(m, n, A, k, p, passes_per_stab, Omega.data(), use_lu, seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A, m, Omega.data(), n, 0.0, Q, m);
    std::vector<T> tau(k, 2.0);
    geqrf(m, k, Q, m, tau);
    ungqr(m, k, k, Q, m, tau);
}
} // end namespace RandLAPACK::comps::rf
