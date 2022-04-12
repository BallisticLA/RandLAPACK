#include <RandLAPACK/comps/rf.hh>
#include <RandLAPACK/comps/rs.hh>
#include <RandLAPACK/comps/qb.hh>
#include <iostream>

namespace RandLAPACK::comps::qb {

template <typename T>
void qb1(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        T* Q, // m by k
        T* B, // k by n
	bool use_lu,
	uint64_t seed
){
    using namespace blas;
    //std::vector<T> Q(m * k, 0.0);
    rf1(m, n, A, k, p, passes_per_stab, Q, use_lu, seed);
    gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, m, n, 1.0, Q, m, A, m, 0.0, B, k);
}

}// end namespace RandLAPACK::comps::rf