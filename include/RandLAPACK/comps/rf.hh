#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

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
);

} // end namespace RandLAPACK::comps::rs