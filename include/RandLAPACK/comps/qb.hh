#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

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
);

} // end namespace RandLAPACK::comps::rs