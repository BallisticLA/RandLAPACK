#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::rs {

template <typename T>
void rs1(
		int64_t m,
		int64_t n,
		T* const A,
		int64_t k,
		int64_t p,
		int64_t passes_per_stab,
		T* Omega, // n by k
		uint32_t seed
);

} // end namespace RandLAPACK::comps::rs
