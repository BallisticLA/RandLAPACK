#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::rf {

template <typename T>
void rf1(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // n by k
	uint32_t seed
);

template <typename T>
bool rf1_safe(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // n by k
        bool use_qr,
	uint32_t seed
);

} // end namespace RandLAPACK::comps::rs