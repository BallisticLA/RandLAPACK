#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::qb {

template <typename T>
void qb1(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // m by k
        std::vector<T>& B, // k by n
	uint32_t seed
);

template <typename T>
int qb2(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t& k,
        int64_t block_sz,
        T tol,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // m by k
        std::vector<T>& B, // k by n
	uint32_t seed
);

template <typename T>
int qb2_safe(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        int64_t& k,
        int64_t block_sz,
        T tol,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // m by k
        std::vector<T>& B, // k by n
	uint32_t seed
);

} // end namespace RandLAPACK::comps::rs