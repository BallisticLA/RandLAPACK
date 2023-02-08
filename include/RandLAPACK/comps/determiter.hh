#ifndef randlapack_comps_determiter_h
#define randlapack_comps_determiter_h

#include <vector>
#include <cstdint>

namespace RandLAPACK::comps::determiter {


template <typename T>
void pcg(
    int64_t m,
    int64_t n,
    const T* A,
    int64_t lda, 
    const T* b, // length m
    const T* c, // length n
    T delta, // >= 0
    std::vector<T>& resid_vec, // re
    T tol, //  > 0
    int64_t k,
    const T* M, // n-by-k
    int64_t ldm,
    const T* x0, // length n
    T* x,  // length n
    T* y // length m
);


void run_pcgls_ex(int n, int m);

} // end namespace RandLAPACK::comps::determiter
#endif
