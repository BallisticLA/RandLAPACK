#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::util {

template <typename T>
void eye(
        int64_t m,
        int64_t n,
        T* A
);

template <typename T>
void householder_ref_gen(
        int64_t m,
        int64_t n,
        T* const A,
        T* Q 
);


template <typename T>
void orth_dcgs2(
        int64_t m,
        int64_t n,
        T* const A,
        T* Q 
);

template <typename T> 
void get_L(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* L
);
} // end namespace RandLAPACK::comps::rs