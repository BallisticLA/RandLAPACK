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
void get_L(
        int64_t m,
        int64_t n,
        T* L
);

template <typename T>
void diag(
        int64_t m,
        int64_t n,
        T* s, // pointer to the beginning
        T* S
);

template <typename T> 
void row_swap(
        int64_t m,
        int64_t n,
        T* A, // pointer to the beginning
        int64_t* p // Pivot vector
);

template <typename T> 
void gen_exp_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        T t, // controls the decay. The higher the value, the slower the decay
        int32_t seed
);

template <typename T> 
void gen_s_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        int32_t seed
);

template <typename T> 
void gen_mat(
        int64_t m,
        int64_t n,
        T* A,
        int64_t k, // vector length
        T* S,
        int32_t seed
);

} // end namespace RandLAPACK::comps::rs