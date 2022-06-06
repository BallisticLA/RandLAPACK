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
        bool col_maj,
        int64_t m,
        int64_t n,
        T* L
);

template <typename T> 
void get_U(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* U // pointer to the beginning
);

template <typename T> 
void scale_diag(
        int64_t m,
        int64_t n,
        T* U, // pointer to the beginning
        T c //scaling factor 
);

template <typename T> 
void get_sym(
        bool col_maj,
        int64_t m,
        int64_t n,
        T* U // pointer to the beginning
);

template <typename T> 
void chol_QR(
        int64_t m,
        int64_t k,
        T* Q // pointer to the beginning
);

template <typename T>
void diag(
        int64_t m,
        int64_t n,
        T* s, // pointer to the beginning
        T* S
);

template <typename T> 
void pivot_swap(
        int64_t m,
        int64_t n,
        T* A, // pointer to the beginning
        int64_t* p // Pivot vector
);

} // end namespace RandLAPACK::comps::rs