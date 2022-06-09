#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::orth {

template <typename T> 
int chol_QR(
        int64_t m,
        int64_t k,
        T* Q // pointer to the beginning
);

} // end namespace RandLAPACK::comps::rs