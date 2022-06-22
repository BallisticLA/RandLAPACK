#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::orth {

template <typename T> 
int chol_QR(
        int64_t m,
        int64_t k,
        std::vector<T>& Q // pointer to the beginning
);

template <typename T> 
void stab_LU(
        int64_t m,
        int64_t n,
        std::vector<T>& A
);

template <typename T> 
void stab_QR(
        int64_t m,
        int64_t n,
        std::vector<T>& A
);

template <typename T> 
void orth_Chol_QR(
        int64_t m,
        int64_t n,
        std::vector<T>& A
);

} // end namespace RandLAPACK::comps::rs