#ifndef BLAS_HH
#include <blas.hh>
#define BLAS_HH
#endif

namespace RandLAPACK::comps::preconditioners {

template <typename T>
int64_t rpc_svd_sjlt(
    int64_t m, // number of rows in A
    int64_t n, // number of columns in A
    int64_t d, // number of rows in sketch of A
    int64_t k, // number of nonzeros in each column of the sketching operator
    std::vector<T>& A, // buffer of size m*n.
    std::vector<T>& M_wk, // length at least d*n; 
    T mu,
    int64_t threads, // number of OpenMP threads to use in sketching.
    uint64_t seed_key,
    uint32_t seed_ctr,
    blas::Layout layout_A = blas::Layout::RowMajor
);

}