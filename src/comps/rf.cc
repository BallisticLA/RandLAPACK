//#include <RandLAPACK/comps/rf.hh>
//#include <RandLAPACK/comps/rs.hh>

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <iostream>

namespace RandLAPACK::comps::rf {

template <typename T>
void rf1(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        T* Q, // n by k
	    bool use_lu,
	    uint64_t seed
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);

    RandLAPACK::comps::rs::rs1<T>(m, n, A, k, p, passes_per_stab, Omega.data(), use_lu, seed);
    //RandBLAS::dense_op::gen_rmat_norm<T>(n, k, Omega.data(), seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega.data(), n, 0.0, Q, m);
    std::vector<T> tau(k, 2.0);

    //char name5[] = "Q";
    //RandBLAS::util::print_colmaj(m, k, Q, name5);

    // Done via regular LAPACK's QR
    // Leave here for future testing
    geqrf(m, k, Q, m, tau.data());
    ungqr(m, k, k, Q, m, tau.data());

    //RandLAPACK::comps::util::chol_QR<T>(m, k, Q);
    // Performing the alg twice for better orthogonality	
    //RandLAPACK::comps::util::chol_QR<T>(m, k, Q);
}

template void rf1<float>(int64_t m, int64_t n, float* const A, int64_t k, int64_t p, int64_t passes_per_stab, float* Q, bool use_lu, uint64_t seed);
template void rf1<double>(int64_t m, int64_t n, double* const A, int64_t k, int64_t p, int64_t passes_per_stab, double* Q, bool use_lu, uint64_t seed);
} // end namespace RandLAPACK::comps::rf
