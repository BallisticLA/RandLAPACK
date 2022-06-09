#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

//#define USE_QR

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
	    uint32_t seed
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);

    RandLAPACK::comps::rs::rs1<T>(m, n, A, k, p, passes_per_stab, Omega.data(), seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega.data(), n, 0.0, Q, m);

#ifdef USE_QR
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default
    std::vector<T> tau(k, 2.0);
    geqrf(m, k, Q, m, tau.data());
    ungqr(m, k, k, Q, m, tau.data());
#else
    RandLAPACK::comps::orth::chol_QR<T>(m, k, Q);
    // Performing the alg twice for better orthogonality	
    RandLAPACK::comps::orth::chol_QR<T>(m, k, Q);
#endif
}

template <typename T>
bool rf1_safe(
        int64_t m,
        int64_t n,
        T* const A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        T* Q, // n by k
        bool use_qr,
	    uint32_t seed
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);

    RandLAPACK::comps::rs::rs1<T>(m, n, A, k, p, passes_per_stab, Omega.data(), seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A, m, Omega.data(), n, 0.0, Q, m);


    if (!use_qr)
    {
        if(RandLAPACK::comps::orth::chol_QR<T>(m, k, Q))
        {
            // chol_QR failed
            std::vector<T> tau(k, 2.0);
            geqrf(m, k, Q, m, tau.data());
            ungqr(m, k, k, Q, m, tau.data());
            // Return "use Householder QR now"
            return true;
        }
        // Performing the alg twice for better orthogonality	
        RandLAPACK::comps::orth::chol_QR<T>(m, k, Q);
    }
    else
    {
        // Done via regular LAPACK's QR
        // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
        // tau needs to be a vector of all 2's by default
        std::vector<T> tau(k, 2.0);
        geqrf(m, k, Q, m, tau.data());
        ungqr(m, k, k, Q, m, tau.data());
    }

    return false;
}

template void rf1<float>(int64_t m, int64_t n, float* const A, int64_t k, int64_t p, int64_t passes_per_stab, float* Q, uint32_t seed);
template void rf1<double>(int64_t m, int64_t n, double* const A, int64_t k, int64_t p, int64_t passes_per_stab, double* Q, uint32_t seed);

template bool rf1_safe<float>(int64_t m, int64_t n, float* const A, int64_t k, int64_t p, int64_t passes_per_stab, float* Q, bool use_qr, uint32_t seed);
template bool rf1_safe<double>(int64_t m, int64_t n, double* const A, int64_t k, int64_t p, int64_t passes_per_stab, double* Q, bool use_qr, uint32_t seed);
} // end namespace RandLAPACK::comps::rf
