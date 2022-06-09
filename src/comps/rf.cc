#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

//#define USE_QR

namespace RandLAPACK::comps::rf {

template <typename T>
void rf1(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // n by k
	    uint32_t seed
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);

    RandLAPACK::comps::rs::rs1<T>(m, n, A, k, p, passes_per_stab, Omega, seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q.data(), m);

#ifdef USE_QR
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default
    std::vector<T> tau(k, 2.0);
    geqrf(m, k, Q.data(), m, tau.data());
    ungqr(m, k, Q.data(), Q, m, tau.data());
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
        const std::vector<T>& A,
        int64_t k,
        int64_t p,
        int64_t passes_per_stab,
        std::vector<T>& Q, // n by k
        bool use_qr,
	    uint32_t seed
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);
    T* Q_dat = Q.data();

    RandLAPACK::comps::rs::rs1<T>(m, n, A, k, p, passes_per_stab, Omega, seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q_dat, m);


    if (!(use_qr || RandLAPACK::comps::orth::chol_QR<T>(m, k, Q)))
    {
        // Performing the alg twice for better orthogonality	
        RandLAPACK::comps::orth::chol_QR<T>(m, k, Q);
        return false;
    }
    else
    {
        // Done via regular LAPACK's QR
        // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
        // tau needs to be a vector of all 2's by default
        std::vector<T> tau(k, 2.0);
        T* tau_dat = tau.data();

        geqrf(m, k, Q_dat, m, tau_dat);
        ungqr(m, k, k, Q_dat, m, tau_dat);
        return true;
    }

    return false;
}

template void rf1<float>(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, uint32_t seed);
template void rf1<double>(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, uint32_t seed);

template bool rf1_safe<float>(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, bool use_qr, uint32_t seed);
template bool rf1_safe<double>(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, bool use_qr, uint32_t seed);
} // end namespace RandLAPACK::comps::rf
