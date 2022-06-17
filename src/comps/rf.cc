#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#define USE_QR
#define COND_CHECK
#define VERBOSE

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
	    uint32_t seed,
        T& cond_num
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);
    T* Q_dat = Q.data();

    RandLAPACK::comps::rs::rs1<T>(m, n, A, k, p, passes_per_stab, Omega, seed);
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q_dat, m);

#ifdef COND_CHECK

    // Copy to avoid any changes
    std::vector<T> Q_cpy (m * k, 0.0);
    T* Q_cpy_dat = Q_cpy.data();
    lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
    /*
    T* rcond = 0;
    T one_norm = lange(Norm::Inf, m, k, Q_dat, m);
    //I don't understand why thid does not work
    gecon(Norm::Inf, k, Q_cpy_dat, k, one_norm, rcond);		
    */
    std::vector<T> s(k, 0.0);
    T* s_dat = s.data();
    gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
    cond_num = *s_dat / *(s_dat + k - 1);
#ifdef VERBOSE
    printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);
#endif
#endif

#ifdef USE_QR
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default
    std::vector<T> tau(k, 2.0);
    T* tau_dat = tau.data();

    geqrf(m, k, Q_dat, m, tau_dat);
    ungqr(m, k, k, Q_dat, m, tau_dat);
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

#ifdef COND_CHECK

    // Copy to avoid any changes
    std::vector<T> Q_cpy (m * k, 0.0);
    T* Q_cpy_dat = Q_cpy.data();
    lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
    /*
    T* rcond = 0;
    T one_norm = lange(Norm::Inf, m, k, Q_dat, m);
    //I don't understand why thid does not work
    gecon(Norm::Inf, k, Q_cpy_dat, k, one_norm, rcond);		
    */
    std::vector<T> s(k, 0.0);
    T* s_dat = s.data();
    gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
    T cond_num = *s_dat / *(s_dat + k - 1);
#ifdef VERBOSE
    printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);
#endif
#endif

    if (!(use_qr || RandLAPACK::comps::orth::chol_QR<T>(m, k, Q)))
    {
        // Performing the alg twice for better orthogonality	
        RandLAPACK::comps::orth::chol_QR<T>(m, k, Q);
        return false;
    }
    else
    {
        printf("CHOL QR FAILED\n");
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

template void rf1<float>(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, uint32_t seed, float& cond_num);
template void rf1<double>(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, uint32_t seed, double& cond_num);

template bool rf1_safe<float>(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<float>& Q, bool use_qr, uint32_t seed);
template bool rf1_safe<double>(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, int64_t p, int64_t passes_per_stab, std::vector<double>& Q, bool use_qr, uint32_t seed);
} // end namespace RandLAPACK::comps::rf
