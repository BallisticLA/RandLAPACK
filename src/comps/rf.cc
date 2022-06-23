#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#define USE_QR
#define COND_CHECK
#define VERBOSE

namespace RandLAPACK::comps::rf {

template <typename T>
void RF1<T>::call(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        std::vector<T>& Q,
        T& cond_num
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);
    T* Q_dat = Q.data();

    RF1::RS_Obj.call(m, n, A, k, Omega);
    
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q_dat, m);

    if (RF1::cond_check)
    {
        // Copy to avoid any changes
        std::vector<T> Q_cpy (m * k, 0.0);
        std::vector<T> s(k, 0.0);
        T* Q_cpy_dat = Q_cpy.data();
        T* s_dat = s.data();

        lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
        gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
        cond_num = *s_dat / *(s_dat + k - 1);

        if (RF1::verbosity)
            printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);
    }

    RF1::Orthogonalization(m, k, Q);
}

template <typename T>
bool RF1<T>::call(
        int64_t m,
        int64_t n,
        const std::vector<T>& A,
        int64_t k,
        std::vector<T>& Q, // n by k
        bool use_qr
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    std::vector<T> Omega(n * k, 0.0);
    T* Q_dat = Q.data();

    RF1::RS_Obj.call(m, n, A, k, Omega);

    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q_dat, m);

    if (RF1::cond_check)
    {
        // Copy to avoid any changes
        std::vector<T> Q_cpy (m * k, 0.0);
        std::vector<T> s(k, 0.0);
        T* Q_cpy_dat = Q_cpy.data();
        T* s_dat = s.data();

        lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
        gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
        T cond_num = *s_dat / *(s_dat + k - 1);

        if (RF1::verbosity)
            printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);
    }

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

template void RF1<float>::call(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q, float& cond_num);
template void RF1<double>::call(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q, double& cond_num);

template bool RF1<float>::call(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q, bool use_qr);
template bool RF1<double>::call(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q, bool use_qr);

} // end namespace RandLAPACK::comps::rf
