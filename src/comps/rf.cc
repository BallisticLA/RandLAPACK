#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#define USE_QR
#define COND_CHECK
#define VERBOSE

namespace RandLAPACK::comps::rf {

template <typename T>
void RF<T>::rf1(
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

    RF::RS_Obj.call(m, n, A, k, Omega);

    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q_dat, m);

    if (RF::cond_check)
    {
        // Copy to avoid any changes
        std::vector<T> Q_cpy (m * k, 0.0);
        std::vector<T> s(k, 0.0);
        T* Q_cpy_dat = Q_cpy.data();
        T* s_dat = s.data();

        lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
        gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
        T cond_num = *s_dat / *(s_dat + k - 1);

        if (RF::verbosity)
            printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);
        
        RF::cond_num = cond_num;
    }

    // Orthogonalization
    switch(RF::Orth_Obj.decision_orth)
    {
        case 0:
            RF::Orth_Obj.call(m, k, Q);
            // Check if CholQR failure flag is set to 1
            if (!RF::Orth_Obj.chol_fail)
            {
                // Performing the alg twice for better orthogonality	
                RF::Orth_Obj.call(m, k, Q);
            }
            else
            {
                // Switch to HQR
                RF::Orth_Obj.decision_orth = 1;
                
                if (RF::verbosity)
                    printf("CHOL QR FAILED\n");
                // Done via regular LAPACK's QR
                // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
                // tau needs to be a vector of all 2's by default
                RF::Orth_Obj.tau.resize(k);
                RF::Orth_Obj.call(m, k, Q);
            }
            break;
        case 1:
            RF::Orth_Obj.tau.resize(k);
            RF::Orth_Obj.call(m, k, Q);
            break;
    }
}

template <typename T>
void RF<T>::rf1_test_mode(
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

    RF::RS_Obj.call(m, n, A, k, Omega);
    
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, Omega.data(), n, 0.0, Q_dat, m);

    if (RF::cond_check)
    {
        // Copy to avoid any changes
        std::vector<T> Q_cpy (m * k, 0.0);
        std::vector<T> s(k, 0.0);
        T* Q_cpy_dat = Q_cpy.data();
        T* s_dat = s.data();

        lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
        gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
        cond_num = *s_dat / *(s_dat + k - 1);

        if (RF::verbosity)
            printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);

        RF::cond_num = cond_num;
    }

    // Orthogonalization
    switch(RF::Orth_Obj.decision_orth)
    {
        case 0:
            RF::Orth_Obj.call(m, k, Q);
            // Check if CholQR failure flag is set to 1
            if (!RF::Orth_Obj.chol_fail)
            {
                // Performing the alg twice for better orthogonality	
                RF::Orth_Obj.call(m, k, Q);
            }
            else
            {
                // Switch to HQR
                RF::Orth_Obj.decision_orth = 1;
                
                if (RF::verbosity)
                    printf("CHOL QR FAILED\n");
                // Done via regular LAPACK's QR
                // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
                // tau needs to be a vector of all 2's by default
                RF::Orth_Obj.tau.resize(k);
                RF::Orth_Obj.call(m, k, Q);
            }
            break;
        case 1:
            RF::Orth_Obj.tau.resize(k);
            RF::Orth_Obj.call(m, k, Q);
            break;
    }
}

template void RF<float>::rf1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q, bool use_qr);
template void RF<double>::rf1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q, bool use_qr);

template void RF<float>::rf1_test_mode(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q, float& cond_num);
template void RF<double>::rf1_test_mode(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q, double& cond_num);
} // end namespace RandLAPACK::comps::rf
