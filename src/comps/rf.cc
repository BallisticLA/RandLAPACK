/*
TODO #1: Merge test_mod and normal routine
*/
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
    if(this->Omega.size() != n * k)
        this->Omega.resize(n * k);

    T* Q_dat = Q.data();

    this->RS_Obj.call(m, n, A, k, this->Omega);

    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, this->Omega.data(), n, 0.0, Q_dat, m);

    if (this->cond_check)
    {
        // Copy to avoid any changes
        if(this->Q_cpy.size() != m * k)
            this->Q_cpy.resize(m * k);
        if(this->s.size() != k)
            this->s.resize(k);
        
        T* Q_cpy_dat = this->Q_cpy.data();
        T* s_dat = this->s.data();

        lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
        gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
        T cond_num = s_dat[0] / s_dat[k - 1];

        if (this->verbosity)
            printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);

        this->cond_nums.push_back(cond_num);
    }

    // Orthogonalization
    switch(this->Orth_Obj.decision_orth)
    {
        case 0:
            this->Orth_Obj.call(m, k, Q);
            // Check if CholQR failure flag is set to 1
            if (!this->Orth_Obj.chol_fail)
            {
                // Performing the alg twice for better orthogonality	
                this->Orth_Obj.call(m, k, Q);
            }
            else
            {
                // Switch to HQR
                this->Orth_Obj.decision_orth = 1;
                
                if (this->verbosity)
                    printf("CHOL QR FAILED\n");
                // Done via regular LAPACK's QR
                // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
                // tau needs to be a vector of all 2's by default
                this->Orth_Obj.tau.resize(k);
                this->Orth_Obj.call(m, k, Q);
            }
            break;
        case 1:
            this->Orth_Obj.tau.resize(k);
            this->Orth_Obj.call(m, k, Q);
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
    std::vector<T>& cond_nums
){
    using namespace blas;
    using namespace lapack;

    // Get the sketching operator Omega
    if(this->Omega.size() != n * k)
        this->Omega.resize(n * k);

    T* Q_dat = Q.data();

    this->RS_Obj.call(m, n, A, k, this->Omega);
    
    // Q = orth(A * Omega)
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, n, 1.0, A.data(), m, this->Omega.data(), n, 0.0, Q_dat, m);

    if (this->cond_check)
    {
        // Copy to avoid any changes
        if(this->Q_cpy.size() != m * k)
            this->Q_cpy.resize(m * k);
        if(this->s.size() != k)
            this->s.resize(k);

        T* Q_cpy_dat = this->Q_cpy.data();
        T* s_dat = this->s.data();

        lacpy(MatrixType::General, m, k, Q_dat, m, Q_cpy_dat, m);
        gesdd(Job::NoVec, m, k, Q_cpy_dat, m, s_dat, NULL, m, NULL, k);
        T cond_num = s_dat[0] / s_dat[k - 1];

        if (this->verbosity)
            printf("CONDITION NUMBER OF SKETCH Q_i: %f\n", cond_num);
        
        cond_nums.push_back(cond_num);
    }

    switch(this->Orth_Obj.decision_orth)
    {
        case 0:
            this->Orth_Obj.call(m, k, Q);
            // Check if CholQR failure flag is set to 1
            if (!this->Orth_Obj.chol_fail)
            {
                // Performing the alg twice for better orthogonality	
                this->Orth_Obj.call(m, k, Q);
            }
            else
            {
                // Switch to HQR
                this->Orth_Obj.decision_orth = 1;
                
                if (this->verbosity)
                    printf("CHOL QR FAILED\n");
                // Done via regular LAPACK's QR
                // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
                // tau needs to be a vector of all 2's by default
                this->Orth_Obj.tau.resize(k);
                this->Orth_Obj.call(m, k, Q);
            }
            break;
        case 1:
            this->Orth_Obj.tau.resize(k);
            this->Orth_Obj.call(m, k, Q);
            break;
    }
}

template void RF<float>::rf1(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q, bool use_qr);
template void RF<double>::rf1(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q, bool use_qr);

template void RF<float>::rf1_test_mode(int64_t m, int64_t n, const std::vector<float>& A, int64_t k, std::vector<float>& Q, std::vector<float>& cond_num);
template void RF<double>::rf1_test_mode(int64_t m, int64_t n, const std::vector<double>& A, int64_t k, std::vector<double>& Q, std::vector<double>& cond_num);
} // end namespace RandLAPACK::comps::rf
