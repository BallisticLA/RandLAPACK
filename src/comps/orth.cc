#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::orth {

// Perfoms a Cholesky QR factorization
template <typename T> 
int Orth<T>::CholQRQ(
        int64_t m,
        int64_t k,
        std::vector<T>& Q
){
        printf("CHOL QR BEGIN\n");
        using namespace blas;
        using namespace lapack;
        
        T* Q_gram_dat = upsize<T>(k * k, this->Q_gram);
        T* Q_dat = Q.data();

        // Find normal equation Q'Q - Just the upper triangular portion        
        syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, Q_gram_dat, k);

        // Positive definite cholesky factorization
        if (potrf(Uplo::Upper, k, Q_gram_dat, k))
        {
                if(this->verbosity)
                {
                        printf("CHOLESKY QR FAILED\n");
                }
                this->chol_fail = true; // scheme failure 
                return 1;
        }

        // Scheme may succeed, but output garbage
        if(this->cond_check)
        {
                if(cond_num_check<T>(k, k, Q_gram, this->Q_gram_cpy, this->s, this->verbosity) > (1 / std::sqrt(std::numeric_limits<T>::epsilon()))){
                //        return 1;
                }
        }

        trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_gram_dat, k, Q_dat, m);
        printf("CHOL QR END\n");
        return 0;
}

template <typename T> 
int Stab<T>::PLUL(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<int64_t>& ipiv
){
        using namespace lapack;

        // Not using utility bc vector of int
        if(ipiv.size() < n) 
                ipiv.resize(n);

        T* A_dat = A.data();
        int64_t* ipiv_dat = ipiv.data(); 

        if(getrf(m, n, A_dat, m, ipiv_dat))
                return 1; // failure condition

        get_L<T>(m, n, A);
        laswp(n, A_dat, m, 1, n, ipiv_dat, 1);

        return 0;
}

template <typename T> 
int Orth<T>::HQRQ(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<T>& tau
){
        printf("HQR BEGIN\n");
        // Done via regular LAPACK's QR
        // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
        // tau needs to be a vector of all 2's by default
        using namespace lapack;

        upsize<T>(n, tau);

        T* A_dat = A.data();
	T* tau_dat = tau.data();
        if(geqrf(m, n, A_dat, m, tau_dat))
                return 1; // Failure condition
        ungqr(m, n, n, A_dat, m, tau_dat);
        printf("HQR END BEGIN\n");
        return 0;
}

// GEQR lacks "unpacking" of Q
template <typename T> 
int Orth<T>::GEQR(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<T>& tvec
){
        using namespace lapack;

        tvec.resize(5);

        T* A_dat = A.data();
        
        geqr(m, n, A_dat, m, tvec.data(), -1);
        int64_t tsize = (int64_t) tvec[0]; 
        tvec.resize(tsize);
        if(geqr(m, n, A_dat, m, tvec.data(), tsize))
                return 1;

        return 0;
}

template int Orth<float>::CholQRQ(int64_t m, int64_t k, std::vector<float>& Q);
template int Orth<double>::CholQRQ(int64_t m, int64_t k, std::vector<double>& Q);

template int Stab<float>::PLUL(int64_t m, int64_t n, std::vector<float>& A, std::vector<int64_t>& ipiv);
template int Stab<double>::PLUL(int64_t m, int64_t n, std::vector<double>& A, std::vector<int64_t>& ipiv);

template int Orth<float>::HQRQ(int64_t m, int64_t n, std::vector<float>& A, std::vector<float>& tau);
template int Orth<double>::HQRQ(int64_t m, int64_t n, std::vector<double>& A, std::vector<double>& tau);

template int Orth<float>::GEQR(int64_t m, int64_t n, std::vector<float>& A, std::vector<float>& tvec);
template int Orth<double>::GEQR(int64_t m, int64_t n, std::vector<double>& A, std::vector<double>& tvec); 
} // end namespace orth