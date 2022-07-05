/*
TODO #1: only store the upper triangle of the gram matrix in gram_vec,
so that it can be of size k*(k+1)/2 instead of k*k.
*/

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

namespace RandLAPACK::comps::orth {

// Perfoms a Cholesky QR factorization
template <typename T> 
void Orth<T>::CholQR(
        int64_t m,
        int64_t k,
        std::vector<T>& Q // pointer to the beginning
){
        using namespace blas;
        using namespace lapack;

        if (this->Q_gram.size() != k * k)
                this->Q_gram.resize(k * k); // TODO #1

        T* Q_dat = Q.data();
        T* Q_gram_dat = this->Q_gram.data();

        // Find normal equation Q'Q - Just the upper triangular portion
        syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, Q_gram_dat, k);

        // Positive definite cholesky factorization
        if (potrf(Uplo::Upper, k, Q_gram_dat, k) != 0)
                this->chol_fail = true; // scheme failure 
                
        // Q = Q * R^(-1)
        trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_gram_dat, k, Q_dat, m);	    
        this->chol_fail = false;
}

template <typename T> 
void Stab<T>::PLU(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<int64_t>& ipiv
){
        using namespace lapack;

        getrf(m, n, A.data(), m, ipiv.data());
        RandLAPACK::comps::util::swap_rows<T>(m, n, A, ipiv);
        RandLAPACK::comps::util::get_L<T>(m, n, A);
}

template <typename T> 
void Orth<T>::HQR(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<T>& tau
){
        // Done via regular LAPACK's QR
        // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
        // tau needs to be a vector of all 2's by default
        using namespace lapack;

        std::vector<T> Tvec (5, 0);

        T* A_dat = A.data();
	T* tau_dat = tau.data();
        /*
        // To perform TSQR-like orthogonalization. Currently, Q extraction step is not in lapackpp.
        geqr(m, n, A_dat, m, Tvec.data(), -1);
        int64_t tsize = (int64_t) Tvec[0]; 
        Tvec.resize(tsize);
        geqr(m, n, A_dat, m, Tvec.data(), tsize);
        */
        geqrf(m, n, A_dat, m, tau_dat);
        ungqr(m, n, n, A_dat, m, tau_dat);
}

template void Orth<float>::CholQR(int64_t m, int64_t k, std::vector<float>& Q);
template void Orth<double>::CholQR(int64_t m, int64_t k, std::vector<double>& Q);

template void Stab<float>::PLU(int64_t m, int64_t n, std::vector<float>& A, std::vector<int64_t>& ipiv);
template void Stab<double>::PLU(int64_t m, int64_t n, std::vector<double>& A, std::vector<int64_t>& ipiv);

template void Orth<float>::HQR(int64_t m, int64_t n, std::vector<float>& A, std::vector<float>& tau);
template void Orth<double>::HQR(int64_t m, int64_t n, std::vector<double>& A, std::vector<double>& tau);

} // end namespace orth