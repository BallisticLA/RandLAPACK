/*
TODO #1: only store the upper triangle of the gram matrix in gram_vec,
so that it can be of size k*(k+1)/2 instead of k*k.
*/

#include <cstdint>
#include <vector>

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::orth {

// Perfoms a Cholesky QR factorization
template <typename T> 
int Orth<T>::CholQR(
        int64_t m,
        int64_t k,
        std::vector<T>& Q
){
        using namespace blas;
        using namespace lapack;
        
        T* Q_gram_dat = upsize<T>(k * (k + 1) / 2, this->Q_gram);
        T* Q_dat = Q.data();

        // Find normal equation Q'Q - Just the upper triangular portion        
        sfrk(Op::NoTrans, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, Q_gram_dat);

        // Positive definite cholesky factorization
        if (pftrf(Op::NoTrans, Uplo::Upper, k, Q_gram_dat))
        {
                this->chol_fail = true; // scheme failure 
                return 1;
        }

        tfsm(Op::NoTrans, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_gram_dat, Q_dat, m);
        
       return 0;
}

template <typename T> 
int Stab<T>::PLU(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<int64_t>& ipiv
){
        using namespace lapack;

        // Not using utility bc vector of int
        if(ipiv.size() < (uint64_t)n)
                ipiv.resize(n);

        if(getrf(m, n, A.data(), m, ipiv.data()))
                return 1; // failure condition

        get_L<T>(m, n, A);
        swap_rows<T>(m, n, A, ipiv);

        return 0;
}

template <typename T> 
int Orth<T>::HQR(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<T>& tau
){
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

        return 0;
}

#if !defined(__APPLE__)
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
#endif

template int Orth<float>::CholQR(int64_t m, int64_t k, std::vector<float>& Q);
template int Orth<double>::CholQR(int64_t m, int64_t k, std::vector<double>& Q);

template int Stab<float>::PLU(int64_t m, int64_t n, std::vector<float>& A, std::vector<int64_t>& ipiv);
template int Stab<double>::PLU(int64_t m, int64_t n, std::vector<double>& A, std::vector<int64_t>& ipiv);

template int Orth<float>::HQR(int64_t m, int64_t n, std::vector<float>& A, std::vector<float>& tau);
template int Orth<double>::HQR(int64_t m, int64_t n, std::vector<double>& A, std::vector<double>& tau);

#if !defined(__APPLE__)
template int Orth<float>::GEQR(int64_t m, int64_t n, std::vector<float>& A, std::vector<float>& tvec);
template int Orth<double>::GEQR(int64_t m, int64_t n, std::vector<double>& A, std::vector<double>& tvec); 
#endif
} // end namespace orth
