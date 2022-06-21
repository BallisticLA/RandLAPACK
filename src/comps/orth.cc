#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

namespace RandLAPACK::comps::orth {

// Perfoms a Cholesky QR factorization
template <typename T> 
int chol_QR(
        int64_t m,
        int64_t k,
        std::vector<T>& Q // pointer to the beginning
) {
        using namespace blas;
        using namespace lapack;

        std::vector<T> Q_buf(k * k, 0.0);

        T* Q_dat = Q.data();
        T* Q_buf_dat = Q_buf.data();

        // Find normal equation Q'Q - Just the upper triangular portion
        syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 1.0, Q_buf_dat, k);

        // Positive definite cholesky factorization
        if (potrf(Uplo::Upper, k, Q_buf_dat, k) != 0)
                return 1; // scheme failure 
                
        // Q = Q * R^(-1)
        trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_buf_dat, k, Q_dat, m);	    
        return 0;
}

/*
Issue: 
It seems that if we want to pass around the function pointers to stabilization routines,
those would need to have the same parameter lists.
If that is true, then we would be re-allocating vectors at every iteration of an alg.
*/

// CholQR can also be used for stabilization, but it is defined separately.

template <typename T> 
void stab_LU(
        int64_t m,
        int64_t n,
        std::vector<T>& A
){
        using namespace lapack;
        std::vector<int64_t> ipiv(n, 0);

        getrf(m, n, A.data(), m, ipiv.data());
        RandLAPACK::comps::util::row_swap<T>(m, n, A, ipiv);
        RandLAPACK::comps::util::get_L<T>(m, n, A);
}

template <typename T> 
void stab_QR(
        int64_t m,
        int64_t n,
        std::vector<T>& A
){
        // Done via regular LAPACK's QR
        // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
        // tau needs to be a vector of all 2's by default
        using namespace lapack;
        T* A_dat = A.data();
        std::vector<T> tau(n, 2.0);
	T* tau_dat = tau.data();

        geqrf(m, n, A_dat, m, tau_dat);
        ungqr(m, n, n, A_dat, m, tau_dat);
}

template int chol_QR(int64_t m, int64_t k, std::vector<float>& Q);
template int chol_QR(int64_t m, int64_t k, std::vector<double>& Q);

template void stab_LU(int64_t m, int64_t n, std::vector<float>& A);
template void stab_LU(int64_t m, int64_t n, std::vector<double>& A);

template void stab_QR(int64_t m, int64_t n, std::vector<float>& A);
template void stab_QR(int64_t m, int64_t n, std::vector<double>& A);

} // end namespace orth