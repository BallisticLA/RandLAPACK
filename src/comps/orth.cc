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

template int chol_QR(int64_t m, int64_t k, std::vector<float>& Q);
template int chol_QR(int64_t m, int64_t k, std::vector<double>& Q);

} // end namespace orth