#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

namespace RandLAPACK::comps::orth {

// Perfoms a Cholesky QR factorization
template <typename T> 
void chol_QR(
        int64_t m,
        int64_t k,
        T* Q // pointer to the beginning
) {
        using namespace blas;
        using namespace lapack;

        std::vector<T> Q_buf(k * k, 0.0);
        // Find normal equation Q'Q - Just the upper triangular portion
        syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q, m, 1.0, Q_buf.data(), k);

        // Positive definite cholesky factorization
        potrf(Uplo::Upper, k, Q_buf.data(), k);

        // Inverse of an upper-triangular matrix
        trtri(Uplo::Upper, Diag::NonUnit, k, Q_buf.data(), k);
        // Q = Q * R^(-1)
        std::vector<T> Q_chol(m * k, 0.0);
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, Q, m, Q_buf.data(), k, 0.0, Q_chol.data(), m);

        // Copy the result into Q
        lacpy(MatrixType::General, m, k, Q_chol.data(), m, Q, m);
}

template void chol_QR(int64_t m, int64_t k, float* Q);
template void chol_QR(int64_t m, int64_t k, double* Q);

} // end namespace orth