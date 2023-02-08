#include <cstdint>
#include <vector>

#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::orth {

// -----------------------------------------------------------------------------
/// Performs Cholesky QR factorization. Outputs the Q-factor only.
/// Optionally checks the condition number of R-factor before computing the Q-factor.
///
/// Templated for `float` and `double` types.
///
/// @param[in] m
///     The number of rows in the matrix Q.
///
/// @param[in] k
///     The number of columns in the matrix Q.
///
/// @param[in] Q
///     The m-by-k matrix, stored in a column-major format.
///
/// @param[out] Q
///     Overwritten with an orthogonal Q-factor.
///     
///
/// @return = 0: successful exit
///
template <typename T> 
int CholQRQ<T>::cholqrq(
    int64_t m,
    int64_t k,
    std::vector<T>& Q
){
    using namespace blas;
    using namespace lapack;
        
    T* Q_gram_dat = upsize(k * k, this->Q_gram);
    T* Q_dat = Q.data();

    // Find normal equation Q'Q - Just the upper triangular portion        
    syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, Q_gram_dat, k);

    // Positive definite cholesky factorization
    if (potrf(Uplo::Upper, k, Q_gram_dat, k)) {
        if(this->verbosity) {
            printf("CHOLESKY QR FAILED\n");
        }
        this->chol_fail = true; // scheme failure 
        return 1;
    }

    // Scheme may succeed, but output garbage
    if(this->cond_check) {
        if(cond_num_check(k, k, Q_gram, this->Q_gram_cpy, this->s, this->verbosity) > (1 / std::sqrt(std::numeric_limits<T>::epsilon()))){
        //        return 1;
        }
    }

    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_gram_dat, k, Q_dat, m);
    return 0;
}

// -----------------------------------------------------------------------------
/// Performs an unpivoted LU factorization. Outputs the L-factor only.
/// Uses L-extraction routine and LASWP().
///
/// Templated for `float` and `double` types.
///
/// @param[in] m
///     The number of rows in the matrix A.
///
/// @param[in] n
///     The number of columns in the matrix A.
///
/// @param[in] A
///     The m-by-n matrix, stored in a column-major format.
///
/// @param[in] ipiv
///     Buffer for the pivot vector.
///     
/// @param[out] A
///     Overwritten by the lower-triangular factor L with interchanged rows,
///     L[ipiv,:].
///
/// @return = 0: successful exit
///
template <typename T> 
int PLUL<T>::plul(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    std::vector<int64_t>& ipiv
){
    using namespace lapack;

    // Not using utility bc vector of int
    if(ipiv.size() < (uint64_t)n)
        ipiv.resize(n);

    T* A_dat = A.data();
    int64_t* ipiv_dat = ipiv.data(); 

    if(getrf(m, n, A_dat, m, ipiv_dat))
        return 1; // failure condition

    get_L(m, n, A);
    laswp(n, A_dat, m, 1, n, ipiv_dat, 1);

    return 0;
}

// -----------------------------------------------------------------------------
/// Performs a Householder QR factorization. Outputs the Q-factor only.
/// Uses UNGQR() to form Q explicitly.
///
/// Templated for `float` and `double` types.
///
/// @param[in] m
///     The number of rows in the matrix A.
///
/// @param[in] n
///     The number of columns in the matrix A.
///
/// @param[in] A
///     The m-by-n matrix, stored in a column-major format.
///
/// @param[in] tau
///     Buffer for the scalar factor array.
///     
/// @param[out] A
///     Overwritten explicitly with an orthogonal Q-factor.
///
/// @return = 0: successful exit
///
template <typename T> 
int HQRQ<T>::hqrq(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    std::vector<T>& tau
){
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default
    using namespace lapack;

    upsize(n, tau);

    T* A_dat = A.data();
    T* tau_dat = tau.data();
    if(geqrf(m, n, A_dat, m, tau_dat))
        return 1; // Failure condition

    ungqr(m, n, n, A_dat, m, tau_dat);
    return 0;
}

template int CholQRQ<float>::cholqrq(int64_t m, int64_t k, std::vector<float>& Q);
template int CholQRQ<double>::cholqrq(int64_t m, int64_t k, std::vector<double>& Q);

template int PLUL<float>::plul(int64_t m, int64_t n, std::vector<float>& A, std::vector<int64_t>& ipiv);
template int PLUL<double>::plul(int64_t m, int64_t n, std::vector<double>& A, std::vector<int64_t>& ipiv);

template int HQRQ<float>::hqrq(int64_t m, int64_t n, std::vector<float>& A, std::vector<float>& tau);
template int HQRQ<double>::hqrq(int64_t m, int64_t n, std::vector<double>& A, std::vector<double>& tau);

} // end namespace orth
