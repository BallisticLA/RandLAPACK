#ifndef randlapack_comps_orth_h
#define randlapack_comps_orth_h

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_orth.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

using namespace RandLAPACK::comps::util;

namespace RandLAPACK::comps::orth {

template <typename T>
class Stabilization {
    public:
        virtual ~Stabilization() {}

        virtual int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ) = 0;
};

template <typename T>
class CholQRQ : public Stabilization<T> {
    public:

        CholQRQ(bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
            chol_fail = false;
        };

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
        int cholqrq(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        );

        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ) {
            return cholqrq(m, k, Q);
        }

    public:
        bool chol_fail;
        bool cond_check;
        bool verbosity;

        // CholQR-specific
        std::vector<T> Q_gram;
        std::vector<T> Q_gram_cpy;
        std::vector<T> s;
};

// -----------------------------------------------------------------------------
template <typename T>
int CholQRQ<T>::cholqrq(
    int64_t m,
    int64_t k,
    std::vector<T>& Q
){

    T* Q_gram_dat = upsize(k * k, this->Q_gram);
    T* Q_dat = Q.data();

    // Find normal equation Q'Q - Just the upper triangular portion
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, Q_gram_dat, k);

    // Positive definite cholesky factorization
    if (lapack::potrf(Uplo::Upper, k, Q_gram_dat, k)) {
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

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, Q_gram_dat, k, Q_dat, m);
    return 0;
}





template <typename T>
class HQRQ : public Stabilization<T> {
    public:

        HQRQ(bool c_check, bool verb) {
            cond_check = c_check;
            verbosity = verb;
        };

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
        int hqrq(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            std::vector<T>& tau
        );

        int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            return hqrq(m, k, Q, this->tau);;
        }

    public:
        std::vector<T> tau;
        bool cond_check;
        bool verbosity;
};

// -----------------------------------------------------------------------------
template <typename T>
int HQRQ<T>::hqrq(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    std::vector<T>& tau
) {
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default

    upsize(n, tau);

    T* A_dat = A.data();
    T* tau_dat = tau.data();
    if(lapack::geqrf(m, n, A_dat, m, tau_dat))
        return 1; // Failure condition

    lapack::ungqr(m, n, n, A_dat, m, tau_dat);
    return 0;
}






template <typename T>
class PLUL : public Stabilization<T> {
    public:

        PLUL(bool c_check, bool verb) {
            this->cond_check = c_check;
            this->verbosity = verb;
        };

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
        int plul(
            int64_t m,
            int64_t k,
            std::vector<T>& Q,
            std::vector<int64_t>& ipiv
        );

        // Control of Stab types calls.
        virtual int call(
            int64_t m,
            int64_t k,
            std::vector<T>& Q
        ){
            return plul(m, k, Q, this->ipiv);
        }

    public:
        std::vector<int64_t> ipiv;
        bool cond_check;
        bool verbosity;
};


// -----------------------------------------------------------------------------
template <typename T>
int PLUL<T>::plul(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    std::vector<int64_t>& ipiv
){

    // Not using utility bc vector of int
    if(ipiv.size() < (uint64_t)n)
        ipiv.resize(n);

    T* A_dat = A.data();
    int64_t* ipiv_dat = ipiv.data();

    if(lapack::getrf(m, n, A_dat, m, ipiv_dat))
        return 1; // failure condition

    get_L(m, n, A);
    lapack::laswp(n, A_dat, m, 1, n, ipiv_dat, 1);

    return 0;
}

} // end namespace RandLAPACK::comps::rs
#endif
