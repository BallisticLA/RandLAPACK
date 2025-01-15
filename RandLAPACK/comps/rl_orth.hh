#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

namespace RandLAPACK {

template <typename T>
class Stabilization {
    public:
        virtual ~Stabilization() {}

        virtual int call(
            int64_t m,
            int64_t k,
            T* A
        ) = 0;
};

template <typename T>
class CholQRQ : public Stabilization<T> {
    public:

        CholQRQ(bool c_check, bool verb) {
            cond_check = c_check;
            verbose = verb;
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
        /// @param[in] A
        ///     The m-by-k matrix, stored in a column-major format.
        ///
        /// @param[out] A
        ///     Overwritten with an orthogonal Q-factor.
        ///
        ///
        /// @return = 0: successful exit
        ///
        int call(
            int64_t m,
            int64_t k,
            T* A
        );

    public:
        bool chol_fail;
        bool cond_check;
        bool verbose;
};

// -----------------------------------------------------------------------------
template <typename T>
int CholQRQ<T>::call(
    int64_t m,
    int64_t k,
    T* A
){

    T* A_gram  = new T[k * k]();

    // Find normal equation Q'Q - Just the upper triangular portion
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A, m, 0.0, A_gram, k);

    // Positive definite cholesky factorization
    if (lapack::potrf(Uplo::Upper, k, A_gram, k)) {
        if(this->verbose) {
            printf("CHOLESKY QR FAILED\n");
        }
        this->chol_fail = true; // scheme failure
        delete[] A_gram;
        return 1;
    }

    // Scheme may succeed, but output garbage
    if(this->cond_check) {
        if(util::cond_num_check(k, k, A_gram, this->verbose) > (1 / std::sqrt(std::numeric_limits<T>::epsilon()))){
                delete[] A_gram;
                return 1;
        }
    }

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, A_gram, k, A, m);
    delete[] A_gram;
    return 0;
}

template <typename T>
class HQRQ : public Stabilization<T> {
    public:

        HQRQ(bool c_check, bool verb) {
            cond_check = c_check;
            verbose = verb;
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

        int call(
            int64_t m,
            int64_t k,
            T* A
        );

    public:
        bool cond_check;
        bool verbose;
};

// -----------------------------------------------------------------------------
template <typename T>
int HQRQ<T>::call(
    int64_t m,
    int64_t n,
    T* A
) {
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default

    T* tau  = new T[n]();

    if(lapack::geqrf(m, n, A, m, tau)) {
        delete[] tau;
        return 1; // Failure condition
    }

    lapack::ungqr(m, n, n, A, m, tau);
    delete[] tau;
    return 0;
}

template <typename T>
class PLUL : public Stabilization<T> {
    public:

        PLUL(bool c_check, bool verb) {
            this->cond_check = c_check;
            this->verbose = verb;
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
        int call(
            int64_t m,
            int64_t k,
            T* A
        );

    public:
        bool cond_check;
        bool verbose;
};


// -----------------------------------------------------------------------------
template <typename T>
int PLUL<T>::call(
    int64_t m,
    int64_t n,
    T* A
){
    int64_t* ipiv  = new int64_t[n]();

    if(lapack::getrf(m, n, A, m, ipiv)) {
        delete[] ipiv;
        return 1; // failure condition
    }

    util::get_L(m, n, A, 1);
    lapack::laswp(n, A, m, 1, n, ipiv, 1);

    delete[] ipiv;
    return 0;
}

} // end namespace RandLAPACK
