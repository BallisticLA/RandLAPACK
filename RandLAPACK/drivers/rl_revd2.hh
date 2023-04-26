#ifndef randlapack_drivers_revd2_h
#define randlapack_drivers_revd2_h

#include "rl_qb.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

namespace RandLAPACK {

template <typename T>
class REVD2alg {
    public:

        virtual ~REVD2alg() {}

        virtual int call(
            int64_t m,
            std::vector<T>& A,
            int64_t& k,
            std::vector<T>& V,
            std::vector<T>& E
        ) = 0;
};

template <typename T>
class REVD2 : public REVD2alg<T> {
    public:

        // Constructor
        REVD2(
            RandLAPACK::RangeFinder<T>& rf_obj,
            RandBLAS::base::RNGState<r123::Philox4x32> s,
            bool verb
        ) : RF_Obj(rf_obj) {
            state = s;
            verbosity = verb;
        }

        /// Computes an EVD of a symmetric positive definite matrix:
        ///     A= V diag(E) V*,
        /// where V is a matrix of eigenvectors and E is a vector of eigenvalues.
        /// Detailed description of this algorithm may be found in Section 4.2.2.
        /// of "the RandLAPACK book". 
        /// Uses RangeFinder for constructing a sketching operator.
        /// Has a lot of potential in terms of storage space optimization, 
        /// which, however, will affect readability.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-m matrix A, stored in a column-major format.
        ///     Must be SPD.
        ///
        /// @param[in] k
        ///     Column dimension of a sketch, k <= n.
        ///
        /// @param[in] V
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[in] E
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[out] V
        ///     Stores m-by-k matrix matrix of eigenvectors.
        ///
        /// @param[out] E
        ///     Stores k eigenvalues.
        ///
        /// @return = 0: successful exit
        ///

        int call(
            int64_t m,
            std::vector<T>& A,
            int64_t& k,
            std::vector<T>& V,
            std::vector<T>& E
        ) override;

    public:
        RandLAPACK::RangeFinder<T>& RF_Obj;
        RandBLAS::base::RNGState<r123::Philox4x32> state;
        bool verbosity;

        std::vector<T> Y;
        std::vector<T> Omega;

        std::vector<T> R;
        std::vector<T> S;
};

// -----------------------------------------------------------------------------
template <typename T>
int REVD2<T>::call(
        int64_t m, // m = n
        std::vector<T>& A,
        int64_t& k,
        std::vector<T>& V,
        std::vector<T>& E
){
    T* A_dat = A.data();
    T* V_dat = util::upsize(m * k, V);
    util::upsize(k, E);
    T* Y_dat = util::upsize(m * k, this->Y);
    T* Omega_dat = util::upsize(m * k, this->Omega);
    T* R_dat = util::upsize(k * k, this->R);
    T* S_dat = util::upsize(k * k, this->S);
    
    // Construnct a sketching operator
    if(this->RF_Obj.call(m, m, A, k, this->Omega))
        return 2;

    // Y = A * S
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, m, 1.0, A_dat, m, Omega_dat, m, 0.0, Y_dat, m);

    T v = std::sqrt(m) * std::numeric_limits<double>::epsilon() * lapack::lange(lapack::Norm::Fro, m, k, Y_dat, m);

    // We need Y = Y + vS
    // We further need R = chol(S' Y)
    // Solve this as R = chol(S' Y + vS'S)
    // Compute vS'S - syrk only computes the lower triangular part. Need full.
    blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, k, m, v, Omega_dat, m, 0.0, R_dat, k);
    // Fill the upper triangular part of S'S
    for(int i = 1; i < k; ++i)
        blas::copy(k - i, R_dat + i + ((i-1) * k), 1, R_dat + (i - 1) + (i * k), k);
    // Compute S' Y + vS'S
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Omega_dat, m, Y_dat, m, 1.0, R_dat, k);

    // Compute R = chol(S' Y + vS'S)
    // Looks like if POTRF gets passed a non-triangular matrix, it will also output a non-triangular one
    if(lapack::potrf(Uplo::Lower, k, R_dat, k)){
        printf("CHOLESKY FAILED\n");
        return 1;
    }
    RandLAPACK::util::get_L(k, k, R, 0);

    // B = Y(R')^-1 - need to transpose R
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Lower, Op::Trans, Diag::NonUnit, m, k, 1.0, R_dat, k, Y_dat, m);

    //[V, S, ~] = SVD(B)
    // Although we don't need the right singular vectors, we need to give space for those.
    // Use R as a buffer for that.
    lapack::gesdd(Job::SomeVec, m, k, Y_dat, m, S_dat, V_dat, m, R_dat, k);
    
    // E = diag(S^2)
    T buf;
    int64_t r = 0;
    int i;
    for(i = 0; i < k; ++i) {
        buf = std::pow(S[i], 2);
        E[i] = buf;
        // r = number of entries in E that are greater than v
        if(buf > v)
            ++r;
    }

    // Undo regularlization
    for(i = 0; i < r; ++i)
        E[i] -=v;
    
    std::fill(V.begin() + m * r, V.end(), 0.0);
    
    return 0;
}

} // end namespace RandLAPACK
#endif
