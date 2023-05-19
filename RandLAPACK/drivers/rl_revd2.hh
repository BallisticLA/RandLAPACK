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
            blas::Uplo uplo,
            std::vector<T>& A,
            int64_t& k,
            std::vector<T>& V,
            std::vector<T>& eigvals
        ) = 0;
};

template <typename T, typename RNG>
class REVD2 : public REVD2alg<T> {
    public:

        // Constructor
        REVD2(
            RandLAPACK::RangeFinder<T>& rf_obj,
            T tolerance,
            int power_iters,
            RandBLAS::base::RNGState<RNG> s,
            bool verb
        ) : RF_Obj(rf_obj) {
            tol = tolerance;
            p = power_iters;
            state = s;
            verbosity = verb;
        }

        /// Computes a rank-k approximation to an EVD of a symmetric positive definite matrix:
        ///     A= V diag(E) V*,
        /// where V is a matrix of eigenvectors and E is a vector of eigenvalues.
        /// Does so adaptively. 
        /// If the tolerance is not met, increases the rank estimation parameter by 2.
        /// Tolerance is the maximum of user-specified tol times 5 and computed 'nu' times 5.
        /// This is motivated by the fact that the approx error will never be smaller than nu.
        /// This code is identical to ALgorithm E2 from https://arxiv.org/pdf/2110.02820.pdf.
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
        ///     We are assuming it is either full or upper-triangular always.
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
        /// @param[out] eigvals
        ///     Stores k eigenvalues.
        ///
        /// @return = 0: successful exit
        ///

        int call(
            int64_t m,
            blas::Uplo uplo,
            std::vector<T>& A,
            int64_t& k,
            std::vector<T>& V,
            std::vector<T>& eigvals
        ) override;

    public:
        RandLAPACK::RangeFinder<T>& RF_Obj;
        RandBLAS::base::RNGState<RNG> state;
        T tol;
        int p;
        bool verbosity;

        std::vector<T> Y;
        std::vector<T> Omega;

        std::vector<T> R;
        std::vector<T> S;

        std::vector<T> A_cpy;
};

// -----------------------------------------------------------------------------
/// Power scheme for error estimation, based on algorithm E.1 from https://arxiv.org/pdf/2110.02820.pdf.
/// This routine is too specialized to be included into RandLAPACK::utils
/// p - number of algorithm iterations
/// vector_buf - buffer for vector operations
/// Mat_buf - buffer for matrix operations
/// All other parameters come from REVD2
template <typename T>
T power_error_est(
    int64_t m,
    int64_t k,
    int p,
    T* vector_buf,
    T* V,
    blas::Uplo uplo,
    T* A,
    T* Mat_buf,
    T* eigvals
) {
    T err = 0;
        
    for(int i = 0; i < p; ++i) {
        T g_norm = blas::nrm2(m, vector_buf, 1);
        // Compute g = g / ||g|| - we need this because dot product does not take in an alpha
        blas::scal(m, 1 / g_norm, vector_buf, 1);

        // Compute V'*g / ||g||
        // Using the second column of Omega as a buffer for matrix-vector product
        gemv(Layout::ColMajor, Op::Trans, m, k, 1.0, V, m, vector_buf, 1, 0.0, &vector_buf[m], 1);


        // Compute V*E, eigvals diag
        // Using Y as a buffer for V*E
        for (int i = 0, j = 0; i < m * k; ++i) {
            Mat_buf[i] = V[i] * eigvals[j];
            if((i + 1) % m == 0 && i != 0)
                ++j;
        }

        // Compute V*E*V'*g / ||g||
        // Using the third column of Omega as a buffer for matrix-vector product
        gemv(Layout::ColMajor, Op::NoTrans, m, k, 1.0, Mat_buf, m, &vector_buf[m], 1, 0.0, &vector_buf[2 * m], 1);
        // Compute A*g / ||g||
        // Using the forth column of Omega as a buffer for matrix-vector product
        symv(Layout::ColMajor, uplo, m, 1.0, A, m, vector_buf, 1, 0.0, &vector_buf[3 * m], 1);

        // Compute A*g / ||g|| - V*E*V'*g / ||g||
        // Result is stored in the 4th column of Omega
        blas::axpy(m, -1.0, &vector_buf[2 * m], 1, &vector_buf[3 * m], 1);
        // Compute (g / ||g||)' * (A*g / ||g|| - V*E*V'*g / ||g||) - this is our measure for the error
        err = blas::dot(m, vector_buf, 1, &vector_buf[3 * m], 1);	
        // v_0 <- v
        std::copy(&vector_buf[3 * m], &vector_buf[4 * m], vector_buf);
    }
    return err;
}



template <typename T, typename RNG>
int REVD2<T, RNG>::call(
        int64_t m, // m = n
        blas::Uplo uplo,
        std::vector<T>& A,
        int64_t& k,
        std::vector<T>& V,
        std::vector<T>& eigvals
){
    T err = 0;
    while(1) {
        T* A_dat = A.data();
        T* V_dat = util::upsize(m * k, V);
        util::upsize(k, eigvals);
        T* Y_dat = util::upsize(m * k, this->Y);
        T* Omega_dat = util::upsize(m * k, this->Omega);
        T* R_dat = util::upsize(k * k, this->R);
        T* S_dat = util::upsize(k * k, this->S);

        // Construnct a sketching operator
        // If CholeskyQR is used for stab/orth here, RF fails
        if(this->RF_Obj.call(m, m, A, k, this->Omega)) {
            printf("RF/RS FAILED\n");
            return 2;
        }

        // Y = A * S
        blas::symm(Layout::ColMajor, Side::Left, uplo, m, k, 1.0, A_dat, m, Omega_dat, m, 0.0, Y_dat, m);

        T nu = std::numeric_limits<double>::epsilon() * lapack::lange(lapack::Norm::Fro, m, k, Y_dat, m);

        // We need Y = Y + vS
        // We further need R = chol(S' Y)
        // Solve this as R = chol(S' Y + vS'S)
        // Compute vS'S - syrk only computes the lower triangular part. Need full.
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, k, m, nu, Omega_dat, m, 0.0, R_dat, k);
        // Fill the upper triangular part of S'S
        for(int i = 1; i < k; ++i)
            blas::copy(k - i, &R_dat[i + ((i-1) * k)], 1, &R_dat[(i - 1) + (i * k)], k);
        // Compute S' Y + vS'S
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Omega_dat, m, Y_dat, m, 1.0, R_dat, k);

        // Compute R = chol(S' Y + vS'S)
        // Looks like if POTRF gets passed a non-triangular matrix, it will also output a non-triangular one
        if(lapack::potrf(Uplo::Upper, k, R_dat, k)) {
            printf("CHOLESKY FACTORIZATION FAILED\n");
            return 1;
        }
        RandLAPACK::util::get_U(k, k, R);

        // B = Y(R')^-1 - need to transpose R
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_dat, k, Y_dat, m);

        //[V, S, ~] = SVD(B)
        // Although we don't need the right singular vectors, we need to give space for those.
        // Use R as a buffer for that.
        lapack::gesdd(Job::SomeVec, m, k, Y_dat, m, S_dat, V_dat, m, R_dat, k);

        // eigvals = diag(S^2)
        T buf;
        int64_t r = 0;
        int i;
        for(i = 0; i < k; ++i) {
            buf = std::pow(S[i], 2);
            eigvals[i] = buf;
            // r = number of entries in eigvals that are greater than v
            if(buf > nu)
                ++r;
        }

        // Undo regularlization
        // Need to make sure no eigenvalue is negative
        for(i = 0; i < r; ++i)
            (eigvals[i] - nu < 0) ? 0 : eigvals[i] -=nu;

        std::fill(&V_dat[m * r], &V_dat[m * k], 0.0);

        // Error estimation
        // Using the first column of Omega as a buffer for a random vector
        // To perform the following safely, need to make sure Omega has at least 4 columns
        Omega_dat = util::upsize(m * 4, this->Omega);
        RandBLAS::dense::DenseDist  g{.n_rows = m, .n_cols = 1};
        RandBLAS::dense::fill_buff(Omega_dat, g, state);

        err = power_error_est(m, k, p, Omega_dat, V_dat, uplo, A_dat, Y_dat, eigvals.data()); 

        if(err <= 5 * std::max(this->tol, nu) || k == m) {
            break;
        } else if (2 * k > m) {
            k = m;
        } else {
            k = 2 * k;
        }
    }
    return 0;
}

} // end namespace RandLAPACK
#endif