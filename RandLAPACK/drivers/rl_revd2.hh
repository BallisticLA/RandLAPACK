#ifndef randlapack_drivers_revd2_h
#define randlapack_drivers_revd2_h

#include "rl_syps.hh"
#include "rl_syrf.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

namespace RandLAPACK {

template <typename T, typename RNG>
class REVD2alg {
    public:

        virtual ~REVD2alg() {}

        virtual RandBLAS::RNGState<RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const T* A,
            int64_t& k,
            T tol,
            std::vector<T>& V,
            std::vector<T>& eigvals,
            RandBLAS::RNGState<RNG>& state
        ) = 0;

        virtual RandBLAS::RNGState<RNG> call(
            SymmetricLinearOperator<T> &A,
            int64_t& k,
            T tol,
            std::vector<T>& V,
            std::vector<T>& eigvals,
            RandBLAS::RNGState<RNG>& state
        ) = 0;
};

template <typename T, typename RNG>
class REVD2 : public REVD2alg<T, RNG> {
    public:

        // Constructor
        REVD2(
            RandLAPACK::SymmetricRangeFinder<T, RNG>& syrf_obj,
            int error_est_power_iters,
            bool verb = false
        ) : SYRF_Obj(syrf_obj) {
            error_est_p = error_est_power_iters;
            verbose = verb;
        }

        /// Computes a rank-k approximation to an EVD of a symmetric positive semidefinite matrix:
        ///     A_hat = V diag(eigvals) V^*,
        /// where V is a matrix of eigenvectors and eigvals is a vector of eigenvalues.
        /// 
        /// This function is adaptive. If the tolerance is not met, increases the rank
        /// estimation parameter by 2. Tolerance is the maximum of user-specified tol times
        /// 5 and computed 'nu' times 5. This is motivated by the fact that the approximation
        /// error will never be smaller than nu.
        /// 
        /// This code is identical to Algorithm E2 from https://arxiv.org/pdf/2110.02820.pdf.
        /// It uses a SymmetricRangeFinder for constructing a sketching operator.
        /// It has a lot of potential in terms of storage space optimization, 
        /// which, however, will affect readability.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-m matrix A, stored in a column-major format.
        ///     Must be SPD.
        ///
        /// @param[in] k
        ///     Column dimension of a sketch, k <= n.
        ///
        /// @param[in, out] V
        ///     On entry, is empty and may not have any space allocated for it.
        ///     On exit, stores m-by-k matrix matrix of (approximate) eigenvectors.
        ///
        /// @param[in, out] eigvals
        ///     On entry, is empty and may not have any space allocated for it.
        ///     On exit, stores k eigenvalues.
        ///
        /// @returns
        ///     An RNGState that the calling function should use the next
        ///     time it needs an RNGState.
        ///

        RandBLAS::RNGState<RNG> call(
            blas::Uplo uplo,
            int64_t m,
            const T* A,
            int64_t& k,
            T tol,
            std::vector<T>& V,
            std::vector<T>& eigvals,
            RandBLAS::RNGState<RNG>& state
        ) override;

        RandBLAS::RNGState<RNG> call(
            SymmetricLinearOperator<T> &A,
            int64_t& k,
            T tol,
            std::vector<T>& V,
            std::vector<T>& eigvals,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        RandLAPACK::SymmetricRangeFinder<T, RNG>& SYRF_Obj;
        int error_est_p;
        bool verbose;

        std::vector<T> Y;
        std::vector<T> Omega;
        std::vector<T> R;
        std::vector<T> S;
        std::vector<T> symrf_work;
};

// -----------------------------------------------------------------------------
/// Power scheme for error estimation, based on Algorithm E.1 from https://arxiv.org/pdf/2110.02820.pdf.
/// This routine is too specialized to be included into RandLAPACK::utils
/// p - number of algorithm iterations
/// vector_buf - buffer for vector operations
/// Mat_buf - buffer for matrix operations
/// All other parameters come from REVD2
template <typename T>
T power_error_est(
    SymmetricLinearOperator<T> &A,
    int64_t k,
    int p,
    T* vector_buf,
    T* V,
    T* Mat_buf,
    T* eigvals
) {
    int64_t m = A.m;
    T err = 0;
    for(int i = 0; i < p; ++i) {
        T g_norm = blas::nrm2(m, vector_buf, 1);
        // Compute g = g / ||g|| - we need this because dot product does not take in an alpha
        blas::scal(m, 1 / g_norm, vector_buf, 1);

        // Compute V' * g / ||g||
        // Using the second column of vector_buff as a buffer for matrix-vector product
        gemv(Layout::ColMajor, Op::Trans, m, k, 1.0, V, m, vector_buf, 1, 0.0, &vector_buf[m], 1);

        // Compute V*E, eigvals diag
        // Using Mat_buf as a buffer for V * diag(eigvals).
        for (int i = 0, j = 0; i < m * k; ++i) {
            Mat_buf[i] = V[i] * eigvals[j];
            if((i + 1) % m == 0 && i != 0)
                ++j;
        }

        // Compute V * diag(eigvals) * V' * g / ||g||
        // Using the third column of vector_buf as a buffer for matrix-vector product
        gemv(Layout::ColMajor, Op::NoTrans, m, k, 1.0, Mat_buf, m, &vector_buf[m], 1, 0.0, &vector_buf[2 * m], 1);
        // Compute A * g / ||g||
        // Using the forth column of vector_buff as a buffer for matrix-vector product
        A(Layout::ColMajor, 1, 1.0, vector_buf, m, 0.0, &vector_buf[3*m], m);
        // symv(Layout::ColMajor, uplo, m, 1.0, A, m, vector_buf, 1, 0.0, &vector_buf[3 * m], 1);

        // Compute w = (A * g / ||g|| - V * diag(eigvals) * V' * g / ||g||)
        // Result is stored in the 4th column of vector_buf
        blas::axpy(m, -1.0, &vector_buf[2 * m], 1, &vector_buf[3 * m], 1);
        // Compute (g / ||g||)' * w - this is our measure for the error
        err = blas::dot(m, vector_buf, 1, &vector_buf[3 * m], 1);	
        // v_0 <- v
        std::copy(&vector_buf[3 * m], &vector_buf[4 * m], vector_buf);
    }
    return err;
}

template <typename T>
T power_error_est(
    int64_t m,
    int64_t k,
    int p,
    T* vector_buf,
    T* V,
    blas::Uplo uplo,
    const T* A,
    T* Mat_buf,
    T* eigvals
) {
    ExplicitSymLinOp<T> A_linop(m, uplo, A, m);
    power_error_est(A_linop, k, p, vector_buf, V, Mat_buf, eigvals);
}


template <typename T, typename RNG>
RandBLAS::RNGState<RNG> REVD2<T, RNG>::call(
        SymmetricLinearOperator<T> &A,
        int64_t& k,
        T tol,
        std::vector<T>& V,
        std::vector<T>& eigvals,
        RandBLAS::RNGState<RNG>& state
) {
    int64_t m = A.m;
    T err = 0;
    RandBLAS::RNGState<RNG> error_est_state(state.counter, state.key);
    error_est_state.key.incr(1);
    while(true) {
        util::upsize(k, eigvals);
        T* V_dat = util::upsize(m * k, V);
        T* Y_dat = util::upsize(m * k, this->Y);
        T* Omega_dat = util::upsize(m * k, this->Omega);
        T* R_dat = util::upsize(k * k, this->R);
        T* S_dat = util::upsize(k * k, this->S);
        T* symrf_work_dat = util::upsize(m * k, this->symrf_work);

        // Construnct a sketching operator
        // If CholeskyQR is used for stab/orth here, RF can fail
        this->SYRF_Obj.call(A, k, this->Omega, state, symrf_work_dat);

        // Y = A * Omega
        A(blas::Layout::ColMajor, k, 1.0, Omega_dat, m, 0.0, Y_dat, m);

        T nu = std::numeric_limits<T>::epsilon() * lapack::lange(lapack::Norm::Fro, m, k, Y_dat, m);

        // We need Y = Y + v Omega
        // We further need R = chol(Omega' Y)
        // Solve this as R = chol(Omega' Y + v Omega'Omega)
        // Compute v Omega' Omega; syrk only computes the lower triangular part. Need full.
        blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, k, m, nu, Omega_dat, m, 0.0, R_dat, k);
        for(int i = 1; i < k; ++i)
            blas::copy(k - i, &R_dat[i + ((i-1) * k)], 1, &R_dat[(i - 1) + (i * k)], k);
        // Compute Omega' Y + v Omega' Omega
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Omega_dat, m, Y_dat, m, 1.0, R_dat, k);

        // Compute R = chol(Omega' Y + v Omega' Omega)
        // Looks like if POTRF gets passed a non-triangular matrix, it will also output a non-triangular one
        if(lapack::potrf(Uplo::Upper, k, R_dat, k))
            throw std::runtime_error("Cholesky decomposition failed.");
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
        RandBLAS::DenseDist  g{.n_rows = m, .n_cols = 1};
        error_est_state = RandBLAS::fill_dense(g, Omega_dat, error_est_state);

        err = power_error_est(A, k, this->error_est_p, Omega_dat, V_dat, Y_dat, eigvals.data()); 

        if(err <= 5 * std::max(tol, nu) || k == m) {
            break;
        } else if (2 * k > m) {
            k = m;
        } else {
            k = 2 * k;
        }
    }
    return 0;
}

template <typename T, typename RNG>
RandBLAS::RNGState<RNG> REVD2<T, RNG>::call(
        blas::Uplo uplo,
        int64_t m,
        const T* A,
        int64_t& k,
        T tol,
        std::vector<T>& V,
        std::vector<T>& eigvals,
        RandBLAS::RNGState<RNG> &state
) {
    ExplicitSymLinOp<T> A_linop(m, uplo, A, m);
    return this->call(A_linop, k, tol, V, eigvals, state);
}

} // end namespace RandLAPACK
#endif
