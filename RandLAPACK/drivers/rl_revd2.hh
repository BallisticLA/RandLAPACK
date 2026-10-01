#pragma once

#include "rl_exceptions.hh"
#include "rl_syps.hh"
#include "rl_syrf.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "rl_nystrom_recovery.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>

namespace RandLAPACK {


// -----------------------------------------------------------------------------
/// Power scheme for error estimation, based on Algorithm E.1 from https://arxiv.org/pdf/2110.02820.pdf.
/// This routine is too specialized to be included into RandLAPACK::utils
/// p - number of algorithm iterations
/// vector_buf - buffer for vector operations
/// Mat_buf - buffer for matrix operations
/// All other parameters come from REVD2
template <typename T, linops::SymmetricLinearOperator SLO>
T power_error_est(
    SLO &A,
    int64_t k,
    int p,
    T* vector_buf,
    T* V,
    T* Mat_buf,
    T* eigvals
) {
    int64_t m = A.dim;
    T err = 0;
    for(int i = 0; i < p; ++i) {
        T g_norm = blas::nrm2(m, vector_buf, 1);
        if (g_norm == (T)0) return (T)0;
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
        blas::axpy(m, (T) -1.0, &vector_buf[2 * m], 1, &vector_buf[3 * m], 1);
        // Compute (g / ||g||)' * w - this is our measure for the error
        err = blas::dot(m, vector_buf, 1, &vector_buf[3 * m], 1);	
        // v_0 <- v
        std::copy(&vector_buf[3 * m], &vector_buf[4 * m], vector_buf);
    }
    return err;
}


template <typename SYRF_t>
class REVD2 {
    public:
        using T   = typename SYRF_t::T;
        using RNG = typename SYRF_t::RNG;
        SYRF_t &syrf;
        int error_est_p;
        bool verbose;

        std::vector<T> Y;
        std::vector<T> Omega;
        std::vector<T> R;
        std::vector<T> S;
        std::vector<T> symrf_work;

        // Constructor
        REVD2(
            SYRF_t &syrf_obj,
            int error_est_power_iters,
            bool verb = false
        ) : syrf(syrf_obj) {
            error_est_p = error_est_power_iters;
            verbose = verb;
        }

        /// Computes a rank-k approximation to an EVD of a symmetric positive semidefinite matrix:
        ///     A_hat = V diag(eigvals) V^*,
        /// where V is a matrix of eigenvectors and eigvals is a vector of eigenvalues.
        /// 
        /// This function is adaptive. If the tolerance is not met, doubles the rank
        /// estimation parameter. The stopping threshold is 5*max(tol, nu),
        /// where nu accounts for rounding error in the spectral recovery.
        /// 
        /// The adaptive scheme follows Algorithm E2 from https://arxiv.org/pdf/2110.02820.pdf.
        /// A SymmetricRangeFinder constructs the sketch. Spectral recovery uses
        /// the same shifted Nyström kernel as NystromEVD, with
        /// nu = sqrt(m)*epsilon*||A*Omega||_F. It shifts both the sampled image
        /// and the Gram matrix, and retains orthonormal columns when eigenvalues
        /// are clamped to zero. Set error_est_power_iters to zero for fixed rank;
        /// NystromEVD also provides a fixed-rank interface with sparse sketches.
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
        int call(
            Uplo uplo,
            int64_t m,
            const T* A,
            int64_t &k,
            T tol,
            std::vector<T> &V,
            std::vector<T> &eigvals,
            RandBLAS::RNGState<RNG> &state
        ) {
            // Input parameter validation. Bad inputs would otherwise propagate to a
            // downstream BLAS/LAPACK failure or a segfault, the latter fatal when
            // REVD2 is called through a binding layer (e.g. MEX/MATLAB).
            randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
            randlapack_require(k > 0) << "target rank k=" << k << " must be > 0";
            randlapack_require(tol >= (T)0) << "tol=" << tol << " must be >= 0";
            randlapack_require(!(A == nullptr && m > 0)) << "A buffer is null but m=" << m << " > 0";
            linops::ExplicitSymLinOp<T> A_linop(m, uplo, A, m, Layout::ColMajor);
            return this->call(A_linop, k, tol, V, eigvals, state);
        }

        template <linops::SymmetricLinearOperator SLO>
        int call(
            SLO &A,
            int64_t &k,
            T tol,
            std::vector<T> &V,
            std::vector<T> &eigvals,
            RandBLAS::RNGState<RNG> &state
        ) {
            // Input parameter validation; same MEX-safety motivation as above.
            randlapack_require(k > 0) << "target rank k=" << k << " must be > 0";
            randlapack_require(tol >= (T)0) << "tol=" << tol << " must be >= 0";
            int64_t m = A.dim;
            randlapack_require(k <= m) << "target rank k=" << k << " must be <= m=" << m;
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
                this->syrf.call(A, k, this->Omega, state, symrf_work_dat);

                // Y = A * Omega
                A(Layout::ColMajor, k, 1.0, Omega_dat, m, 0.0, Y_dat, m);

                int64_t clamped_eigenvalues = 0;
                T nu = detail::nystrom_recovery<T>(m, k,
                    {Y_dat, R_dat, S_dat, R_dat, clamped_eigenvalues},
                    V_dat, eigvals.data(),
                    [&](T shift) {
                        blas::axpy(m * k, shift, Omega_dat, 1, Y_dat, 1);
                    },
                    [&] {
                        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m,
                                   (T)1, Omega_dat, m, Y_dat, m, (T)0, R_dat, k);
                    }, -1, "REVD2");

                // Error estimation
                // Using the first column of Omega as a buffer for a random vector
                // To perform the following safely, need to make sure Omega has at least 4 columns
                Omega_dat = util::upsize(m * 4, this->Omega);
                RandBLAS::DenseDist  g(m, 1);
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

};


} // end namespace RandLAPACK
