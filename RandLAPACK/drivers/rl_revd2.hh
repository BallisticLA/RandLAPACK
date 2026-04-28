#pragma once

#include "rl_syps.hh"
#include "rl_syrf.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace RandLAPACK {


// -----------------------------------------------------------------------------
/// Power scheme for error estimation, based on Algorithm E.1 from https://arxiv.org/pdf/2110.02820.pdf.
/// p - number of algorithm iterations
/// vector_buf - buffer for vector operations (must hold at least 4*m elements)
/// Mat_buf    - buffer for matrix operations (must hold at least m*k elements)
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
        blas::scal(m, 1 / g_norm, vector_buf, 1);

        // V' * g/||g|| — result in column 1 of vector_buf
        gemv(Layout::ColMajor, Op::Trans, m, k, 1.0, V, m, vector_buf, 1, 0.0, &vector_buf[m], 1);

        // V * diag(eigvals) into Mat_buf
        for (int i = 0, j = 0; i < m * k; ++i) {
            Mat_buf[i] = V[i] * eigvals[j];
            if((i + 1) % m == 0 && i != 0)
                ++j;
        }

        // V * diag(eigvals) * V' * g/||g|| — result in column 2 of vector_buf
        gemv(Layout::ColMajor, Op::NoTrans, m, k, 1.0, Mat_buf, m, &vector_buf[m], 1, 0.0, &vector_buf[2 * m], 1);
        // A * g/||g|| — result in column 3 of vector_buf
        A(Layout::ColMajor, 1, 1.0, vector_buf, m, 0.0, &vector_buf[3*m], m);

        // w = A*g/||g|| - V*diag(eigvals)*V'*g/||g||
        blas::axpy(m, -1.0, &vector_buf[2 * m], 1, &vector_buf[3 * m], 1);
        // error estimate: (g/||g||)' * w
        err = blas::dot(m, vector_buf, 1, &vector_buf[3 * m], 1);
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

        // Internal working buffers — owned by this object, grown with calloc/free
        T* Y          = nullptr; int64_t Y_sz          = 0;
        T* Omega      = nullptr; int64_t Omega_sz      = 0;
        T* R          = nullptr; int64_t R_sz          = 0;
        T* S          = nullptr; int64_t S_sz          = 0;
        T* symrf_work = nullptr; int64_t symrf_work_sz = 0;

        REVD2(
            SYRF_t &syrf_obj,
            int error_est_power_iters,
            bool verb = false
        ) : syrf(syrf_obj) {
            error_est_p = error_est_power_iters;
            verbose = verb;
        }

        ~REVD2() {
            free(Y);
            free(Omega);
            free(R);
            free(S);
            free(symrf_work);
        }

        /// Computes a rank-k approximation to an EVD of a symmetric positive semidefinite matrix:
        ///     A_hat = V diag(eigvals) V^*,
        /// where V is a matrix of eigenvectors and eigvals is a vector of eigenvalues.
        ///
        /// Adaptive: if tolerance is not met, doubles k until convergence or k == m.
        /// Set error_est_power_iters=0 and tol=0 for fixed-rank single-pass behavior
        /// (k never changes, exits after one iteration).
        ///
        /// This code is identical to Algorithm E2 from https://arxiv.org/pdf/2110.02820.pdf.
        ///
        /// @param[in] m       Number of rows/cols of A.
        /// @param[in] A       m-by-m matrix, column-major, must be SPD.
        /// @param[in,out] k   Sketch rank on entry; final rank on exit (may grow in adaptive mode).
        /// @param[in] tol     Convergence tolerance. Use 0.0 for fixed-rank.
        /// @param[in,out] V   On exit, m-by-k matrix of approximate eigenvectors.
        /// @param[in,out] eigvals  On exit, k approximate eigenvalues.
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
            int64_t m = A.dim;
            T err = 0;
            RandBLAS::RNGState<RNG> error_est_state(state.counter, state.key);
            error_est_state.key.incr(1);
            while(true) {
                // Resize caller-owned output buffers (std::vector handles dynamic k)
                util::upsize(k, eigvals);
                T* V_dat = util::upsize(m * k, V);

                // Grow internal working buffers if needed
                if (m * k > Y_sz) {
                    free(Y);
                    Y = (T*) calloc(m * k, sizeof(T));
                    Y_sz = m * k;
                }
                // Omega must hold at least max(m*k, m*4) — the error estimator needs 4 columns
                int64_t omega_needed = std::max(m * k, m * (int64_t)4);
                if (omega_needed > Omega_sz) {
                    free(Omega);
                    Omega = (T*) calloc(omega_needed, sizeof(T));
                    Omega_sz = omega_needed;
                }
                if (k * k > R_sz) {
                    free(R);
                    R = (T*) calloc(k * k, sizeof(T));
                    R_sz = k * k;
                }
                if (k * k > S_sz) {
                    free(S);
                    S = (T*) calloc(k * k, sizeof(T));
                    S_sz = k * k;
                }
                if (m * k > symrf_work_sz) {
                    free(symrf_work);
                    symrf_work = (T*) calloc(m * k, sizeof(T));
                    symrf_work_sz = m * k;
                }

                // Construct sketching operator: Omega = orth(A * sketch)
                this->syrf.call(A, k, this->Omega, state, symrf_work);

                // Y = A * Omega
                A(Layout::ColMajor, k, 1.0, Omega, m, 0.0, Y, m);

                // Stabilization parameter nu = eps * ||Y||_F
                T nu = std::numeric_limits<T>::epsilon() * lapack::lange(Norm::Fro, m, k, Y, m);

                // R = chol(Omega'*Y + nu * Omega'*Omega) — regularized to ensure PD
                // syrk fills lower triangle; copy to upper for full symmetric matrix
                blas::syrk(Layout::ColMajor, Uplo::Lower, Op::Trans, k, m, nu, Omega, m, 0.0, R, k);
                for(int i = 1; i < k; ++i)
                    blas::copy(k - i, &R[i + ((i-1) * k)], 1, &R[(i - 1) + (i * k)], k);
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Omega, m, Y, m, 1.0, R, k);

                if(lapack::potrf(Uplo::Upper, k, R, k))
                    throw std::runtime_error("Cholesky decomposition failed.");
                RandLAPACK::util::get_U(k, k, R, k);

                // B = Y * (R')^{-1} — overwrites Y
                blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R, k, Y, m);

                // [V, S, ~] = SVD(B); use R as buffer for discarded right singular vectors
                lapack::gesdd(Job::SomeVec, m, k, Y, m, S, V_dat, m, R, k);

                // eigvals = S^2 - nu (undo regularization; clamp to zero)
                T buf;
                int64_t r = 0;
                int i;
                for(i = 0; i < k; ++i) {
                    buf = std::pow(S[i], 2);
                    eigvals[i] = buf;
                    if(buf > nu)
                        ++r;
                }
                for(i = 0; i < r; ++i)
                    (eigvals[i] - nu < 0) ? 0 : eigvals[i] -= nu;

                std::fill(&V_dat[m * r], &V_dat[m * k], 0.0);

                // Error estimation using Omega as a scratch buffer (needs 4 columns = 4*m)
                RandBLAS::DenseDist g(m, 1);
                error_est_state = RandBLAS::fill_dense(g, Omega, error_est_state);

                err = power_error_est(A, k, this->error_est_p, Omega, V_dat, Y, eigvals.data());

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
