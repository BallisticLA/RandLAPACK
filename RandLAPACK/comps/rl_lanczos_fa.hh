#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <concepts>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RandLAPACK {


/// d-step block Lanczos for matrix function application f(A)B.
///
/// Given a SymmetricLinearOperator A and a matrix B (n×s), approximates
/// f(A)B column-wise using a Krylov subspace of dimension d per column.
///
/// Each column b_j of B generates an independent Krylov sequence. The
/// recurrence produces a d×d symmetric tridiagonal T_j per column; f(A)b_j
/// is approximated by the first column of Q_j * f(T_j) * Q_j' * b_j, where
/// Q_j = [q_1,...,q_d] is the Lanczos basis and f(T_j) is evaluated via a
/// dense d×d tridiagonal eigendecomposition (lapack::stev).
///
/// Key formula (Tyler Chen): since q_1 = b_j/||b_j||, we have
///   f(A)b_j ≈ ||b_j|| * Q_j * S_j * diag(f(θ_j)) * S_j[0,:]^T
/// where (S_j, θ_j) = eig(T_j) from stev and S_j[0,:] is the first row
/// of the eigenvector matrix.
///
/// f is any scalar callable T → T (e.g., sqrt, log, pow).
/// The stev calls per column are independent and run in parallel (OpenMP).
///
/// Memory layout: K is stored as (d+1) contiguous n×s blocks.
///   K[step * n * s + col * n + row] = entry (row, col) of the step-th block.
/// This keeps each step's n×s matrix contiguous for batch matvec via SLO,
/// while allowing strided gemv (lda = n*s) for per-column reconstruction.
///
/// @tparam T    Floating-point scalar type.
/// @tparam RNG  Random number generator type (unused here; kept for API uniformity).
template <typename T, typename RNG>
class LanczosFA {
public:
    /// Reorthogonalization control.
    /// -1 = full (project out all previous Krylov vectors after each step).
    ///  0 = none (vanilla Lanczos, per Persson's reference implementation).
    /// Lanczos-FA tolerates loss of orthogonality better than eigenvalue
    /// Lanczos (Paige-Greenbaum theory), so vanilla often works in practice.
    /// Full reorthogonalization is the safe default for a numerical library.
    int64_t reorth = -1;

    // Internal buffers — grown with new/delete[], never shrunk between calls.
    // K:     (d+1) × n × s — Krylov basis blocks; see layout note above.
    // alpha: s × d         — tridiagonal diagonals, alpha[j*d + i] = α_{i,j}.
    // beta:  s × (d-1)     — tridiagonal subdiagonals, beta[j*(d-1) + i] = β_{i+1,j}.
    // normb: s             — column norms of B before normalization.
    T*      K      = nullptr; int64_t K_sz      = 0;
    T*      alpha  = nullptr; int64_t alpha_sz  = 0;
    T*      beta   = nullptr; int64_t beta_sz   = 0;
    T*      normb  = nullptr; int64_t normb_sz  = 0;

    ~LanczosFA() { delete[] K; delete[] alpha; delete[] beta; delete[] normb; }

    // ------------------------------------------------------------------
    /// Run d-step block Lanczos recurrence.
    /// Fills K, alpha, beta, normb from B (n×s column-major).
    /// Calls A exactly d times, each application to an n×s matrix.
    ///
    /// @param[in]  A    SymmetricLinearOperator — matvec oracle.
    /// @param[in]  B    n×s input matrix (column-major); not modified.
    /// @param[in]  n    Dimension of A.
    /// @param[in]  s    Number of right-hand sides (Hutchinson samples).
    /// @param[in]  d    Number of Lanczos steps.
    template <linops::SymmetricLinearOperator SLO>
    void run(SLO& A, const T* B, int64_t n, int64_t s, int64_t d) {
        // Grow buffers if needed
        if ((d + 1) * n * s > K_sz) {
            delete[] K;
            K = new T[(d + 1) * n * s];
            K_sz = (d + 1) * n * s;
        }
        if (d * s > alpha_sz) {
            delete[] alpha;
            alpha = new T[d * s];
            alpha_sz = d * s;
        }
        if ((d - 1) * s > beta_sz && d > 1) {
            delete[] beta;
            beta = new T[(d - 1) * s];
            beta_sz = (d - 1) * s;
        }
        if (s > normb_sz) {
            delete[] normb;
            normb = new T[s];
            normb_sz = s;
        }

        // Step 0: q_1 = column-normalize B; store in K[:,:,0]
        T* K0 = K;
        std::copy(B, B + n * s, K0);
        for (int64_t j = 0; j < s; ++j) {
            T nrm = blas::nrm2(n, K0 + j * n, 1);
            normb[j] = nrm;
            blas::scal(n, (T)1.0 / nrm, K0 + j * n, 1);
        }

        // Step 0 matvec: K[:,:,1] = A * K[:,:,0]
        T* K1 = K + n * s;
        A(Layout::ColMajor, s, (T)1.0, K0, n, (T)0.0, K1, n);

        // α[0, j] = dot(K1[:,j], K0[:,j]) — diagonal entry for step 0
        for (int64_t j = 0; j < s; ++j)
            alpha[j * d + 0] = blas::dot(n, K1 + j * n, 1, K0 + j * n, 1);

        // Main Lanczos loop: steps 1..d-1
        // At the start of iteration i:
        //   K[:,:,i]   = A*q_i - β_i*q_{i-1}  (partial three-term, β part done last iter)
        //   K[:,:,i+1] is free scratch
        // This iteration:
        //   (1) subtract α_i*q_i to complete three-term → K_{i} becomes unnormalized q_{i+1}
        //   (2) optional reorthogonalization
        //   (3) β_{i+1} = ||K_{i+1}||
        //   (4) normalize K_{i+1} → q_{i+1}
        //   (5) K_{i+2} = A*q_{i+1} - β_{i+1}*q_i  (start of next three-term)
        //   (6) α_{i+1} = dot(K_{i+2}, q_{i+1})
        for (int64_t i = 0; i < d - 1; ++i) {
            T* K_prev = K + i * n * s;           // K[:,:,i] = q_i (normalized)
            T* K_curr = K + (i + 1) * n * s;     // K[:,:,i+1] = partial (A*q_i - β_i*q_{i-1})
            T* K_new  = K + (i + 2) * n * s;     // K[:,:,i+2] = scratch for A*q_{i+1}

            // (1) Complete three-term: K_curr -= α_i * K_prev, per column
            for (int64_t j = 0; j < s; ++j)
                blas::axpy(n, -alpha[j * d + i], K_prev + j * n, 1, K_curr + j * n, 1);

            // (2) Optional full reorthogonalization against all q_0,...,q_i
            // Subtract projections onto previous Krylov vectors to recover lost
            // orthogonality accumulated in floating-point arithmetic.
            int64_t reorth_steps = (reorth < 0) ? (i + 1) : reorth;
            for (int64_t prev = 0; prev < reorth_steps; ++prev) {
                T* K_p = K + prev * n * s;
                for (int64_t j = 0; j < s; ++j) {
                    T coeff = blas::dot(n, K_curr + j * n, 1, K_p + j * n, 1);
                    blas::axpy(n, -coeff, K_p + j * n, 1, K_curr + j * n, 1);
                }
            }

            // (3) β_{i+1} = column norms and (4) normalize to get q_{i+1}
            for (int64_t j = 0; j < s; ++j) {
                T nrm = blas::nrm2(n, K_curr + j * n, 1);
                beta[j * (d - 1) + i] = nrm;
                blas::scal(n, (T)1.0 / nrm, K_curr + j * n, 1);
            }

            // (5) K_new = A*q_{i+1} - β_{i+1}*q_i  (one batch matvec + axpy per column)
            A(Layout::ColMajor, s, (T)1.0, K_curr, n, (T)0.0, K_new, n);
            for (int64_t j = 0; j < s; ++j)
                blas::axpy(n, -beta[j * (d - 1) + i], K_prev + j * n, 1, K_new + j * n, 1);

            // (6) α_{i+1} = dot(K_new[:,j], q_{i+1}[:,j])
            for (int64_t j = 0; j < s; ++j)
                alpha[j * d + (i + 1)] = blas::dot(n, K_new + j * n, 1, K_curr + j * n, 1);
        }
    }

    // ------------------------------------------------------------------
    /// Evaluate f(A)B from precomputed Krylov data (K, alpha, beta, normb).
    /// Per column j: eigendecompose T_j via lapack::stev, then compute
    ///   out[:,j] = normb[j] * Q_j * S_j * diag(f(θ_j)) * S_j[0,:]^T
    /// where S_j[0,:] is the first row of the eigenvector matrix.
    /// Per-column stev calls are independent — parallelized with OpenMP.
    ///
    /// @param[in]  f    Scalar callable T→T applied to tridiagonal eigenvalues.
    /// @param[in]  n    Dimension of A.
    /// @param[in]  s    Number of right-hand sides.
    /// @param[in]  d    Number of Lanczos steps (tridiagonal size).
    /// @param[out] out  n×s output matrix (column-major); overwritten.
    template <std::invocable<T> F>
    void apply(F f, int64_t n, int64_t s, int64_t d, T* out) {
        // Per-thread scratch: alpha_j(d), beta_j(d-1), Z_j(d*d), c_j(d), v_j(d)
        // Total per thread = d + (d-1) + d*d + d + d = d^2 + 4d - 1
        int64_t scratch_per_thread = d * d + 3 * d + std::max(d - 1, (int64_t)0);
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        T* scratch = new T[nthreads * scratch_per_thread];

#pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < s; ++j) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            T* base    = scratch + tid * scratch_per_thread;
            T* alpha_j = base;
            T* beta_j  = alpha_j + d;
            T* Z_j     = beta_j  + std::max(d - 1, (int64_t)0);
            T* c_j     = Z_j     + d * d;
            T* v_j     = c_j     + d;

            // Copy tridiagonal for column j (stev destroys alpha and beta in-place)
            std::copy(alpha + j * d,       alpha + j * d + d,             alpha_j);
            if (d > 1)
                std::copy(beta + j * (d-1), beta + j * (d-1) + (d-1), beta_j);

            // d×d tridiagonal eigendecomposition: T_j = Z_j * diag(θ) * Z_j^T
            // alpha_j → eigenvalues θ (ascending); Z_j → eigenvectors (column-major)
            lapack::stev(lapack::Job::Vec, d, alpha_j, beta_j, Z_j, d);

            // c_j[i] = f(θ_i) * S_j[0, i]
            // In column-major Z_j (d×d): entry (row=0, col=i) = Z_j[i*d + 0]
            for (int64_t i = 0; i < d; ++i)
                c_j[i] = f(alpha_j[i]) * Z_j[i * d + 0];

            // v_j = Z_j * c_j  (d×d matrix times d-vector)
            blas::gemv(Layout::ColMajor, Op::NoTrans, d, d,
                       (T)1.0, Z_j, d, c_j, 1, (T)0.0, v_j, 1);

            // out[:,j] = normb[j] * Q_j * v_j
            // Q_j is n×d with column stride n*s (strided view into K buffer)
            blas::gemv(Layout::ColMajor, Op::NoTrans, n, d,
                       normb[j], K + j * n, n * s, v_j, 1, (T)0.0, out + j * n, 1);
        }

        delete[] scratch;
    }

    // ------------------------------------------------------------------
    /// Combined run + apply: compute f(A)B in one call.
    ///
    /// @param[in]  A    SymmetricLinearOperator.
    /// @param[in]  B    n×s input matrix (column-major).
    /// @param[in]  n    Dimension of A.
    /// @param[in]  s    Number of right-hand sides.
    /// @param[in]  f    Scalar function T→T.
    /// @param[in]  d    Lanczos steps.
    /// @param[out] out  n×s output, overwritten with f(A)B approximation.
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        run(A, B, n, s, d);
        apply(f, n, s, d, out);
    }
};


} // end namespace RandLAPACK
