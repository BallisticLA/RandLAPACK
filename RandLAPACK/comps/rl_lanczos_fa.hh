#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <concepts>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RandLAPACK {


/// d-step block Lanczos for matrix function application f(A)B.
/// Approximates f(A)B column-wise via independent Krylov subspaces of dimension d.
/// See: T. Chen, "A Lanczos-FA algorithm for matrix function approximation" (2022).
///
/// @tparam T    Floating-point scalar type.
/// @tparam RNG  Random number generator type (unused here; kept for API uniformity).
template <typename T, typename RNG>
class LanczosFA {
public:
    /// Reorthogonalization control.
    ///  1 = full (project out all previous Krylov vectors after each step).
    ///  0 = none (vanilla Lanczos, per Persson's reference implementation).
    /// Lanczos-FA tolerates loss of orthogonality better than eigenvalue
    /// Lanczos (Paige-Greenbaum theory), so vanilla often works in practice.
    /// Full reorthogonalization is the safe default for a numerical library.
    int64_t reorth = 1;

    // Internal buffers — grown with new/delete[], never shrunk between calls.
    // Dimension key: n = operator dimension, s = number of RHS vectors (columns of B),
    //                d = number of Lanczos steps.
    //
    // K:     (d+1) × n × s — Krylov basis blocks.
    //   Layout: K[step * n*s + col * n + row] = row-th entry of step-th basis vector for column col.
    //   Storing steps as contiguous n×s slices keeps each batch matvec contiguous,
    //   while the per-column stride (n*s) lets apply() use strided gemv for reconstruction.
    // alpha: s × d         — tridiagonal diagonals, alpha[j*d + i] = α_{i,j}.
    // beta:  s × (d-1)     — tridiagonal subdiagonals, beta[j*(d-1) + i] = β_{i+1,j}.
    //   lapack::stevd expects the diagonal and subdiagonal as separate arrays,
    //   so alpha and beta are stored separately rather than interleaved.
    // normb: s             — column norms of B before normalization.
    T*      K      = nullptr; int64_t K_sz      = 0;
    T*      alpha  = nullptr; int64_t alpha_sz  = 0;
    T*      beta   = nullptr; int64_t beta_sz   = 0;
    T*      normb  = nullptr; int64_t normb_sz  = 0;

    ~LanczosFA() { delete[] K; delete[] alpha; delete[] beta; delete[] normb; }

    // ------------------------------------------------------------------
    /// Run the d-step block Lanczos recurrence on B.
    /// Fills K, alpha, beta, normb from B (n×s column-major).
    /// Calls A exactly d times, each application to an n×s matrix.
    ///
    /// @param[in]  A    SymmetricLinearOperator — matvec oracle.
    /// @param[in]  B    n×s input matrix (column-major); not modified.
    /// @param[in]  n    Dimension of A.
    /// @param[in]  s    Number of right-hand sides (Hutchinson samples).
    /// @param[in]  d    Number of Lanczos steps.
    template <linops::SymmetricLinearOperator SLO>
    void run_lanczos(SLO& A, const T* B, int64_t n, int64_t s, int64_t d) {
        // Grow buffers if needed
        util::resize(K,     K_sz,     (d + 1) * n * s);
        util::resize(alpha, alpha_sz, d * s);
        if (d > 1) util::resize(beta, beta_sz, (d - 1) * s);
        util::resize(normb, normb_sz, s);

        // Step 0: q_1 = column-normalize B; store in K[:,:,0]
        T* K0 = K;
        lapack::lacpy(lapack::MatrixType::General, n, s, B, n, K0, n);
#pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < s; ++j) {
            T nrm = blas::nrm2(n, K0 + j * n, 1);
            normb[j] = nrm;
            blas::scal(n, (T)1.0 / nrm, K0 + j * n, 1);
        }

        // Step 0 matvec: K[:,:,1] = A * K[:,:,0]
        T* K1 = K + n * s;
        A(Layout::ColMajor, s, (T)1.0, K0, n, (T)0.0, K1, n);

        // α[0, j] = q_1[:,j] · (A q_1)[:,j] — s independent inner products,
        // one tridiagonal diagonal entry per column.
#pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < s; ++j)
            alpha[j * d + 0] = blas::dot(n, K1 + j * n, 1, K0 + j * n, 1);

        // Main Lanczos loop: steps 1..d-1
        // At the start of iteration i:
        //   K[:,:,i]   = A*q_i - β_i*q_{i-1}  (partial three-term, β part done last iter)
        //   K[:,:,i+1] is free workspace
        // This iteration:
        //   (1) subtract α_i*q_i to complete three-term → K_{i+1} becomes unnormalized q_{i+1}
        //   (2) optional reorthogonalization
        //   (3) β_{i+1} = ||K_{i+1}||, (4) normalize → q_{i+1}
        //   (5) K_{i+2} = A*q_{i+1} - β_{i+1}*q_i  (start of next three-term)
        //   (6) α_{i+1} = q_{i+1}[:,j] · K_{i+2}[:,j]
        // All per-column loops below are independent across j and parallelized.
        for (int64_t i = 0; i < d - 1; ++i) {
            T* K_prev = K + i * n * s;           // K[:,:,i] = q_i (normalized)
            T* K_curr = K + (i + 1) * n * s;     // K[:,:,i+1] = partial (A*q_i - β_i*q_{i-1})
            T* K_new  = K + (i + 2) * n * s;     // K[:,:,i+2] = workspace for A*q_{i+1}

            // (1) Complete three-term: K_curr -= α_i * K_prev
            // Each column has a different scalar α_{i,j}, so GEMM would cost O(n·s²); axpy is optimal.
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < s; ++j)
                blas::axpy(n, -alpha[j * d + i], K_prev + j * n, 1, K_curr + j * n, 1);

            // (2) Optional full reorthogonalization: project K_curr[:,j] out of all q_0..q_i.
            // Outer loop over j is parallel (columns are independent); inner prev-loop is
            // sequential per column (each projection modifies K_curr[:,j] in place).
            int64_t reorth_steps = reorth ? (i + 1) : 0;
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < s; ++j) {
                for (int64_t prev = 0; prev < reorth_steps; ++prev) {
                    T* K_p = K + prev * n * s;
                    T coeff = blas::dot(n, K_curr + j * n, 1, K_p + j * n, 1);
                    blas::axpy(n, -coeff, K_p + j * n, 1, K_curr + j * n, 1);
                }
            }

            // (3) β_{i+1} = column norms, (4) normalize → q_{i+1}
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < s; ++j) {
                T nrm = blas::nrm2(n, K_curr + j * n, 1);
                beta[j * (d - 1) + i] = nrm;
                blas::scal(n, (T)1.0 / nrm, K_curr + j * n, 1);
            }

            // (5) K_new = A*q_{i+1} - β_{i+1}*q_i
            // Different β per column — same reasoning as (1), axpy is optimal.
            A(Layout::ColMajor, s, (T)1.0, K_curr, n, (T)0.0, K_new, n);
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < s; ++j)
                blas::axpy(n, -beta[j * (d - 1) + i], K_prev + j * n, 1, K_new + j * n, 1);

            // (6) α_{i+1} = q_{i+1}[:,j] · K_new[:,j]
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < s; ++j)
                alpha[j * d + (i + 1)] = blas::dot(n, K_new + j * n, 1, K_curr + j * n, 1);
        }
    }

    // ------------------------------------------------------------------
    /// Evaluate f(A)B from precomputed Krylov data (K, alpha, beta, normb).
    /// Per column j: eigendecompose T_j = S_j diag(θ_j) S_j^T via lapack::stev, then:
    ///   out[:,j] = normb[j] * Q_j * S_j * diag(f(θ_j)) * S_j[0,:]^T
    /// where Q_j is the n×d Lanczos basis stored in K and S_j[0,:] is the first row
    /// of the eigenvector matrix (Chen 2022, eq. 2.3).
    /// Per-column stev calls are independent — parallelized with OpenMP.
    ///
    /// @param[in]  f    Scalar callable T→T applied to tridiagonal eigenvalues.
    /// @param[in]  n    Dimension of A.
    /// @param[in]  s    Number of right-hand sides.
    /// @param[in]  d    Number of Lanczos steps (tridiagonal size).
    /// @param[out] out  n×s output matrix (column-major); overwritten.
    // F must be callable as T f(T x) — lambda, function pointer, or functor all work.
    // std::invocable<T> is a C++20 concept that enforces this at the call site,
    // giving a readable error instead of a cryptic substitution failure deep inside.
    template <std::invocable<T> F>
    void apply_f(F f, int64_t n, int64_t s, int64_t d, T* out) {
        // Per-thread workspace: alpha_j(d), beta_j(d-1), Z_j(d*d), c_j(d), v_j(d)
        // Total per thread = d + (d-1) + d*d + d + d = d^2 + 4d - 1
        int64_t workspace_per_thread = d * d + 3 * d + std::max(d - 1, (int64_t)0);
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        T* workspace = new T[nthreads * workspace_per_thread];

#pragma omp parallel for schedule(static)
        for (int64_t j = 0; j < s; ++j) {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            T* base    = workspace + tid * workspace_per_thread;
            T* alpha_j = base;
            T* beta_j  = alpha_j + d;
            T* Z_j     = beta_j  + std::max(d - 1, (int64_t)0);
            T* c_j     = Z_j     + d * d;
            T* v_j     = c_j     + d;

            // Copy per-column tridiagonal entries into per-thread workspace.
            // stevd overwrites its alpha/beta arrays in-place (they become eigenvalues
            // and workspace), so we must work on copies — otherwise the member alpha/beta
            // would be destroyed and a second apply_f call (with a different f) would
            // produce garbage without re-running run_lanczos.
            // alpha[j*d .. j*d+d-1] and beta[j*(d-1) .. j*(d-1)+(d-2)] are flat 1D
            // vectors of length d and d-1, not 2D matrices — blas::copy is correct here;
            // lacpy is for 2D matrices with distinct leading dimensions.
            blas::copy(d, alpha + j * d, 1, alpha_j, 1);
            if (d > 1)
                blas::copy(d - 1, beta + j * (d - 1), 1, beta_j, 1);

            // d×d tridiagonal eigendecomposition: T_j = Z_j * diag(θ) * Z_j^T
            // alpha_j → eigenvalues θ (ascending); Z_j → eigenvectors (column-major)
            lapack::stevd(lapack::Job::Vec, d, alpha_j, beta_j, Z_j, d);

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

        delete[] workspace;
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
        run_lanczos(A, B, n, s, d);
        apply_f(f, n, s, d, out);
    }
};


} // end namespace RandLAPACK
