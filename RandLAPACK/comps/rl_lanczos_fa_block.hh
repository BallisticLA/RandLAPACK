#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <chrono>
#include <cstdint>
#include <concepts>
#include <algorithm>
#include <vector>
#include <cstring>

namespace RandLAPACK {


/// d-step block Lanczos for matrix function application f(A)B.
/// Builds a single joint block Krylov subspace using BLAS-3 throughout,
/// replacing the s independent scalar Lanczos sequences in LanczosFA.
///
/// Algorithm reference: T. Chen, "A handbook for matrix-function-based Krylov methods",
///   arXiv:2410.11090 (2024). Algorithm 9.2 (block Lanczos recurrence),
///   Definition 9.6 (block Lanczos-FA).
///
/// At each recurrence step all updates (alpha, beta, Z) are BLAS-3 (GEMM + GEQRF).
/// apply_f calls a single syevd on the (d*s)×(d*s) block tridiagonal instead of
/// s separate stevd calls of size d.
///
/// Block Lanczos-FA formula:  out ≈ Q_basis * f(T_k) * E₁ * R₀
///   Q_basis = [Q₀|...|Q_{d-1}] (n×d*s), T_k = d*s×d*s block tridiagonal,
///   E₁ = first d*s×s columns of identity, B = Q₀*R₀ (initial QR).
///
/// Known limitation (v1): deflation is not implemented.  When the block Krylov
/// space fills before d steps (B_step develops near-zero singular values), accuracy
/// degrades but the algorithm does not crash.  For problems where d*s approaches the
/// effective rank of A, use a smaller d or the scalar LanczosFA.
///
/// @tparam T    Floating-point scalar type.
template <typename T>
class BlockLanczosFA {
public:
    /// Reorthogonalization control.
    ///  1 = full (project each new block Z out of all previous Krylov blocks).
    ///  0 = none.
    int64_t reorth = 1;

    // Internal buffers — grown with new/delete[], never shrunk between calls.
    // Dimension key: n = operator dimension, s = block size, d = Lanczos steps.
    //
    // K_big:     (d+1)*n*s  — block Krylov basis + matvec scratch.
    //   Layout: K_big[step*n*s .. (step+1)*n*s-1] = Q_step (n×s col-major, ld=n).
    //   First d blocks form Q_basis (n×d*s col-major, ld=n) used by apply_f.
    //   Block d is scratch for the current-step matvec output.
    // R0_buf:    s*s         — upper triangular factor from initial QR of B.
    // tau_buf:   n           — Householder scalars; reused at each geqrf/orgqr call.
    // A_blk:     d*s*s       — block alphas (s×s symmetric), A_blk[step*s*s] = A_step.
    // B_blk:     d*s*s       — block betas  (s×s upper triangular), B_blk[step*s*s] = B_step.
    //   Only d-1 entries are populated (no beta after the last step);
    //   size d is allocated to keep indexing uniform.
    // workspace: apply_f scratch — T_dense (d*s×d*s) + eig_vals (d*s) + G (d*s×s) + C1 (d*s×s).
    // proj_buf:  s*s — reorthogonalization scratch (Q_p^T * Y projection); reused across steps.
    T* K_big     = nullptr; int64_t K_big_sz     = 0;
    T* R0_buf    = nullptr; int64_t R0_sz        = 0;
    T* tau_buf   = nullptr; int64_t tau_buf_sz   = 0;
    T* A_blk     = nullptr; int64_t A_blk_sz     = 0;
    T* B_blk     = nullptr; int64_t B_blk_sz     = 0;
    T* workspace = nullptr; int64_t workspace_sz = 0;
    T* proj_buf  = nullptr; int64_t proj_buf_sz  = 0;

    bool timing = false;
    std::vector<long> times;
    long _t_matvec_us = 0;

    BlockLanczosFA()                                 = default;
    BlockLanczosFA(const BlockLanczosFA&)            = delete;
    BlockLanczosFA& operator=(const BlockLanczosFA&) = delete;

    ~BlockLanczosFA() {
        delete[] K_big; delete[] R0_buf; delete[] tau_buf;
        delete[] A_blk; delete[] B_blk; delete[] workspace; delete[] proj_buf;
    }

    // ------------------------------------------------------------------
    /// Run the d-step block Lanczos recurrence on B (n×s col-major).
    /// Fills K_big, R0_buf, A_blk, B_blk.
    /// Calls A exactly d times, each applied to an n×s block.
    template <linops::SymmetricLinearOperator SLO>
    void run_lanczos(SLO& A, const T* B, int64_t n, int64_t s, int64_t d) {
        using namespace std::chrono;
        steady_clock::time_point _mv_t0, _mv_t1;
        _t_matvec_us = 0;

        util::resize(K_big,   K_big_sz,   (d + 1) * n * s);
        util::resize(R0_buf,  R0_sz,      s * s);
        util::resize(tau_buf, tau_buf_sz, n);
        util::resize(A_blk,   A_blk_sz,   d * s * s);
        util::resize(B_blk,   B_blk_sz,   d * s * s);
        if (reorth) util::resize(proj_buf, proj_buf_sz, s * s);

        // Initial QR: B = Q0 * R0.  Q0 overwrites K_big[0..n*s-1].
        T* Q0 = K_big;
        lapack::lacpy(lapack::MatrixType::General, n, s, B, n, Q0, n);
        lapack::geqrf(n, s, Q0, n, tau_buf);
        lapack::laset(lapack::MatrixType::General, s, s, (T)0, (T)0, R0_buf, s);
        lapack::lacpy(lapack::MatrixType::Upper, s, s, Q0, n, R0_buf, s);
        lapack::orgqr(n, s, s, Q0, n, tau_buf);

        for (int64_t step = 0; step < d; ++step) {
            T* Q_step = K_big + step       * n * s;
            T* Q_prev = (step > 0) ? K_big + (step - 1) * n * s : nullptr;
            T* Y      = K_big + (step + 1) * n * s;   // matvec output, then Z in-place
            T* A_step = A_blk + step * s * s;
            T* B_prev = (step > 0) ? B_blk + (step - 1) * s * s : nullptr;

            // Y = A * Q_step
            if (this->timing) _mv_t0 = steady_clock::now();
            A(Layout::ColMajor, s, (T)1.0, Q_step, n, (T)0.0, Y, n);
            if (this->timing) { _mv_t1 = steady_clock::now(); _t_matvec_us += duration_cast<microseconds>(_mv_t1 - _mv_t0).count(); }

            // Y -= Q_{step-1} * B_{step-1}^T
            if (step > 0)
                blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                           n, s, s, (T)-1.0, Q_prev, n, B_prev, s, (T)1.0, Y, n);

            // A_step = Q_step^T * Y  (block alpha, s×s)
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                       s, s, n, (T)1.0, Q_step, n, Y, n, (T)0.0, A_step, s);
            for (int64_t j = 0; j < s; ++j) {
                for (int64_t i = j + 1; i < s; ++i) {
                    T avg = (T)0.5 * (A_step[j * s + i] + A_step[i * s + j]);
                    A_step[j * s + i] = avg;
                    A_step[i * s + j] = avg;
                }
            }

            // Y -= Q_step * A_step  (Z = Y - Q_step*A_step, in-place)
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, s, s, (T)-1.0, Q_step, n, A_step, s, (T)1.0, Y, n);

            // Optional full reorthogonalization: Z -= Q_p * (Q_p^T * Z) for each prev block
            if (reorth) {
                for (int64_t prev = 0; prev <= step; ++prev) {
                    T* Q_p = K_big + prev * n * s;
                    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                               s, s, n, (T)1.0, Q_p, n, Y, n, (T)0.0, proj_buf, s);
                    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                               n, s, s, (T)-1.0, Q_p, n, proj_buf, s, (T)1.0, Y, n);
                }
            }

            // QR of Z → Q_{step+1} (in Y) and B_step (upper R factor).
            // Skipped at the last step: Q_d is never needed by apply_f.
            if (step < d - 1) {
                T* B_step = B_blk + step * s * s;
                lapack::geqrf(n, s, Y, n, tau_buf);
                lapack::laset(lapack::MatrixType::General, s, s, (T)0, (T)0, B_step, s);
                lapack::lacpy(lapack::MatrixType::Upper, s, s, Y, n, B_step, s);
                lapack::orgqr(n, s, s, Y, n, tau_buf);
            }
        }
    }

    // ------------------------------------------------------------------
    /// Evaluate f(A)B from precomputed Krylov data (K_big, R0_buf, A_blk, B_blk).
    ///
    /// Computation:
    ///  1. Assemble T_dense (d*s × d*s block tridiagonal) from A_blk and B_blk.
    ///  2. syevd: T_dense → eigenvectors V (in-place), eigenvalues λ.
    ///  3. W (s×m): W[i,j] = f(λⱼ)*V[i,j] for i=0..s-1 (first s rows of V, col-scaled).
    ///  4. C1 (d*s × s) = V * W^T  — this equals f(T_k) * E₁.
    ///  5. C1 *= R₀  (TRMM: right-multiply by upper-triangular R₀).
    ///  6. out (n × s) = Q_basis * C1  (GEMM).
    template <std::invocable<T> F>
    void apply_f(F f, int64_t n, int64_t s, int64_t d, T* out) {
        int64_t m = d * s;
        util::resize(workspace, workspace_sz, m * m + m + 2 * m * s);

        T* T_dense = workspace;
        T* eig_vals = T_dense + m * m;
        T* G        = eig_vals + m;
        T* C1       = G + m * s;

        // 1. Assemble T_dense.
        std::memset(T_dense, 0, m * m * sizeof(T));
        for (int64_t step = 0; step < d; ++step) {
            int64_t b0 = step * s;
            // Diagonal block
            lapack::lacpy(lapack::MatrixType::General, s, s,
                          A_blk + step * s * s, s, T_dense + b0 * m + b0, m);
            // Off-diagonal blocks (B_step in lower, B_step^T in upper)
            if (step < d - 1) {
                T* B_step = B_blk + step * s * s;
                int64_t b1 = (step + 1) * s;
                // Lower off-diagonal: T(b1:b1+s, b0:b0+s) = B_step
                lapack::lacpy(lapack::MatrixType::General, s, s,
                              B_step, s, T_dense + b0 * m + b1, m);
                // Upper off-diagonal: T(b0:b0+s, b1:b1+s) = B_step^T
                for (int64_t j = 0; j < s; ++j)
                    for (int64_t i = 0; i < s; ++i)
                        T_dense[(b1 + j) * m + (b0 + i)] = B_step[i * s + j];
            }
        }
        // Symmetrize to eliminate any floating-point asymmetry
        for (int64_t j = 0; j < m; ++j)
            for (int64_t i = j + 1; i < m; ++i) {
                T avg = (T)0.5 * (T_dense[j * m + i] + T_dense[i * m + j]);
                T_dense[j * m + i] = T_dense[i * m + j] = avg;
            }

        // 2. Eigendecomposition: T_dense → V (eigenvectors overwrite T_dense), eig_vals → λ.
        lapack::syevd(lapack::Job::Vec, blas::Uplo::Lower, m, T_dense, m, eig_vals);

        // 3. W (s×m col-major, ld=s; stored in the G buffer): W[i,j] = f(λⱼ)*V[i,j]
        //    for i=0..s-1, j=0..m-1.  Each column j of W is the first s elements of
        //    column j of V, scaled by f(λⱼ).
        //
        //    Derivation: f(T_k)*E₁ = V*diag(f(λ))*V^T*E₁.
        //    V^T*E₁ = first s columns of V^T = {row i of V} for i=0..s-1 stacked as cols.
        //    diag(f(λ))*(V^T*E₁): scale row j → f(λⱼ)*(row j of V^T*E₁) = f(λⱼ)*V[:,j][0:s].
        //    That intermediate is W^T (m×s), so f(T_k)*E₁ = V * W^T.
        T* W = G;   // reuse G buffer; W is s×m col-major (ld=s)
        for (int64_t j = 0; j < m; ++j) {
            T fev = f(std::max(eig_vals[j], (T)0));
            const T* V_col = T_dense + j * m;   // col j of V (contiguous, length m)
            T*       W_col = W       + j * s;   // col j of W (contiguous, length s)
            for (int64_t i = 0; i < s; ++i)
                W_col[i] = fev * V_col[i];      // first s rows of V[:,j], scaled
        }

        // 4. C1 (m × s) = V * W^T  — equals f(T_k) * E₁.
        //    GEMM(NoTrans, Trans, m, s, m): C = V(m×m) * W^T  where W is s×m (ld=s).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
                   m, s, m, (T)1.0, T_dense, m, W, s, (T)0.0, C1, m);

        // 5. C1 *= R₀  (TRMM: C1 = C1 * R₀, right upper triangular).
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   m, s, (T)1.0, R0_buf, s, C1, m);

        // 6. out (n × s) = Q_basis (n × m) * C1 (m × s).
        //    Q_basis = K_big[0..m*n-1] (n×m col-major, ld=n).
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   n, s, m, (T)1.0, K_big, n, C1, m, (T)0.0, out, n);
    }

    // ------------------------------------------------------------------
    /// Combined run + apply.
    ///
    /// Drop-in replacement for LanczosFA::call — same signature, slots into
    /// FunNystromPP and ResidualOp as the LanczosFA_t template parameter.
    template <linops::SymmetricLinearOperator SLO, std::invocable<T> F>
    void call(SLO& A, const T* B, int64_t n, int64_t s, F f, int64_t d, T* out) {
        using namespace std::chrono;
        _t_matvec_us = 0;
        steady_clock::time_point t_total_start, t_lanczos_end, t_end;
        if (this->timing) t_total_start = steady_clock::now();
        run_lanczos(A, B, n, s, d);
        if (this->timing) t_lanczos_end = steady_clock::now();
        apply_f(f, n, s, d, out);
        if (this->timing) {
            t_end = steady_clock::now();
            long total_us   = duration_cast<microseconds>(t_end         - t_total_start).count();
            long lanczos_us = duration_cast<microseconds>(t_lanczos_end - t_total_start).count();
            long apply_f_us = duration_cast<microseconds>(t_end         - t_lanczos_end).count();
            long rest_us    = total_us - lanczos_us - apply_f_us;
            this->times = {_t_matvec_us, lanczos_us, apply_f_us, rest_us, total_us};
        }
    }
};


} // end namespace RandLAPACK
