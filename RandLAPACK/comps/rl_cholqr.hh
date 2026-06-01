#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_bqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <chrono>
#include <limits>
#include <algorithm>

namespace RandLAPACK {

/// Method for inverting the upper-triangular preconditioner P inside pcholqr_primitive.
///
/// TRSM_IDENTITY  Solve P * R_pre = I via column-by-column TRSM(Side::Left).
///                Backward-stable for the subsequent product A * R_pre. O(n^3/2).
/// TRTRI          Direct triangular inverse via LAPACK trtri. Same cost; less stable
///                than TRSM_IDENTITY when P is ill-conditioned in practice.
/// GEQP3          R_pre = P_perm * R_buf^{-1} * Q_buf^T via LAPACK geqp3 on P followed
///                by ungqr + TRSM + scatter. Stable for ill-conditioned P. O(11 n^3 / 6).
/// BQRRP          Same output format as GEQP3 but via RandLAPACK::BQRRP (blocked +
///                sketched pivoting). Faster for large n; requires RNG state.
enum class PCholQRPrecondMethod {
    TRSM_IDENTITY,
    TRTRI,
    GEQP3,
    BQRRP
};


// ============================================================================
// Layer 0 — blocked_preconditioned_gram
// ============================================================================
//
// Compute the Gram matrix G = R_pre^T * (A^T A) * R_pre via blocked linop calls.
// When R_pre is nullptr, computes the unpreconditioned Gram G = A^T A (M = I).
//
// Peak memory: O(n^2 + (m + n) * b_eff)
//   - A_temp: m × b_eff  scratch for A * R_pre[:, j:j+b]
//   - Z_buf:  n × b_eff  scratch for A^T * A_temp (used only when R_pre != nullptr;
//                        when R_pre == nullptr we write A^T * A_temp directly into G)
//   - When R_pre == nullptr an n × b_eff identity scratch is allocated internally.
//
// Timing accumulators are output parameters; ignored when timing == false.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
void blocked_preconditioned_gram(
    GLO& A,
    const T* R_pre,
    T* G,
    int64_t m, int64_t n, int64_t b_eff,
    T* A_temp,
    T* Z_buf,
    bool skip_left_factor,
    long& fwd_us, long& adj_us, long& gemm_us,
    bool timing)
{
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    steady_clock::time_point t0, t1;
    long fwd_accum = 0, adj_accum = 0, gemm_accum = 0;

    T* I_block = nullptr;
    if (R_pre == nullptr) {
        I_block = new T[n * b_eff]();
    }

    for (int64_t j = 0; j < n; j += b_eff) {
        int64_t b_j = std::min(b_eff, n - j);

        const T* B_in;
        if (R_pre) {
            B_in = R_pre + j * n;
        } else {
            for (int64_t c = 0; c < b_j; ++c)
                I_block[(j + c) + c * n] = (T)1.0;
            B_in = I_block;
        }

        // A_temp = A * B_in   (m × b_j)
        if (timing) t0 = steady_clock::now();
        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          m, b_j, n, (T)1.0, B_in, n, (T)0.0, A_temp, m);
        if (timing) { t1 = steady_clock::now(); fwd_accum += duration_cast<microseconds>(t1 - t0).count(); }

        if (R_pre && !skip_left_factor) {
            // Z_buf = A^T * A_temp   (n × b_j)
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }

            // G[:, j:j+b_j] = R_pre^T * Z_buf   (n × b_j)
            if (timing) t0 = steady_clock::now();
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                       n, b_j, n, (T)1.0, R_pre, n, Z_buf, n, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); gemm_accum += duration_cast<microseconds>(t1 - t0).count(); }
        } else if (R_pre) {
            // skip_left_factor: write Z directly into G[:, j:j+b_j]. Caller applies the left
            // factor (typically via TRSM with the original P) once the loop completes.
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }
        } else {
            // G[:, j:j+b_j] = A^T * A_temp   (direct, M = I case)
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }

            // Restore I_block columns to zero for next iter
            for (int64_t c = 0; c < b_j; ++c)
                I_block[(j + c) + c * n] = (T)0.0;
        }
    }

    if (I_block) delete[] I_block;

    if (timing) {
        fwd_us = fwd_accum;
        adj_us = adj_accum;
        gemm_us = gemm_accum;
    }
}


// ============================================================================
// Layer 1a — cholqr_primitive: Algorithm 1 / sCholQR3 iter-1 (shifted)
// ============================================================================
//
// Computes R upper-triangular such that A * R^{-1} has orthonormal columns:
//     G = A^T A   (+ shift * I  if shift_factor > 0)
//     G = R^T R  (Cholesky)
//
// shift_factor: caller-supplied multiplier on trace(G) = ||A||_F^2. The sCholQR3
//               iter-1 spec wants s = 11 * eps * n * ||A||_F^2 so the caller
//               passes shift_factor = 11 * eps * n. Pass 0 for unshifted CholQR.
//
// Workspaces (caller-owned):
//   R       — n × n output buffer (lower triangle returned zeroed)
//   G       — n × n Gram scratch
//   A_temp  — m × b_eff
//
// Returns potrf info (0 on success; >0 on Cholesky breakdown).
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int cholqr_primitive(
    GLO& A,
    T* R, int64_t ldr,
    T shift_factor,
    int64_t block_size,
    T* G,
    T* A_temp,
    long& fwd_us, long& adj_us, long& chol_us,
    bool timing)
{
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;

    long gemm_unused = 0;
    blocked_preconditioned_gram<T, GLO>(A, (const T*)nullptr, G, m, n, b_eff,
                                         A_temp, (T*)nullptr,
                                         /*skip_left_factor=*/false,
                                         fwd_us, adj_us, gemm_unused, timing);

    steady_clock::time_point t0, t1;
    if (timing) t0 = steady_clock::now();

    if (shift_factor > T(0)) {
        T trace = 0;
        for (int64_t i = 0; i < n; ++i) trace += G[i * (n + 1)];
        T shift = shift_factor * trace;
        for (int64_t i = 0; i < n; ++i) G[i * (n + 1)] += shift;
    }

    if (n > 1)
        lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), &G[1], n);
    int info = lapack::potrf(Uplo::Upper, n, G, n);
    if (info) return info;

    lapack::lacpy(MatrixType::Upper, n, n, G, n, R, ldr);
    if (n > 1)
        lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R + 1, ldr);

    if (timing) {
        t1 = steady_clock::now();
        chol_us = duration_cast<microseconds>(t1 - t0).count();
    }
    return 0;
}


// ============================================================================
// Layer 1b — pcholqr_primitive: Algorithm 2 (preconditioned Q-less Cholesky QR)
// ============================================================================
//
// Given upper-triangular P (n × n), computes R upper-triangular such that
// A * R^{-1} has orthonormal columns. Mirrors the collaborator's pseudocode:
//   1. R_pre = invert(P) via 'method'   (TRSM_IDENTITY / TRTRI / GEQP3 / BQRRP)
//   2. G = R_pre^T * A^T * A * R_pre   (blocked_preconditioned_gram)
//   3. G = (R^chol)^T R^chol           (Cholesky)
//   4. R = R^chol * P                   (in-place trmm into R)
//
// Workspaces (caller-owned):
//   R_pre   — n × n  (output of step 1; input to step 2)
//   G       — n × n  (Gram scratch, then R^chol after potrf)
//   A_temp  — m × b_eff
//   Z_buf   — n × b_eff
//
// state: RNG state for the BQRRP method only. Pass &state for BQRRP; nullptr is OK
//        for TRSM_IDENTITY / TRTRI / GEQP3.
//
// bqrrp_block_ratio: BQRRP block-size knob. Pass 1.0 to use the CQRRPT-style
//                    adaptive heuristic (1.0 for n<=2000, 0.5 for n<=8000, 1/32 else).
//
// Returns 0 on success; nonzero on diag-zero check failure / Cholesky breakdown.
template <typename T, RandLAPACK::linops::LinearOperator GLO,
          typename RNG = RandBLAS::DefaultRNG>
int pcholqr_primitive(
    GLO& A,
    const T* P,
    T* R, int64_t ldr,
    PCholQRPrecondMethod method,
    int64_t block_size,
    T bqrrp_block_ratio,
    T* R_pre,
    T* G,
    T* A_temp,
    T* Z_buf,
    RandBLAS::RNGState<RNG>* state,
    long& precond_inv_us,
    long& fwd_us, long& adj_us, long& gemm_us, long& chol_us, long& update_us,
    bool timing)
{
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;

    steady_clock::time_point t0, t1;
    if (timing) t0 = steady_clock::now();

    // ---- Step 1: R_pre = invert(P) ----
    switch (method) {
        case PCholQRPrecondMethod::TRSM_IDENTITY: {
            if (!RandLAPACK::util::diag_is_nonzero(n, P, n)) return 1;
            RandLAPACK::util::eye(n, n, R_pre);
            blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper,
                       Op::NoTrans, Diag::NonUnit,
                       n, n, T(1), P, n, R_pre, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R_pre + 1, n);
            break;
        }
        case PCholQRPrecondMethod::TRTRI: {
            if (!RandLAPACK::util::diag_is_nonzero(n, P, n)) return 1;
            lapack::lacpy(MatrixType::Upper, n, n, P, n, R_pre, n);
            if (n > 1)
                lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R_pre + 1, n);
            int trtri_info = lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_pre, n);
            if (trtri_info) return 1;
            break;
        }
        case PCholQRPrecondMethod::GEQP3:
        case PCholQRPrecondMethod::BQRRP: {
            T* P_copy = new T[n * n]();
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i <= j; ++i)
                    P_copy[i + j * n] = P[i + j * n];
            if (!RandLAPACK::util::diag_is_nonzero(n, P_copy, n)) { delete[] P_copy; return 1; }

            int64_t* jpiv = new int64_t[n]();
            T* tau_qr = new T[n];

            if (method == PCholQRPrecondMethod::GEQP3) {
                lapack::geqp3(n, n, P_copy, n, jpiv, tau_qr);
            } else {
                if (state == nullptr) {
                    delete[] P_copy; delete[] jpiv; delete[] tau_qr;
                    return 1;
                }
                T ratio = bqrrp_block_ratio;
                if (ratio == T(1.0)) {
                    if (n <= 2000)      ratio = T(1.0);
                    else if (n <= 8000) ratio = T(0.5);
                    else                ratio = T(1.0) / T(32);
                }
                int64_t bqrrp_b = std::max(int64_t(1), (int64_t)(n * ratio));
                RandLAPACK::BQRRP<T, RNG> bqrrp(false, bqrrp_b);
                bqrrp.call(n, n, P_copy, n, T(1), tau_qr, jpiv, *state);
            }

            T* R_buf = new T[n * n]();
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i <= j; ++i)
                    R_buf[i + j * n] = P_copy[i + j * n];
            lapack::ungqr(n, n, n, P_copy, n, tau_qr);

            T* W = new T[n * n];
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < n; ++j)
                    W[i + j * n] = P_copy[j + i * n];
            blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper,
                       Op::NoTrans, Diag::NonUnit,
                       n, n, T(1), R_buf, n, W, n);

            std::fill(R_pre, R_pre + n * n, T(0));
            for (int64_t k = 0; k < n; ++k)
                for (int64_t j = 0; j < n; ++j)
                    R_pre[(jpiv[k] - 1) + j * n] = W[k + j * n];

            delete[] P_copy;
            delete[] R_buf;
            delete[] W;
            delete[] jpiv;
            delete[] tau_qr;
            break;
        }
    }

    if (timing) {
        t1 = steady_clock::now();
        precond_inv_us = duration_cast<microseconds>(t1 - t0).count();
    }

    // ---- Step 2: G = R_pre^T * A^T * A * R_pre ----
    //
    // Dispatch by precond method on how to apply the *left* R_pre^T factor:
    //
    //   TRSM_IDENTITY / TRTRI : per-block step writes A^T A R_pre directly into G,
    //       then a single TRSM(P^T, G) applies P^{-T} = R_pre^T at the end.
    //       Cheaper (O(n^3/2)) and equally stable since P is preserved.
    //   GEQP3 / BQRRP : per-block GEMM with the explicit R_pre^T preserves the
    //       QRCP stability advantage (R_pre is computed accurately by pivoting,
    //       but TRSM(P^T, ...) would re-amplify the kappa(P) error we just dodged).
    bool use_trsm_at_end = (method == PCholQRPrecondMethod::TRSM_IDENTITY
                          || method == PCholQRPrecondMethod::TRTRI);

    blocked_preconditioned_gram<T, GLO>(A, R_pre, G, m, n, b_eff,
                                         A_temp, Z_buf,
                                         /*skip_left_factor=*/use_trsm_at_end,
                                         fwd_us, adj_us, gemm_us, timing);

    if (use_trsm_at_end) {
        // G := P^{-T} * G  (= R_pre^T * A^T * A * R_pre, since R_pre = P^{-1})
        if (timing) t0 = steady_clock::now();
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans,
                   Diag::NonUnit, n, n, T(1), P, n, G, n);
        if (timing) {
            t1 = steady_clock::now();
            gemm_us += duration_cast<microseconds>(t1 - t0).count();
        }
    }

    // ---- Step 3: G = (R^chol)^T R^chol ----
    if (timing) t0 = steady_clock::now();
    if (n > 1)
        lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), &G[1], n);
    int info = lapack::potrf(Uplo::Upper, n, G, n);
    if (info) return info;
    if (timing) {
        t1 = steady_clock::now();
        chol_us = duration_cast<microseconds>(t1 - t0).count();
    }

    // ---- Step 4: R = R^chol * P  (in-place trmm; R starts as P) ----
    if (timing) t0 = steady_clock::now();
    lapack::lacpy(MatrixType::Upper, n, n, P, n, R, ldr);
    if (n > 1)
        lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R + 1, ldr);
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper,
               Op::NoTrans, Diag::NonUnit,
               n, n, T(1), G, n, R, ldr);
    if (timing) {
        t1 = steady_clock::now();
        update_us = duration_cast<microseconds>(t1 - t0).count();
    }

    return 0;
}

} // namespace RandLAPACK
