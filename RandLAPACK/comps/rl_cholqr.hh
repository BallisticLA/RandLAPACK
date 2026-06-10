#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_bqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <limits>
#include <algorithm>

namespace RandLAPACK {

/// Below enum lists the methods cholqr_primitive can use to invert the
/// upper-triangular preconditioner P (forming R_pre = P^{-1}) in the
/// preconditioned path. They trade speed for stability when P is ill-conditioned:
///
/// TRSM_IDENTITY  Solve P * R_pre = I via TRSM(Side::Left). Backward-stable for the
///                subsequent product A * R_pre. O(n^3/2). Default.
/// TRTRI          Direct triangular inverse via LAPACK trtri. Same cost; less stable
///                than TRSM_IDENTITY when P is ill-conditioned in practice.
/// GEQP3          R_pre via column-pivoted QR of P (geqp3), giving P^{-1} = Pi R^{-1} Q^T.
///                Stable for ill-conditioned P. O(11 n^3 / 6).
/// BQRRP          Same output as GEQP3 but pivots via RandLAPACK::BQRRP (blocked +
///                sketched). Faster for large n; requires an RNG state.
enum class PCholQRPrecondMethod {
    TRSM_IDENTITY,
    TRTRI,
    GEQP3,
    BQRRP
};


// ============================================================================
// blocked_preconditioned_gram
// ============================================================================
//
// Forms the (optionally preconditioned) Gram matrix that cholqr_primitive then
// Cholesky-factorizes. The operator A is matrix-free (only A * B and A^T * B are
// available), so the Gram is built one column block at a time:
//
//   R_pre != nullptr :  G = R_pre^T (A^T A) R_pre   (preconditioned Gram)
//   R_pre == nullptr :  G = A^T A                   (plain Gram)
//
// This is the only place A is touched, so it dominates the cost (2 linop applies
// per block); keeping it blocked bounds the scratch to O(n^2 + (m+n) b_eff)
// instead of materializing A (m*n).
//
//   A_temp  m x b_eff : holds A * B_in for the current block.
//   Z_buf   n x b_eff : holds A^T * A_temp (used only on the R_pre != nullptr,
//                       non-skip path; otherwise A^T * A_temp goes straight to G).
//   When R_pre == nullptr, an n x n identity is allocated internally so each block
//   B_in is a column block of I (i.e. we apply A to the identity to read its columns).
//
// skip_left_factor (R_pre != nullptr only): write A^T A R_pre into G and let the
// caller apply the left R_pre^T factor afterwards (a single TRSM with P) instead
// of a per-block GEMM. Timing accumulators are outputs; ignored when timing==false.
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

    // Unpreconditioned path reads A's columns by applying A to column blocks of
    // the identity. We keep a single n x b_eff scratch block (rather than a full
    // n x n identity via util::eye, which would cost n^2 memory at scale) and set
    // its b_j shifted-diagonal ones each iteration, clearing them again after use.
    T* I_block = nullptr;
    if (R_pre == nullptr) I_block = new T[n * b_eff]();

    for (int64_t j = 0; j < n; j += b_eff) {
        int64_t b_j = std::min(b_eff, n - j);

        // B_in is the j-th column block of R_pre (preconditioned) or of I_n.
        const T* B_in;
        if (R_pre != nullptr) {
            B_in = R_pre + j * n;
        } else {
            for (int64_t c = 0; c < b_j; ++c) I_block[(j + c) + c * n] = (T)1.0;
            B_in = I_block;
        }

        // A_temp = A * B_in   (m x b_j)
        if (timing) t0 = steady_clock::now();
        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          m, b_j, n, (T)1.0, B_in, n, (T)0.0, A_temp, m);
        if (timing) { t1 = steady_clock::now(); fwd_accum += duration_cast<microseconds>(t1 - t0).count(); }

        if (R_pre && !skip_left_factor) {
            // Z_buf = A^T * A_temp ; then G[:, blk] = R_pre^T * Z_buf.
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }

            if (timing) t0 = steady_clock::now();
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
                       n, b_j, n, (T)1.0, R_pre, n, Z_buf, n, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); gemm_accum += duration_cast<microseconds>(t1 - t0).count(); }
        } else {
            // skip_left_factor (preconditioned) and the unpreconditioned path both
            // write A^T * A_temp straight into G[:, blk]; any left factor is applied
            // once after the loop by the caller.
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }

            // Clear this block's identity ones before the next iteration.
            if (R_pre == nullptr)
                for (int64_t c = 0; c < b_j; ++c) I_block[(j + c) + c * n] = (T)0.0;
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
// cholqr_primitive — Q-less (preconditioned) Cholesky QR
// ============================================================================
//
// Computes upper-triangular R such that A * R^{-1} has orthonormal columns. A
// single primitive covers both the plain and preconditioned variants via the
// optional preconditioner P:
//
//   P == nullptr (unpreconditioned CholQR):
//       G = A^T A;  G = R^T R (Cholesky);  output R.
//   P != nullptr (preconditioned, P upper-triangular):
//       R_pre = invert(P) via 'method';  G = R_pre^T A^T A R_pre;
//       G = (R^chol)^T R^chol (Cholesky);  output R = R^chol * P.
//
// CholQR2 / sCholQR3 chain this: pass the previous iterate's R as P so the
// returned R is the accumulated factor R_k ... R_1.
//
// Adaptive shift: before Cholesky a diagonal shift s = shift_factor * trace(G) is
// added (s = 0 when shift_factor = 0). On a non-PD pivot the shift is grown by
// shift_growth and potrf retried, up to max_retries times; max_retries < 0 means
// unbounded (retry until PD — geometric growth guarantees termination once the
// shift reaches trace(G), where the Gram is diagonally dominant). The Gram is
// computed once and snapshotted, so retries are O(n^2), not O(m n^2).
//
// Caller-owned scratch: R_pre (n x n, preconditioned only), G (n x n), A_temp
// (m x b_eff), Z_buf (n x b_eff, preconditioned non-skip only). state is the RNG
// state for the BQRRP method (nullptr otherwise). Timing args are outputs.
//
// Returns 0 on success; nonzero on a diag-zero preconditioner or a Cholesky
// breakdown that survived all retries.
template <typename T, RandLAPACK::linops::LinearOperator GLO,
          typename RNG = RandBLAS::DefaultRNG>
int cholqr_primitive(
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
    bool timing,
    T shift_factor = T(0),
    int max_retries = 0,
    T shift_growth = T(10))
{
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    const bool preconditioned = (P != nullptr);

    steady_clock::time_point t0, t1;
    if (timing) t0 = steady_clock::now();

    // ---- Step 1: R_pre = invert(P) (preconditioned only) ----
    if (preconditioned) {
        switch (method) {
            case PCholQRPrecondMethod::TRSM_IDENTITY: {
                if (!RandLAPACK::util::diag_is_nonzero(n, P, n)) {
                    std::fprintf(stderr, "[cholqr_primitive] FAIL: TRSM_IDENTITY diag_is_nonzero(P) failed (P has ~0 diagonal entry)\n");
                    return 1;
                }
                RandLAPACK::util::eye(n, n, R_pre);
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper,
                           Op::NoTrans, Diag::NonUnit,
                           n, n, T(1), P, n, R_pre, n);
                if (n > 1)
                    lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R_pre + 1, n);
                break;
            }
            case PCholQRPrecondMethod::TRTRI: {
                if (!RandLAPACK::util::diag_is_nonzero(n, P, n)) {
                    std::fprintf(stderr, "[cholqr_primitive] FAIL: TRTRI diag_is_nonzero(P) failed (P has ~0 diagonal entry)\n");
                    return 1;
                }
                lapack::lacpy(MatrixType::Upper, n, n, P, n, R_pre, n);
                if (n > 1)
                    lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R_pre + 1, n);
                int trtri_info = lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_pre, n);
                if (trtri_info) {
                    std::fprintf(stderr, "[cholqr_primitive] FAIL: lapack::trtri returned info=%d\n", trtri_info);
                    return 1;
                }
                break;
            }
            case PCholQRPrecondMethod::GEQP3:
            case PCholQRPrecondMethod::BQRRP: {
                // Invert P stably via column-pivoted QR. Column pivoting gives
                //     P Pi = Q R_tri   (Pi the pivot permutation),
                // so  P^{-1} = Pi R_tri^{-1} Q^T. We build that below from the QRCP
                // outputs; this avoids the kappa(P) error amplification a direct
                // triangular solve against an ill-conditioned P would incur.
                T* P_copy = new T[n * n]();
                lapack::lacpy(MatrixType::Upper, n, n, P, n, P_copy, n);   // P_copy = P (upper); lower stays 0
                if (!RandLAPACK::util::diag_is_nonzero(n, P_copy, n)) {
                    std::fprintf(stderr, "[cholqr_primitive] FAIL: GEQP3/BQRRP diag_is_nonzero(P) failed\n");
                    delete[] P_copy; return 1;
                }

                int64_t* jpiv = new int64_t[n]();
                T* tau_qr = new T[n];

                if (method == PCholQRPrecondMethod::GEQP3) {
                    lapack::geqp3(n, n, P_copy, n, jpiv, tau_qr);
                } else {
                    if (state == nullptr) {
                        std::fprintf(stderr, "[cholqr_primitive] FAIL: BQRRP called with state=nullptr\n");
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

                // After QRCP, P_copy holds R_tri (upper) + Householder vectors (lower).
                // Pull out R_tri, rebuild Q in place, then form R_tri^{-1} Q^T.
                T* R_buf = new T[n * n]();
                lapack::lacpy(MatrixType::Upper, n, n, P_copy, n, R_buf, n);   // R_buf = R_tri
                lapack::ungqr(n, n, n, P_copy, n, tau_qr);                     // P_copy = Q
                RandLAPACK::util::transpose_square(P_copy, n);                 // P_copy = Q^T
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper,
                           Op::NoTrans, Diag::NonUnit,
                           n, n, T(1), R_buf, n, P_copy, n);                   // P_copy = R_tri^{-1} Q^T

                // Apply the pivot permutation Pi to the rows: R_pre = Pi (R_tri^{-1} Q^T).
                // (Row scatter by jpiv; col_swap is column-oriented so it doesn't apply here.)
                std::fill(R_pre, R_pre + n * n, T(0));
                for (int64_t k = 0; k < n; ++k)
                    for (int64_t j = 0; j < n; ++j)
                        R_pre[(jpiv[k] - 1) + j * n] = P_copy[k + j * n];

                delete[] P_copy;
                delete[] R_buf;
                delete[] jpiv;
                delete[] tau_qr;
                break;
            }
        }
    }

    if (timing) {
        t1 = steady_clock::now();
        precond_inv_us = preconditioned ? duration_cast<microseconds>(t1 - t0).count() : 0;
    }

    // ---- Step 2: form the Gram G ----
    // Preconditioned TRSM_IDENTITY/TRTRI defer the left R_pre^T factor to a single
    // TRSM (cheaper, O(n^3/2), and equally stable since P is preserved); GEQP3/BQRRP
    // keep the explicit per-block GEMM with R_pre^T to preserve the QRCP-accurate
    // R_pre (a TRSM with the ill-conditioned P would re-amplify the error). The
    // unpreconditioned path forms plain A^T A.
    const bool use_trsm_at_end = preconditioned
                              && (method == PCholQRPrecondMethod::TRSM_IDENTITY
                               || method == PCholQRPrecondMethod::TRTRI);

    blocked_preconditioned_gram<T, GLO>(A, preconditioned ? R_pre : (const T*)nullptr,
                                         G, m, n, b_eff, A_temp, Z_buf,
                                         /*skip_left_factor=*/use_trsm_at_end,
                                         fwd_us, adj_us, gemm_us, timing);

    if (use_trsm_at_end) {
        // G := P^{-T} G  (= R_pre^T A^T A R_pre, since R_pre = P^{-1}).
        if (timing) t0 = steady_clock::now();
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans,
                   Diag::NonUnit, n, n, T(1), P, n, G, n);
        if (timing) { t1 = steady_clock::now(); gemm_us += duration_cast<microseconds>(t1 - t0).count(); }
    }

    // ---- Step 3: Cholesky G = (R^chol)^T R^chol, with adaptive-shift retry ----
    //
    // The retry block exists because the (preconditioned) Gram can pick up a
    // non-PD pivot from rounding -- amplified by kappa(R_pre)^2 in the iter-2/3
    // Gram of CholQR2/sCholQR3. We snapshot G and trace(G) once, then on each
    // potrf failure restore G, grow the diagonal shift, and retry. trace(G) is
    // the natural scale for the shift; G_backup lets retries stay O(n^2). Seeding
    // from eps on the first bump lets an unshifted (shift_factor==0) caller keep a
    // clean first attempt while still being rescued if the Gram is non-PD.
    T* G_backup = (max_retries != 0) ? new T[n * n] : nullptr;
    T trace_G = 0;
    for (int64_t i = 0; i < n; ++i) trace_G += G[i * (n + 1)];
    if (G_backup) std::copy(G, G + n * n, G_backup);

    if (timing) t0 = steady_clock::now();

    int info = 0;
    T current_shift_factor = shift_factor;
    for (int attempt = 0; max_retries < 0 || attempt <= max_retries; ++attempt) {
        if (attempt > 0) {
            // Restore the Gram and grow the shift (seed at eps if we started at 0).
            std::copy(G_backup, G_backup + n * n, G);
            current_shift_factor = (current_shift_factor > T(0))
                                 ? current_shift_factor * shift_growth
                                 : std::numeric_limits<T>::epsilon();
        }
        if (current_shift_factor > T(0)) {
            T shift = current_shift_factor * trace_G;
            for (int64_t i = 0; i < n; ++i) G[i * (n + 1)] += shift;
        }
        if (n > 1)
            lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), &G[1], n);
        info = lapack::potrf(Uplo::Upper, n, G, n);
        if (info == 0) break;
    }

    if (info) {
        std::fprintf(stderr,
            "[cholqr_primitive] FAIL: lapack::potrf returned info=%d after %d "
            "retries (final shift_factor=%g)\n",
            info, max_retries, (double)current_shift_factor);
        delete[] G_backup;
        return info;
    }
    delete[] G_backup;

    if (timing) { t1 = steady_clock::now(); chol_us = duration_cast<microseconds>(t1 - t0).count(); }

    // ---- Step 4: output R ----
    // Unpreconditioned: R = R^chol. Preconditioned: R = R^chol * P (accumulates
    // the running factor), computed in place by seeding R with P then a TRMM.
    if (timing) t0 = steady_clock::now();
    if (preconditioned) {
        lapack::lacpy(MatrixType::Upper, n, n, P, n, R, ldr);
        if (n > 1)
            lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R + 1, ldr);
        blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper,
                   Op::NoTrans, Diag::NonUnit, n, n, T(1), G, n, R, ldr);
    } else {
        lapack::lacpy(MatrixType::Upper, n, n, G, n, R, ldr);
        if (n > 1)
            lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R + 1, ldr);
    }
    if (timing) { t1 = steady_clock::now(); update_us = preconditioned ? duration_cast<microseconds>(t1 - t0).count() : 0; }

    return 0;
}


// Unpreconditioned convenience overload: plain CholQR (P = nullptr). Owns the G /
// A_temp scratch the general primitive needs and forwards. This is the entry point
// for CholQR and for iteration 1 of CholQR2 / sCholQR3.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int cholqr_primitive(
    GLO& A,
    T* R, int64_t ldr,
    T shift_factor,
    int64_t block_size,
    long& fwd_us, long& adj_us, long& chol_us,
    bool timing,
    int max_retries = 0,
    T shift_growth = T(10))
{
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;

    T* G      = new T[n * n]();
    T* A_temp = new T[m * b_eff];

    long precond_inv_us = 0, gemm_us = 0, update_us = 0;
    int info = cholqr_primitive<T, GLO>(
        A, /*P=*/(const T*)nullptr, R, ldr,
        PCholQRPrecondMethod::TRSM_IDENTITY,   // ignored when P == nullptr
        block_size, /*bqrrp_block_ratio=*/T(1), /*R_pre=*/(T*)nullptr,
        G, A_temp, /*Z_buf=*/(T*)nullptr,
        /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
        precond_inv_us, fwd_us, adj_us, gemm_us, chol_us, update_us,
        timing, shift_factor, max_retries, shift_growth);

    delete[] G;
    delete[] A_temp;
    return info;
}

} // namespace RandLAPACK
