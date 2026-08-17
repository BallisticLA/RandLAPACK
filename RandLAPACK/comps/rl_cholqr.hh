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
#include <cmath>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <string>

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
            lapack::laset(MatrixType::General, b_j, b_j, (T)0, (T)1, I_block + j, n);
            B_in = I_block;
        }

        // A_temp = A * B_in   (m x b_j)
        if (timing) t0 = steady_clock::now();
        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, b_j, n, (T)1.0, B_in, n, (T)0.0, A_temp, m);
        if (timing) { t1 = steady_clock::now(); fwd_accum += duration_cast<microseconds>(t1 - t0).count(); }

        if (R_pre && !skip_left_factor) {
            // Z_buf = A^T * A_temp ; then G[:, blk] = R_pre^T * Z_buf.
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }

            if (timing) t0 = steady_clock::now();
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_j, n, (T)1.0, R_pre, n, Z_buf, n, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); gemm_accum += duration_cast<microseconds>(t1 - t0).count(); }
        } else {
            // skip_left_factor (preconditioned) and the unpreconditioned path both
            // write A^T * A_temp straight into G[:, blk]; any left factor is applied
            // once after the loop by the caller.
            if (timing) t0 = steady_clock::now();
            A(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n);
            if (timing) { t1 = steady_clock::now(); adj_accum += duration_cast<microseconds>(t1 - t0).count(); }

            // Clear this block's identity ones before the next iteration.
            if (R_pre == nullptr)
                lapack::laset(MatrixType::General, b_j, b_j, (T)0, (T)0, I_block + j, n);
        }
    }

    if (I_block) delete[] I_block;

    if (timing) {
        fwd_us = fwd_accum;
        adj_us = adj_accum;
        gemm_us = gemm_accum;
    }
}


// Materialize Q = A * R^{-1} (m x n) block-by-block via the linop, for the
// test/verify paths of the CholQR-family drivers. R is n x n upper-triangular
// (ld = ldr); Q_out (m x n) is caller-allocated. Forms R^{-1} once (n x n trsm),
// then applies A to its column blocks. Not on any timed/algorithmic path.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
void materialize_Q_from_R(GLO& A, const T* R, int64_t ldr,
                          int64_t m, int64_t n, int64_t b_eff, T* Q_out) {
    T* R_inv = new T[n * n];
    RandLAPACK::util::eye(n, n, R_inv);
    blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, T(1), R, ldr, R_inv, n);
    for (int64_t j = 0; j < n; j += b_eff) {
        int64_t b_j = std::min(b_eff, n - j);
        A(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, b_j, n, T(1), R_inv + j * n, n, T(0), Q_out + j * m, m);
    }
    delete[] R_inv;
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
    T shift_growth = T(10),
    int* n_retries = nullptr)   // out: number of shift retries used (0 = clean first attempt)
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
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, T(1), P, n, R_pre, n);
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
            #if defined(__APPLE__)
                // The whole QRCP-based preconditioner path (GEQP3/BQRRP, then ungqr +
                // lapmr to form Pi R_tri^{-1} Q^T) pulls in LAPACK / BQRRP routines that
                // are unsupported under Apple Accelerate; the sibling rl_cqrrpt.hh /
                // rl_hqrrp.hh guard the whole QRCP path the same way. The standard
                // CholQR / CholQR2 / sCholQR3 methods use TRSM_IDENTITY and never reach
                // here, so only the stabilized QRCP preconditioner is disabled on macOS.
                (void)bqrrp_block_ratio; (void)state;
                std::fprintf(stderr, "[cholqr_primitive] FAIL: GEQP3/BQRRP preconditioning is unsupported on Apple Accelerate.\n");
                return 1;
            #else
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
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, T(1), R_buf, n, P_copy, n);  // P_copy = R_tri^{-1} Q^T

                // R_pre = Pi (R_tri^{-1} Q^T): jpiv is the column-pivot permutation, applied to
                // the ROWS here, so copy R^{-1} Q^T over and apply it with lapmr (row permute).
                lapack::lacpy(MatrixType::General, n, n, P_copy, n, R_pre, n);
                lapack::lapmr(false, n, n, R_pre, n, jpiv);

                delete[] P_copy;
                delete[] R_buf;
                delete[] jpiv;
                delete[] tau_qr;
                break;
            #endif
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
    //
    // RANDLAPACK_GRAM_LEFT=gemm forces the per-block GEMM with R_pre^T even for
    // TRSM_IDENTITY/TRTRI (2026-08-12 experiment: the paper's stability analysis
    // models the explicit-multiply path, the default runs the TRSM path; this knob
    // lets a campaign measure both). Only the gemm direction can be forced: making
    // GEQP3/BQRRP use the TRSM would reintroduce the error their construction avoids.
    static const bool force_gemm_left = []() {
        const char* s = std::getenv("RANDLAPACK_GRAM_LEFT");
        return s != nullptr && std::string(s) == "gemm";
    }();
    const bool use_trsm_at_end = preconditioned
                              && !force_gemm_left
                              && (method == PCholQRPrecondMethod::TRSM_IDENTITY
                               || method == PCholQRPrecondMethod::TRTRI);

    blocked_preconditioned_gram<T, GLO>(A, preconditioned ? R_pre : (const T*)nullptr, G, m, n, b_eff, A_temp, Z_buf, /*skip_left_factor=*/use_trsm_at_end, fwd_us, adj_us, gemm_us, timing);

    if (use_trsm_at_end) {
        // G := P^{-T} G  (= R_pre^T A^T A R_pre, since R_pre = P^{-1}).
        if (timing) t0 = steady_clock::now();
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit, n, n, T(1), P, n, G, n);
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
    int attempt = 0;
    T current_shift_factor = shift_factor;
    // max_retries < 0 means "unbounded", which with a geometrically growing shift is
    // *usually* fine: the shift eventually makes G diagonally dominant and potrf
    // succeeds. But the argument fails on non-finite data -- an inf/NaN in G, or a shift
    // that overflows to inf, gives a Gram potrf can never factor, and the loop then never
    // terminates. kUnboundedRetryCeiling is a backstop for exactly that case: it is far
    // above any legitimate retry count (each attempt multiplies the shift by
    // shift_growth, so tens of attempts already span the entire exponent range), so it
    // cannot truncate a run that would otherwise have succeeded.
    constexpr int kUnboundedRetryCeiling = 128;
    for (; (max_retries < 0) ? (attempt < kUnboundedRetryCeiling) : (attempt <= max_retries); ++attempt) {
        if (attempt > 0) {
            // Restore the Gram and grow the shift (seed at eps if we started at 0).
            std::copy(G_backup, G_backup + n * n, G);
            current_shift_factor = (current_shift_factor > T(0))
                                 ? current_shift_factor * shift_growth
                                 : std::numeric_limits<T>::epsilon();
            // A non-finite shift can never rescue the factorization; bail out rather
            // than spin. Same for a non-finite trace, which poisons every shift below.
            if (!std::isfinite(current_shift_factor) || !std::isfinite(trace_G)) {
                info = -1;
                break;
            }
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
    if (n_retries) *n_retries = attempt;   // 0 = succeeded on the unshifted first attempt

    if (info) {
        // Report the retries actually made (`attempt`), not the configured limit --
        // with max_retries = -1 the old message printed "-1 retries".
        std::fprintf(stderr,
            "[cholqr_primitive] FAIL: lapack::potrf returned info=%d after %d "
            "attempt(s) (final shift_factor=%g)\n",
            info, attempt, (double)current_shift_factor);
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
        blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, T(1), G, n, R, ldr);
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
    T shift_growth = T(10),
    int* n_retries = nullptr)
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
        timing, shift_factor, max_retries, shift_growth, n_retries);

    delete[] G;
    delete[] A_temp;
    return info;
}


// ============================================================================
// cholqr_iterate — the shared CholQR-family engine
// ============================================================================
//
// Runs `num_iters` CholQR passes and returns the accumulated R (= R_k ... R_1):
//   iter 1            : unpreconditioned CholQR with shift_iter1.
//   iter 2..num_iters : preconditioned CholQR with the previous iterate as P
//                       (TRSM_IDENTITY) and shift_iter_rest.
// This is the single engine behind CholQR (num_iters = 1), CholQR2 (2), and
// sCholQR3 (3); the only differences are the pass count and the per-pass shift
// (CholQR/CholQR2 start unshifted, shift_iter1 = shift_iter_rest = 0; sCholQR3
// uses eps). The adaptive-shift retry inside cholqr_primitive handles potrf
// breakdown; max_retries < 0 means unbounded.
//
// Scratch for the preconditioned passes (G, R_pre, P_prev, A_temp, Z_buf) is owned
// internally and only allocated when num_iters > 1, so the num_iters = 1 (plain
// CholQR) path keeps its lean footprint.
//
// iter_times (optional, length 5*num_iters): per pass [fwd, adj, gemm, chol, upd];
// gemm and upd are 0 for the unpreconditioned first pass.
//
// Returns 0 on success, or the 1-based index of the pass whose Cholesky failed.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int cholqr_iterate(
    GLO& A, T* R, int64_t ldr, int64_t block_size,
    int num_iters, T shift_iter1, T shift_iter_rest,
    int max_retries, T shift_growth, bool timing,
    long* iter_times = nullptr,
    int* n_retries_total = nullptr)   // out: total shift retries summed across all passes
{
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = (block_size > 0 && block_size < n) ? block_size : n;
    int total_retries = 0, pass_retries = 0;
    auto rec = [&](int it, long fwd, long adj, long gemm, long chol, long upd) {
        if (iter_times) {
            long* t = iter_times + 5 * (it - 1);
            t[0] = fwd; t[1] = adj; t[2] = gemm; t[3] = chol; t[4] = upd;
        }
    };

    // ---- Iter 1: unpreconditioned CholQR ----
    long fwd1 = 0, adj1 = 0, chol1 = 0;
    int info = cholqr_primitive<T, GLO>(A, R, ldr, shift_iter1, block_size, fwd1, adj1, chol1, timing, max_retries, shift_growth, &pass_retries);
    total_retries += pass_retries;
    if (info != 0) { if (n_retries_total) *n_retries_total = total_retries; return 1; }
    rec(1, fwd1, adj1, 0, chol1, 0);
    if (num_iters <= 1) { if (n_retries_total) *n_retries_total = total_retries; return 0; }

    // ---- Iters 2..num_iters: preconditioned CholQR with P = previous R ----
    T* G      = new T[n * n]();
    T* R_pre  = new T[n * n]();
    T* P_prev = new T[n * n]();
    T* A_temp = new T[m * b_eff];
    T* Z_buf  = new T[n * b_eff];
    for (int it = 2; it <= num_iters; ++it) {
        lapack::lacpy(MatrixType::Upper, n, n, R, ldr, P_prev, n);
        if (n > 1) lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), P_prev + 1, n);
        long precond_inv = 0, fwd = 0, adj = 0, gemm = 0, chol = 0, upd = 0;
        info = cholqr_primitive<T, GLO>(
            A, P_prev, R, ldr, PCholQRPrecondMethod::TRSM_IDENTITY,
            block_size, /*bqrrp_block_ratio=*/T(1), R_pre, G, A_temp, Z_buf,
            /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
            precond_inv, fwd, adj, gemm, chol, upd, timing,
            shift_iter_rest, max_retries, shift_growth, &pass_retries);
        total_retries += pass_retries;
        if (info != 0) {
            delete[] G; delete[] R_pre; delete[] P_prev; delete[] A_temp; delete[] Z_buf;
            if (n_retries_total) *n_retries_total = total_retries;
            return it;
        }
        rec(it, fwd, adj, gemm, chol, precond_inv + upd);
    }
    delete[] G; delete[] R_pre; delete[] P_prev; delete[] A_temp; delete[] Z_buf;
    if (n_retries_total) *n_retries_total = total_retries;
    return 0;
}

} // namespace RandLAPACK
