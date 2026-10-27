#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_bqrrp.hh"
#include "rl_exceptions.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace RandLAPACK {

/// How cholqr_primitive inverts the upper-triangular preconditioner P (R_pre = P^{-1}) in the
/// preconditioned path, trading speed for stability when P is ill-conditioned:
///
/// trsm_identity  Solve P * R_pre = I via TRSM(Side::Left), backward-stable for the
///                subsequent product A * R_pre. n^3/3 flops.
/// trtri          Direct triangular inverse via LAPACK trtri. Same cost; less stable than
///                trsm_identity when P is ill-conditioned in practice.
/// geqp3          R_pre via column-pivoted QR of P (geqp3), giving P^{-1} = Pi R^{-1} Q^T.
///                Stable for ill-conditioned P. About 11 n^3/3 flops.
/// bqrrp          Same output as geqp3, but pivots via RandLAPACK::BQRRP (blocked and
///                sketched). Faster for large n; requires an RNG state.
enum class CholQRPrecondMethod {
    trsm_identity,
    trtri,
    geqp3,
    bqrrp
};

/// Default column-block width of the blocked Gram computation, so the tall intermediate
/// costs O(m b) memory rather than O(m n). block_size = 0 forms it unblocked.
inline constexpr int64_t kDefaultGramBlockSize = 256;

/// Column-block width for a requested block_size: block_size <= 0 or >= n means n.
inline int64_t gram_block_width(int64_t block_size, int64_t n) {
    return (block_size > 0 && block_size < n) ? block_size : n;
}

/// Whether a preconditioned Gram applies its left factor R_pre^T by a GEMM per block, which
/// needs Z_buf (n x b). Always for geqp3 and bqrrp: a TRSM with an ill-conditioned P would
/// reintroduce the error their construction avoids. trsm_identity and trtri do so only with
/// gram_left_gemm, and otherwise apply P^{-T} as one TRSM after the Gram loop.
inline bool gram_left_by_gemm(CholQRPrecondMethod method, bool gram_left_gemm) {
    return gram_left_gemm || method == CholQRPrecondMethod::geqp3 || method == CholQRPrecondMethod::bqrrp;
}

/// Whether a preconditioned call forms P^{-1} explicitly in R_pre: all but trsm_identity with
/// the left factor by TRSM, whose Gram solves each column panel of P^{-1} as it goes.
inline bool forms_explicit_inverse(CholQRPrecondMethod method, bool gram_left_gemm) {
    return method != CholQRPrecondMethod::trsm_identity || gram_left_gemm;
}

/// cholqr_primitive return codes besides 0 (success) and a positive potrf info (a Cholesky
/// breakdown that survived every attempt).
inline constexpr int kCholQRNonFinite              = -1;  ///< trace(G), a shift, or a shifted diagonal entry is not finite
inline constexpr int kCholQRInvalidShift           = -2;  ///< a shift factor is negative or NaN, or shift_growth <= 1
inline constexpr int kCholQRSingularPreconditioner = -3;  ///< P, or the triangular factor of its pivoted QR, has a zero diagonal entry

/// Cap on Cholesky attempts when max_retries < 0. With shift growth g and an unshifted first
/// attempt, the last attempt's shift factor is eps * g^126, far beyond the diagonal dominance
/// that makes a finite Gram positive definite for g = 10; the cap ends the loop when growth is
/// slow or no shift rescues the data.
inline constexpr int kMaxShiftAttempts = 128;

/// potrf(Upper) that also fails at the first column whose factor diagonal is not finite: a NaN
/// or Inf in G's upper triangle always reaches it, but only some LAPACKs (MKL) flag a NaN pivot.
template <typename T>
inline int64_t potrf_upper_finite(int64_t n, T* G, int64_t ldg) {
    int64_t info = lapack::potrf(Uplo::Upper, n, G, ldg);
    for (int64_t j = 0; info == 0 && j < n; ++j)
        if (!std::isfinite(G[j + j * ldg])) info = j + 1;
    return info;
}

/// Retry and Gram options of cholqr_primitive. The defaults are a single attempt on the
/// symmetrized Gram.
template <typename T>
struct CholQROptions {
    int  max_retries     = 0;      ///< retries with a grown shift after a non-PD pivot; < 0: until
                                   ///< positive definite, at most kMaxShiftAttempts attempts
    T    shift_growth    = T(10);  ///< shift multiplier per retry; must be > 1
    bool symmetrize_gram = true;   ///< factor (G + G^T)/2 rather than G's upper triangle as formed
    bool gram_left_gemm  = false;  ///< trsm_identity/trtri: apply R_pre^T to each Gram block by
                                   ///< GEMM (see gram_left_by_gemm)
};

/// What the Cholesky step of one cholqr_primitive call did.
template <typename T>
struct CholQRShiftRecord {
    int n_retries     = 0;      ///< shift retries used (0 = the first attempt succeeded)
    T   applied_shift = T(0);   ///< absolute diagonal shift of the last attempt (0 = unshifted)
    T   gram_trace    = T(0);   ///< trace(G), the scale the shift multiplies
};

namespace detail {
/// Runs f, adding its wall-clock in microseconds to acc when timing is on.
template <typename F>
inline void timed_us(bool timing, long& acc, F&& f) {
    if (!timing) { f(); return; }
    auto t0 = std::chrono::steady_clock::now();
    f();
    acc += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace detail


// ============================================================================
// blocked_preconditioned_gram
// ============================================================================
//
// Forms the (optionally preconditioned) Gram matrix one column block at a time, using only
// A * B and A^T * B:
//
//   R_pre != nullptr :  G = R_pre^T (A^T A) R_pre
//   P != nullptr     :  G = A^T A P^{-1}, each column panel of P^{-1} solved just before its
//                       apply by the TRSM against the identity that invert_upper_into does
//   neither          :  G = A^T A     (A's column blocks from materialize_cols when A provides
//                                      it, otherwise A applied to column blocks of the identity)
//
// A_temp (m x b_eff) holds A * B_in for the current block; Z_buf (n x b_eff) holds
// A^T * A_temp on the R_pre path when skip_left_factor is false. With skip_left_factor (which
// the P path requires), G receives A^T A R_pre (or A^T A P^{-1}) and the caller applies the
// left factor. Timing accumulators are outputs, written only when timing is true; left_us is
// the left factor's GEMMs and inv_us the P path's panel solves.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
void blocked_preconditioned_gram(
    GLO& A,
    const T* R_pre,
    const T* P, int64_t ldp,
    T* G,
    int64_t m, int64_t n, int64_t b_eff,
    T* A_temp,
    T* Z_buf,
    bool skip_left_factor,
    long& fwd_us, long& adj_us, long& left_us, long& inv_us,
    bool timing)
{
    if (timing) fwd_us = adj_us = left_us = inv_us = 0;
    const bool panel = (R_pre == nullptr && P != nullptr);
    randlapack_require(!panel || skip_left_factor)
        << "blocked_preconditioned_gram: the P path needs the left factor applied by the caller";
    constexpr bool has_cols = RandLAPACK::linops::HasMaterializeCols<GLO>;
    const bool direct = has_cols && R_pre == nullptr && !panel;   // A's columns, no identity block

    // One n x b_eff block, unless A's columns come from materialize_cols: identity columns whose
    // shifted-diagonal ones are set and cleared per block, or on the P path the panel
    // P^{-1}[:, j:j+b], zero below row j + b.
    std::unique_ptr<T[]> I_block((R_pre == nullptr && !direct) ? new T[n * b_eff]() : nullptr);

    for (int64_t j = 0; j < n; j += b_eff) {
        int64_t b_j = std::min(b_eff, n - j);

        const T* B_in = I_block.get();
        if (R_pre != nullptr) {
            B_in = R_pre + j * n;
        } else if (!direct) {
            lapack::laset(MatrixType::General, b_j, b_j, (T)0, (T)1, I_block.get() + j, n);
            if (panel)
                detail::timed_us(timing, inv_us, [&] {
                    blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                               j + b_j, b_j, T(1), P, ldp, I_block.get(), n); });
        }

        detail::timed_us(timing, fwd_us, [&] {
            if constexpr (has_cols) {
                if (direct) { A.materialize_cols(j, b_j, A_temp, m); return; }
            }
            A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, b_j, n, (T)1.0, B_in, n, (T)0.0, A_temp, m); });

        if (R_pre && !skip_left_factor) {
            // Z_buf = A^T * A_temp; G[:, blk] = R_pre^T * Z_buf.
            detail::timed_us(timing, adj_us, [&] {
                A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_j, m, (T)1.0, A_temp, m, (T)0.0, Z_buf, n); });
            detail::timed_us(timing, left_us, [&] {
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_j, n, (T)1.0, R_pre, n, Z_buf, n, (T)0.0, G + j * n, n); });
        } else {
            detail::timed_us(timing, adj_us, [&] {
                A(Layout::ColMajor, Op::Trans, Op::NoTrans, n, b_j, m, (T)1.0, A_temp, m, (T)0.0, G + j * n, n); });
            if (panel)
                lapack::laset(MatrixType::General, j + b_j, b_j, (T)0, (T)0, I_block.get(), n);
            else if (R_pre == nullptr && !direct)
                lapack::laset(MatrixType::General, b_j, b_j, (T)0, (T)0, I_block.get() + j, n);
        }
    }
}


// X := P^{-1} for P n x n upper triangular, by TRSM against the identity in column panels of
// width b_panel (TRSM rather than trtri: the more stable choice for ill-conditioned P).
// X is upper triangular, so panel [j, j+b) involves only the leading j+b rows: n^3/3 flops
// instead of the full-width solve's n^3. Not bit-identical to the full-width solve (BLAS blocks
// a different operand shape differently), but equal to rounding. Only P's upper triangle is read.
template <typename T>
inline void invert_upper_into(const T* P, int64_t ldp, T* X, int64_t ldx,
                              int64_t n, int64_t b_panel) {
    if (n <= 0) return;
    lapack::laset(MatrixType::General, n, n, T(0), T(1), X, ldx);
    const int64_t b = std::max<int64_t>(1, std::min(b_panel, n));
    for (int64_t j = 0; j < n; j += b) {
        const int64_t bj   = std::min(b, n - j);
        const int64_t lead = j + bj;
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   lead, bj, T(1), P, ldp, X + j * ldx, ldx);
    }
}

// B := U * B for n x n upper triangular U and B, in column panels of width b_panel, touching
// only each panel's leading rows (n^3/3 flops instead of n^3). Equal to the full-width trmm to
// rounding, not bit-identical. U's strict lower triangle is not read; B must hold zeros there.
template <typename T>
inline void trmm_upper_upper_left(const T* U, int64_t ldu, T* B, int64_t ldb,
                                  int64_t n, int64_t b_panel) {
    if (n <= 0) return;
    const int64_t b = std::max<int64_t>(1, std::min(b_panel, n));
    for (int64_t j = 0; j < n; j += b) {
        const int64_t bj   = std::min(b, n - j);
        const int64_t lead = j + bj;
        blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   lead, bj, T(1), U, ldu, B + j * ldb, ldb);
    }
}


// ============================================================================
// cholqr_primitive: Q-less (preconditioned) Cholesky QR
// ============================================================================
//
// Computes upper-triangular R such that A * R^{-1} has orthonormal columns:
//
//   P == nullptr:  G = A^T A;  G = R^T R (Cholesky);  output R.
//   P != nullptr:  R_pre = P^{-1} via `method`;  G = R_pre^T A^T A R_pre;
//                  G = (R^chol)^T R^chol;  output R = R^chol * P.
//
// trsm_identity with the left factor by TRSM (the default) never forms R_pre: the Gram solves
// each column panel of P^{-1} just before applying A to it, with the same TRSMs.
//
// Chaining passes with the previous R as P (CholQR2, sCholQR3) accumulates R_k ... R_1.
//
// Adaptive shift: the first attempt factors G + s I with s = shift_factor * trace(G). On a
// non-PD pivot the shift factor is grown by opts.shift_growth (seeded at eps when it is 0) and
// potrf retried, up to opts.max_retries times. G is formed once: a failed attempt is undone
// from G's untouched strict lower triangle plus an O(n) copy of its diagonal. With a nonzero
// shift s the output satisfies R^T R = A^T A + s P^T P in exact arithmetic (P = I when
// unpreconditioned), so A R^{-1} is only near-orthonormal: a shifted success does not certify
// orthogonality. The record reports the shift.
//
// P is n x n upper triangular with leading dimension ldp; only its upper triangle is read. P
// may be R itself (ldp == ldr), as cholqr_iterate passes the previous pass's factor.
// bqrrp_block_ratio is BQRRP's block size as a fraction of n; the sentinel 1.0 selects 1 for
// n <= 2000, 1/2 for n <= 8000 and 1/32 above. Caller-owned scratch: R_pre (n x n, only when
// P^{-1} is formed: trtri, geqp3, bqrrp, or trsm_identity with the left factor by GEMM; may be
// nullptr otherwise), G (n x n), A_temp (m x b_eff), Z_buf (n x b_eff, only when a
// preconditioned call applies the left factor by GEMM, see gram_left_by_gemm).
// state is the RNG state for bqrrp (nullptr otherwise). Timing arguments are outputs;
// precond_inv_us includes the Gram's panel solves.
//
// Returns 0 on success, a positive potrf info (or the first column whose factor diagonal is not
// finite) on a breakdown that survived every attempt, or kCholQRNonFinite, kCholQRInvalidShift,
// kCholQRSingularPreconditioner (bqrrp reports the latter when its rank truncation zeroes a
// pivot). R is written only on success. Caller bugs (m < n, ldr < n, ldp < n, P == R with
// ldp != ldr, null R, a missing R_pre, Z_buf or bqrrp state, a bqrrp_block_ratio outside
// (0, 1] with bqrrp) throw.
template <typename T, RandLAPACK::linops::LinearOperator GLO,
          typename RNG = RandBLAS::DefaultRNG>
int cholqr_primitive(
    GLO& A,
    const T* P, int64_t ldp,
    T* R, int64_t ldr,
    CholQRPrecondMethod method,
    int64_t block_size,
    T bqrrp_block_ratio,
    T* R_pre,
    T* G,
    T* A_temp,
    T* Z_buf,
    RandBLAS::RNGState<RNG>* state,
    long& precond_inv_us,
    long& fwd_us, long& adj_us, long& left_us, long& chol_us, long& update_us,
    bool timing,
    T shift_factor = T(0),
    const CholQROptions<T>& opts = CholQROptions<T>(),
    CholQRShiftRecord<T>* record = nullptr)
{
    using std::chrono::steady_clock;
    using std::chrono::duration_cast;
    using std::chrono::microseconds;
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = gram_block_width(block_size, n);
    const bool preconditioned = (P != nullptr);
    const bool use_trsm_at_end = preconditioned && !gram_left_by_gemm(method, opts.gram_left_gemm);
    const bool panel_inverse = preconditioned && !forms_explicit_inverse(method, opts.gram_left_gemm);

    randlapack_require(m >= n) << "cholqr_primitive: operator must be tall (m=" << m << " < n=" << n << ")";
    randlapack_require(ldr >= n) << "cholqr_primitive: ldr=" << ldr << " < n=" << n;
    randlapack_require(R != nullptr) << "cholqr_primitive: R buffer is null";
    randlapack_require(!preconditioned || ldp >= n) << "cholqr_primitive: ldp=" << ldp << " < n=" << n;
    randlapack_require(P != R || ldp == ldr)
        << "cholqr_primitive: P aliases R with ldp=" << ldp << " != ldr=" << ldr;
    randlapack_require(!preconditioned || panel_inverse || R_pre != nullptr)
        << "cholqr_primitive: R_pre is required unless trsm_identity applies the left factor by TRSM";
    randlapack_require(!preconditioned || use_trsm_at_end || Z_buf != nullptr)
        << "cholqr_primitive: Z_buf is required when the left factor is applied by GEMM";
    randlapack_require(!(preconditioned && method == CholQRPrecondMethod::bqrrp && state == nullptr))
        << "cholqr_primitive: the bqrrp preconditioner inversion needs an RNG state";
    randlapack_require(!(preconditioned && method == CholQRPrecondMethod::bqrrp)
                       || (bqrrp_block_ratio > T(0) && bqrrp_block_ratio <= T(1)))
        << "cholqr_primitive: bqrrp_block_ratio=" << bqrrp_block_ratio << " must lie in (0, 1]";

    if (record) *record = CholQRShiftRecord<T>();
    if (!(shift_factor >= T(0)) || !(opts.shift_growth > T(1)))
        return kCholQRInvalidShift;

    steady_clock::time_point t0, t1;
    if (timing) t0 = steady_clock::now();

    // ---- Step 1: R_pre = P^{-1} (preconditioned only) ----
    if (preconditioned) {
        if (!RandLAPACK::util::diag_is_nonzero(n, P, ldp))
            return kCholQRSingularPreconditioner;
        switch (method) {
            case CholQRPrecondMethod::trsm_identity: {
                if (!panel_inverse)
                    invert_upper_into<T>(P, ldp, R_pre, n, n, b_eff);
                break;
            }
            case CholQRPrecondMethod::trtri: {
                lapack::lacpy(MatrixType::Upper, n, n, P, ldp, R_pre, n);
                RandLAPACK::util::get_U(n, n, R_pre, n);
                lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_pre, n);   // P's diagonal is nonzero
                break;
            }
            case CholQRPrecondMethod::geqp3:
            case CholQRPrecondMethod::bqrrp: {
                // Column-pivoted QR P Pi = Q R_tri gives P^{-1} = Pi R_tri^{-1} Q^T, avoiding
                // the kappa(P) error amplification of a direct triangular inverse.
                std::unique_ptr<T[]> P_copy(new T[n * n]());           // P with an explicit zero lower triangle
                std::unique_ptr<int64_t[]> jpiv(new int64_t[n]());     // geqp3 reads jpiv: 0 = free column
                std::unique_ptr<T[]> tau_qr(new T[n]);
                lapack::lacpy(MatrixType::Upper, n, n, P, ldp, P_copy.get(), n);

                if (method == CholQRPrecondMethod::geqp3) {
                    lapack::geqp3(n, n, P_copy.get(), n, jpiv.get(), tau_qr.get());
                } else {
                    T ratio = bqrrp_block_ratio;
                    if (ratio == T(1.0)) {
                        if (n <= 2000)      ratio = T(1.0);
                        else if (n <= 8000) ratio = T(0.5);
                        else                ratio = T(1.0) / T(32);
                    }
                    int64_t bqrrp_b = std::max(int64_t(1), (int64_t)(n * ratio));
                    RandLAPACK::BQRRP<T, RNG> bqrrp(false, bqrrp_b);
                    bqrrp.call(n, n, P_copy.get(), n, T(1), tau_qr.get(), jpiv.get(), *state);
                }
                if (!RandLAPACK::util::diag_is_nonzero(n, P_copy.get(), n))
                    return kCholQRSingularPreconditioner;

                // P_copy holds R_tri (upper) and the Householder vectors: form Q in R_pre, then
                // R_tri^{-1} Q^T with R_tri read in place, and permute its rows by the pivots.
                lapack::lacpy(MatrixType::General, n, n, P_copy.get(), n, R_pre, n);
                lapack::ungqr(n, n, n, R_pre, n, tau_qr.get());
                RandLAPACK::util::transpose_square(R_pre, n);
                blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                           n, n, T(1), P_copy.get(), n, R_pre, n);
                lapack::lapmr(false, n, n, R_pre, n, jpiv.get());
                break;
            }
        }
    }

    if (timing) {
        t1 = steady_clock::now();
        precond_inv_us = preconditioned ? duration_cast<microseconds>(t1 - t0).count() : 0;
    }

    // ---- Step 2: form G ----
    long panel_inv_us = 0;
    blocked_preconditioned_gram<T, GLO>(A, (preconditioned && !panel_inverse) ? R_pre : (const T*)nullptr,
                                        panel_inverse ? P : (const T*)nullptr, ldp, G, m, n, b_eff,
                                        A_temp, Z_buf, /*skip_left_factor=*/use_trsm_at_end,
                                        fwd_us, adj_us, left_us, panel_inv_us, timing);
    if (timing) precond_inv_us += panel_inv_us;

    if (use_trsm_at_end)
        detail::timed_us(timing, left_us, [&] {
            blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit, n, n, T(1), P, ldp, G, n); });

    // G <- (G + G^T)/2 in both triangles, so a retry restores exactly the matrix attempt 0
    // factored. Without symmetrization a retry restores the upper triangle from the strict
    // lower one, which differs from it by formation rounding only.
    if (opts.symmetrize_gram)
        detail::timed_us(timing, left_us, [&] {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < j; ++i) {
                    T avg = (G[i + j * n] + G[j + i * n]) / T(2);
                    G[i + j * n] = avg;
                    G[j + i * n] = avg;
                }
            }
        });

    // ---- Step 3: Cholesky with adaptive-shift retry ----
    // potrf(Upper) reads and overwrites only the upper triangle, so G's strict lower triangle
    // survives a failed attempt and, with the saved diagonal, restores the upper.
    std::unique_ptr<T[]> diag_backup((opts.max_retries != 0) ? new T[n] : nullptr);
    T trace_G = 0;
    for (int64_t i = 0; i < n; ++i) {
        T d = G[i * (n + 1)];
        trace_G += d;
        if (diag_backup) diag_backup[i] = d;
    }
    if (record) record->gram_trace = trace_G;
    if (!std::isfinite(trace_G))
        return kCholQRNonFinite;

    if (timing) t0 = steady_clock::now();

    int info = 0;
    int attempt = 0;
    T current_shift_factor = shift_factor;
    T last_shift = T(0);
    for (; (opts.max_retries < 0) ? (attempt < kMaxShiftAttempts) : (attempt <= opts.max_retries); ++attempt) {
        if (attempt > 0) {
            RandBLAS::symmetrize(Layout::ColMajor, Uplo::Lower, n, G, n);
            for (int64_t j = 0; j < n; ++j) G[j * (n + 1)] = diag_backup[j];
            current_shift_factor = (current_shift_factor > T(0))
                                 ? current_shift_factor * opts.shift_growth
                                 : std::numeric_limits<T>::epsilon();
        }
        if (current_shift_factor > T(0)) {
            T shift = current_shift_factor * trace_G;
            bool finite = std::isfinite(shift);
            for (int64_t i = 0; finite && i < n; ++i) {
                G[i * (n + 1)] += shift;
                finite = std::isfinite(G[i * (n + 1)]);
            }
            if (!finite) {
                info = kCholQRNonFinite;
                break;
            }
            last_shift = shift;
        }
        info = (int)potrf_upper_finite(n, G, n);
        if (info == 0) break;
    }
    if (record) {
        // On failure the attempt indexed by `attempt` never succeeded; count retries before it.
        record->n_retries     = (info == 0) ? attempt : (attempt > 0 ? attempt - 1 : 0);
        record->applied_shift = last_shift;
    }
    if (info != 0)
        return info;

    if (timing) { t1 = steady_clock::now(); chol_us = duration_cast<microseconds>(t1 - t0).count(); }

    // ---- Step 4: R = R^chol (unpreconditioned) or R^chol * P (preconditioned) ----
    if (timing) t0 = steady_clock::now();
    if (preconditioned) {
        if (P != R)
            lapack::lacpy(MatrixType::Upper, n, n, P, ldp, R, ldr);
        RandLAPACK::util::get_U(n, n, R, ldr);
        trmm_upper_upper_left<T>(G, n, R, ldr, n, b_eff);
    } else {
        lapack::lacpy(MatrixType::Upper, n, n, G, n, R, ldr);
        RandLAPACK::util::get_U(n, n, R, ldr);
    }
    if (timing) { t1 = steady_clock::now(); update_us = preconditioned ? duration_cast<microseconds>(t1 - t0).count() : 0; }

    return 0;
}


// ============================================================================
// cholqr_iterate: the CholQR-family engine
// ============================================================================
//
// Runs `num_iters` (>= 1) CholQR passes and returns the accumulated R (= R_k ... R_1): pass 1
// is unpreconditioned with shift factor shift_factor_iter1, passes 2..num_iters are
// preconditioned on the previous iterate (trsm_identity) with shift_factor_rest. CholQR,
// CholQR2 and sCholQR3 are num_iters = 1, 2, 3. Both shift factors are validated before pass 1;
// an invalid one fails as pass 1. Later passes read R itself as their preconditioner, so they
// need no scratch beyond pass 1's unless gram_left_gemm asks for R_pre and Z_buf, which are
// allocated when pass 2 starts.
//
// iter_times (optional, length 5*num_iters): per pass [fwd, adj, left, chol, upd], where fwd
// is A's products (or, on pass 1 with an operator that provides materialize_cols, its column
// copy), left is the Gram's left factor (its TRSM or GEMMs) plus the symmetrization and upd is
// the preconditioner inversion plus the R update; left and upd are always 0 for pass 1 (its
// symmetrization is not recorded). records (optional, length num_iters): each pass's shift
// record, default-initialized for passes not reached. pass_status (optional):
// cholqr_primitive's return code for the failed pass, 0 on success.
//
// Returns 0 on success, or the 1-based index of the pass that failed; R then holds the factor
// of the last successful pass (untouched if pass 1 failed or a shift factor was invalid).
template <typename T, RandLAPACK::linops::LinearOperator GLO>
int cholqr_iterate(
    GLO& A, T* R, int64_t ldr, int64_t block_size,
    int num_iters, T shift_factor_iter1, T shift_factor_rest,
    const CholQROptions<T>& opts, bool timing,
    long* iter_times = nullptr,
    CholQRShiftRecord<T>* records = nullptr,
    int* pass_status = nullptr)
{
    randlapack_require(num_iters >= 1) << "cholqr_iterate: num_iters=" << num_iters << " must be >= 1";
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    int64_t b_eff = gram_block_width(block_size, n);
    if (records) std::fill(records, records + num_iters, CholQRShiftRecord<T>());
    if (pass_status) *pass_status = 0;
    auto fail = [&](int it, int info) {
        if (pass_status) *pass_status = info;
        return it;
    };
    if (!(shift_factor_iter1 >= T(0)))
        return fail(1, kCholQRInvalidShift);
    if (num_iters > 1 && !(shift_factor_rest >= T(0)))
        return fail(1, kCholQRInvalidShift);
    auto rec = [&](int it, long fwd, long adj, long left, long chol, long upd) {
        if (iter_times) {
            long* t = iter_times + 5 * (it - 1);
            t[0] = fwd; t[1] = adj; t[2] = left; t[3] = chol; t[4] = upd;
        }
    };

    std::unique_ptr<T[]> G(new T[n * n]), A_temp(new T[m * b_eff]);
    std::unique_ptr<T[]> R_pre, Z_buf;   // gram_left_gemm only, allocated after pass 1's Gram scratch is freed
    auto* no_state = (RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr;

    for (int it = 1; it <= num_iters; ++it) {
        const bool first = (it == 1);
        if (it == 2 && forms_explicit_inverse(CholQRPrecondMethod::trsm_identity, opts.gram_left_gemm)) {
            R_pre.reset(new T[n * n]);
            Z_buf.reset(new T[n * b_eff]);
        }
        long precond_inv = 0, fwd = 0, adj = 0, left = 0, chol = 0, upd = 0;
        int info = cholqr_primitive<T, GLO>(
            A, first ? (const T*)nullptr : R, ldr, R, ldr, CholQRPrecondMethod::trsm_identity,
            block_size, /*bqrrp_block_ratio=*/T(1), R_pre.get(), G.get(), A_temp.get(), Z_buf.get(),
            no_state, precond_inv, fwd, adj, left, chol, upd, timing,
            first ? shift_factor_iter1 : shift_factor_rest, opts, records ? &records[it - 1] : nullptr);
        if (info != 0) return fail(it, info);
        if (first) rec(it, fwd, adj, 0, chol, 0);
        else       rec(it, fwd, adj, left, chol, precond_inv + upd);
    }
    return 0;
}

/// Q := A R^{-1} (m x n, leading dimension m): A's columns, from materialize_cols when A provides
/// it and otherwise A applied to block_size-wide column blocks of the identity (so no operand of
/// a composite A is materialized), then a right triangular solve with R.
template <typename T, RandLAPACK::linops::LinearOperator GLO>
void form_q_from_r(GLO& A, const T* R, int64_t ldr, int64_t block_size, T* Q) {
    int64_t m = A.n_rows;
    int64_t n = A.n_cols;
    if constexpr (RandLAPACK::linops::HasMaterializeCols<GLO>) {
        A.materialize_cols(0, n, Q, m);
    } else {
        int64_t b = gram_block_width(block_size, n);
        std::unique_ptr<T[]> I_block(new T[n * b]());
        for (int64_t j = 0; j < n; j += b) {
            int64_t b_j = std::min(b, n - j);
            lapack::laset(MatrixType::General, b_j, b_j, T(0), T(1), I_block.get() + j, n);
            A(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, b_j, n, T(1), I_block.get(), n, T(0), Q + j * m, m);
            lapack::laset(MatrixType::General, b_j, b_j, T(0), T(0), I_block.get() + j, n);
        }
    }
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
               m, n, T(1), R, ldr, Q, m);
}

// ============================================================================
// CholQRLinopsBase: members shared by the CholQR-family drivers
// ============================================================================

/// Members, options and test-mode Q shared by the Cholesky QR drivers on linear operators.
/// NumPasses sizes the per-pass shift record.
///
/// Every call() starts by resetting times, n_chol_retries, chol_status and the shift record,
/// before any argument check, so a call that throws RandLAPACK::Error on invalid arguments
/// leaves them reset and R and Q as they were. An exception from the operator propagates as
/// is: times stays empty and Q as it was, but R and the shift records of the passes that
/// completed may already be written.
template <typename T, int NumPasses>
class CholQRLinopsBase {
    public:
        bool timing;
        bool test_mode;

        /// Test-mode Q = A R^{-1} (m x n) of the last successful test-mode call, owned by the
        /// driver; nullptr until one.
        T* Q = nullptr;
        int64_t Q_rows = 0;
        int64_t Q_cols = 0;

        /// Timing breakdown of the last call (layout per driver); empty if it was untimed,
        /// failed or threw.
        std::vector<long> times;
        /// Total wall-clock (us) of the last call(), or -1 if it was untimed or failed.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        /// Column-block width of the Gram computation; <= 0 or >= n means unblocked.
        int64_t block_size = kDefaultGramBlockSize;
        int  max_retries     = -1;     ///< Cholesky retries with a grown shift; < 0: until positive
                                       ///< definite, at most kMaxShiftAttempts attempts
        T    shift_growth    = T(10);  ///< shift multiplier per retry
        bool symmetrize_gram = true;   ///< factor (G + G^T)/2 rather than G as formed

        int n_chol_retries = 0;   ///< shift retries of the last call, summed over passes
        int chol_status    = 0;   ///< cholqr_primitive's status for the failed pass (0 = success)
        /// Per-pass shift record of the last call: each pass's last-attempt shift (0 = unshifted)
        /// and Gram trace. After a failure the passes that ran keep theirs (a non-finite Gram
        /// leaves its non-finite trace); later passes hold 0.
        T chol_applied_shifts[NumPasses] = {};
        T chol_gram_traces[NumPasses]    = {};

        CholQRLinopsBase(const CholQRLinopsBase&) = delete;
        CholQRLinopsBase& operator=(const CholQRLinopsBase&) = delete;

    protected:
        CholQRLinopsBase(bool time_subroutines, bool enable_test_mode)
            : timing(time_subroutines), test_mode(enable_test_mode) {}

        ~CholQRLinopsBase() { delete[] Q; }

        void reset_outputs() {
            times.clear();
            n_chol_retries = 0;
            chol_status    = 0;
            std::fill(chol_applied_shifts, chol_applied_shifts + NumPasses, T(0));
            std::fill(chol_gram_traces, chol_gram_traces + NumPasses, T(0));
        }

        CholQROptions<T> chol_options() const {
            CholQROptions<T> opts;
            opts.max_retries     = max_retries;
            opts.shift_growth    = shift_growth;
            opts.symmetrize_gram = symmetrize_gram;
            return opts;
        }

        void store_records(const CholQRShiftRecord<T>* records) {
            n_chol_retries = 0;
            for (int p = 0; p < NumPasses; ++p) {
                n_chol_retries        += records[p].n_retries;
                chol_applied_shifts[p] = records[p].applied_shift;
                chol_gram_traces[p]    = records[p].gram_trace;
            }
        }

        /// Q := A R^{-1} (form_q_from_r), owned by the driver. Returns its wall-clock in us.
        template <RandLAPACK::linops::LinearOperator GLO>
        long form_test_Q(GLO& A, const T* R, int64_t ldr) {
            auto t0 = std::chrono::steady_clock::now();
            std::unique_ptr<T[]> Q_buf(new T[A.n_rows * A.n_cols]);
            form_q_from_r<T>(A, R, ldr, block_size, Q_buf.get());
            delete[] Q;
            Q = Q_buf.release();
            Q_rows = A.n_rows;
            Q_cols = A.n_cols;
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count();
        }

        /// The call() of CholQR_linops, CholQR2_linops and sCholQR3_linops: resets the outputs,
        /// runs NumPasses passes (pass 1 with shift factor sf1, later passes with sf_rest),
        /// stores the shift records and status, forms the test-mode Q and, when timing, fills
        /// times in the layout each driver documents. Returns cholqr_iterate's code.
        template <RandLAPACK::linops::LinearOperator GLO>
        int run_passes(GLO& A, T* R, int64_t ldr, T sf1, T sf_rest, bool gram_left_gemm) {
            reset_outputs();
            auto t0 = std::chrono::steady_clock::now();
            long it[5 * NumPasses] = {0};
            CholQRShiftRecord<T> records[NumPasses];
            CholQROptions<T> opts = chol_options();
            opts.gram_left_gemm = gram_left_gemm;
            int info = cholqr_iterate<T, GLO>(A, R, ldr, block_size, NumPasses, sf1, sf_rest, opts,
                                              timing, timing ? it : nullptr, records, &chol_status);
            store_records(records);
            if (info != 0) return info;

            const long q_us = test_mode ? form_test_Q(A, R, ldr) : 0;
            if (timing) {
                const long total = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0).count() - q_us;
                long sum = 0;
                for (long t : it) sum += t;
                times = {0L, it[0], it[1], it[3]};   // pass 1's left is not recorded
                if (NumPasses > 1) {
                    times.push_back(it[4]);
                    times.insert(times.end(), it + 5, it + 5 * NumPasses);
                    times.push_back(q_us);
                }
                times.push_back(total - sum);
                times.push_back(total);
            }
            return 0;
        }
};

} // namespace RandLAPACK
