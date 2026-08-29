#pragma once

#include "rl_exceptions.hh"
#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"
#include "rl_bqrrp.hh"
#include "rl_linops.hh"
#include "../comps/rl_cholqr.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <limits>

using namespace std::chrono;

namespace RandLAPACK {

/// Backwards-compatible alias for the precond-method enum, which now lives in
/// comps/rl_cholqr.hh as PCholQRPrecondMethod (shared across CholQR/sCholQR3/CQRRT).
using CQRRTLinopPrecond = PCholQRPrecondMethod;


// ============================================================================
// CQRRT: dense Q-less Cholesky QR with sketch preconditioning.
// ============================================================================
//
// Operates on a dense column-major A (m × n).  Modifies A in place via TRSM
// (A := A * R_sk^{-1}) so the Q factor is materialized implicitly inside A.
// This is faster than the linop variant when A is already dense in memory.
//
// Reference: arXiv:2111.11148.
template <typename T, typename RNG>
class CQRRTalg {
    public:

        virtual ~CQRRTalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG = RandBLAS::DefaultRNG>
class CQRRT : public CQRRTalg<T, RNG> {
    public:

        CQRRT(
            bool time_subroutines,
            T ep
        ) {
            timing = time_subroutines;
            eps = ep;
            orthogonalization = false;
            compute_Q = true;
            nnz = 4;              // SASO nonzeros per column; paper uses 4 or 8 (2 causes sporadic spikes)
            max_retries  = -1;    // unbounded retries (no ceiling), as CQRRT_linops
            shift_growth = T(10);
        }

        /// Computes an unpivoted QR factorization of the form:
        ///     A= QR,
        /// where Q and R are of size m-by-n and n-by-n.
        ///
        /// @note This algorithm expects A to be full-rank (rank = n). Rank-deficient inputs may result
        ///       in loss of orthogonality in the Q-factor and numerical instability in the R-factor.
        ///
        /// @return 0 on success; 1 if the preconditioner's diagonal is singular or the
        ///         preconditioned-Gram Cholesky (potrf) failed on every shift-retry
        ///         attempt (see stderr for which); -2 if shift_growth is invalid
        ///         (<= 1, which defeats the retry's geometric-growth termination
        ///         argument), a caller-configuration bug, checked before any work,
        ///         same sentinel cholqr_primitive uses for its own invalid-shift
        ///         inputs. Dense CQRRT has no caller-exposed shift_factor to validate
        ///         separately: the first attempt always starts unshifted, same as
        ///         CQRRT_linops, so shift_growth is the only invalid-shift input here.
        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        bool timing;
        T eps;

        // 10 entries: saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total.
        // NOT index-compatible with CQRRT_linops's 11-entry layout (that one starts
        // with an alloc slot); a plotter must dispatch on the layout, not assume the
        // indices line up.
        std::vector<long> times;
        /// Total measured wall-clock (microseconds) of the last call(), or -1 if timing
        /// was off. Every driver in this family packs the total as the LAST times[] entry,
        /// but the entry COUNT differs per driver (6 / 11 / 15 / 18). Callers used to hard-
        /// code that index (times[5], times[10], times[14], times[17]), so adding or
        /// removing one slot silently wrote the wrong number into every CSV with no compile
        /// error. Read the total through here instead.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        int64_t nnz;
        bool orthogonalization;
        bool compute_Q;   // skip Q materialization when false (R-only mode)

        // Adaptive-shift safety net on the preconditioned Gram's Cholesky, matching
        // CQRRT_linops: the first attempt is always unshifted; only on potrf
        // breakdown does the retry seed the shift at eps*trace(G) and grow it
        // x shift_growth. max_retries < 0 = unbounded (retry until PD). The clean
        // (unshifted, first-attempt-succeeds) path is unaffected.
        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;   ///< shift retries used on the last call (0 = clean)
        /// Shift record from the last call's preconditioned-Gram Cholesky: absolute
        /// shift the successful potrf carried (0 = unshifted) and the Gram's trace.
        T chol_applied_shifts[1] = {T(0)};
        T chol_gram_traces[1]    = {T(0)};
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRT<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* R,
    int64_t ldr,
    T d_factor,
    RandBLAS::RNGState<RNG> &state
){
    // Input parameter validation. Bad inputs would otherwise propagate to a
    // downstream BLAS/LAPACK failure or a segfault, the latter fatal when
    // CQRRT is called through a binding layer (e.g. MEX/MATLAB).
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
    randlapack_require(m >= n) << "CQRRT: operator must be tall (m=" << m << " < n=" << n << ")";
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (lda must be >= m for ColMajor)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n << " (ldr must be >= n)";
    randlapack_require(d_factor >= (T)1.0) << "d_factor=" << d_factor << " must be >= 1.0";
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
    randlapack_require(R != nullptr) << "CQRRT: R buffer is null";

    // Reset the shift-record out-params up front so a stale value from a
    // previous call never survives an early-return failure path.
    this->n_chol_retries = 0;
    this->chol_applied_shifts[0] = T(0);
    this->chol_gram_traces[0] = T(0);

    // Fail-fast shift-config validation, same argument as cholqr_primitive's
    // shift check (rl_cholqr.hh): shift_growth <= 1 defeats the retry loop's
    // geometric-growth termination argument (it would spin at a constant or
    // shrinking shift instead of escalating toward diagonal dominance). No
    // separate shift_factor check is needed here: unlike cholqr_primitive,
    // dense CQRRT does not expose a caller-supplied starting shift_factor
    // (the first attempt is always unshifted, matching CQRRT_linops), so
    // there is nothing else to validate before the retry loop below.
    if (this->shift_growth <= T(1)) {
        std::fprintf(stderr,
            "[CQRRT] FAIL: shift_growth (%g) must be > 1 (<= 1 defeats the "
            "geometric-growth retry termination argument)\n",
            (double)this->shift_growth);
        return -2;
    }

    ///--------------------TIMING VARS--------------------/
    steady_clock::time_point saso_t_start, saso_t_stop;
    steady_clock::time_point qr_t_start, qr_t_stop;
    steady_clock::time_point precond_t_start, precond_t_stop;
    steady_clock::time_point gram_t_start, gram_t_stop;
    steady_clock::time_point potrf_t_start, potrf_t_stop;
    steady_clock::time_point q_t_start, q_t_stop;
    steady_clock::time_point finalize_t_start, finalize_t_stop;
    steady_clock::time_point total_t_start, total_t_stop;
    long saso_t_dur = 0, qr_t_dur = 0, precond_t_dur = 0, gram_t_dur = 0;
    long potrf_t_dur = 0, q_t_dur = 0, finalize_t_dur = 0, total_t_dur = 0;

    if(this -> timing) total_t_start = steady_clock::now();

    int64_t d = (int64_t) (d_factor * (T) n);
    if (d < n) d = n;   // same clamp as CQRRT_linops: truncation must not undershoot n
    T* A_hat = new T[d * n]();
    T* tau   = new T[n]();

    // Sketch + small QR
    if(this -> timing) saso_t_start = steady_clock::now();
    RandBLAS::SparseDist DS(d, m, this->nnz);
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = S.next_state;
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                             d, n, m, (T)1.0, S, 0, 0, A, lda, (T)0.0, A_hat, d);
    if(this -> timing) { saso_t_stop = steady_clock::now(); qr_t_start = steady_clock::now(); }

    lapack::geqrf(d, n, A_hat, d, tau);
    if(this -> timing) qr_t_stop = steady_clock::now();

    T* R_sk = R;
    lapack::lacpy(MatrixType::Upper, n, n, A_hat, d, R_sk, ldr);
    // The caller's R buffer is otherwise uninitialized; the trmm below reads all
    // of R_sk (not just its upper triangle), so a garbage strict lower would
    // contaminate the output R's upper triangle (mirrors cholqr_primitive's own
    // lacpy+laset pattern at rl_cholqr.hh).
    if (n > 1)
        lapack::laset(MatrixType::Lower, n - 1, n - 1, T(0), T(0), R_sk + 1, ldr);

    if(this -> timing) precond_t_start = steady_clock::now();
    if (!RandLAPACK::util::diag_is_nonzero(n, R_sk, ldr)) {
        delete[] A_hat; delete[] tau; return 1;
    }
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
               m, n, 1.0, R_sk, ldr, A, lda);
    if(this -> timing) { precond_t_stop = steady_clock::now(); gram_t_start = steady_clock::now(); }

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, lda, 0.0, R_sk, ldr);
    if(this -> timing) { gram_t_stop = steady_clock::now(); potrf_t_start = steady_clock::now(); }

    // Adaptive-shift retry (same policy as cholqr_primitive's Step 3, parity with
    // CQRRT_linops): the clean path (unshifted potrf succeeds first try) is still
    // bit-identical to before, but is NOT free of extra work. With the default
    // max_retries = -1 (unbounded), gram_backup below is allocated and filled via
    // lacpy on every call, clean or not; that O(n^2) alloc+copy is timed into the
    // potrf slot even when no retry ever happens. Only max_retries == 0 skips it.
    T* gram_backup = (this->max_retries != 0) ? new T[n * n] : nullptr;
    T trace_G = 0;
    for (int64_t i = 0; i < n; ++i) trace_G += R_sk[i * (ldr + 1)];
    this->chol_gram_traces[0] = trace_G;
    if (gram_backup) lapack::lacpy(MatrixType::Upper, n, n, R_sk, ldr, gram_backup, n);

    int potrf_info = 0;
    int attempt = 0;
    T current_shift_factor = T(0);
    T last_shift = T(0);
    constexpr int kUnboundedRetryCeiling = 128;
    for (; (this->max_retries < 0) ? (attempt < kUnboundedRetryCeiling) : (attempt <= this->max_retries); ++attempt) {
        if (attempt > 0) {
            lapack::lacpy(MatrixType::Upper, n, n, gram_backup, n, R_sk, ldr);
            current_shift_factor = (current_shift_factor > T(0))
                                 ? current_shift_factor * this->shift_growth
                                 : std::numeric_limits<T>::epsilon();
            if (!std::isfinite(current_shift_factor) || !std::isfinite(trace_G)) {
                potrf_info = -1;
                break;
            }
        }
        if (current_shift_factor > T(0)) {
            T shift = current_shift_factor * trace_G;
            if (!std::isfinite(shift)) {
                potrf_info = -1;
                break;
            }
            for (int64_t i = 0; i < n; ++i) R_sk[i * (ldr + 1)] += shift;
            last_shift = shift;
        }
        potrf_info = lapack::potrf(Uplo::Upper, n, R_sk, ldr);
        if (potrf_info == 0) break;
    }
    this->n_chol_retries = (potrf_info == 0) ? attempt : (attempt > 0 ? attempt - 1 : 0);
    this->chol_applied_shifts[0] = last_shift;
    delete[] gram_backup;
    if (potrf_info) {
        delete[] A_hat; delete[] tau; return 1;
    }
    if(this -> timing) potrf_t_stop = steady_clock::now();

    if (this->compute_Q) {
        if(this -> timing) q_t_start = steady_clock::now();
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   m, n, 1.0, R_sk, ldr, A, lda);
        if(this -> timing) q_t_stop = steady_clock::now();
    }

    if(this -> timing) finalize_t_start = steady_clock::now();
    if (!this->orthogonalization) {
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   n, n, 1.0, A_hat, d, R_sk, ldr);
    }
    if(this -> timing) finalize_t_stop = steady_clock::now();

    if(this -> timing) {
        total_t_stop = steady_clock::now();
        saso_t_dur     = duration_cast<microseconds>(saso_t_stop     - saso_t_start).count();
        qr_t_dur       = duration_cast<microseconds>(qr_t_stop       - qr_t_start).count();
        precond_t_dur  = duration_cast<microseconds>(precond_t_stop  - precond_t_start).count();
        gram_t_dur     = duration_cast<microseconds>(gram_t_stop     - gram_t_start).count();
        potrf_t_dur    = duration_cast<microseconds>(potrf_t_stop    - potrf_t_start).count();
        finalize_t_dur = duration_cast<microseconds>(finalize_t_stop - finalize_t_start).count();
        total_t_dur    = duration_cast<microseconds>(total_t_stop    - total_t_start).count();
        if (this->compute_Q) {
            q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
            total_t_dur -= q_t_dur;
        }
        long t_rest = total_t_dur - (saso_t_dur + qr_t_dur + precond_t_dur +
                                      gram_t_dur + potrf_t_dur + finalize_t_dur);
        this -> times = {saso_t_dur, qr_t_dur, 0L, precond_t_dur,
                         gram_t_dur, 0L, potrf_t_dur, finalize_t_dur,
                         t_rest, total_t_dur};
    }

    delete[] A_hat;
    delete[] tau;
    return 0;
}


// ============================================================================
// CQRRT_linops: sketch-preconditioned Q-less Cholesky QR for abstract operators
// ============================================================================
//
// Algorithm 4 from the collaborator's spec.  Cannot modify the operator in place,
// so it forms R_sk explicitly and delegates to cholqr_primitive (which handles
// the precondition-inversion strategy via PCholQRPrecondMethod).
//
template <typename T, typename RNG = RandBLAS::DefaultRNG>
class CQRRT_linops {
    public:

        bool timing;
        bool test_mode;

        // Q-factor for test mode (only allocated if test_mode = true)
        T* Q;
        int64_t Q_rows;
        int64_t Q_cols;

        // 11 entries (preserved for matlab plotter compatibility):
        // [0] alloc, [1] sketch, [2] qr, [3] tri_inv, [4] fwd, [5] adj, [6] trsm_gram,
        // [7] chol, [8] finalize, [9] rest, [10] total
        std::vector<long> times;
        /// Total measured wall-clock (microseconds) of the last call(), or -1 if timing
        /// was off. Every driver in this family packs the total as the LAST times[] entry,
        /// but the entry COUNT differs per driver (6 / 11 / 15 / 18). Callers used to hard-
        /// code that index (times[5], times[10], times[14], times[17]), so adding or
        /// removing one slot silently wrote the wrong number into every CSV with no compile
        /// error. Read the total through here instead.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        int64_t nnz;
        CQRRTLinopPrecond precond_method;
        T bqrrp_block_ratio;
        int64_t block_size;

        // Adaptive-shift safety net (same as CholQR/CholQR2): the preconditioned
        // Gram Cholesky's first attempt is always unshifted; only if potrf breaks
        // down does the primitive seed the shift at eps*trace(G) and grow it x
        // shift_growth. max_retries < 0 = unbounded, retry until PD. This lets
        // CQRRT survive an ill-conditioned (e.g. single-precision) Gram instead
        // of failing outright, matching the CholQR family.
        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;   ///< shift retries used on the last call (0 = clean)
        /// Shift record from the last call's preconditioned-Gram Cholesky: absolute
        /// shift the successful potrf carried (0 = unshifted) and the Gram's trace.
        T chol_applied_shifts[1] = {T(0)};
        T chol_gram_traces[1]    = {T(0)};

        CQRRT_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            (void)ep;   // kept in the signature for call-site compatibility; unused
            nnz = 4;    // SASO nonzeros per column; paper uses 4 or 8 (2 causes sporadic spikes)
            block_size = kDefaultGramBlockSize;
            precond_method = PCholQRPrecondMethod::TRSM_IDENTITY;
            bqrrp_block_ratio = (T)1.0;
            max_retries  = -1;       // unbounded retries (no ceiling), as CholQR/CholQR2
            shift_growth = T(10);
            test_mode = enable_test_mode;
            Q = nullptr;
            Q_rows = 0;
            Q_cols = 0;
        }

        ~CQRRT_linops() {
            if (Q != nullptr) {
                delete[] Q;
            }
        }

        template <RandLAPACK::linops::LinearOperator GLO>
        int call(
            GLO& A,
            T* R,
            int64_t ldr,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) {
            steady_clock::time_point t0, t1, total_t_start, total_t_stop;
            long alloc_dur = 0, saso_dur = 0, qr_dur = 0;
            long precond_inv_dur = 0, fwd_dur = 0, adj_dur = 0, gemm_dur = 0;
            long chol_dur = 0, finalize_dur = 0, q_dur = 0, total_dur = 0;

            if (this->timing) total_t_start = steady_clock::now();

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            // Input validation: d_factor < 1 gives d < n, which reads out of bounds
            // in the lacpy of the upper n x n block below.
            randlapack_require(m >= n) << "CQRRT_linops: operator must be tall (m=" << m << " < n=" << n << ")";
            randlapack_require(n >= 1) << "CQRRT_linops: n must be >= 1";
            randlapack_require(d_factor >= (T)1.0) << "CQRRT_linops: d_factor=" << d_factor << " must be >= 1.0";
            randlapack_require(ldr >= n) << "CQRRT_linops: ldr=" << ldr << " < n=" << n;
            randlapack_require(R != nullptr) << "CQRRT_linops: R buffer is null";
            int64_t d = (int64_t)(d_factor * (T)n);
            if (d < n) d = n;
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // ---- Allocations, PHASED ----
            // The sketch phase and the Gram/Cholesky phase have disjoint working sets,
            // so allocating both up front sums their peaks. Phasing drops the peak
            // from (d*n + n + n^2) + (3n^2 + (m+n)*b_eff + n) to
            // max(d*n + n + n^2, 3n^2 + (m+n)*b_eff + n): sketch moment is
            // A_hat(d*n) + tau(n) + P(n*n); Gram moment is P + R_pre + G (3*n*n)
            // + A_temp(m*b_eff) + Z_buf(n*b_eff) + cholqr_primitive's O(n)
            // diag_backup. See cqrrt_linops_analytical_kb in rl_memory_tracker.hh,
            // the source of truth this comment must agree with.
            if (this->timing) t0 = steady_clock::now();
            T* A_hat  = new T[d * n];
            T* tau    = new T[n];
            T* P      = new T[n * n]();
            T* R_pre  = nullptr;   // Gram-phase buffers: allocated after the sketch QR,
            T* G      = nullptr;   // once A_hat and tau have been released.
            T* A_temp = nullptr;
            T* Z_buf  = nullptr;
            if (this->timing) { t1 = steady_clock::now(); alloc_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Step 1: Sketch M^sk = S * A ----
            // (Sparse SASO only.)
            if (this->timing) t0 = steady_clock::now();
            {
                RandBLAS::SparseDist DS(d, m, this->nnz);
                RandBLAS::SparseSkOp<T, RNG> S(DS, state);
                state = S.next_state;
                RandBLAS::fill_sparse(S);
                A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  d, n, m, (T)1.0, S, (T)0.0, A_hat, d);
            }
            if (this->timing) { t1 = steady_clock::now(); saso_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Step 2: [~, R^sk] = qr(M^sk) ----
            if (this->timing) t0 = steady_clock::now();
            lapack::geqrf(d, n, A_hat, d, tau);
            lapack::lacpy(MatrixType::Upper, n, n, A_hat, d, P, n);   // R^sk = upper(A_hat); P's lower stays 0
            // The sketch and its Householder scalars are dead here: CQRRT never forms
            // or applies the sketch Q, only R^sk, which now lives in P. Release them
            // BEFORE the Gram-phase buffers exist.
            delete[] A_hat; A_hat = nullptr;
            delete[] tau;   tau   = nullptr;
            if (this->timing) { t1 = steady_clock::now(); qr_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // Gram/Cholesky working set, allocated now that the sketch is gone.
            if (this->timing) t0 = steady_clock::now();
            R_pre  = new T[n * n]();
            G      = new T[n * n]();
            A_temp = new T[m * b_eff];
            Z_buf  = new T[n * b_eff];
            if (this->timing) { t1 = steady_clock::now(); alloc_dur += duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Step 3: PCholQR(A, P = R^sk) ----
            int info = cholqr_primitive<T, GLO, RNG>(
                A, P, R, ldr,
                this->precond_method,
                this->block_size,
                this->bqrrp_block_ratio,
                R_pre, G, A_temp, Z_buf,
                &state,
                precond_inv_dur, fwd_dur, adj_dur, gemm_dur, chol_dur, finalize_dur,
                this->timing,
                /*shift_factor=*/T(0), this->max_retries, this->shift_growth, &this->n_chol_retries,
                &this->chol_applied_shifts[0], &this->chol_gram_traces[0]);
            if (info != 0) {
                // cholqr_primitive's status (retry exhaustion, singular
                // preconditioner, non-finite shift) is flattened to 1 here;
                // see stderr for the specific cause.
                delete[] A_hat; delete[] tau; delete[] P; delete[] R_pre;
                delete[] G; delete[] A_temp; delete[] Z_buf;
                return 1;
            }

            // ---- Test mode: materialize Q = A * R^{-1} ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();

                T* Q_buf = new T[m * n];
                RandLAPACK::materialize_Q_from_R(A, R, ldr, m, n, b_eff, Q_buf, m);
                this->Q_rows = m;
                this->Q_cols = n;
                // The class owns Q (the destructor frees it), so release any buffer from a
                // previous call() before taking ownership of this one.
                delete[] this->Q;
                this->Q = Q_buf;

                if (this->timing) { t1 = steady_clock::now(); q_dur = duration_cast<microseconds>(t1 - t0).count(); }
            }

            // ---- Finalize timing ----
            if (this->timing) {
                total_t_stop = steady_clock::now();
                total_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                total_dur -= q_dur;

                long rest_dur = total_dur - (alloc_dur + saso_dur + qr_dur + precond_inv_dur
                                            + fwd_dur + adj_dur + gemm_dur + chol_dur + finalize_dur);

                this->times = {alloc_dur, saso_dur, qr_dur, precond_inv_dur,
                               fwd_dur, adj_dur, gemm_dur,
                               chol_dur, finalize_dur, rest_dur, total_dur};
            }

            delete[] A_hat;
            delete[] tau;
            delete[] P;
            delete[] R_pre;
            delete[] G;
            delete[] A_temp;
            delete[] Z_buf;
            return 0;
        }
};

} // end namespace RandLAPACK
