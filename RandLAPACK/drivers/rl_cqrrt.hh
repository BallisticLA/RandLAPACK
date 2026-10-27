#pragma once

#include "rl_exceptions.hh"
#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_cholqr.hh"

#include <RandBLAS.hh>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <chrono>

using namespace std::chrono;

namespace RandLAPACK {

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
            nnz = 2;
        }

        /// Computes an unpivoted QR factorization of the form:
        ///     A= QR,
        /// where Q and R are of size m-by-n and n-by-n.
        /// Detailed description of this algorithm may be found in https://arxiv.org/pdf/2111.11148.
        ///
        /// @note This algorithm expects A to be full-rank (rank = n). Rank-deficient inputs may result
        ///       in loss of orthogonality in the Q-factor and numerical instability in the R-factor.
        ///
        /// @param[in] m
        ///     The number of rows in the matrix A.
        ///
        /// @param[in] n
        ///     The number of columns in the matrix A.
        ///
        /// @param[in] A
        ///     The m-by-n matrix A, stored in a column-major format.
        ///
        /// @param[in] d_factor
        ///     Sketch size factor, finite and >= 1: the sketch has d = d_factor * n rows, the
        ///     product rounded in T and truncated, at least n.
        ///
        /// @param[in] R
        ///     n-by-n buffer with leading dimension ldr; its contents on entry are ignored.
        ///
        /// @param[in] state
        ///     RNG state parameter, required for sketching operator generation.
        ///
        /// @param[out] A
        ///     Overwritten by an m-by-n orthogonal Q factor.
        ///     Matrix is stored explicitly.
        ///
        /// @param[out] R
        ///     Stores n-by-n matrix with upper-triangular R factor.
        ///     Zero entries are not compressed.
        ///
        /// @return = 0: successful exit
        ///         = 1: the sketch's R factor is singular or the Gram's Cholesky failed (a non-PD
        ///              pivot or a non-finite factor)
        ///

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

        // 10 entries, matching CQRRTO_linops slots 1-10: saso, qr, tri_inv (always 0), precond,
        // gram, left (always 0), potrf, finalize, rest, total. Cleared at the start of every call.
        std::vector<long> times;
        /// Total wall-clock (us) of the last call(), or -1 if it was untimed or failed.
        long total_us() const { return times.empty() ? -1L : times.back(); }

        // tuning SASOS: nonzeros per short-axis vector of the sketch, capped at min(d, m)
        int64_t nnz;

        // Mode of operation that allows to use CQRRT for orthogonalization of the input matrix.
        bool orthogonalization;

        // If false, skip the Q-factor computation (R-only mode).
        // When false, A is NOT overwritten with Q on output.
        bool compute_Q;
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
    this->times.clear();
    // Input parameter validation. Bad inputs would otherwise propagate to a
    // downstream BLAS/LAPACK failure or a segfault, the latter fatal when
    // CQRRT is called through a binding layer (e.g. MEX/MATLAB).
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 1) << "n=" << n << " must be >= 1";
    randlapack_require(m >= n) << "m=" << m << " must be >= n=" << n;
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (lda must be >= m for ColMajor)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n << " (ldr must be >= n)";
    randlapack_require(d_factor >= (T)1.0 && std::isfinite(d_factor)) << "d_factor=" << d_factor << " must be finite and >= 1.0";
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
    randlapack_require(!(R == nullptr && n > 0)) << "R buffer is null but n=" << n << " > 0";

    ///--------------------TIMING VARS--------------------/
    steady_clock::time_point saso_t_start, saso_t_stop;
    steady_clock::time_point qr_t_start, qr_t_stop;
    steady_clock::time_point precond_t_start, precond_t_stop;
    steady_clock::time_point gram_t_start, gram_t_stop;
    steady_clock::time_point potrf_t_start, potrf_t_stop;
    steady_clock::time_point q_t_start, q_t_stop;
    steady_clock::time_point finalize_t_start, finalize_t_stop;
    steady_clock::time_point total_t_start, total_t_stop;
    long saso_t_dur      = 0;
    long qr_t_dur        = 0;
    long precond_t_dur   = 0;
    long gram_t_dur      = 0;
    long potrf_t_dur     = 0;
    long q_t_dur         = 0;
    long finalize_t_dur  = 0;
    long total_t_dur     = 0;

    if(this -> timing)
        total_t_start = steady_clock::now();

    int64_t d = cqrrt_sketch_rows(d_factor, n);

    T* A_hat = new T[d * n]();
    T* tau   = new T[n]();

    if(this -> timing)
        saso_t_start = steady_clock::now();

    /// Generating a SASO. RandBLAS requires nnz <= min(d, m); a single-column input truncates
    /// d to 1 for any d_factor < 2, below the default nnz = 2.
    RandBLAS::SparseDist DS(d, m, std::min({this->nnz, d, m}));
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = S.next_state;

    /// Applying a SASO
    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, (T) 1.0, S, 0, 0, A, lda, (T) 0.0, A_hat, d
    );

    if(this -> timing) {
        saso_t_stop = steady_clock::now();
        qr_t_start = steady_clock::now();
    }

    /// Performing QR on a sketch
    lapack::geqrf(d, n, A_hat, d, tau);

    if(this -> timing)
        qr_t_stop = steady_clock::now();

    /// Extracting a k by k R representation. The final trmm reads all of R_sk, so its strict
    /// lower triangle (the caller's buffer) must be zero.
    T* R_sk  = R;
    lapack::lacpy(MatrixType::Upper, n, n, A_hat, d, R_sk, ldr);
    RandLAPACK::util::get_U(n, n, R_sk, ldr);

    if(this -> timing)
        precond_t_start = steady_clock::now();

    // Precondition: A := A * R_sk^{-1}
    if (!RandLAPACK::util::diag_is_nonzero(n, R_sk, ldr)) {
        delete[] A_hat;
        delete[] tau;
        return 1;
    }
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, R_sk, ldr, A, lda);

    if(this -> timing) {
        precond_t_stop = steady_clock::now();
        gram_t_start = steady_clock::now();
    }

    // Gram matrix: G = A^T * A (SYRK, upper triangle only)
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, (T) 1.0, A, lda, (T) 0.0, R_sk, ldr);

    if(this -> timing) {
        gram_t_stop = steady_clock::now();
        potrf_t_start = steady_clock::now();
    }

    // Cholesky factorization
    if (potrf_upper_finite(n, R_sk, ldr)) {
        delete[] A_hat;
        delete[] tau;
        return 1;
    }

    if(this -> timing)
        potrf_t_stop = steady_clock::now();

    // Obtain the output Q-factor (only if requested, timed separately and excluded from total)
    if (this->compute_Q) {
        if(this -> timing)
            q_t_start = steady_clock::now();

        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, (T) 1.0, R_sk, ldr, A, lda);

        if(this -> timing)
            q_t_stop = steady_clock::now();
    }

    if(this -> timing)
        finalize_t_start = steady_clock::now();

    if (!this->orthogonalization) {
        // Get the final R-factor - undoing the preconditioning
        // R := R_chol * R_sk (returned by QR on sketch)
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, (T) 1.0, A_hat, d, R_sk, ldr);
    }

    if(this -> timing) {
        finalize_t_stop = steady_clock::now();

        total_t_stop = steady_clock::now();

        saso_t_dur     = duration_cast<microseconds>(saso_t_stop     - saso_t_start).count();
        qr_t_dur       = duration_cast<microseconds>(qr_t_stop       - qr_t_start).count();
        precond_t_dur  = duration_cast<microseconds>(precond_t_stop  - precond_t_start).count();
        gram_t_dur     = duration_cast<microseconds>(gram_t_stop     - gram_t_start).count();
        potrf_t_dur    = duration_cast<microseconds>(potrf_t_stop    - potrf_t_start).count();
        finalize_t_dur = duration_cast<microseconds>(finalize_t_stop - finalize_t_start).count();

        total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

        // Subtract Q-factor computation time (excluded from total)
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

/// Sketch-preconditioned Q-less Cholesky QR (CQRRTO) for abstract linear operators.
///
///   1. Sketch S A (d x n, d = d_factor * n) with a sparse sign sketch S;
///   2. P = R factor of a Householder QR of S A;
///   3. one CholQR pass on A preconditioned on P (cholqr_primitive), giving R = R^chol P.
///
/// A is used only through A * B, A^T * B and its sketch overload. One max(d, 2n) x n buffer
/// holds S A, then P (its R factor, moved to the front) and the Gram G, so the Gram phase adds
/// no n x n beyond the sketch's; the sketch operator and tau are freed before the Gram's other
/// scratch is allocated. With test_mode on, Q = A R^{-1} is formed for verification,
/// from A's columns (see form_q_from_r), outside the timed region. A potrf breakdown is
/// retried with the adaptive shift (see cholqr_primitive), after which R factors a shifted Gram.
///
/// call() returns 0 on success, or 1 when the factorization fails: R and Q are then untouched,
/// times is empty, chol_status holds cholqr_primitive's code (a positive potrf info,
/// kCholQRNonFinite, kCholQRInvalidShift or kCholQRSingularPreconditioner), and n_chol_retries,
/// chol_applied_shifts and chol_gram_traces hold what the failed attempt recorded (a non-finite
/// Gram leaves its trace non-finite). shift_growth <= 1 or NaN fails this way with
/// kCholQRInvalidShift before the sketch is drawn, so state is untouched. call() throws
/// RandLAPACK::Error on m < n, n < 1, ldr < n, a null R, nnz < 1, a d_factor that is not
/// finite and >= 1, or (with bqrrp) a bqrrp_block_ratio outside (0, 1], also before the sketch:
/// R, Q and state are then untouched, and times, chol_status and the shift record are reset.
/// Q is formed only in test mode.
///
/// times (11 entries): alloc, sketch, qr, tri_inv, fwd, adj, left (the Gram's left factor, by
/// TRSM or GEMMs, plus the symmetrization), chol, finalize, rest, total.
template <typename T, typename RNG = RandBLAS::DefaultRNG>
class CQRRTO_linops : public CholQRLinopsBase<T, 1> {
    public:
        int64_t nnz = 4;   ///< nonzeros per short-axis vector of the sketch, capped at min(d, m)
        /// How P is inverted (see CholQRPrecondMethod); bqrrp draws from state, and its rank
        /// truncation of a P whose diagonal spans more than about 1/eps fails with
        /// kCholQRSingularPreconditioner.
        CholQRPrecondMethod precond_method = CholQRPrecondMethod::trsm_identity;
        T bqrrp_block_ratio = T(1);   ///< in (0, 1]; 1.0 selects an n-dependent default (see
                                      ///< cholqr_primitive)
        bool gram_left_gemm = false;  ///< trsm_identity, trtri: apply R_pre^T by GEMM per block
                                      ///< instead of one TRSM (see gram_left_by_gemm)

        /// @param ep  unused; kept for call-site compatibility with the other QR drivers.
        CQRRTO_linops(bool time_subroutines, T ep, bool enable_test_mode = false)
            : CholQRLinopsBase<T, 1>(time_subroutines, enable_test_mode) { (void)ep; }

        /// Computes the R factor of A = QR (Q not formed), A m x n with m >= n >= 1.
        /// @param[out]    R         n x n buffer; on exit the upper-triangular factor, zeros below.
        /// @param[in]     ldr       leading dimension of R.
        /// @param[in]     d_factor  sketch rows per column, >= 1.
        /// @param[in,out] state     RNG state of the sketch (and of bqrrp); advanced on exit.
        template <RandLAPACK::linops::LinearOperator GLO>
        int call(GLO& A, T* R, int64_t ldr, T d_factor, RandBLAS::RNGState<RNG>& state) {
            this->reset_outputs();
            steady_clock::time_point t0, total_t_start;
            if (this->timing) total_t_start = steady_clock::now();
            auto lap = [&](long& slot) {   // charge the time since the last lap to slot
                if (!this->timing) return;
                auto t1 = steady_clock::now();
                slot += duration_cast<microseconds>(t1 - t0).count();
                t0 = t1;
            };

            int64_t m = A.n_rows;
            int64_t n = A.n_cols;
            randlapack_require(m >= n) << "CQRRTO_linops: operator must be tall (m=" << m << " < n=" << n << ")";
            randlapack_require(n >= 1) << "CQRRTO_linops: n must be >= 1";
            randlapack_require(d_factor >= T(1) && std::isfinite(d_factor))
                << "CQRRTO_linops: d_factor=" << d_factor << " must be finite and >= 1";
            randlapack_require(ldr >= n) << "CQRRTO_linops: ldr=" << ldr << " < n=" << n;
            randlapack_require(R != nullptr) << "CQRRTO_linops: R is null (n=" << n << ")";
            randlapack_require(this->nnz >= 1) << "CQRRTO_linops: nnz=" << this->nnz << " must be >= 1";
            randlapack_require(this->precond_method != CholQRPrecondMethod::bqrrp
                               || (this->bqrrp_block_ratio > T(0) && this->bqrrp_block_ratio <= T(1)))
                << "CQRRTO_linops: bqrrp_block_ratio=" << this->bqrrp_block_ratio << " must lie in (0, 1]";
            if (!(this->shift_growth > T(1))) {
                this->chol_status = kCholQRInvalidShift;
                return 1;
            }
            int64_t d = cqrrt_sketch_rows(d_factor, n);
            int64_t b_eff = gram_block_width(this->block_size, n);
            CholQROptions<T> opts = this->chol_options();
            opts.gram_left_gemm = this->gram_left_gemm;
            const bool left_by_gemm = gram_left_by_gemm(this->precond_method, opts.gram_left_gemm);

            long alloc_dur = 0, saso_dur = 0, qr_dur = 0;
            long precond_inv_dur = 0, fwd_dur = 0, adj_dur = 0, left_dur = 0, chol_dur = 0, finalize_dur = 0;
            if (this->timing) t0 = steady_clock::now();

            CholQRShiftRecord<T> records[1];
            int info;
            {
                // S A (ld d), then P and G (ld n): P's strict lower triangle is never read.
                std::unique_ptr<T[]> buf(new T[std::max(d, 2 * n) * n]);
                T* P = buf.get();
                T* G = P + n * n;
                {   // Sketch phase: the sketch and tau are released before the Gram phase allocates.
                    std::unique_ptr<T[]> tau(new T[n]);
                    lap(alloc_dur);
                    {
                        RandBLAS::SparseDist DS(d, m, std::min({this->nnz, d, m}));
                        RandBLAS::SparseSkOp<T, RNG> S(DS, state);
                        state = S.next_state;
                        RandBLAS::fill_sparse(S);
                        A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                          d, n, m, (T)1.0, S, (T)0.0, buf.get(), d);
                    }
                    lap(saso_dur);
                    lapack::geqrf(d, n, buf.get(), d, tau.get());
                    // Move each column's R entries from ld d to ld n; destinations precede sources.
                    if (d != n)
                        for (int64_t j = 1; j < n; ++j)
                            std::copy(P + j * d, P + j * d + j + 1, P + j * n);
                    lap(qr_dur);
                }

                // Gram phase, freed with P and G before the test-mode Q is formed.
                std::unique_ptr<T[]> R_pre(forms_explicit_inverse(this->precond_method, opts.gram_left_gemm)
                                           ? new T[n * n] : nullptr);
                std::unique_ptr<T[]> A_temp(new T[m * b_eff]);
                std::unique_ptr<T[]> Z_buf(left_by_gemm ? new T[n * b_eff] : nullptr);
                lap(alloc_dur);

                info = cholqr_primitive<T, GLO, RNG>(
                    A, P, n, R, ldr, this->precond_method, this->block_size, this->bqrrp_block_ratio,
                    R_pre.get(), G, A_temp.get(), Z_buf.get(), &state,
                    precond_inv_dur, fwd_dur, adj_dur, left_dur, chol_dur, finalize_dur, this->timing,
                    /*shift_factor=*/T(0), opts, &records[0]);
            }
            this->store_records(records);
            this->chol_status = info;
            if (info != 0) return 1;

            long q_dur = this->test_mode ? this->form_test_Q(A, R, ldr) : 0;

            if (this->timing) {
                long total = duration_cast<microseconds>(steady_clock::now() - total_t_start).count() - q_dur;
                long rest = total - (alloc_dur + saso_dur + qr_dur + precond_inv_dur
                                     + fwd_dur + adj_dur + left_dur + chol_dur + finalize_dur);
                this->times = {alloc_dur, saso_dur, qr_dur, precond_inv_dur,
                               fwd_dur, adj_dur, left_dur, chol_dur, finalize_dur, rest, total};
            }
            return 0;
        }
};

/// Former name of CQRRTO_linops.
template <typename T, typename RNG = RandBLAS::DefaultRNG>
using CQRRT_linops [[deprecated("renamed CQRRTO_linops")]] = CQRRTO_linops<T, RNG>;

} // end namespace RandLAPACK
