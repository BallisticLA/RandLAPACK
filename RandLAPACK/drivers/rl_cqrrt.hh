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
#include <vector>
#include <chrono>
#include <numeric>

using namespace std::chrono;

namespace RandLAPACK {

/// Backwards-compatible alias for the precond-method enum, which now lives in
/// comps/rl_cholqr.hh as PCholQRPrecondMethod (shared across CholQR/sCholQR3/CQRRT).
using CQRRTLinopPrecond = PCholQRPrecondMethod;


// ============================================================================
// CQRRT — dense Q-less Cholesky QR with sketch preconditioning.
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
            nnz = 2;
        }

        /// Computes an unpivoted QR factorization of the form:
        ///     A= QR,
        /// where Q and R are of size m-by-n and n-by-n.
        ///
        /// @note This algorithm expects A to be full-rank (rank = n). Rank-deficient inputs may result
        ///       in loss of orthogonality in the Q-factor and numerical instability in the R-factor.
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

        // 10 entries: saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total
        // Matches CQRRT_linops timing indices for direct comparison.
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
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (lda must be >= m for ColMajor)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n << " (ldr must be >= n)";
    randlapack_require(d_factor >= (T)1.0) << "d_factor=" << d_factor << " must be >= 1.0";
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
    long saso_t_dur = 0, qr_t_dur = 0, precond_t_dur = 0, gram_t_dur = 0;
    long potrf_t_dur = 0, q_t_dur = 0, finalize_t_dur = 0, total_t_dur = 0;

    if(this -> timing) total_t_start = steady_clock::now();

    int64_t d = d_factor * n;
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

    if(this -> timing) precond_t_start = steady_clock::now();
    if (!RandLAPACK::util::diag_is_nonzero(n, R_sk, ldr)) {
        delete[] A_hat; delete[] tau; return 1;
    }
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
               m, n, 1.0, R_sk, ldr, A, lda);
    if(this -> timing) { precond_t_stop = steady_clock::now(); gram_t_start = steady_clock::now(); }

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, lda, 0.0, R_sk, ldr);
    if(this -> timing) { gram_t_stop = steady_clock::now(); potrf_t_start = steady_clock::now(); }

    if (lapack::potrf(Uplo::Upper, n, R_sk, ldr)) {
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
// CQRRT_linops — sketch-preconditioned Q-less Cholesky QR for abstract operators
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
        T eps;

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
        bool use_dense_sketch;
        CQRRTLinopPrecond precond_method;
        T bqrrp_block_ratio;
        int64_t block_size;

        // Adaptive-shift safety net (Oleg's prescription; same as CholQR/CholQR2):
        // the preconditioned Gram Cholesky's first attempt is always unshifted; only
        // if potrf breaks down does the primitive seed the shift at eps*trace(G) and
        // grow it x shift_growth. max_retries < 0 = unbounded — retry until PD. This
        // lets CQRRT survive an ill-conditioned (e.g. single-precision) Gram instead
        // of failing outright, matching the CholQR family.
        int max_retries;
        T   shift_growth;
        int n_chol_retries = 0;   ///< shift retries used on the last call (0 = clean)

        CQRRT_linops(
            bool time_subroutines,
            T ep,
            bool enable_test_mode = false
        ) {
            timing = time_subroutines;
            eps = ep;
            nnz = 2;
            use_dense_sketch = false;
            block_size = 0;
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
            int64_t d = (int64_t)(d_factor * (T)n);
            int64_t b_eff = (this->block_size > 0 && this->block_size < n)
                          ? this->block_size : n;

            // ---- Allocations ----
            if (this->timing) t0 = steady_clock::now();
            T* A_hat  = new T[d * n];
            T* tau    = new T[n];
            T* P      = new T[n * n]();
            T* R_pre  = new T[n * n]();
            T* G      = new T[n * n]();
            T* A_temp = new T[m * b_eff];
            T* Z_buf  = new T[n * b_eff];
            if (this->timing) { t1 = steady_clock::now(); alloc_dur = duration_cast<microseconds>(t1 - t0).count(); }

            // ---- Step 1: Sketch M^sk = S * A ----
            if (this->timing) t0 = steady_clock::now();
            if (this->use_dense_sketch) {
                RandBLAS::DenseDist DD(d, m);
                RandBLAS::DenseSkOp<T, RNG> S(DD, state);
                state = S.next_state;
                RandBLAS::fill_dense(S);
                A(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                  d, n, m, (T)1.0, S, (T)0.0, A_hat, d);
            } else {
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
            if (this->timing) { t1 = steady_clock::now(); qr_dur = duration_cast<microseconds>(t1 - t0).count(); }

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
                /*shift_factor=*/T(0), this->max_retries, this->shift_growth, &this->n_chol_retries);
            if (info != 0) {
                delete[] A_hat; delete[] tau; delete[] P; delete[] R_pre;
                delete[] G; delete[] A_temp; delete[] Z_buf;
                return 1;
            }

            // ---- Test mode: materialize Q = A * R^{-1} ----
            if (this->test_mode) {
                if (this->timing) t0 = steady_clock::now();

                T* Q_buf = new T[m * n];
                RandLAPACK::materialize_Q_from_R(A, R, ldr, m, n, b_eff, Q_buf);
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
