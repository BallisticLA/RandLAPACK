#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"
#include "rl_bqrrp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>

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
        /// @param[in] d
        ///     Embedding dimension of a sketch, m >= d >= n.
        ///
        /// @param[in] R
        ///     Represents the upper-triangular R factor of QR factorization.
        ///     On entry, is empty and may not have any space allocated for it.
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

        // 10 entries: saso, qr, trtri(=0), precond, gram, trmm_gram(=0), potrf, finalize, rest, total
        // Matches CQRRT_linops timing indices for direct comparison.
        std::vector<long> times;

        // tuning SASOS
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

    int64_t d = d_factor * n;

    T* A_hat = new T[d * n]();
    T* tau   = new T[n]();

    if(this -> timing)
        saso_t_start = steady_clock::now();
    
    /// Generating a SASO
    RandBLAS::SparseDist DS(d, m, this->nnz);
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

    /// Extracting a k by k R representation
    T* R_sk  = R;
    lapack::lacpy(MatrixType::Upper, n, n, A_hat, d, R_sk, ldr);

    if(this -> timing)
        precond_t_start = steady_clock::now();

    // Precondition: A := A * R_sk^{-1}
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sk, ldr, A, lda);

    if(this -> timing) {
        precond_t_stop = steady_clock::now();
        gram_t_start = steady_clock::now();
    }

    // Gram matrix: G = A^T * A (SYRK, upper triangle only)
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, lda, 0.0, R_sk, ldr);

    if(this -> timing) {
        gram_t_stop = steady_clock::now();
        potrf_t_start = steady_clock::now();
    }

    // Cholesky factorization
    int msg = lapack::potrf(Uplo::Upper, n, R_sk, ldr);

    if(this -> timing)
        potrf_t_stop = steady_clock::now();

    // Obtain the output Q-factor (only if requested, timed separately and excluded from total)
    if (this->compute_Q) {
        if(this -> timing)
            q_t_start = steady_clock::now();

        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sk, ldr, A, lda);

        if(this -> timing)
            q_t_stop = steady_clock::now();
    }

    if(this -> timing)
        finalize_t_start = steady_clock::now();

    if (!this->orthogonalization) {
        // Get the final R-factor - undoing the preconditioning
        // R := R_chol * R_sk (returned by QR on sketch)
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, n, n, 1.0, A_hat, d, R_sk, ldr);
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

        // Subtract Q-factor computation time (excluded from total, matching CQRRT_linops test_mode)
        if (this->compute_Q) {
            q_t_dur = duration_cast<microseconds>(q_t_stop - q_t_start).count();
            total_t_dur -= q_t_dur;
        }

        long t_rest = total_t_dur - (saso_t_dur + qr_t_dur + precond_t_dur +
                                      gram_t_dur + potrf_t_dur + finalize_t_dur);

        // Fill the data vector (10 entries, matching CQRRT_linops indices)
        // Index: 0=saso, 1=qr, 2=trtri(=0), 3=precond, 4=gram, 5=trmm_gram(=0), 6=potrf, 7=finalize, 8=rest, 9=total
        this -> times = {saso_t_dur, qr_t_dur, 0L, precond_t_dur,
                         gram_t_dur, 0L, potrf_t_dur, finalize_t_dur,
                         t_rest, total_t_dur};
    }

    delete[] A_hat;
    delete[] tau;

    return 0;
}
} // end namespace RandLAPACK
