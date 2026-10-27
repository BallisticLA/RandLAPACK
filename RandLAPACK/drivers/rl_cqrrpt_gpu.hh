#pragma once

#include "rl_exceptions.hh"
#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "lapack/device.hh"

#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace std::chrono;

namespace RandLAPACK {

template <typename T, typename RNG>
class CQRRPT_GPU_alg {
    public:

        virtual ~CQRRPT_GPU_alg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            int64_t* J,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class CQRRPT_GPU : public CQRRPT_GPU_alg<T, RNG> {
    public:

        /// The sketch and its QRCP run on the host; the rest of the algorithm runs on the device.
        /// The QRCP of the sketch is LAPACK's GEQP3 when 'no_hqrrp' is nonzero (the default), and HQRRP
        /// otherwise, tuned by 'nb_alg', 'oversampling', 'panel_pivoting' and 'use_cholqr'.
        /// The sparse sketching operator has 'nnz' nonzeros per column, 2 by default, as in CQRRPT.
        ///
        /// @param[in] verb
        ///     Stored in 'verbosity'; the algorithm prints nothing.
        ///
        /// @param[in] time_subroutines
        ///     If true, the call fills 'times' with the durations of its steps, in microseconds.
        ///
        /// @param[in] ep
        ///     The tolerance 'eps' of the a posteriori rank estimate after a failed Cholesky factorization.
        CQRRPT_GPU(
            bool verb,
            bool time_subroutines,
            T ep
        ) {
            verbosity = verb;
            timing = time_subroutines;
            eps = ep;
            no_hqrrp = 1;
            nb_alg = 64;
            oversampling = 10;
            use_cholqr = 0;
            panel_pivoting = 1;
            nnz = 2;
            rank = 0;
        }

        /// Computes a QR factorization with column pivots of the form:
        ///     A[:, J] = QR,
        /// where Q and R are of size m-by-k and k-by-n, with rank(A) = k.
        /// Detailed description of this algorithm may be found in Section 5.1.2.
        /// of "the RandLAPACK book". 
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
        ///     The sketch has d = d_factor * n rows, m >= d >= n.
        ///
        /// @param[in] R
        ///     Represents the upper-triangular R factor of QR factorization.
        ///     On entry, an n-by-n buffer with leading dimension ldr >= n; its contents are ignored.
        ///
        /// @param[out] A
        ///     Overwritten by an m-by-k orthogonal Q factor.
        ///     Matrix is stored explicitly.
        ///
        /// @param[out] R
        ///     Rows 0:rank hold the upper-trapezoidal R factor; all other entries
        ///     of the n-by-n block are zero.
        ///
        /// @param[out] J
        ///     Stores n integer type pivot index entries. For an empty A (m = 0 or n = 0),
        ///     J = 1, ..., n and the rank is 0.
        ///
        /// @return = 0: successful exit. If the Cholesky factorization fails, the rank
        ///     is re-estimated from the leading part of the factor, as in CQRRPT.
        ///

        int call(
            int64_t m,
            int64_t n,
            T* A,
            int64_t lda,
            T* R,
            int64_t ldr,
            int64_t* J,
            T d_factor,
            RandBLAS::RNGState<RNG> &state
        ) override;

    public:
        bool verbosity;
        bool timing;
        T eps;
        int64_t rank;

        // 8 entries
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // HQRRP-related
        int no_hqrrp;
        int64_t nb_alg;
        int64_t oversampling;
        int64_t panel_pivoting;
        int64_t use_cholqr;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRPT_GPU<T, RNG>::call(
    int64_t m,
    int64_t n,
    T* A,
    int64_t lda,
    T* R,
    int64_t ldr,
    int64_t* J,
    T d_factor,
    RandBLAS::RNGState<RNG> &state
){
    // Input parameter validation, as in CQRRPT.
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (lda must be >= m for ColMajor)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n << " (ldr must be >= n)";
    randlapack_require(d_factor >= (T)1.0) << "d_factor=" << d_factor << " must be >= 1.0";
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
    randlapack_require(!(R == nullptr && n > 0)) << "R buffer is null but n=" << n << " > 0";
    randlapack_require(!(J == nullptr && n > 0)) << "J buffer is null but n=" << n << " > 0";

    // R's contents on entry are ignored: the parts the algorithm does not write are read as zeros.
    lapack::laset(MatrixType::General, n, n, (T) 0.0, (T) 0.0, R, ldr);

    // An empty A has nothing to factor. Return before touching the device, with rank 0 and the
    // identity pivots that GEQP3 returns for an empty matrix, as BQRRP_GPU does.
    if (std::min(m, n) == 0) {
        this->rank = 0;
        std::iota(J, J + n, (int64_t) 1);
        return 0;
    }

    ///--------------------TIMING VARS--------------------/
    steady_clock::time_point saso_t_stop;
    steady_clock::time_point saso_t_start;
    steady_clock::time_point qrcp_t_start;
    steady_clock::time_point qrcp_t_stop;
    steady_clock::time_point rank_reveal_t_start;
    steady_clock::time_point rank_reveal_t_stop;
    steady_clock::time_point cholqr_t_start;
    steady_clock::time_point cholqr_t_stop;
    steady_clock::time_point a_mod_piv_t_start;
    steady_clock::time_point a_mod_piv_t_stop;
    steady_clock::time_point a_mod_trsm_t_start;
    steady_clock::time_point a_mod_trsm_t_stop;
    steady_clock::time_point total_t_start;
    steady_clock::time_point total_t_stop;
    long saso_t_dur        = 0;
    long qrcp_t_dur        = 0;
    long rank_reveal_t_dur = 0;
    long cholqr_t_dur      = 0;
    long a_mod_piv_t_dur   = 0;
    long a_mod_trsm_t_dur  = 0;
    long total_t_dur       = 0;

    if(this -> timing)
        total_t_start = steady_clock::now();

    int i;
    int64_t k = n;
    int64_t d = d_factor * n;
    // A constant for initial rank estimation.
    T eps_initial_rank_estimation = 2 * std::pow(std::numeric_limits<T>::epsilon(), 0.95);

    T* A_hat = new T[d * n]();
    T* tau   = new T[n]();

    // geqp3 reads jpvt on entry (nonzero marks a fixed column), so zero the
    // caller's J to keep its prior contents from steering the pivoting.
    std::fill(J, J + n, (int64_t) 0);

    if(this -> timing)
        saso_t_start = steady_clock::now();
    /***********************************************************************************/
    // I will avoid performing skething on a GPU for now

    /// Generating a SASO
    RandBLAS::SparseDist DS(d, m, this->nnz);
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    RandBLAS::fill_sparse(S);
    state = S.next_state;

    /// Applying a SASO
    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, (T) 1.0, S, 0, 0, A, lda, (T) 0.0, A_hat, d
    );

    if(this -> timing) {
        saso_t_stop = steady_clock::now();
        qrcp_t_start = steady_clock::now();
    }

    /// Performing QRCP on a sketch
    if(this->no_hqrrp) {
        lapack::geqp3(d, n, A_hat, d, J, tau);
    } else {
        hqrrp(d, n, A_hat, d, J, tau, this->nb_alg, this->oversampling, this->panel_pivoting, this->use_cholqr, state, (T**) nullptr);
    }

    if(this -> timing) {
        qrcp_t_stop = steady_clock::now();
        rank_reveal_t_start = steady_clock::now();
    }

    // Check if the input is all zeros, as CQRRPT does.
    if (!A_hat[0]) {
        this->rank = 0;
        delete[] A_hat;
        delete[] tau;
        return 0;
    }

    /// Using naive rank estimation to ensure that R used for preconditioning is invertible.
    /// The actual rank estimate k will be computed a posteriori. 
    /// Using R[i,i] to approximate the i-th singular value of A_hat. 
    /// Truncate at the largest i where R[i,i] / R[0,0] >= eps.
    for(i = 0; i < n; ++i) {
        if(std::abs(A_hat[i * d + i]) / std::abs(A_hat[0]) < eps_initial_rank_estimation) {
            k = i;
            break;
        }
    }
    this->rank = k;

    if(this -> timing)
        rank_reveal_t_stop = steady_clock::now();

    // Allocating space for a preconditioner buffer.
    T* R_sp = new T[k * k]();
    /// Extracting a k by k upper-triangular R.
    lapack::lacpy(MatrixType::Upper, k, k, A_hat, d, R_sp, k);
    /// Extracting a k by n R representation (k by k upper-triangular, rest - general)
    lapack::lacpy(MatrixType::Upper, k, k, A_hat, d, R, ldr);
    lapack::lacpy(MatrixType::General, k, n - k, &A_hat[d * k], d, &R[ldr * k], ldr);

    /******************************GPU REGION BEGIN*********************************/
    // The reasons for using GPUs for this part only ar the following: 
    // 1. There is no geqp3 available in any GPU linalg libraries. 
    //    We could port HQRRP to GPUs, but that takes additional time.
    // 2. There are no lacpy functions on any GPU linalg libraries. Those, however, 
    //    can be substituted with blas::copy.
    // 3. We do not have GPU-based skething support in RandBLAS at the moment.
    //
    // If the above points would be resolved, we may perform the entirety of CQRRPT on the device.
    //
    // Allocating device data & performing copies:
    T* A_device;
    T* R_device;
    T* R_sp_device;
    int64_t* J_device;
    // Gather buffer for pivoting A, rb rows at a time. rb keeps the gather's grid within 65,535
    // blocks of 8 rows, and W within 2^25 entries (256 MB in double).
    T* W_device;
    int64_t rb = std::min({m, (int64_t) 524280, std::max((int64_t) 1, ((int64_t) 1 << 25) / n)});

    // Rank estimated a posteriori.
    int64_t new_rank;

    using lapack::device_info_int;
    // One queue (one stream) for all device work: trsm, syrk, potrf, trsm and trmm
    // each read what the previous call wrote, and only same-stream launches are ordered.
    lapack::Queue lapack_queue(0);
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
    
    cudaMalloc(&A_device, lda * n * sizeof(T));
    cudaMalloc(&R_device, ldr * n * sizeof(T));
    cudaMalloc(&R_sp_device, k * k * sizeof(T));
    cudaMalloc(&J_device, n * sizeof(int64_t));
    cudaMalloc(&W_device, rb * n * sizeof(T));

    // A and R keep their leading dimensions on the device; only rows 0:m and 0:n are copied.
    // Every copy runs on lapack_queue's stream, so it is ordered against the device work even
    // when the default stream does not synchronize with other streams (per-thread default stream).
    blas::device_copy_matrix(m, n, A, lda, A_device, lda, lapack_queue);
    blas::device_copy_matrix(n, n, R, ldr, R_device, ldr, lapack_queue);
    blas::device_memcpy(R_sp_device, R_sp, k * k, lapack_queue);
    blas::device_memcpy(J_device, J, n, lapack_queue);

    if(this -> timing) {
        lapack_queue.sync();
        a_mod_piv_t_start = steady_clock::now();
    }

    // Permute the columns of A with J on the device: gather each block of rows into W, then copy it
    // back. Column j of A[:, J] is column J[j] - 1 of A, as util::col_swap gives on the host.
    for (int64_t r0 = 0; r0 < m; r0 += rb) {
        int64_t rows = std::min(rb, m - r0);
        RandLAPACK::cuda_kernels::col_swap_gpu(lapack_queue.stream(), rows, n, n, W_device, rb, &A_device[r0], lda, J_device);
        blas::device_copy_matrix(rows, n, W_device, rb, &A_device[r0], lda, lapack_queue);
    }

    if(this -> timing) {
        lapack_queue.sync();
        a_mod_piv_t_stop = steady_clock::now();
        a_mod_trsm_t_start = steady_clock::now();
    }

    //char name [] = "A";
    //RandBLAS::util::print_colmaj(m, n, A, name);

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, (T) 1.0, R_sp_device, k, A_device, lda, lapack_queue);

    // Device calls are asynchronous: when timing, wait for each step's work before stopping its timer.
    if(this -> timing) {
        lapack_queue.sync();
        a_mod_trsm_t_stop = steady_clock::now();
        cholqr_t_start = steady_clock::now();
    }

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, (T) 1.0, A_device, lda, (T) 0.0, R_sp_device, k, lapack_queue);
    lapack::potrf(Uplo::Upper, k, R_sp_device, k, d_info, lapack_queue);
    device_info_int info = 0;
    blas::device_memcpy(&info, d_info, 1, lapack_queue);
    lapack_queue.sync();

    // If CholQR failed, re-estimate the rank a posteriori from the completed part of R_sp's diagonal,
    // with CQRRPT's rule; cuSOLVER may leave NaN past the failed column, which the rule never keeps.
    //
    // The approach is slightly complicated due the fact that R_sp is currently allocated on device.
    // In order to avoid creating a specialized kerken function, we would copy the diagonal of R_sp 
    // back to the host using a strided cuda Memcopy (beware of performance issues).

    // In elements: the host pitch is 1, since we have a column vector; the device pitch is k + 1, so each
    // "row" starts at a diagonal entry of the k by k matrix; each of the k rows is 1 element wide.
    T* R_sp_diag = new T[k]();
    blas::device_memcpy_2d(R_sp_diag, (int64_t) 1, R_sp_device, k + 1, (int64_t) 1, k, lapack_queue);
    lapack_queue.sync();

    new_rank = util::cholqr_failure_rank(k, R_sp_diag, (int64_t) 1, (int64_t) info, this->eps);

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, (T) 1.0, R_sp_device, k, A_device, lda, lapack_queue);

    if(this -> timing) {
        lapack_queue.sync();
        cholqr_t_stop = steady_clock::now();
    }

    // Get the final R-factor: R[0:new_rank, :] = R_sp[0:new_rank, 0:k] * R[0:k, :].
    // After a failed potrf, R_sp[0:new_rank, new_rank:k] may not be a factor, so rebuild it as Q' * A_pre[:, new_rank:k].
    if(new_rank < k)
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, new_rank, k - new_rank, m, (T) 1.0, A_device, lda, &A_device[lda * new_rank], lda, (T) 0.0, &R_sp_device[k * new_rank], k, lapack_queue);
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, (T) 1.0, R_sp_device, k, R_device, ldr, lapack_queue);
    if(new_rank < k)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, new_rank, n, k - new_rank, (T) 1.0, &R_sp_device[k * new_rank], k, &R_device[new_rank], ldr, (T) 1.0, R_device, ldr, lapack_queue);

    // Set the rank parameter to the value comuted a posteriori.
    this->rank = new_rank;

    if(this -> timing) {
        lapack_queue.sync();
        saso_t_dur        = duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
        qrcp_t_dur        = duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
        rank_reveal_t_dur = duration_cast<microseconds>(rank_reveal_t_stop - rank_reveal_t_start).count();
        a_mod_piv_t_dur   = duration_cast<microseconds>(a_mod_piv_t_stop - a_mod_piv_t_start).count();
        a_mod_trsm_t_dur  = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
        cholqr_t_dur      = duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();

        total_t_stop = steady_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest  = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqr_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqr_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, t_rest, total_t_dur};
    }

    blas::device_copy_matrix(m, n, A_device, lda, A, lda, lapack_queue);
    blas::device_copy_matrix(n, n, R_device, ldr, R, ldr, lapack_queue);
    lapack_queue.sync();
    // Rows past the rank hold leftovers of the sketch's R factor.
    lapack::laset(MatrixType::General, n - new_rank, n, (T) 0.0, (T) 0.0, &R[new_rank], ldr);
    
    cudaFree(A_device);
    cudaFree(R_device);
    cudaFree(R_sp_device);
    cudaFree(J_device);
    cudaFree(W_device);
    blas::device_free(d_info, lapack_queue);
    delete[] A_hat;
    delete[] R_sp;
    delete[] tau;    
    delete[] R_sp_diag;

    return 0;
}
} // end namespace RandLAPACK
