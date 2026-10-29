#pragma once

#include "rl_exceptions.hh"
#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "lapack/device.hh"

#include "RandLAPACK/gpu_functions/rl_cuda_kernels.cuh"
#include "RandLAPACK/drivers/rl_bqrrp_gpu.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <memory>
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

// Struct outside of the CQRRPT_GPU class to make symbols shorter
struct CQRRPT_GPU_Subroutines {
    // The QRCP of the d by n sketch A_hat, on the device; CQRRPT_GPU's member 'qrcp' picks one.
    enum QRCP {
        // The default. The column pivots come from a row-pivoted LU of the transpose of G * A_hat, where G is
        // an n by d Gaussian that the host applies to A_hat; an unpivoted QR of the pivoted A_hat then gives R,
        // as in each block of BQRRP_GPU. The LU sees this Gaussian mix of the sketch rather than the sketch
        // itself because a sparse sketch can have zero rows, which stall LU with partial pivoting: at such a
        // step it finds no pivot and keeps the next column in place, even a zero or a dependent one.
        // With these pivots, the rank estimate can come out a little higher than with GEQP3, CQRRPT's default,
        // on numerically rank-deficient inputs, while the factors stay accurate.
        luqr,
        // BQRRP_GPU on A_hat, guided by a Gaussian sketch of it, G * A_hat with G as many rows by d as BQRRP_GPU's
        // block size, CQRRPT_GPU's member 'bqrrp_block_size'. Its default, 0, takes the block size of CQRRPT's
        // bqrrp option: n for n <= 2000, n / 2 for n <= 8000, and n / 32 beyond. BQRRP_GPU gathers the d rows of
        // the sketch in one launch, which covers at most 524,280 rows (65,535 thread blocks of 8 rows): for
        // d = d_factor * n above that, the call rejects its arguments as it does others, with RandLAPACK::Error,
        // before it writes R or touches the device.
        bqrrp
    };
};

template <typename T, typename RNG>
class CQRRPT_GPU : public CQRRPT_GPU_alg<T, RNG> {
    public:

        using Subroutines = CQRRPT_GPU_Subroutines;

        /// The sketch runs on the host, as RandBLAS does not sketch on a GPU yet; everything after it
        /// runs on the device. The QRCP of the sketch is chosen by 'qrcp', luqr by default; see
        /// CQRRPT_GPU_Subroutines::QRCP.
        /// The sparse sketching operator has 'nnz' nonzeros per column, 2 by default, as in CQRRPT.
        /// A in pinned host memory (cudaMallocHost) is copied to the device while the host sketches it;
        /// the driver does not pin A itself.
        ///
        /// @param[in] verb
        ///     Stored in 'verbosity'; the algorithm prints nothing.
        ///
        /// @param[in] time_subroutines
        ///     If true, a call that factors A fills 'times' with the durations of its steps; see 'times'.
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
            qrcp = Subroutines::QRCP::luqr;
            bqrrp_block_size = 0;
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
        /// @param[in,out] state
        ///     RNG state of the sketching operators. A call on a nonempty A advances it past the sparse operator
        ///     and past the Gaussian one, which is drawn from where the sparse one ends, so the two are independent
        ///     and the same state gives the same result; an empty A leaves it unchanged. CQRRPT with its default
        ///     QRCP (GEQP3) draws only the sparse operator, so the state it returns differs.
        ///
        /// @param[out] J
        ///     Stores n integer type pivot index entries. For an empty A (m = 0 or n = 0) or a zero A,
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

        // With timing on, 8 durations in microseconds: the sketch and its Gaussian sketch (the first issues the
        // copy of A), the QRCP of the sketch (with the copies of both sketches to the device, which wait for the
        // copy of A), the zero test and rank estimate, the pivoting of A, CholQR (syrk, potrf and the trsm that
        // forms Q), the preconditioning trsm, the rest, and the total. The total covers the whole call: the copies
        // back and the release of the device memory and queue too. Each device step ends with a sync of the queue.
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // QRCP of the sketch
        Subroutines::QRCP qrcp;
        int64_t bqrrp_block_size;
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
    ///--------------------TIMING VARS--------------------/
    steady_clock::time_point sketch_t_start;
    steady_clock::time_point sketch_t_stop;
    steady_clock::time_point qrcp_t_start;
    steady_clock::time_point qrcp_t_stop;
    steady_clock::time_point rank_est_t_start;
    steady_clock::time_point rank_est_t_stop;
    steady_clock::time_point piv_t_start;
    steady_clock::time_point piv_t_stop;
    steady_clock::time_point cholqr_t_start;
    steady_clock::time_point cholqr_t_stop;
    steady_clock::time_point trsm_t_start;
    steady_clock::time_point trsm_t_stop;
    steady_clock::time_point total_t_start;
    steady_clock::time_point total_t_stop;

    if(this -> timing)
        total_t_start = steady_clock::now();

    // The most rows that one launch of col_swap_gpu's gather covers: it runs a thread block per 8 rows,
    // and a grid holds at most 65,535 blocks in y.
    const int64_t max_gather_rows = 8 * 65535;

    // Input parameter validation, as in CQRRPT.
    randlapack_require(m >= 0) << "m=" << m << " must be >= 0";
    randlapack_require(n >= 0) << "n=" << n << " must be >= 0";
    randlapack_require(lda >= m) << "lda=" << lda << " < m=" << m << " (lda must be >= m for ColMajor)";
    randlapack_require(ldr >= n) << "ldr=" << ldr << " < n=" << n << " (ldr must be >= n)";
    randlapack_require(d_factor >= (T)1.0) << "d_factor=" << d_factor << " must be >= 1.0";
    randlapack_require(!(A == nullptr && m > 0 && n > 0)) << "A buffer is null but m=" << m << " and n=" << n << " imply a nonempty matrix";
    randlapack_require(!(R == nullptr && n > 0)) << "R buffer is null but n=" << n << " > 0";
    randlapack_require(!(J == nullptr && n > 0)) << "J buffer is null but n=" << n << " > 0";
    // BQRRP_GPU gathers all d = d_factor * n rows of the sketch in one launch; luqr gathers them in blocks of rows.
    randlapack_require(this->qrcp == Subroutines::QRCP::luqr || (int64_t) (d_factor * n) <= max_gather_rows)
        << "d=" << (int64_t) (d_factor * n) << " > " << max_gather_rows << " with qrcp = bqrrp (BQRRP_GPU gathers the rows of the sketch"
        << " in one launch; use luqr, or a smaller d_factor)";

    // R's contents on entry are ignored: the parts the algorithm does not write are read as zeros.
    lapack::laset(MatrixType::General, n, n, (T) 0.0, (T) 0.0, R, ldr);

    // An empty A has nothing to factor. Return before touching the device, with rank 0 and the
    // identity pivots that GEQP3 returns for an empty matrix, as BQRRP_GPU does.
    if (std::min(m, n) == 0) {
        this->rank = 0;
        std::iota(J, J + n, (int64_t) 1);
        return 0;
    }

    int64_t k = n;
    int64_t d = d_factor * n;
    // A constant for initial rank estimation.
    T eps_initial_rank_estimation = 2 * std::pow(std::numeric_limits<T>::epsilon(), 0.95);
    // Rank estimated a posteriori.
    int64_t new_rank;

    bool luqr = this->qrcp == Subroutines::QRCP::luqr;
    // BQRRP_GPU's block size, and the rows of the Gaussian sketch of the sketch, which guides the QRCP of the
    // sketch: n for luqr, BQRRP_GPU's block size for bqrrp.
    int64_t b_sz = this->bqrrp_block_size;
    if (b_sz <= 0)
        b_sz = n <= 2000 ? n : (n <= 8000 ? n / 2 : n / 32);
    b_sz = std::max((int64_t) 1, std::min(b_sz, n));
    int64_t d_sk = luqr ? n : b_sz;

    // The host buffers of this call, freed on every exit, exceptions included.
    struct HostBuffers {
        T* A_hat = nullptr;
        T* A_hat_sk = nullptr;
        T* R_sp_lead = nullptr;
        char* work_getrf = nullptr;
        char* work_geqrf = nullptr;
        void release() {
            delete[] A_hat;
            delete[] A_hat_sk;
            delete[] R_sp_lead;
            delete[] work_getrf;
            delete[] work_geqrf;
            A_hat = A_hat_sk = R_sp_lead = nullptr;
            work_getrf = work_geqrf = nullptr;
        }
        ~HostBuffers() { release(); }
    } host;
    host.A_hat = new T[d * n]();
    host.A_hat_sk = new T[d_sk * n]();

    /******************************DEVICE SETUP*********************************/
    using lapack::device_info_int;
    using lapack::device_pivot_int;
    // One queue (one stream) for all device work: each call reads what the previous ones wrote,
    // and only same-stream launches are ordered. A normal exit destroys it before the total timer stops.
    auto queue_owner = std::make_unique<lapack::Queue>(0);
    lapack::Queue& lapack_queue = *queue_owner;
    cudaStream_t strm = lapack_queue.stream();

    // The device buffers of this call. release() runs on every exit from here on, exceptions included,
    // through the destructor: unless the call's final sync has run, it waits for the queue, whose copy of
    // A may still be reading the caller's A, and it then frees the buffers. Neither cudaStreamSynchronize
    // nor cudaFree (which matches blas::device_malloc) throws, as a destructor must not.
    struct DeviceBuffers {
        lapack::Queue& queue;
        bool synced = false;
        T* A = nullptr;
        T* R = nullptr;
        T* R_sp = nullptr;
        T* W = nullptr;
        T* A_sk = nullptr;
        T* A_sk_work = nullptr;
        T* A_sk_sk = nullptr;
        T* tau = nullptr;
        int64_t* J = nullptr;
        device_pivot_int* J_lu = nullptr;
        int64_t* rank = nullptr;
        bool* zero = nullptr;
        device_info_int* info = nullptr;
        char* work_getrf = nullptr;
        char* work_geqrf = nullptr;
        void release() {
            if (!synced)
                cudaStreamSynchronize(queue.stream());
            synced = true;
            auto free = [](auto*& p) { if (p) cudaFree(p); p = nullptr; };
            free(A); free(R); free(R_sp); free(W); free(A_sk); free(A_sk_work); free(A_sk_sk); free(tau);
            free(J); free(J_lu); free(rank); free(zero); free(info); free(work_getrf); free(work_geqrf);
        }
        ~DeviceBuffers() { release(); }
    } dev{lapack_queue};

    // The NVTX range of the current step, for profilers. The destructor ends an open range, so that
    // the caller's ranges stay balanced when the call returns early or throws.
    struct NvtxStep {
        bool open = false;
        void begin(const char* name) { end(); nvtxRangePushA(name); open = true; }
        void end() { if (open) nvtxRangePop(); open = false; }
        ~NvtxStep() { end(); }
    } step;

    // Gathers the columns J of the rows by n src into dst, at most max_gather_rows rows per launch.
    auto gather_rows = [&](int64_t rows, T* dst, int64_t ldd, T* src, int64_t lds) {
        for (int64_t r0 = 0; r0 < rows; r0 += max_gather_rows)
            RandLAPACK::cuda_kernels::col_swap_gpu(strm, std::min(max_gather_rows, rows - r0), n, n, &dst[r0], ldd, &src[r0], lds, dev.J);
    };

    // All device memory is allocated here, before any work is queued, and freed after the last sync.
    // A and R keep their leading dimensions on the device. R_sp, sized for k = n, holds the k by k
    // preconditioner, and then the Cholesky factor, with leading dimension k. W is the gather buffer
    // for pivoting A, rb rows at a time: rb keeps each block of rows to one gather, and W within 2^25
    // entries (256 MB in double). A_sk is the d by n sketch, and A_sk_sk the d_sk by n sketch of the
    // sketch. With luqr, A_sk_work holds the n by d_sk transpose of A_sk_sk for the LU, and then the
    // pivoted sketch and its QR, with leading dimension d; with bqrrp, the copy of the sketch that
    // BQRRP_GPU factors. BQRRP_GPU allocates its own workspace.
    int64_t rb = std::min({m, max_gather_rows, std::max((int64_t) 1, ((int64_t) 1 << 25) / n)});
    dev.A         = blas::device_malloc< T >( lda * n, lapack_queue );
    dev.R         = blas::device_malloc< T >( ldr * n, lapack_queue );
    dev.R_sp      = blas::device_malloc< T >( n * n, lapack_queue );
    dev.W         = blas::device_malloc< T >( rb * n, lapack_queue );
    dev.A_sk      = blas::device_malloc< T >( d * n, lapack_queue );
    dev.A_sk_work = blas::device_malloc< T >( d * n, lapack_queue );
    dev.A_sk_sk   = blas::device_malloc< T >( d_sk * n, lapack_queue );
    dev.tau       = blas::device_malloc< T >( n, lapack_queue );
    dev.J         = blas::device_malloc< int64_t >( n, lapack_queue );
    dev.rank      = blas::device_malloc< int64_t >( 1, lapack_queue );
    dev.zero      = blas::device_malloc< bool >( 1, lapack_queue );
    dev.info      = blas::device_malloc< device_info_int >( 1, lapack_queue );
    size_t d_size_getrf = 0, h_size_getrf = 0, d_size_geqrf = 0, h_size_geqrf = 0;
    if (luqr) {
        // Workspaces of the LU of the n by d_sk transpose and of the QR of the d by n pivoted sketch.
        dev.J_lu = blas::device_malloc< device_pivot_int >( n, lapack_queue );
        lapack::getrf_work_size_bytes(n, d_sk, dev.A_sk_work, n, &d_size_getrf, &h_size_getrf, lapack_queue);
        lapack::geqrf_work_size_bytes(d, n, dev.A_sk_work, d, &d_size_geqrf, &h_size_geqrf, lapack_queue);
        dev.work_getrf = blas::device_malloc< char >( d_size_getrf, lapack_queue );
        dev.work_geqrf = blas::device_malloc< char >( d_size_geqrf, lapack_queue );
        if (h_size_getrf > 0)
            host.work_getrf = new char[h_size_getrf];
        if (h_size_geqrf > 0)
            host.work_geqrf = new char[h_size_geqrf];
    }
    // potrf grows the queue's workspace when it is too small, which syncs the queue and allocates.
    // Size it now, with the query potrf makes (under the same cuSOLVER version guard), for n. Should
    // the query fail, or the k by k factor need more, potrf grows the workspace itself.
#if CUSOLVER_VERSION >= 11000
    size_t d_size_potrf = 0, h_size_potrf = 0;
    cudaDataType potrf_type = std::is_same_v<T, double> ? CUDA_R_64F : CUDA_R_32F;
    if (cusolverDnXpotrf_bufferSize(lapack_queue.solver(), lapack_queue.solver_params(), CUBLAS_FILL_MODE_UPPER, n, potrf_type, dev.R_sp, n, potrf_type, &d_size_potrf, &h_size_potrf) == CUSOLVER_STATUS_SUCCESS)
        lapack_queue.work_ensure_size< char >( d_size_potrf );
#endif

    if(this -> timing)
        sketch_t_start = steady_clock::now();
    step.begin("sketch");

    // Copy A to the device before the sketch, which only reads A: a pinned A is copied while the host
    // sketches, a pageable one before. Only rows 0:m are copied.
    // Every copy runs on lapack_queue's stream, so it is ordered against the device work even
    // when the default stream does not synchronize with other streams (per-thread default stream).
    blas::device_copy_matrix(m, n, A, lda, dev.A, lda, lapack_queue);

    // The sketching operators live in this block only: the Gaussian one holds d_sk by d entries.
    {
        /// Generating a SASO
        RandBLAS::SparseDist DS(d, m, this->nnz);
        RandBLAS::SparseSkOp<T, RNG> S(DS, state);
        RandBLAS::fill_sparse(S);
        state = S.next_state;

        /// Applying a SASO
        RandBLAS::sketch_general(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, (T) 1.0, S, 0, 0, A, lda, (T) 0.0, host.A_hat, d
        );

        // The Gaussian sketch of the sketch, G * A_hat with G d_sk by d, whose rows mix all of A_hat's; it
        // guides both QRCPs of the sketch, as CPU BQRRP sketches its input.
        RandBLAS::DenseDist D_sk(d_sk, d);
        RandBLAS::DenseSkOp<T, RNG> S_sk(D_sk, state);
        state = S_sk.next_state;
        RandBLAS::sketch_general(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d_sk, n, d, (T) 1.0, S_sk, 0, 0, host.A_hat, d, (T) 0.0, host.A_hat_sk, d_sk
        );
    }

    step.end();
    if(this -> timing) {
        sketch_t_stop = steady_clock::now();
        qrcp_t_start = steady_clock::now();
    }
    step.begin("qrcp");

    blas::device_copy_matrix(d, n, host.A_hat, d, dev.A_sk, d, lapack_queue);
    blas::device_copy_matrix(d_sk, n, host.A_hat_sk, d_sk, dev.A_sk_sk, d_sk, lapack_queue);
    if (luqr) {
        // QRCP of the sketch through LUQR, as in a block of BQRRP_GPU: the row pivots of an LU of the
        // transpose of the sketch of the sketch, which is left in A_sk_work, are the column pivots J; then
        // an unpivoted QR of the pivoted sketch, gathered into A_sk_work, gives R_sk in its upper triangle.
        RandLAPACK::cuda_kernels::transposition_gpu(strm, d_sk, n, dev.A_sk_sk, d_sk, dev.A_sk_work, n, 0);
        lapack::getrf(n, d_sk, dev.A_sk_work, n, dev.J_lu, dev.work_getrf, d_size_getrf, host.work_getrf, h_size_getrf, dev.info, lapack_queue);
        RandLAPACK::cuda_kernels::LUQRCP_piv_process_gpu(strm, d_sk, n, dev.J, dev.J_lu);
        gather_rows(d, dev.A_sk_work, d, dev.A_sk, d);
        lapack::geqrf(d, n, dev.A_sk_work, d, dev.tau, dev.work_geqrf, d_size_geqrf, host.work_geqrf, h_size_geqrf, dev.info, lapack_queue);
    } else {
        // QRCP of the sketch through BQRRP_GPU, on a copy, since the zero test below reads the sketch.
        // BQRRP_GPU leaves R_sk in the upper triangle of A_sk_work, as GEQP3 does. It runs on its own
        // queue: sync before the call, so that its inputs are on the device; it syncs its queue before
        // it returns.
        blas::device_copy_matrix(d, n, dev.A_sk, d, dev.A_sk_work, d, lapack_queue);
        lapack_queue.sync();
        BQRRP_GPU<T, RNG> bqrrp_gpu(false, b_sz);
        bqrrp_gpu.call(d, n, dev.A_sk_work, d, dev.A_sk_sk, d_sk, dev.tau, dev.J);
    }
    T* R_sk = dev.A_sk_work;

    step.end();
    if(this -> timing) {
        lapack_queue.sync();
        qrcp_t_stop = steady_clock::now();
        rank_est_t_start = steady_clock::now();
    }
    step.begin("rank_est");

    // Check if the input is all zeros, as CQRRPT does, on the sketch itself: with LU pivoting,
    // R_sk(0, 0) = 0 does not imply a zero sketch. A NaN entry counts as nonzero.
    RandLAPACK::cuda_kernels::all_of_async(strm, d * n, (T) 0.0, dev.A_sk, dev.zero);
    /// Using naive rank estimation to ensure that R used for preconditioning is invertible.
    /// The actual rank estimate k will be computed a posteriori.
    /// Using R[i,i] to approximate the i-th singular value of A_hat.
    /// Truncate at the largest i where R[i,i] / R[0,0] >= eps.
    RandLAPACK::cuda_kernels::naive_rank_est_async(strm, n, eps_initial_rank_estimation, R_sk, d, dev.rank);
    bool sketch_zero = false;
    blas::device_memcpy(&sketch_zero, dev.zero, 1, lapack_queue);
    blas::device_memcpy(&k, dev.rank, 1, lapack_queue);
    lapack_queue.sync();

    step.end();
    // dev waits for the copy of A and frees.
    if (sketch_zero) {
        this->rank = 0;
        std::iota(J, J + n, (int64_t) 1);
        return 0;
    }
    this->rank = k;

    if(this -> timing)
        rank_est_t_stop = steady_clock::now();

    // R = [R_sk(0:k, 0:k) upper triangular, R_sk(0:k, k:n)], zero elsewhere. The preconditioner is its
    // leading k by k block.
    blas::device_memset(dev.R, 0, ldr * n, lapack_queue);
    RandLAPACK::cuda_kernels::copy_mat_gpu(strm, k, n, R_sk, d, dev.R, ldr, true);
    blas::device_copy_matrix(k, k, dev.R, ldr, dev.R_sp, k, lapack_queue);

    if(this -> timing) {
        lapack_queue.sync();
        piv_t_start = steady_clock::now();
    }
    step.begin("piv_A");

    // Permute the columns of A with J on the device: gather each block of rows into W, then copy it
    // back. Column j of A[:, J] is column J[j] - 1 of A, as util::col_swap gives on the host.
    for (int64_t r0 = 0; r0 < m; r0 += rb) {
        int64_t rows = std::min(rb, m - r0);
        gather_rows(rows, dev.W, rb, &dev.A[r0], lda);
        blas::device_copy_matrix(rows, n, dev.W, rb, &dev.A[r0], lda, lapack_queue);
    }

    step.end();
    if(this -> timing) {
        lapack_queue.sync();
        piv_t_stop = steady_clock::now();
        trsm_t_start = steady_clock::now();
    }
    step.begin("precond_A");

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, (T) 1.0, dev.R_sp, k, dev.A, lda, lapack_queue);

    step.end();
    // Device calls are asynchronous: when timing, wait for each step's work before stopping its timer.
    if(this -> timing) {
        lapack_queue.sync();
        trsm_t_stop = steady_clock::now();
        cholqr_t_start = steady_clock::now();
    }
    step.begin("cholqr");

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, (T) 1.0, dev.A, lda, (T) 0.0, dev.R_sp, k, lapack_queue);
    lapack::potrf(Uplo::Upper, k, dev.R_sp, k, dev.info, lapack_queue);
    device_info_int info = 0;
    blas::device_memcpy(&info, dev.info, 1, lapack_queue);
    lapack_queue.sync();

    // If CholQR failed, re-estimate the rank a posteriori from the completed columns of R_sp, with CQRRPT's
    // rule; cuSOLVER may leave NaN from the failed column on, which the rule never reads.
    new_rank = k;
    if (info > 0) {
        // Copy the leading (info - 1)-by-(info - 1) block, all of which the rule reads: in elements, each of its
        // b columns (a "row" of the 2D copy) is b wide, at pitch k on the device and pitch b on the host.
        int64_t b = (int64_t) info - 1;
        int64_t ld_lead = std::max(b, (int64_t) 1);
        host.R_sp_lead = new T[ld_lead * b]();
        if (b > 0)
            blas::device_memcpy_2d(host.R_sp_lead, ld_lead, dev.R_sp, k, b, b, lapack_queue);
        lapack_queue.sync();
        new_rank = util::cholqr_failure_rank(k, host.R_sp_lead, ld_lead, (int64_t) info, this->eps);
    }

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, (T) 1.0, dev.R_sp, k, dev.A, lda, lapack_queue);

    step.end();
    if(this -> timing) {
        lapack_queue.sync();
        cholqr_t_stop = steady_clock::now();
    }
    step.begin("form_R");

    // Get the final R-factor: R[0:new_rank, :] = R_sp[0:new_rank, 0:k] * R[0:k, :].
    // After a failed potrf, R_sp[0:new_rank, new_rank:k] may not be a factor, so rebuild it as Q' * A_pre[:, new_rank:k].
    if(new_rank < k)
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, new_rank, k - new_rank, m, (T) 1.0, dev.A, lda, &dev.A[lda * new_rank], lda, (T) 0.0, &dev.R_sp[k * new_rank], k, lapack_queue);
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, (T) 1.0, dev.R_sp, k, dev.R, ldr, lapack_queue);
    if(new_rank < k)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, new_rank, n, k - new_rank, (T) 1.0, &dev.R_sp[k * new_rank], k, &dev.R[new_rank], ldr, (T) 1.0, dev.R, ldr, lapack_queue);

    step.end();
    // Set the rank parameter to the value comuted a posteriori.
    this->rank = new_rank;

    step.begin("copy_back");
    blas::device_copy_matrix(m, n, dev.A, lda, A, lda, lapack_queue);
    blas::device_copy_matrix(n, n, dev.R, ldr, R, ldr, lapack_queue);
    blas::device_memcpy(J, dev.J, n, lapack_queue);
    lapack_queue.sync();
    dev.synced = true;
    step.end();
    // Rows past the rank hold leftovers of the sketch's R factor.
    lapack::laset(MatrixType::General, n - new_rank, n, (T) 0.0, (T) 0.0, &R[new_rank], ldr);

    // Release the memory and the queue here, rather than on return, so that total covers them.
    dev.release();
    host.release();
    queue_owner.reset();

    if(this -> timing) {
        total_t_stop = steady_clock::now();
        auto us = [](steady_clock::time_point start, steady_clock::time_point stop) {
            return (long) duration_cast<microseconds>(stop - start).count();
        };
        long sketch_t_dur   = us(sketch_t_start, sketch_t_stop);
        long qrcp_t_dur     = us(qrcp_t_start, qrcp_t_stop);
        long rank_est_t_dur = us(rank_est_t_start, rank_est_t_stop);
        long piv_t_dur      = us(piv_t_start, piv_t_stop);
        long cholqr_t_dur   = us(cholqr_t_start, cholqr_t_stop);
        long trsm_t_dur     = us(trsm_t_start, trsm_t_stop);
        long total_t_dur    = us(total_t_start, total_t_stop);
        long t_rest = total_t_dur - (sketch_t_dur + qrcp_t_dur + rank_est_t_dur + piv_t_dur + cholqr_t_dur + trsm_t_dur);
        this -> times = {sketch_t_dur, qrcp_t_dur, rank_est_t_dur, piv_t_dur, cholqr_t_dur, trsm_t_dur, t_rest, total_t_dur};
    }

    return 0;
}
} // end namespace RandLAPACK
