#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "lapack/device.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>
#include <numeric>

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

        /// The algorithm allows for choosing how QRCP is emplemented: either thropught LAPACK's GEQP3
        /// or through a custom HQRRP function. This decision is controlled through 'no_hqrrp' parameter,
        /// which defaults to 1.
        ///
        /// The algorithm allows for choosing the rank estimation scheme either naively, through looking at the
        /// diagonal entries of an R-factor from QRCP or via finding the smallest k such that ||A[k:, k:]||_F <= tau_trunk * ||A||_x.
        /// This decision is controlled through 'naive_rank_estimate' parameter, which defaults to 1.
        /// The choice of norm ||A||_x, either 2 or F, is controlled via 'use_fro_norm'.
        ///
        /// The algorithm optionally times all of its subcomponents through a user-defined 'verbosity' parameter.
        ///
        /// The algorithm optionally computes a condition number of a preconditioned matrix A through a 'cond_check'
        /// parameter, which defaults to 0. This requires extra n * (m + 1) * sizeof(T) bytes of space, which will be 
        /// internally allocated by a utility routine. 
        /// A computation is handled by a utility method that finds the l2 condition number by computing all singular
        /// values of the R-factor via an appropriate LAPACK function.
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
        /// @param[in] d
        ///     Embedding dimension of a sketch, m >= d >= n.
        ///
        /// @param[in] R
        ///     Represents the upper-triangular R factor of QR factorization.
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[out] A
        ///     Overwritten by an m-by-k orthogonal Q factor.
        ///     Matrix is stored explicitly.
        ///
        /// @param[out] R
        ///     Stores k-by-n matrix with upper-triangular R factor.
        ///     Zero entries are not compressed.
        ///
        /// @param[out] J
        ///     Stores k integer type pivot index extries.
        ///
        /// @return = 0: successful exit
        ///
        /// @return = 1: cholesky factorization failed
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
    // Buffer for column pivoting.
    std::vector<int64_t> J_buf(n, 0);

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
        d, n, m, 1.0, S, 0, 0, A, lda, 0.0, A_hat, d
    );

    if(this -> timing) {
        saso_t_stop = steady_clock::now();
        qrcp_t_start = steady_clock::now();
    }

    /// Performing QRCP on a sketch
    if(this->no_hqrrp) {
        lapack::geqp3(d, n, A_hat, d, J, tau);
    } else {
        hqrrp(d, n, A_hat, d, J, tau, this->nb_alg, this->oversampling, this->panel_pivoting, this->use_cholqr, state, (T*) nullptr);
    }

    if(this -> timing) {
        qrcp_t_stop = steady_clock::now();
        rank_reveal_t_start = steady_clock::now();
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
    lapack::lacpy(MatrixType::General, k, n - k, &A_hat[d * k], d, &R[n * k], ldr);

    if(this -> timing)
        a_mod_piv_t_start = steady_clock::now();

    // Swap k columns of A with pivots from J
    blas::copy(n, J, 1, J_buf.data(), 1);
    util::col_swap(m, n, k, A, lda, J_buf);

    if(this -> timing) {
        a_mod_piv_t_stop = steady_clock::now();
        a_mod_trsm_t_start = steady_clock::now();
    }

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

    // Variables for a posteriori rank estimation.
    int64_t new_rank;
    T running_max, running_min, curr_entry;

    using lapack::device_info_int;
    blas::Queue blas_queue(0);
    lapack::Queue lapack_queue(0);
    device_info_int* d_info = blas::device_malloc< device_info_int >( 1, lapack_queue );
    
    cudaMalloc(&A_device, m * n * sizeof(T));
    cudaMalloc(&R_device, ldr * n * sizeof(T));
    cudaMalloc(&R_sp_device, k * k * sizeof(T));

    cudaMemcpy(A_device, A, m * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(R_device, R, ldr * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(R_sp_device, R_sp, k * k * sizeof(T), cudaMemcpyHostToDevice);

    //char name [] = "A";
    //RandBLAS::util::print_colmaj(m, n, A, name);

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_device, k, A_device, lda, blas_queue);

    if(this -> timing) {
        a_mod_trsm_t_stop = steady_clock::now();
        cholqr_t_start = steady_clock::now();
    }

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A_device, lda, 0.0, R_sp_device, k, blas_queue);
    lapack::potrf(Uplo::Upper, k, R_sp_device, k, d_info, lapack_queue);
    blas_queue.sync();

    // Re-estimate rank after we have the R-factor form Cholesky QR.
    // The strategy here is the same as in naive rank estimation.
    // This also automatically takes care of any potentical failures in Cholesky factorization.
    // Note that the diagonal of R_sp may not be sorted, so we need to keep the running max/min
    // We expect the loss in the orthogonality of Q to be approximately equal to u * cond(R_sp)^2, where u is the unit roundoff for the numerical type T.
    //
    // The approach is slightly complicated due the fact that R_sp is currently allocated on device.
    // In order to avoid creating a specialized kerken function, we would copy the diagonal of R_sp 
    // back to the host using a strided cuda Memcopy (beware of performance issues).

    // spitch  (2nd argument) - width of the row vectors in the source array - sizeof(T), since we have a column vector.
    // dpitch  (4nd argument) - width of the row vectors in the source array - (k + 1) * sizeof(T), since we have a k by k matrix and 
    //                                                                         want the "vector" to start with a diagonal element.
    // width   (5th argument) - number of columns in data transfer           - sizeof(T), since we transfer one element per column.
    // heighth (6th argument) - numer of rows in data transfer               - k, since we will be transferring k elements total.
    T* R_sp_diag = new T[k]();
    cudaMemcpy2D(R_sp_diag, sizeof(T), R_sp_device, (k + 1) * sizeof(T), sizeof(T), k, cudaMemcpyDeviceToHost);

    new_rank = k;
    running_max = R_sp_diag[0];
    running_min = R_sp_diag[0];

    for(i = 0; i < k; ++i) {
        curr_entry = std::abs(R_sp_diag[i]);
        running_max = std::max(running_max, curr_entry);
        running_min = std::min(running_min, curr_entry);
        if(running_max / running_min >= std::sqrt(this->eps / std::numeric_limits<T>::epsilon())) {
            new_rank = i - 1;
            break;
        }
    }

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, 1.0, R_sp_device, k, A_device, lda, blas_queue);

    if(this -> timing)
        cholqr_t_stop = steady_clock::now();

    // Get the final R-factor.
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, 1.0, R_sp_device, k, R_device, ldr, blas_queue);

    // Set the rank parameter to the value comuted a posteriori.
    this->rank = k;

    if(this -> timing) {
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

    cudaMemcpy(A, A_device, m * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(R, R_device, ldr * n * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(A_device);
    cudaFree(R_device);
    cudaFree(R_sp_device);
    delete[] A_hat;
    delete[] R_sp;
    delete[] tau;    
    delete[] R_sp_diag;

    return 0;
}
} // end namespace RandLAPACK
