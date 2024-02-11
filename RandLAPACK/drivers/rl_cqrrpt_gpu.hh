#ifndef randlapack_cqrrpt_gpu_h
#define randlapack_cqrrpt_gpu_h


#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

#include "rl_cuda_macros.hh"
#include <cuda.h>
#include <cuda_runtime.h>

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
    high_resolution_clock::time_point saso_t_stop;
    high_resolution_clock::time_point saso_t_start;
    high_resolution_clock::time_point qrcp_t_start;
    high_resolution_clock::time_point qrcp_t_stop;
    high_resolution_clock::time_point rank_reveal_t_start;
    high_resolution_clock::time_point rank_reveal_t_stop;
    high_resolution_clock::time_point cholqr_t_start;
    high_resolution_clock::time_point cholqr_t_stop;
    high_resolution_clock::time_point a_mod_piv_t_start;
    high_resolution_clock::time_point a_mod_piv_t_stop;
    high_resolution_clock::time_point a_mod_trsm_t_start;
    high_resolution_clock::time_point a_mod_trsm_t_stop;
    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;
    long saso_t_dur        = 0;
    long qrcp_t_dur        = 0;
    long rank_reveal_t_dur = 0;
    long cholqr_t_dur      = 0;
    long a_mod_piv_t_dur   = 0;
    long a_mod_trsm_t_dur  = 0;
    long total_t_dur       = 0;

    if(this -> timing)
        total_t_start = high_resolution_clock::now();

    int i;
    int64_t k = n;
    int64_t d = d_factor * n;
    // A constant for initial rank estimation.
    T eps_initial_rank_estimation = 2 * std::pow(std::numeric_limits<T>::epsilon(), 0.95);
    // Variables for a posteriori rank estimation.
    int64_t new_rank;
    T running_max, running_min, curr_entry;

    T* A_hat = ( T * ) calloc( d * n, sizeof( T ) );
    T* tau   = ( T * ) calloc( n, sizeof( T ) );
    // Buffer for column pivoting.
    std::vector<int64_t> J_buf(n, 0);

    T* A_hat_device;
    cudaMalloc(&A_hat_device, d * n);

    return 0;
}
} // end namespace RandLAPACK
#endif
