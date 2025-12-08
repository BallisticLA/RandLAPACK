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
            nnz = 2;
        }

        /// Computes an unpivoted QR factorization of the form:
        ///     A= QR,
        /// where Q and R are of size m-by-n and n-by-n.
        /// Detailed description of this algorithm may be found in https://arxiv.org/pdf/2111.11148. 
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
        int64_t rank;

        // 6 entries
        std::vector<long> times;

        // tuning SASOS
        int64_t nnz;

        // Mode of operation that allows to use CQRRT for orthogonalization of the input matrix.
        // In this case, we do not compute the R-factor and if the input is rank-deficient, the algorithm would
        // append orthonormal columns at the end.
        bool orthogonalization;
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
    steady_clock::time_point saso_t_stop;
    steady_clock::time_point saso_t_start;
    steady_clock::time_point qr_t_start;
    steady_clock::time_point qr_t_stop;
    steady_clock::time_point cholqr_t_start;
    steady_clock::time_point cholqr_t_stop;
    steady_clock::time_point a_mod_trsm_t_start;
    steady_clock::time_point a_mod_trsm_t_stop;
    steady_clock::time_point total_t_start;
    steady_clock::time_point total_t_stop;
    long saso_t_dur        = 0;
    long qr_t_dur          = 0;
    long rank_reveal_t_dur = 0;
    long cholqr_t_dur      = 0;
    long a_mod_piv_t_dur   = 0;
    long a_mod_trsm_t_dur  = 0;
    long total_t_dur       = 0;

    if(this -> timing)
        total_t_start = steady_clock::now();

    int i;
    int64_t d = d_factor * n;
    // Variables for a posteriori rank estimation.
    int64_t new_rank = n;
    T running_max, running_min, curr_entry;

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
        a_mod_trsm_t_start = steady_clock::now();

    // A_pre * R_sk = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sk, ldr, A, lda);

    if(this -> timing) {
        a_mod_trsm_t_stop = steady_clock::now();
        cholqr_t_start = steady_clock::now();
    }

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A, lda, 0.0, R_sk, ldr);
    int msg = lapack::potrf(Uplo::Upper, n, R_sk, ldr);
    //printf("CholQR output message: %d\n", msg);
    //if(msg) {
        // Perform aposteriori rank estimation of CholQR failed?

        // Estimate rank after we have the R-factor form Cholesky QR.
        // The strategy here is the same as in naive rank estimation.
        // This also automatically takes care of any potential failures in Cholesky factorization.
        // Note that the diagonal of R_sk may not be sorted, so we need to keep the running max/min
        // We expect the loss in the orthogonality of Q to be approximately equal to u * cond(R_sk)^2, where u is the unit roundoff for the numerical type T.
        running_max = R_sk[0];
        running_min = R_sk[0];
        T cond_threshold = std::sqrt(this->eps / std::numeric_limits<T>::epsilon());

        for(i = 0; i < n; ++i) {
            curr_entry = std::abs(R_sk[i * ldr + i]);
            running_max = std::max(running_max, curr_entry);
            running_min = std::min(running_min, curr_entry);
            if((running_min * cond_threshold < running_max) && i > 1) {
                new_rank = i - 1;
                break;
            }
        }
    //}

    // Set the rank parameter to the value computed a posteriori.
    this->rank = new_rank;

    // Obtain the output Q-factor
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, 1.0, R_sk, ldr, A, lda);

    if(this -> timing)
        cholqr_t_stop = steady_clock::now();

    if (!this->orthogonalization) {
        // Get the final R-factor -- undoing the preconditioning
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, 1.0, A_hat, d, R_sk, ldr); 
    } 
    
    // Function always full econ Q
    if (new_rank != n) {
        // Complete the orthonormal set
        // Generate Gaussian matrix G of size (m, n - new_rank) in trailing columns of A
        int64_t cols_to_fill = n - new_rank;
        RandBLAS::DenseDist D(m, cols_to_fill);
        RandBLAS::fill_dense(D, &A[new_rank * lda], state);

        // Project out G = (I - QQ^T)G
        // First compute QQ^T * G and store temporarily
        T* temp = new T[m * cols_to_fill]();
        // temp = Q^T * G
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, new_rank, cols_to_fill, m, 1.0, A, lda, &A[new_rank * lda], lda, 0.0, temp, new_rank);
        // G := G - Q * temp (i.e., G = G - QQ^T * G)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, cols_to_fill, new_rank, -1.0, A, lda, temp, new_rank, 1.0, &A[new_rank * lda], lda);
        delete[] temp;
        
        // Orthogonalize G using QRF + ORGQR
        T* tau_orth = new T[cols_to_fill]();
        lapack::geqrf(m, cols_to_fill, &A[new_rank * lda], lda, tau_orth);
        lapack::orgqr(m, cols_to_fill, cols_to_fill, &A[new_rank * lda], lda, tau_orth);
        delete[] tau_orth;
    } 

    if(this -> timing) {
        saso_t_dur       = duration_cast<microseconds>(saso_t_stop       - saso_t_start).count();
        qr_t_dur         = duration_cast<microseconds>(qr_t_stop         - qr_t_start).count();
        a_mod_trsm_t_dur = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
        cholqr_t_dur     = duration_cast<microseconds>(cholqr_t_stop     - cholqr_t_start).count();

        total_t_stop = steady_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest  = total_t_dur - (saso_t_dur + qr_t_dur + cholqr_t_dur + a_mod_trsm_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qr_t_dur, cholqr_t_dur, a_mod_trsm_t_dur, t_rest, total_t_dur};
    }

    delete[] A_hat;
    delete[] tau;

    return 0;
}
} // end namespace RandLAPACK
