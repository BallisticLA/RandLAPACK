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
class CQRRPTalg {
    public:

        virtual ~CQRRPTalg() {}

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

// Struct outside of CQRRPT class to make symbols shorter
struct CQRRPTSubroutines {
    enum QRCP {hqrrp, bqrrp, geqp3};
};

template <typename T, typename RNG = RandBLAS::DefaultRNG>
class CQRRPT : public CQRRPTalg<T, RNG> {
    public:

        using Subroutines = CQRRPTSubroutines;

        /// The algorithm allows for choosing how QRCP is emplemented: either thropught LAPACK's GEQP3
        /// or through a custom HQRRP function, or through BQRRP function. GEQP3 is the default option.
        CQRRPT(
            bool time_subroutines,
            T ep
        ) {
            timing = time_subroutines;
            eps = ep;
            qrcp = Subroutines::QRCP::geqp3;
            bqrrp_block_ratio = 1;
            nb_alg = 64;
            oversampling = 10;
            use_cholqr = 0;
            panel_pivoting = 1;
            orthogonalization = false;
            nnz = 2;
        }

        /// Computes a QR factorization with column pivots of the form:
        ///     A[:, J] = QR,
        /// where Q and R are of size m-by-k and k-by-n, with rank(A) = k.
        /// Detailed description of this algorithm may be found in https://arxiv.org/abs/2311.08316. 
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
        bool timing;
        T eps;
        int64_t rank;

        // 8 entries
        std::vector<long> times;

        // tuning SASOS
        int64_t nnz;

        // QRCP-related
        Subroutines::QRCP qrcp;
        // Ratio of BQRRP block size to the number of columns n; 
        // used to set the BQRRP block size.
        double bqrrp_block_ratio;
        int64_t nb_alg;
        int64_t oversampling;
        int64_t panel_pivoting;
        int64_t use_cholqr;

        // Mode of operation that allows to use CQRRPT for orthogonalization of the input matrix.
        // In this case, we do not compute the R-factor and if the input is rank-deficient, the algorithm would
        // append orthonormal columns at the end.
        bool orthogonalization;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRPT<T, RNG>::call(
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
    // Variables for a posteriori rank estimation.
    int64_t new_rank;
    T running_max, running_min, curr_entry;

    T* A_hat = new T[d * n]();
    T* tau   = new T[n]();
    // Buffer for column pivoting.
    std::vector<int64_t> J_buf(n, 0);

    if(this -> timing)
        saso_t_start = steady_clock::now();
    
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
    if(this -> qrcp == Subroutines::QRCP::hqrrp) {
        hqrrp(d, n, A_hat, d, J, tau, this->nb_alg, this->oversampling, this->panel_pivoting, this->use_cholqr, state, (T**) nullptr);
    } else if(this -> qrcp == Subroutines::QRCP::bqrrp) {

        #if !defined(__APPLE__)
        if (n <= 2000) {
            this->bqrrp_block_ratio = 1.0;
        } else if (n <= 8000) {
            this->bqrrp_block_ratio = 0.5;
        } else {
            this->bqrrp_block_ratio = (T) 1 / (T) 32;
        }

        RandLAPACK::BQRRP<T, RNG> BQRRP(false, n * this->bqrrp_block_ratio);
        BQRRP.call(d, n, A_hat, d, 1.0, tau, J, state);
        #endif
    } else {
        lapack::geqp3(d, n, A_hat, d, J, tau);
    }

    if(this -> timing) {
        qrcp_t_stop = steady_clock::now();
        rank_reveal_t_start = steady_clock::now();
    }

    // Check if the input is all zeros
    if (!A_hat[0]) {
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

    /// Extracting a k by k R representation
    T* R_sp  = R;
    lapack::lacpy(MatrixType::Upper, k, k, A_hat, d, R_sp, ldr);

    if(this -> timing)
        a_mod_piv_t_start = steady_clock::now();

    // Swap k columns of A with pivots from J
    blas::copy(n, J, 1, J_buf.data(), 1);
    util::col_swap(m, n, k, A, lda, J_buf);

    if(this -> timing) {
        a_mod_piv_t_stop = steady_clock::now();
        a_mod_trsm_t_start = steady_clock::now();
    }

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp, ldr, A, lda);

    if(this -> timing) {
        a_mod_trsm_t_stop = steady_clock::now();
        cholqr_t_start = steady_clock::now();
    }

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A, lda, 0.0, R_sp, ldr);
    lapack::potrf(Uplo::Upper, k, R_sp, ldr);

    // Re-estimate rank after we have the R-factor form Cholesky QR.
    // The strategy here is the same as in naive rank estimation.
    // This also automatically takes care of any potential failures in Cholesky factorization.
    // Note that the diagonal of R_sp may not be sorted, so we need to keep the running max/min
    // We expect the loss in the orthogonality of Q to be approximately equal to u * cond(R_sp)^2, where u is the unit roundoff for the numerical type T.
    new_rank = k;
    running_max = R_sp[0];
    running_min = R_sp[0];
    T cond_threshold = std::sqrt(this->eps / std::numeric_limits<T>::epsilon());

    for(i = 0; i < k; ++i) {
        curr_entry = std::abs(R_sp[i * ldr + i]);
        running_max = std::max(running_max, curr_entry);
        running_min = std::min(running_min, curr_entry);
        if((running_min * cond_threshold < running_max) && i > 1) {
            new_rank = i - 1;
            break;
        }
    }

    // Set the rank parameter to the value computed a posteriori.
    this->rank = new_rank;

    // Obtain the output Q-factor
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, 1.0, R_sp, ldr, A, lda);

    if(this -> timing)
        cholqr_t_stop = steady_clock::now();

    if (!this->orthogonalization) {
        // Get the final R-factor -- undoing the preconditioning
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, 1.0, A_hat, d, R_sp, ldr); 
    } 
    /*
    else if (new_rank != n) {
        // Complete the orthonormal set
        // Need to write gram-schmidt - fill in later.
    } 
    */


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

    delete[] A_hat;
    delete[] tau;

    return 0;
}
} // end namespace RandLAPACK
