#ifndef randlapack_cqrrpt_h
#define randlapack_cqrrpt_h

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_hqrrp.hh"

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
            std::vector<T> &A,
            int64_t d,
            std::vector<T> &R,
            std::vector<int64_t> &J,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class CQRRPT : public CQRRPTalg<T, RNG> {
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
        CQRRPT(
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
            std::vector<T> &A,
            int64_t d,
            std::vector<T> &R,
            std::vector<int64_t> &J,
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
int CQRRPT<T, RNG>::call(
    int64_t m,
    int64_t n,
    std::vector<T> &A,
    int64_t d,
    std::vector<T> &R,
    std::vector<int64_t> &J,
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

    std::vector<T> A_hat;
    std::vector<T> tau;
    std::vector<T> R_sp;


    T* A_dat       = A.data();
    T* A_hat_dat   = util::upsize(d * n, A_hat);
    T* tau_dat     = util::upsize(n, tau);
    J.resize(n);
    int64_t* J_dat = J.data();

    int i;
    int64_t k = n;
    // A constant for initial rank estimation.
    T eps_initial_rank_estimation = 2 * std::pow(std::numeric_limits<T>::epsilon(), 0.95);
    // Variables for a posteriori rank estimation.
    int64_t new_rank;
    T running_max, running_min, curr_entry;

    if(this -> timing)
        saso_t_start = high_resolution_clock::now();
    
    /// Generating a SASO
    RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = m, .vec_nnz = this->nnz};
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    /// Applying a SASO
    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, 1.0, S, 0, 0, A.data(), m, 0.0, A_hat_dat, d
    );

    if(this -> timing) {
        saso_t_stop = high_resolution_clock::now();
        qrcp_t_start = high_resolution_clock::now();
    }

    /// Performing QRCP on a sketch
    if(this->no_hqrrp) {
        lapack::geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);
    } else {
        std::iota(J.begin(), J.end(), 1);
        hqrrp(d, n, A_hat_dat, d, J_dat, tau_dat, this->nb_alg, this->oversampling, this->panel_pivoting, this->use_cholqr, state, (T*) nullptr);
    }

    if(this -> timing)
        qrcp_t_stop = high_resolution_clock::now();

    T* R_dat = util::upsize(n * n, R);
    
    if(this -> timing)
        rank_reveal_t_start = high_resolution_clock::now();

    /// Using naive rank estimation to ensure that R used for preconditioning is invertible.
    /// The actual rank estimate k will be computed a posteriori. 
    /// Using R[i,i] to approximate the i-th singular value of A_hat. 
    /// Truncate at the largest i where R[i,i] / R[0,0] >= eps.
    for(i = 0; i < n; ++i) {
        if(std::abs(A_hat_dat[i * d + i]) / std::abs(A_hat_dat[0]) < eps_initial_rank_estimation) {
            k = i;
            break;
        }
    }
    this->rank = k;

    if(this -> timing)
        rank_reveal_t_stop = high_resolution_clock::now();

    // Allocating space for a preconditioner buffer.
    T* R_sp_dat  = util::upsize(k * k, R_sp);
    /// Extracting a k by k upper-triangular R.
    lapack::lacpy(MatrixType::Upper, k, k, A_hat_dat, d, R_sp_dat, k);
    /// Extracting a k by n R representation (k by k upper-triangular, rest - general)
    lapack::lacpy(MatrixType::Upper, k, k, A_hat_dat, d, R_dat, n);
    lapack::lacpy(MatrixType::General, k, n - k, &A_hat_dat[d * k], d, &R_dat[n * k], n);

    if(this -> timing)
        a_mod_piv_t_start = high_resolution_clock::now();

    // Swap k columns of A with pivots from J
    util::col_swap(m, n, k, A_dat, m, J);

    if(this -> timing) {
        a_mod_piv_t_stop = high_resolution_clock::now();
        a_mod_trsm_t_start = high_resolution_clock::now();
    }

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    if(this -> timing) {
        a_mod_trsm_t_stop = high_resolution_clock::now();
        cholqr_t_start = high_resolution_clock::now();
    }

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A_dat, m, 0.0, R_sp_dat, k);
    lapack::potrf(Uplo::Upper, k, R_sp_dat, k);

    // Re-estimate rank after we have the R-factor form Cholesky QR.
    // The strategy here is the same as in naive rank estimation.
    // This also automatically takes care of any potentical failures in Cholesky factorization.
    // Note that the diagonal of R_sp_dat may not be sorted, so we need to keep the running max/min
    // We expect the loss in the orthogonality of Q to be approximately equal to u * cond(R_sp_dat)^2, where u is the unit roundoff for the numerical type T.
    new_rank = k;
    running_max = R_sp_dat[0];
    running_min = R_sp_dat[0];
    
    for(i = 0; i < k; ++i) {
        curr_entry = std::abs(R_sp[i * k + i]);
        if(curr_entry > running_max) running_max = curr_entry;
        if(curr_entry < running_min) running_max = running_min;
        if(running_max / running_min >= std::sqrt(this->eps / std::numeric_limits<T>::epsilon())) {
            new_rank = i - 1;
            break;
        }
    }

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, new_rank, 1.0, R_sp_dat, k, A_dat, m);

    if(this -> timing)
        cholqr_t_stop = high_resolution_clock::now();

    // Get the final R-factor.
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, new_rank, n, 1.0, R_sp_dat, k, R_dat, n);

    // Set the rank parameter to the value comuted a posteriori.
    this->rank = k;

    if(this -> timing) {
        saso_t_dur        = duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
        qrcp_t_dur        = duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
        rank_reveal_t_dur = duration_cast<microseconds>(rank_reveal_t_stop - rank_reveal_t_start).count();
        a_mod_piv_t_dur   = duration_cast<microseconds>(a_mod_piv_t_stop - a_mod_piv_t_start).count();
        a_mod_trsm_t_dur  = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
        cholqr_t_dur      = duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();

        total_t_stop = high_resolution_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest  = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqr_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqr_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, t_rest, total_t_dur};
    }
    return 0;
}
} // end namespace RandLAPACK
#endif