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
            panel_pivoting = 1;
            naive_rank_estimate = 1;
            use_fro_norm = 1;
            cond_check = 0;
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
        bool cond_check;
        T eps;
        int64_t rank;

        // 10 entries
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // Buffers
        std::vector<T> A_hat;
        std::vector<T> tau;
        std::vector<T> R_sp;

        // HQRRP-related
        int no_hqrrp;
        int64_t nb_alg;
        int64_t oversampling;
        int64_t panel_pivoting;

        // Rank estimate-related
        int naive_rank_estimate;
        int use_fro_norm;

        // Preconditioning-related
        T cond_num_A_pre;
        T cond_num_A_norm_pre;
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
    //-------TIMING VARS--------/
    high_resolution_clock::time_point saso_t_stop;
    high_resolution_clock::time_point saso_t_start;
    long saso_t_dur = 0;

    high_resolution_clock::time_point qrcp_t_start;
    high_resolution_clock::time_point qrcp_t_stop;
    long qrcp_t_dur = 0;

    high_resolution_clock::time_point rank_reveal_t_start;
    high_resolution_clock::time_point rank_reveal_t_stop;
    long rank_reveal_t_dur = 0;

    high_resolution_clock::time_point cholqr_t_start;
    high_resolution_clock::time_point cholqr_t_stop;
    long cholqr_t_dur = 0;

    high_resolution_clock::time_point a_mod_piv_t_start;
    high_resolution_clock::time_point a_mod_piv_t_stop;
    long a_mod_piv_t_dur = 0;

    high_resolution_clock::time_point a_mod_trsm_t_start;
    high_resolution_clock::time_point a_mod_trsm_t_stop;
    long a_mod_trsm_t_dur = 0;

    high_resolution_clock::time_point copy_t_start;
    high_resolution_clock::time_point copy_t_stop;
    long copy_t_dur = 0;

    high_resolution_clock::time_point resize_t_start;
    high_resolution_clock::time_point resize_t_stop;
    long resize_t_dur = 0;

    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;
    long total_t_dur = 0;

    if(this -> timing) {
        total_t_start = high_resolution_clock::now();
        resize_t_start = high_resolution_clock::now();
    }

    T* A_dat       = A.data();
    T* A_hat_dat   = util::upsize(d * n, this->A_hat);
    T* tau_dat     = util::upsize(n, this->tau);
    J.resize(n);
    int64_t* J_dat = J.data();

    if(this -> timing) {
        resize_t_stop = high_resolution_clock::now();
        resize_t_dur = duration_cast<microseconds>(resize_t_stop - resize_t_start).count();
        saso_t_start = high_resolution_clock::now();
    }
    
    RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = m, .vec_nnz = this->nnz};
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, 1.0, S, 0, 0, A.data(), m, 0.0, A_hat_dat, d
    );

    if(this -> timing) {
        saso_t_stop = high_resolution_clock::now();
        qrcp_t_start = high_resolution_clock::now();
    }

    // QRCP - add failure condition
    if(this->no_hqrrp) {
        lapack::geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);
    }
    else {
        std::iota(J.begin(), J.end(), 1);
        hqrrp(d, n, A_hat_dat, d, J_dat, tau_dat, this->nb_alg, this->oversampling, this->panel_pivoting, state);
    }

    if(this -> timing) {
        qrcp_t_stop = high_resolution_clock::now();
        resize_t_start = high_resolution_clock::now();
    }

    T* R_dat = util::upsize(n * n, R);
    
    if(this -> timing) {
        resize_t_stop = high_resolution_clock::now();
        resize_t_dur  += duration_cast<microseconds>(resize_t_stop - resize_t_start).count();
        rank_reveal_t_start = high_resolution_clock::now();
    }

    int64_t k = n;
    int i;
    T eps_initial_rank_estimation = 2 * std::pow(std::numeric_limits<T>::epsilon(), 0.95);
    if(this->naive_rank_estimate) {
        /// Using R[i,i] to approximate the i-th singular value of A_hat. 
        /// Truncate at the largest i where R[i,i] / R[0,0] >= eps.
        for(i = 0; i < n; ++i) {
            if(std::abs(A_hat_dat[i * d + i]) / std::abs(A_hat_dat[0]) < eps_initial_rank_estimation) {
                k = i;
                break;
            }
        }
        this->rank = k;
    }
    else {
        // Oleg's scheme for rank estimation
        for(i = 0; i < n; ++i) {
            // copy over an upper-triangular matrix R
            // from col-maj to row-maj format
            blas::copy(i + 1, &A_hat_dat[i * d], 1, &R_dat[i], n);
        }

        T norm_R = 0.0;
        if(this->use_fro_norm) {
            // find fro norm of the full R
            norm_R = lapack::lange(Norm::Fro, n, n, R_dat, n);
        } else {
            // find l2 norm of the full R
            norm_R = RandLAPACK::util::estimate_spectral_norm(n, n, R_dat, 10, state);
            eps_initial_rank_estimation = 5 * eps_initial_rank_estimation;
        }

        T norm_R_sub = lapack::lange(Norm::Fro, 1, n, &R_dat[(n - 1) * n], 1);
        // Check if R is full column rank checking if||A[n - 1:, n - 1:]||_F > tau_trunk * ||A||_F
        if ((norm_R_sub > eps_initial_rank_estimation * norm_R)) {
            k = n;
        } else {
            k = RandLAPACK::util::rank_search_binary(0, n + 1, std::floor(n / 2), n, norm_R, eps_initial_rank_estimation, R_dat);
        }

        this->rank = k;
        // Clear R
        std::fill(R.begin(), R.end(), 0.0);
    }

    if(this -> timing) {
        rank_reveal_t_stop = high_resolution_clock::now();
        resize_t_start = high_resolution_clock::now();
    }

    T* R_sp_dat  = util::upsize(k * k, this->R_sp);

    if(this -> timing) {
        resize_t_stop = high_resolution_clock::now();
        copy_t_start = high_resolution_clock::now();
    }

    // performing a copy column by column
    for(i = 0; i < k; ++i) {
        // extract k by k R
        blas::copy(i + 1, &A_hat_dat[i * d], 1, &R_sp_dat[i * k], 1);
        // extract full R
        blas::copy(i + 1, &A_hat_dat[i * d], 1, &R_dat[i * k], 1);
    }
    for(i = k; i < n; ++i) {
        blas::copy(k, &A_hat_dat[i * d], 1, &R_dat[i * k], 1);
    }

    if(this -> timing) {
        copy_t_stop = high_resolution_clock::now();
        a_mod_piv_t_start = high_resolution_clock::now();
    }

    // Swap k columns of A with pivots from J
    util::col_swap(m, n, k, A_dat, J);

    if(this -> timing) {
        a_mod_piv_t_stop = high_resolution_clock::now();
        a_mod_trsm_t_start = high_resolution_clock::now();
    }

    // A_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    if(this -> timing)
        a_mod_trsm_t_stop = high_resolution_clock::now();

    // Check the condition number of a A_pre
    if(this -> cond_check)
    {
        // Check cond(A_pre)
        std::vector<T> A_pre_cpy;
        std::vector<T> s;
        this->cond_num_A_pre = RandLAPACK::util::cond_num_check(m, k, A, A_pre_cpy, s, false);

        A_pre_cpy.clear();
        s.clear();
        // Check cond(normc(A_pre))
        std::vector<T> A_norm_pre;
        RandLAPACK::util::normc(m, k, A, A_norm_pre);
        this->cond_num_A_norm_pre = RandLAPACK::util::cond_num_check(m, k, A_norm_pre, A_pre_cpy, s, false);
    }

    if(this -> timing)
        cholqr_t_start = high_resolution_clock::now();

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A_dat, m, 0.0, R_sp_dat, k);
    if(lapack::potrf(Uplo::Upper, k, R_sp_dat, k)){
        if(this->verbosity)
            throw std::runtime_error("Cholesky decomposition failed.");
        return 1;
    }

    // Re-estimate rank after we have the R-factor form Cholesky QR.
    // The strategy here is the same as in naive rank estimation.
    // This also automatically takes care of any potentical failures in Cholesky factorization.
    // Note that the diagonal of R_sp_dat may not be sorted, so we need to keep the running max/min
    // We expect the loss in the orthogonality of Q to be approximately equal to u * cond(R_sp_dat)^2, where u is the unit roundoff for the numerical type T.
    int64_t new_rank = k;
    T running_max = R_sp_dat[0];
    T running_min = R_sp_dat[0];
    T curr_entry;
    
    for(i = 0; i < k; ++i) {
        curr_entry = std::abs(R_sp[i * k + i]);
        
        if(curr_entry > running_max) running_max = curr_entry;
        if(curr_entry < running_min) running_max = running_min;

        if(running_max / running_min >= std::sqrt(this->eps / std::numeric_limits<T>::epsilon())) {
            new_rank = i - 1;
            break;
        }
    }

    // Beware of that R_sp and R have k rows and need to be downsized by rows
    RandLAPACK::util::row_resize(k, k, R_sp, new_rank);
    RandLAPACK::util::row_resize(k, n, R, new_rank);
    
    k = new_rank;
    this->rank = k;

    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    if(this -> timing)
        cholqr_t_stop = high_resolution_clock::now();

    // Get R
    // trmm
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, k, n, 1.0, R_sp_dat, k, R_dat, k);

    if(this -> timing) {
        saso_t_dur        = duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
        qrcp_t_dur        = duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
        rank_reveal_t_dur = duration_cast<microseconds>(rank_reveal_t_stop - rank_reveal_t_start).count();
        resize_t_dur     += duration_cast<microseconds>(resize_t_stop - resize_t_start).count();
        copy_t_dur       += duration_cast<microseconds>(copy_t_stop - copy_t_start).count();
        a_mod_piv_t_dur   = duration_cast<microseconds>(a_mod_piv_t_stop - a_mod_piv_t_start).count();
        a_mod_trsm_t_dur  = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
        cholqr_t_dur      = duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();

        total_t_stop = high_resolution_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest  = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqr_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur + copy_t_dur + resize_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqr_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, copy_t_dur, resize_t_dur, t_rest, total_t_dur};
    }
    return 0;
}





















template <typename T, typename RNG>
class CQRRP_blocked : public CQRRPTalg<T, RNG> {
    public:

        CQRRP_blocked(
            bool verb,
            bool time_subroutines,
            T ep,
            int64_t b_sz
        ) {
            verbosity = verb;
            timing = time_subroutines;
            eps = ep;
            block_size = b_sz;
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
        bool cond_check;
        RandBLAS::RNGState<RNG> state;
        T eps;
        int64_t rank;
        int64_t block_size;

        // 10 entries
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // Buffers
        std::vector<T> A_hat;
        std::vector<T> tau;
        std::vector<T> R_sk;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRP_blocked<T, RNG>::call(
    int64_t m,
    int64_t n,
    std::vector<T> &A,
    int64_t d,
    std::vector<T> &R,
    std::vector<int64_t> &J,
    RandBLAS::RNGState<RNG> &state
){

    int64_t rows = m;
    int64_t cols = n;
    int64_t b_sz = this->block_size;

    // Originam m by n matrix A, will be "changing" size to rows by cols at every iteration
    T* A_dat       = A.data();
    // We now actually need the full Q vector
    std::vector Q (m * m, 0.0);
    T* Q_dat       = Q.data(); 
    // The full R-factor, must have pre-allocated size of m by n;
    T* R_dat       = util::upsize(m * n, R); 
    // Sketching buffer that is accessed at every iteration. Must be b_sz by b_sz;
    T* R_sk_dat    = util::upsize(b_sz * b_sz, this->R_sk); 
    // Buffer array for sketching. Max size d by n, actual size d by col. Needed until R-factor is extracted.
    T* A_hat_dat   = util::upsize(d * n, this->A_hat);
    T* tau_dat     = util::upsize(n, this->tau);
    J.resize(n);
    int64_t* J_dat = J.data();
    // This is a buffer for the "Q_full" a row by col matrix, which will be found 
    // via applying householder reconstruction to Q_econ, which itself is found by chol_QR on A_pre
    std::vector<T> Q_full(m * n, 0.0);
    T* Q_full_dat = Q_full.data();
    // We will need a copy of A_piv to use in gemms
    std::vector<T> A_piv(m * n, 0.0);
    T* A_piv_dat = A_piv.data();
    // We will need a copy of A_pre to use in gemms
    std::vector<T> A_pre(m * b_sz, 0.0);
    T* A_pre_dat = A_pre.data();
    // These are required for householder reconstruction
    std::vector<T> D_1(n, 0.0);
    std::vector<T> T_1(n * n, 0.0);
    
    // We will be operating on this at every iteration;
    std::vector<int64_t> J_buffer (n, 0);
    int64_t* J_buffer_dat = J_buffer.data();


    char name [] = "A";
    


    std::vector<T> R11 (b_sz * b_sz, 0.0);
    T* R11_dat = R11.data();

    std::vector<T> R12 (b_sz * (n - b_sz), 0.0);
    T* R12_dat = R12.data();

    for(int iter = 0; iter < n / b_sz; ++iter)
    {
        RandBLAS::util::print_colmaj(rows, cols, A.data(), name);

        // Zero-out data
        std::fill(J_buffer.begin(), J_buffer.end(), 0);
        std::fill(this->tau.begin(), this->tau.end(), 0.0);

        // Skethcing in an embedding regime
        RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = rows, .vec_nnz = this->nnz};
        RandBLAS::SparseSkOp<T, RNG> S(DS, state);
        state = RandBLAS::fill_sparse(S);

        RandBLAS::sketch_general(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, cols, rows, 1.0, S, 0, 0, A.data(), rows, 0.0, A_hat_dat, d
        );

        // QRCP
        lapack::geqp3(d, cols, A_hat_dat, d, J_buffer_dat, tau_dat);

        printf("Printing out J_buffer\n");
        for(int i = 0; i < cols; ++i)  
            printf("%ld\n", J_buffer[i]);





        char name2 [] = "R";
        // Need to premute trailing columns of the full R-factor
        if(iter != 0)
            util::col_swap(m, cols, cols, &R_dat[m * (b_sz * iter)], J_buffer);

        // extract b_sz by b_sz R_sk
        for(int i = 0; i < b_sz; ++i)
            blas::copy(i + 1, &A_hat_dat[i * d], 1, &R_sk_dat[i * b_sz], 1);


        char nameRsk [] = "R_sk";
            RandBLAS::util::print_colmaj(b_sz, b_sz, R_sk.data(), nameRsk);


        // A_piv = Need to pivot full matrix A
        // This is a wors by cols permuted version of the current A
        util::col_swap(rows, cols, cols, A_dat, J_buffer);

        char nameA_piv [] = "A_piv"; 
        RandBLAS::util::print_colmaj(rows, cols, A_dat, nameA_piv);

        // Below two copies are required, as later there is an expression where A, Q_full and A_piv are all incorporated
        // We need to save A_piv for the future
        lapack::lacpy(MatrixType::General, rows, cols, A_dat, rows, Q_full_dat, rows);
        // We will need this copy to use in gemms
        lapack::lacpy(MatrixType::General, rows, cols, A_dat, rows, A_piv_dat, rows);

        // A_pre = AJ(:, 1:b_sz) * R_sk
        // This is a preconditioned rows by b_sz version of the current A, written into Q_full because of the way trsm works
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, R_sk_dat, b_sz, Q_full_dat, rows);

        // We need to save A_pre, is required for computations below 
        lapack::lacpy(MatrixType::General, rows, b_sz, Q_full_dat, rows, A_pre_dat, rows);

        // Performing Cholesky QR
        std::vector<T> R_econ;
        T* R_econ_dat = util::upsize(b_sz * b_sz, R_econ);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, b_sz, rows, 1.0, Q_full_dat, rows, 0.0, R_econ_dat, b_sz);
        lapack::potrf(Uplo::Upper, b_sz, R_econ_dat, b_sz);

        // Compute Q_econ
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, R_econ_dat, b_sz, Q_full_dat, rows);

        // Find Q_full using Householder reconstruction. 
        // Remember that Q_econ has b_sz orthonormal columns
        lapack::orhr_col(rows, b_sz, b_sz, Q_full_dat, rows, T_1.data(), b_sz, D_1.data());
                    /*
                    std::vector<T> I_buf(rows * rows, 0.0);
                    RandLAPACK::util::eye(rows, rows, I_buf);

                    lapack::gemqrt(Side::Left, Op::NoTrans, m, m, b_sz, b_sz, Q_full_dat, m, T_1.data(), b_sz, I_buf.data(), m);

                    char name_QQ [] = "Q_just_assembled";
                    RandBLAS::util::print_colmaj(rows, rows, I_buf.data(), name_QQ);
                    */

        // Remember that even though Q_full is represented by "b_sz" columns, it is actually of size "rows by rows."
        // Compute R11_full = Q_full' * A_pre - gets written into A_pre space
        lapack::gemqrt(Side::Left, Op::Trans, rows, b_sz, b_sz, b_sz, Q_full_dat, rows, T_1.data(), b_sz, A_pre_dat, rows);

        // Compute R11 = R11_full(1:b_sz, :) * R_sk.
        // We will have to resize A_pre_dat down to b_sz by b_sz
        RandLAPACK::util::row_resize(rows, b_sz, A_pre, b_sz);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, b_sz, b_sz, 1.0, A_pre_dat, b_sz, R_sk_dat, b_sz, 0.0, R11.data(), b_sz);

        // Looks like we can substitute the two multiplications with a single one:
        // A_piv is a rows by cols matrix, its last "cols-b_sz" columns and "rows" rows will be updated.
        // The first b_sz rows will represent R12
        // The last rows-b_sz rows will represent the new A

        lapack::gemqrt(Side::Left, Op::Trans, rows, cols - b_sz, b_sz, b_sz,  Q_full_dat, rows, T_1.data(), b_sz, &A_piv_dat[rows * b_sz], rows);

        //char nameA_piv [] = "A_piv that stores R12 and Updated A"; 
        //RandBLAS::util::print_colmaj(rows, cols, A_piv_dat, nameA_piv);

        for(int i = 0; i < (cols - b_sz); ++i) {
            blas::copy(b_sz, &A_piv_dat[rows * (b_sz + i)], 1, &R12[i * b_sz], 1);
            blas::copy(rows - b_sz, &A_piv_dat[rows * (b_sz + i) + b_sz], 1, &A[i * (rows - b_sz)], 1);
        }

        //char name3 [] = "R11";
        //RandBLAS::util::print_colmaj(b_sz, b_sz, R11.data(), name3);


        // Updating Q, Pivots
        if(iter == 0) {
            blas::copy(cols, J_buffer_dat, 1, J_dat, 1);
            RandLAPACK::util::eye(rows, rows, Q);
            lapack::gemqrt(Side::Right, Op::NoTrans, rows, rows, b_sz, b_sz, Q_full_dat, rows, T_1.data(), b_sz, Q_dat, rows);
        } else {
            
            // THINK ABOUT HOW COMPLEX INDEX EXPRESSION RELATE TO ROWS/COLS
            lapack::gemqrt(Side::Right, Op::NoTrans, m, rows, b_sz, b_sz, Q_full_dat, rows, T_1.data(), b_sz, &Q_dat[m * (b_sz * iter)], m);
            

            // Find a simpler way to do this
            RandLAPACK::util::col_swap<T>(cols, cols, &J_dat[iter * b_sz], J_buffer);
        }

        // Updating R-factor. The full R-factor is m by n
        for(int i = iter * b_sz, j = -1, k = -1; i < n; ++i) {
            if (i < (iter + 1) * b_sz) {
                blas::copy(b_sz, &R11_dat[b_sz * ++j], 1, &R_dat[(m * i) + (iter * b_sz)], 1);
            } else {
                blas::copy(b_sz, &R12_dat[b_sz * ++k], 1, &R_dat[(m * i) + (iter * b_sz)], 1);
            }
        }

        //char name1 [] = "Q";
        //RandBLAS::util::print_colmaj(m, m, Q.data(), name1);
        //RandBLAS::util::print_colmaj(m, n, R.data(), name2);

        char nameA [] = "Updated A";
        RandBLAS::util::print_colmaj(m, n, A.data(), nameA);

        RandBLAS::util::print_colmaj(m, n, R.data(), name2);

        char nameQ [] = "Q";
        RandBLAS::util::print_colmaj(m, m, Q.data(), nameQ);


        printf("Printing out Full J\n");
        for(int i = 0; i < n; ++i)
            printf("%ld\n", J[i]);

        // MATRIX A IS CORRECTLY UPDATED AT THE END
        // NEED TO CHECK WHAT HAPPENS AT ITERATION TWO

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;
    }
    lapack::lacpy(MatrixType::General, m, n, Q.data(), m, A.data(), m);
    char nameQ [] = "Q";
    RandBLAS::util::print_colmaj(m, n, A.data(), nameQ);
    this -> rank = n;
    RandLAPACK::util::row_resize(m, n, R, n);

    return 0;
}

} // end namespace RandLAPACK
#endif