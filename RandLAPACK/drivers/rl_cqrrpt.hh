#ifndef randlapack_cqrrpt_h
#define randlapack_cqrrpt_h

#include "rl_cqrrpt.hh"
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

template <typename T>
class CQRRPTalg {
    public:

        virtual ~CQRRPTalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t d,
            std::vector<T>& R,
            std::vector<int64_t>& J
        ) = 0;
};

template <typename T, typename RNG>
class CQRRPT : public CQRRPTalg<T> {
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
            RandBLAS::base::RNGState<RNG> st,
            T ep
        ) {
            verbosity = verb;
            timing = time_subroutines;
            state = st;
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
            std::vector<T>& A,
            int64_t d,
            std::vector<T>& R,
            std::vector<int64_t>& J
        ) override;

    public:
        bool verbosity;
        bool timing;
        bool cond_check;
        RandBLAS::base::RNGState<RNG> state;
        T eps;
        int64_t rank;
        int64_t b_sz;

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
    std::vector<T>& A,
    int64_t d,
    std::vector<T>& R,
    std::vector<int64_t>& J
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
    
    RandBLAS::sparse::SparseDist DS = {RandBLAS::sparse::SparseDistName::SASO, d, m, this->nnz};
    RandBLAS::sparse::SparseSkOp<T> S(DS, state, NULL, NULL, NULL);
    RandBLAS::sparse::fill_sparse(S);

    RandBLAS::sparse::lskges<T, RandBLAS::sparse::SparseSkOp<T>>(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, 1.0, S, 0, 0, A.data(), m, 0.0, A_hat_dat, d);

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
        hqrrp(d, n, A_hat_dat, d, J_dat, tau_dat, this->nb_alg, this->oversampling, this->panel_pivoting, this->state);
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
    if(this->naive_rank_estimate) {
        /// Using R[i,i] to approximate the i-th singular value of A_hat. 
        /// Truncate at the largest i where R[i,i] / R[0,0] >= eps.
        for(i = 0; i < n; ++i) {
            if(std::abs(A_hat_dat[i * d + i]) / std::abs(A_hat_dat[0]) < this->eps) {
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
            this->eps = 5 * this->eps;
        }

        T norm_R_sub = lapack::lange(Norm::Fro, 1, n, &R_dat[(n - 1) * n], 1);
        // Check if R is full column rank checking if||A[n - 1:, n - 1:]||_F > tau_trunk * ||A||_F
        if ((norm_R_sub > this->eps * norm_R)) {
            k = n;
        } else {
            k = RandLAPACK::util::rank_search_binary(0, n + 1, std::floor(n / 2), n, norm_R, this->eps, R_dat);
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
    util::col_swap(m, n, k, A, J);

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
            printf("CHOLESKY FACTORIZATION FAILED.\n");
        return 1;
    }

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
        cholqr_t_dur    = duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();

        total_t_stop = high_resolution_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqr_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur + copy_t_dur + resize_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqr_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, copy_t_dur, resize_t_dur, t_rest, total_t_dur};
    }

    return 0;
}

} // end namespace RandLAPACK
#endif