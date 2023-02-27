#ifndef randlapack_cholqrcp_h
#define randlapack_cholqrcp_h

#include "rl_cholqrcp.hh"
#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>
#include <chrono>

using namespace std::chrono;

namespace RandLAPACK {

template <typename T>
class CholQRCPalg {
    public:

        virtual ~CholQRCPalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t d,
            std::vector<T>& R,
            std::vector<int64_t>& J
        ) = 0;
};

template <typename T>
class CholQRCP : public CholQRCPalg<T> {
    public:

        // Constructor
        CholQRCP(
            bool verb,
            bool t,
            uint32_t sd,
            T ep
        ) {
            verbosity = verb;
            timing = t;
            seed = sd;
            eps = ep;
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
        int CholQRCP1(
            int64_t m,
            int64_t n,
            std::vector<T>& A,
            int64_t d,
            std::vector<T>& R,
            std::vector<int64_t>& J
        );

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
        uint32_t seed;
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
};

// -----------------------------------------------------------------------------
template <typename T>
int CholQRCP<T>::call(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t d,
    std::vector<T>& R,
    std::vector<int64_t>& J
) {
    int termination = CholQRCP1(m, n, A, d, R, J);

    if(this->verbosity) {
        switch(termination) {
        case 1:
            printf("\nCholQRCP TERMINATED VIA: 1.\n");
            break;
        case 0:
            printf("\nCholQRCP TERMINATED VIA: normal termination.\n");
            break;
        }
    }
    return termination;
}

// -----------------------------------------------------------------------------
template <typename T>
int CholQRCP<T>::CholQRCP1(
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

    high_resolution_clock::time_point cholqrcp_t_start;
    high_resolution_clock::time_point cholqrcp_t_stop;
    long cholqrcp_t_dur = 0;

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
    RandBLAS::sparse::SparseSkOp<T> S(DS, this->seed, 0, NULL, NULL, NULL);
    RandBLAS::sparse::fill_sparse(S);

    RandBLAS::sparse::lskges<T, RandBLAS::sparse::SparseSkOp<T>>(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, n, m, 1.0, S, 0, 0, A.data(), m, 0.0, A_hat_dat, d);

    if(this -> timing) {
        saso_t_stop = high_resolution_clock::now();
        qrcp_t_start = high_resolution_clock::now();
    }

    // QRCP - add failure condition
<<<<<<< HEAD:RandLAPACK/drivers/rl_cholqrcp.hh
    lapack::geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);
=======
    if(this->no_hqrrp) {
        geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);
    }
    else {
        // Add below line to ensure best HQRRP performance
        omp_set_num_threads(8);
        std::iota(J.begin(), J.end(), 1);
        HQRRP::hqrrp(d, n, (double *)A_hat_dat, d, J_dat, (double *)tau_dat, this->nb_alg, this->oversampling, this->panel_pivoting);
        //HQRRP::dgeqpr(d, n, (double *)A_hat_dat, d, J_dat, (double *)tau_dat);
    }
<<<<<<< HEAD:RandLAPACK/drivers/rl_cholqrcp.hh
    omp_set_num_threads(36);
>>>>>>> 39fdf89 (Changing the number of threads used by HQRRP.):src/drivers/cholqrcp.cc
=======
    // Continue with the best threading for the remainder of routines
<<<<<<< HEAD:RandLAPACK/drivers/rl_cholqrcp.hh
    //omp_set_num_threads(36);
>>>>>>> 01b03ab (Preparing to run HQRRP-backed CholQRCP without thread optimization, but with b_sz of 32.):src/drivers/cholqrcp.cc
=======
    omp_set_num_threads(36);
>>>>>>> 873b209 (Re-running best case for HQRRP):src/drivers/cholqrcp.cc

    if(this -> timing) {
        qrcp_t_stop = high_resolution_clock::now();
        rank_reveal_t_start = high_resolution_clock::now();
    }

    // Find rank
    int k = n;
    int i;
    for(i = 0; i < n; ++i) {
        if(std::abs(A_hat_dat[i * d + i]) < this->eps) {
            k = i;
            break;
        }
    }
    this->rank = k;

    if(this -> timing) {
        rank_reveal_t_stop = high_resolution_clock::now();
        resize_t_start = high_resolution_clock::now();
    }

    T* R_sp_dat  = util::upsize(k * k, this->R_sp);
    T* R_dat     = util::upsize(k * n, R);

    if(this -> timing) {
        resize_t_stop = high_resolution_clock::now();
        copy_t_start = high_resolution_clock::now();
    }

    // extract k by k R
    // Copy data over to R_sp_dat col by col
    for(i = 0; i < k; ++i) {
        blas::copy(i + 1, &A_hat_dat[i * d], 1, &R_sp_dat[i * k], 1);
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

    // A_sp_pre * R_sp = AP
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    if(this -> timing)
        a_mod_trsm_t_stop = high_resolution_clock::now();

    if(this -> timing)
        cholqrcp_t_start = high_resolution_clock::now();

    // Do Cholesky QR
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A_dat, m, 0.0, R_sp_dat, k);
    lapack::potrf(Uplo::Upper, k, R_sp_dat, k);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    if(this -> timing)
        cholqrcp_t_stop = high_resolution_clock::now();

    // Get R
    // trmm
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, k, n, 1.0, R_sp_dat, k, R_dat, k);

    if(this -> timing) {
        saso_t_dur        = duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
        qrcp_t_dur        = duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
        rank_reveal_t_dur = duration_cast<microseconds>(rank_reveal_t_stop - rank_reveal_t_start).count();
        resize_t_dur     += duration_cast<microseconds>(resize_t_stop - resize_t_start).count();
        copy_t_dur        = duration_cast<microseconds>(copy_t_stop - copy_t_start).count();
        a_mod_piv_t_dur   = duration_cast<microseconds>(a_mod_piv_t_stop - a_mod_piv_t_start).count();
        a_mod_trsm_t_dur  = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
        cholqrcp_t_dur    = duration_cast<microseconds>(cholqrcp_t_stop - cholqrcp_t_start).count();

        total_t_stop = high_resolution_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqrcp_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur + copy_t_dur + resize_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqrcp_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, copy_t_dur, resize_t_dur, t_rest, total_t_dur};
    }

    return 0;
}

} // end namespace RandLAPACK
#endif
