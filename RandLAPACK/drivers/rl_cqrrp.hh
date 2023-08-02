#ifndef randlapack_cqrrp_h
#define randlapack_cqrrp_h

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
class CQRRPalg {
    public:

        virtual ~CQRRPalg() {}

        virtual int call(
            int64_t m,
            int64_t n,
            std::vector<T> &A,
            int64_t d_factor,
            std::vector<T> &Q,
            std::vector<T> &R,
            std::vector<int64_t> &J,
            RandBLAS::RNGState<RNG> &state
        ) = 0;
};

template <typename T, typename RNG>
class CQRRP_blocked : public CQRRPalg<T, RNG> {
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
        /// @param[in] d_factor
        ///     Embedding dimension of a sketch factor, m >= (d_factor * n) >= n.
        ///
        /// @param[in] Q
        ///     Represents an orthonormal Q factor of QR factorization.
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[in] R
        ///     Represents the upper-triangular R factor of QR factorization.
        ///     On entry, is empty and may not have any space allocated for it.
        ///
        /// @param[out] A
        ///     Overwritten by garbage.
        ///
        /// @param[out] Q
        ///     Contains an m-by-k orthogonal Q factor.
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
            int64_t d_factor,
            std::vector<T> &Q,
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

        // Work buffer 1, of size m by n. Used for (in order):
        // Buffer array for sketching. of size d by cols. 
        // Copy of A_piv of size rows by cols.
        std::vector<T> Work1;
        // Work buffer 2, of size b_sz * max((n - b_sz), b_sz). Used for (in order):
        // Buffer for R_sk of size b_sz * b_sz.
        // Buffer to store R12 of size b_sz * (cols - b_sz).
        std::vector<T> Work2;
        // Work buffer 3, of size b_sz * b_sz. Used for (in order):
        // Buffer for R in Cholesky QR, of size b_sz * b_sz.
        // Buffer required by Householder reconstruction routine, name "T" in orhr_col. Max size b_sz by b_sz, can have any number of rows < b_sz. 
        // Buffer for R11, of size b_sz * b_sz.
        std::vector<T> Work3;
        // Work buffer 4, of size n. Used for (in order):
        // Vector tau in qr factorization, of size col.
        // Vector D in Householder reconstruction, of size col.
        std::vector<T> Work4;
};

// -----------------------------------------------------------------------------
template <typename T, typename RNG>
int CQRRP_blocked<T, RNG>::call(
    int64_t m,
    int64_t n,
    std::vector<T> &A,
    int64_t d_factor,
    std::vector<T> &Q,
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

    high_resolution_clock::time_point cholqr_t_start;
    high_resolution_clock::time_point cholqr_t_stop;
    long cholqr_t_dur = 0;

    high_resolution_clock::time_point reconstruction_t_start;
    high_resolution_clock::time_point reconstruction_t_stop;
    long reconstruction_t_dur = 0;

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

    int64_t rows = m;
    int64_t cols = n;
    int64_t curr_sz = 0;
    int64_t b_sz = this->block_size;
    int64_t maxiter = (int64_t) std::ceil(n / (T) b_sz);
    int64_t d = 0;
    
    // BELOW ARE MATRICES THAT WE CANNOT PUT INTO COMMON BUFFERS
    // We will need a copy of A_pre to use in gemms, is of size col by b_sz.
    std::vector<T> A_pre(m * b_sz, 0.0);
    // We will be operating on this at every iteration, is of size cols.
    std::vector<int64_t> J_buffer (n, 0);

    T* A_dat     = A.data();
    T* Q_dat     = util::upsize(m * m, Q);
    T* R_dat     = util::upsize(m * n, R); 
    T* Work1_dat = util::upsize(m * n, this->Work1);
    T* Work2_dat = util::upsize(b_sz * std::max((n - b_sz), b_sz), this->Work2);
    T* Work3_dat = util::upsize(b_sz * b_sz, this->Work3);
    T* Work4_dat = util::upsize(n, this->Work4);
    T* A_pre_dat = A_pre.data();
    J.resize(n);
    int64_t* J_dat        = J.data();
    int64_t* J_buffer_dat = J_buffer.data();

    T norm_A     = lapack::lange(Norm::Fro, m, n, A_dat, m);
    T norm_R     = 0.0;
    T norm_R11   = 0.0;
    T norm_R12   = 0.0;
    T norm_R_i   = 0.0;
    T approx_err = 0.0;

    int i, j, k, iter = 0;

    for(iter = 0; iter < maxiter; ++iter) {
        // Make sure we fit into the available space
        b_sz = std::min(this->block_size, n - curr_sz);
        // Update the row dimension of a sketch
        d = std::min(rows, d_factor * cols);

        // Zero-out data
        std::fill(J_buffer.begin(), J_buffer.end(), 0);
        std::fill(Work4.begin(), Work4.end(), 0.0);
        std::fill(Work2.begin(), Work2.end(), 0.0);

        if(this -> timing) {
            saso_t_start = high_resolution_clock::now();
        }

        // Skethcing in an embedding regime
        RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = rows, .vec_nnz = this->nnz};
        RandBLAS::SparseSkOp<T, RNG> S(DS, state);
        state = RandBLAS::fill_sparse(S);

        RandBLAS::sketch_general(
            Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, cols, rows, 1.0, S, 0, 0, A.data(), rows, 0.0, Work1_dat, d
        );

        if(this -> timing) {
            saso_t_stop  = high_resolution_clock::now();
            saso_t_dur  += duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
            qrcp_t_start = high_resolution_clock::now();
        }

        // QRCP
        lapack::geqp3(d, cols, Work1_dat, d, J_buffer_dat, Work4_dat);

        if(this -> timing) {
            qrcp_t_stop = high_resolution_clock::now();
            qrcp_t_dur += duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
        }

        // Need to premute trailing columns of the full R-factor
        if(iter != 0)
            util::col_swap(m, cols, cols, &R_dat[m * curr_sz], J_buffer);

        // extract b_sz by b_sz R_sk (Work2)
        for(i = 0; i < b_sz; ++i)
            blas::copy(i + 1, &Work1_dat[i * d], 1, &Work2_dat[i * b_sz], 1);

        // A_piv (Work1) = Need to pivot full matrix A
        // This is a wors by cols permuted version of the current A
        util::col_swap(rows, cols, cols, A_dat, J_buffer);

        // Below copy is required to preserve the true state of a pivoted A.
        // The actual space of A will be used to store intermediate representation of the current iteration's Q.
        lapack::lacpy(MatrixType::General, rows, cols, A_dat, rows, Work1_dat, rows);

        // A_pre = AJ(:, 1:b_sz) * R_sk (Work2)
        // This is a preconditioned rows by b_sz version of the current A, written into Q_full because of the way trsm works
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, Work2_dat, b_sz, A_dat, rows);

        // We need to save A_pre, is required for computations below 
        lapack::lacpy(MatrixType::General, rows, b_sz, A_dat, rows, A_pre_dat, rows);

        // Performing Cholesky QR
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, b_sz, rows, 1.0, A_dat, rows, 0.0, Work3_dat, b_sz);
        lapack::potrf(Uplo::Upper, b_sz, Work3_dat, b_sz);

        // Compute Q_econ
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, Work3_dat, b_sz, A_dat, rows);

        // Find Q (stored in A) using Householder reconstruction. 
        // Remember that Q (stored in A) has b_sz orthonormal columns
        lapack::orhr_col(rows, b_sz, b_sz, A_dat, rows, Work3_dat, b_sz, Work4_dat);

        // Remember that at the moment even though Q (stored in A) is represented by "b_sz" columns, it is actually of size "rows by rows."
        // Compute R11_full = Q' * A_pre - gets written into A_pre space
        lapack::gemqrt(Side::Left, Op::Trans, rows, b_sz, b_sz, b_sz, A_dat, rows, Work3_dat, b_sz, A_pre_dat, rows);

        // Looks like we can substitute the two multiplications with a single one:
        // A_piv (Work1) is a rows by cols matrix, its last "cols-b_sz" columns and "rows" rows will be updated.
        // The first b_sz rows will represent R12 (Stored in Work2)
        // The last rows-b_sz rows will represent the new A
        lapack::gemqrt(Side::Left, Op::Trans, rows, cols - b_sz, b_sz, b_sz,  A_dat, rows, Work3_dat, b_sz, &Work1_dat[rows * b_sz], rows);

        // Updating Q, Pivots
        if(iter == 0) {
            blas::copy(cols, J_buffer_dat, 1, J_dat, 1);
            RandLAPACK::util::eye(rows, rows, Q);
            lapack::gemqrt(Side::Right, Op::NoTrans, rows, rows, b_sz, b_sz, A_dat, rows, Work3_dat, b_sz, Q_dat, rows);
        } else {
            lapack::gemqrt(Side::Right, Op::NoTrans, m, rows, b_sz, b_sz, A_dat, rows, Work3_dat, b_sz, &Q_dat[m * curr_sz], m);
            RandLAPACK::util::col_swap<T>(cols, cols, &J_dat[curr_sz], J_buffer);
        }

        // Compute R11 (stored in Work3) = R11_full(1:b_sz, :) * R_sk (Work2)
        // We will have to resize A_pre_dat down to b_sz by b_sz
        RandLAPACK::util::row_resize(rows, b_sz, A_pre, b_sz);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, b_sz, b_sz, 1.0, A_pre_dat, b_sz, Work2_dat, b_sz, 0.0, Work3_dat, b_sz);

        // Filling R12 and updating A
        for(i = 0; i < (cols - b_sz); ++i) {
            blas::copy(b_sz, &Work1_dat[rows * (b_sz + i)], 1, &Work2_dat[i * b_sz], 1);
            blas::copy(rows - b_sz, &Work1_dat[rows * (b_sz + i) + b_sz], 1, &A_dat[i * (rows - b_sz)], 1);
        }

        // Updating R-factor. The full R-factor is m by n
        for(i = curr_sz, j = -1, k = -1; i < n; ++i) {
            if (i < curr_sz + b_sz) {
                blas::copy(b_sz, &Work3_dat[b_sz * ++j], 1, &R_dat[(m * i) + curr_sz], 1);
            } else {
                blas::copy(b_sz, &Work2_dat[b_sz * ++k], 1, &R_dat[(m * i) + curr_sz], 1);
            }
        }

        // Estimate R norm, use Fro norm trick to compute the approximation error
        norm_R11 = lapack::lange(Norm::Fro, b_sz, b_sz, Work3_dat, b_sz);
        norm_R12 = lapack::lange(Norm::Fro, b_sz, n - curr_sz - b_sz, Work2_dat, b_sz);
        norm_R_i = std::hypot(norm_R11, norm_R12);
        norm_R = std::hypot(norm_R, norm_R_i);
        // Updating approximation error
        approx_err = std::sqrt((norm_A - norm_R) * (norm_A + norm_R)) / norm_A;
        
        // Size of the factors is updated;
        curr_sz += b_sz;

        if(approx_err < std::sqrt(this->eps))
        {
            // Termination criteria reached
            this -> rank = curr_sz;
            RandLAPACK::util::row_resize(m, n, R, curr_sz);
            return 0;
        }

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;
    }
    this -> rank = n;
    RandLAPACK::util::row_resize(m, n, R, n);

    // may not be a proper place for this function
    if(this -> timing) {
        total_t_stop = high_resolution_clock::now();
        total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
        long t_rest  = total_t_dur - (saso_t_dur + qrcp_t_dur + reconstruction_t_dur + cholqr_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur + copy_t_dur + resize_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, reconstruction_t_dur, cholqr_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, copy_t_dur, resize_t_dur, t_rest, total_t_dur};
    }
    return 0;
}

} // end namespace RandLAPACK
#endif
