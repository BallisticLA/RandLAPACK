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

        // 8 entries
        std::vector<long> times;

        // tuning SASOS
        int num_threads;
        int64_t nnz;

        // Work buffer 1, of size m by n. Used for (in order):
        // Buffer array for sketching. of size d by cols. - NOT ANYMORE
        // Copy of A_piv of size rows by cols.
        std::vector<T> Work1;
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

    high_resolution_clock::time_point preconditioning_t_start;
    high_resolution_clock::time_point preconditioning_t_stop;
    long preconditioning_t_dur = 0;

    high_resolution_clock::time_point updating_t_start;
    high_resolution_clock::time_point updating_t_stop;
    long updating_t_dur = 0;

    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;
    long total_t_dur = 0;

    if(this -> timing) {
        total_t_start = high_resolution_clock::now();
    }

    int64_t rows = m;
    int64_t cols = n;
    int64_t curr_sz = 0;
    int64_t b_sz = this->block_size;
    int64_t maxiter = (int64_t) std::ceil(n / (T) b_sz);
    // Update the row dimension of a sketch
    int64_t d =  d_factor * b_sz;
    
    // BELOW ARE MATRICES THAT WE CANNOT PUT INTO COMMON BUFFERS
    // We will need a copy of A_pre to use in gemms, is of size col by b_sz.
    std::vector<T> A_pre(m * b_sz, 0.0);
    // We will be operating on this at every iteration, is of size cols.
    std::vector<int64_t> J_buffer (n, 0);
    // EXPERIMENTAL ADDITION - buffer for sketching update
    std::vector<T> A_sk(d * n, 0.0);

    T* A_dat     = A.data();
    T* Q_dat     = util::upsize(m * m, Q);
    T* R_dat     = util::upsize(m * n, R); 
    T* Work1_dat = util::upsize(m * n, this->Work1);
    // Work buffer 2, stored in R. Used for (in order):
    // Buffer for R_sk of size b_sz * b_sz.
    // Buffer to store R12 of size b_sz * (cols - b_sz).
    T* Work2_dat = NULL;
    // Work buffer 3, stored in R. Used for (in order):
    // Buffer for R in Cholesky QR, of size b_sz * b_sz.
    // Buffer required by Householder reconstruction routine, name "T" in orhr_col. Max size b_sz by b_sz, can have any number of rows < b_sz. 
    // Buffer for R11, of size b_sz * b_sz.
    T* Work3_dat = NULL;
    T* Work4_dat = util::upsize(n, this->Work4);
    T* A_pre_dat = A_pre.data();
    T* A_sk_dat  = A_sk.data();
    J.resize(n);
    int64_t* J_dat        = J.data();
    int64_t* J_buffer_dat = J_buffer.data();

    T norm_A     = lapack::lange(Norm::Fro, m, n, A_dat, m);
    T norm_A_sq  = std::pow(norm_A, 2);
    T norm_R     = 0.0;
    T norm_R11   = 0.0;
    T norm_R12   = 0.0;
    T norm_R_i   = 0.0;
    T approx_err = 0.0;

    // TEMPORARY SPACE BEGIN
    std::vector<T> T_full(b_sz * n, 0.0);
    T* T_full_dat = T_full.data();
    T* T_dat = NULL;
    T* Q_implicit_dat = A.data();


    std::vector<T> R_sk(b_sz * b_sz, 0.0);
    T* R_sk_dat = R_sk.data();

    // TEMPORARY SPACE END

    // Skethcing in an embedding regime
    RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = rows, .vec_nnz = this->nnz};
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, cols, rows, 1.0, S, 0, 0, A.data(), rows, 0.0, A_sk_dat, d
    );

    for(int iter = 0; iter < maxiter; ++iter) {
        // Make sure we fit into the available space
        b_sz = std::min(this->block_size, n - curr_sz);

        // Zero-out data
        std::fill(J_buffer.begin(), J_buffer.end(), 0);
        std::fill(Work4.begin(), Work4.end(), 0.0);
        std::fill(R_sk.begin(), R_sk.end(), 0.0);

        if(this -> timing) {
            saso_t_start = high_resolution_clock::now();
        }

        if(this -> timing) {
            saso_t_stop  = high_resolution_clock::now();
            saso_t_dur  += duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
            qrcp_t_start = high_resolution_clock::now();
        }

        // QRCP
        lapack::geqp3(d, cols, A_sk_dat, d, J_buffer_dat, Work4_dat);

        if(this -> timing) {
            qrcp_t_stop = high_resolution_clock::now();
            qrcp_t_dur += duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
            updating_t_start = high_resolution_clock::now();
        }

        // Need to premute trailing columns of the full R-factor
        if(iter != 0)
            util::col_swap(m, cols, cols, &R_dat[m * curr_sz], m, J_buffer);

        if(this -> timing) {
            updating_t_stop  = high_resolution_clock::now();
            updating_t_dur  += duration_cast<microseconds>(updating_t_stop - updating_t_start).count();
        }

        // CONSIDER TAKING BELOW DIRECTLY FROM WORK1

        // Need to zero-out the lower-triangular portion in S11
        RandLAPACK::util::get_U(b_sz, b_sz, A_sk_dat, d);
        
        // extract b_sz by b_sz R_sk (Work2)

        lapack::lacpy(MatrixType::General, b_sz, b_sz, A_sk_dat, d, R_sk_dat, b_sz);

        if(this -> timing) {
            preconditioning_t_start = high_resolution_clock::now();
        }

        // A_piv (Work1) = Need to pivot full matrix A
        // This is a wors by cols permuted version of the current A
        util::col_swap(rows, cols, cols, A_dat, m, J_buffer);

        // Below copy is required to preserve the true state of a pivoted A.
        // The actual space of A will be used to store intermediate representation of the current iteration's Q.
        lapack::lacpy(MatrixType::General, rows, cols, A_dat, m, Work1_dat, rows);

        // A_pre = AJ(:, 1:b_sz) * R_sk (Work2)
        // This is a preconditioned rows by b_sz version of the current A, written into Q_full because of the way trsm works
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, R_sk_dat, b_sz, A_dat, m);

        if(this -> timing) {
            preconditioning_t_stop  = high_resolution_clock::now();
            preconditioning_t_dur  += duration_cast<microseconds>(preconditioning_t_stop - preconditioning_t_start).count();
        }

        // We need to save A_pre, is required for computations below 
        lapack::lacpy(MatrixType::General, rows, b_sz, A_dat, m, A_pre_dat, rows);

        if(this -> timing) {
            cholqr_t_start = high_resolution_clock::now();
        }

        // Performing Cholesky QR
        Work3_dat = &R_dat[(m + 1) * curr_sz];
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, b_sz, rows, 1.0, A_dat, m, 0.0, Work3_dat, m);
        lapack::potrf(Uplo::Upper, b_sz, Work3_dat, m);

        // Compute Q_econ
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, Work3_dat, m, A_dat, m);

        if(this -> timing) {
            cholqr_t_stop  = high_resolution_clock::now();
            cholqr_t_dur  += duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();
            reconstruction_t_start = high_resolution_clock::now();
        }




        // Find Q (stored in A) using Householder reconstruction. 
        // Remember that Q (stored in A) has b_sz orthonormal columns




        T_dat = &T_full_dat[b_sz * curr_sz]; 
        lapack::orhr_col(rows, b_sz, b_sz, A_dat, m, T_dat, b_sz, Work4_dat);



        if(this -> timing) {
            reconstruction_t_stop  = high_resolution_clock::now();
            reconstruction_t_dur  += duration_cast<microseconds>(reconstruction_t_stop - reconstruction_t_start).count();
            updating_t_start = high_resolution_clock::now();
        }

        // Remember that at the moment even though Q (stored in A) is represented by "b_sz" columns, it is actually of size "rows by rows."
        // Compute R11_full = Q' * A_pre - gets written into A_pre space
        lapack::gemqrt(Side::Left, Op::Trans, rows, b_sz, b_sz, b_sz, A_dat, m, T_dat, b_sz, A_pre_dat, rows);

        // Looks like we can substitute the two multiplications with a single one:
        // A_piv (Work1) is a rows by cols matrix, its last "cols-b_sz" columns and "rows" rows will be updated.
        // The first b_sz rows will represent R12 (Stored in Work2)
        // The last rows-b_sz rows will represent the new A
        lapack::gemqrt(Side::Left, Op::Trans, rows, cols - b_sz, b_sz, b_sz,  A_dat, m, T_dat, b_sz, &Work1_dat[rows * b_sz], rows);

        // Updating Q, Pivots
        if(iter == 0) {
            blas::copy(cols, J_buffer_dat, 1, J_dat, 1);
        } else {
            RandLAPACK::util::col_swap<T>(cols, cols, &J_dat[curr_sz], J_buffer);
        }

        // Compute R11 (stored in Work3) = R11_full(1:b_sz, :) * R_sk (Work2)
        // We will have to resize A_pre_dat down to b_sz by b_sz
        RandLAPACK::util::row_resize(rows, b_sz, A_pre, b_sz);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, b_sz, b_sz, 1.0, A_pre_dat, b_sz, R_sk_dat, b_sz, 0.0, Work3_dat, m);

        lapack::lacpy(MatrixType::Upper, b_sz, b_sz, Work3_dat, m, &Q_implicit_dat[(m + 1) * curr_sz], m);
        T* R11_dat = &Q_implicit_dat[(m + 1) * curr_sz];

        if(this -> timing) {
            updating_t_stop  = high_resolution_clock::now();
            updating_t_dur  += duration_cast<microseconds>(updating_t_stop - updating_t_start).count();
        }

        // Filling R12 and updating A
        T* R12_dat = &Q_implicit_dat[(m * (curr_sz + b_sz)) + curr_sz];
        lapack::lacpy(MatrixType::General, b_sz, cols - b_sz, &Work1_dat[rows * b_sz], rows, R12_dat, m);
        A_dat = &A_dat[(m + 1) * b_sz];

        lapack::lacpy(MatrixType::General, rows - b_sz, cols - b_sz, &Work1_dat[(rows + 1) * b_sz], rows, A_dat, m);

        // Updating the skethcing buffer
        // trsm (S11, R11) -> S11
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, b_sz, b_sz, 1.0, R11_dat, m, A_sk_dat, d);
        // CAREFUL WHEN d = b_sz
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, cols - b_sz, b_sz, -1.0, A_sk_dat, d, R12_dat, m, 1.0, &A_sk_dat[d * b_sz], d);

        // Changing the pointer to relevant data in A_sk - this is equaivalent to copying data over to the beginning of A_sk
        A_sk_dat = &A_sk_dat[d * b_sz];

        // Estimate R norm, use Fro norm trick to compute the approximation error
        norm_R11 = lapack::lange(Norm::Fro, b_sz, b_sz, R11_dat, m);
        norm_R12 = lapack::lange(Norm::Fro, b_sz, n - curr_sz - b_sz, R12_dat, m);
        norm_R_i = std::hypot(norm_R11, norm_R12);
        norm_R = std::hypot(norm_R, norm_R_i);
        // Updating approximation error
        approx_err = ((norm_A - norm_R) * (norm_A + norm_R)) / norm_A_sq;

        // Size of the factors is updated;
        curr_sz += b_sz;



        if((approx_err < this->eps) || (curr_sz >= n)) {
            // Termination criteria reached
            this -> rank = curr_sz;

            RandLAPACK::util::householder_unpacking(m, curr_sz, b_sz, Q_implicit_dat, Q_dat, T_full_dat);


            RandLAPACK::util::get_U(m, n, Q_implicit_dat, m);
            lapack::lacpy(MatrixType::Upper, m, n, Q_implicit_dat, m, R_dat, m);
            RandLAPACK::util::row_resize(m, n, R, curr_sz);

            if(this -> timing) {
                total_t_stop = high_resolution_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest  = total_t_dur - (saso_t_dur + qrcp_t_dur + reconstruction_t_dur + preconditioning_t_dur + updating_t_dur);
                this -> times.resize(8);
                this -> times = {saso_t_dur, qrcp_t_dur, preconditioning_t_dur, cholqr_t_dur, reconstruction_t_dur, updating_t_dur, t_rest, total_t_dur};

                printf("\n\n/------------CQRRP TIMING RESULTS BEGIN------------/\n");
                printf("SASO time: %34ld μs,\n",                           saso_t_dur);
                printf("QRCP time: %36ld μs,\n",                           qrcp_t_dur);
                printf("Preconditioning time: %23ld μs,\n",                preconditioning_t_dur);
                printf("CholQR time: %31ld μs,\n",                         cholqr_t_dur);
                printf("Householder vector restoration time: %7ld μs,\n",  reconstruction_t_dur);
                printf("Factors updating time: %23ld μs,\n",               updating_t_dur);
                printf("Other routines time: %24ld μs,\n",                 t_rest);
                printf("Total time: %35ld μs.\n",                          total_t_dur);

                printf("\nSASO generation and application takes %2.2f%% of runtime.\n", 100 * ((T) saso_t_dur            / (T) total_t_dur));
                printf("QRCP takes %32.2f%% of runtime.\n",                             100 * ((T) qrcp_t_dur            / (T) total_t_dur));
                printf("Preconditioning takes %20.2f%% of runtime.\n",                  100 * ((T) preconditioning_t_dur / (T) total_t_dur));
                printf("Cholqr takes %29.2f%% of runtime.\n",                           100 * ((T) cholqr_t_dur          / (T) total_t_dur));
                printf("Householder restoration takes %12.2f%% of runtime.\n",          100 * ((T) reconstruction_t_dur  / (T) total_t_dur));
                printf("Factors updating time takes %14.2f%% of runtime.\n",            100 * ((T) updating_t_dur        / (T) total_t_dur));
                printf("Everything else takes %20.2f%% of runtime.\n",                  100 * ((T) t_rest                / (T) total_t_dur));
                printf("/-------------CQRRP TIMING RESULTS END-------------/\n\n");
            }
            return 0;
        }

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;
    }
    return 0;
}

} // end namespace RandLAPACK
#endif