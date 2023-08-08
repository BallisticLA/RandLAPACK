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
    high_resolution_clock::time_point preallocation_t_stop;
    high_resolution_clock::time_point preallocation_t_start;
    long preallocation_t_dur = 0;

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
        preallocation_t_start = high_resolution_clock::now();
    }

    int64_t rows = m;
    int64_t cols = n;
    // Describes sizes of full Q and R factors at a given iteration.
    int64_t curr_sz = 0;
    int64_t b_sz = this->block_size;
    int64_t maxiter = (int64_t) std::ceil(n / (T) b_sz);
    // Update the row dimension of a sketch
    int64_t d =  d_factor * b_sz;
    
    // Make sure the pivots vector has enough space
    J.resize(n);
    // Setting up space for skethcing buffer
    std::vector<T> A_sk(d * n, 0.0);
    // Setting up space for a buffer with a copy of preconditioned A
    std::vector<T> A_pre(m * b_sz, 0.0);
    // Setting up space for a buffer for R-factor from Cholesky QR
    std::vector<T> R_cholqr (b_sz * b_sz, 0.0);
    // Setting up space for a buffer for pivot vector
    std::vector<int64_t> J_buffer (n, 0);
    // Setting up space for a buffer for tau in GEQP3 and D in orhr_col
    std::vector<T> Work4 (n, 0.0);

    //*********************************POINTERS TO A BEGIN*********************************
    // LDA for all of the below is m

    // Pointer to the beginning of the original space of A.
    // It will always point to the same memory location.
    T* A_dat      = A.data();
    // Pointer to the beginning of A's "work zone," 
    // will shift at every iteration of an algorithm by (m * b_sz) + b_sz.
    T* A_work_dat = A.data();
    // Workspace 1 pointer - will serve as a buffer for computing R12 and updated matrix A.
    // Points to a location, offset by m * b_sz from the current "A_work_dat."
    T* Work1_dat  = NULL;
    // Points to R11 factor, right above the compact Q, of size b_sz by b_sz.
    T* R11_dat    = NULL;
    // Points to R12 factor, to the right of R11 and above Work1 of size b_sz by n - cols.
    T* R12_dat    = NULL;
    //**********************************POINTERS TO A END**********************************

    //*********************************POINTERS TO PIVOTS BEGIN*********************************
    int64_t* J_dat = J.data();;
    //**********************************POINTERS TO PIVOTS END**********************************

    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE BEGIN*******************
    // BELOW ARE MATRICES THAT WE CANNOT PUT INTO COMMON BUFFERS
    // A copy of a preconditioned matrix A (of size m by b_sz), lda m.
    // Required in order to find the proper R11_full-factor after the 
    // full Q has been restored form economy Q (the has been found via Cholesky QR).
    // While the full A_pre is required for a computation, we would only
    // need the first b_sz rows of R11_full.
    // Eventually, the b_sz by b_sz space is use to store R11 (computed via trmm)
    // which is then copied into its appropriate space in the matrix A.
    T* A_pre_dat = A_pre.data();

    // J_buffer serves as a buffer for the pivots found at every iteration, of size n.
    // At every iteration, it would only hold "cols" entries.
    int64_t* J_buffer_dat = J_buffer.data();

    // A_sk serves as a skething matrix, of size d by n, lda d
    // Below algorithm does not perform repeated sampling, hence A_sk
    // is updated at the end of every iteration.
    // Should remain unchanged throughout the algorithm,
    // As the algorithm needs to have access to the upper-triangular factor R
    // (stored in this matrix after geqp3) at all times. 
    T* A_sk_dat = A_sk.data();
    // Pointer to the b_sz by b_sz upper-triangular facor R stored in A_sk after GEQP3.
    T* R_sk_dat = NULL;

    // Buffer for the R-factor in Cholesky QR, of size b_sz by b_sz, lda b_sz.
    T* R_cholqr_dat = R_cholqr.data();

    // Buffer for Tau in GEQP3 and D in orhr_col, of size n.
    T* Work4_dat = Work4.data();
    //*******************POINTERS TO DATA REQUIRING ADDITIONAL STORAGE END*******************

    // Below Q and R buffers are only required for current implementation.
    // In practice, we are hoping for the output of CQRRPT to be identical to that of GEQP3.
    T* Q_dat      = util::upsize(m * m, Q);
    T* R_dat      = util::upsize(m * n, R); 
    std::vector<T> T_full(b_sz * n, 0.0);
    // Pointer to the full matrix T from orhr_col.
    T* T_full_dat = T_full.data();
    // Pointer to matrix T from orhr_col at currect iteration.
    T* T_dat      = NULL;


    T norm_A     = lapack::lange(Norm::Fro, m, n, A_dat, m);
    T norm_A_sq  = std::pow(norm_A, 2);
    T norm_R     = 0.0;
    T norm_R11   = 0.0;
    T norm_R12   = 0.0;
    T norm_R_i   = 0.0;
    T approx_err = 0.0;

    if(this -> timing) {
        preallocation_t_stop  = high_resolution_clock::now();
        preallocation_t_dur   = duration_cast<microseconds>(preallocation_t_stop - preallocation_t_start).count();
        saso_t_start = high_resolution_clock::now();
    }

    // Skethcing in an embedding regime
    RandBLAS::SparseDist DS = {.n_rows = d, .n_cols = rows, .vec_nnz = this->nnz};
    RandBLAS::SparseSkOp<T, RNG> S(DS, state);
    state = RandBLAS::fill_sparse(S);

    RandBLAS::sketch_general(
        Layout::ColMajor, Op::NoTrans, Op::NoTrans,
        d, cols, rows, 1.0, S, 0, 0, A.data(), rows, 0.0, A_sk_dat, d
    );

    if(this -> timing) {
        saso_t_stop  = high_resolution_clock::now();
        saso_t_dur   = duration_cast<microseconds>(saso_t_stop - saso_t_start).count();
    }

    for(int iter = 0; iter < maxiter; ++iter) {
        // Make sure we fit into the available space
        b_sz = std::min(this->block_size, n - curr_sz);

        // Zero-out data - may not be necessary
        std::fill(J_buffer.begin(), J_buffer.end(), 0);
        std::fill(Work4.begin(), Work4.end(), 0.0);

        if(this -> timing)
            qrcp_t_start = high_resolution_clock::now();

        // Performing QR with column pivoting
        lapack::geqp3(d, cols, A_sk_dat, d, J_buffer_dat, Work4_dat);

        if(this -> timing) {
            qrcp_t_stop = high_resolution_clock::now();
            qrcp_t_dur += duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();
            updating_t_start = high_resolution_clock::now();
        }

        // Need to premute trailing columns of the full R-factor.
        // Remember that the R-factor is stored the upper-triangular portion of A.
        if(iter != 0)
            util::col_swap(curr_sz, cols, cols, &A_dat[m * curr_sz], m, J_buffer);

        if(this -> timing) {
            updating_t_stop  = high_resolution_clock::now();
            updating_t_dur  += duration_cast<microseconds>(updating_t_stop - updating_t_start).count();
            preconditioning_t_start = high_resolution_clock::now();
        }

        // Pivoting the current matrix A.
        util::col_swap(rows, cols, cols, A_work_dat, m, J_buffer);

        // Defining the new "working subportion" of matrix A.
        // In a global sense, below is identical to:
        // Work1_dat = &A_dat[(m * (iter + 1) * b_sz) + curr_sz];
        Work1_dat = &A_work_dat[m * b_sz];

        // Define the space representing R_sk (stored in A_sk)
        R_sk_dat = A_sk_dat;

        // A_pre = AJ(:, 1:b_sz) * R_sk
        // Performing preconditioning of the current matrix A.
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, R_sk_dat, d, A_work_dat, m);

        if(this -> timing) {
            preconditioning_t_stop  = high_resolution_clock::now();
            preconditioning_t_dur  += duration_cast<microseconds>(preconditioning_t_stop - preconditioning_t_start).count();
        }

        // A copy of A_pre is necessary to later obraing the full R-factor from Cholesky QR.
        lapack::lacpy(MatrixType::General, rows, b_sz, A_work_dat, m, A_pre_dat, rows);

        if(this -> timing) {
            cholqr_t_start = high_resolution_clock::now();
        }

        // Performing Cholesky QR
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, b_sz, rows, 1.0, A_work_dat, m, 0.0, R_cholqr_dat, b_sz);
        lapack::potrf(Uplo::Upper, b_sz, R_cholqr_dat, b_sz);

        // Compute Q_econ from Cholesky QR
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, rows, b_sz, 1.0, R_cholqr_dat, b_sz, A_work_dat, m);

        if(this -> timing) {
            cholqr_t_stop  = high_resolution_clock::now();
            cholqr_t_dur  += duration_cast<microseconds>(cholqr_t_stop - cholqr_t_start).count();
            reconstruction_t_start = high_resolution_clock::now();
        }

        // Define a pointer to matrix T from orhr_col at current iteration.
        T_dat = &T_full_dat[b_sz * curr_sz]; 
        // Find Q (stored in A) using Householder reconstruction. 
        // This will represent the full (rows by rows) Q factor form Cholesky QR
        lapack::orhr_col(rows, b_sz, b_sz, A_work_dat, m, T_dat, b_sz, Work4_dat);

        if(this -> timing) {
            reconstruction_t_stop  = high_resolution_clock::now();
            reconstruction_t_dur  += duration_cast<microseconds>(reconstruction_t_stop - reconstruction_t_start).count();
            updating_t_start = high_resolution_clock::now();
        }

        // Compute R11_full = Q' * A_pre - gets written into A_pre space.
        // This may be simplifiable, as we only need R11_full[1:b_sz, 1:b_sz].
        // How can we perform an equivalent of (Q[:, 1:b_sz])' * A_pre?
        lapack::gemqrt(Side::Left, Op::Trans, rows, b_sz, b_sz, b_sz, A_work_dat, m, T_dat, b_sz, A_pre_dat, rows);


        // Perform Q_full' * A_piv(:, b_sz:end) to find R12 and the new "current A."
        // A_piv (Work1) is a rows by cols - b_sz matrix, stored in space of the original A.
        // The first b_sz rows will represent R12.
        // The last rows-b_sz rows will represent the new A.
        // With that, everything is placed where it should be, no copies required.
        lapack::gemqrt(Side::Left, Op::Trans, rows, cols - b_sz, b_sz, b_sz, A_work_dat, m, T_dat, b_sz, Work1_dat, m);

        // Updating pivots
        if(iter == 0) {
            blas::copy(cols, J_buffer_dat, 1, J_dat, 1);
        } else {
            RandLAPACK::util::col_swap<T>(cols, cols, &J_dat[curr_sz], J_buffer);
        }

        // Compute R11 = R11_full(1:b_sz, :) * R_sk
        // R11_full is stored in A_pre_dat space, R_sk is stored in A_sk space.
        blas::trmm(Layout::ColMajor, Side::Right,  Uplo::Upper, Op::NoTrans, Diag::NonUnit, b_sz, b_sz, 1.0, R_sk_dat, d, A_pre_dat, rows);

        // Need to copy R11 over form A_pre_dat into the appropriate space in A.
        // In a global sense, this is identical to:
        // R11_dat =  &A_dat[(m + 1) * curr_sz];
        R11_dat = A_work_dat;
        lapack::lacpy(MatrixType::Upper, b_sz, b_sz, A_pre_dat, rows, A_work_dat, m);

        // Updating the pointer to R12
        // In a global sense, this is identical to:
        // R12_dat =  &A_dat[(m * (curr_sz + b_sz)) + curr_sz];
        R12_dat = &R11_dat[m * b_sz];

        if(this -> timing) {
            updating_t_stop  = high_resolution_clock::now();
            updating_t_dur  += duration_cast<microseconds>(updating_t_stop - updating_t_start).count();
        }

        // Estimate R norm, use Fro norm trick to compute the approximation error
        // Keep in mind that R11 is Upper triangular and R12 is rectangular.
        norm_R11 = lapack::lantr(Norm::Fro, Uplo::Upper, Diag::NonUnit, b_sz, b_sz, R11_dat, m);
        norm_R12 = lapack::lange(Norm::Fro, b_sz, n - curr_sz - b_sz, R12_dat, m);
        norm_R_i = std::hypot(norm_R11, norm_R12);
        norm_R   = std::hypot(norm_R, norm_R_i);
        // Updating approximation error
        approx_err = ((norm_A - norm_R) * (norm_A + norm_R)) / norm_A_sq;

        // Size of the factors is updated;
        curr_sz += b_sz;

        if((approx_err < this->eps) || (curr_sz >= n)) {
            // Termination criteria reached
            this -> rank = curr_sz;

            if(this -> timing) {
                total_t_stop = high_resolution_clock::now();
                total_t_dur  = duration_cast<microseconds>(total_t_stop - total_t_start).count();
                long t_rest  = total_t_dur - (preallocation_t_dur + saso_t_dur + qrcp_t_dur + reconstruction_t_dur + preconditioning_t_dur + updating_t_dur);
                this -> times.resize(8);
                this -> times = {saso_t_dur, preallocation_t_dur, qrcp_t_dur, preconditioning_t_dur, cholqr_t_dur, reconstruction_t_dur, updating_t_dur, t_rest, total_t_dur};

                printf("\n\n/------------CQRRP TIMING RESULTS BEGIN------------/\n");
                printf("Preallocation time: %25ld μs,\n",                  preallocation_t_dur);
                printf("SASO time: %34ld μs,\n",                           saso_t_dur);
                printf("QRCP time: %36ld μs,\n",                           qrcp_t_dur);
                printf("Preconditioning time: %23ld μs,\n",                preconditioning_t_dur);
                printf("CholQR time: %31ld μs,\n",                         cholqr_t_dur);
                printf("Householder vector restoration time: %7ld μs,\n",  reconstruction_t_dur);
                printf("Factors updating time: %23ld μs,\n",               updating_t_dur);
                printf("Other routines time: %24ld μs,\n",                 t_rest);
                printf("Total time: %35ld μs.\n",                          total_t_dur);

                printf("\nPreallocation takes %22.2f%% of runtime.\n",                  100 * ((T) preallocation_t_dur   / (T) total_t_dur));
                printf("SASO generation and application takes %2.2f%% of runtime.\n",   100 * ((T) saso_t_dur            / (T) total_t_dur));
                printf("QRCP takes %32.2f%% of runtime.\n",                             100 * ((T) qrcp_t_dur            / (T) total_t_dur));
                printf("Preconditioning takes %20.2f%% of runtime.\n",                  100 * ((T) preconditioning_t_dur / (T) total_t_dur));
                printf("Cholqr takes %29.2f%% of runtime.\n",                           100 * ((T) cholqr_t_dur          / (T) total_t_dur));
                printf("Householder restoration takes %12.2f%% of runtime.\n",          100 * ((T) reconstruction_t_dur  / (T) total_t_dur));
                printf("Factors updating time takes %14.2f%% of runtime.\n",            100 * ((T) updating_t_dur        / (T) total_t_dur));
                printf("Everything else takes %20.2f%% of runtime.\n",                  100 * ((T) t_rest                / (T) total_t_dur));
                printf("/-------------CQRRP TIMING RESULTS END-------------/\n\n");
            }
            // WE ARE HOPING THAT BELOW WORK WILL BE UNNCECESSARY
            // WARNING: UNPACKING DOES NOT WHORK IF BLOCK SIZE HAS BEEN CHANGED!!!!!!!!!!!!!!!!!!
            RandLAPACK::util::householder_unpacking(m, curr_sz, b_sz, A_dat, Q_dat, T_full_dat);
            RandLAPACK::util::get_U(m, n, A_dat, m);
            lapack::lacpy(MatrixType::Upper, m, n, A_dat, m, R_dat, m);
            RandLAPACK::util::row_resize(m, n, R, curr_sz);

            return 0;
        }

        if(this -> timing)
            updating_t_start = high_resolution_clock::now();

        // Updating the pointer to "Current A."
        // In a global sense, below is identical to:
        // Work1_dat = &A_dat[(m * (iter + 1) * b_sz) + curr_sz + b_sz];
        // Also, Below is identical to:
        // A_work_dat = &A_work_dat[(m + 1) * b_sz];
        A_work_dat = &Work1_dat[b_sz];

        // Updating the skethcing buffer
        // trsm (R_sk_dat, R11) -> R_sk_dat
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, b_sz, b_sz, 1.0, R11_dat, m, R_sk_dat, d);
        // R_sk_12 - R_sk_11 * inv(R_11) * R_12
        // Side note: might need to be careful when d = b_sz.
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_sz, cols - b_sz, b_sz, -1.0, R_sk_dat, d, R12_dat, m, 1.0, &R_sk_dat[d * b_sz], d);

        // Changing the pointer to relevant data in A_sk - this is equaivalent to copying data over to the beginning of A_sk
        A_sk_dat = &A_sk_dat[d * b_sz];

        if(this -> timing) {
            updating_t_stop  = high_resolution_clock::now();
            updating_t_dur  += duration_cast<microseconds>(updating_t_stop - updating_t_start).count();
        }

        // Data size decreases by block_size per iteration.
        rows -= b_sz;
        cols -= b_sz;
    }
    return 0;
}

} // end namespace RandLAPACK
#endif