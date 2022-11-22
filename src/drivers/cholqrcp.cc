#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>
#include <chrono>

using namespace RandLAPACK::comps::util;
using namespace blas;
using namespace lapack;

using namespace std::chrono;

namespace RandLAPACK::drivers::cholqrcp {

// This vesrion of the code overwrites matrix A with Q
// Note that Q is now useless here
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
    long saso_t_dur;

    high_resolution_clock::time_point qrcp_t_start;
    high_resolution_clock::time_point qrcp_t_stop;
    long qrcp_t_dur;
    
    high_resolution_clock::time_point rank_reveal_t_start;
    high_resolution_clock::time_point rank_reveal_t_stop;
    long rank_reveal_t_dur;

    high_resolution_clock::time_point cholqrcp_t_start;
    high_resolution_clock::time_point cholqrcp_t_stop;
    long cholqrcp_t_dur;

    high_resolution_clock::time_point a_mod_piv_t_start;
    high_resolution_clock::time_point a_mod_piv_t_stop;
    long a_mod_piv_t_dur;

    high_resolution_clock::time_point a_mod_trsm_t_start;
    high_resolution_clock::time_point a_mod_trsm_t_stop;
    long a_mod_trsm_t_dur;

    high_resolution_clock::time_point copy_t_start;
    high_resolution_clock::time_point copy_t_stop;
    long copy_t_dur;

    high_resolution_clock::time_point resize_t_start;
    high_resolution_clock::time_point resize_t_stop;
    long resize_t_dur;

    high_resolution_clock::time_point total_t_start;
    high_resolution_clock::time_point total_t_stop;
    long total_t_dur;
    //-------TIMING VARS--------/

    //-------TIMING--------/
    if(this -> timing)
    {
        total_t_start = high_resolution_clock::now();
    
        resize_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/

    T* A_dat       = A.data();
    T* A_hat_dat   = upsize(d * n, this->A_hat);
    T* tau_dat     = upsize(n, this->tau);
    J.resize(n);
    int64_t* J_dat = J.data();

    //-------TIMING--------/
    if(this -> timing)
    {
        resize_t_stop = high_resolution_clock::now();
        resize_t_dur = duration_cast<microseconds>(resize_t_stop - resize_t_start).count();
    }
    
    //-------TIMING--------/
    if(this -> timing)
    {
        saso_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/

    struct RandBLAS::sasos::SASO sas;
    sas.n_rows = d; // > n
    sas.n_cols = m;
    sas.vec_nnz = this->nnz; // Arbitrary constant, Riley likes 8
    sas.rows = new int64_t[sas.vec_nnz * m];
    sas.cols = new int64_t[sas.vec_nnz * m];
    sas.vals = new double[sas.vec_nnz * m];
    RandBLAS::sasos::fill_colwise(sas, this->seed, 0);

    RandBLAS::sasos::sketch_csccol(sas, n, (double*) A_dat, (double*) A_hat_dat, this->num_threads);

    //-------TIMING--------/
    if(this -> timing)
    {
        saso_t_stop = high_resolution_clock::now();
        saso_t_dur = duration_cast<microseconds>(saso_t_stop - saso_t_start).count();

        qrcp_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/
    
    // QRCP - add failure condition
    geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);

    //-------TIMING--------/
    if(this -> timing)
    {
        qrcp_t_stop = high_resolution_clock::now();
        qrcp_t_dur = duration_cast<microseconds>(qrcp_t_stop - qrcp_t_start).count();

        rank_reveal_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/

    // Find rank
    int64_t k = n;
    this->rank = k;
    int i;
    for(i = 0; i < n; ++i)
    {
        if(std::abs(A_hat_dat[i * d + i]) < this->eps)
        {
            // "Rank" is k, but the index should be k - 1
            k = i;
            this->rank = k;
            break;
        }
    }

    //-------TIMING--------/
    if(this -> timing)
    {
        rank_reveal_t_stop = high_resolution_clock::now();
        rank_reveal_t_dur = duration_cast<microseconds>(rank_reveal_t_stop - rank_reveal_t_start).count();

        resize_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/
    
    T* R_sp_dat  = upsize(k * k, this->R_sp);
    T* R_dat     = upsize(k * n, R);
    T* R_buf_dat = upsize(k * n, this -> R_buf);

    //-------TIMING--------/
    if(this -> timing)
    {
        resize_t_stop = high_resolution_clock::now();
        resize_t_dur += duration_cast<microseconds>(resize_t_stop - resize_t_start).count();

        copy_t_start = high_resolution_clock::now();
    }

    // extract k by k R
    // Copy data over to R_sp_dat col by col
    for(i = 0; i < k; ++i)
    {
        copy<T, T>(i + 1, &A_hat_dat[i * d], 1, &R_sp_dat[i * k], 1);
        copy<T, T>(i + 1, &A_hat_dat[i * d], 1, &R_dat[i * k], 1);
    }
    for(i = k; i < n; ++i)
    {
        copy<T, T>(k, &A_hat_dat[i * d], 1, &R_dat[i * k], 1);
    }
    
    //-------TIMING--------/
    if(this -> timing)
    {
        copy_t_stop = high_resolution_clock::now();
        copy_t_dur = duration_cast<microseconds>(copy_t_stop - copy_t_start).count();

        a_mod_piv_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/

    // Swap k columns of A with pivots from J
    col_swap(m, n, k, A, J);

    //-------TIMING--------/
    if(this -> timing)
    {
        a_mod_piv_t_stop = high_resolution_clock::now();
        a_mod_piv_t_dur = duration_cast<microseconds>(a_mod_piv_t_stop - a_mod_piv_t_start).count();

        a_mod_trsm_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/

    // A_sp_pre * R_sp = AP
    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    //-------TIMING--------/
    if(this -> timing)
    {
        a_mod_trsm_t_stop = high_resolution_clock::now();
        a_mod_trsm_t_dur = duration_cast<microseconds>(a_mod_trsm_t_stop - a_mod_trsm_t_start).count();
    }
    //-------TIMING--------/

    //-------TIMING--------/
    if(this -> timing)
    {
        cholqrcp_t_start = high_resolution_clock::now();
    }
    //-------TIMING--------/

    // Do Cholesky QR
    syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, A_dat, m, 0.0, R_sp_dat, k);
    potrf(Uplo::Upper, k, R_sp_dat, k);

    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, A_dat, m);

    // Get R
    // trmm
    trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, k, n, 1.0, R_sp_dat, k, R_dat, k);	

    //-------TIMING--------/
    if(this -> timing)
    {
        cholqrcp_t_stop = high_resolution_clock::now();
        cholqrcp_t_dur = duration_cast<microseconds>(cholqrcp_t_stop - cholqrcp_t_start).count();

        total_t_stop = high_resolution_clock::now();
        total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

        printf("\n\n/------------CholQRCP1 TIMING RESULTS BEGIN------------/\n");

        long t_rest = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqrcp_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur + copy_t_dur + resize_t_dur);

        printf("SASO time: %ld μs,\n", saso_t_dur);
        printf("QRCP time: %ld μs,\n", qrcp_t_dur);
        printf("Rank revealing time: %ld μs,\n", rank_reveal_t_dur);
        printf("CholQRCP time: %ld μs,\n", cholqrcp_t_dur);
        printf("A modification pivoting time: %ld μs,\n", a_mod_piv_t_dur);
        printf("A modification TRSM time: %ld μs,\n", a_mod_trsm_t_dur);
        printf("Copying time: %ld μs,\n", copy_t_dur);
        printf("Resizing time: %ld μs,\n", resize_t_dur);
        printf("Other routines time: %ld μs,\n", t_rest);
        printf("Total time: %ld μs.\n", total_t_dur);

        printf("\nSASO generation and application takes %.1f%% of runtime.\n", 100 * ((double) saso_t_dur / (double) total_t_dur));
        printf("QRCP takes %.1f%% of runtime.\n", 100 * ((double) qrcp_t_dur / (double) total_t_dur));
        printf("Rank revealing takes %.1f%% of runtime.\n", 100 * ((double) rank_reveal_t_dur / (double) total_t_dur));
        printf("Modifying matrix (pivoting) A %.1f%% of runtime.\n", 100 * ((double) a_mod_piv_t_dur / (double) total_t_dur));
        printf("Modifying matrix (trsm) A %.1f%% of runtime.\n", 100 * ((double) a_mod_trsm_t_dur / (double) total_t_dur));
        printf("Cholqrcp takes %.1f%% of runtime.\n", 100 * ((double) cholqrcp_t_dur / (double) total_t_dur));
        printf("Copying takes %.1f%% of runtime.\n", 100 * ((double) copy_t_dur / (double) total_t_dur));
        printf("Resizing takes %.1f%% of runtime.\n", 100 * ((double) resize_t_dur / (double) total_t_dur));
        printf("Everything else takes %.1f%% of runtime.\n", 100 * ((double) t_rest / (double) total_t_dur));
        printf("/-------------CholQRCP1 TIMING RESULTS END-------------/\n\n");
    }
    //-------TIMING--------/

    return 0;
}

template <typename T>
int CholQRCP<T>::CholQRCP2(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t d,
    int64_t b_sz,
    std::vector<T>& R,
    std::vector<int64_t>& J
){
    /*
    T* Q_dat       = upsize(m * n, Q);
    T* A_dat       = A.data();
    T* A_hat_dat   = upsize(d * n, this->A_hat);
    T* tau_dat     = upsize(n, this->tau);
    J.resize(n);
    int64_t* J_dat = J.data();

    copy<T, T>(m * n, A_dat, 1, Q_dat, 1);


    // Preset the size of R
    T* R_data = upsize(n *n, R);

    for (int i = 0; i < n / b_sz; ++i)
    {
        // View to the row dimension of A_cpy, changes with every iteration
        int64_t sz1 = m;
        int64_t sz2 = n;
        
        // Need a wrapper for SASO
        struct RandBLAS::sasos::SASO saso;
        sjl.ori = RandBLAS::sasos::ColumnWise;
        sjl.n_rows = d;
        sjl.n_cols = sz1;
        sjl.vec_nnz = 8; // Arbitrary constant, Riley likes 8s
        sjl.rows = new uint64_t[sjl.vec_nnz * sz1];
        sjl.cols = new uint64_t[sjl.vec_nnz * sz1];
        sjl.vals = new double[sjl.vec_nnz * sz1];
        RandBLAS::sasos::fill_colwise(sjl, this->seed, 0);

        // A_hat = S * A
        RandBLAS::sasos::sketch_csccol(sjl, d, sz2, (double*) Q_dat, (double*) A_hat_dat);

        // At every iteration, size of J will be different
        geqp3(d, sz2, A_hat_dat, d, J_dat, tau_dat);

        if(i != 0)
        {
            // Swap size_j columns in R
            col_swap(n, n, sz2, R, J);
        }
        // A = AJ[:, b_sz]
        col_swap(sz1, sz2, sz2, Q, J); // need a copy of this

        // Figure out what to do about the isze of this
        copy<T, T>(sz1 * sz2, Q_dat, 1, Q_cpy_dat, 1);

        // [Q_, R_11_full] = qr(AJ[:, b_sz])
        geqrf(sz1, b_sz, Q_cpy_dat, sz, this->tau_dat);

        // need to grab the upper-triangular part and store into a buffer
        // R11 = R11_full(1:b_sz, :);
        // get_U(sz1, b_sz, R11)

        //ungqr(sz1, sz2, Q_cpy_dat, sz2, this->tau_dat);

        //R12 = Q_(:, 1:b_sz)' * A * J(:, b_sz + 1:end);
        // do a gemm on Q_cpy and offset Q, write into a buffer

        //A = Q_(:, b_sz + 1:end)' * A * J(:, b_sz + 1:end);
        

        //QI = eye(m, m);
        //PI = eye(n, n);

        //QI((j - 1) * b_sz + 1: end, (j - 1) * b_sz + 1 : end) = Q_;
        //PI((j - 1) * b_sz + 1: end, (j - 1) * b_sz + 1 : end) = J;

        //Q = Q * QI;
        //P = P * PI;
        
        //R1 = [R11 R12];

        //R((j - 1) * b_sz + 1: j * b_sz, (j - 1) * b_sz + 1: end) = R1;  
    }
    
   */
    return 0;
}

template int CholQRCP<float>::CholQRCP1(int64_t m, int64_t n, std::vector<float>& A, int64_t d, std::vector<float>& R, std::vector<int64_t>& J);
template int CholQRCP<double>::CholQRCP1(int64_t m, int64_t n, std::vector<double>& A, int64_t d, std::vector<double>& R, std::vector<int64_t>& J);
}