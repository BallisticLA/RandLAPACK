#include <cstdint>
#include <vector>

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
    sas.vec_nnz = this->nnz;
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

    //-------TIMING--------/
    if(this -> timing)
    {
        cholqrcp_t_stop = high_resolution_clock::now();
        cholqrcp_t_dur = duration_cast<microseconds>(cholqrcp_t_stop - cholqrcp_t_start).count();
    }
    //-------TIMING--------/

    // Get R
    // trmm
    trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, k, n, 1.0, R_sp_dat, k, R_dat, k);	

    //-------TIMING--------/
    if(this -> timing)
    {
        total_t_stop = high_resolution_clock::now();
        total_t_dur = duration_cast<microseconds>(total_t_stop - total_t_start).count();

        long t_rest = total_t_dur - (saso_t_dur + qrcp_t_dur + rank_reveal_t_dur + cholqrcp_t_dur + a_mod_piv_t_dur + a_mod_trsm_t_dur + copy_t_dur + resize_t_dur);

        // Fill the data vector
        this -> times = {saso_t_dur, qrcp_t_dur, rank_reveal_t_dur, cholqrcp_t_dur, a_mod_piv_t_dur, a_mod_trsm_t_dur, copy_t_dur, resize_t_dur, t_rest, total_t_dur};
    }
    //-------TIMING--------/

    return 0;
}

template int CholQRCP<float>::CholQRCP1(int64_t m, int64_t n, std::vector<float>& A, int64_t d, std::vector<float>& R, std::vector<int64_t>& J);
template int CholQRCP<double>::CholQRCP1(int64_t m, int64_t n, std::vector<double>& A, int64_t d, std::vector<double>& R, std::vector<int64_t>& J);
}