#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>

using namespace RandLAPACK::comps::util;
using namespace blas;
using namespace lapack;

namespace RandLAPACK::drivers::cholqrcp {
/*
template <typename T>
int CholQRCP<T>::CholQRCP1(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t& d,
    std::vector<T>& Q,
    std::vector<T>& R,
    std::vector<int64_t>& J
){
    // What to do about these?

    T* A_cpy_dat   = upsize(m * n, this->A_cpy);
    T* A_dat       = A.data();
    T* A_hat_dat   = upsize(d * n, this->A_hat);
    T* tau_dat     = upsize(n, this->tau);
    J.resize(n);
    int64_t* J_dat = J.data();

    copy<T, T>(m * n, A_dat, 1, A_cpy_dat, 1);
    
    // Generate a random matrix
    T* Omega_dat = upsize(d * m, this->Omega);
    RandBLAS::dense_op::gen_rmat_norm<T>(d, m, Omega_dat, this->seed);
    
    gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, Omega_dat, d, A_cpy_dat, m, 0.0, A_hat_dat, d);

    // QRCP - add failure condition
    geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);
    
    // Find rank
    int64_t k = n;
    int i;
    for(i = 0; i < n; ++i)
    {
        if(std::abs(A_hat_dat[i * d + i]) < this->eps)
        {
            k = i;
            this->rank = i;
            break;
        }
    }
    
    T* R_sp_dat  = upsize(k * k, this->R_sp);
    T* Q_dat     = upsize(m * k, Q);
    T* R_dat     = upsize(k * n, R);
    T* R_buf_dat = upsize(k * n, this -> R_buf);
    T* A_buf_dat = upsize(k * k, this->A_buf);

    // extract k by k R
    // Copy data over to R_sp_dat col by col
    for(i = 0; i < k; ++i)
    {
        copy<T, T>(i + 1, &A_hat_dat[i * d], 1, &R_sp_dat[i * k], 1);
        copy<T, T>(i + 1, &A_hat_dat[i * d], 1, &R_buf_dat[i * k], 1);
    }
    for(i = k; i < n; ++i)
    {
        copy<T, T>(k, &A_hat_dat[i * d], 1, &R_buf_dat[i * k], 1);
    }
    
    // Swap k columns of A with pivots from J
    col_swap(m, n, k, A_cpy, J);
    
    // IS THIS FASTER THAN DOING TRSM?

    // Get an inverse of R_sp
    trtri(Uplo::Upper, Diag::NonUnit, k, R_sp_dat, k);

    // Do AJ_k * R_sp^(-1)
    // switch this for trmm
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k, k, 1.0, A_cpy_dat, m, R_sp_dat, k, 0.0, Q_dat, m);
    
    // Do Cholesky QR
    syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, A_buf_dat, k);
    potrf(Uplo::Upper, k, A_buf_dat, k);
    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, A_buf_dat, k, Q_dat, m);

    // Get R
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, k, 1.0, A_buf_dat, k, R_buf_dat, k, 0.0, R_dat, k);
    
    return 0;
}
*/

// Same as above, but with TRSM
template <typename T>
int CholQRCP<T>::CholQRCP1(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t d,
    std::vector<T>& Q,
    std::vector<T>& R,
    std::vector<int64_t>& J
){
    T* A_dat       = A.data();
    T* Q_dat       = upsize(m * n, Q);
    T* A_hat_dat   = upsize(d * n, this->A_hat);
    T* tau_dat     = upsize(n, this->tau);
    J.resize(n);
    int64_t* J_dat = J.data();

    copy<T, T>(m * n, A_dat, 1, Q_dat, 1);
    
    // Need a wrapper for SJLT
    struct RandBLAS::sjlts::SJLT sjl;
    sjl.ori = RandBLAS::sjlts::ColumnWise;
    sjl.n_rows = d; // > n
    sjl.n_cols = m;
    sjl.vec_nnz = 8; // Arbitrary constant, Riley likes 8s
    sjl.rows = new uint64_t[sjl.vec_nnz * m];
    sjl.cols = new uint64_t[sjl.vec_nnz * m];
    sjl.vals = new double[sjl.vec_nnz * m];
    RandBLAS::sjlts::fill_colwise(sjl, this->seed, 0);

    RandBLAS::sjlts::sketch_csccol(sjl, m, n, (double*) Q_dat, (double*) A_hat_dat);

    // QRCP - add failure condition
    geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);
    
    // Find rank
    int64_t k = n;
    int i;
    for(i = 0; i < n; ++i)
    {
        if(std::abs(A_hat_dat[i * d + i]) < this->eps)
        {
            k = i;
            this->rank = i;
            break;
        }
    }
    
    T* R_sp_dat  = upsize(k * k, this->R_sp);
    T* R_dat     = upsize(k * n, R);
    T* R_buf_dat = upsize(k * n, this -> R_buf);

    // extract k by k R
    // Copy data over to R_sp_dat col by col
    for(i = 0; i < k; ++i)
    {
        copy<T, T>(i + 1, &A_hat_dat[i * d], 1, &R_sp_dat[i * k], 1);
        copy<T, T>(i + 1, &A_hat_dat[i * d], 1, &R_buf_dat[i * k], 1);
    }
    for(i = k; i < n; ++i)
    {
        copy<T, T>(k, &A_hat_dat[i * d], 1, &R_buf_dat[i * k], 1);
    }
    
    // Swap k columns of A with pivots from J
    col_swap(m, n, k, Q, J);
    
    // A_sp_pre * R_sp = AP
    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, Q_dat, m);
    
    // Do Cholesky QR
    syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, k, m, 1.0, Q_dat, m, 0.0, R_sp_dat, k);
    potrf(Uplo::Upper, k, R_sp_dat, k);
    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, k, 1.0, R_sp_dat, k, Q_dat, m);

    // Get R
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, n, k, 1.0, R_sp_dat, k, R_buf_dat, k, 0.0, R_dat, k);
    
    return 0;
}

template <typename T>
int CholQRCP<T>::CholQRCP2(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    int64_t d,
    int64_t b_sz,
    std::vector<T>& Q,
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

    for (int i = 0; i < n / b_sz; ++i)
    {
        // View to the row dimension of A_cpy, changes with every iteration
        int_64_t sz = m;
        
        // Need a wrapper for SJLT
        struct RandBLAS::sjlts::SJLT sjl;
        sjl.ori = RandBLAS::sjlts::ColumnWise;
        sjl.n_rows = d;
        sjl.n_cols = sz;
        sjl.vec_nnz = 8; // Arbitrary constant, Riley likes 8s
        sjl.rows = new uint64_t[sjl.vec_nnz * sz];
        sjl.cols = new uint64_t[sjl.vec_nnz * sz];
        sjl.vals = new double[sjl.vec_nnz * sz];
        RandBLAS::sjlts::fill_colwise(sjl, this->seed, 0);

        // A_hat = S * A
        RandBLAS::sjlts::sketch_csccol(sjl, sz, n, (double*) Q_dat, (double*) A_hat_dat);

        geqp3(d, n, A_hat_dat, d, J_dat, tau_dat);

        // Need to do something about the size of R
        col_swap(m, n, k, R, J);
        // A = AJ[:, b_sz]
        col_swap(sz, n, n, Q, J); // need a copy of this
        // Figure out what to do about the isze of this
        copy<T, T>(sz * n, Q_dat, 1, Q_cpy_dat, 1);

        // [Q_, R_11_full] = qr(AJ[:, b_sz])
        geqrf(sz, b_sz, Q_cpy_dat, sz, this->tau_dat);

        //R12 = Q_(:, 1:b_sz)' * A * J(:, b_sz + 1:end);


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

template int CholQRCP<float>::CholQRCP1(int64_t m, int64_t n, std::vector<float>& A, int64_t d, std::vector<float>& Q, std::vector<float>& R, std::vector<int64_t>& J);
template int CholQRCP<double>::CholQRCP1(int64_t m, int64_t n, std::vector<double>& A, int64_t d, std::vector<double>& Q, std::vector<double>& R, std::vector<int64_t>& J);
}