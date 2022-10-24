#include <blas.hh>
#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>
#include <math.h>

#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

#include <lapack/fortran.h>
#include <lapack/config.h>

using namespace std::chrono;

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;

#include <valgrind/callgrind.h>

/*
// For calling LAPACK directly (didnt work)
void _LAPACK_sorhr_col(
  int64_t m, int64_t n, int64_t nb, double *A, int64_t lda, double *T, int64_t ldt, double *D, int64_t info)
{
  lapack_int m_ = (lapack_int) m;
  lapack_int n_ = (lapack_int) n;
  lapack_int nb_ = (lapack_int) nb;
  lapack_int lda_ = (lapack_int) lda;
  lapack_int ldt_ = (lapack_int) ldt;  
  lapack_int info_ = (lapack_int) info;
  LAPACK_sorhr_col(&m_, &n_, &nb_, A, &lda_, T, &ldt_, D, info_);
  info = (int64_t) info_;
  return;
}
*/

// Different from RandLAPACK's HQRQ, as this one produces a full Q
template <typename T> 
void HQRQ_full(int64_t m, int64_t n, std::vector<T>& A, std::vector<T>& tau)
{
    // Done via regular LAPACK's QR
    // tau The vector tau of length min(m,n). The scalar factors of the elementary reflectors (see Further Details).
    // tau needs to be a vector of all 2's by default

    using namespace lapack;

    upsize<T>(n, tau);

    T* A_dat = A.data();
    T* tau_dat = tau.data();
    geqrf(m, n, A_dat, m, tau_dat);

    // FOR EXPLICIT STORAGE
    //ungqr(m, m, n, A_dat, m, tau_dat);
}

template <typename T> 
int GEQR(
        int64_t m,
        int64_t n,
        std::vector<T>& A,
        std::vector<T>& tvec
){
        using namespace lapack;

        tvec.resize(5);

        T* A_dat = A.data();
        
        geqr(m, n, A_dat, m, tvec.data(), -1);
        int64_t tsize = (int64_t) tvec[0]; 
        tvec.resize(tsize);
        if(geqr(m, n, A_dat, m, tvec.data(), tsize))
                return 1;

        return 0;
}

// Grabs the main diagonal of a matrix and stores it in a vector
template <typename T> 
void undiag(
    int64_t m,
    int64_t n,
    std::vector<T>& A,
    std::vector<T>& a
) {     

    int64_t k;
    if(m > n)
    {
        k = n;
    }
    else
    {
        k = m;
    }
    upsize<T>(k, a);

    T* A_dat = A.data();
    T* a_dat = a.data();

    for(int i = 0; i < k; ++i)
    {
        a_dat[i] = (A_dat[(i * m) + i]);
    }
}

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, int64_t n, uint32_t seed) {

    using namespace blas;
    using namespace lapack;

    int64_t size = m * n;
    std::vector<T> A(size, 0.0);
    // Allocate more space for A_cpy used in HQR
    std::vector<T> A_cpy(m * n, 0.0);
    std::vector<T> B(size, 0.0);
    std::vector<T> B_cpy(size, 0.0);

    T* A_dat = A.data();
    T* A_cpy_dat = A_cpy.data();
    T* B_dat = B.data();
    T* B_cpy_dat = B_cpy.data();

    // Needed for full QR through Chol
    std::vector<T> tau ;
    std::vector<T> D (n, 0.0);
    std::vector<T> T_mat (n * n, 0.0);
    std::vector<T> TVT (n * m, 0.0);
    std::vector<T> t;
    std::vector<T> Q_gram;

    // Random Gaussian test matrix
    RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_dat, seed);
    RandBLAS::dense_op::gen_rmat_norm<T>(m, n, B_dat, seed);
    // Make a copy
    std::copy(A_dat, A_dat + size, A_cpy_dat);
    std::copy(B_dat, B_dat + size, B_cpy_dat);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Stabilization Constructor - not needed for HQRQ, as we're using a custom function
    Orth<T> Orth_CholQR(use_CholQRQ, false, false);
    //Orth<T> Orth_HQR(use_HQRQ, false, false);

    // CHOL QR
    // Orthonormalize A

    for(int i = 0; i < m * n; ++i)
    {
       A[i] = (float) A[i];
    }

    auto start_chol = high_resolution_clock::now();
    Orth_CholQR.call(m, n, A);
    // Restore Householder vectors
    //orhr_col(m, n, n, A.data(), m, T_mat.data(), n, D.data());

    auto stop_chol = high_resolution_clock::now();
    long dur_chol = duration_cast<microseconds>(stop_chol - start_chol).count();

    // HQR
    auto start_qr = high_resolution_clock::now();
    //HQRQ_full(m, n, A_cpy, tau);
    GEQR(m, n, A_cpy, tau);

    auto stop_qr = high_resolution_clock::now();
    long dur_qr = duration_cast<microseconds>(stop_qr - start_qr).count();

    std::vector<long> res{dur_qr, dur_chol}; 
  
    return res; 
}

template <typename T>
static std::vector<long> 
test_speed_orhr_col(int64_t m, int64_t n, uint32_t seed) {

    using namespace blas;
    using namespace lapack;

    int64_t size = m * n;
    std::vector<T> A(size, 0.0);

    T* A_dat = A.data();

    // Needed for full QR through Chol
    std::vector<T> tau ;
    std::vector<T> D (n, 0.0);
    std::vector<T> T_mat (n * n, 0.0);

    // Random Gaussian test matrix
    RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_dat, seed);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Stabilization Constructor - not needed for HQRQ, as we're using a custom function
    Orth<T> Orth_CholQR(use_CholQRQ, false, false);
    //Orth<T> Orth_HQR(use_HQRQ, false, false);

    // CHOL QR
    // Orthonormalize A
    auto start_qr = high_resolution_clock::now();
    Orth_CholQR.call(m, n, A);
    auto stop_qr = high_resolution_clock::now();
    long dur_qr = duration_cast<microseconds>(stop_qr - start_qr).count();

    // Restore Householder vectors
    auto start_rest = high_resolution_clock::now();

    // PERFORMANCE PROFILING
    // CALLGRIND_START_INSTRUMENTATION;
    // CALLGRIND_TOGGLE_COLLECT;

    orhr_col(m, n, n, A.data(), m, T_mat.data(), n, D.data());

    // CALLGRIND_TOGGLE_COLLECT;
    // CALLGRIND_STOP_INSTRUMENTATION;

    auto stop_rest = high_resolution_clock::now();
    long dur_rest = duration_cast<microseconds>(stop_rest - start_rest).count();

    std::vector<long> res{dur_qr, dur_rest}; 
  
    return res; 
}

template <typename T>
static void 
test_speed_max(int r_pow, int r_pow_max, int col, int col_max, int runs)
{

    // Clear all files
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    int64_t rows = 0;
    int64_t cols = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols += 64)
        {
            std::vector<long> res;
            long t_chol = 0;
            long t_qr   = 0;

            long curr_t_chol = 0;
            long curr_t_qr   = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, 0);
                //res = test_speed_orhr_col<T>(rows, cols, 0);
                curr_t_chol = res[1];
                curr_t_qr   = res[0];

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    if(t_chol > curr_t_chol || t_chol == 0)
                        t_chol = curr_t_chol;
                    if(t_qr > curr_t_qr || t_qr == 0)
                        t_qr  = curr_t_qr;
                }
            }

            // Save the output into .dat file
            std::fstream file("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::fstream::app);
            file << t_qr << "  " << t_chol << "\n";

            //printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            //printf("Best timing of Chol QR for %d runs: %ld μs.\n", runs - 1, t_chol);
            //printf("Best timing of Householder reflector restoration for %d runs: %ld μs.\n", runs - 1, t_qr);
            //printf("Result: rest is %f times faster than CholQR.\n\n", (float) t_qr / (float) t_chol);

            printf("Best timing of Chol QR + restoration for %d runs: %ld μs.\n", runs - 1, t_chol);
            printf("Best timing of Full Householder QR for %d runs: %ld μs.\n", runs - 1, t_qr);
            printf("Result: cholQR + rest is %f times faster then Full HQR.\n\n", (float) t_qr / (float) t_chol);
        }
    }
}

template <typename T>
static void 
test_speed_mean(int r_pow, int r_pow_max, int col, int col_max, int runs)
{

    // Clear all files
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    int64_t rows = 0;
    int64_t cols = 0;

    T chol_avg = 0;
    T qr_avg = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols += 64)
        {
            std::vector<long> res;
            long t_chol = 0;
            long t_qr   = 0;

            long curr_t_chol = 0;
            long curr_t_qr   = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, i);
                curr_t_chol = res[1];
                curr_t_qr   = res[0];

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    t_chol += curr_t_chol;
                    t_qr   += curr_t_qr;
                }
            }

            chol_avg = (T)t_chol / (T)(runs - 1);
            qr_avg   = (T)t_qr   / (T)(runs - 1);

            // Save the output into .dat file
            std::fstream file("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::fstream::app);
            file << qr_avg << "  " << chol_avg << "\n";

            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Average timing of Chol QR + restoration for %d runs: %f μs.\n", runs - 1, chol_avg);
            printf("Average timing of Full Householder QR for %d runs: %f μs.\n", runs - 1, qr_avg);
            printf("Result: cholQR + rest is %f times faster than Full HQR.\n\n", qr_avg / chol_avg);
        }
    }
}


int main(int argc, char **argv){
    test_speed_max<double>(17, 17, 64, 64, 5);
    //test_speed_max<double>(18, 18, 64, 1024, 50);
    //test_speed_max<double>(12, 14, 32, 256, 50);
    //test_speed_helper<double>(5, 3, 0);
    return 0;
}