#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>

template <typename T>
static void
test_cond_helper_0(int64_t m, 
                  int64_t n, 
                  const std::tuple<int, T, bool>& mat_type, 
                  RandBLAS::base::RNGState<r123::Philox4x32> state) {

    std::vector<T> A(m * n, 0.0);
    std::vector<T> A_hat(m * n, 0.0);
    std::vector<T> I_ref(n * n, 0.0);
    std::vector<T> R_sp(n * n, 0.0);

    // Generate random matrix
    RandLAPACK::util::gen_mat_type(m, n, A, n, state, mat_type);
    RandLAPACK::util::eye(n, n, I_ref);

    std::copy(A.data(), A.data() + (m * n), A_hat.data());

    T* A_dat = A.data();
    T* A_hat_dat = A_hat.data();
    T* I_ref_dat = I_ref.data();
    T* R_sp_dat  = R_sp.data();

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A_dat, m, 0.0, R_sp_dat, n);
    lapack::potrf(Uplo::Upper, n, R_sp_dat, n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_sp_dat, n, A_dat, m);

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, A_dat, m, -1.0, I_ref_dat, n);
    T norm_Q = lapack::lange(lapack::Norm::Fro, n, n, I_ref_dat, n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_dat, m, R_sp_dat, n, -1.0, A_hat_dat, m);
    T norm_A = lapack::lange(Norm::Fro, m, n, A_hat_dat, m);

    printf("COND(A^{pre}): %21e\n", std::get<1>(mat_type));
    printf("FRO NORM OF Q' * Q - I: %e\n", norm_Q);
    printf("FRO NORM OF AP - QR: %15e\n\n", norm_A);
}

using namespace std::chrono;

template <typename T>
static int
test_cond_helper_1(int64_t m, 
                  int64_t n, 
                  int64_t true_k,
                  int64_t d,
                  int64_t nnz,
                  const std::tuple<int, T, bool>& mat_type, 
                  RandBLAS::base::RNGState<r123::Philox4x32> state,
                  int naive_rank_estimate,
                  int cond_check) {

    std::vector<T> A(m * n, 0.0);
    std::vector<T> A_1(m * n, 0.0);
    std::vector<T> A_2(m * n, 0.0);
    std::vector<T> R(n * n, 0.0);
    std::vector<int64_t> J;

    // Generate random matrix
    RandLAPACK::util::gen_mat_type(m, n, A, true_k, state, mat_type);

    /*    
    std::vector<T> A_pre_cpy;
    std::vector<T> s;
    RandLAPACK::util::cond_num_check(m, n, A, A_pre_cpy, s, false);

    auto start = high_resolution_clock::now();
    T norm_2 = RandLAPACK::util::get_2_norm(m, n, A.data(), state);
    auto stop = high_resolution_clock::now();
    long dur = duration_cast<microseconds>(stop - start).count();
    printf("Time for L2 norm %ld\n", dur);

    printf("THE LARGEST SINGULAR VALUE IS %f\n", s[0]);
    printf("THE LARGEST SINGULAR VALUE EST IS %f\n", norm_2);
    */

    std::copy(A.data(), A.data() + (m * n), A_1.data());
    std::copy(A.data(), A.data() + (m * n), A_2.data());

    // CQRRPT constructor
    RandLAPACK::CQRRPT<T> CQRRPT(false, true, state, std::numeric_limits<double>::epsilon());//std::pow(std::numeric_limits<double>::epsilon(), 0.75));
    CQRRPT.nnz                 = nnz;
    CQRRPT.num_threads         = 4;
    CQRRPT.cond_check          = cond_check;
    CQRRPT.naive_rank_estimate = naive_rank_estimate;
    CQRRPT.path = "../../../"; 
    CQRRPT.use_fro_norm = 1;

    // CQRRPT
    CQRRPT.call(m, n, A, d, R, J);

    printf("COND(A): %27e\n", std::get<1>(mat_type));
    printf("COND(A^{pre}): %21e\n", CQRRPT.cond_num_A_pre);
    printf("TRUE RANK(A): %14ld\n", true_k);
    printf("RANK(A) ESTIMATE: %10ld\n", CQRRPT.rank);
    int k = CQRRPT.rank;

    T* A_dat = A.data();
    T* A_1_dat = A_1.data();
    T* A_2_dat = A_2.data();
    T* R_dat = R.data();
    std::vector<T> I_ref(k * k, 0.0);
    RandLAPACK::util::eye(k, k, I_ref);
    T* I_ref_dat = I_ref.data();

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, A_dat, m, A_dat, m, -1.0, I_ref_dat, k);
    T norm_Q = lapack::lange(lapack::Norm::Fro, k, k, I_ref_dat, k);

    // Check approximation quality
    RandLAPACK::util::col_swap(m, n, n, A_1, J);
    RandLAPACK::util::col_swap(m, n, n, A_2, J);
    // AP - QR
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_1_dat, m);
    
    // Implementing max col norm metric
    T max_col_norm = 0.0;
    T col_norm = 0.0;
    int max_idx = 0;
    for(int i = 0; i < n; ++i) {
        col_norm = lapack::lange(Norm::Fro, m, 1, A_1_dat + (m * i), m);
        if(max_col_norm < col_norm) {
            max_col_norm = col_norm;
            max_idx = i;
        }
    }
    T col_norm_AP =  lapack::lange(Norm::Fro, m, 1, A_2_dat + (m * max_idx), m);

    T norm_APQR = lapack::lange(Norm::Fro, m, n, A_1_dat, m);
    T norm_AP = lapack::lange(Norm::Fro, m, n, A_2_dat, m);
    
    printf("REL NORM OF AP - QR: %15e\n", norm_APQR / norm_AP);
    printf("MAX COL NORM METRIC: %15e\n", max_col_norm / col_norm_AP);
    printf("FRO NORM OF Q' * Q - I: %2e\n\n", norm_Q);

    if(k != true_k)
        return 1;

    return 0;
}

template <typename T>
static void 
test_cond_orth(int row, 
           int col, 
           int64_t k,
           int64_t d,
           int64_t nnz,
           T cond_start,
           T cond_end,
           T cond_step,
           RandBLAS::base::RNGState<r123::Philox4x32> state,
           int naive_rank_estimate,
           int cond_check,
           int alg_type,
           int mat_type_num) {
    //printf("\n/-----------------------------------------CQRRPT CONDITION NUMBER BENCHMARK START-----------------------------------------/\n");
    if(naive_rank_estimate && alg_type) {
        printf("USING NAIVE RANK DETECTION\n\n");
    } else if(alg_type){
        printf("USING PRINCIPLED RANK DETECTION\n\n");
    }
    int detect_rank_underestimate = 1;
    T rank_underestimate_cond = 0;

    for (; cond_start <= cond_end; cond_start *= cond_step) {
        auto mat_type = std::make_tuple(mat_type_num, cond_start, false);
        //auto mat_type = std::make_tuple(0, cond_start, false);
        if(alg_type) {
            if(test_cond_helper_1<T>(row, col, k, d, nnz, mat_type, state, naive_rank_estimate, cond_check) && detect_rank_underestimate)
            {
                detect_rank_underestimate = 0;
                rank_underestimate_cond = cond_start;
            }
        } else {
            test_cond_helper_0<T>(row, col, mat_type, state);
        }
    }
    if(!detect_rank_underestimate)
    {
        printf("FAILED TO ACCURATELY ESTIMATE RANK FOR COND(A): %e\n", rank_underestimate_cond);
    }
    //printf("/-----------------------------------------CQRRPT CONDITION NUMBER EFFECT BENCHMARK STOP-----------------------------------------/\n\n");
}

int main(){
    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename  
    auto state = RandBLAS::base::RNGState(0, 0);
    // CholQR check
    //test_cond_orth<double>(10000, 1024, 1024, 1024, 1, 10, 10e16, 10, state, 0, 1, 0);
    //test_cond_orth<double>(1024, 5, 5, 5, 1, 10, 10, 10, state, 0, 1, 1);
    // CQRRPT check
    //test_cond_orth<double>(10000, 1024, 1024, 1024, 1, 1, 1, 10, state, 0, 1, 1);
    //test_cond_orth<double>(10000, 1024, 1024, 1024, 1, 1, 10e16, 10, state, 1, 1, 1);

    // Spiked data results for Oleg
    // Condition number parameter here is unused
    /*
    test_cond_orth<double>(10000, 2000, 2000, 2000, 1, 10e16, 10e16, 10, state, 0, 1, 1, 8);
    test_cond_orth<double>(10000, 2000, 2000, 3000, 4, 10e16, 10e16, 10, state, 0, 1, 1, 8);
    test_cond_orth<double>(10000, 2000, 2000, 4000, 4, 10e16, 10e16, 10, state, 0, 1, 1, 8);
    test_cond_orth<double>(10000, 2000, 2000, 6000, 4, 10e16, 10e16, 10, state, 0, 1, 1, 8);
    test_cond_orth<double>(10000, 2000, 2000, 8000, 4, 10e16, 10e16, 10, state, 0, 1, 1, 8);
    */

    // Scaled data results for Oleg
    // Condition number here acts as scaling "sigma"
    /*
    test_cond_orth<double>(10000, 2000, 2000, 2000, 1, 10e15, 10e15, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10000, 2000, 2000, 3000, 4, 10e15, 10e15, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10000, 2000, 2000, 4000, 4, 10e15, 10e15, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10000, 2000, 2000, 6000, 4, 10e15, 10e15, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10000, 2000, 2000, 8000, 4, 10e15, 10e15, 10, state, 0, 1, 1, 9);
    */

    // Oleg's testing approach
    // Condition number here acts as scaling "sigma"
    
    test_cond_orth<double>(10e6, 300, 295, 2 * 300, 4, 10e7, 10e7, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10e6, 300, 295, 2 * 300, 4, 10e9, 10e9, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10e6, 300, 295, 2 * 300, 4, 10e11, 10e11, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10e6, 300, 295, 2 * 300, 4, 10e13, 10e13, 10, state, 0, 1, 1, 9);
    test_cond_orth<double>(10e6, 300, 295, 2 * 300, 4, 10e15, 10e15, 10, state, 0, 1, 1, 9);
    
    // Not enough space for these
    
    //test_cond_orth<double>(10e6, 1000, 984, 2 * 1000, 4, 10e13, 10e13, 10, state, 1, 1, 1, 9);
    //test_cond_orth<double>(10e6, 1000, 984, 2 * 1000, 4, 10e15, 10e15, 10, state, 1, 1, 1, 9);

    //test_cond_orth<double>(10e7, 300, 300, 2 * 300, 4, 10e13, 10e13, 10, state, 1, 1, 1, 9);
    //test_cond_orth<double>(10e7, 300, 300, 2 * 300, 4, 10e15, 10e15, 10, state, 1, 1, 1, 9);

    return 0;
}