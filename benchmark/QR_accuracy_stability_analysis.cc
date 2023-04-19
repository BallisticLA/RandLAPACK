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
error_check(int64_t m,
            int64_t n,
            int64_t k,
            T norm_A,
            T* A_dat,
            T* A_1_dat,
            T* Q_dat,
            T* R_dat,
            T* I_ref_dat
            ) {

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, 1.0, Q_dat, m, Q_dat, m, -1.0, I_ref_dat, k);
    T norm_0 = lapack::lange(lapack::Norm::Fro, k, k, I_ref_dat, k);

    // AP - QR
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, R_dat, k, -1.0, A_dat, m);
    
    // Implementing max col norm metric
    T max_col_norm = 0.0;
    T col_norm = 0.0;
    int max_idx = 0;
    for(int i = 0; i < n; ++i) {
        col_norm = lapack::lange(Norm::Fro, m, 1, A_dat + (m * i), m);
        if(max_col_norm < col_norm) {
            max_col_norm = col_norm;
            max_idx = i;
        }
    }
    T col_norm_A =  lapack::lange(Norm::Fro, m, 1, A_1_dat + (m * max_idx), m);
    T norm_AQR = lapack::lange(Norm::Fro, m, n, A_dat, m);
    
    printf("REL NORM OF AP - QR: %15e\n", norm_AQR / norm_A);
    printf("MAX COL NORM METRIC: %15e\n", max_col_norm / col_norm_A);
    printf("FRO NORM OF Q' * Q - I: %2e\n\n", norm_0);
}


template <typename T>
static void
cholqr_helper(int64_t m, 
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

template <typename T>
static void
cqrrpt_helper(int64_t m, 
                int64_t n, 
                const std::tuple<int, T, bool>& mat_type, 
                RandBLAS::base::RNGState<r123::Philox4x32> state,
                const std::vector<T>& additional_params) {

    int64_t true_k = additional_params[0];
    int64_t d = additional_params[1];

    std::vector<T> A(m * n, 0.0);
    std::vector<T> A_1(m * n, 0.0);
    std::vector<T> A_2(m * n, 0.0);
    std::vector<T> R(n * n, 0.0);
    std::vector<int64_t> J;

    // Generate random matrix
    RandLAPACK::util::gen_mat_type(m, n, A, true_k, state, mat_type);

    std::copy(A.data(), A.data() + (m * n), A_1.data());
    std::copy(A.data(), A.data() + (m * n), A_2.data());

    // CQRRPT constructor
    RandLAPACK::CQRRPT<T> CQRRPT(false, true, state, std::numeric_limits<double>::epsilon());
    CQRRPT.nnz                 = additional_params[2];
    CQRRPT.num_threads         = additional_params[3];
    CQRRPT.cond_check          = additional_params[4];
    CQRRPT.naive_rank_estimate = additional_params[5];
    CQRRPT.path = "../../../"; 
    CQRRPT.use_fro_norm = 0;

    // CQRRPT
    CQRRPT.call(m, n, A, d, R, J);

    printf("COND(A): %27e\n", std::get<1>(mat_type));
    printf("COND(A^{pre}): %21e\n", CQRRPT.cond_num_A_pre);
    printf("TRUE RANK(A): %14ld\n", true_k);
    printf("RANK(A) ESTIMATE: %10ld\n", CQRRPT.rank);
    int k = CQRRPT.rank;

    std::vector<T> I_ref(k * k, 0.0);
    RandLAPACK::util::eye(k, k, I_ref);

    // Check approximation quality
    RandLAPACK::util::col_swap(m, n, n, A_1, J);
    RandLAPACK::util::col_swap(m, n, n, A_2, J);

    error_check(m, n, k, lapack::lange(Norm::Fro, m, n, A.data(), m), A_1.data(), A_2.data(), A.data(), R.data(), I_ref.data()); 
}

template <typename T>
static void
scholqr_helper(int64_t m, 
                  int64_t n, 
                  const std::tuple<int, T, bool>& mat_type, 
                  RandBLAS::base::RNGState<r123::Philox4x32> state,
                  const std::vector<T>& additional_params) {

    int64_t k = additional_params[0];
    T shift_c = additional_params[1];

    std::vector<T> A(m * n, 0.0);
    std::vector<T> A_1(m * n, 0.0);
    std::vector<T> A_2(m * n, 0.0);

    std::vector<T> ATA(n * n, 0.0);
    std::vector<T> ATA1(n * n, 0.0);
    std::vector<T> ATA2(n * n, 0.0);
    std::vector<T> ATA_buf(n * n, 0.0);
    std::vector<T> R(n * n, 0.0);

    // Generate random matrix
    RandLAPACK::util::gen_mat_type(m, n, A, k, state, mat_type);

    std::copy(A.data(), A.data() + (m * n), A_1.data());
    std::copy(A.data(), A.data() + (m * n), A_2.data());

    T norm_A;    
    // Decide which norm we're using
    if(additional_params[2]) {
        norm_A = lapack::lange(Norm::Fro, m, n, A.data(), m);
    } else {
        norm_A = RandLAPACK::util::get_2_norm(m, n, A.data(), state);
    }

    T shift = std::numeric_limits<double>::epsilon() * shift_c * std::pow(norm_A, 2);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A.data(), m, 0.0, ATA.data(), n);

    for (int i = 0; i < n; ++i) {
        ATA[i * (n + 1)] += shift;
    }
    
    lapack::potrf(Uplo::Upper, n, ATA.data(), n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, ATA.data(), n, A.data(), m);
    // Define R1
    std::copy(ATA.data(), ATA.data() + (n * n), ATA1.data());

    // CholeskyQR2
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A.data(), m, 0.0, ATA.data(), n);
    lapack::potrf(Uplo::Upper, n, ATA.data(), n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, ATA.data(), n, A.data(), m);
    // Define R2
    std::copy(ATA.data(), ATA.data() + (n * n), ATA2.data());

    // CholeskyQR3
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A.data(), m, 0.0, ATA.data(), n);
    lapack::potrf(Uplo::Upper, n, ATA.data(), n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, ATA.data(), n, A.data(), m);

    // Re-combine R's
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n, 1.0, ATA.data(), n, ATA2.data(), n, 0.0, ATA_buf.data(), n);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n, 1.0, ATA_buf.data(), n, ATA1.data(), n, 0.0, R.data(), n);

    std::vector<T> I_ref(n * n, 0.0);
    RandLAPACK::util::eye(n, n, I_ref);

    error_check(m, n, n, norm_A, A_1.data(), A_2.data(), A.data(), R.data(), I_ref.data());
}

template <typename T>
static void 
test_cond_orth(int row, 
           int col, 
           T cond_start,
           T cond_end,
           T cond_step,
           RandBLAS::base::RNGState<r123::Philox4x32> state,
           int alg_type,
           int mat_type_num,
           std::vector<T> additional_params) {

    for (; cond_start <= cond_end; cond_start *= cond_step) {

        auto mat_type = std::make_tuple(mat_type_num, cond_start, true);
        switch(alg_type) {
            case 0:
                    // CholQR
                    cholqr_helper<T>(row, col, mat_type, state);
                    break;
            case 1:
                    // CQRRPT
                    cqrrpt_helper<T>(row, col, mat_type, state, additional_params);
                    break;
            case 2:
                    // sCholQR
                    scholqr_helper(row, col, mat_type, state, additional_params);
                    break;
            default:
            throw std::runtime_error(std::string("Unrecognized case."));
            break;
        }
    }
}

int main(){
    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename  
    auto state = RandBLAS::base::RNGState(0, 0);
    // Old tests
    //test_cond_orth<double>(10000, 1024, 1024, 1024, 1, 10, 10e16, 10, state, 0, 1, 0);
    //test_cond_orth<double>(1024, 5, 5, 5, 1, 10, 10, 10, state, 0, 1, 1);
    // CQRRPT check
    //test_cond_orth<double>(10000, 1024, 1024, 1024, 1, 1, 1, 10, state, 0, 1, 1);
    //test_cond_orth<double>(10000, 1024, 1024, 1024, 1, 1, 10e16, 10, state, 1, 1, 1);

    // Things presented on 04/17/2023

    // Spiked data Max test
    // Condition number parameter here is unused
    /*
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 1, 8, {2000, 2000, 1, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 1, 8, {2000, 3000, 4, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 1, 8, {2000, 4000, 4, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 1, 8, {2000, 6000, 4, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 1, 8, {2000, 8000, 4, 4, 1, 1});
    */

    // Scaled data Max test
    // Condition number here acts as scaling "sigma"
    /*
    test_cond_orth<double>(10000, 2000, 10e15, 10e15, 10, state, 1, 9, {2000, 2000, 1, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e15, 10e15, 10, state, 1, 9, {2000, 3000, 4, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e15, 10e15, 10, state, 1, 9, {2000, 4000, 4, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e15, 10e15, 10, state, 1, 9, {2000, 6000, 4, 4, 1, 1});
    test_cond_orth<double>(10000, 2000, 10e15, 10e15, 10, state, 1, 9, {2000, 8000, 4, 4, 1, 1});
    */

    // Oleg test
    // Condition number here acts as scaling "sigma"
    //test_cond_orth<double>(10e6, 300, 10e7, 10e15, 100, state, 1, 9, {295, 2 * 300, 4, 4, 0, 1});

    // Things to present on 04/14/2023

    // Checking sCholQR3
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 2, 8, {2000, 11 * 1, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 2, 8, {2000, 11 * 5, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 2, 8, {2000, 11 * 2000, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 2, 8, {2000, 11 * 10000, 1});
    test_cond_orth<double>(10000, 2000, 10e16, 10e16, 10, state, 2, 8, {2000, 11 * (10000 * 2000 + 2000 * (2000 + 1)), 1});
    
    // Oleg test - An attempt to get an even higher condition number of A^{pre}
    // Condition number here acts as scaling "sigma"
    //test_cond_orth<double>(10e6, 500, 10e15, 10e21, 100, state, 1, 9, {500, 2 * 500, 4, 4, 0, 1});
    return 0;
}