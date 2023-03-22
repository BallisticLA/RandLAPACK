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
test_speed_helper(int64_t m, 
                  int64_t n, 
                  const std::tuple<int, T, bool>& mat_type, 
                  RandBLAS::base::RNGState<r123::Philox4x32> state) {

    std::vector<T> A(m * n, 0.0);
    std::vector<T> I_ref(n * n, 0.0);

    // Generate random matrix
    RandLAPACK::util::gen_mat_type(m, n, A, n, state, mat_type);
    RandLAPACK::util::eye(n, n, I_ref);

    // CHOL QR
    RandLAPACK::CholQRQ<T> Orth_CholQR(false, false);
    // Orthonormalize A
    Orth_CholQR.call(m, n, A);

    T* A_dat = A.data();
    T* I_ref_dat = I_ref.data();

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_dat, m, A_dat, m, -1.0, I_ref_dat, n);
    T norm_fro = lapack::lange(lapack::Norm::Fro, n, n, I_ref_dat, n);
    printf("FRO NORM OF Q' * Q - I: %2.20f\n", norm_fro);
    printf("COND(A): %2.20f\n", std::get<1>(mat_type));
}

template <typename T>
static void 
test_speed(int r_pow, 
           int col, 
           T cond_start,
           T cond_end,
           T cond_step,
           RandBLAS::base::RNGState<r123::Philox4x32> state) {
    printf("\n/-----------------------------------------CholQR CONDITION NUMBER BENCHMARK START-----------------------------------------/\n");

    for (; cond_start <= cond_end; cond_start *= cond_step) {
        auto mat_type = std::make_tuple(0, cond_start, false);
        test_speed_helper<T>(std::pow(2, r_pow), col, mat_type, state);
    }
    printf("\n/-----------------------------------------CholQR CONDITION NUMBER EFFECT BENCHMARK STOP-----------------------------------------/\n\n");
}

int main(){
    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename  
    auto state = RandBLAS::base::RNGState(0, 0);
    test_speed<double>(17, 1024, 1, 10e7, 10, state);
    return 0;
}