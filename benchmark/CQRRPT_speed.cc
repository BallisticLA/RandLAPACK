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
/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/
/*
Compares speed of CQRRPT to other pivoted and unpivoted QR factorizations
*/

using namespace std::chrono;

template <typename T>
static void 
print_info(int64_t rows, 
    int64_t cols,
    T d_multiplier, 
    int64_t nnz, 
    int64_t num_threads, 
    T cqrrpt_time, 
    T geqp3_time, 
    T geqr_time, 
    T tsqrp_time, 
    T geqrf_time, 
    T scholqr_time,
    T chol_full_time,
    T geqp3_full_time,
    T geqr_full_time,
    T tsqrp_full_time,
    T geqrf_full_time,
    T scholqr_full_time,
    const std::string& test_type,
    int runs) {
    const char * test_type_print = test_type.c_str();

    printf("\n/-------------------------------------QR TIMING INFO BEGIN-------------------------------------/\n");
            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Embedding size: %f.\n", d_multiplier * cols);
            printf("Number of nonzeros per column in SASO: %ld\n", nnz);
            printf("Number of threads used in SASO application: %ld\n", num_threads);

            printf("%s timing of CQRRPT for %d runs: %54.2f μs. Full timing: %f μs.\n",                                test_type_print, runs - 1, cqrrpt_time, chol_full_time);
            printf("%s timing of GEQP3 for %d runs: %57.2f μs. Full timing: %f μs.\n",                                   test_type_print, runs - 1, geqp3_time, geqp3_full_time);
            printf("%s timing of GEQRF for %d runs: %57.2f μs. Full timing: %f μs.\n",                                   test_type_print, runs - 1, geqrf_time, geqrf_full_time);
            printf("%s timing of GEQR for %d runs: %58.2f μs. Full timing: %f μs.\n",                                    test_type_print, runs - 1, geqr_time, geqr_full_time);
            printf("%s timing of TSQRP for %d runs: %57.2f μs. Full timing: %f μs.\n\n",                                 test_type_print, runs - 1, tsqrp_time, tsqrp_full_time);
            printf("%s timing of SCHOLQR3 for %d runs: %54.2f μs. Full timing: %f μs.\n\n",                                 test_type_print, runs - 1, scholqr_time, scholqr_full_time);

            /*CQRRPT vs GEQP3*/
            if(cqrrpt_time < geqp3_time) {
                printf("Result: CQRRPT is %33.2f times faster than GEQP3.\n",                         geqp3_time / cqrrpt_time);
            } else {
                printf("Result: CQRRPT is %33.2f times slower than GEQP3.\n",                         cqrrpt_time / geqp3_time);
            }

            if(chol_full_time < geqp3_full_time) {
                printf("With space allocation + application: %3.2f times faster than GEQP3.\n\n", geqp3_full_time / chol_full_time);
            } else {
                printf("With space allocation + application: %3.2f times slower than GEQP3.\n\n", chol_full_time / geqp3_full_time);
            }

            /*CQRRPT vs TSQRP*/
            if(cqrrpt_time < tsqrp_time) {
                printf("Result: CQRRPT is %33.2f times faster than TSQRP.\n",                         tsqrp_time / cqrrpt_time);
            } else {
                printf("Result: CQRRPT is %33.2f times slower than TSQRP.\n",                         cqrrpt_time / tsqrp_time);
            }
            if(chol_full_time < tsqrp_full_time) {
                printf("With space allocation + application: %3.2f times faster than TSQRP.\n\n", tsqrp_full_time / chol_full_time);
            } else {
                printf("With space allocation + application: %3.2f times slower than TSQRP.\n\n", chol_full_time / tsqrp_full_time);
            }

            /*CQRRPT vs GEQRF*/
            if(cqrrpt_time < geqrf_time) {
                printf("Result: CQRRPT is %33.2f times faster than GEQRF.\n",                         geqrf_time / cqrrpt_time);
            } else {
                printf("Result: CQRRPT is %33.2f times slower than GEQRF.\n",                         cqrrpt_time / geqrf_time);
            }
            if(chol_full_time < geqrf_full_time) {
                printf("With space allocation + application: %3.2f times faster than GEQRF.\n\n", geqrf_full_time / chol_full_time);
            } else {
                printf("With space allocation + application: %3.2f times slower than GEQRF.\n\n", chol_full_time / geqrf_full_time);
            }

            /*CQRRPT vs GEQR*/
            if(cqrrpt_time < geqr_time) {
                printf("Result: CQRRPT is %33.2f times faster than GEQR.\n",                          geqr_time / cqrrpt_time);
            } else {
                printf("Result: CQRRPT is %33.2f times slower than GEQR.\n",                          cqrrpt_time / geqr_time);
            }
            if(chol_full_time < geqr_full_time) {
                printf("With space allocation + application: %3.2f times faster than GEQR.\n\n",  geqr_full_time / chol_full_time);
            } else {
                printf("With space allocation + application: %3.2f times slower than GEQR.\n\n",  chol_full_time / geqr_full_time);
            }

            /*CQRRPT vs SCHOLQR3*/
            if(cqrrpt_time < scholqr_time) {
                printf("Result: CQRRPT is %33.2f times faster than SCHOLQR3.\n",                          scholqr_time / cqrrpt_time);
            } else {
                printf("Result: CQRRPT is %33.2f times slower than SCHOLQR3.\n",                          cqrrpt_time / scholqr_time);
            }
            if(chol_full_time < scholqr_full_time) {
                printf("With space allocation + application: %3.2f times faster than SCHOLQR3.\n\n",  scholqr_full_time / chol_full_time);
            } else {
                printf("With space allocation + application: %3.2f times slower than SCHOLQR3.\n\n",  chol_full_time / scholqr_full_time);
            }

            printf("\n/---------------------------------------QR TIMING INFO END---------------------------------------/\n\n");

}

template <typename T>
static void 
log_info(int64_t rows, 
           int64_t cols,
           T d_multiplier,
           T k_multiplier, 
           T tol,
           int64_t nnz, 
           int64_t num_threads,
           const std::tuple<int, T, bool>& mat_type,
           T cqrrpt_time, 
           T geqp3_time, 
           T geqr_time, 
           T tsqrp_time, 
           T geqrf_time, 
           T scholqr_time,
           T chol_full_time,
           T geqp3_full_time,
           T geqr_full_time,
           T tsqrp_full_time,
           T geqrf_full_time,
           T scholqr_full_time,
           const std::string& test_type,
           int runs,
           int apply_to_large,
           std::string path) {
    // Save the output into .dat file
    std::fstream file(path + "CQRRPT_comp_time_" + test_type 
                                + "_m_"            + std::to_string(rows) 
                                + "_d_multiplier_" + std::to_string(d_multiplier)
                                + "_k_multiplier_" + std::to_string(k_multiplier)
                                + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                + "_nnz_"          + std::to_string(nnz)
                                + "_runs_per_sz_"  + std::to_string(runs)
                                + "_OMP_threads_"  + std::to_string(num_threads) 
                                + "_apply_to_large_" + std::to_string(apply_to_large)
                                + ".dat", std::fstream::app);
    file << cqrrpt_time       << "  " 
         << geqp3_time        << "  " 
         << geqr_time         << "  "
         << tsqrp_time        << "  " 
         << geqrf_time        << "  "  
         << scholqr_time      << "  "
         << chol_full_time    << "  " 
         << geqp3_full_time   << "  " 
         << geqr_full_time    << "  "
         << tsqrp_full_time   << "  " 
         << geqrf_full_time   << "  "
         << scholqr_full_time << "\n";

    print_info(rows, cols, d_multiplier, nnz, num_threads, 
               cqrrpt_time,
               geqp3_time,
               geqr_time,
               tsqrp_time,
               geqrf_time,  
               scholqr_time,
               chol_full_time, 
               geqp3_full_time, 
               geqr_full_time,
               tsqrp_full_time, 
               geqrf_full_time,
               scholqr_full_time,
               test_type, runs);
}

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, 
                  int64_t n, 
                  int64_t d, 
                  int64_t k, 
                  T tol, 
                  int64_t nnz, 
                  int64_t num_threads, 
                  const std::tuple<int, T, bool>& mat_type, 
                  RandBLAS::base::RNGState<r123::Philox4x32> state,
                  int apply_to_large) {

    int64_t size = m * n;
    int64_t b_dim = 10;
    if(apply_to_large)
        b_dim = n;
    std::vector<T> A_1(size, 0.0);
    std::vector<T> B_1(b_dim * m, 0.0);
    std::vector<T> R_1;
    std::vector<int64_t> J_1;
    std::vector<T> Res_1;
    RandLAPACK::util::upsize(b_dim * n, Res_1);
    std::vector<int64_t> J_2;
    std::vector<T> tau_2;
    std::vector<T> t_3;
    std::vector<T> R_3;
    std::vector<T> tau_3;
    std::vector<int64_t> J_3;
    std::vector<T> tau_4;
    std::vector<T> ATA;

    // Generate random matrix
    RandLAPACK::util::gen_mat_type(m, n, A_1, k, state, mat_type);

    // Generate random matrix that we will apply Q to
    RandLAPACK::util::gen_mat_type(b_dim, m, B_1, b_dim, state, mat_type);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // CQRRPT constructor
    bool log_times = false;
    RandLAPACK::CQRRPT<T> CQRRPT(false, log_times, state, tol);
    CQRRPT.nnz = nnz;
    CQRRPT.num_threads = num_threads;
    /*
    //-TEST POINT 1 BEGIN-------------------------------------------------------------------------------------------------------------------------------------------/
    
    // Pre-allocation for CQRRPT
    auto start_alloc1 = high_resolution_clock::now();
    if(log_times) {
        (CQRRPT.times).resize(10);
    }
    RandLAPACK::util::upsize(d * n, (CQRRPT.A_hat));
    RandLAPACK::util::upsize(n, (CQRRPT.A_hat));
    J_1.resize(n);
    RandLAPACK::util::upsize(n * n, (CQRRPT.R_sp));
    RandLAPACK::util::upsize(n * n, R_1);
    auto stop_alloc1 = high_resolution_clock::now();
    long dur_alloc1 = duration_cast<microseconds>(stop_alloc1 - start_alloc1).count();
    
    // CQRRPT
    auto start_cqrrpt = high_resolution_clock::now();
    CQRRPT.call(m, n, A_1, d, R_1, J_1);
    auto stop_cqrrpt = high_resolution_clock::now();
    long dur_cqrrpt = duration_cast<microseconds>(stop_cqrrpt - start_cqrrpt).count();
    
    // CQRRPT verbose info print
    if (log_times) {
        printf("\n\n/------------CQRRPT TIMING RESULTS BEGIN------------/\n");
        printf("SASO time: %33ld μs,\n",                    (CQRRPT.times)[0]);
        printf("QRCP time: %33ld μs,\n",                    (CQRRPT.times)[1]);
        printf("Rank revealing time: %23ld μs,\n",          (CQRRPT.times)[2]);
        printf("CholQR time: %31ld μs,\n",                  (CQRRPT.times)[3]);
        printf("A modification pivoting time: %14ld μs,\n", (CQRRPT.times)[4]);
        printf("A modification TRSM time: %18ld μs,\n",     (CQRRPT.times)[5]);
        printf("Copying time: %30ld μs,\n",                 (CQRRPT.times)[6]);
        printf("Resizing time: %29ld μs,\n",                (CQRRPT.times)[7]);
        printf("Other routines time: %23ld μs,\n",          (CQRRPT.times)[8]);
        printf("Total time: %32ld μs.\n",                   (CQRRPT.times)[9]);

        printf("\nSASO generation and application takes %2.2f%% of runtime.\n", 100 * ((double) (CQRRPT.times)[0] / (double) (CQRRPT.times)[9]));
        printf("QRCP takes %32.2f%% of runtime.\n",                            100 * ((double) (CQRRPT.times)[1] / (double) (CQRRPT.times)[9]));
        printf("Rank revealing takes %22.2f%% of runtime.\n",                  100 * ((double) (CQRRPT.times)[2] / (double) (CQRRPT.times)[9]));
        printf("Cholqr takes %30.2f%% of runtime.\n",                          100 * ((double) (CQRRPT.times)[3] / (double) (CQRRPT.times)[9]));
        printf("Modifying matrix (pivoting) A %13.2f%% of runtime.\n",         100 * ((double) (CQRRPT.times)[4] / (double) (CQRRPT.times)[9]));
        printf("Modifying matrix (trsm) A %17.2f%% of runtime.\n",             100 * ((double) (CQRRPT.times)[5] / (double) (CQRRPT.times)[9]));
        printf("Copying takes %29.2f%% of runtime.\n",                         100 * ((double) (CQRRPT.times)[6] / (double) (CQRRPT.times)[9]));
        printf("Resizing takes %28.2f%% of runtime.\n",                        100 * ((double) (CQRRPT.times)[7] / (double) (CQRRPT.times)[9]));
        printf("Everything else takes %21.2f%% of runtime.\n",                 100 * ((double) (CQRRPT.times)[8] / (double) (CQRRPT.times)[9]));
        printf("/-------------CQRRPT TIMING RESULTS END-------------/\n\n");
    }

    // Apply Q_1
    auto start_appl1 = high_resolution_clock::now();
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_dim, n, m, 1.0, B_1.data(), b_dim, A_1.data(), m, 0.0, Res_1.data(), b_dim);
    auto stop_appl1 = high_resolution_clock::now();
    long dur_appl1 = duration_cast<microseconds>(stop_appl1 - start_appl1).count();

    //-TEST POINT 1 END---------------------------------------------------------------------------------------------------------------------------------------------/
    // Re-generate matrix
    RandLAPACK::util::gen_mat_type(m, n, A_1, k, state, mat_type);

    // Pre-allocation for GEQP3
    auto start_alloc2 = high_resolution_clock::now();
    RandLAPACK::util::upsize(n, tau_2);
    J_2.resize(n);
    auto stop_alloc2 = high_resolution_clock::now();
    long dur_alloc2 = duration_cast<microseconds>(stop_alloc2 - start_alloc2).count();

    // GEQP3
    auto start_geqp3 = high_resolution_clock::now();
    lapack::geqp3(m, n, A_1.data(), m, J_2.data(), tau_2.data());
    auto stop_geqp3 = high_resolution_clock::now();
    long dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();

    // Apply Q_2
    // Re-generate the random matrix
    RandLAPACK::util::gen_mat_type(b_dim, m, B_1, b_dim, state, mat_type);
    auto start_appl2 = high_resolution_clock::now();
    lapack::ormqr(Side::Right, Op::NoTrans, b_dim, m, b_dim, A_1.data(), m, tau_2.data(), B_1.data(), b_dim);
    auto stop_appl2 = high_resolution_clock::now();
    long dur_appl2 = duration_cast<microseconds>(stop_appl2 - start_appl2).count();

    //-TEST POINT 2 END---------------------------------------------------------------------------------------------------------------------------------------------/
    // Re-generate matrix
    RandLAPACK::util::gen_mat_type(m, n, A_1, k, state, mat_type);

    // Pre-allocation for GEQR
    auto start_alloc3 = high_resolution_clock::now();
    RandLAPACK::util::upsize(5, t_3);
    auto stop_alloc3 = high_resolution_clock::now();
    long dur_alloc3 = duration_cast<microseconds>(stop_alloc3 - start_alloc3).count();

    // Pre-allocation for GEQP3
    auto start_alloc4 = high_resolution_clock::now();
    RandLAPACK::util::upsize(n * n, R_3);
    RandLAPACK::util::upsize(n, tau_3);
    J_3.resize(n);
    auto stop_alloc4 = high_resolution_clock::now();
    long dur_alloc4 = duration_cast<microseconds>(stop_alloc4 - start_alloc4).count();

    // GEQR + GEQP3
    auto start_tsqrp = high_resolution_clock::now();
    // GEQR part
    auto sart_geqr = high_resolution_clock::now();
#if !defined(__APPLE__)
    lapack::geqr(m, n, A_1.data(), m, t_3.data(), -1);
    int64_t tsize = (int64_t) t_3[0]; 
    t_3.resize(tsize);
    lapack::geqr(m, n, A_1.data(), m, t_3.data(), tsize);
#endif
    auto stop_geqr = high_resolution_clock::now();
    long dur_geqr = duration_cast<microseconds>(stop_geqr - sart_geqr).count();

    // Apply Q_3
    // Re-generate the random matrix
    RandLAPACK::util::gen_mat_type(b_dim, m, B_1, b_dim, state, mat_type);
    auto start_appl3 = high_resolution_clock::now();
    lapack::ormqr(Side::Right, Op::NoTrans, b_dim, m, b_dim, A_1.data(), m, t_3.data(), B_1.data(), b_dim);
    auto stop_appl3 = high_resolution_clock::now();
    long dur_appl3 = duration_cast<microseconds>(stop_appl3 - start_appl3).count();
#if !defined(__APPLE__)
    // GEQP3 on R part
    // We are not timing the pre-allocation of R, as it expected to take very small time
    RandLAPACK::util::get_U(m, n, A_1, R_3);
    lapack::geqp3(n, n, R_3.data(), n, J_3.data(), tau_3.data());
#endif
    auto stop_tsqrp = high_resolution_clock::now();
    long dur_tsqrp = duration_cast<microseconds>(stop_tsqrp - start_tsqrp).count() - dur_appl3;

    // Apply Q_4
    // Re-generate the random matrix
    RandLAPACK::util::gen_mat_type(b_dim, m, B_1, b_dim, state, mat_type);
    auto start_appl4 = high_resolution_clock::now();
    lapack::ormqr(Side::Right, Op::NoTrans, b_dim, m, b_dim, A_1.data(), m, tau_3.data(), B_1.data(), b_dim);
    auto stop_appl4 = high_resolution_clock::now();
    long dur_appl4 = duration_cast<microseconds>(stop_appl4 - start_appl4).count();

    //-TEST POINT 3&4 END-------------------------------------------------------------------------------------------------------------------------------------------/
    // Re-generate matrix
    RandLAPACK::util::gen_mat_type(m, n, A_1, k, state, mat_type);

    // Pre-allocation for GEQRF
    auto start_alloc5 = high_resolution_clock::now();
    RandLAPACK::util::upsize(n, tau_4);
    auto stop_alloc5 = high_resolution_clock::now();
    long dur_alloc5 = duration_cast<microseconds>(stop_alloc5 - start_alloc5).count();

    // GEQRF
    auto start_geqrf = high_resolution_clock::now();
    lapack::geqrf(m, n, A_1.data(), m, tau_4.data());
    auto stop_geqrf = high_resolution_clock::now();
    long dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

    // Apply Q_5
    // Re-generate the random matrix
    RandLAPACK::util::gen_mat_type(b_dim, m, B_1, b_dim, state, mat_type);
    auto start_appl5 = high_resolution_clock::now();
    lapack::ormqr(Side::Right, Op::NoTrans, b_dim, m, b_dim, A_1.data(), m, tau_4.data(), B_1.data(), b_dim);
    auto stop_appl5 = high_resolution_clock::now();
    long dur_appl5 = duration_cast<microseconds>(stop_appl5 - start_appl5).count();
    */
    //-TEST POINT 5 END---------------------------------------------------------------------------------------------------------------------------------------------/
    // Re-generate matrix
    RandLAPACK::util::gen_mat_type(m, n, A_1, k, state, mat_type);

    // Pre-allocation for Shifted Cholesky QR 3
    auto start_alloc6 = high_resolution_clock::now();
    RandLAPACK::util::upsize(n * n, ATA);
    auto stop_alloc6 = high_resolution_clock::now();
    long dur_alloc6 = duration_cast<microseconds>(stop_alloc6 - start_alloc6).count();
    
    // Shifted Cholesky QR 3
    auto start_scholqr = high_resolution_clock::now();
    
    // Below line is impractical
    //T norm_A = RandLAPACK::util::get_2_norm(m, n, A_1.data(), state);
    T norm_A = lapack::lange(Norm::Fro, m, n, A_1.data(), m);
    
    T shift = 11 * std::numeric_limits<double>::epsilon() * m * std::pow(norm_A, 2);

    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A_1.data(), m, 0.0, ATA.data(), n);

    for (int i = 0; i < n; ++i) {
        ATA[i * (n + 1)] += shift;
    }
    
    lapack::potrf(Uplo::Upper, n, ATA.data(), n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, ATA.data(), n, A_1.data(), m);

    // CholeskyQR2
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A_1.data(), m, 0.0, ATA.data(), n);
    lapack::potrf(Uplo::Upper, n, ATA.data(), n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, ATA.data(), n, A_1.data(), m);
    
    // CholeskyQR3
    //blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A_1.data(), m, 0.0, ATA.data(), n);
    //lapack::potrf(Uplo::Upper, n, ATA.data(), n);
    //blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, ATA.data(), n, A_1.data(), m);
    
    std::vector<T> I_ref(n * n, 0.0);
    RandLAPACK::util::eye(n, n, I_ref);
    T* I_ref_dat = I_ref.data();

    // Check orthogonality of Q
    // Q' * Q  - I = 0
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m, 1.0, A_1.data(), m, A_1.data(), m, -1.0, I_ref_dat, n);
    T norm_Q = lapack::lange(lapack::Norm::Fro, n, n, I_ref_dat, n);
    printf("ORTHOGONALITY MEASURE OF Q:%e\n", norm_Q);

    /*
    auto stop_scholqr = high_resolution_clock::now();
    long dur_scholqr = duration_cast<microseconds>(stop_scholqr - start_scholqr).count();

    // Apply Q_6
    // Re-generate the random matrix
    auto start_appl6 = high_resolution_clock::now();
    gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, b_dim, n, m, 1.0, B_1.data(), b_dim, A_1.data(), m, 0.0, Res_1.data(), b_dim);
    auto stop_appl6 = high_resolution_clock::now();
    long dur_appl6 = duration_cast<microseconds>(stop_appl6 - start_appl6).count();


    //-TEST POINT 6 END---------------------------------------------------------------------------------------------------------------------------------------------/

    std::vector<long> res{dur_alloc1, dur_cqrrpt, dur_appl1, 
                            dur_alloc2, dur_geqp3,    dur_appl2,
                            dur_alloc3, dur_geqr,     dur_appl3,
                            dur_alloc4, dur_tsqrp,    dur_appl4,
                            dur_alloc5, dur_geqrf,    dur_appl5,
                            dur_alloc6, dur_scholqr,  dur_appl6}; 
    */
    std::vector<long> res(18, 0);
    return res;
}

template <typename T>
static void 
test_speed(int r_pow, 
           int r_pow_max, 
           int col, 
           int col_max, 
           int runs, 
           int nnz, 
           int num_threads, 
           T tol, 
           T k_multiplier, 
           T d_multiplier, 
           const std::tuple<int, T, bool> & mat_type,
           int apply_to_large,
           std::string path,
           RandBLAS::base::RNGState<r123::Philox4x32> state) {

    printf("\n/-----------------------------------------QRCP SPEED BENCHMARK START-----------------------------------------/\n");
    // This variable is controls an additional iteration, used for initialization work
    int initialization = 1;
    int curr_runs = 0;
    
    // We are now filling 3 types of data - best, mean and raw
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf) {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open(path + "CQRRPT_comp_time_Best_m_"   + std::to_string(rows) 
                                    + "_d_multiplier_" + std::to_string(d_multiplier)
                                    + "_k_multiplier_" + std::to_string(k_multiplier)
                                    + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                    + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                    + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                    + "_nnz_"          + std::to_string(nnz)
                                    + "_runs_per_sz_"  + std::to_string(runs)
                                    + "_OMP_threads_"  + std::to_string(num_threads) 
                                    + "_apply_to_large_" + std::to_string(apply_to_large)
                                    + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open(path + "CQRRPT_comp_time_Mean_m_"   + std::to_string(rows) 
                                    + "_d_multiplier_" + std::to_string(d_multiplier)
                                    + "_k_multiplier_" + std::to_string(k_multiplier)
                                    + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                    + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                    + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                    + "_nnz_"          + std::to_string(nnz)
                                    + "_runs_per_sz_"  + std::to_string(runs)
                                    + "_OMP_threads_"  + std::to_string(num_threads) 
                                    + "_apply_to_large_" + std::to_string(apply_to_large)
                                    + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();

        ofs.open(path + "CQRRPT_comp_time_Raw_m_"    + std::to_string(rows) 
                                    + "_d_multiplier_" + std::to_string(d_multiplier)
                                    + "_k_multiplier_" + std::to_string(k_multiplier)
                                    + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                    + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                    + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                    + "_nnz_"          + std::to_string(nnz)
                                    + "_runs_per_sz_"  + std::to_string(runs)
                                    + "_OMP_threads_"  + std::to_string(num_threads) 
                                    + "_apply_to_large_" + std::to_string(apply_to_large)
                                    + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }
    
    int64_t rows = 0;
    for(; r_pow <= r_pow_max; ++r_pow) {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols *= 2) {
            std::vector<long> res;

            long t_alloc1   = 0;
            long t_cqrrpt = 0;
            long t_appl1    = 0;
            long t_alloc2   = 0;
            long t_geqp3    = 0;
            long t_appl2    = 0;
            long t_alloc3   = 0;
            long t_geqr     = 0;
            long t_appl3    = 0;
            long t_alloc4   = 0;
            long t_tsqrp    = 0;
            long t_appl4    = 0;
            long t_alloc5   = 0;
            long t_geqrf    = 0;
            long t_appl5    = 0;
            long t_alloc6   = 0;
            long t_scholqr  = 0;
            long t_appl6    = 0;

            T alloc1_best   = 0;
            T cqrrpt_best = 0;
            T appl1_best    = 0;
            T alloc2_best   = 0;
            T geqp3_best    = 0;
            T appl2_best    = 0;
            T alloc3_best   = 0;
            T geqr_best     = 0;
            T appl3_best    = 0;
            T alloc4_best   = 0;
            T tsqrp_best    = 0;
            T appl4_best    = 0;
            T alloc5_best   = 0;
            T geqrf_best    = 0;
            T appl5_best    = 0;
            T alloc6_best   = 0;
            T scholqr_best  = 0;
            T appl6_best    = 0;

            T alloc1_mean   = 0;
            T cqrrpt_mean = 0;
            T appl1_mean    = 0;
            T alloc2_mean   = 0;
            T geqp3_mean    = 0;
            T appl2_mean    = 0;
            T alloc3_mean   = 0;
            T geqr_mean     = 0;
            T appl3_mean    = 0;
            T alloc4_mean   = 0;
            T tsqrp_mean    = 0;
            T appl4_mean    = 0;
            T alloc5_mean   = 0;
            T geqrf_mean    = 0;
            T appl5_mean    = 0;
            T alloc6_mean   = 0;
            T scholqr_mean  = 0;
            T appl6_mean    = 0;

            curr_runs = runs + initialization;
            for(int i = 0; i < curr_runs; ++i) {
                res = test_speed_helper<T>(rows, cols, d_multiplier * cols, k_multiplier * cols, tol, nnz, num_threads, mat_type, state, apply_to_large);

                // Skip first iteration, as it tends to produce garbage results
                if (!initialization) {
                    t_alloc1   += res[0];
                    t_cqrrpt += res[1];
                    t_appl1    += res[2];
                    t_alloc2   += res[3];
                    t_geqp3    += res[4];
                    t_appl2    += res[5];
                    t_alloc3   += res[6];
                    t_geqr     += res[7];
                    t_appl3    += res[8];
                    t_alloc4   += res[9];
                    t_tsqrp    += res[10];
                    t_appl4    += res[11];
                    t_alloc5   += res[12];
                    t_geqrf    += res[13];
                    t_appl5    += res[14];
                    t_alloc6   += res[15];
                    t_scholqr  += res[16];
                    t_appl6    += res[17];
                    
                    // Log every run in the raw data file
                    std::fstream file(path + "CQRRPT_comp_time_Raw_m_"   + std::to_string(rows) 
                                                        + "_d_multiplier_" + std::to_string(d_multiplier)
                                                        + "_k_multiplier_" + std::to_string(k_multiplier)
                                                        + "_log10(tol)_"   + std::to_string(long(log10(tol)))
                                                        + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                        + "_cond_"         + std::to_string(long(std::get<1>(mat_type)))
                                                        + "_nnz_"          + std::to_string(nnz)
                                                        + "_runs_per_sz_"  + std::to_string(runs)
                                                        + "_OMP_threads_"  + std::to_string(num_threads) 
                                                        + "_apply_to_large_" + std::to_string(apply_to_large)
                                                        + ".dat", std::fstream::app);
                    file << res[0]  << "  " 
                         << res[1]  << "  " 
                         << res[2]  << "  "
                         << res[3]  << "  " 
                         << res[4]  << "  "  
                         << res[5]  << "  " 
                         << res[6]  << "  " 
                         << res[7]  << "  "
                         << res[8]  << "  " 
                         << res[9]  << "  "
                         << res[10] << "  " 
                         << res[11] << "  " 
                         << res[12] << "  "
                         << res[13] << "  "
                         << res[14] << "  " 
                         << res[15] << "  "
                         << res[16] << "  "
                         << res[17] << "\n";
                    
                    // For best timing
                    if(alloc1_best > res[0] || alloc1_best == 0) {
                        alloc1_best = res[0];
                    }
                    if(cqrrpt_best > res[1] || cqrrpt_best == 0) {
                        cqrrpt_best = res[1];
                    }
                    if(appl1_best > res[2] || appl1_best == 0) {
                        appl1_best = res[2];
                    }
                    if(alloc2_best > res[3] || alloc2_best == 0) {
                        alloc2_best = res[3];
                    }
                    if(geqp3_best > res[4] || geqp3_best == 0) {
                        geqp3_best = res[4];
                    }
                    if(appl2_best > res[5] || appl2_best == 0) {
                        appl2_best = res[5];
                    }
                    if(alloc3_best > res[6] || alloc3_best == 0) {
                        alloc3_best = res[6];
                    }
                    if(geqr_best > res[7] || geqr_best == 0) {
                        geqr_best = res[7];
                    }
                    if(appl3_best > res[8] || appl3_best == 0) {
                        appl3_best = res[8];
                    }
                    if(alloc4_best > res[9] || alloc4_best == 0) {
                        alloc4_best = res[9];
                    }
                    if(tsqrp_best > res[10] || tsqrp_best == 0) {
                        tsqrp_best = res[10];
                    }
                    if(appl4_best > res[11] || appl4_best == 0) {
                        appl4_best = res[11];
                    }
                    if(alloc5_best > res[12] || alloc5_best == 0) {
                        alloc5_best = res[12];
                    }
                    if(geqrf_best > res[13] || geqrf_best == 0) {
                        geqrf_best = res[13];
                    }
                    if(appl5_best > res[14] || appl5_best == 0) {
                        appl5_best = res[14];
                    }
                    if(alloc6_best > res[15] || alloc6_best == 0) {
                        alloc6_best = res[15];
                    }
                    if(scholqr_best > res[16] || scholqr_best == 0) {
                        scholqr_best = res[16];
                    }
                    if(appl6_best > res[17] || appl6_best == 0) {
                        appl6_best = res[17];
                    }
                }
                initialization = 0;
            }

            // For mean timing
            alloc1_mean   = (T)t_alloc1  / (T)(curr_runs);
            cqrrpt_mean   = (T)t_cqrrpt  / (T)(curr_runs);
            appl1_mean    = (T)t_appl1   / (T)(curr_runs);
            alloc2_mean   = (T)t_alloc2  / (T)(curr_runs);
            geqp3_mean    = (T)t_geqp3   / (T)(curr_runs);
            appl2_mean    = (T)t_appl2   / (T)(curr_runs);
            alloc3_mean   = (T)t_alloc3  / (T)(curr_runs);
            geqr_mean     = (T)t_geqr    / (T)(curr_runs);
            appl3_mean    = (T)t_appl3   / (T)(curr_runs);
            alloc4_mean   = (T)t_alloc4  / (T)(curr_runs);
            tsqrp_mean    = (T)t_tsqrp   / (T)(curr_runs);
            appl4_mean    = (T)t_appl4   / (T)(curr_runs);
            alloc5_mean   = (T)t_alloc5  / (T)(curr_runs);
            geqrf_mean    = (T)t_geqrf   / (T)(curr_runs);
            appl5_mean    = (T)t_appl5   / (T)(curr_runs);
            alloc6_mean   = (T)t_alloc6  / (T)(curr_runs);
            scholqr_mean  = (T)t_scholqr / (T)(curr_runs);
            appl6_mean    = (T)t_appl6   / (T)(curr_runs);
            
            log_info(rows, cols, d_multiplier, k_multiplier, tol, nnz, num_threads, mat_type, 
                     cqrrpt_best,
                     geqp3_best,
                     geqr_best,
                     tsqrp_best,
                     geqrf_best,  
                     scholqr_best,
                     cqrrpt_best + alloc1_best + appl1_best, 
                     geqp3_best    + alloc2_best + appl2_best, 
                     geqr_best     + alloc3_best + appl3_best,
                     tsqrp_best    + alloc4_best + appl4_best, 
                     geqrf_best    + alloc5_best + appl5_best,
                     scholqr_best  + alloc6_best + appl6_best,
                     "Best", 
                     runs,
                     apply_to_large,
                     path);

            log_info(rows, cols, d_multiplier, k_multiplier, tol, nnz, num_threads, mat_type, 
                     cqrrpt_mean,
                     geqp3_mean,
                     geqr_mean,
                     tsqrp_mean,
                     geqrf_mean,
                     scholqr_mean,
                     cqrrpt_mean + alloc1_mean + appl1_mean, 
                     geqp3_mean    + alloc2_mean + appl2_mean, 
                     geqr_mean     + alloc3_mean + appl3_mean,
                     tsqrp_mean    + alloc4_mean + appl4_mean, 
                     geqrf_mean    + alloc5_mean + appl5_mean,
                     scholqr_mean  + alloc6_mean + appl6_mean,
                     "Mean", 
                     runs,
                     apply_to_large,
                     path);
            
            printf("Done with size %ld by %ld\n", rows, cols);
        }
    }
    printf("\n/-----------------------------------------QRCP SPEED BENCHMARK STOP-----------------------------------------/\n\n");
}

int main(){
    // Run with env OMP_NUM_THREADS=36 numactl --interleave all ./filename 
    auto state = RandBLAS::base::RNGState(0, 0);
    
    //test_speed<double>(17, 17, 512, 8192, 5, 1, 36, std::pow(std::numeric_limits<double>::epsilon(), 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), 1, "../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/apply_Q_to_large/", state);
    //test_speed<double>(17, 17, 512, 8192, 5, 1, 36, std::pow(std::numeric_limits<double>::epsilon(), 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), 0, "../../testing/RandLAPACK-Testing/test_benchmark/QR/speed/raw_data/", state);
    
    
        test_speed<double>(17, 17, 4096, 8192, 1, 1, 36, std::pow(std::numeric_limits<double>::epsilon(), 0.75), 1.0, 1.0, std::make_tuple(6, 0, false), 0, "../aaa/", state);
    return 0;
}
