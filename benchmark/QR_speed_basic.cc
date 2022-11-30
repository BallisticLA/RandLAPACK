/*
TODO #1: Switch tuples to vectors.
*/

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
using namespace std::chrono;

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::util;

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, int64_t n, uint32_t seed) {
    
    using namespace blas;
    using namespace lapack;

    int64_t size = m * n;
    std::vector<T> A_1(size, 0.0);
    std::vector<T> A_2(size, 0.0);
    std::vector<T> A_3(size, 0.0);
    std::vector<T> A_4(size, 0.0);
    
    std::vector<T> R_1_sp(n * n, 0.0);
    std::vector<T> D_1(n, 0.0);
    std::vector<T> T_1(n * n, 0.0);

    std::vector<int64_t> J_2(n, 0);
    std::vector<T> tau_2(n, 0);

    std::vector<T> R_3(n * n, 0);
    std::vector<T> t_3(5, 0);

    std::vector<T> tau_4(n, 0);

    // Random Gaussian test matrix
    RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_1.data(), seed);


    // Make copies
    std::copy(A_1.data(), A_1.data() + size, A_2.data());
    std::copy(A_1.data(), A_1.data() + size, A_3.data());
    std::copy(A_1.data(), A_1.data() + size, A_4.data());

    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // CholQR
    auto start_cholqr = high_resolution_clock::now();
    syrk(Layout::ColMajor, Uplo::Upper, Op::Trans, n, m, 1.0, A_1.data(), m, 0.0, R_1_sp.data(), n);
    potrf(Uplo::Upper, n, R_1_sp.data(), n);
    trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, m, n, 1.0, R_1_sp.data(), n, A_1.data(), m);
    auto stop_cholqr = high_resolution_clock::now();
    long dur_cholqr = duration_cast<microseconds>(stop_cholqr - start_cholqr).count();

    /*
    // Householder reflectors restoring
    auto start_rest = high_resolution_clock::now();
    orhr_col(m, n, n, A_1.data(), m, T_1.data(), n, D_1.data());
    auto stop_rest = high_resolution_clock::now();
    long dur_rest = duration_cast<microseconds>(stop_rest - start_rest).count();
    */

    // GEQP3
    auto start_geqp3 = high_resolution_clock::now();
    geqp3(m, n, A_2.data(), m, J_2.data(), tau_2.data());
    auto stop_geqp3 = high_resolution_clock::now();
    long dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();

    // GEQRF
    auto start_geqrf = high_resolution_clock::now();
    geqrf(m, n, A_4.data(), m, tau_4.data());
    auto stop_geqrf = high_resolution_clock::now();
    long dur_geqrf = duration_cast<microseconds>(stop_geqrf - start_geqrf).count();

    std::vector<long> res{dur_cholqr, dur_geqp3, dur_geqrf}; 
 
    return res;
}


template <typename T>
static void 
test_speed_mean(int r_pow, int r_pow_max, int col, int col_max, int runs)
{
    printf("\n/-----------------------------------------MEAN SPEED TEST START-----------------------------------------/\n");
    // Clear all files
    
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../testing/test_basic_qr_speed_mean_time_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.open("../../../testing/test_basic_qr_speed_mean_time_ratio_" + std::to_string(rows) + ".dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    int64_t rows = 0;
    int64_t cols = 0;

    T cholqr_avg = 0;
    T geqp3_avg    = 0;
    T geqrf_avg    = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols *= 2)
        {
            std::vector<long> res;
            long t_cholqr = 0;
            long t_geqp3    = 0;
            long t_geqrf    = 0;

            long curr_t_cholqr = 0;
            long curr_t_geqp3    = 0;
            long curr_t_geqrf    = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, i);

                //res{dur_alloc, dur_cholqrcp, dur_rest, dur_geqp3, dur_geqrf, dur_geqr, dur_tsqrp}; 

                curr_t_cholqr = res[0];
                curr_t_geqp3    = res[1];
                curr_t_geqrf    = res[2];

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    t_cholqr += curr_t_cholqr;
                    t_geqp3    += curr_t_geqp3;
                    t_geqrf    += curr_t_geqrf;
                }
            }

            cholqr_avg = (T)t_cholqr / (T)(runs - 1);
            geqp3_avg    = (T)t_geqp3    / (T)(runs - 1);
            geqrf_avg    = (T)t_geqrf    / (T)(runs - 1);

            // Save the output into .dat file
            std::fstream file("../../../testing/test_basic_qr_speed_mean_time_" + std::to_string(rows) + ".dat", std::fstream::app);
            file << cholqr_avg << "  " << geqp3_avg << "  " << geqrf_avg << "  " << geqr_avg << "\n";

            std::fstream file1("../../../testing/test_basic_qr_speed_mean_time_ratio_" + std::to_string(rows) + ".dat", std::fstream::app);
            file1 << cholqr_avg / geqrf_avg << "  " << geqp3_avg / geqrf_avg <<  "\n";

            printf("\n/-------------------------------------QRCP MEAN TIMING BEGIN-------------------------------------/\n");
            printf("\nMatrix size: %ld by %ld.\n", rows, cols);

            printf("Average timing of CholQR for %d runs: %54.2f μs.\n",                                runs - 1, cholqr_avg);
            printf("Average timing of GEQP3 for %d runs: %57.2f μs.\n",                                   runs - 1, geqp3_avg);
            printf("Average timing of GEQRF for %d runs: %57.2f μs.\n",                                   runs - 1, geqrf_avg);

            printf("Result: CholQR is %33.2f times faster than GEQP3.\n",                              geqp3_avg / cholqr_avg);
            printf("Result: CholQRCP is %33.2f times faster than GEQRF.\n",                              geqrf_avg / cholqr_avg);

            printf("\n/---------------------------------------QRCP MEAN TIMING END---------------------------------------/\n\n");
        }
    }
    printf("\n/-----------------------------------------MEAN SPEED TEST STOP-----------------------------------------/\n\n");
}

int main(int argc, char **argv){
        
    //test_speed_mean<double>(14, 17, 64, 1024, 64, 10);
    test_speed_mean<double>(14, 14, 256, 16384, 5);
    test_speed_mean<double>(16, 16, 512, 65536, 5);
    
    return 0;
}
