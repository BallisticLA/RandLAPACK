/*
TODO #1: Switch tuples to vectors.
*/

#include <blas.hh>
#include <RandBLAS.hh>
#include <lapack.hh>
#include <RandLAPACK.hh>
#include <math.h>

/*
Compares speed of various QR algorithms
*/

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
using std::string;

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, 
                  int64_t n, 
                  int64_t k, 
                  std::tuple<int, T, bool> mat_type, 
                  uint32_t seed) 
    {
    
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

    // Generate random matrix
    gen_mat_type<T>(m, n, A_1, k, seed, mat_type);


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
test_speed(int r_pow, 
                int r_pow_max, 
                int col, 
                int col_max, 
                T k_multiplier, 
                int runs, 
                std::tuple<int, T, bool> mat_type, 
                string test_type)
{
    printf("\n/-----------------------------------------QR SPEED TEST START-----------------------------------------/\n");
    
    // Clear all files
    for(int r_buf = r_pow; r_buf <= r_pow_max; ++r_buf)
    {
        int rows = std::pow(2, r_buf);
        std::ofstream ofs;
        ofs.open("../../../testing/test_benchmark/QR/speed/raw_data/QR_comp_time_" + test_type 
                                                                                   + "_m_"            + std::to_string(rows) 
                                                                                   + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                   + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                   + "_cond_"         + std::to_string(int(std::get<1>(mat_type)))
                                                                                   + "_runs_per_sz_"  + std::to_string(runs)
                                                                                   + ".dat", std::ofstream::out | std::ofstream::trunc);

        ofs.close();
        ofs.open("../../../testing/test_benchmark/QR/speed/raw_data/QR_comp_time_ratios_" + test_type 
                                                                                          + "_m_"            + std::to_string(rows) 
                                                                                          + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                          + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                          + "_cond_"         + std::to_string(int(std::get<1>(mat_type)))
                                                                                          + "_runs_per_sz_"  + std::to_string(runs)
                                                                                          + ".dat", std::ofstream::out | std::ofstream::trunc);

        ofs.close();
    }

    int64_t rows = 0;
    int64_t cols = 0;

    T cholqr_total = 0;
    T geqp3_total  = 0;
    T geqrf_total  = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols *= 2)
        {
            std::vector<long> res;
            long t_cholqr = 0;
            long t_geqp3  = 0;
            long t_geqrf  = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, k_multiplier * cols, mat_type, i);

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    if(!test_type.compare("Mean"))
                    {
                        t_cholqr += res[0];
                        t_geqp3  += res[1];
                        t_geqrf  += res[2];
                    }
                    else
                    {
                        if(cholqr_total > res[0] || cholqr_total == 0)
                        {
                            cholqr_total = res[0];
                        }
                        if(geqp3_total > res[1] || geqp3_total == 0)
                        {
                            geqp3_total = res[1];
                        }
                        if(geqrf_total > res[2] || geqrf_total == 0)
                        {
                            geqrf_total = res[2];
                        }
                    }
                }
            }

            if(!test_type.compare("Mean"))
            {
                cholqr_total = (T)t_cholqr     / (T)(runs - 1);
                geqp3_total    = (T)t_geqp3    / (T)(runs - 1);
                geqrf_total    = (T)t_geqrf    / (T)(runs - 1);
            }

            // Save the output into .dat file
            std::fstream file("../../../testing/test_benchmark/QR/speed/raw_data/QR_comp_time_" + test_type 
                                                                             + "_m_"            + std::to_string(rows) 
                                                                             + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                             + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                             + "_cond_"         + std::to_string(int(std::get<1>(mat_type)))
                                                                             + "_runs_per_sz_"  + std::to_string(runs)
                                                                             + ".dat", std::fstream::app);
            file << cholqr_total << "  " << geqp3_total << "  " << geqrf_total << "\n";

            std::fstream file1("../../../testing/test_benchmark/QR/speed/raw_data/QR_comp_time_ratios_" + test_type 
                                                                                                        + "_m_"            + std::to_string(rows) 
                                                                                                        + "_k_multiplier_" + std::to_string(k_multiplier)
                                                                                                        + "_mat_type_"     + std::to_string(std::get<0>(mat_type))
                                                                                                        + "_cond_"         + std::to_string(int(std::get<1>(mat_type)))
                                                                                                        + "_runs_per_sz_"  + std::to_string(runs)
                                                                                                        + ".dat", std::fstream::app);
            file1 << cholqr_total / geqrf_total << "  " << geqp3_total / geqrf_total <<  "\n";

            const char * test_type_print = test_type.c_str();

            printf("\n/---------------------------------------QR TIMING INFO BEGIN------------------------------------------/\n");
            printf("\nMatrix size: %ld by %ld.\n", rows, cols);

            printf("%s timing of CholQR for %d runs: %14.2f μs.\n", test_type_print, runs - 1, cholqr_total);
            printf("%s timing of GEQP3 for %d runs: %15.2f μs.\n",  test_type_print, runs - 1, geqp3_total);
            printf("%s timing of GEQRF for %d runs: %15.2f μs.\n",  test_type_print, runs - 1, geqrf_total);

            printf("\nResult: CholQR is %30.2f times faster than GEQP3.\n", geqp3_total / cholqr_total);
            printf("Result: CholQR is %28.2f times faster than GEQRF.\n", geqrf_total / cholqr_total);

            printf("\n/-----------------------------------------QR TIMING INFO END------------------------------------------/\n\n");
        }
    }
    printf("\n/-----------------------------------------QR SPEED TEST STOP------------------------------------------/\n\n");
}

int main(int argc, char **argv){
    test_speed<double>(14, 14, 64, 1024, 1, 5, std::make_tuple(6, 0, false), "Mean");
    test_speed<double>(14, 14, 64, 1024, 1, 5, std::make_tuple(6, 0, false), "Best");

    test_speed<double>(16, 16, 256, 4096, 1, 5, std::make_tuple(6, 0, false), "Mean");
    test_speed<double>(16, 16, 256, 4096, 1, 5, std::make_tuple(6, 0, false), "Best");

    test_speed<double>(17, 17, 512, 8192, 1, 5, std::make_tuple(6, 0, false), "Mean");
    test_speed<double>(17, 17, 512, 8192, 1, 5, std::make_tuple(6, 0, false), "Best");
    return 0;
}
