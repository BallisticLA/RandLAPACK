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
using namespace RandLAPACK::drivers::cholqrcp;

template <typename T>
static std::vector<long> 
test_speed_helper(int64_t m, int64_t n, uint32_t seed) {
    
    using namespace blas;
    using namespace lapack;

    int64_t size = m * n;
    std::vector<T> A_1(size, 0.0);
    std::vector<T> A_2(size, 0.0);
    std::vector<T> A_3(size, 0.0);
    
    std::vector<T> Q_1;
    std::vector<T> R_1;
    std::vector<int64_t> J_1(n, 0);

    std::vector<int64_t> J_2(n, 0.0);
    std::vector<T> tau_2(n, 0);

    std::vector<T> R_3(m * n, 0);
    std::vector<T> t_3(5, 0);
    std::vector<T> tau_3(n, 0);
    std::vector<int64_t> J_3(n, 0);

    // Random Gaussian test matrix
    RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A_1.data(), seed);


    // Make copies
    std::copy(A_1.data(), A_1.data() + size, A_2.data());
    std::copy(A_1.data(), A_1.data() + size, A_3.data());

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // CholQRCP constructor
    CholQRCP<T> CholQRCP(false, true, seed, 1.0e-16, use_cholqrcp1);
    CholQRCP.nnz = 4;
    CholQRCP.num_threads = 4;
    // Upsizing buffers
    upsize(m * n, Q_1);
    upsize((n + 1) * n, (CholQRCP.A_hat));
    upsize(n, (CholQRCP.A_hat));
    J_1.resize(n);
    upsize(n * n, (CholQRCP.R_sp));
    upsize(n * n, R_1);
    upsize(n * n, (CholQRCP.R_buf));
    
    // CholQRCP
    auto start_cholqrcp = high_resolution_clock::now();
    CholQRCP.call(m, n, A_1, n + 1, Q_1, R_1, J_1);
    auto stop_cholqrcp = high_resolution_clock::now();
    long dur_cholqrcp = duration_cast<microseconds>(stop_cholqrcp - start_cholqrcp).count();

    // GEQP3
    auto start_geqp3 = high_resolution_clock::now();
    geqp3(m, n, A_2.data(), m, J_2.data(), tau_2.data());
    auto stop_geqp3 = high_resolution_clock::now();
    long dur_geqp3 = duration_cast<microseconds>(stop_geqp3 - start_geqp3).count();

    // TSQR + GEQP3
    auto start_tsqrp = high_resolution_clock::now();
    // TSQR part
    geqr(m, n, A_3.data(), m, t_3.data(), -1);
    int64_t tsize = (int64_t) t_3[0]; 
    t_3.resize(tsize);
    geqr(m, n, A_3.data(), m, t_3.data(), tsize);

    // GEQP3 on R part
    get_U(m, n, A_3, R_3);
    geqp3(m, n, A_3.data(), m, J_3.data(), tau_3.data());

    // We then want to combine the representations of GEQP3 and GEQR

    auto stop_tsqrp = high_resolution_clock::now();
    long dur_tsqrp = duration_cast<microseconds>(stop_tsqrp - start_tsqrp).count();
    
    std::vector<long> res{dur_cholqrcp, dur_geqp3, dur_tsqrp}; 
 
    return res;
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

    T cholqrcp_avg = 0;
    T geqp3_avg    = 0;
    T tsqrp_avg    = 0;

    for(; r_pow <= r_pow_max; ++r_pow)
    {
        rows = std::pow(2, r_pow);
        int64_t cols = col;

        for (; cols <= col_max; cols += 64)
        {
            std::vector<long> res;
            long t_cholqrcp = 0;
            long t_geqp3    = 0;
            long t_tsqrp    = 0;

            long curr_t_cholqrcp = 0;
            long curr_t_geqp3    = 0;
            long curr_t_tsqrp    = 0;

            for(int i = 0; i < runs; ++i)
            {
                res = test_speed_helper<T>(rows, cols, i);
                curr_t_cholqrcp = res[0];
                curr_t_geqp3    = res[1];
                curr_t_tsqrp    = res[2];

                // Skip first iteration, as it tends to produce garbage results
                if (i != 0)
                {
                    t_cholqrcp += curr_t_cholqrcp;
                    t_geqp3    += curr_t_geqp3;
                    t_tsqrp    += curr_t_tsqrp;
                }
            }

            cholqrcp_avg = (T)t_cholqrcp / (T)(runs - 1);
            geqp3_avg    = (T)t_geqp3    / (T)(runs - 1);
            tsqrp_avg    = (T)t_tsqrp    / (T)(runs - 1);

            // Save the output into .dat file
            std::fstream file("../../../test_plots/test_speed_full_Q/raw_data/test_mean_time_" + std::to_string(rows) + ".dat", std::fstream::app);
            file << cholqrcp_avg << "  " << geqp3_avg << tsqrp_avg << "\n";

            printf("\nMatrix size: %ld by %ld.\n", rows, cols);
            printf("Average timing of CholQRCP for %d runs: %f μs.\n", runs - 1, cholqrcp_avg);
            printf("Average timing of GEQP3 for %d runs: %f μs.\n", runs - 1, geqp3_avg);
            printf("Average timing of TSQRP for %d runs: %f μs.\n", runs - 1, tsqrp_avg);
            printf("Result: CholQRCP is %f times faster than GEQP3.\n\n", geqp3_avg / cholqrcp_avg);
            printf("Result: CholQRCP is %f times faster than TSQRP.\n\n", tsqrp_avg / cholqrcp_avg);
        }
    }
}

int main(int argc, char **argv){
    test_speed_mean<double>(14, 16, 32, 128, 5);
    return 0;
}
