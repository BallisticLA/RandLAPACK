#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <chrono>
/*
Auxillary benchmark routine, computes flops using GEMM for a given system
*/

using namespace std::chrono;
using namespace RandLAPACK;

template <typename T, typename RNG>
static void 
test_flops(int64_t k, 
        RandBLAS::RNGState<RNG> state) {
    int size = k * k;
    // Flops in gemm of given size
    int64_t flop_cnt = 2 * std::pow(k, 3);

    int runs = 50;
    T GFLOPS_rate_best = 0;

    T* A = ( T * ) calloc( size, sizeof( T ) );
    T* B = ( T * ) calloc( size, sizeof( T ) );
    T* C = ( T * ) calloc( size, sizeof( T ) );

    RandLAPACK::gen::mat_gen_info<double> m_info(k, k, RandLAPACK::gen::gaussian);  

    for (int i = 0; i < runs; ++i) {
        RandLAPACK::gen::mat_gen(m_info, A, state);
        RandLAPACK::gen::mat_gen(m_info, B, state);

        // Get the timing
        auto start = steady_clock::now();
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, k, k, 1.0, A, k, B, k, 0.0, C, k);
        auto stop = steady_clock::now();
        long dur = duration_cast<microseconds>(stop - start).count();
    
        T dur_s = dur / 1e+6;
        T GFLOPS_rate = (flop_cnt / dur_s) / 1e+9;

        if(GFLOPS_rate > GFLOPS_rate_best)
            GFLOPS_rate_best = GFLOPS_rate;
    }

    printf("THE SYSTEM IS CAPABLE OF %f GFLOPs/sec.\n\n", GFLOPS_rate_best);
}

int main() {
    auto state = RandBLAS::RNGState();
    test_flops<double>(10000, state);
    return 0;
}
