#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <chrono>
/*
Auxillary benchmark routine, computes flops using GEQRF for a given system
*/

using namespace std::chrono;
using namespace RandLAPACK;

template <typename T, typename RNG>
static void 
test_flops(int64_t k, 
        RandBLAS::RNGState<RNG> state) {
    int size = k * k;
    // Flops in gemm of given size
    int64_t flop_cnt = 2 * std::pow(k, 3) - (2/3) * std::pow(k, 3) + 3 * std::pow(k, 2) - std::pow(k, 2) + (14 / 3) * k;

    int runs = 50;
    T GFLOPS_rate_best = 0;

    T* A   = new T[size]();
    T* tau = new T[k]();

    RandLAPACK::gen::mat_gen_info<double> m_info(k, k, RandLAPACK::gen::gaussian);  

    for (int i = 0; i < runs; ++i) {
        RandLAPACK::gen::mat_gen(m_info, A, state);

        // Get the timing
        auto start = steady_clock::now();
        lapack::geqrf(k, k, A, k, tau);
        auto stop = steady_clock::now();
        long dur  = duration_cast<microseconds>(stop - start).count();
    
        T dur_s = dur / 1e+6;
        T GFLOPS_rate = (flop_cnt / dur_s) / 1e+9;

        if(GFLOPS_rate > GFLOPS_rate_best)
            GFLOPS_rate_best = GFLOPS_rate;
    }

    printf("THE SYSTEM IS CAPABLE OF %f GFLOPs/sec.\n\n", GFLOPS_rate_best);

    delete[] A;
    delete[] tau;
}

int main() {
    auto state = RandBLAS::RNGState();
    test_flops<double>(10000, state);
    return 0;
}