#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>

#include <chrono>
using namespace std::chrono;

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class BenchmarkUtil : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void 
    test_flops(int k, uint32_t seed)
    {
        printf("|===================================TEST SYSTEM FLOPS BEGIN====================================|\n");
        int size = k * k;

        // Flops in gemm of given size - overflows
        long buf = k * k;
        long flop_cnt = buf * (2 * k - 1);

        int runs = 10;
        T DUR_sum = 0;
        T GFLOPS_sum = 0;

        for (int i = 0; i < runs; ++i)
        {
            using namespace blas;
            std::vector<T> A(size, 0.0);
            std::vector<T> B(size, 0.0);
            std::vector<T> C(size, 0.0);

            T* A_dat = A.data();
            T* B_dat = B.data();
            T* C_dat = C.data();

            RandBLAS::dense_op::gen_rmat_norm<T>(k, k, A_dat, ++seed);
            RandBLAS::dense_op::gen_rmat_norm<T>(k, k, B_dat, ++seed);

            // Get the timing
            auto start = high_resolution_clock::now();
            gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, k, k, 1.0, A_dat, k, B_dat, k, 0.0, C_dat, k);
            auto stop = high_resolution_clock::now();
            long dur = duration_cast<microseconds>(stop - start).count();
        
            T dur_s = dur / 1e+6;
            T GFLOPS = (flop_cnt / dur_s) / 1e+9;

            DUR_sum += dur_s;
            GFLOPS_sum += GFLOPS;
        }

        printf("\nTHIS GEMM REQUIRES %ld flops.\n", flop_cnt);
        printf("COMPUTATION TIME: %f sec.\n", DUR_sum / runs);
        printf("THE SYATEM IS CAPABLE OF %f GFLOPs/sec.\n\n", GFLOPS_sum / runs);

        printf("|=====================================TEST SYSTEM FLOPS END====================================|\n");
    }
};
/*
// Check how many fps the machine is capable of
TEST_F(BenchmarkUtil, GemmFlop)
{
    //__builtin___clear_cache();   
    test_flops<double>(10000, 0);
}
*/