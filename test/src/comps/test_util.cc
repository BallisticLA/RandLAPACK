#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>

#include <chrono>
using namespace std::chrono;

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestUtil : public ::testing::Test
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
        long buf = 10000 * 10000;
        long flop_cnt = buf * (2 * 10000 - 1);

        using namespace blas;
        std::vector<T> A(size, 0.0);
        std::vector<T> B(size, 0.0);
        std::vector<T> C(size, 0.0);

        T* A_dat = A.data();
        T* B_dat = B.data();
        T* C_dat = C.data();

        RandBLAS::dense_op::gen_rmat_norm<T>(k, k, A_dat, seed);
        RandBLAS::dense_op::gen_rmat_norm<T>(k, k, B_dat, seed + 1);
        RandBLAS::dense_op::gen_rmat_norm<T>(k, k, C_dat, seed + 2);

        // Get the timing
        auto start = high_resolution_clock::now();
        gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, k, k, k, 1.0, A_dat, k, B_dat, k, 0.0, C_dat, k);
        auto stop = high_resolution_clock::now();
        long dur = duration_cast<microseconds>(stop - start).count();
    
        T dur_s = dur / 1e+6;
        printf("\nTHIS GEMM REQUIRES %ld flops.\n", flop_cnt);
        printf("COMPUTATION TIME: %f sec.\n", dur_s);
        printf("THE SYATEM IS CAPABLE OF %f GFLOPs/sec.\n\n", (flop_cnt / dur_s) / 1e+9);

        printf("|=====================================TEST SYSTEM FLOPS END====================================|\n");
    }
};
/*
// Check how many fps the machine is capable of
TEST_F(TestUtil, GemmFlop)
{
    test_flops<double>(10000, 0);
}
*/