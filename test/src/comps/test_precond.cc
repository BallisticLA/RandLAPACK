#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;


class Test_rpc_svd_sjlt : public ::testing::Test
{
    // only tests column-sparse SJLTs for now.
    protected:
        uint64_t m = 50000;
        uint64_t n = 1000;
        uint64_t d = 3000;
        std::vector<uint64_t> keys = {42, 0, 1};
        std::vector<uint64_t> vec_nnzs = {8, 10};     
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    virtual void apply(uint64_t key_index, uint64_t nnz_index, int threads)
    {
        uint64_t a_seed = 99;
        int64_t k = vec_nnzs[nnz_index];
        uint64_t seed_key = keys[key_index];
        uint64_t seed_ctr = 0;
        
        // construct test data: A
        std::vector<double> A(m*n, 0.0);
        double *a = A.data();
        RandBLAS::dense_op::gen_rmat_norm(m, n, a, (uint32_t) a_seed);        

        // compute expected result
        std::vector<double> M_wk(d*n, 0.0);
        int64_t rank;
        rank = RandLAPACK::comps::preconditioners::rpc_svd_sjlt(m, n, d, k,
            A, M_wk, 0.0, 2, seed_key, seed_ctr    
        );
        std::vector<double> A_pc(m*n, 0.0);
        double *at = A.data(); // interpret as transpose in column-major
        double *M = M_wk.data();
        blas::gemm(
            blas::Layout::ColMajor,
            blas::Op::Trans,
            blas::Op::NoTrans,
            m, n, n,
            1.0, at, n, M, n,
            0.0, A_pc.data(), m 
        );
        
        // check the result
        double reldtol = RELDTOL;
        EXPECT_EQ(rank, n);
    }
};

TEST_F(Test_rpc_svd_sjlt, SimpleTest)
{
    apply(0, 0, 2);
}