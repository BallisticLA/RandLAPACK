#include <gtest/gtest.h>
#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <math.h>
#include <lapack.hh>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;


class Test_rpc_svd_sjlt : public ::testing::Test
{

    protected:
        uint64_t m = 5000;
        uint64_t n = 100;
        uint64_t d = 300;
        std::vector<uint64_t> keys = {42, 1};
        std::vector<uint64_t> vec_nnzs = {8, 10};
        double sqrt_cond = 1e5;
        double mu = 1e-6; // only used in "full_rank_after_reg" test.  
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    /*
     * Generate an ill-conditioned 5000-by-100 matrix "A," which we
     * interpret as being stored in row-major format.
     * 
     * Use an SJLT to sketch A down to 300-by-100; process the sketch to
     * obtain an SVD-based preconditioner "M" in column-major format.
     * 
     * Check that rank(M) == rank(A).
     * Compute A*M, check that its condition number is <= 10.
     * 
    */
    virtual void test_full_rank_without_reg(
        uint64_t key_index,
        uint64_t nnz_index
    ){
        uint64_t a_seed = 99;
        int64_t k = vec_nnzs[nnz_index];
        uint64_t seed_key = keys[key_index];
        uint64_t seed_ctr = 0;
        
        // construct "A" with cond(A) >= sqrt_cond^2.
        std::vector<double> A(m*n, 0.0);
        double *a = A.data();
        RandBLAS::dense_op::gen_rmat_unif(m, n, a, (uint32_t) a_seed);        
        blas::scal(n, sqrt_cond, a, 1);
        double invscale = 1.0 / sqrt_cond;
        blas::scal(n, invscale, &a[n], 1);

        // apply the function under test (rpc_svd_sjlt)
        std::vector<double> M_wk(d*n, 0.0);
        int64_t rank;
        rank = RandLAPACK::comps::preconditioners::rpc_svd_sjlt(m, n, d, k,
            A, M_wk, 0.0, 1, seed_key, seed_ctr    
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
        EXPECT_EQ(rank, n);
        double *ignore = nullptr;
        std::vector<double> s(n, 0.0);
        lapack::gesvd(
            lapack::Job::NoVec,
            lapack::Job::NoVec,
            m, n, A_pc.data(), m,
            s.data(), ignore, 1, ignore, 1
        );
        double cond = s[0] / s[n-1];
        EXPECT_LE(cond, 10);
    }

    /*
     * Generate a 5000-by-100 matrix "A" in row-major format.
     * Zero-out its first column, so it's rank-deficient.
     * 
     * Use an SJLT "S" to sketch A down to 300-by-100; process
     * the augmented sketch \hat{A}_sk = [S*A; sqrt(mu)*I] to
     * obtain an SVD-based preconditioner "M" in column-major format.
     * 
     * Check that rank(M) == n.
     * Check that cond([A; sqrt(mu)*I]*M) <= 10.
     * 
    */
    virtual void test_full_rank_after_reg(
        uint64_t key_index,
        uint64_t nnz_index
    ){
        uint64_t a_seed = 99;
        int64_t k = vec_nnzs[nnz_index];
        uint64_t seed_key = keys[key_index];
        uint64_t seed_ctr = 0;
        
        // construct an ill-conditioned matrix, then zero out first column.
        std::vector<double> A(m*n, 0.0);
        double *a = A.data();
        RandBLAS::dense_op::gen_rmat_unif(m, n, a, (uint32_t) a_seed);        
        blas::scal(n, sqrt_cond, a, 1);
        double invscale = 1.0 / sqrt_cond;
        blas::scal(n, invscale, &a[n], 1);
        blas::scal(m, 0.0, a, n);

        // apply the function under test (rpc_svd_sjlt)
        std::vector<double> M_wk(d*n, 0.0);
        int64_t rank;
        rank = RandLAPACK::comps::preconditioners::rpc_svd_sjlt(m, n, d, k,
            A, M_wk, mu, 1, seed_key, seed_ctr    
        );
        std::vector<double> A_aug_pc((m + n)*n, 0.0);
        double *a_aug_pc = A_aug_pc.data(); // interpret as column-major
        double *at = A.data(); // interpret as transpose in column-major
        double *M = M_wk.data();
        blas::gemm(
            blas::Layout::ColMajor,
            blas::Op::Trans,
            blas::Op::NoTrans,
            m, n, n,
            1.0, at, n, M, n,
            0.0, a_aug_pc, m + n 
        );
        double sqrt_mu = std::sqrt(mu);
        double *sqrt_mu_eye = &a_aug_pc[m];  // offset by m. 
        for (uint64_t i = 0; i < n; ++i) {
            sqrt_mu_eye[(m+n)*i + i] = sqrt_mu;
        }
        
        // check the result
        EXPECT_EQ(rank, n);
        double *ignore = nullptr;
        std::vector<double> s(n, 0.0);
        lapack::gesvd(
            lapack::Job::NoVec,
            lapack::Job::NoVec,
            m, n, a_aug_pc, m,
            s.data(), ignore, 1, ignore, 1
        );
        double cond = s[0] / s[n-1];
        EXPECT_LE(cond, 10);
    }
};

TEST_F(Test_rpc_svd_sjlt, FullRankNoReg)
{
    test_full_rank_without_reg(0, 0);
    test_full_rank_without_reg(0, 1);
    test_full_rank_without_reg(1, 0);
    test_full_rank_without_reg(1, 1);
}

TEST_F(Test_rpc_svd_sjlt, FullRankAfterReg)
{
    test_full_rank_after_reg(0, 0);
    test_full_rank_after_reg(0, 1);
    test_full_rank_after_reg(1, 0);
    test_full_rank_after_reg(1, 1);
}
