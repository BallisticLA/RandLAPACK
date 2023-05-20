#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>


#define RELTOL_POWER 0.6
#define ABSTOL_POWER 0.75


template <typename T>
void check_condnum_after_precond(
    std::vector<T> &A,
    std::vector<T> &M_wk,
    int64_t rank,
    int64_t m,
    int64_t n,
    blas::Layout layout    
) {
    T *M = M_wk.data();
    std::vector<T> A_pc(m*n, 0.0);
    if (layout == blas::Layout::RowMajor) {
        // interpret as transpose in column-major
        // multiply by transpose(M) on the left.
        T *at = A.data();
        blas::gemm(
            blas::Layout::ColMajor,
            blas::Op::Trans,
            blas::Op::NoTrans,
            m, rank, n,
            1.0, at, n, M, n,
            0.0, A_pc.data(), m );
    } else {
        // A_pc = A @ M
        T *a = A.data();
        blas::gemm(
            blas::Layout::ColMajor,
            blas::Op::NoTrans,
            blas::Op::NoTrans,
            m, rank, n,
            1.0, a, m, M, n,
            0.0, A_pc.data(), m);
    }
    
    // check the result
    //T reltol = std::pow(std::numeric_limits<T>::epsilon(), RELTOL_POWER);
    // EXPECT_EQ(rank, n);
    T *ignore = nullptr;
    std::vector<T> s(rank, 0.0);
    lapack::gesvd(
        lapack::Job::NoVec,
        lapack::Job::NoVec,
        m, rank, A_pc.data(), m,
        s.data(), ignore, 1, ignore, 1
    );
    T cond = s[0] / s[rank-1];
    EXPECT_LE(cond, 10.0);
}


class Test_rpc_svd_saso : public ::testing::Test
{

    protected:
        static inline int64_t m = 5000;
        static inline int64_t n = 100;
        static inline int64_t d = 300;
        static inline std::vector<uint64_t> keys = {42, 1};
        static inline std::vector<uint64_t> vec_nnzs = {8, 10};
        static inline double sqrt_cond = 1e5;
        static inline double mu = 1e-6; // only used in "full_rank_after_reg" test.  
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    /*
     * Generate an ill-conditioned 5000-by-100 matrix "A," which we
     * interpret as being stored in either row-major or column-major
     * format.
     * 
     * Use an SJLT to sketch A down to 300-by-100; process the sketch to
     * obtain an SVD-based preconditioner "M" in column-major format.
     * 
     * Check that rank(M) == rank(A).
     * Compute A*M, check that its condition number is <= 10.
     * 
    */
   template <typename T>
    void test_full_rank_without_reg(
        int key_index,
        int nnz_index,
        blas::Layout layout
    ){  
        // construct "A" with cond(A) >= sqrt_cond^2.
        std::vector<T> A(m*n, 0.0);
        T *a = A.data();
        RandBLAS::dense::DenseDist D{
            .n_rows = m,
            .n_cols = n,
            .family = RandBLAS::dense::DenseDistName::Uniform
        };
        auto state = RandBLAS::base::RNGState(99);
        RandBLAS::dense::fill_buff(a, D, state); // dead-store the next state
    
        if (layout == blas::Layout::RowMajor) {
            // scale first row up by sqrt_cond
            // scale second row down by sqrt_cond
            blas::scal(n, sqrt_cond, a, 1);
            T invscale = 1.0 / sqrt_cond;
            blas::scal(n, invscale, &a[n], 1);
        } else {
            // scale first column up by sqrt_cond
            // scale second column down by sqrt_cond
            blas::scal(m, sqrt_cond, a, 1);
            T invscale = 1.0 / sqrt_cond;
            blas::scal(m, invscale, &a[m], 1);
        }

        // apply the function under test (rpc_svd_saso)
        auto alg_state = RandBLAS::base::RNGState((uint32_t) keys[key_index]);
        int64_t k = vec_nnzs[nnz_index];
        std::vector<T> M_wk(d*n, 0.0);
        int64_t rank;
        rank = RandLAPACK::rpc_svd_saso(
            layout, m, n, d, k,
            A, M_wk, (T) 0.0, alg_state
        );
        check_condnum_after_precond<T>(A, M_wk, rank, m, n, layout);
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
        // construct an ill-conditioned matrix, then zero out first column.
        std::vector<double> A(m*n, 0.0);
        double *a = A.data();
        RandBLAS::dense::DenseDist D{
            .family = RandBLAS::dense::DenseDistName::Uniform,
            .n_rows = m,
            .n_cols = n
        };
        auto state = RandBLAS::base::RNGState(99);
        RandBLAS::dense::fill_buff(a, D, state);  // dead-store
                      
        blas::scal(n, sqrt_cond, a, 1);
        double invscale = 1.0 / sqrt_cond;
        blas::scal(n, invscale, &a[n], 1);
        blas::scal(m, 0.0, a, n);

        // apply the function under test (rpc_svd_saso)
        std::vector<double> M_wk(d*n, 0.0);
        auto alg_state = RandBLAS::base::RNGState(keys[key_index]);
        int64_t k = vec_nnzs[nnz_index];
        int64_t rank = RandLAPACK::rpc_svd_saso(
            blas::Layout::RowMajor, m, n, d, k,
            A, M_wk, mu, alg_state    
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
        for (int i = 0; i < n; ++i) {
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

TEST_F(Test_rpc_svd_saso, FullRankNoReg_rowmajor_double)
{
    test_full_rank_without_reg<double>(0, 0, blas::Layout::RowMajor);
    test_full_rank_without_reg<double>(0, 1, blas::Layout::RowMajor);
    test_full_rank_without_reg<double>(1, 0, blas::Layout::RowMajor);
    test_full_rank_without_reg<double>(1, 1, blas::Layout::RowMajor);
}

TEST_F(Test_rpc_svd_saso, FullRankNoReg_colmajor_double)
{
    test_full_rank_without_reg<double>(0, 0, blas::Layout::ColMajor);
    test_full_rank_without_reg<double>(0, 1, blas::Layout::ColMajor);
    test_full_rank_without_reg<double>(1, 0, blas::Layout::ColMajor);
    test_full_rank_without_reg<double>(1, 1, blas::Layout::ColMajor);
}

TEST_F(Test_rpc_svd_saso, FullRankAfterReg)
{
    test_full_rank_after_reg(0, 0);
    test_full_rank_after_reg(0, 1);
    test_full_rank_after_reg(1, 0);
    test_full_rank_after_reg(1, 1);
}