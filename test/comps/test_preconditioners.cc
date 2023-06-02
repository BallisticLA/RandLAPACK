#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>


template <typename T>
void check_condnum_after_precond(
    blas::Layout layout,
    std::vector<T> &A,
    std::vector<T> &M_wk,
    int64_t rank,
    int64_t m,
    int64_t n
) {
    std::vector<T> A_pc(m * rank, 0.0);
    bool is_colmajor = layout == blas::Layout::ColMajor;
    int64_t lda = (is_colmajor) ? m : n;
    int64_t ldm = (is_colmajor) ? n : rank;
    int64_t ldapc = (is_colmajor) ? m : rank;
    blas::gemm(
        layout,
        blas::Op::NoTrans,
        blas::Op::NoTrans,
        m, rank, n,
        1.0, A.data(), lda, M_wk.data(), ldm,
        0.0, A_pc.data(), ldapc
    );
    
    std::vector<T> s(rank, 0.0);
    if (is_colmajor) {
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
            m, rank, A_pc.data(), ldapc, s.data(), nullptr, 1, nullptr, 1
        );
    } else {
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec,
            rank, m, A_pc.data(), ldapc, s.data(), nullptr, 1, nullptr, 1
        );
    }
    T cond = s[0] / s[rank-1];
    EXPECT_LE(cond, 10.0);
}


class Test_rpc_svd : public ::testing::Test
{

    protected:
        static inline int64_t m = 500;
        static inline int64_t n = 10;
        static inline int64_t d = 30;
        static inline std::vector<uint64_t> keys = {42, 1};
        static inline double sqrt_cond = 1e5;
        static inline double mu = 1e-6; // only used in "full_rank_after_reg" test.  
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    /*
     * Generate an ill-conditioned 500-by-10 matrix "A."
     * 
     * Use an SJLT to sketch A down to 30-by-10; process the sketch to
     * obtain an SVD-based preconditioner "M."
     * 
     * Check that rank(M) == rank(A) == 10.
     * Compute A*M, check that its condition number is <= 10.
     * (This latter "10" has nothing to do with the former "10.")
    */
    template <typename T>
    void test_full_rank_without_reg(
        int key_index,
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
        RandBLAS::dense::fill_buff(a, D, state);
    
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

        // apply the function under test (rpc_data_svd_saso)
        auto alg_state = RandBLAS::base::RNGState((uint32_t) keys[key_index]);
        std::vector<T> M_wk(d*n, 0.0);
        std::vector<T> sigma_sk(n, 0.0);
        int64_t lda = (layout == blas::Layout::ColMajor) ? m : n;
        RandBLAS::sparse::SparseDist SDist{.n_rows=d, .n_cols=m, .vec_nnz=8};
        RandBLAS::sparse::SparseSkOp<T> S(SDist, alg_state);
        RandBLAS::sparse::fill_sparse(S);
        
        RandLAPACK::rpc_data_svd(
            layout, m, n, A.data(), lda, S, M_wk.data(), sigma_sk.data()
        );
        int64_t rank = RandLAPACK::make_right_orthogonalizer(
            layout, n, M_wk.data(), sigma_sk.data(), 0.0
        );

        // Check for correct output
        check_condnum_after_precond<T>(layout, A, M_wk, rank, m, n);
    }

    /*
     * Generate a 500-by-10 matrix "A" in row-major format.
     * Zero-out its first column, so it's rank-deficient.
     * 
     * Use an SJLT "S" to sketch A down to 30-by-10; process
     * the augmented sketch \hat{A}_sk = [S*A; sqrt(mu)*I] to
     * obtain an SVD-based preconditioner "M" in row-major format.
     * 
     * Check that rank(M) == n.
     * Check that cond([A; sqrt(mu)*I]*M) <= 10.
     * 
    */
    virtual void test_full_rank_after_reg(
        uint64_t key_index
    ){    
        // construct an ill-conditioned matrix, then zero out first column.
        // After regularization the augmented matrix will still be full-rank.
        std::vector<double> A(m*n, 0.0);
        double *a = A.data();
        RandBLAS::dense::DenseDist D{
            .n_rows = m,
            .n_cols = n,
            .family = RandBLAS::dense::DenseDistName::Uniform
        };
        auto state = RandBLAS::base::RNGState(99);
        RandBLAS::dense::fill_buff(a, D, state);
                      
        blas::scal(n, sqrt_cond, a, 1);
        double invscale = 1.0 / sqrt_cond;
        blas::scal(n, invscale, &a[n], 1);
        blas::scal(m, 0.0, a, n);

        // apply the function under test (rpc_svd_saso)
        std::vector<double> M_wk(d*n, 0.0);
        std::vector<double> sigma_sk(n, 0.0);
        auto alg_state = RandBLAS::base::RNGState(keys[key_index]);
        RandLAPACK::rpc_data_svd_saso(
            blas::Layout::RowMajor, m, n, d, 8,
            A.data(), n, M_wk.data(), sigma_sk.data(), alg_state
        );
        int64_t rank = RandLAPACK::make_right_orthogonalizer(
            blas::Layout::RowMajor,
            n, M_wk.data(), sigma_sk.data(), mu
        );
        EXPECT_EQ(rank, n);
        
        std::vector<double> A_aug((m + n)*n, 0.0);
        double *a_aug = A_aug.data();
        blas::copy(m*n, a, 1, a_aug, 1);
        double sqrt_mu = std::sqrt(mu);
        double *sqrt_mu_eye = &a_aug[m * n];
        for (int i = 0; i < n; ++i)
            sqrt_mu_eye[n*i + i] = sqrt_mu;

        check_condnum_after_precond(blas::Layout::RowMajor, A_aug, M_wk, rank, m + n, n);
    }
};

TEST_F(Test_rpc_svd, FullRankNoReg_rowmajor_double)
{
    test_full_rank_without_reg<double>(0, blas::Layout::RowMajor);
    test_full_rank_without_reg<double>(1, blas::Layout::RowMajor);
}

TEST_F(Test_rpc_svd, FullRankNoReg_colmajor_double)
{
    test_full_rank_without_reg<double>(0, blas::Layout::ColMajor);
    test_full_rank_without_reg<double>(1, blas::Layout::ColMajor);
}

TEST_F(Test_rpc_svd, FullRankAfterReg)
{
    test_full_rank_after_reg(0);
    test_full_rank_after_reg(1);
}