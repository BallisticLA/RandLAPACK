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
        RandBLAS::DenseDist D(m, n, RandBLAS::DenseDistName::Uniform);
        auto state = RandBLAS::RNGState(99);
        RandBLAS::fill_dense(D, a, state);
    
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
        auto alg_state = RandBLAS::RNGState((uint32_t) keys[key_index]);
        std::vector<T> M_wk(d*n, 0.0);
        std::vector<T> sigma_sk(n, 0.0);
        int64_t lda = (layout == blas::Layout::ColMajor) ? m : n;
        RandBLAS::SparseDist SDist{.n_rows=d, .n_cols=m, .vec_nnz=8};
        RandBLAS::SparseSkOp<T> S(SDist, alg_state);
        RandBLAS::fill_sparse(S);
        
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
        RandBLAS::DenseDist D(m, n, RandBLAS::DenseDistName::Uniform);
        auto state = RandBLAS::RNGState(99);
        RandBLAS::fill_dense(D, a, state);
                      
        blas::scal(n, sqrt_cond, a, 1);
        double invscale = 1.0 / sqrt_cond;
        blas::scal(n, invscale, &a[n], 1);
        blas::scal(m, 0.0, a, n);

        // apply the function under test (rpc_svd_saso)
        std::vector<double> M_wk(d*n, 0.0);
        std::vector<double> sigma_sk(n, 0.0);
        auto alg_state = RandBLAS::RNGState(keys[key_index]);
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

class TestNystromPrecond : public ::testing::Test
{

    protected:
        static inline int64_t m = 500;
        static inline std::vector<uint64_t> keys = {42, 1};
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    void set_invP(int64_t m, int64_t k, T* V, T* lambda, T mu, T* invP) {
        // compute invP = V * diag((min(lambda) + mu)/(lambda + mu)) * V' + (I - VV').
        RandLAPACK::util::eye(m, m, invP);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
            m, m, k, -1.0, V, m, V, m, 1.0, invP, m
        );
        for (int i = 0; i < k; ++i) {
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                m, m, 1, (lambda[k-1] + mu) / (lambda[i] + mu),
                &V[i*m], m, &V[i*m], m, 1.0, invP, m 
            );
        }
    };

    template <typename T>
    void set_G_mu_pre(int64_t m, T* G, T mu, T* invP, T* G_mu_pre) {
        // G_mu_pre = (G + mu)*invP
        blas::copy(m * m, invP, 1, G_mu_pre, 1);
        blas::scal(m * m, mu, G_mu_pre, 1);
        blas::symm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Lower,
            m, m, 1.0, G, m, invP, m, 1.0, G_mu_pre, m
        );
        for(int i = 1; i < m; ++i)
            blas::copy(m - i, &G_mu_pre[i + ((i-1) * m)], 1, &G_mu_pre[(i - 1) + (i * m)], m);
    };

    template <typename T>
    void run(int key_index, std::vector<T> &G) {
        /* Run the algorithm under test */
        RandBLAS::RNGState alg_state(keys[key_index]);
        alg_state.key.incr();
        std::vector<T> V(0);
        std::vector<T> lambda(0);
        int64_t k = 1;
        T mu_min = 1e-5;
        RandLAPACK::nystrom_pc_data(
            Uplo::Lower, G.data(), m, V, lambda, k, mu_min, alg_state
        ); // k has been updated.

        /* Verify algorithm output */
        EXPECT_TRUE(k > 5);
        EXPECT_TRUE(k < m);
        std::vector<T> invP(m * m, 0.0);
        std::vector<T> G_mu_pre(m * m, 0.0);

        T cond_lim = 5;
        T mu = mu_min;
        std::vector<T> s(m);
        set_invP(m, k, V.data(), lambda.data(), mu, invP.data());
        set_G_mu_pre(m, G.data(), mu, invP.data(), G_mu_pre.data());
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, m, m, G_mu_pre.data(), m, s.data(), nullptr, 1, nullptr, 1);
        T cond = s[0] / s[m-1];
        EXPECT_LE(cond, cond_lim);

        mu *= 10;
        cond_lim /= 2;
        set_invP(m, k, V.data(), lambda.data(), mu, invP.data());
        set_G_mu_pre(m, G.data(), mu, invP.data(), G_mu_pre.data());
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, m, m, G_mu_pre.data(), m, s.data(), nullptr, 1, nullptr, 1);
        cond = s[0] / s[m-1];
        EXPECT_LE(cond, cond_lim);
    
        mu *= 10;
        cond_lim /= 2;
        set_invP(m, k, V.data(), lambda.data(), mu, invP.data());
        set_G_mu_pre(m, G.data(), mu, invP.data(), G_mu_pre.data());
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, m, m, G_mu_pre.data(), m, s.data(), nullptr, 1, nullptr, 1);
        cond = s[0] / s[m-1];
        EXPECT_LE(cond, cond_lim);
    };
};

TEST_F(TestNystromPrecond, basictest) {
    RandLAPACK::gen::mat_gen_info<double> mat_info(m, m, RandLAPACK::gen::polynomial);
    mat_info.cond_num = 1e6;
    mat_info.rank = m;
    mat_info.exponent = 2.0;
    std::vector<double> A(m * m, 0.0);
    RandBLAS::RNGState data_state(0);
    RandLAPACK::gen::mat_gen<double>(mat_info, A.data(), data_state);
    std::vector<double> G(m * m, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Lower, Op::NoTrans, m, m, 1.0,
        A.data(), m, 0.0, G.data(), m
    ); // Note: G is PSD with squared spectrum of A.
    run<double>(0, G);
}
