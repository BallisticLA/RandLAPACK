#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "/Users/rjmurr/Documents/randnla/RandLAPACK/RandBLAS/test/comparison.hh"


using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::DenseDist;
using RandBLAS::SparseDist;
using RandBLAS::RNGState;


template <typename T>
void check_condnum_after_precond(
    Layout layout,
    vector<T> &A,
    vector<T> &M_wk,
    int64_t rank,
    int64_t m,
    int64_t n
) {
    vector<T> A_pc(m * rank, 0.0);
    bool is_colmajor = layout == Layout::ColMajor;
    int64_t lda = (is_colmajor) ? m : n;
    int64_t ldm = (is_colmajor) ? n : rank;
    int64_t ldapc = (is_colmajor) ? m : rank;
    blas::gemm(
        layout,
        Op::NoTrans,
        Op::NoTrans,
        m, rank, n,
        1.0, A.data(), lda, M_wk.data(), ldm,
        0.0, A_pc.data(), ldapc
    );
    
    vector<T> s(rank, 0.0);
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
        static inline vector<uint64_t> keys = {42, 1};
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
        Layout layout
    ){  
        // construct "A" with cond(A) >= sqrt_cond^2.
        vector<T> A(m*n, 0.0);
        T *a = A.data();
        DenseDist D(m, n, RandBLAS::DenseDistName::Uniform);
        auto state = RNGState(99);
        RandBLAS::fill_dense(D, a, state);
    
        if (layout == Layout::RowMajor) {
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
        auto alg_state = RNGState((uint32_t) keys[key_index]);
        vector<T> M_wk(d*n, 0.0);
        vector<T> sigma_sk(n, 0.0);
        int64_t lda = (layout == Layout::ColMajor) ? m : n;
        SparseDist SDist{.n_rows=d, .n_cols=m, .vec_nnz=8};
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
        vector<double> A(m*n, 0.0);
        double *a = A.data();
        DenseDist D(m, n, RandBLAS::DenseDistName::Uniform);
        auto state = RNGState(99);
        RandBLAS::fill_dense(D, a, state);
                      
        blas::scal(n, sqrt_cond, a, 1);
        double invscale = 1.0 / sqrt_cond;
        blas::scal(n, invscale, &a[n], 1);
        blas::scal(m, 0.0, a, n);

        // apply the function under test (rpc_svd_saso)
        vector<double> M_wk(d*n, 0.0);
        vector<double> sigma_sk(n, 0.0);
        auto alg_state = RNGState(keys[key_index]);
        RandLAPACK::rpc_data_svd_saso(
            Layout::RowMajor, m, n, d, 8,
            A.data(), n, M_wk.data(), sigma_sk.data(), alg_state
        );
        int64_t rank = RandLAPACK::make_right_orthogonalizer(
            Layout::RowMajor,
            n, M_wk.data(), sigma_sk.data(), mu
        );
        EXPECT_EQ(rank, n);
        
        vector<double> A_aug((m + n)*n, 0.0);
        double *a_aug = A_aug.data();
        blas::copy(m*n, a, 1, a_aug, 1);
        double sqrt_mu = std::sqrt(mu);
        double *sqrt_mu_eye = &a_aug[m * n];
        for (int i = 0; i < n; ++i)
            sqrt_mu_eye[n*i + i] = sqrt_mu;

        check_condnum_after_precond(Layout::RowMajor, A_aug, M_wk, rank, m + n, n);
    }
};

TEST_F(Test_rpc_svd, FullRankNoReg_rowmajor_double)
{
    test_full_rank_without_reg<double>(0, Layout::RowMajor);
    test_full_rank_without_reg<double>(1, Layout::RowMajor);
}

TEST_F(Test_rpc_svd, FullRankNoReg_colmajor_double)
{
    test_full_rank_without_reg<double>(0, Layout::ColMajor);
    test_full_rank_without_reg<double>(1, Layout::ColMajor);
}

TEST_F(Test_rpc_svd, FullRankAfterReg)
{
    test_full_rank_after_reg(0);
    test_full_rank_after_reg(1);
}


template <typename T>
vector<T> polynomial_decay_psd(int64_t m, T cond_num, T exponent, uint32_t seed) {
    RandLAPACK::gen::mat_gen_info<T> mat_info(m, m, RandLAPACK::gen::polynomial);
    mat_info.cond_num = std::sqrt(cond_num);
    mat_info.rank = m;
    mat_info.exponent = exponent;
    vector<T> A(m * m, 0.0);
    RNGState data_state(seed);
    RandLAPACK::gen::mat_gen(mat_info, A.data(), data_state);
    vector<T> G(m * m, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::NoTrans, m, m, 1.0,
        A.data(), m, 0.0, G.data(), m
    ); // Note: G is PSD with squared spectrum of A.
    RandBLAS::util::symmetrize(Layout::ColMajor, Uplo::Upper, G.data(), m, m);
    return G;
}

/**
 * Note: a few implicit linear operators are tested implicitly (ha) in
 * test_determiter.cc. It's important to have tests for these things
 * since bugs in their implementation can be hard to track down. 
 */


class TestSpectralPrecondLinearOperator: public ::testing::Test {

    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    // Run on a diagonal matrix with an optimal rank-k preconditioner.
    template <typename T>
    void run_diag(int64_t n, int64_t k, T mu) {
        int64_t i;
        vector<T> alleigs(n);
        vector<T> allV(n*n, 0.0);
        for (i = 0; i < n; ++i) {
            alleigs[i] = std::pow((T)i + (T)1.0, (T) -3.0);
            allV[i + i*n] = 1.0;
        }

        vector<T> G_mu(n*n, 0.0);
        for (i = 0; i < n; ++i) {
            G_mu[i + i*n] = alleigs[i] + mu;
        }

        vector<T> pceigs(k);
        vector<T> pcV(n*k, 0.0);
        for (i = 0; i < k; ++i) {
            pceigs[i] = alleigs[i];
            pcV[i + i*n] = 1.0;
        }
        vector<T> G_mu_pre_expect(n*n, 0.0);
        T scale_on_precond_subspace = alleigs[k-1] + mu;
        for (i = 0; i < n; ++i) {
            if (i < k) {
                G_mu_pre_expect[i + i*n] = scale_on_precond_subspace;
            } else {
                G_mu_pre_expect[i + i*n] = alleigs[i] + mu;
            }
        }
        RandLAPACK::linops::SpectralPrecond<T> invP_operator(n);
        vector<T> mus(1, mu);
        invP_operator.prep(pcV, pceigs, mus, n);
        vector<T> G_mu_pre_actual(n*n, 0.0);
        invP_operator.evaluate(n, G_mu.data(), G_mu_pre_actual.data());
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, n, n, G_mu_pre_actual.data(), n,
            G_mu_pre_expect.data(), n, __PRETTY_FUNCTION__, 
            __FILE__, __LINE__
        );
        return;
    }
};

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n3_k1) {
    run_diag<float>(3, 1, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n3_k2) {
    run_diag<float>(3, 2, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n4_k1) {
    run_diag<float>(4, 1, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n4_k2) {
    run_diag<float>(4, 2, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n4_k3) {
    run_diag<float>(4, 3, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k1) {
    run_diag<float>(5, 1, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k2) {
    run_diag<float>(5, 2, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k3) {
    run_diag<float>(5, 3, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k4) {
    run_diag<float>(5, 4, 0.1);
}


/***
 * This actually assesses quality of the Nystrom preconditioner.
 */
class TestNystromPrecond : public ::testing::Test {

    protected:
        static inline int64_t m = 500;
        static inline vector<uint32_t> keys = {42, 1};
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    void run(int key_index, vector<T> &G) {
        /* Run the algorithm under test */
        RNGState alg_state(keys[key_index]);
        alg_state.key.incr();
        vector<T> V(0);
        vector<T> lambda(0);
        int64_t k = 1;
        T mu_min = 1e-5;
        RandLAPACK::nystrom_pc_data(
            Uplo::Lower, G.data(), m, V, lambda, k, mu_min, alg_state
        ); // k has been updated.

        /* Verify algorithm output */
        EXPECT_TRUE(k > 5);
        EXPECT_TRUE(k < m);
        RandLAPACK::linops::SpectralPrecond<T> invP(m);
        vector<T> G_mu_pre(m * m, 0.0);
        vector<T> G_mu(m * m);
        vector<T> mus(1);
        vector<T> s(m);

        mus[0] = mu_min;
        G_mu = G;
        for (int64_t i = 0; i < m; ++i)
            G_mu[i + i*m] += mus[0];
        invP.prep(V, lambda, mus, m);
        invP.evaluate(m, G_mu.data(), G_mu_pre.data());
        T cond_lim = 5;
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, m, m, G_mu_pre.data(), m, s.data(), nullptr, 1, nullptr, 1);
        T cond = s[0] / s[m-1];
        EXPECT_LE(cond, cond_lim);

        mus[0] *= 10;
        G_mu = G;
        for (int64_t i = 0; i < m; ++i)
            G_mu[i + i*m] += mus[0];
        invP.prep(V, lambda, mus, m);
        invP.evaluate(m, G_mu.data(), G_mu_pre.data());
        cond_lim /= 2;
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, m, m, G_mu_pre.data(), m, s.data(), nullptr, 1, nullptr, 1);
        cond = s[0] / s[m-1];
        EXPECT_LE(cond, cond_lim);
    
        mus[0] *= 10;
        G_mu = G;
        for (int64_t i = 0; i < m; ++i)
            G_mu[i + i*m] += mus[0];
        invP.prep(V, lambda, mus, m);
        invP.evaluate(m, G_mu.data(), G_mu_pre.data());
        cond_lim /= 2;
        lapack::gesvd(lapack::Job::NoVec, lapack::Job::NoVec, m, m, G_mu_pre.data(), m, s.data(), nullptr, 1, nullptr, 1);
        cond = s[0] / s[m-1];
        EXPECT_LE(cond, cond_lim);
    }
};


TEST_F(TestNystromPrecond, basictest) {
    auto G = polynomial_decay_psd<double>(m, 1e12, 2.0, 99);
    run(0, G);
    run(1, G);
}

