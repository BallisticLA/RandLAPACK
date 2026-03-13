#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"


using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::DenseDist;
using RandBLAS::SparseDist;
using RandBLAS::RNGState;


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
        invP_operator(blas::Layout::ColMajor, n, (T) 1.0,  G_mu.data(), n, (T)0.0, G_mu_pre_actual.data(), n);
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
