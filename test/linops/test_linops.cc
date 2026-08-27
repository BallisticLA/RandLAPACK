#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include <RandBLAS/testing/comparison.hh>


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
        RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, n, n, G_mu_pre_actual.data(), n,
            G_mu_pre_expect.data(), n, __RANDBLAS_PRETTY_FUNCTION__, 
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

// Issue #124: the generic materialize fallback forms buf = A * I by allocating
// an n by n identity matrix behind the caller's back. That hidden allocation is
// capped; the type-specific overloads (DenseLinOp, SparseLinOp, CompositeOperator)
// materialize without forming an identity and carry no cap.
namespace {
struct IdentityFallbackProbeOp {
    using scalar_t = double;
    void operator()(blas::Side, Layout, Op, Op,
                    int64_t, int64_t, int64_t, double,
                    const double*, int64_t, double,
                    double*, int64_t) {}
};
}

TEST(TestMaterializeGuard, generic_fallback_refuses_huge_identity) {
    IdentityFallbackProbeOp A;
    int64_t n_over = RandLAPACK::MATERIALIZE_IDENTITY_MAX_DIM + 1;
    double stub = 0.0;
    // The guard must fire before the output buffer is touched, so a one-element
    // stub standing in for the (never-written) output is safe here.
    EXPECT_THROW(RandLAPACK::materialize(A, n_over, n_over, &stub, n_over), RandLAPACK::Error);

    // Below the cap the generic path still works.
    int64_t n_small = 4;
    std::vector<double> buf(n_small * n_small, 1.0);
    RandLAPACK::materialize(A, n_small, n_small, buf.data(), n_small);
    // The probe operator writes nothing, so all that must have happened is the
    // zeroing of the output buffer.
    for (auto v : buf)
        ASSERT_EQ(v, 0.0);
}

// The CompositeOperator overload materializes each operand and multiplies the
// dense factors with one gemm; it never builds an identity and never calls the
// composite's operator(), so it is exempt from the fallback's dimension cap
// and is safe to use when testing operator() itself.
TEST(TestMaterializeComposite, matches_reference_product) {
    int64_t m = 7;
    int64_t k = 4;
    int64_t n = 5;
    int64_t ldb = m + 2;

    // Deterministic operand fills.
    vector<double> L_buf(m * k), R_buf(k * n);
    for (int64_t i = 0; i < m * k; ++i)
        L_buf[i] = 0.5 + (double)((3 * i) % 11);
    for (int64_t i = 0; i < k * n; ++i)
        R_buf[i] = -1.0 + (double)((5 * i) % 7);

    RandLAPACK::linops::DenseLinOp<double> L_op(m, k, L_buf.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> R_op(k, n, R_buf.data(), k, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

    // Sentinel-fill past-the-column entries to check ldb handling.
    vector<double> buf(ldb * n, -42.0);
    RandLAPACK::materialize(comp, m, n, buf.data(), ldb);

    vector<double> ref(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
               1.0, L_buf.data(), m, R_buf.data(), k, 0.0, ref.data(), m);

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i)
            ASSERT_DOUBLE_EQ(buf[i + j * ldb], ref[i + j * m]);
        for (int64_t i = m; i < ldb; ++i)
            ASSERT_EQ(buf[i + j * ldb], -42.0) << "materialize wrote past column height";
    }
}

TEST(TestMaterializeComposite, wide_composite_is_exempt_from_identity_cap) {
    // Wider than the generic fallback allows, but with a small inner dimension,
    // so the operand-product path materializes it in m*k + k*n scratch where
    // the fallback would have refused to build the n*n identity.
    int64_t m = 3;
    int64_t k = 2;
    int64_t n = RandLAPACK::MATERIALIZE_IDENTITY_MAX_DIM + 1;

    vector<double> L_buf(m * k, 1.0), R_buf(k * n);
    for (int64_t i = 0; i < k * n; ++i)
        R_buf[i] = (double)(i % 5);

    RandLAPACK::linops::DenseLinOp<double> L_op(m, k, L_buf.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> R_op(k, n, R_buf.data(), k, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

    vector<double> buf(m * n, 0.0);
    RandLAPACK::materialize(comp, m, n, buf.data(), m);

    // With every row of L equal to ones, each column j of the product holds
    // the column sum of R in every entry.
    for (int64_t j = 0; j < n; ++j) {
        double col_sum = R_buf[j * k] + R_buf[j * k + 1];
        for (int64_t i = 0; i < m; ++i)
            ASSERT_DOUBLE_EQ(buf[i + j * m], col_sum);
    }
}
