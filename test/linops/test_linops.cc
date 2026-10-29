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
using blas::Side;
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
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, n, n, G_mu_pre_actual.data(), n,
            G_mu_pre_expect.data(), n, __RANDBLAS_PRETTY_FUNCTION__, 
            __FILE__, __LINE__
        ), "");
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

// The generic materialize fallback allocates an n by n identity behind the
// caller's back; that hidden allocation is capped.
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
    // The guard fires before the output buffer is touched, so a stub is safe.
    EXPECT_THROW(RandLAPACK::materialize(A, n_over, n_over, &stub, n_over), RandLAPACK::Error);

    // Below the cap the generic path still works.
    int64_t n_small = 4;
    std::vector<double> buf(n_small * n_small, 1.0);
    RandLAPACK::materialize(A, n_small, n_small, buf.data(), n_small);
    // The probe writes nothing, so only the zeroing of the buffer happened.
    for (auto v : buf)
        ASSERT_EQ(v, 0.0);
}

// The CompositeOperator overload multiplies materialized operands with one
// gemm: no identity, no cap, no call through the composite's operator().
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
    // Wider than the generic fallback allows; the operand-product path handles
    // it in m*k + k*n scratch.
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

// At beta = 0 the composite's C is output only, as in BLAS: a NaN-filled C must give the same result as
// a zero-filled one, for each side and transposition, and for a sketch applied blocked and unblocked.
TEST(TestCompositeOperator, beta_zero_does_not_read_C) {
    int64_t M = 6, K = 4, N = 5, p = 3;
    vector<double> L_buf(M * K), R_buf(K * N), B(std::max(M, N) * p);
    for (int64_t i = 0; i < M * K; ++i)
        L_buf[i] = 0.5 + (double)((3 * i) % 11);
    for (int64_t i = 0; i < K * N; ++i)
        R_buf[i] = -1.0 + (double)((5 * i) % 7);
    for (std::size_t i = 0; i < B.size(); ++i)
        B[i] = 1.0 + (double)((2 * i) % 5);
    RandLAPACK::linops::DenseLinOp<double> L_op(M, K, L_buf.data(), M, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> R_op(K, N, R_buf.data(), K, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator comp(M, N, L_op, R_op);
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // Runs apply into a zero-filled and a NaN-filled rows-by-cols C and compares them.
    auto check = [&](const char* name, int64_t rows, int64_t cols, auto apply) {
        SCOPED_TRACE(name);
        vector<double> C0(rows * cols, 0.0), C1(rows * cols, nan);
        apply(C0.data());
        apply(C1.data());
        for (int64_t i = 0; i < rows * cols; ++i) {
            ASSERT_TRUE(std::isfinite(C0[i])) << "entry " << i;
            ASSERT_EQ(C1[i], C0[i]) << "entry " << i;
        }
    };
    check("left", M, p, [&](double* C) {
        comp(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, M, p, N, 1.0, B.data(), N, 0.0, C, M); });
    check("left, transposed", N, p, [&](double* C) {
        comp(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, N, p, M, 1.0, B.data(), M, 0.0, C, N); });
    check("right", p, N, [&](double* C) {
        comp(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, p, N, M, 1.0, B.data(), p, 0.0, C, p); });
    for (int64_t block_size : {(int64_t) 0, (int64_t) 2}) {
        comp.block_size = block_size;
        check(block_size ? "sketch, blocked" : "sketch", p, N, [&](double* C) {
            RandBLAS::DenseSkOp<double> S(DenseDist(p, M), RNGState(3));
            comp(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, p, N, M, 1.0, S, 0.0, C, p); });
    }
}

// A sparse B whose dimensions differ from those (m, n, k, trans_B) give is rejected before anything is
// written: densifying it into a buffer sized for op(B) would write past the buffer.
TEST(TestSparseLinOp, rejects_a_sparse_B_of_the_wrong_size) {
    double identity[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    RandBLAS::sparse_data::CSCMatrix<double> A_sp(3, 3), B_sp(4, 3);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, identity, 0.0, A_sp);
    double B_dense[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, B_dense, 0.0, B_sp);
    RandLAPACK::linops::SparseLinOp<decltype(A_sp)> A_op(3, 3, A_sp);

    vector<double> C(9, -1.0);
    for (Op trans_B : {Op::NoTrans, Op::Trans}) {
        EXPECT_THROW(A_op(Side::Left, Layout::ColMajor, Op::NoTrans, trans_B, 3, 3, 3, 1.0, B_sp, 0.0, C.data(), 3),
                     RandLAPACK::Error);
        EXPECT_THROW(A_op(Side::Right, Layout::ColMajor, Op::NoTrans, trans_B, 3, 3, 3, 1.0, B_sp, 0.0, C.data(), 3),
                     RandLAPACK::Error);
    }
    EXPECT_EQ(C, vector<double>(9, -1.0));
}

// With a sparse B, a request for a leading submatrix of A must multiply only that submatrix, on every
// build: MKL's sparse-times-sparse product covers all of A, so it serves only whole-operator requests.
// RandBLAS multiplies submatrices of COO matrices only; with a CSC A the request is rejected.
TEST(TestSparseLinOp, sparse_B_with_a_submatrix_of_A) {
    double A_dense[9] = {1, 4, 7, 2, 5, 8, 3, 6, 10};  // 3 x 3, column-major
    RandBLAS::sparse_data::COOMatrix<double> A_sp(3, 3);
    RandBLAS::sparse_data::coo::dense_to_coo<double>(Layout::ColMajor, A_dense, 0.0, A_sp);
    RandLAPACK::linops::SparseLinOp<decltype(A_sp)> A_op(3, 3, A_sp);
    auto A = [&](int64_t i, int64_t j) { return A_dense[i + 3 * j]; };
    const double sentinel = -99.0;

    // Left: C (2 x 2, ldc 3) = A[0:2, 0:3] * B with B 3 x 2.
    double B_left[6] = {1, 0, 2, -1, 3, 1};
    RandBLAS::sparse_data::CSCMatrix<double> B_left_sp(3, 2);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, B_left, 0.0, B_left_sp);
    vector<double> C(6, sentinel);
    A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 2, 3, 1.0, B_left_sp, 0.0, C.data(), 3);
    for (int64_t j = 0; j < 2; ++j) {
        for (int64_t i = 0; i < 2; ++i) {
            double ref = 0;
            for (int64_t l = 0; l < 3; ++l)
                ref += A(i, l) * B_left[l + 3 * j];
            EXPECT_DOUBLE_EQ(C[i + 3 * j], ref) << "left, entry (" << i << ", " << j << ")";
        }
        EXPECT_EQ(C[2 + 3 * j], sentinel) << "left wrote row 2 of C";
    }

    // Right: C (2 x 2, ldc 3) = B * A[0:3, 0:2] with B 2 x 3; a third column of sentinels follows C.
    double B_right[6] = {1, 2, 0, -1, 3, 1};
    RandBLAS::sparse_data::CSCMatrix<double> B_right_sp(2, 3);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, B_right, 0.0, B_right_sp);
    C.assign(9, sentinel);
    A_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 2, 3, 1.0, B_right_sp, 0.0, C.data(), 3);
    for (int64_t j = 0; j < 2; ++j) {
        for (int64_t i = 0; i < 2; ++i) {
            double ref = 0;
            for (int64_t l = 0; l < 3; ++l)
                ref += B_right[i + 2 * l] * A(l, j);
            EXPECT_DOUBLE_EQ(C[i + 3 * j], ref) << "right, entry (" << i << ", " << j << ")";
        }
        EXPECT_EQ(C[2 + 3 * j], sentinel) << "right wrote row 2 of C";
    }
    for (int64_t i = 6; i < 9; ++i)
        EXPECT_EQ(C[i], sentinel) << "right wrote past the last column of C";

    RandBLAS::sparse_data::CSCMatrix<double> A_csc(3, 3);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, A_dense, 0.0, A_csc);
    RandLAPACK::linops::SparseLinOp<decltype(A_csc)> A_csc_op(3, 3, A_csc);
    C.assign(6, sentinel);
    EXPECT_THROW(A_csc_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 2, 3, 1.0, B_left_sp, 0.0, C.data(), 3),
                 RandBLAS::Error);
    EXPECT_EQ(C, vector<double>(6, sentinel));
}

// A sparse sketch larger than the request is rejected on every build without writing C: MKL's sparse
// product would multiply all of S, and RandBLAS's sketch_general takes only a sketch of the requested size.
TEST(TestSparseLinOp, rejects_a_sparse_sketch_larger_than_the_request) {
    double A_dense[9] = {1, 4, 7, 2, 5, 8, 3, 6, 10};  // 3 x 3, column-major
    RandBLAS::sparse_data::CSCMatrix<double> A_sp(3, 3);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, A_dense, 0.0, A_sp);
    RandLAPACK::linops::SparseLinOp<decltype(A_sp)> A_op(3, 3, A_sp);
    const double sentinel = -99.0;

    // Left: C (3 x 2) = A * S with S 3 x 4, two columns more than requested.
    RandBLAS::SparseSkOp<double> S_left(SparseDist(3, 4, 1), RNGState(5));
    vector<double> C(12, sentinel);
    EXPECT_THROW(A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 3, 2, 3, 1.0, S_left, 0.0, C.data(), 3),
                 RandLAPACK::Error);
    EXPECT_EQ(C, vector<double>(12, sentinel)) << "left";
    EXPECT_LT(S_left.nnz, 0) << "the rejected sketch was sampled";

    // Right: C (2 x 3) = S * A with S 4 x 3, two rows more than requested.
    RandBLAS::SparseSkOp<double> S_right(SparseDist(4, 3, 1), RNGState(6));
    EXPECT_THROW(A_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 3, 3, 1.0, S_right, 0.0, C.data(), 4),
                 RandLAPACK::Error);
    EXPECT_EQ(C, vector<double>(12, sentinel)) << "right";
    EXPECT_LT(S_right.nnz, 0) << "the rejected sketch was sampled";
}

// The operator's sparse matrix must have the dimensions it is given.
TEST(TestSparseLinOp, rejects_a_matrix_of_other_dimensions) {
    double A_dense[6] = {1, 0, 0, 1, 1, 1};
    RandBLAS::sparse_data::CSCMatrix<double> A_sp(3, 2);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, A_dense, 0.0, A_sp);
    using Op_t = RandLAPACK::linops::SparseLinOp<decltype(A_sp)>;
    EXPECT_THROW(Op_t(3, 3, A_sp), RandLAPACK::Error);
    EXPECT_THROW(Op_t(2, 2, A_sp), RandLAPACK::Error);
    EXPECT_NO_THROW(Op_t(3, 2, A_sp));
}
