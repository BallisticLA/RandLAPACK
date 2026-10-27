// Tests for linop-based orthogonalization algorithms: CholQR_linops, CholQR2_linops,
// sCholQR3_linops and sCholQR3_linops_basic, and CQRRT_linops. The primitive under the first
// three is tested in test/comps/test_cholqr.cc.
//
// Composite operators are DenseLinOp * SparseLinOp, to avoid an Eigen dependency;
// CholSolverLinOp tests are in extras/test/linops/test_ext_solver_linop_unified.cc.
//
// Verification:
//   - Factorization: ||A - Q R||_F / ||A||_F  <  tol
//   - Orthogonality: ||Q^T Q - I||_F / sqrt(n)  <  tol

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_test_utils.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include "../test_helpers.hh"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

using RandLAPACK::testing::verify_qr;
using RandLAPACK::testing::verify_R_factor;
using test_helpers::ConceptOnlyLinOp;
using test_helpers::random_matrix;

template <typename T>
T default_tol() {
    return std::pow(std::numeric_limits<T>::epsilon(), (T)0.75);
}

// Convenience: verify QR and assert both errors are below tolerance.
template <typename T>
void assert_qr_ok(const T* A, const T* Q, const T* R,
                   int64_t m, int64_t n, int64_t ldr) {
    T tol = default_tol<T>();
    auto [fact_err, orth_err] = verify_qr(A, Q, R, m, n, ldr);
    ASSERT_LE(fact_err, tol);
    ASSERT_LE(orth_err, tol);
}

// Convenience: verify R-factor only and assert both errors are below tolerance.
template <typename T>
void assert_R_ok(const T* A, int64_t m, int64_t n,
                  const T* R, int64_t ldr) {
    T tol = default_tol<T>();
    auto [fact_err, orth_err] = verify_R_factor(A, m, n, R, ldr);
    ASSERT_LE(fact_err, tol);
    ASSERT_LE(orth_err, tol);
}

// ============================================================================
// CholQR_linops
// ============================================================================

class TestCholQRLinops : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(TestCholQRLinops, dense_matrix) {
    int64_t m = 100, n = 50;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCholQRLinops, dense_matrix_float) {
    int64_t m = 100, n = 50;

    std::vector<float> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<float> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<float> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<float> R(n * n, 0.0f);
    RandLAPACK::CholQR_linops<float> algo(false, default_tol<float>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCholQRLinops, sparse_matrix) {
    int64_t m = 100, n = 50;
    RandBLAS::RNGState<> state(0);

    auto A_coo = RandLAPACK::gen::gen_sparse_coo<double>(m, n, 0.2, state);
    auto A_csc = A_coo.as_owning_csc();

    RandLAPACK::linops::SparseLinOp<decltype(A_csc)> A_linop(m, n, A_csc);

    std::vector<double> A_dense(m * n, 0.0);
    RandLAPACK::util::sparse_to_dense(A_csc, Layout::ColMajor, A_dense.data());

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCholQRLinops, composite_dense_sparse) {
    int64_t m = 100, k = 50, n = 20;
    RandBLAS::RNGState<> state(0);

    // Left operand: dense m x k
    std::vector<double> L_data(m * k);
    RandBLAS::DenseDist DL(m, k);
    RandBLAS::fill_dense(DL, L_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> L_linop(m, k, L_data.data(), m, Layout::ColMajor);

    // Right operand: sparse k x n
    auto R_coo = RandLAPACK::gen::gen_sparse_coo<double>(k, n, 0.3, state);
    auto R_csc = R_coo.as_owning_csc();
    RandLAPACK::linops::SparseLinOp<decltype(R_csc)> R_linop(k, n, R_csc);

    RandLAPACK::linops::CompositeOperator A_comp(m, n, L_linop, R_linop);

    // Dense reference: L * R_dense
    std::vector<double> R_dense(k * n, 0.0);
    RandLAPACK::util::sparse_to_dense(R_csc, Layout::ColMajor, R_dense.data());
    std::vector<double> A_dense(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
               1.0, L_data.data(), m, R_dense.data(), k, 0.0, A_dense.data(), m);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_comp, R.data(), n), 0);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCholQRLinops, blocked) {
    int64_t m = 100, n = 50;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>(), true);
    algo.block_size = 10;
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

// ============================================================================
// CQRRT_linops
// ============================================================================

class TestCQRRTLinops : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(TestCQRRTLinops, dense_matrix) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), true);
    state = RandBLAS::RNGState<>(1);
    algo.call(A_linop, R.data(), n, d_factor, state);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCQRRTLinops, dense_matrix_float) {
    int64_t m = 100, n = 50;
    float d_factor = 2.0f;

    std::vector<float> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<float> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<float> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<float> R(n * n, 0.0f);
    RandLAPACK::CQRRT_linops<float> algo(false, default_tol<float>(), true);
    state = RandBLAS::RNGState<>(1);
    algo.call(A_linop, R.data(), n, d_factor, state);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCQRRTLinops, composite_dense_sparse) {
    int64_t m = 100, k = 50, n = 20;
    double d_factor = 2.0;
    RandBLAS::RNGState<> state(0);

    std::vector<double> L_data(m * k);
    RandBLAS::DenseDist DL(m, k);
    RandBLAS::fill_dense(DL, L_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> L_linop(m, k, L_data.data(), m, Layout::ColMajor);

    auto R_coo = RandLAPACK::gen::gen_sparse_coo<double>(k, n, 0.3, state);
    auto R_csc = R_coo.as_owning_csc();
    RandLAPACK::linops::SparseLinOp<decltype(R_csc)> R_linop(k, n, R_csc);

    RandLAPACK::linops::CompositeOperator A_comp(m, n, L_linop, R_linop);

    std::vector<double> R_dense(k * n, 0.0);
    RandLAPACK::util::sparse_to_dense(R_csc, Layout::ColMajor, R_dense.data());
    std::vector<double> A_dense(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
               1.0, L_data.data(), m, R_dense.data(), k, 0.0, A_dense.data(), m);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), true);
    algo.nnz = 2;
    algo.call(A_comp, R.data(), n, d_factor, state);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}

// --- Block processing tests ---

TEST_F(TestCQRRTLinops, block_processing_even_division) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), false);
    algo.block_size = 10;
    state = RandBLAS::RNGState<>(1);
    algo.call(A_linop, R.data(), n, d_factor, state);

    assert_R_ok(A_data.data(), m, n, R.data(), n);
}

TEST_F(TestCQRRTLinops, block_processing_with_remainder) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), false);
    algo.block_size = 12;  // 50 / 12 = 4 blocks of 12, remainder of 2
    state = RandBLAS::RNGState<>(1);
    algo.call(A_linop, R.data(), n, d_factor, state);

    assert_R_ok(A_data.data(), m, n, R.data(), n);
}

TEST_F(TestCQRRTLinops, block_processing_single_column) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), false);
    algo.block_size = 1;
    state = RandBLAS::RNGState<>(1);
    algo.call(A_linop, R.data(), n, d_factor, state);

    assert_R_ok(A_data.data(), m, n, R.data(), n);
}

TEST_F(TestCQRRTLinops, block_vs_full_agreement) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(42);
    RandBLAS::fill_dense(D, A_data.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    // Full path
    std::vector<double> R_full(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> alg_full(false, default_tol<double>(), false);
    state = RandBLAS::RNGState<>(1);
    alg_full.call(A_linop, R_full.data(), n, d_factor, state);

    // Block path
    std::vector<double> R_block(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> alg_block(false, default_tol<double>(), false);
    alg_block.block_size = 10;
    state = RandBLAS::RNGState<>(1);  // same seed
    alg_block.call(A_linop, R_block.data(), n, d_factor, state);

    // ||R_full - R_block|| / ||R_full||
    double norm_R = lapack::lange(Norm::Fro, n, n, R_full.data(), n);
    std::vector<double> diff(n * n);
    for (int64_t i = 0; i < n * n; ++i)
        diff[i] = R_full[i] - R_block[i];
    double norm_diff = lapack::lange(Norm::Fro, n, n, diff.data(), n);

    ASSERT_LE(norm_diff / norm_R, 1000 * std::numeric_limits<double>::epsilon());
}

// ============================================================================
// sCholQR3_linops (fully-blocked)
// ============================================================================

class TestSCholQR3Linops : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(TestSCholQR3Linops, dense_matrix) {
    int64_t m = 100, n = 50;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestSCholQR3Linops, dense_matrix_float) {
    int64_t m = 100, n = 50;

    std::vector<float> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<float> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<float> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<float> R(n * n, 0.0f);
    RandLAPACK::sCholQR3_linops<float> algo(false, default_tol<float>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestSCholQR3Linops, sparse_matrix) {
    int64_t m = 100, n = 50;
    RandBLAS::RNGState<> state(0);

    auto A_coo = RandLAPACK::gen::gen_sparse_coo<double>(m, n, 0.2, state);
    auto A_csc = A_coo.as_owning_csc();

    RandLAPACK::linops::SparseLinOp<decltype(A_csc)> A_linop(m, n, A_csc);

    std::vector<double> A_dense(m * n, 0.0);
    RandLAPACK::util::sparse_to_dense(A_csc, Layout::ColMajor, A_dense.data());

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestSCholQR3Linops, composite_dense_sparse) {
    int64_t m = 100, k = 50, n = 20;
    RandBLAS::RNGState<> state(0);

    std::vector<double> L_data(m * k);
    RandBLAS::DenseDist DL(m, k);
    RandBLAS::fill_dense(DL, L_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> L_linop(m, k, L_data.data(), m, Layout::ColMajor);

    auto R_coo = RandLAPACK::gen::gen_sparse_coo<double>(k, n, 0.3, state);
    auto R_csc = R_coo.as_owning_csc();
    RandLAPACK::linops::SparseLinOp<decltype(R_csc)> R_linop(k, n, R_csc);

    RandLAPACK::linops::CompositeOperator A_comp(m, n, L_linop, R_linop);

    std::vector<double> R_dense(k * n, 0.0);
    RandLAPACK::util::sparse_to_dense(R_csc, Layout::ColMajor, R_dense.data());
    std::vector<double> A_dense(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
               1.0, L_data.data(), m, R_dense.data(), k, 0.0, A_dense.data(), m);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_comp, R.data(), n), 0);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestSCholQR3Linops, blocked) {
    int64_t m = 100, n = 50;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>(), true);
    algo.block_size = 10;
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

// The 18 timing slots: alloc and upd1 are 0, and the total excludes q_mat.
TEST_F(TestSCholQR3Linops, timing_layout) {
    int64_t m = 120, n = 30;
    auto A_data = random_matrix<double>(m, n, 3);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(/*timing=*/true, default_tol<double>(), /*test_mode=*/true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    const auto& t = algo.times;
    ASSERT_EQ(t.size(), 18u);
    EXPECT_EQ(t[0], 0);
    EXPECT_EQ(t[4], 0);
    EXPECT_GE(t[15], 0);
    EXPECT_EQ(std::accumulate(t.begin() + 1, t.begin() + 15, 0L) + t[16], t[17]);
    EXPECT_EQ(algo.total_us(), t[17]);
}

// ============================================================================
// sCholQR3_linops_basic (non-blocked, standard algorithm)
// ============================================================================

class TestSCholQR3LinopsBasic : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(TestSCholQR3LinopsBasic, dense_matrix) {
    int64_t m = 100, n = 50;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops_basic<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestSCholQR3LinopsBasic, dense_matrix_float) {
    int64_t m = 100, n = 50;

    std::vector<float> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<float> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<float> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<float> R(n * n, 0.0f);
    RandLAPACK::sCholQR3_linops_basic<float> algo(false, default_tol<float>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestSCholQR3LinopsBasic, composite_dense_sparse) {
    int64_t m = 100, k = 50, n = 20;
    RandBLAS::RNGState<> state(0);

    std::vector<double> L_data(m * k);
    RandBLAS::DenseDist DL(m, k);
    RandBLAS::fill_dense(DL, L_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> L_linop(m, k, L_data.data(), m, Layout::ColMajor);

    auto R_coo = RandLAPACK::gen::gen_sparse_coo<double>(k, n, 0.3, state);
    auto R_csc = R_coo.as_owning_csc();
    RandLAPACK::linops::SparseLinOp<decltype(R_csc)> R_linop(k, n, R_csc);

    RandLAPACK::linops::CompositeOperator A_comp(m, n, L_linop, R_linop);

    std::vector<double> R_dense(k * n, 0.0);
    RandLAPACK::util::sparse_to_dense(R_csc, Layout::ColMajor, R_dense.data());
    std::vector<double> A_dense(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
               1.0, L_data.data(), m, R_dense.data(), k, 0.0, A_dense.data(), m);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops_basic<double> algo(false, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_comp, R.data(), n), 0);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}

// ============================================================================
// CholQR2_linops
// ============================================================================

class TestCholQR2Linops : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

TEST_F(TestCholQR2Linops, dense_matrix) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<double>(m, n, 30);
    std::vector<double> A_copy = A_data;
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR2_linops<double> algo(true, default_tol<double>(), true);
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
    ASSERT_EQ(algo.times.size(), (size_t)13);
    ASSERT_EQ(algo.total_us(), algo.times.back());
}

TEST_F(TestCholQR2Linops, dense_matrix_float_blocked) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<float>(m, n, 31);
    std::vector<float> A_copy = A_data;
    RandLAPACK::linops::DenseLinOp<float> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<float> R(n * n, 0.0f);
    RandLAPACK::CholQR2_linops<float> algo(false, default_tol<float>(), true);
    algo.block_size = 12;
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

// Ill-conditioned input (kappa = 1e7): one CholQR pass loses orthogonality like kappa^2 u,
// the second pass restores it. Guards the pass-2 preconditioning.
TEST_F(TestCholQR2Linops, second_pass_restores_orthogonality) {
    int64_t m = 200, n = 20;
    // Singular values from 1 down to 1e-7 with random singular vectors (column scaling alone
    // would not do: CholQR is invariant to it).
    std::vector<double> S(n * n, 0.0), A_data(m * n);
    for (int64_t i = 0; i < n; ++i) S[i + i * n] = std::pow(10.0, -7.0 * (double)i / (double)(n - 1));
    RandBLAS::RNGState<> state(32);
    RandLAPACK::gen::gen_singvec(m, n, A_data.data(), n, S.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R1(n * n, 0.0), R2(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> one(false, default_tol<double>(), true);
    RandLAPACK::CholQR2_linops<double> two(false, default_tol<double>(), true);
    ASSERT_EQ(one.call(A_linop, R1.data(), n), 0);
    ASSERT_EQ(two.call(A_linop, R2.data(), n), 0);
    const double orth1 = RandLAPACK::testing::orthogonality_error(one.Q, m, n);
    const double orth2 = RandLAPACK::testing::orthogonality_error(two.Q, m, n);
    // Q = A R^{-1} is itself formed with error ~ kappa(R) u ~ 1e-9, which floors orth2.
    EXPECT_GT(orth1, 1e-6);
    EXPECT_LT(orth2, 1e-8);
    EXPECT_LT(orth2, 1e-3 * orth1);
}

// The two left-factor routes of the preconditioned passes (one TRSM with P after the Gram
// loop, or a GEMM with R_pre^T per block) and the unsymmetrized Gram all give a valid QR.
TEST_F(TestCholQR2Linops, gram_options_all_factor) {
    int64_t m = 120, n = 30;
    auto A_data = random_matrix<double>(m, n, 33);
    std::vector<double> A_copy = A_data;
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    for (bool gemm_left : {false, true}) {
        for (bool symmetrize : {false, true}) {
            std::vector<double> R(n * n, 0.0);
            RandLAPACK::CholQR2_linops<double> algo(false, default_tol<double>(), true);
            algo.block_size = 8;
            algo.gram_left_gemm  = gemm_left;
            algo.symmetrize_gram = symmetrize;
            ASSERT_EQ(algo.call(A_linop, R.data(), n), 0) << gemm_left << symmetrize;
            assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
        }
    }
}

// ============================================================================
// Adaptive-shift record and input validation
// ============================================================================

class TestCholQRShiftRecord : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// A clean (well-conditioned) input must factor on the unshifted first attempt
// and record zero shift with a positive Gram trace.
TEST_F(TestCholQRShiftRecord, clean_input_records_no_shift) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<double>(m, n, 0);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_EQ(algo.n_chol_retries, 0);
    ASSERT_EQ(algo.chol_status, 0);
    ASSERT_EQ(algo.chol_applied_shifts[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
}

// An exactly zero column gives a Gram with an exact zero pivot, so unshifted potrf fails
// deterministically: the retry must fire and record the applied shift, which for the eps
// seed grown k-1 times is eps * trace(G) * growth^(retries - 1).
TEST_F(TestCholQRShiftRecord, rank_deficient_input_records_seed_shift) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<double>(m, n, 1);
    for (int64_t i = 0; i < m; ++i) A_data[m + i] = 0.0;
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_GE(algo.n_chol_retries, 1);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    ASSERT_GT(algo.chol_applied_shifts[0], 0.0);
    const double eps = std::numeric_limits<double>::epsilon();
    double expected = eps * algo.chol_gram_traces[0];
    for (int k = 1; k < algo.n_chol_retries; ++k) expected *= algo.shift_growth;
    ASSERT_NEAR(algo.chol_applied_shifts[0], expected, 1e-12 * expected);
}

// The same input with max_retries = 0 must fail and say why: pass 1, potrf's positive info.
TEST_F(TestCholQRShiftRecord, rank_deficient_input_without_retries_reports_cause) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<double>(m, n, 1);
    for (int64_t i = 0; i < m; ++i) A_data[m + i] = 0.0;
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>());
    algo.max_retries = 0;
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 1);
    EXPECT_GT(algo.chol_status, 0);
    EXPECT_EQ(algo.n_chol_retries, 0);

    RandLAPACK::sCholQR3_linops<double> s3(false, default_tol<double>());
    s3.max_retries = 0;
    s3.shift_factor_iter1 = 0.0;
    ASSERT_EQ(s3.call(A_linop, R.data(), n), 1);
    EXPECT_GT(s3.chol_status, 0);
}

// CholQR2 on the same rank-deficient input must fill the per-pass record: the pass-1 shift
// fires (singular Gram) and every reported trace is positive.
TEST_F(TestCholQRShiftRecord, cholqr2_per_pass_record) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<double>(m, n, 2);
    for (int64_t i = 0; i < m; ++i) A_data[m + i] = 0.0;
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR2_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_GE(algo.n_chol_retries, 1);
    ASSERT_GT(algo.chol_applied_shifts[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[1], 0.0);
}

// sCholQR3's pass-1 shift defaults to s = 11 * n * eps * trace(G), so pass 1 carries a
// shift even on a clean input, with zero retries; a caller-set shift_factor_iter1 >= 0 is
// used as given. Passes 2-3 stay unshifted on a clean input.
TEST_F(TestCholQRShiftRecord, scholqr3_clean_input_records_pass1_shift) {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix<double>(m, n, 5);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_EQ(algo.n_chol_retries, 0);

    const double eps = std::numeric_limits<double>::epsilon();
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    const double expected_pass1_shift = 11.0 * (double)n * eps * algo.chol_gram_traces[0];
    ASSERT_NEAR(algo.chol_applied_shifts[0], expected_pass1_shift, 1e-12 * expected_pass1_shift);
    ASSERT_EQ(algo.chol_applied_shifts[1], 0.0);
    ASSERT_EQ(algo.chol_applied_shifts[2], 0.0);

    RandLAPACK::sCholQR3_linops<double> algo_override(false, default_tol<double>());
    algo_override.shift_factor_iter1 = eps;
    ASSERT_EQ(algo_override.call(A_linop, R.data(), n), 0);
    const double expected_override_shift = eps * algo_override.chol_gram_traces[0];
    ASSERT_NEAR(algo_override.chol_applied_shifts[0], expected_override_shift, 1e-12 * expected_override_shift);
}

// Only a negative shift_factor_iter1 selects the default: NaN reaches the primitive and is
// rejected as an invalid shift on pass 1.
TEST_F(TestCholQRShiftRecord, scholqr3_nan_shift_factor_is_rejected) {
    int64_t m = 60, n = 20;
    auto A_data = random_matrix<double>(m, n, 7);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>());
    algo.shift_factor_iter1 = std::numeric_limits<double>::quiet_NaN();
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 1);
    EXPECT_EQ(algo.chol_status, RandLAPACK::kCholQRInvalidShift);
}

// ============================================================================
// Behavior shared by CholQR_linops, CholQR2_linops, sCholQR3_linops, sCholQR3_linops_basic
// ============================================================================

namespace {

// A dense operand that records the widest block it is applied to (the column count of the
// left-side product).
struct WidthRecordingOp {
    using scalar_t = double;
    const int64_t n_rows;
    const int64_t n_cols;
    RandLAPACK::linops::DenseLinOp<double>& A;
    int64_t max_width = 0;
    void operator()(Layout layout, blas::Op trans_A, blas::Op trans_B, int64_t m, int64_t n, int64_t k,
                    double alpha, const double* B, int64_t ldb, double beta, double* C, int64_t ldc) {
        max_width = std::max(max_width, n);
        A(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }
    void operator()(blas::Side side, Layout layout, blas::Op trans_A, blas::Op trans_B, int64_t m, int64_t n,
                    int64_t k, double alpha, const double* B, int64_t ldb, double beta, double* C, int64_t ldc) {
        max_width = std::max(max_width, side == blas::Side::Left ? n : m);
        A(side, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }
};

template <typename QR>
constexpr bool validates_input = requires(QR qr) { qr.max_retries; };   // the CholQRLinopsBase drivers

// A concept-only operator compiles and factors, with test mode off and on.
template <typename QR>
void check_concept_only_operator() {
    int64_t m = 90, n = 15;
    auto A_data = random_matrix(m, n, 41);
    ConceptOnlyLinOp<double> A_op{m, n, A_data.data()};
    for (bool test_mode : {false, true}) {
        std::vector<double> R(n * n, 0.0);
        QR qr(false, default_tol<double>(), test_mode);
        ASSERT_EQ(qr.call(A_op, R.data(), n), 0) << "test_mode " << test_mode;
        if (test_mode) assert_qr_ok(A_data.data(), qr.Q, R.data(), m, n, n);
        else           assert_R_ok(A_data.data(), m, n, R.data(), n);
    }
}

// A second call on one object, with a different operator, replaces the test-mode Q:
// A2 = Q R2 with Q orthonormal.
template <typename QR>
void check_repeated_call_replaces_Q() {
    int64_t m = 300, n = 16;
    auto A1 = random_matrix(m, n, 37), A2 = random_matrix(m, n, 38);
    std::vector<double> A2_copy = A2;
    RandLAPACK::linops::DenseLinOp<double> A1_linop(m, n, A1.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> A2_linop(m, n, A2.data(), m, Layout::ColMajor);
    QR qr(false, 0.0, /*test_mode=*/true);
    std::vector<double> R1(n * n, 0.0), R2(n * n, 0.0);
    ASSERT_EQ(qr.call(A1_linop, R1.data(), n), 0);
    ASSERT_EQ(qr.call(A2_linop, R2.data(), n), 0);
    ASSERT_NE(qr.Q, nullptr);
    EXPECT_EQ(qr.Q_rows, m);
    EXPECT_EQ(qr.Q_cols, n);
    assert_qr_ok(A2_copy.data(), qr.Q, R2.data(), m, n, n);
}

// times is empty after a failing call (a NaN entry) and, for the drivers that validate, after
// a call that throws (ldr < n); the throw also resets the failed call's status and record.
template <typename QR>
void check_failed_call_resets_outputs() {
    int64_t m = 100, n = 50;
    auto A_data = random_matrix(m, n, 8);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    QR qr(/*timing=*/true, default_tol<double>());
    if constexpr (validates_input<QR>) {
        ASSERT_EQ(qr.call(A_linop, R.data(), n), 0);
        ASSERT_GE(qr.total_us(), 0);
        EXPECT_THROW(qr.call(A_linop, R.data(), n - 1), RandLAPACK::Error);
        EXPECT_TRUE(qr.times.empty());
    }
    ASSERT_EQ(qr.call(A_linop, R.data(), n), 0);
    ASSERT_GE(qr.total_us(), 0);
    A_data[3 + 5 * m] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NE(qr.call(A_linop, R.data(), n), 0);
    EXPECT_TRUE(qr.times.empty());
    EXPECT_EQ(qr.total_us(), -1);
    if constexpr (validates_input<QR>) {
        EXPECT_EQ(qr.chol_status, RandLAPACK::kCholQRNonFinite);
        EXPECT_TRUE(std::isnan(qr.chol_gram_traces[0]));
        EXPECT_THROW(qr.call(A_linop, R.data(), n - 1), RandLAPACK::Error);
        EXPECT_EQ(qr.chol_status, 0);
        EXPECT_EQ(qr.n_chol_retries, 0);
        EXPECT_EQ(qr.chol_gram_traces[0], 0.0);
        EXPECT_EQ(qr.chol_applied_shifts[0], 0.0);
    }
}

} // namespace

class TestCholQRFamilyDrivers : public ::testing::Test {};

TEST_F(TestCholQRFamilyDrivers, concept_only_operator) {
    check_concept_only_operator<RandLAPACK::CholQR_linops<double>>();
    check_concept_only_operator<RandLAPACK::CholQR2_linops<double>>();
    check_concept_only_operator<RandLAPACK::sCholQR3_linops<double>>();
    check_concept_only_operator<RandLAPACK::sCholQR3_linops_basic<double>>();
}

TEST_F(TestCholQRFamilyDrivers, repeated_call_replaces_Q) {
    check_repeated_call_replaces_Q<RandLAPACK::CholQR_linops<double>>();
    check_repeated_call_replaces_Q<RandLAPACK::CholQR2_linops<double>>();
    check_repeated_call_replaces_Q<RandLAPACK::sCholQR3_linops<double>>();
    check_repeated_call_replaces_Q<RandLAPACK::sCholQR3_linops_basic<double>>();
}

TEST_F(TestCholQRFamilyDrivers, failed_call_resets_outputs) {
    check_failed_call_resets_outputs<RandLAPACK::CholQR_linops<double>>();
    check_failed_call_resets_outputs<RandLAPACK::CholQR2_linops<double>>();
    check_failed_call_resets_outputs<RandLAPACK::sCholQR3_linops<double>>();
    check_failed_call_resets_outputs<RandLAPACK::sCholQR3_linops_basic<double>>();
}

// The test-mode Q of a composite L V with a square left operand applies both operands to
// block_size-wide blocks only: neither is materialized.
TEST_F(TestCholQRFamilyDrivers, test_mode_q_applies_operands_blockwise) {
    int64_t m = 120, n = 20, b = 8;
    auto L_data = random_matrix(m, m, 51), V_data = random_matrix(m, n, 52);
    RandLAPACK::linops::DenseLinOp<double> L_dense(m, m, L_data.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> V_dense(m, n, V_data.data(), m, Layout::ColMajor);
    std::vector<double> A_dense(m * n);
    blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, m, 1.0,
               L_data.data(), m, V_data.data(), m, 0.0, A_dense.data(), m);
    for (int variant = 0; variant < 2; ++variant) {
        WidthRecordingOp L{m, m, L_dense}, V{m, n, V_dense};
        RandLAPACK::linops::CompositeOperator A(m, n, L, V);
        std::vector<double> R(n * n, 0.0);
        if (variant == 0) {
            RandLAPACK::CholQR_linops<double> qr(false, 0.0, /*test_mode=*/true);
            qr.block_size = b;
            ASSERT_EQ(qr.call(A, R.data(), n), 0);
            assert_qr_ok(A_dense.data(), qr.Q, R.data(), m, n, n);
        } else {
            RandLAPACK::sCholQR3_linops<double> qr(false, 0.0, /*test_mode=*/true);
            qr.block_size = b;
            ASSERT_EQ(qr.call(A, R.data(), n), 0);
            assert_qr_ok(A_dense.data(), qr.Q, R.data(), m, n, n);
        }
        EXPECT_LE(L.max_width, b) << "variant " << variant;
        EXPECT_LE(V.max_width, b) << "variant " << variant;
    }
}

// An operator exception while forming the test-mode Q (the Gram of an unblocked CholQR takes
// two operator calls, the Q the third) propagates: no Q is installed and times stays empty, but
// R and the completed pass's shift record are already written.
TEST_F(TestCholQRFamilyDrivers, test_mode_q_failure_installs_no_q) {
    int64_t m = 60, n = 12;
    auto A_data = random_matrix(m, n, 9);
    ConceptOnlyLinOp<double> A_op{m, n, A_data.data(), /*calls_left=*/2};
    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(/*timing=*/true, default_tol<double>(), /*test_mode=*/true);
    algo.block_size = 0;
    EXPECT_THROW(algo.call(A_op, R.data(), n), std::runtime_error);
    EXPECT_EQ(algo.Q, nullptr);
    EXPECT_TRUE(algo.times.empty());
    EXPECT_EQ(algo.chol_status, 0);
    EXPECT_GT(algo.chol_gram_traces[0], 0.0);
    EXPECT_GT(R[0], 0.0);
    assert_R_ok(A_data.data(), m, n, R.data(), n);
}

namespace {

// With an operator that has materialize_cols, the unpreconditioned first Gram, sCholQR3_basic's
// first product and the test-mode Q read its columns, so A is applied only in the preconditioned
// passes (and in sCholQR3_basic's Q = A R1^{-1}): product_cols columns in all, direct_cols read.
// R and Q equal those of a DenseLinOp on the same matrix, whose products with identity columns
// are exact. Blocks of 8 over n = 20 end with a partial block.
template <typename QR>
void check_materialize_cols_path(int64_t product_cols, int64_t direct_cols) {
    const int64_t m = 90, n = 20;
    auto A_data = random_matrix(m, n, 43);
    test_helpers::MaterializeColsProbe<double> probe{m, n, A_data.data()};
    RandLAPACK::linops::DenseLinOp<double> dense(m, n, A_data.data(), m, Layout::ColMajor);
    std::vector<double> R(n * n, 0.0), R_dense(n * n, 0.0);
    QR qr(false, 0.0, /*test_mode=*/true), qr_dense(false, 0.0, /*test_mode=*/true);
    if constexpr (requires { qr.block_size; }) qr.block_size = qr_dense.block_size = 8;
    ASSERT_EQ(qr.call(probe, R.data(), n), 0);
    ASSERT_EQ(qr_dense.call(dense, R_dense.data(), n), 0);
    EXPECT_EQ(probe.product_cols, product_cols);
    EXPECT_EQ(probe.direct_cols, direct_cols);
    const double tol = 10 * std::numeric_limits<double>::epsilon();
    EXPECT_LE(RandLAPACK::testing::rel_fro_error(R.data(), R_dense.data(), n * n), tol);
    EXPECT_LE(RandLAPACK::testing::rel_fro_error(qr.Q, qr_dense.Q, m * n), tol);
}

} // namespace

TEST_F(TestCholQRFamilyDrivers, materialize_cols_replaces_identity_products) {
    const int64_t n = 20;
    check_materialize_cols_path<RandLAPACK::CholQR_linops<double>>(0, 2 * n);
    check_materialize_cols_path<RandLAPACK::CholQR2_linops<double>>(n, 2 * n);
    check_materialize_cols_path<RandLAPACK::sCholQR3_linops<double>>(2 * n, 2 * n);
    check_materialize_cols_path<RandLAPACK::sCholQR3_linops_basic<double>>(n, n);
}
