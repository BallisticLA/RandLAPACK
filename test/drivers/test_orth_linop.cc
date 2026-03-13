// Tests for linop-based orthogonalization algorithms: CholQR_linops,
// CQRRT_linops, sCholQR3_linops, sCholQR3_linops_basic.
//
// Ported from PR #115 (demos/test/drivers/test_dm_{cholqr,cqrrt,scholqr3}_linops.cc).
// CholSolverLinOp-based composite tests are replaced with DenseLinOp * SparseLinOp
// composites to avoid Eigen dependency. CholSolverLinOp tests will be added in PR C.
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
#include <vector>
#include <cmath>

using RandLAPACK::testing::verify_qr;
using RandLAPACK::testing::verify_R_factor;

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_comp, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_comp, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_linop, R.data(), n);

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
    algo.call(A_comp, R.data(), n);

    assert_qr_ok(A_dense.data(), algo.Q, R.data(), m, n, n);
}
