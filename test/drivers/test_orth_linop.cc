// Tests for linop-based orthogonalization algorithms: CholQR_linops,
// CQRRT_linops, sCholQR3_linops, sCholQR3_linops_basic.
//
// Ported from demos/test/drivers/test_dm_{cholqr,cqrrt,scholqr3}_linops.cc.
// CholSolverLinOp-based composite tests are replaced with DenseLinOp * SparseLinOp
// composites to avoid Eigen dependency. CholSolverLinOp tests are in
// extras/test/linops/test_ext_solver_linop_unified.cc.
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
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

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
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

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
    ASSERT_EQ(algo.call(A_comp, R.data(), n, d_factor, state), 0);

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
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

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
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

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
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

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
    ASSERT_EQ(alg_full.call(A_linop, R_full.data(), n, d_factor, state), 0);

    // Block path
    std::vector<double> R_block(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> alg_block(false, default_tol<double>(), false);
    alg_block.block_size = 10;
    state = RandBLAS::RNGState<>(1);  // same seed
    ASSERT_EQ(alg_block.call(A_linop, R_block.data(), n, d_factor, state), 0);

    // ||R_full - R_block|| / ||R_full||
    double norm_R = lapack::lange(Norm::Fro, n, n, R_full.data(), n);
    std::vector<double> diff(n * n);
    for (int64_t i = 0; i < n * n; ++i)
        diff[i] = R_full[i] - R_block[i];
    double norm_diff = lapack::lange(Norm::Fro, n, n, diff.data(), n);

    ASSERT_LE(norm_diff / norm_R, 1000 * std::numeric_limits<double>::epsilon());
}

// --- Precond-method coverage: TRTRI / GEQP3 / BQRRP all should produce a
//     valid Q-less QR (Q = A * R^{-1} has orthonormal columns). Exercises
//     the dispatch in cholqr_primitive via the CQRRT_linops wrapper.

TEST_F(TestCQRRTLinops, precond_method_TRTRI) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(7);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), true);
    algo.precond_method = RandLAPACK::PCholQRPrecondMethod::TRTRI;
    state = RandBLAS::RNGState<>(11);
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

TEST_F(TestCQRRTLinops, precond_method_GEQP3) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(7);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), true);
    algo.precond_method = RandLAPACK::PCholQRPrecondMethod::GEQP3;
    state = RandBLAS::RNGState<>(11);
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}

// BQRRP throws on macOS (BLAS = Apple Accelerate lacks the required LAPACK
// routines), so the BQRRP preconditioner path is unavailable there. Skip this
// case on Apple: same guard the dedicated BQRRP/CQRRPT/HQRRP tests use.
#if !defined(__APPLE__)
TEST_F(TestCQRRTLinops, precond_method_BQRRP) {
    int64_t m = 100, n = 50;
    double d_factor = 2.0;

    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(7);
    RandBLAS::fill_dense(D, A_data.data(), state);
    std::vector<double> A_copy = A_data;

    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> algo(false, default_tol<double>(), true);
    algo.precond_method = RandLAPACK::PCholQRPrecondMethod::BQRRP;
    state = RandBLAS::RNGState<>(11);
    ASSERT_EQ(algo.call(A_linop, R.data(), n, d_factor, state), 0);

    assert_qr_ok(A_copy.data(), algo.Q, R.data(), m, n, n);
}
#endif  // !defined(__APPLE__)

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
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(0);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_EQ(algo.n_chol_retries, 0);
    ASSERT_EQ(algo.chol_applied_shifts[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
}

// A rank-deficient input (an exactly zero column) gives a Gram with an exact
// zero pivot, so unshifted potrf fails deterministically: the retry must fire
// and record the applied shift, which for the eps seed grown k-1 times is
// eps * trace(G) * growth^(retries - 1).
TEST_F(TestCholQRShiftRecord, rank_deficient_input_records_seed_shift) {
    int64_t m = 100, n = 50;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(1);
    RandBLAS::fill_dense(D, A_data.data(), state);
    // Zero out column 1: its Gram row/column is exactly zero, so the potrf pivot
    // there is exactly 0 and the unshifted attempt cannot succeed.
    for (int64_t i = 0; i < m; ++i) A_data[m + i] = 0.0;
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_GE(algo.n_chol_retries, 1);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    ASSERT_GT(algo.chol_applied_shifts[0], 0.0);
    const double eps = std::numeric_limits<double>::epsilon();
    const double expected_seed = eps * algo.chol_gram_traces[0];
    // shift = seed * growth^(retries - 1); with one retry this is the seed itself.
    double expected = expected_seed;
    for (int k = 1; k < algo.n_chol_retries; ++k) expected *= algo.shift_growth;
    ASSERT_NEAR(algo.chol_applied_shifts[0], expected, 1e-12 * expected);
}

// CholQR2 on the same rank-deficient input must fill the per-pass record: the
// pass-1 shift fires (singular Gram) and every reported trace is positive.
TEST_F(TestCholQRShiftRecord, cholqr2_per_pass_record) {
    int64_t m = 100, n = 50;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(2);
    RandBLAS::fill_dense(D, A_data.data(), state);
    for (int64_t i = 0; i < m; ++i) A_data[m + i] = 0.0;   // exact zero column
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR2_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_GE(algo.n_chol_retries, 1);
    ASSERT_GT(algo.chol_applied_shifts[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    ASSERT_GT(algo.chol_gram_traces[1], 0.0);
}

// A negative shift_factor is invalid input and must be rejected up front, not
// silently treated as "no shift".
TEST_F(TestCholQRShiftRecord, negative_shift_factor_rejected) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(3);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    long fwd = 0, adj = 0, chol = 0;
    int n_retries = -7;
    double applied_shift = -7.0, gram_trace = -7.0;
    int info = RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, R.data(), n, /*shift_factor=*/-1.0, /*block_size=*/0,
        fwd, adj, chol, /*timing=*/false, /*max_retries=*/-1, /*shift_growth=*/10.0,
        &n_retries, &applied_shift, &gram_trace);
    ASSERT_EQ(info, -2);
    // Out-params must be reset even on the early rejection (stale-value guard).
    ASSERT_EQ(n_retries, 0);
    ASSERT_EQ(applied_shift, 0.0);
    ASSERT_EQ(gram_trace, 0.0);
}

// shift_growth <= 1 defeats the geometric-growth retry termination argument
// (a growth factor of 1 or less never escalates the shift) and must be
// rejected the same way a negative shift_factor is.
TEST_F(TestCholQRShiftRecord, shift_growth_leq_one_rejected) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(4);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    long fwd = 0, adj = 0, chol = 0;
    int n_retries = -7;
    double applied_shift = -7.0, gram_trace = -7.0;
    int info = RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, R.data(), n, /*shift_factor=*/0.0, /*block_size=*/0,
        fwd, adj, chol, /*timing=*/false, /*max_retries=*/-1, /*shift_growth=*/1.0,
        &n_retries, &applied_shift, &gram_trace);
    ASSERT_EQ(info, -2);
    // Out-params must be reset even on the early rejection (stale-value guard).
    ASSERT_EQ(n_retries, 0);
    ASSERT_EQ(applied_shift, 0.0);
    ASSERT_EQ(gram_trace, 0.0);
}

// sCholQR3's pass-1 shift defaults to the paper's prescription
// s = 11 * n * eps * trace(G) (unlike CholQR/CholQR2, which default to 0), so
// pass 1 always carries a shift, even on a clean, well-conditioned input, with
// zero retries. A caller-set shift_factor_iter1 >= 0 overrides the default and
// is used verbatim. Passes 2-3 default to shift_factor_iter23 = 0 and a clean
// input never trips their retry path, so their recorded shifts stay 0.
TEST_F(TestCholQRShiftRecord, scholqr3_clean_input_records_pass1_shift) {
    int64_t m = 100, n = 50;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(5);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    RandLAPACK::sCholQR3_linops<double> algo(false, default_tol<double>());
    ASSERT_EQ(algo.call(A_linop, R.data(), n), 0);
    ASSERT_EQ(algo.n_chol_retries, 0);

    const double eps = std::numeric_limits<double>::epsilon();
    ASSERT_GT(algo.chol_gram_traces[0], 0.0);
    const double expected_pass1_shift = 11.0 * (double)n * eps * algo.chol_gram_traces[0];
    ASSERT_NEAR(algo.chol_applied_shifts[0], expected_pass1_shift,
                1e-12 * expected_pass1_shift);
    ASSERT_EQ(algo.chol_applied_shifts[1], 0.0);
    ASSERT_EQ(algo.chol_applied_shifts[2], 0.0);

    // Explicit override: a caller-set factor >= 0 replaces the default.
    RandLAPACK::sCholQR3_linops<double> algo_override(false, default_tol<double>());
    algo_override.shift_factor_iter1 = eps;
    ASSERT_EQ(algo_override.call(A_linop, R.data(), n), 0);
    const double expected_override_shift = eps * algo_override.chol_gram_traces[0];
    ASSERT_NEAR(algo_override.chol_applied_shifts[0], expected_override_shift,
                1e-12 * expected_override_shift);
}

// ============================================================================
// cholqr_primitive failure paths
// ============================================================================

class TestCholQRPrimitiveFailurePaths : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// TRSM_IDENTITY must refuse a singular preconditioner (an exact zero diagonal
// entry in P) rather than dividing by zero in the TRSM inverse.
TEST_F(TestCholQRPrimitiveFailurePaths, trsm_identity_singular_preconditioner_returns_1) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(201);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> P(n * n, 0.0);
    RandLAPACK::util::eye(n, n, P.data());
    P[5 + 5 * n] = 0.0;   // zero out one diagonal entry: P is singular

    std::vector<double> R(n * n, 0.0), R_pre(n * n, 0.0), G(n * n, 0.0);
    std::vector<double> A_temp(m * n, 0.0), Z_buf(n * n, 0.0);
    long precond_inv_us = 0, fwd_us = 0, adj_us = 0, gemm_us = 0, chol_us = 0, update_us = 0;

    int info = RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, P.data(), R.data(), n,
        RandLAPACK::PCholQRPrecondMethod::TRSM_IDENTITY,
        /*block_size=*/0, /*bqrrp_block_ratio=*/1.0,
        R_pre.data(), G.data(), A_temp.data(), Z_buf.data(),
        /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
        precond_inv_us, fwd_us, adj_us, gemm_us, chol_us, update_us,
        /*timing=*/false);

    EXPECT_EQ(info, 1);
}

// TRTRI must refuse a singular preconditioner (an exact zero diagonal entry
// in P) the same way TRSM_IDENTITY does above, before ever calling trtri.
TEST_F(TestCholQRPrimitiveFailurePaths, trtri_singular_preconditioner_returns_1) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(202);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> P(n * n, 0.0);
    RandLAPACK::util::eye(n, n, P.data());
    P[5 + 5 * n] = 0.0;   // zero out one diagonal entry: P is singular

    std::vector<double> R(n * n, 0.0), R_pre(n * n, 0.0), G(n * n, 0.0);
    std::vector<double> A_temp(m * n, 0.0), Z_buf(n * n, 0.0);
    long precond_inv_us = 0, fwd_us = 0, adj_us = 0, gemm_us = 0, chol_us = 0, update_us = 0;

    int info = RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, P.data(), R.data(), n,
        RandLAPACK::PCholQRPrecondMethod::TRTRI,
        /*block_size=*/0, /*bqrrp_block_ratio=*/1.0,
        R_pre.data(), G.data(), A_temp.data(), Z_buf.data(),
        /*state=*/(RandBLAS::RNGState<RandBLAS::DefaultRNG>*)nullptr,
        precond_inv_us, fwd_us, adj_us, gemm_us, chol_us, update_us,
        /*timing=*/false);

    EXPECT_EQ(info, 1);
}

// max_retries = 0 on a rank-deficient (exactly singular) Gram must return potrf's
// own positive breakdown code with no retry, and must leave the caller's R buffer
// untouched (R is written only in Step 4, reached solely on success).
TEST_F(TestCholQRPrimitiveFailurePaths, potrf_exhaustion_max_retries_zero_returns_positive_info_and_leaves_R_untouched) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(211);
    RandBLAS::fill_dense(D, A_data.data(), state);
    for (int64_t i = 0; i < m; ++i) A_data[3 * m + i] = 0.0;  // exact zero column -> singular Gram
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    const double sentinel = -777.0;
    std::vector<double> R(n * n, sentinel);
    long fwd = 0, adj = 0, chol = 0;
    int n_retries = -7;
    double applied_shift = -7.0, gram_trace = -7.0;
    int info = RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, R.data(), n, /*shift_factor=*/0.0, /*block_size=*/0,
        fwd, adj, chol, /*timing=*/false, /*max_retries=*/0, /*shift_growth=*/10.0,
        &n_retries, &applied_shift, &gram_trace);

    EXPECT_GT(info, 0);              // potrf's own positive breakdown code
    EXPECT_EQ(n_retries, 0);         // single unshifted attempt, no retry allowed
    EXPECT_EQ(applied_shift, 0.0);
    EXPECT_GT(gram_trace, 0.0);      // trace is computed before the failed factorization
    for (double v : R) EXPECT_EQ(v, sentinel);
}

// Non-finite bail: an infinite shift_factor makes shift = shift_factor * trace(G)
// non-finite on the very first attempt, so cholqr_primitive must bail with info
// == -1 without ever calling potrf, and must leave n_retries/applied_shift at
// their reset values. A directly non-finite trace(G) is not reachable cheaply
// through the public path (it would need an inf/NaN entry in the operator's own
// data, which DenseLinOp/gen helpers do not expose without hand-poking the
// buffer); shift_factor = inf reaches the same `!std::isfinite(shift)` bail
// (rl_cholqr.hh) on attempt 0, so it is used here instead of fabricating an
// unreachable input.
TEST_F(TestCholQRPrimitiveFailurePaths, nonfinite_shift_bails_with_info_negative_one) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(231);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    const double sentinel = -777.0;
    std::vector<double> R(n * n, sentinel);
    long fwd = 0, adj = 0, chol = 0;
    int n_retries = -7;
    double applied_shift = -7.0, gram_trace = -7.0;
    int info = RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, R.data(), n, /*shift_factor=*/std::numeric_limits<double>::infinity(),
        /*block_size=*/0, fwd, adj, chol, /*timing=*/false, /*max_retries=*/0,
        /*shift_growth=*/10.0, &n_retries, &applied_shift, &gram_trace);

    EXPECT_EQ(info, -1);
    EXPECT_EQ(n_retries, 0);
    EXPECT_EQ(applied_shift, 0.0);   // never applied: the bail precedes G's update
    for (double v : R) EXPECT_EQ(v, sentinel);
}

// The shared m/ldr/R validation is a caller-bug guard, not a runtime
// condition, so it throws RandLAPACK::Error rather than returning a sentinel.
TEST_F(TestCholQRPrimitiveFailurePaths, invalid_input_throws_randlapack_error) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(221);
    RandBLAS::fill_dense(D, A_data.data(), state);
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);
    std::vector<double> R(n * n, 0.0);
    long fwd = 0, adj = 0, chol = 0;

    // m < n: build an operator that is actually short.
    {
        int64_t m_short = 10;
        std::vector<double> A_short(m_short * n);
        RandBLAS::DenseDist D_short(m_short, n);
        RandBLAS::RNGState<> s2(222);
        RandBLAS::fill_dense(D_short, A_short.data(), s2);
        RandLAPACK::linops::DenseLinOp<double> A_short_linop(m_short, n, A_short.data(), m_short, Layout::ColMajor);
        std::vector<double> R_short(n * n, 0.0);
        EXPECT_THROW(
            (RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
                A_short_linop, R_short.data(), n, 0.0, 0, fwd, adj, chol, false)),
            RandLAPACK::Error);
    }
    // ldr < n
    EXPECT_THROW(
        (RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
            A_linop, R.data(), /*ldr=*/n - 1, 0.0, 0, fwd, adj, chol, false)),
        RandLAPACK::Error);
    // R == nullptr
    EXPECT_THROW(
        (RandLAPACK::cholqr_primitive<double, RandLAPACK::linops::DenseLinOp<double>>(
            A_linop, (double*)nullptr, n, 0.0, 0, fwd, adj, chol, false)),
        RandLAPACK::Error);
}

// cholqr_iterate's 1-based failing-pass contract, pass 1: a rank-deficient input
// with max_retries = 0 (no rescue) fails pass 1's unpreconditioned potrf
// deterministically, so cholqr_iterate must return 1, not 0 or some other index.
//
// A pass-2-SPECIFIC failure (pass 1 succeeds, pass 2's preconditioned Gram then
// breaks) was evaluated and is NOT covered here: cholqr_iterate threads a single
// max_retries value through both cholqr_primitive calls, and pass 2's Gram
// R_pre^T (A^T A) R_pre is, for any exact (unshifted) pass-1 Cholesky factor R1,
// algebraically the identity (R_pre = R1^{-1} exactly inverts the same Gram pass 1
// just factored), so it is better-conditioned than pass 1's own Gram for the same
// input, not worse. Making pass 2 fail while pass 1 succeeds therefore needs a
// narrow, input-dependent floating-point coincidence (kappa(A) large enough that
// pass 1's rounding error survives amplification through R_pre without pass 1
// itself already failing under the SAME retry budget) rather than a construction
// that holds for a chosen, deterministic input. Per the dispatch's own guidance,
// that contract is left to the primitive-level failure tests above (which cover
// every one of cholqr_primitive's return causes) plus TestCholQRShiftRecord's
// cholqr2_per_pass_record (which already exercises a real, successful pass 2).
TEST_F(TestCholQRPrimitiveFailurePaths, cholqr_iterate_reports_1_based_pass_1_failure) {
    int64_t m = 60, n = 20;
    std::vector<double> A_data(m * n);
    RandBLAS::DenseDist D(m, n);
    RandBLAS::RNGState<> state(231);
    RandBLAS::fill_dense(D, A_data.data(), state);
    for (int64_t i = 0; i < m; ++i) A_data[2 * m + i] = 0.0;   // exact zero column
    RandLAPACK::linops::DenseLinOp<double> A_linop(m, n, A_data.data(), m, Layout::ColMajor);

    std::vector<double> R(n * n, 0.0);
    int info = RandLAPACK::cholqr_iterate<double, RandLAPACK::linops::DenseLinOp<double>>(
        A_linop, R.data(), n, /*block_size=*/0,
        /*num_iters=*/2, /*shift_iter1=*/0.0, /*shift_iter_rest=*/0.0,
        /*max_retries=*/0, /*shift_growth=*/10.0, /*timing=*/false);

    EXPECT_EQ(info, 1);   // 1-based: pass 1 failed
}
