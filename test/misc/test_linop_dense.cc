// Unit tests for DenseLinOp
// Tests basic dense matrix linear operator functionality

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"
#include "../../RandLAPACK/misc/rl_util_test_linop.hh"

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::util::test;

class TestDenseLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Unified test function for DenseLinOp
    // Handles both Side::Left and Side::Right, both dense and sparse B, both ColMajor and RowMajor
    template <typename T>
    void test_dense_linop(
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density = 0.0  // Only used if sparse_B == true
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Calculate dimensions using utility function
        auto dims = calculate_dimensions<T>(side, layout, trans_A, trans_B, m, n, k);

        // Generate dense matrix A using utility function
        vector<T> A_dense = generate_dense_matrix<T>(dims.rows_A, dims.cols_A, layout, state);

        // Create output buffers
        vector<T> C_dense_op(m * n);
        vector<T> C_reference(m * n);

        // Initialize test buffers using utility function
        initialize_test_buffers(C_dense_op, C_reference);

        // Create the DenseLinOp operator
        RandLAPACK::linops::DenseLinOp<T> A_op(dims.rows_A, dims.cols_A, A_dense.data(), dims.lda, layout);

        if (sparse_B) {
            // Generate sparse matrix B using utility function
            auto B_csc = generate_sparse_matrix<T>(dims.rows_B, dims.cols_B, density, state);

            // Compute using DenseLinOp with sparse B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_dense_op.data(), dims.ldc);

            // Compute reference: densify B and use BLAS GEMM
            // NOTE: gen_sparse_mat can generate duplicates, so we must SUM them, not overwrite
            vector<T> B_dense(dims.rows_B * dims.cols_B);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, layout, B_dense.data());

            // Compute reference using utility function
            compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                                   A_dense.data(), dims.lda, B_dense.data(), dims.ldb,
                                   beta, C_reference.data(), dims.ldc);

            // Compare results with relaxed tolerance for sparse operations
            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_dense_op.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );
        } else {
            // Generate dense matrix B using utility function
            vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

            // Compute using DenseLinOp with dense B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_dense_op.data(), dims.ldc);

            // Compute reference using utility function
            compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                                   A_dense.data(), dims.lda, B_dense.data(), dims.ldb,
                                   beta, C_reference.data(), dims.ldc);

            // Compare results with standard tolerance
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_dense_op.data(), m,
                C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__
            );
        }
    }
};

// ============================================================================
// Side::Left with dense B - ColMajor
// ============================================================================

TEST_F(TestDenseLinOp, left_dense_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_dense_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_dense_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_dense_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// Side::Left with dense B - RowMajor
// ============================================================================

TEST_F(TestDenseLinOp, left_dense_rowmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_dense_rowmajor_notrans_trans) {
    test_dense_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_dense_rowmajor_trans_notrans) {
    test_dense_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, left_dense_rowmajor_trans_trans) {
    test_dense_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// Side::Right with dense B - ColMajor
// ============================================================================

TEST_F(TestDenseLinOp, right_dense_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_dense_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_dense_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_dense_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// Side::Right with dense B - RowMajor
// ============================================================================

TEST_F(TestDenseLinOp, right_dense_rowmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_dense_rowmajor_notrans_trans) {
    test_dense_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_dense_rowmajor_trans_notrans) {
    test_dense_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12);
}

TEST_F(TestDenseLinOp, right_dense_rowmajor_trans_trans) {
    test_dense_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12);
}

// ============================================================================
// Side::Left with sparse B - ColMajor
// ============================================================================

TEST_F(TestDenseLinOp, left_sparse_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left with sparse B - RowMajor
// ============================================================================

TEST_F(TestDenseLinOp, left_sparse_rowmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_rowmajor_notrans_trans) {
    test_dense_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_rowmajor_trans_notrans) {
    test_dense_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, left_sparse_rowmajor_trans_trans) {
    test_dense_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with sparse B - ColMajor
// ============================================================================

TEST_F(TestDenseLinOp, right_sparse_colmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_colmajor_notrans_trans) {
    test_dense_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_colmajor_trans_notrans) {
    test_dense_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_colmajor_trans_trans) {
    test_dense_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with sparse B - RowMajor
// ============================================================================

TEST_F(TestDenseLinOp, right_sparse_rowmajor_notrans_notrans) {
    test_dense_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_rowmajor_notrans_trans) {
    test_dense_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_rowmajor_trans_notrans) {
    test_dense_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestDenseLinOp, right_sparse_rowmajor_trans_trans) {
    test_dense_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}
