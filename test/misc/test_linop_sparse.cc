// Comprehensive unit tests for SparseLinOp
// Tests all combinations of Side, Layout, trans_A, trans_B with both dense and sparse B matrices

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include "../RandLAPACK/RandBLAS/test/comparison.hh"
#include "../../RandLAPACK/misc/rl_util_test_linop.hh"  // Test utilities, not part of public API

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::util::test;

class TestSparseLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Unified test function for SparseLinOp
    // Handles both Side::Left and Side::Right, both dense and sparse B, and both ColMajor and RowMajor layouts
    template <typename T>
    void test_sparse_linop(
        Side side,
        bool sparse_B,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density_A,
        T density_B = 0.0  // Only used if sparse_B == true
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Calculate dimensions using utility function
        auto dims = calculate_dimensions<T>(side, layout, trans_A, trans_B, m, n, k);

        // Generate sparse matrix A using utility function
        auto A_csc = generate_sparse_matrix<T>(dims.rows_A, dims.cols_A, density_A, state);

        // Create and initialize output buffers with random data for beta testing
        T* C_sparse_op = generate_dense_matrix<T>(m, n, Layout::ColMajor, state);
        T* C_reference = new T[m * n];
        std::copy(C_sparse_op, C_sparse_op + m * n, C_reference);

        // Create the SparseLinOp operator
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(dims.rows_A, dims.cols_A, A_csc);

        if (sparse_B) {
            // Generate sparse matrix B using utility function
            auto B_csc = generate_sparse_matrix<T>(dims.rows_B, dims.cols_B, density_B, state);

            // Compute using SparseLinOp with sparse B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_sparse_op, dims.ldc);

            // Compute reference: densify both matrices and use BLAS GEMM
            // NOTE: gen_sparse_mat can generate duplicates, so we must SUM them, not overwrite
            T* A_dense = new T[dims.rows_A * dims.cols_A];
            T* B_dense = new T[dims.rows_B * dims.cols_B];

            RandLAPACK::util::sparse_to_dense_summing_duplicates(A_csc, layout, A_dense);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, layout, B_dense);

            // Compute reference using utility function
            compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                                   A_dense, dims.lda, B_dense, dims.ldb,
                                   beta, C_reference, dims.ldc);

            // Compare results with relaxed tolerance for sparse operations
            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op, m,
                C_reference, m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );

            delete[] A_dense;
            delete[] B_dense;
        } else {
            // Generate dense matrix B using utility function
            T* B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

            // Compute using SparseLinOp with dense B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, dims.ldb, beta, C_sparse_op, dims.ldc);

            // Compute reference using dense GEMM
            // Densify A
            T* A_dense = new T[dims.rows_A * dims.cols_A];
            RandLAPACK::util::sparse_to_dense_summing_duplicates(A_csc, layout, A_dense);

            // Compute reference using utility function
            compute_gemm_reference(side, layout, trans_A, trans_B, m, n, k, alpha,
                                   A_dense, dims.lda, B_dense, dims.ldb,
                                   beta, C_reference, dims.ldc);

            // Compare results with relaxed tolerance for sparse operations
            T atol = 100 * std::numeric_limits<T>::epsilon();
            T rtol = 10 * std::numeric_limits<T>::epsilon();
            test::comparison::matrices_approx_equal(
                Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op, m,
                C_reference, m, __PRETTY_FUNCTION__, __FILE__, __LINE__,
                atol, rtol
            );

            delete[] A_dense;
            delete[] B_dense;
        }

        // Clean up
        delete[] C_sparse_op;
        delete[] C_reference;
    }
};

// ============================================================================
// Side::Left with dense B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_dense_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left with dense B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_dense_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with dense B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_dense_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right with dense B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_dense_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left with sparse B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_sparse_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Left with sparse B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_sparse_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Right with sparse B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_sparse_colmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Right with sparse B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_sparse_rowmajor_notrans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_notrans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_trans_notrans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_trans_trans) {
    test_sparse_linop<double>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}