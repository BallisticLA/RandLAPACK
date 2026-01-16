// Comprehensive unit tests for SparseLinOp
// Tests all combinations of Side, Layout, trans_A, trans_B with both dense and sparse B matrices

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
using blas::Side;
using RandBLAS::DenseDist;
using RandBLAS::RNGState;
using namespace RandLAPACK::util;

class TestSparseLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    /// Unified test function for SparseLinOp.
    ///
    /// This test verifies that SparseLinOp correctly computes matrix products of the form
    /// C := alpha * op(A) * op(B) + beta * C (side=Left) or
    /// C := alpha * op(B) * op(A) + beta * C (side=Right), where A is a sparse matrix
    /// stored in CSC format and wrapped as a SparseLinOp.
    ///
    /// Test structure:
    ///
    /// 1. MATRIX GENERATION
    ///    - Generate random sparse matrix A in CSC format with specified density.
    ///    - Generate random matrix B, either dense or sparse (CSC format) based on sparse_B flag.
    ///    - Initialize output matrix C with random values (to test beta != 0 case).
    ///
    /// 2. LINEAR OPERATOR CONSTRUCTION
    ///    - Wrap A in a SparseLinOp<CSCMatrix> object, which stores a reference to the CSC data.
    ///
    /// 3. COMPUTATION VIA LINEAR OPERATOR
    ///    - Apply the SparseLinOp to compute:
    ///        Side::Left:  C_sparse_op := alpha * op(A) * op(B) + beta * C_sparse_op
    ///        Side::Right: C_sparse_op := alpha * op(B) * op(A) + beta * C_sparse_op
    ///    - The operator handles both dense and sparse B inputs via overloaded operator().
    ///    - Internally uses RandBLAS sparse matrix-matrix multiplication (spmm).
    ///
    /// 4. REFERENCE COMPUTATION
    ///    - Convert A to dense format.
    ///    - If B is sparse, convert it to dense format as well.
    ///    - Compute C_reference using BLAS gemm directly on dense matrices.
    ///
    /// 5. VERIFICATION
    ///    - Compare C_sparse_op and C_reference entry-wise.
    ///    - Uses relaxed tolerance (atol = 100 * eps, rtol = 10 * eps) to account for
    ///      potential differences in sparse vs dense computation order.
    ///
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
        auto A_csc = RandLAPACK::gen::gen_sparse_csc<T>(dims.rows_A, dims.cols_A, density_A, state);

        // Create and initialize output buffers with random data for beta testing
        T* C_sparse_op = new T[m * n];
        RandLAPACK::gen::gen_random_dense(m, n, C_sparse_op, Layout::ColMajor, state);
        T* C_reference = new T[m * n];
        std::copy(C_sparse_op, C_sparse_op + m * n, C_reference);

        // Create the SparseLinOp operator
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(dims.rows_A, dims.cols_A, A_csc);

        if (sparse_B) {
            // Generate sparse matrix B using utility function
            auto B_csc = RandLAPACK::gen::gen_sparse_csc<T>(dims.rows_B, dims.cols_B, density_B, state);

            // Compute using SparseLinOp with sparse B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_sparse_op, dims.ldc);

            // Compute reference: densify both matrices and use BLAS GEMM
            T* A_dense = new T[dims.rows_A * dims.cols_A]();
            T* B_dense = new T[dims.rows_B * dims.cols_B]();

            RandBLAS::sparse_data::csc::csc_to_dense(A_csc, layout, A_dense);
            RandBLAS::sparse_data::csc::csc_to_dense(B_csc, layout, B_dense);

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
            // Generate dense matrix B
            T* B_dense = new T[dims.rows_B * dims.cols_B];
            RandLAPACK::gen::gen_random_dense(dims.rows_B, dims.cols_B, B_dense, layout, state);

            // Compute using SparseLinOp with dense B
            A_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, dims.ldb, beta, C_sparse_op, dims.ldc);

            // Compute reference using dense GEMM
            // Densify A
            T* A_dense = new T[dims.rows_A * dims.cols_A]();
            RandBLAS::sparse_data::csc::csc_to_dense(A_csc, layout, A_dense);

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