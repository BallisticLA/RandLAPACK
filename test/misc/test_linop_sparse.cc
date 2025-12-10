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
using RandBLAS::SparseDist;
using RandBLAS::RNGState;

class TestSparseLinOp : public ::testing::Test {

protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    // Helper: Generate sparse matrix in CSC format
    template <typename T>
    static RandBLAS::sparse_data::csc::CSCMatrix<T> generate_sparse_csc(
        int64_t m, int64_t n, T density, RNGState<RandBLAS::r123::Philox4x32>& state
    ) {
        auto coo = RandLAPACK::gen::gen_sparse_mat<T>(m, n, density, state);
        RandBLAS::sparse_data::csc::CSCMatrix<T> csc(m, n);
        RandBLAS::sparse_data::conversions::coo_to_csc(coo, csc);
        return csc;
    }

    // Helper: Convert sparse CSC to dense
    template <typename T>
    static void sparse_csc_to_dense(
        const RandBLAS::sparse_data::csc::CSCMatrix<T>& A_csc,
        Layout layout,
        T* A_dense,
        int64_t lda
    ) {
        int64_t m = A_csc.n_rows;
        int64_t n = A_csc.n_cols;

        // Initialize to zero
        for (int64_t i = 0; i < m * n; ++i) A_dense[i] = 0.0;

        // Fill in nonzeros
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t idx = A_csc.colptr[j]; idx < A_csc.colptr[j+1]; ++idx) {
                int64_t i = A_csc.rowidxs[idx];
                T val = A_csc.vals[idx];
                if (layout == Layout::ColMajor) {
                    A_dense[i + j * lda] = val;
                } else {  // RowMajor
                    A_dense[j + i * lda] = val;
                }
            }
        }
    }

    // Helper: Compute reference result using dense GEMM
    template <typename T>
    static void compute_reference_gemm(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        const T* A,
        int64_t lda,
        const T* B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    // Test Side::Left with dense B
    template <typename T>
    void test_sparse_left_dense_B(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Generate sparse matrix A in CSC format
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
        auto A_csc = generate_sparse_csc<T>(rows_A, cols_A, density, state);

        // Generate dense matrix B
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        vector<T> B_dense(rows_B * cols_B);
        RandBLAS::DenseDist D_B(rows_B, cols_B);
        RandBLAS::fill_dense(D_B, B_dense.data(), state);
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

        // Create output buffers
        vector<T> C_sparse_op(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        // Initialize C with random data (to test beta scaling)
        for (auto& c : C_sparse_op) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_sparse_op;

        // Compute using SparseLinOp
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(rows_A, cols_A, A_csc);
        A_op(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), ldb, beta, C_sparse_op.data(), ldc);

        // Compute reference using dense GEMM
        vector<T> A_dense(rows_A * cols_A);
        int64_t lda = (layout == Layout::ColMajor) ? rows_A : cols_A;
        sparse_csc_to_dense(A_csc, layout, A_dense.data(), lda);
        compute_reference_gemm(layout, trans_A, trans_B, m, n, k, alpha,
                             A_dense.data(), lda, B_dense.data(), ldb, beta, C_reference.data(), ldc);

        // Compare results
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op.data(), m,
            C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    // Test Side::Right with dense B
    template <typename T>
    void test_sparse_right_dense_B(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Generate sparse matrix A in CSC format
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
        auto A_csc = generate_sparse_csc<T>(rows_A, cols_A, density, state);

        // Generate dense matrix B
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
        vector<T> B_dense(rows_B * cols_B);
        RandBLAS::DenseDist D_B(rows_B, cols_B);
        RandBLAS::fill_dense(D_B, B_dense.data(), state);
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

        // Create output buffers
        vector<T> C_sparse_op(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        // Initialize C with random data (to test beta scaling)
        for (auto& c : C_sparse_op) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_sparse_op;

        // Compute using SparseLinOp
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(rows_A, cols_A, A_csc);
        A_op(Side::Right, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), ldb, beta, C_sparse_op.data(), ldc);

        // Compute reference using dense GEMM
        vector<T> A_dense(rows_A * cols_A);
        int64_t lda = (layout == Layout::ColMajor) ? rows_A : cols_A;
        sparse_csc_to_dense(A_csc, layout, A_dense.data(), lda);
        compute_reference_gemm(layout, trans_B, trans_A, m, n, k, alpha,
                             B_dense.data(), ldb, A_dense.data(), lda, beta, C_reference.data(), ldc);

        // Compare results
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op.data(), m,
            C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    // Test Side::Left with sparse B
    template <typename T>
    void test_sparse_left_sparse_B(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density_A,
        T density_B
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Generate sparse matrices A and B in CSC format
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
        auto A_csc = generate_sparse_csc<T>(rows_A, cols_A, density_A, state);

        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        auto B_csc = generate_sparse_csc<T>(rows_B, cols_B, density_B, state);

        // Create output buffers
        vector<T> C_sparse_op(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        // Initialize C with random data (to test beta scaling)
        for (auto& c : C_sparse_op) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_sparse_op;

        // Compute using SparseLinOp
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(rows_A, cols_A, A_csc);
        A_op(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_sparse_op.data(), ldc);

        // Compute reference using dense GEMM
        vector<T> A_dense(rows_A * cols_A);
        vector<T> B_dense(rows_B * cols_B);
        int64_t lda = (layout == Layout::ColMajor) ? rows_A : cols_A;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        sparse_csc_to_dense(A_csc, layout, A_dense.data(), lda);
        sparse_csc_to_dense(B_csc, layout, B_dense.data(), ldb);
        compute_reference_gemm(layout, trans_A, trans_B, m, n, k, alpha,
                             A_dense.data(), lda, B_dense.data(), ldb, beta, C_reference.data(), ldc);

        // Compare results
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op.data(), m,
            C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }

    // Test Side::Right with sparse B
    template <typename T>
    void test_sparse_right_sparse_B(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T density_A,
        T density_B
    ) {
        RNGState state(0);
        T alpha = 1.5;
        T beta = 0.5;

        // Generate sparse matrices A and B in CSC format
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
        auto A_csc = generate_sparse_csc<T>(rows_A, cols_A, density_A, state);

        auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
        auto B_csc = generate_sparse_csc<T>(rows_B, cols_B, density_B, state);

        // Create output buffers
        vector<T> C_sparse_op(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        // Initialize C with random data (to test beta scaling)
        for (auto& c : C_sparse_op) c = static_cast<T>(rand()) / RAND_MAX;
        C_reference = C_sparse_op;

        // Compute using SparseLinOp
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csc::CSCMatrix<T>>
            A_op(rows_A, cols_A, A_csc);
        A_op(Side::Right, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_sparse_op.data(), ldc);

        // Compute reference using dense GEMM
        vector<T> A_dense(rows_A * cols_A);
        vector<T> B_dense(rows_B * cols_B);
        int64_t lda = (layout == Layout::ColMajor) ? rows_A : cols_A;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        sparse_csc_to_dense(A_csc, layout, A_dense.data(), lda);
        sparse_csc_to_dense(B_csc, layout, B_dense.data(), ldb);
        compute_reference_gemm(layout, trans_B, trans_A, m, n, k, alpha,
                             B_dense.data(), ldb, A_dense.data(), lda, beta, C_reference.data(), ldc);

        // Compare results
        test::comparison::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_sparse_op.data(), m,
            C_reference.data(), m, __PRETTY_FUNCTION__, __FILE__, __LINE__
        );
    }
};

// ============================================================================
// Side::Left, Dense B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_dense_colmajor_notrans_notrans) {
    test_sparse_left_dense_B<double>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_notrans_trans) {
    test_sparse_left_dense_B<double>(Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_trans_notrans) {
    test_sparse_left_dense_B<double>(Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_colmajor_trans_trans) {
    test_sparse_left_dense_B<double>(Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left, Dense B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_dense_rowmajor_notrans_notrans) {
    test_sparse_left_dense_B<double>(Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_notrans_trans) {
    test_sparse_left_dense_B<double>(Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_trans_notrans) {
    test_sparse_left_dense_B<double>(Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, left_dense_rowmajor_trans_trans) {
    test_sparse_left_dense_B<double>(Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right, Dense B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_dense_colmajor_notrans_notrans) {
    test_sparse_right_dense_B<double>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_notrans_trans) {
    test_sparse_right_dense_B<double>(Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_trans_notrans) {
    test_sparse_right_dense_B<double>(Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_colmajor_trans_trans) {
    test_sparse_right_dense_B<double>(Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Right, Dense B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_dense_rowmajor_notrans_notrans) {
    test_sparse_right_dense_B<double>(Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_notrans_trans) {
    test_sparse_right_dense_B<double>(Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_trans_notrans) {
    test_sparse_right_dense_B<double>(Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3);
}

TEST_F(TestSparseLinOp, right_dense_rowmajor_trans_trans) {
    test_sparse_right_dense_B<double>(Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3);
}

// ============================================================================
// Side::Left, Sparse B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_sparse_colmajor_notrans_notrans) {
    test_sparse_left_sparse_B<double>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_notrans_trans) {
    test_sparse_left_sparse_B<double>(Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_trans_notrans) {
    test_sparse_left_sparse_B<double>(Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_colmajor_trans_trans) {
    test_sparse_left_sparse_B<double>(Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Left, Sparse B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, left_sparse_rowmajor_notrans_notrans) {
    test_sparse_left_sparse_B<double>(Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_notrans_trans) {
    test_sparse_left_sparse_B<double>(Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_trans_notrans) {
    test_sparse_left_sparse_B<double>(Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, left_sparse_rowmajor_trans_trans) {
    test_sparse_left_sparse_B<double>(Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Right, Sparse B, ColMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_sparse_colmajor_notrans_notrans) {
    test_sparse_right_sparse_B<double>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_notrans_trans) {
    test_sparse_right_sparse_B<double>(Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_trans_notrans) {
    test_sparse_right_sparse_B<double>(Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_colmajor_trans_trans) {
    test_sparse_right_sparse_B<double>(Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

// ============================================================================
// Side::Right, Sparse B, RowMajor
// ============================================================================

TEST_F(TestSparseLinOp, right_sparse_rowmajor_notrans_notrans) {
    test_sparse_right_sparse_B<double>(Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_notrans_trans) {
    test_sparse_right_sparse_B<double>(Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_trans_notrans) {
    test_sparse_right_sparse_B<double>(Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 12, 0.3, 0.25);
}

TEST_F(TestSparseLinOp, right_sparse_rowmajor_trans_trans) {
    test_sparse_right_sparse_B<double>(Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 12, 0.3, 0.25);
}
