// Common utilities for linear operator tests
// Provides helper functions to reduce code duplication across linop tests

#pragma once

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <vector>

namespace RandLAPACK {
namespace util {
namespace test {

// ============================================================================
// Matrix Generation Helpers
// ============================================================================

/// Generate a random dense matrix with the specified layout
/// Note: RandBLAS::fill_dense always generates ColMajor, so we convert if needed
template <typename T>
::std::vector<T> generate_dense_matrix(
    int64_t rows,
    int64_t cols,
    ::blas::Layout layout,
    ::RandBLAS::RNGState<>& state
) {
    ::std::vector<T> mat(rows * cols);
    ::RandBLAS::DenseDist D(rows, cols);
    ::RandBLAS::fill_dense(D, mat.data(), state);

    // Convert to RowMajor if needed (RandBLAS generates ColMajor)
    if (layout == ::blas::Layout::RowMajor) {
        ::std::vector<T> temp = mat;
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                mat[j + i * cols] = temp[i + j * rows];
            }
        }
    }

    return mat;
}

/// Generate a random sparse matrix in CSC format
template <typename T>
::RandBLAS::sparse_data::CSCMatrix<T> generate_sparse_matrix(
    int64_t rows,
    int64_t cols,
    T density,
    ::RandBLAS::RNGState<>& state
) {
    auto coo = ::RandLAPACK::gen::gen_sparse_mat<T>(rows, cols, density, state);
    ::RandBLAS::sparse_data::CSCMatrix<T> csc(rows, cols);
    ::RandBLAS::sparse_data::conversions::coo_to_csc(coo, csc);
    return csc;
}

// ============================================================================
// Dimension Calculation Helpers
// ============================================================================

/// Structure to hold matrix dimensions and leading dimensions
struct MatrixDimensions {
    int64_t rows_A, cols_A;  // Operator matrix dimensions
    int64_t rows_B, cols_B;  // Input matrix dimensions
    int64_t lda, ldb, ldc;   // Leading dimensions
};

/// Calculate all dimensions for a linear operator multiplication
/// Handles both Side::Left and Side::Right cases
template <typename T>
MatrixDimensions calculate_dimensions(
    ::blas::Side side,
    ::blas::Layout layout,
    ::blas::Op trans_A,
    ::blas::Op trans_B,
    int64_t m,
    int64_t n,
    int64_t k
) {
    MatrixDimensions dims;

    if (side == ::blas::Side::Left) {
        // Side::Left: C := alpha * op(A) * op(B) + beta * C
        // A is the operator (m × k), B is the input (k × n)
        auto [ra, ca] = ::RandBLAS::dims_before_op(m, k, trans_A);
        auto [rb, cb] = ::RandBLAS::dims_before_op(k, n, trans_B);
        dims.rows_A = ra;
        dims.cols_A = ca;
        dims.rows_B = rb;
        dims.cols_B = cb;
    } else {
        // Side::Right: C := alpha * op(B) * op(A) + beta * C
        // A is the operator (k × n), B is the input (m × k)
        auto [ra, ca] = ::RandBLAS::dims_before_op(k, n, trans_A);
        auto [rb, cb] = ::RandBLAS::dims_before_op(m, k, trans_B);
        dims.rows_A = ra;
        dims.cols_A = ca;
        dims.rows_B = rb;
        dims.cols_B = cb;
    }

    // Calculate leading dimensions based on layout
    if (layout == ::blas::Layout::ColMajor) {
        dims.lda = dims.rows_A;
        dims.ldb = dims.rows_B;
        dims.ldc = m;
    } else {  // RowMajor
        dims.lda = dims.cols_A;
        dims.ldb = dims.cols_B;
        dims.ldc = n;
    }

    return dims;
}

// ============================================================================
// Reference Computation Helpers
// ============================================================================

/// Compute reference result using BLAS GEMM
/// Handles both Side::Left and Side::Right cases
template <typename T>
void compute_gemm_reference(
    ::blas::Side side,
    ::blas::Layout layout,
    ::blas::Op trans_A,
    ::blas::Op trans_B,
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
    if (side == ::blas::Side::Left) {
        // C := alpha * op(A) * op(B) + beta * C
        ::blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        // C := alpha * op(B) * op(A) + beta * C
        ::blas::gemm(layout, trans_B, trans_A, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
    }
}

// ============================================================================
// Layout Conversion Helper
// ============================================================================

/// Convert matrix layout in-place
/// Note: This assumes the matrix is currently in 'from_layout' and converts to 'to_layout'
template <typename T>
void convert_layout_inplace(
    ::std::vector<T>& mat,
    int64_t rows,
    int64_t cols,
    ::blas::Layout from_layout,
    ::blas::Layout to_layout
) {
    if (from_layout == to_layout) return;

    ::std::vector<T> temp = mat;
    if (from_layout == ::blas::Layout::ColMajor && to_layout == ::blas::Layout::RowMajor) {
        // ColMajor to RowMajor
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                mat[j + i * cols] = temp[i + j * rows];
            }
        }
    } else {
        // RowMajor to ColMajor
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                mat[i + j * rows] = temp[j + i * cols];
            }
        }
    }
}

// ============================================================================
// Test Data Initialization Helper
// ============================================================================

/// Initialize test output buffers with random data for beta testing
template <typename T>
void initialize_test_buffers(::std::vector<T>& C_test, ::std::vector<T>& C_reference) {
    for (auto& c : C_test) {
        c = static_cast<T>(rand()) / RAND_MAX;
    }
    C_reference = C_test;
}

// ============================================================================
// Linear Operator Materialization and Analysis
// ============================================================================

/// Materialize a linear operator into dense column-major storage.
///
/// Computes A_dense = A_linop * I by applying the operator to the identity matrix.
///
/// @tparam T - Scalar type (double, float, etc.)
/// @tparam LinOp - Linear operator type supporting operator() with BLAS calling convention
///
/// @param[in] A_linop - The linear operator to materialize
/// @param[in] m - Number of rows
/// @param[in] n - Number of columns
///
/// @return Dense m x n column-major matrix
///
template <typename T, typename LinOp>
::std::vector<T> materialize_linop(LinOp& A_linop, int64_t m, int64_t n) {
    ::std::vector<T> A_dense(m * n, 0.0);
    ::std::vector<T> Eye(n * n, 0.0);
    ::RandLAPACK::util::eye(n, n, Eye.data());
    A_linop(::blas::Side::Left, ::blas::Layout::ColMajor, ::blas::Op::NoTrans, ::blas::Op::NoTrans,
            m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_dense.data(), m);
    return A_dense;
}

/// Compute singular values of a dense column-major matrix via gesdd.
///
/// Note: the input matrix is destroyed (overwritten) by gesdd.
///
/// @tparam T - Scalar type (double, float, etc.)
///
/// @param[in,out] A_dense - Dense m x n matrix (destroyed on output)
/// @param[in] m - Number of rows
/// @param[in] n - Number of columns
///
/// @return Vector of n singular values in decreasing order
///
template <typename T>
::std::vector<T> compute_singular_values(T* A_dense, int64_t m, int64_t n) {
    ::std::vector<T> sigma(n);
    ::lapack::gesdd(::lapack::Job::NoVec,
                    m, n, A_dense, m, sigma.data(),
                    nullptr, 1, nullptr, 1);
    return sigma;
}

}  // namespace test
}  // namespace util
}  // namespace RandLAPACK
