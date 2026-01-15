// Common utilities for RandLAPACK tests
// Provides helper functions for testing linear operators.
// NOT part of the public API - for test code only.

#pragma once

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <algorithm>
#include <vector>

namespace RandLAPACK {
namespace util {
namespace test {

// ============================================================================
// Dimension Calculation Helpers (for linear operator tests)
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
    T* mat,
    int64_t rows,
    int64_t cols,
    ::blas::Layout from_layout,
    ::blas::Layout to_layout
) {
    if (from_layout == to_layout) return;

    T* temp = new T[rows * cols];
    std::copy(mat, mat + rows * cols, temp);
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
    delete[] temp;
}

}  // namespace test
}  // namespace util
}  // namespace RandLAPACK
