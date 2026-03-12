// Testing utilities for RandLAPACK tests
//
// This header provides helper functions for verifying correctness of linear
// operator implementations. It is not part of the public RandLAPACK API and
// should only be used in test code.
//
// Note: Most of these utilities exist in the RandBLAS tests/ folder as well
// but are not packaged with a RandBLAS installation. Once RandBLAS exposes
// official verification utilities, these should be replaced accordingly.
// See: https://github.com/BallisticLA/RandLAPACK/issues/121

#pragma once

#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <vector>

namespace RandLAPACK {
namespace testing {

// ============================================================================
// Matrix Generation Helpers
// ============================================================================

/// Generate a random dense matrix with the specified layout.
template <typename T>
::std::vector<T> generate_dense_matrix(
    int64_t rows,
    int64_t cols,
    ::blas::Layout layout,
    ::RandBLAS::RNGState<>& state
) {
    ::std::vector<T> mat(rows * cols);
    ::RandBLAS::DenseDist D(rows, cols);
    state = ::RandBLAS::fill_dense_unpacked(layout, D, rows, cols, 0, 0, mat.data(), state);
    return mat;
}

// ============================================================================
// Dimension Calculation Helpers
// ============================================================================

/// Dimensions and leading dimensions for a linear operator multiplication.
struct MatmulDimensions {
    int64_t rows_A, cols_A;  // Operator matrix dimensions
    int64_t rows_B, cols_B;  // Input matrix dimensions
    int64_t lda, ldb, ldc;   // Leading dimensions
};

/// Calculate all dimensions for a linear operator multiplication.
/// Handles both Side::Left (C = op(A)*op(B)) and Side::Right (C = op(B)*op(A)).
template <typename T>
MatmulDimensions calculate_dimensions(
    ::blas::Side side,
    ::blas::Layout layout,
    ::blas::Op trans_A,
    ::blas::Op trans_B,
    int64_t m,
    int64_t n,
    int64_t k
) {
    MatmulDimensions dims;

    if (side == ::blas::Side::Left) {
        // A is the operator (m × k), B is the input (k × n)
        auto [ra, ca] = ::RandBLAS::dims_before_op(m, k, trans_A);
        auto [rb, cb] = ::RandBLAS::dims_before_op(k, n, trans_B);
        dims.rows_A = ra; dims.cols_A = ca;
        dims.rows_B = rb; dims.cols_B = cb;
    } else {
        // A is the operator (k × n), B is the input (m × k)
        auto [ra, ca] = ::RandBLAS::dims_before_op(k, n, trans_A);
        auto [rb, cb] = ::RandBLAS::dims_before_op(m, k, trans_B);
        dims.rows_A = ra; dims.cols_A = ca;
        dims.rows_B = rb; dims.cols_B = cb;
    }

    if (layout == ::blas::Layout::ColMajor) {
        dims.lda = dims.rows_A;
        dims.ldb = dims.rows_B;
        dims.ldc = m;
    } else {
        dims.lda = dims.cols_A;
        dims.ldb = dims.cols_B;
        dims.ldc = n;
    }

    return dims;
}

// ============================================================================
// Reference Computation Helpers
// ============================================================================

/// Compute a GEMM reference result.
/// Side::Left:  C = alpha * op(A) * op(B) + beta * C
/// Side::Right: C = alpha * op(B) * op(A) + beta * C
template <typename T>
void sided_gemm(
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
        ::blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        ::blas::gemm(layout, trans_B, trans_A, m, n, k, alpha, B, ldb, A, lda, beta, C, ldc);
    }
}

// ============================================================================
// Layout Conversion Helper
// ============================================================================

/// Convert matrix layout in-place between ColMajor and RowMajor.
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
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j)
                mat[j + i * cols] = temp[i + j * rows];
    } else {
        for (int64_t i = 0; i < rows; ++i)
            for (int64_t j = 0; j < cols; ++j)
                mat[i + j * rows] = temp[j + i * cols];
    }
}

// ============================================================================
// Test Data Initialization
// ============================================================================

/// Initialize output buffers with random data for beta != 0 tests.
template <typename T>
void initialize_test_buffers(::std::vector<T>& C_test, ::std::vector<T>& C_reference) {
    int64_t len = static_cast<int64_t>(C_test.size());
    ::RandBLAS::DenseDist D(len, 1);
    ::RandBLAS::RNGState<> seed(42);
    ::RandBLAS::fill_dense(D, C_test.data(), seed);
    C_reference = C_test;
}

// ============================================================================
// Linear Operator Materialization and Analysis
// ============================================================================

/// Materialize a linear operator into a dense column-major matrix.
/// Computes A_dense = A_linop * I by applying the operator to the identity.
///
/// WARNING: This function uses operator() to materialize, which makes it
/// unsuitable for testing operator() itself (circular reasoning). For
/// correctness tests of individual linop types, prefer type-specific
/// materialization:
///   - DenseLinOp:   copy A_buff directly
///   - SparseLinOp:  use RandLAPACK::util::sparse_to_dense on A_sp
///   - CompositeOperator: materialize each operand independently, then
///     compute the product with blas::gemm
/// This function is appropriate for tests that assume operator() is correct
/// and need a dense representation for some other purpose (e.g., computing
/// singular values, comparing block views against the full operator).
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
/// Note: input matrix is destroyed on output.
template <typename T>
::std::vector<T> compute_singular_values(T* A_dense, int64_t m, int64_t n) {
    ::std::vector<T> sigma(n);
    int64_t info = ::lapack::gesdd(::lapack::Job::NoVec,
                    m, n, A_dense, m, sigma.data(),
                    nullptr, 1, nullptr, 1);
    randblas_require(info == 0);
    return sigma;
}

}  // namespace testing
}  // namespace RandLAPACK
