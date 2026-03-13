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
#include <fstream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <cmath>

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
    ::std::vector<T> A_dense(m * n, (T)0.0);
    ::RandLAPACK::materialize(A_linop, m, n, A_dense.data(), m);
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

// ============================================================================
// QR Factorization Verification
// ============================================================================

/// Verify a QR factorization: checks both factorization accuracy and
/// orthogonality of Q.  All matrices are column-major with leading
/// dimension equal to their row count (except R, whose leading dimension
/// is ldr).
///
/// Returns {factorization_error, orthogonality_error} where:
///   factorization_error = ||A - Q*R||_F / ||A||_F
///   orthogonality_error = ||Q^T Q - I||_F / sqrt(n)
template <typename T>
::std::pair<T, T> verify_qr(const T* A, const T* Q, const T* R,
                              int64_t m, int64_t n, int64_t ldr) {
    // ||A - Q*R|| / ||A||
    ::std::vector<T> QR(m * n, 0.0);
    ::blas::gemm(::blas::Layout::ColMajor, ::blas::Op::NoTrans, ::blas::Op::NoTrans,
                 m, n, n, (T)1.0, Q, m, R, ldr, (T)0.0, QR.data(), m);
    for (int64_t i = 0; i < m * n; ++i)
        QR[i] = A[i] - QR[i];
    T norm_AQR = ::lapack::lange(::lapack::Norm::Fro, m, n, QR.data(), m);
    T norm_A   = ::lapack::lange(::lapack::Norm::Fro, m, n, A, m);

    // ||Q^T Q - I|| / sqrt(n)
    ::std::vector<T> I_ref(n * n);
    RandLAPACK::util::eye(n, n, I_ref.data());
    ::blas::syrk(::blas::Layout::ColMajor, ::blas::Uplo::Upper, ::blas::Op::Trans,
                 n, m, (T)1.0, Q, m, (T)-1.0, I_ref.data(), n);
    T norm_orth = ::lapack::lansy(::lapack::Norm::Fro, ::blas::Uplo::Upper, n, I_ref.data(), n);

    return {norm_AQR / norm_A, norm_orth / ::std::sqrt((T)n)};
}

/// Verify an R-factor only (when Q is not explicitly available).
/// Recovers Q = A * R^{-1} via TRSM, then checks both metrics.
template <typename T>
::std::pair<T, T> verify_R_factor(const T* A_data, int64_t m, int64_t n,
                                    const T* R, int64_t ldr) {
    ::std::vector<T> Q(m * n);
    ::std::copy(A_data, A_data + m * n, Q.begin());
    ::blas::trsm(::blas::Layout::ColMajor, ::blas::Side::Right, ::blas::Uplo::Upper,
                 ::blas::Op::NoTrans, ::blas::Diag::NonUnit, m, n, (T)1.0, R, ldr, Q.data(), m);
    return verify_qr(A_data, Q.data(), R, m, n, ldr);
}

// ============================================================================
// Condition Number Diagnostics
// ============================================================================

/// Materialize a linear operator and print condition number diagnostics,
/// both raw and after column normalization.
template <typename T, typename LinOp>
void print_condition_diagnostics(LinOp& A_linop, int64_t m, int64_t n,
                                 const ::std::string& label = "operator") {
    printf("\nCondition number diagnostics for %s:\n", label.c_str());

    auto A_dense = materialize_linop<T>(A_linop, m, n);

    // Compute column norms
    ::std::vector<T> col_norms(n);
    for (int64_t j = 0; j < n; ++j)
        col_norms[j] = ::blas::nrm2(m, &A_dense[j * m], 1);

    T cn_min = *::std::min_element(col_norms.begin(), col_norms.end());
    T cn_max = *::std::max_element(col_norms.begin(), col_norms.end());
    printf("  Column norm range: [%.6e, %.6e], ratio: %.6e\n",
           (double)cn_min, (double)cn_max, (double)(cn_max / cn_min));

    // Copy for column-normalized version (gesdd is destructive)
    ::std::vector<T> A_normed(A_dense);
    for (int64_t j = 0; j < n; ++j)
        ::blas::scal(m, (T)1.0 / col_norms[j], &A_normed[j * m], 1);

    // SVD on raw matrix
    auto sigma = compute_singular_values<T>(A_dense.data(), m, n);
    printf("  Raw:     kappa = %.6e (sigma_max=%.6e, sigma_min=%.6e)\n",
           (double)(sigma[0] / sigma[n - 1]), (double)sigma[0], (double)sigma[n - 1]);

    // SVD on column-normalized matrix
    auto sigma_normed = compute_singular_values<T>(A_normed.data(), m, n);
    printf("  ColNorm: kappa = %.6e (sigma_max=%.6e, sigma_min=%.6e)\n",
           (double)(sigma_normed[0] / sigma_normed[n - 1]),
           (double)sigma_normed[0], (double)sigma_normed[n - 1]);

    printf("\n");
}

// ============================================================================
// Matrix Market I/O (Dense)
// ============================================================================

/// Write a dense column-major matrix to a Matrix Market coordinate file.
/// Entries with magnitude <= 1e-14 are treated as structural zeros.
///
/// @param[in] filename - Path to output .mtx file
/// @param[in] A       - Dense column-major array (m * n entries)
/// @param[in] m       - Number of rows
/// @param[in] n       - Number of columns
template <typename T>
void write_dense_to_mtx(
    const ::std::string& filename,
    const T* A,
    int64_t m,
    int64_t n
) {
    ::std::ofstream file(filename);
    if (!file.is_open()) {
        throw ::std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << ::std::scientific << ::std::setprecision(17);
    file << "%%MatrixMarket matrix coordinate real general\n";

    // Count nonzeros
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; ++i) {
        if (::std::abs(A[i]) > 1e-14) ++nnz;
    }

    file << m << " " << n << " " << nnz << "\n";

    // Write entries (Matrix Market uses 1-based indexing)
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            if (::std::abs(A[i + j * m]) > 1e-14) {
                file << (i + 1) << " " << (j + 1) << " " << A[i + j * m] << "\n";
            }
        }
    }

    file.close();
}

// ============================================================================
// Test Matrix Generation + File I/O
// ============================================================================

/// Generate a random n×n SPD matrix (via RandLAPACK::gen::gen_spd_mat)
/// and write it to a Matrix Market file.
template <typename T, typename RNG>
void generate_spd_matrix_file(
    const ::std::string& filename,
    int64_t n,
    T cond_num,
    ::RandBLAS::RNGState<RNG> &state
) {
    ::std::vector<T> A(n * n);
    ::RandLAPACK::gen::gen_spd_mat(n, cond_num, A.data(), state);
    write_dense_to_mtx(filename, A.data(), n, n);
}

/// Generate a random invertible (non-symmetric) n×n matrix and write to
/// a Matrix Market file.
///
/// Constructs A = Q1 * D * Q2^T where Q1, Q2 are independent random
/// orthogonal matrices and D = diag(singvals) with
///   sigma_i = 1 + (cond_num - 1) * (i/(n-1))^2,
/// giving kappa(A) = cond_num.
template <typename T, typename RNG>
void generate_invertible_matrix_file(
    const ::std::string& filename,
    int64_t n,
    T cond_num,
    ::RandBLAS::RNGState<RNG> &state
) {
    // Generate singular values with desired condition number
    ::std::vector<T> singvals(n);
    singvals[0] = 1.0;
    if (n > 1) {
        singvals[n-1] = cond_num;
        for (int64_t i = 1; i < n - 1; ++i) {
            T t = static_cast<T>(i) / static_cast<T>(n - 1);
            singvals[i] = 1.0 + (cond_num - 1.0) * t * t;
        }
    }

    // Generate two independent random orthogonal matrices Q1, Q2
    ::std::vector<T> Q1(n * n);
    ::std::vector<T> Q2(n * n);
    ::std::vector<T> tau(n);

    auto d1 = ::RandBLAS::DenseDist(n, n);
    state = ::RandBLAS::fill_dense(d1, Q1.data(), state);
    ::lapack::geqrf(n, n, Q1.data(), n, tau.data());
    ::lapack::orgqr(n, n, n, Q1.data(), n, tau.data());

    auto d2 = ::RandBLAS::DenseDist(n, n);
    state = ::RandBLAS::fill_dense(d2, Q2.data(), state);
    ::lapack::geqrf(n, n, Q2.data(), n, tau.data());
    ::lapack::orgqr(n, n, n, Q2.data(), n, tau.data());

    // Form A = Q1 * D * Q2^T (scale columns of Q1 by singular values, then GEMM)
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            Q1[i + j * n] *= singvals[j];

    ::std::vector<T> A(n * n);
    ::blas::gemm(::blas::Layout::ColMajor, ::blas::Op::NoTrans, ::blas::Op::Trans,
               n, n, n, (T)1.0, Q1.data(), n, Q2.data(), n, (T)0.0, A.data(), n);

    write_dense_to_mtx(filename, A.data(), n, n);
}

/// Left-multiply a column-major m x n matrix A by a random orthogonal matrix.
/// Generates a random m x m matrix, computes its QR factorization, and applies
/// Q to A via ormqr. A is modified in-place.
///
/// @return Updated RNG state
template <typename T, typename RNG>
::RandBLAS::RNGState<RNG> left_multiply_by_orthmat(
    int64_t m, int64_t n, ::std::vector<T> &A, ::RandBLAS::RNGState<RNG> state
) {
    ::std::vector<T> U(m * m, 0.0);
    ::RandBLAS::DenseDist DU(m, m);
    auto out_state = ::RandBLAS::fill_dense(DU, U.data(), state);
    ::std::vector<T> tau(m, 0.0);
    ::lapack::geqrf(m, m, U.data(), m, tau.data());
    ::lapack::ormqr(::blas::Side::Left, ::blas::Op::NoTrans, m, n, m,
                    U.data(), m, tau.data(), A.data(), m);
    return out_state;
}

}  // namespace testing
}  // namespace RandLAPACK
