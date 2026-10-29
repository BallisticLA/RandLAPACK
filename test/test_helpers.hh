#pragma once

// Helpers shared by the linear-operator and driver tests.

#include <RandLAPACK.hh>

#include <cstdint>
#include <stdexcept>

namespace test_helpers {

/// Satisfies the LinearOperator concept and nothing more (no Side or sketch overload, no
/// materialize overload): applies a column-major n_rows x n_cols matrix by GEMM.
template <typename T>
struct ConceptOnlyLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    const T* A;
    void operator()(blas::Layout layout, blas::Op trans_A, blas::Op trans_B, int64_t m, int64_t n,
                    int64_t k, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc) {
        if (layout != blas::Layout::ColMajor) throw std::runtime_error("ColMajor only");
        blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A, n_rows, B, ldb, beta, C, ldc);
    }
};
static_assert(RandLAPACK::linops::LinearOperator<ConceptOnlyLinOp<double>>);

/// A column-major n_rows x n_cols matrix with the concept's product (GEMM) and materialize_cols
/// (a copy), counting the columns each produced, so a test can tell which one a caller used.
template <typename T>
struct MaterializeColsProbe {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    const T* A;
    int64_t product_cols = 0;   ///< columns of NoTrans products
    int64_t direct_cols  = 0;   ///< columns written by materialize_cols
    void operator()(blas::Layout layout, blas::Op trans_A, blas::Op trans_B, int64_t m, int64_t n,
                    int64_t k, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randlapack_require(layout == blas::Layout::ColMajor)
            << "MaterializeColsProbe: ColMajor only, got " << blas::to_string(layout);
        if (trans_A == blas::Op::NoTrans) product_cols += n;
        blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A, n_rows, B, ldb, beta, C, ldc);
    }
    void materialize_cols(int64_t col_start, int64_t col_count, T* buf, int64_t ldb) {
        direct_cols += col_count;
        lapack::lacpy(lapack::MatrixType::General, n_rows, col_count, A + col_start * n_rows, n_rows, buf, ldb);
    }
};
static_assert(RandLAPACK::linops::HasMaterializeCols<MaterializeColsProbe<double>>);
static_assert(!RandLAPACK::linops::HasMaterializeCols<ConceptOnlyLinOp<double>>);

} // namespace test_helpers
