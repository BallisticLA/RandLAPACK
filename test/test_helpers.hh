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

} // namespace test_helpers
