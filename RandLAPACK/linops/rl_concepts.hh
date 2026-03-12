#pragma once

#include "rl_blaspp.hh"

#include <concepts>
#include <cstdint>


namespace RandLAPACK::linops {

// RandLAPACK's linear operator abstraction.
//
// A LinearOperator represents a matrix that can multiply dense data via a
// GEMM-like callable, without requiring explicit storage. Concrete types
// include DenseLinOp, SparseLinOp, and CompositeOperator (implicit product
// of two operators). Algorithms templated on LinearOperator or
// SymmetricLinearOperator accept any of these interchangeably.
//
// The concrete types also support multiplication with RandBLAS sketching
// operators and sparse matrices, plus block-view extraction — but those
// capabilities are not part of the concept itself.

/*********************************************************/
/*                                                       */
/*            Abstract Linear Operator Concept          */
/*                                                       */
/*********************************************************/
template<typename LinOp, typename T = LinOp::scalar_t>
concept LinearOperator = requires(LinOp A) {
    { A.n_rows }  -> std::same_as<const int64_t&>;
    { A.n_cols }  -> std::same_as<const int64_t&>;
} && requires(LinOp A, Layout layout, Op trans_A, Op trans_B, int64_t m, int64_t n, int64_t k, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
    // A Matmul-like function that updates C := alpha A*B + beta C, where
    // B and C have n columns and are stored in layout order with strides (ldb, ldc).
    { A(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc) } -> std::same_as<void>;
};

/*********************************************************/
/*                                                       */
/*        Symmetric Linear Operator Concept             */
/*                                                       */
/*********************************************************/
template<typename LinOp, typename T = LinOp::scalar_t>
concept SymmetricLinearOperator = requires(LinOp A) {
    { A.dim }  -> std::same_as<const int64_t&>;
    // It's recommended that A also have const int64_t members n_rows and n_cols,
    // both equal to A.dim.
} && requires(LinOp A, Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
    // A SYMM-like function that updates C := alpha A*B + beta C, where
    // B and C have n columns and are stored in layout order with strides (ldb, ldc).
    //
    // If layout is ColMajor then an error will be thrown if min(ldb, ldc) < A.dim.
    //
    { A(layout, n, alpha, B, ldb, beta, C, ldc) } -> std::same_as<void>;
};

} // end namespace RandLAPACK::linops
