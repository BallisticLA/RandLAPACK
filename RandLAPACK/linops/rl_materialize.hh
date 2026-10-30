#pragma once

// Public API: materialize, which writes the dense representation of a linear operator
// into a caller-provided buffer.
//
// Generic fallback: multiply by the identity matrix.
// Overloaded specializations avoid that cost when the operator writes its own
// columns (materialize_cols; DenseLinOp and SparseLinOp among its providers) or
// the columns are reachable through the operands (CompositeOperator).

#include "rl_exceptions.hh"
#include "rl_concepts.hh"
#include "rl_composite_linop.hh"
#include "rl_blaspp.hh"
#include "rl_util.hh"

#include <cstdint>
#include <memory>


namespace RandLAPACK {

/// Cap on the dimension the generic materialize fallback will accept. The
/// fallback allocates an n by n identity (2 GiB in double at the cap) behind
/// the caller's buffer. The type-specific overloads below form no identity
/// and carry no cap.
inline constexpr int64_t MATERIALIZE_IDENTITY_MAX_DIM = 16384;

/// Materialize a linear operator into a dense column-major buffer.
///
/// Generic fallback: forms buf = A * I  by applying the operator to the
/// identity matrix through the concept-required overload, so it works for any
/// type satisfying the LinearOperator concept.
///
/// @param[in]  A    Linear operator (m-by-n).
/// @param[in]  m    Number of rows.
/// @param[in]  n    Number of columns.
/// @param[out] buf  Pre-allocated buffer of length at least m * n.
/// @param[in]  ldb  Leading dimension of buf (>= m).
template <typename LinOp>
void materialize(LinOp& A, int64_t m, int64_t n, typename LinOp::scalar_t* buf, int64_t ldb) {
    using T = typename LinOp::scalar_t;
    randlapack_require(n <= MATERIALIZE_IDENTITY_MAX_DIM) << "n=" << n
        << " exceeds MATERIALIZE_IDENTITY_MAX_DIM=" << MATERIALIZE_IDENTITY_MAX_DIM
        << ": the generic materialize fallback would allocate an n*n identity ("
        << (n * n * (int64_t) sizeof(T)) / (1024 * 1024) << " MiB)."
        << " Use a type-specific materialize overload instead";
    randlapack_require(ldb >= m) << "ldb=" << ldb << " < m=" << m << " (ldb must be >= m)";
    // Zero the output buffer.
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            buf[i + j * ldb] = (T)0.0;
    std::unique_ptr<T[]> Eye(new T[n * n]);
    RandLAPACK::util::eye(n, n, Eye.get());
    // buf = 1.0 * A * I + 0.0 * buf
    A(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
      m, n, n, (T)1.0, Eye.get(), n, (T)0.0, buf, ldb);
}

/// Overload for operators that provide materialize_cols (HasMaterializeCols), DenseLinOp and
/// SparseLinOp among them: writes all columns through it, with no identity and no cap.
template <linops::HasMaterializeCols LinOp>
void materialize(LinOp& A, int64_t m, int64_t n, typename LinOp::scalar_t* buf, int64_t ldb) {
    randlapack_require(m == A.n_rows && n == A.n_cols) << "m=" << m << ", n=" << n
        << " must equal the operator's " << A.n_rows << " x " << A.n_cols;
    randlapack_require(ldb >= m) << "ldb=" << ldb << " < m=" << m << " (ldb must be >= m)";
    A.materialize_cols(0, n, buf, ldb);
}

/// Specialization for CompositeOperator: materialize each operand, then form
/// buf = L * R with one gemm. Builds no identity, carries no cap, and never
/// calls the composite's operator(); nested composites recurse.
template <typename LinOp1, typename LinOp2>
void materialize(linops::CompositeOperator<LinOp1, LinOp2>& A, int64_t m, int64_t n,
                 typename LinOp1::scalar_t* buf, int64_t ldb) {
    using T = typename LinOp1::scalar_t;
    randlapack_require(m == A.n_rows) << "m=" << m << " must equal A.n_rows=" << A.n_rows << " for materialize specialization";
    randlapack_require(n == A.n_cols) << "n=" << n << " must equal A.n_cols=" << A.n_cols << " for materialize specialization";
    randlapack_require(ldb >= m) << "ldb=" << ldb << " < m=" << m << " (ldb must be >= m)";
    int64_t k = A.left_op.n_cols;
    std::unique_ptr<T[]> L(new T[m * k]), R(new T[k * n]);
    materialize(A.left_op, m, k, L.get(), m);
    materialize(A.right_op, k, n, R.get(), k);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               m, n, k, (T)1.0, L.get(), m, R.get(), k, (T)0.0, buf, ldb);
}

} // end namespace RandLAPACK
