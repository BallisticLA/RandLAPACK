#pragma once

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_exceptions.hh"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace RandLAPACK::linops {

/// Represents A - Q * BT^T without changing the base operator. Each update
/// appends column-major factors; operator applications accept either layout.
/// The base operator must outlive this object and support the requested layout.
template <typename T, LinearOperator BaseLinOp>
class DowndatableLinOp {
public:
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;

    DowndatableLinOp(BaseLinOp& base, const int64_t max_rank)
        : n_rows(base.n_rows), n_cols(base.n_cols), base_op(base),
          max_rank(max_rank) {
        randlapack_require(n_rows >= 0 && n_cols >= 0);
        if (max_rank < 0)
            throw Error("maximum downdate rank must be nonnegative");
        Q_data.resize(n_rows * max_rank);
        BT_data.resize(n_cols * max_rank);
    }

    /// Append b_sz columns. Exceeding the capacity is rejected before any copy.
    void update(int64_t b_sz, const T* Q_new, const T* BT_new) {
        randlapack_require(b_sz >= 0 && b_sz <= max_rank - curr_rank)
            << "downdate columns exceed the remaining rank capacity";
        if (b_sz == 0) return;
        randlapack_require(Q_new != nullptr && BT_new != nullptr);
        lapack::lacpy(MatrixType::General, n_rows, b_sz, Q_new, n_rows,
                      Q_data.data() + n_rows * curr_rank, n_rows);
        lapack::lacpy(MatrixType::General, n_cols, b_sz, BT_new, n_cols,
                      BT_data.data() + n_cols * curr_rank, n_cols);
        curr_rank += b_sz;
    }

    /// C := alpha * op(A - Q*BT^T) * op(B) + beta * C.
    void operator()(
        Layout layout, Op trans_A, Op trans_B,
        int64_t m, const int64_t n, int64_t k, T alpha,
        const T* B, int64_t ldb, T beta, T* C, int64_t ldc
    ) {
        randlapack_require(layout == Layout::ColMajor || layout == Layout::RowMajor);
        randlapack_require(trans_A == Op::NoTrans || trans_A == Op::Trans);
        randlapack_require(trans_B == Op::NoTrans || trans_B == Op::Trans);
        if (n < 0)
            throw Error("RHS column count must be nonnegative");
        const bool transpose = trans_A == Op::Trans;
        randlapack_require(m == (transpose ? n_cols : n_rows));
        randlapack_require(k == (transpose ? n_rows : n_cols));

        base_op(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        if (curr_rank == 0 || n == 0 || alpha == T(0)) return;

        // Factors remain column-major irrespective of the RHS/output layout.
        const T* left = transpose ? BT_data.data() : Q_data.data();
        const T* right = transpose ? Q_data.data() : BT_data.data();
        scratch.resize(curr_rank * n);
        const Op rhs_op = layout == Layout::ColMajor ? trans_B
            : (trans_B == Op::NoTrans ? Op::Trans : Op::NoTrans);
        blas::gemm(Layout::ColMajor, Op::Trans, rhs_op, curr_rank, n, k,
                   T(1), right, k, B, ldb, T(0), scratch.data(), curr_rank);

        // In row-major storage, the column-major factors are transposed views.
        const Op factor_op = layout == Layout::ColMajor ? Op::NoTrans : Op::Trans;
        blas::gemm(layout, factor_op, factor_op, m, n, curr_rank,
                   -alpha, left, m, scratch.data(), curr_rank, T(1), C, ldc);
    }

private:
    BaseLinOp& base_op;
    int64_t curr_rank = 0;
    const int64_t max_rank;
    std::vector<T> Q_data;
    std::vector<T> BT_data;
    std::vector<T> scratch;
};

} // namespace RandLAPACK::linops
