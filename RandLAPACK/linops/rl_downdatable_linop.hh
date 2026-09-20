#pragma once

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_exceptions.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace RandLAPACK::linops {

/// Represents A - E * F^T without changing the base operator. Each update
/// appends column-major factors; operator applications accept either layout.
/// E has n_rows rows and F has n_cols rows; neither needs orthogonal columns.
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
        const auto e_size = checked_buffer_size(n_rows, max_rank);
        const auto f_size = checked_buffer_size(n_cols, max_rank);
        E_data.resize(e_size);
        F_data.resize(f_size);
    }

    /// Append b_sz columns. Exceeding the capacity is rejected before any copy.
    void update(int64_t b_sz, const T* E_new, const T* F_new) {
        randlapack_require(b_sz >= 0 && b_sz <= max_rank - curr_rank)
            << "downdate columns exceed the remaining rank capacity";
        if (b_sz == 0) return;
        randlapack_require(E_new != nullptr && F_new != nullptr);
        lapack::lacpy(MatrixType::General, n_rows, b_sz, E_new, n_rows,
                      E_data.data() + n_rows * curr_rank, n_rows);
        lapack::lacpy(MatrixType::General, n_cols, b_sz, F_new, n_cols,
                      F_data.data() + n_cols * curr_rank, n_cols);
        curr_rank += b_sz;
    }

    /// C := alpha * op(A - E*F^T) * op(B) + beta * C.
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

        // Validate and allocate scratch before the base operator can touch output.
        if (curr_rank != 0 && n != 0 && alpha != T(0))
            scratch.resize(checked_buffer_size(curr_rank, n));
        base_op(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        if (curr_rank == 0 || n == 0 || alpha == T(0)) return;

        // Factors remain column-major irrespective of the RHS/output layout.
        const T* left = transpose ? F_data.data() : E_data.data();
        const T* right = transpose ? E_data.data() : F_data.data();
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
    static std::size_t checked_buffer_size(int64_t rows, int64_t cols) {
        // Keep both signed indexing products and the vector allocation valid.
        const auto max_size = std::min<uint64_t>(
            std::numeric_limits<int64_t>::max(), std::vector<T>().max_size());
        if (rows < 0 || cols < 0 || (rows != 0 &&
            static_cast<uint64_t>(cols) > max_size / static_cast<uint64_t>(rows)))
            throw Error("downdate buffer dimensions exceed the supported size");
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    }

    BaseLinOp& base_op;
    int64_t curr_rank = 0;
    const int64_t max_rank;
    std::vector<T> E_data;
    std::vector<T> F_data;
    std::vector<T> scratch;
};

} // namespace RandLAPACK::linops
