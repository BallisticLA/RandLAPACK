#pragma once

#include "rl_concepts.hh"
#include "rl_blaspp.hh"
#include "rl_exceptions.hh"
#include "rl_memory.hh"

#include <cstdint>
#include <memory>

namespace RandLAPACK::linops {

/// A - E * F^T, applied in either layout, with packed column-major views E (n_rows x rank) and F (n_cols x
/// rank) that the owner sets with set_factors and refreshes whenever it moves or extends them. Neither the
/// base operator, which must outlive this object, nor the factors are copied.
template <typename T, LinearOperator BaseLinOp>
class DowndatableLinOp {
public:
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;

    DowndatableLinOp(BaseLinOp& base, const int64_t max_rank)
        : n_rows(base.n_rows), n_cols(base.n_cols), base_op(base), max_rank(max_rank) {
        randlapack_require(n_rows >= 0 && n_cols >= 0);
        if (max_rank < 0)
            throw Error("maximum downdate rank must be nonnegative");
        // The factor views the owner supplies must be indexable with int64_t.
        util::checked_mul(n_rows, max_rank);
        util::checked_mul(n_cols, max_rank);
    }

    /// Point at the first rank columns of E and F, replacing any earlier factors.
    void set_factors(const T* E, const T* F, int64_t rank) {
        randlapack_require(rank >= 0 && rank <= max_rank)
            << "downdate rank " << rank << " must lie in [0, " << max_rank << "]";
        randlapack_require(rank == 0 || (E != nullptr && F != nullptr));
        E_data = E;
        F_data = F;
        curr_rank = rank;
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

        // Allocate scratch before the base operator can touch the output.
        const bool correct = curr_rank != 0 && n != 0 && alpha != T(0);
        std::unique_ptr<T[]> scratch;
        if (correct)
            scratch = util::make_buffer<T>(util::checked_mul(curr_rank, n));
        base_op(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        if (!correct) return;

        // Factors remain column-major irrespective of the RHS/output layout.
        const T* left = transpose ? F_data : E_data;
        const T* right = transpose ? E_data : F_data;
        const Op rhs_op = layout == Layout::ColMajor ? trans_B
            : (trans_B == Op::NoTrans ? Op::Trans : Op::NoTrans);
        blas::gemm(Layout::ColMajor, Op::Trans, rhs_op, curr_rank, n, k,
                   T(1), right, k, B, ldb, T(0), scratch.get(), curr_rank);

        // In row-major storage, the column-major factors are transposed views.
        const Op factor_op = layout == Layout::ColMajor ? Op::NoTrans : Op::Trans;
        blas::gemm(layout, factor_op, factor_op, m, n, curr_rank,
                   -alpha, left, m, scratch.get(), curr_rank, T(1), C, ldc);
    }

private:
    BaseLinOp& base_op;
    const int64_t max_rank;
    const T* E_data = nullptr;
    const T* F_data = nullptr;
    int64_t curr_rank = 0;
};

} // namespace RandLAPACK::linops
