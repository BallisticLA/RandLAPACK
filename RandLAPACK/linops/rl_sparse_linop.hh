#pragma once

// Public API: SparseLinOp — linear operator backed by a sparse matrix (CSR, CSC, or COO).

#include "rl_concepts.hh"
#include "rl_sparse_views.hh"
#include "rl_blaspp.hh"
#include "rl_util.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <memory>
#include <type_traits>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                   SparseLinOp                         */
/*                                                       */
/*********************************************************/

/// @brief Sparse linear operator for matrix multiplication operations
///
/// @details Provides a linear operator interface for sparse matrices, supporting
/// multiplication with both dense and sparse matrices. The operator supports:
/// - Left and right multiplication modes (Side::Left and Side::Right)
/// - Both ColMajor and RowMajor memory layouts
/// - Transpose operations (Op::NoTrans, Op::Trans)
/// - Sparse-dense and sparse-sparse multiplications
///
/// The operator delegates to RandBLAS sparse BLAS routines for optimal performance.
///
/// @tparam SpMat Sparse matrix type (CSC, CSR, or COO format)
///
/// @note For sparse-sparse multiplication, uses RandBLAS::spgemm (MKL-accelerated)
/// when available. Falls back to densifying one operand when MKL is absent or
/// when transpose configurations are not directly supported by spgemm.
template <RandBLAS::sparse_data::SparseMatrix SpMat>
struct SparseLinOp {
    using T = typename SpMat::scalar_t;
    using scalar_t = T;
    using sint_t = typename SpMat::index_t;
    const int64_t n_rows;  ///< Number of rows in the operator matrix
    const int64_t n_cols;  ///< Number of columns in the operator matrix

private:
    // ---- Block data ownership (type-erased) ----
    //
    // When this SparseLinOp is produced by a block method (row_block,
    // col_block, submatrix), block_owner_ holds the block/view struct
    // (e.g., CSRRowBlockView, CSRColBlock, CSCColBlockView, CSCRowBlock)
    // that owns the underlying sparse data arrays.  For SparseLinOps
    // constructed directly from an external SpMat&, block_owner_ is null.
    //
    // Uses std::shared_ptr<void> for type-erased ownership: the correct
    // destructor is captured at construction time via shared_ptr's deleter,
    // so a single member can hold any block type without exposing it in the
    // class template signature.
    //
    // IMPORTANT: block_owner_ is declared BEFORE A_sp so that C++ destruction
    // order (reverse of declaration) destroys A_sp (the non-owning view)
    // first, then block_owner_ releases the data A_sp may point into.
    std::shared_ptr<void> block_owner_;

    /// @brief Create a non-owning SpMat view of an existing sparse matrix.
    static SpMat make_view(SpMat& src) {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        using COO = RandBLAS::sparse_data::COOMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            return CSR(src.n_rows, src.n_cols, src.nnz,
                       src.vals, src.rowptr, src.colidxs);
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            return CSC(src.n_rows, src.n_cols, src.nnz,
                       src.vals, src.rowidxs, src.colptr);
        } else if constexpr (std::is_same_v<SpMat, COO>) {
            return COO(src.n_rows, src.n_cols, src.nnz,
                       src.vals, src.rows, src.cols);
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "make_view only supports CSR, CSC, and COO sparse formats");
        }
    }

    /// @brief Private constructor for block-derived SparseLinOps.
    SparseLinOp(int64_t n_rows, int64_t n_cols,
                SpMat&& view, std::shared_ptr<void> owner)
        : n_rows(n_rows), n_cols(n_cols),
          block_owner_(std::move(owner)), A_sp(std::move(view)) {}

public:
    // ---- Sparse matrix data ----
    //
    // Non-owning view (own_memory == false) of the sparse matrix, created
    // via the SpMat expert constructor.  The actual data resides either in
    // the caller's original SpMat (for externally-constructed operators) or
    // in block_owner_ (for block-derived operators).
    //
    // Declared mutable because const block methods (row_block, col_block,
    // submatrix) must pass A_sp to free functions (e.g., csr_row_block)
    // that take SpMat& (non-const reference).  The underlying matrix data
    // is never modified through A_sp; the mutable qualifier only relaxes
    // the top-level const on the SpMat object itself.
    mutable SpMat A_sp;

    /// @brief Construct a sparse linear operator from an existing sparse matrix.
    SparseLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        SpMat &src
    ) : n_rows(n_rows), n_cols(n_cols), block_owner_(), A_sp(make_view(src)) {

    }

    /// @brief Compute the Frobenius norm of the sparse operator matrix
    T fro_nrm(
    ) {
        return blas::nrm2(A_sp.nnz, A_sp.vals, 1);
    }

    /// @brief Sparse-dense matrix multiplication: C := alpha * op(A_sp) * op(B) + beta * C
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        const T* B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    /// @brief Sparse-dense multiplication with explicit side specification.
    /// Side refers to the side on which this operator appears.
    ///   Side::Left:  C = alpha * op(A_sp) * op(B) + beta * C
    ///   Side::Right: C = alpha * op(B) * op(A_sp) + beta * C
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        const T* B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if (side == Side::Left) {
            // C := alpha * op(A_sp) * op(B) + beta * C
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            RandBLAS::sparse_data::left_spmm(layout, trans_A, trans_B, m, n, k, alpha, A_sp, 0, 0, B, ldb, beta, C, ldc);
        } else {  // Side::Right
            // C := alpha * op(B) * op(A_sp) + beta * C
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            RandBLAS::sparse_data::right_spmm(layout, trans_B, trans_A, m, n, k, alpha, B, ldb, A_sp, 0, 0, beta, C, ldc);
        }
    }

    /// @brief Sparse-sparse matrix multiplication: C := alpha * op(A_sp) * op(B_sp) + beta * C
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SpMatB &B_sp,
        T beta,
        T* C,
        int64_t ldc
    ) {
        (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);
    }

    /// @brief Sparse-sparse multiplication with explicit side specification.
    /// Side refers to the side on which this operator appears.
    ///   Side::Left:  C = alpha * op(A_sp) * op(B_sp) + beta * C
    ///   Side::Right: C = alpha * op(B_sp) * op(A_sp) + beta * C
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_B,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SpMatB &B_sp,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if (side == Side::Left) {
            // C := alpha * op(A_sp) * op(B_sp) + beta * C
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldc >= n);
            }

            #if defined(RandBLAS_HAS_MKL)
            if (trans_B == Op::NoTrans) {
                RandBLAS::spgemm(layout, trans_A, alpha, A_sp, B_sp, beta, C, ldc);
                return;
            }
            #endif

            // Fallback: densify B_sp and use sparse-dense multiplication.
            T* B_dense = new T[rows_B * cols_B]();
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
            RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);
            (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);
            delete[] B_dense;

        } else {  // Side::Right
            // C := alpha * op(B_sp) * op(A_sp) + beta * C
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);

            if (layout == Layout::ColMajor) {
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldc >= n);
            }

            #if defined(RandBLAS_HAS_MKL)
            if (trans_A == Op::NoTrans) {
                RandBLAS::spgemm(layout, trans_B, alpha, B_sp, A_sp, beta, C, ldc);
                return;
            }
            #endif

            // Fallback: densify B_sp and use sparse-dense multiplication.
            T* B_dense = new T[rows_B * cols_B]();
            int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
            RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);
            RandBLAS::sparse_data::right_spmm(layout, trans_B, trans_A, m, n, k, alpha, B_dense, ldb, A_sp, 0, 0, beta, C, ldc);
            delete[] B_dense;
        }
    }

    /// Left-multiply this by a sketching operator.
    /// Computes C = alpha * op(S) * op(A) + beta * C (equivalent to Side::Right).
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Layout layout,
        Op trans_A,
        Op trans_S,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        (*this)(Side::Right, layout, trans_A, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    /// Sketching operator multiplication with sparse linear operator.
    /// Side refers to the side on which this operator appears.
    ///   Side::Left:  C = alpha * op(A) * op(S) + beta * C
    ///   Side::Right: C = alpha * op(S) * op(A) + beta * C
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side,
        Layout layout,
        Op trans_A,
        Op trans_S,
        int64_t m,
        int64_t n,
        int64_t k,
        T alpha,
        SkOp& S,
        T beta,
        T* C,
        int64_t ldc
    ) {
        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            // Dense sketch operator: extract buffer and use SpMM directly
            if (S.buff == nullptr) {
                RandBLAS::fill_dense(S);
            }
            // Handle layout mismatch
            Op adjusted_trans = trans_S;
            if (S.layout != layout) {
                adjusted_trans = (trans_S == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            }
            (*this)(side, layout, trans_A, adjusted_trans, m, n, k, alpha, S.buff, S.dist.dim_major, beta, C, ldc);
        } else {
            // Sparse sketch operator: use spgemm when available.
            #if defined(RandBLAS_HAS_MKL)
            {
                if (S.nnz < 0)
                    RandBLAS::fill_sparse(S);
                auto S_coo = RandBLAS::coo_view_of_skop(S);

                if (side == Side::Left && trans_S == Op::NoTrans) {
                    auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
                    randblas_require(rows_A == n_rows);
                    randblas_require(cols_A == n_cols);
                    if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                    else randblas_require(ldc >= n);
                    RandBLAS::spgemm(layout, trans_A, alpha, A_sp, S_coo, beta, C, ldc);
                    return;
                }
                if (side == Side::Right && trans_A == Op::NoTrans) {
                    auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
                    randblas_require(rows_A == n_rows);
                    randblas_require(cols_A == n_cols);
                    if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                    else randblas_require(ldc >= n);
                    RandBLAS::spgemm(layout, trans_S, alpha, S_coo, A_sp, beta, C, ldc);
                    return;
                }
            }
            #endif

            // Fallback: densify A_sp and use sketch_general.
            T* A_dense = new T[n_rows * n_cols]();
            int64_t lda = (layout == Layout::ColMajor) ? n_rows : n_cols;
            RandLAPACK::util::sparse_to_dense(A_sp, layout, A_dense);

            if (side == Side::Left) {
                auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
                randblas_require(rows_A == n_rows);
                randblas_require(cols_A == n_cols);
                if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                else randblas_require(ldc >= n);
                RandBLAS::sketch_general(layout, trans_A, trans_S, m, n, k, alpha, A_dense, lda, S, beta, C, ldc);
            } else {
                auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
                randblas_require(rows_A == n_rows);
                randblas_require(cols_A == n_cols);
                if (layout == Layout::ColMajor) randblas_require(ldc >= m);
                else randblas_require(ldc >= n);
                RandBLAS::sketch_general(layout, trans_S, trans_A, m, n, k, alpha, S, A_dense, lda, beta, C, ldc);
            }

            delete[] A_dense;
        }
    }

    /// @brief Extract a row block [row_start, row_start + row_count).
    SparseLinOp row_block(int64_t row_start, int64_t row_count) const {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            auto blk = std::make_shared<CSRRowBlockView<T, sint_t>>(
                csr_row_block(A_sp, row_start, row_count));
            auto view = blk->as_csr();
            return SparseLinOp(row_count, A_sp.n_cols,
                               std::move(view), std::move(blk));
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            auto blk = std::make_shared<CSCRowBlock<T, sint_t>>(
                csc_row_block(A_sp, row_start, row_count));
            auto view = blk->as_csc();
            return SparseLinOp(row_count, A_sp.n_cols,
                               std::move(view), std::move(blk));
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "row_block is only supported for CSR and CSC sparse formats");
        }
    }

    /// @brief Extract a column block [col_start, col_start + col_count).
    SparseLinOp col_block(int64_t col_start, int64_t col_count) const {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            auto blk = std::make_shared<CSRColBlock<T, sint_t>>(
                csr_col_block(A_sp, col_start, col_count));
            auto view = blk->as_csr();
            return SparseLinOp(A_sp.n_rows, col_count,
                               std::move(view), std::move(blk));
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            auto blk = std::make_shared<CSCColBlockView<T, sint_t>>(
                csc_col_block(A_sp, col_start, col_count));
            auto view = blk->as_csc();
            return SparseLinOp(A_sp.n_rows, col_count,
                               std::move(view), std::move(blk));
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "col_block is only supported for CSR and CSC sparse formats");
        }
    }

    /// @brief Extract a submatrix at (row_start, col_start).
    SparseLinOp submatrix(int64_t row_start, int64_t col_start,
                          int64_t row_count, int64_t col_count) const {
        using CSR = RandBLAS::sparse_data::CSRMatrix<T, sint_t>;
        using CSC = RandBLAS::sparse_data::CSCMatrix<T, sint_t>;
        if constexpr (std::is_same_v<SpMat, CSR>) {
            // Step 1: natural row-block (O(row_count), temporary view)
            auto row_view = csr_row_block(A_sp, row_start, row_count);
            auto row_csr = row_view.as_csr();
            // Step 2: cross-direction col extraction (O(block_nnz), full copy)
            auto blk = std::make_shared<CSRColBlock<T, sint_t>>(
                csr_col_block(row_csr, col_start, col_count));
            auto view = blk->as_csr();
            return SparseLinOp(row_count, col_count,
                               std::move(view), std::move(blk));
        } else if constexpr (std::is_same_v<SpMat, CSC>) {
            // Step 1: natural col-block (O(col_count), temporary view)
            auto col_view = csc_col_block(A_sp, col_start, col_count);
            auto col_csc = col_view.as_csc();
            // Step 2: cross-direction row extraction (O(block_nnz), full copy)
            auto blk = std::make_shared<CSCRowBlock<T, sint_t>>(
                csc_row_block(col_csc, row_start, row_count));
            auto view = blk->as_csc();
            return SparseLinOp(row_count, col_count,
                               std::move(view), std::move(blk));
        } else {
            static_assert(!std::is_same_v<SpMat, SpMat>,
                "submatrix is only supported for CSR and CSC sparse formats");
        }
    }
};


/*********************************************************/
/*                                                       */
/*                   SparseSymLinOp                      */
/*                                                       */
/*********************************************************/

/// @brief Symmetric sparse linear operator satisfying the SymmetricLinearOperator concept.
///
/// Wraps a sparse matrix assumed to be symmetric with both triangles explicitly stored.
/// Delegates to `left_spmm` with NoTrans for SYMM-like semantics: C = alpha * A * B + beta * C.
///
/// Satisfies SymmetricLinearOperator (provides `dim` and SYMM-like operator()).
/// Also provides `n_rows` and `n_cols` (both equal to `dim`) for LinearOperator compatibility.
///
/// @tparam SpMat Sparse matrix type (CSR, CSC, or COO format).
///
/// @note The sparse matrix must be square (n×n) and symmetric with both triangles stored.
///       The caller's SpMat must outlive this operator.
template <RandBLAS::sparse_data::SparseMatrix SpMat>
struct SparseSymLinOp {
    using T = typename SpMat::scalar_t;
    using scalar_t = T;
    const int64_t dim;
    const int64_t n_rows;
    const int64_t n_cols;
    SpMat& A_sp;

    /// @brief Construct from an existing symmetric sparse matrix.
    /// @param n    Matrix dimension (n × n); A_sp must be n × n.
    /// @param src  Sparse matrix; must be square, symmetric, both triangles stored.
    SparseSymLinOp(int64_t n, SpMat& src)
        : dim(n), n_rows(n), n_cols(n), A_sp(src) {}

    /// @brief Apply: C = alpha * A_sp * B + beta * C.
    void operator()(Layout layout, int64_t n_vecs, T alpha,
                    T* const B, int64_t ldb,
                    T beta, T* C, int64_t ldc) {
        RandBLAS::sparse_data::left_spmm(layout, Op::NoTrans, Op::NoTrans,
            dim, n_vecs, dim, alpha, A_sp, 0, 0, B, ldb, beta, C, ldc);
    }
};

} // end namespace RandLAPACK::linops
