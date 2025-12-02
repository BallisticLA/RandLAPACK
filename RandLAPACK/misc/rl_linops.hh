#pragma once

#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_util.hh"
#include "RandBLAS/sparse_data/base.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <concepts>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*            Abstract Linear Operator Concept          */
/*                                                       */
/*********************************************************/
// Abstract linear operator concept
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
/*            EfficientAndSparseLinOp                    */
/*                                                       */
/*********************************************************/
// Custom linear operator for the fast randomized Q-less Cholesky QR algorithm.
// Represents the implicit product of an efficient operator A and a sparse tall matrix Omega.
// This operator computes (A*Omega)*B by evaluating from right to left: A*(Omega*B).
//
// Template parameters:
//   EffLinOp - Type satisfying LinearOperator concept (e.g., SRFT, dense matrix wrapper)
//   SpMat - Sparse matrix type from RandBLAS
template <LinearOperator EffLinOp, RandBLAS::sparse_data::SparseMatrix SpMat>
struct EfficientAndSparseLinOp {
    using T = typename EffLinOp::scalar_t;
    using scalar_t = T;
    const int64_t n_rows;  // Number of rows in A*Omega
    const int64_t n_cols;  // Number of columns in A*Omega
    EffLinOp &A_eff;       // Reference to the efficient operator A
    SpMat &Omega_sp;       // Reference to the sparse matrix Omega

    EfficientAndSparseLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        EffLinOp &A_eff,
        SpMat &Omega_sp
    ) : n_rows(n_rows), n_cols(n_cols), A_eff(A_eff), Omega_sp(Omega_sp) {
        // Validate dimensions: A is (n_rows x Omega.n_rows), Omega is (Omega.n_rows x n_cols)
        randblas_require(A_eff.n_cols == Omega_sp.n_rows);
        randblas_require(Omega_sp.n_cols == n_cols);
    }

    // Default operator: Computes C := alpha * op(A*Omega) * op(B) + beta * C
    // Strategy: temp = Omega * B, then C = alpha * A * temp + beta * C
    void operator()(
        Layout layout,
        Op trans_AO,  // Transposition of the composite operator (A*Omega)
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
        // Validate input dimensions
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        randblas_require(ldb >= rows_B);
        auto [rows_AO, cols_AO] = RandBLAS::dims_before_op(m, k, trans_AO);
        randblas_require(rows_AO <= n_rows);
        randblas_require(cols_AO <= n_cols);
        randblas_require(ldc >= m);

        if (trans_AO == Op::NoTrans) {
            // C := alpha * (A * Omega) * op(B) + beta * C
            // Note: Omega_sp.n_cols is only known to this sparse matrix.
            // This parameter can have any value.
            int64_t temp_rows = Omega_sp.n_cols;
            T* temp_buffer = new T[temp_rows * n]();
            int64_t ldt = temp_rows;

            // Step 1: temp := Omega * op(B), dimension (n_cols x n)
            // Sparse matrix gets multiplied by a dense matrix on the right,
            // result is stored in a buffer.
            RandBLAS::sparse_data::left_spmm(layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, Omega_sp, 0, 0, B, ldb, (T)0.0, temp_buffer, ldt);

            // Step 2: C := alpha * A * temp + beta * C
            // Multiply the fast operator by temporary buffer on the right
            A_eff(layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);

            delete[] temp_buffer;

        } else {
            // trans_AO == Op::Trans
            // C := alpha * (A * Omega)^T * op(B) + beta * C
            //    = alpha * Omega^T * A^T * op(B) + beta * C
            // Note: A_eff.n_cols is only known to this operator.
            // This parameter can have any value.
            int64_t temp_rows = A_eff.n_cols;
            T* temp_buffer = new T[temp_rows * n]();
            int64_t ldt = temp_rows;

            // Step 1: temp := A^T * op(B), dimension (A.n_cols x n)
            // Efficient operator gets multiplied by a dense matrix on the right,
            // result stored in a buffer.
            A_eff(layout, Op::Trans, trans_B, temp_rows, n, k, (T)1.0, B, ldb, (T)0.0, temp_buffer, ldt);

            // Step 2: C := alpha * Omega^T * temp + beta * C
            // Sparse matrix gets multiplied by a dense matrix on the right,
            // result is stored in a buffer.
            RandBLAS::sparse_data::left_spmm(layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, Omega_sp, 0, 0, temp_buffer, ldt, beta, C, ldc);

            delete[] temp_buffer;
        }
    }

    // Augmented operator to allow for left and right multiplication.
    // Side::Left: C := alpha * (A*Omega) * op(B) + beta * C  (delegates to default operator)
    // Side::Right: C := alpha * op(B) * (A*Omega) + beta * C
    void operator()(
        Side side,
        Layout layout,
        Op trans_AO,  // Transposition of the composite operator (A*Omega)
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
            // Left multiplication: delegate to default operator
            (*this)(layout, trans_AO, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);

        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B) * op(A*Omega) + beta * C
            // We use the transpose trick: compute C^T := alpha * op(A*Omega)^T * op(B)^T + beta * C^T
            // with swapped layout

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            randblas_require(ldb >= rows_B);
            auto [rows_AO, cols_AO] = RandBLAS::dims_before_op(k, n, trans_AO);
            randblas_require(rows_AO <= n_rows);
            randblas_require(cols_AO <= n_cols);
            randblas_require(ldc >= m);

            // Transpose the operation: swap trans_AO and use opposite layout
            auto trans_trans_AO = (trans_AO == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;

            // Now call the default operator with transposed parameters
            (*this)(trans_layout, trans_trans_AO, trans_B, n, m, k, alpha, B, ldb, beta, C, ldc);
        }
    }

    // Sparse matrix multiplication operator: C := alpha * op(A*Omega) * op(B_sp) + beta * C
    // where B_sp is a sparse matrix.
    //
    // Strategy depends on trans_AO and whether A_eff is sparse:
    //
    // Case 1: trans_AO == NoTrans
    //   C := alpha * (A * Omega) * op(B_sp) + beta * C
    //   Since Omega (sparse) * op(B_sp) (sparse) has no sparse-sparse implementation in RandBLAS,
    //   we ALWAYS convert B_sp to dense first, regardless of whether A is sparse or not.
    //
    // Case 2: trans_AO == Trans
    //   C := alpha * (A * Omega)^T * op(B_sp) + beta * C
    //     = alpha * Omega^T * A^T * op(B_sp) + beta * C
    //
    //   Sub-case 2a: A_eff is sparse
    //     Both A^T (sparse) and op(B_sp) (sparse) are sparse, so we convert B_sp to dense
    //     to avoid sparse-sparse issues, then use the existing dense operator.
    //
    //   Sub-case 2b: A_eff is NOT sparse
    //     We can efficiently compute A^T * op(B_sp) using sparse-dense multiplication
    //     (A^T is dense/efficient, B_sp is sparse). We materialize B_sp to dense,
    //     apply A^T, then multiply by Omega^T using left_spmm (sparse * dense).
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Layout layout,
        Op trans_AO,  // Transposition of the composite operator (A*Omega)
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
        // Validate input dimensions
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        auto [rows_AO, cols_AO] = RandBLAS::dims_before_op(m, k, trans_AO);
        randblas_require(rows_AO <= n_rows);
        randblas_require(cols_AO <= n_cols);
        randblas_require(ldc >= m);

        // Check if EffLinOp is a sparse linear operator
        constexpr bool eff_is_sparse = requires(EffLinOp op) {
            typename EffLinOp::T;
            { op.A_sp };
            requires RandBLAS::sparse_data::SparseMatrix<decltype(op.A_sp)>;
        };

        if constexpr (eff_is_sparse) {
            // A_eff is sparse, so convert B_sp to dense to avoid sparse-sparse issues
            int64_t dense_rows = rows_B;
            int64_t dense_cols = cols_B;
            T* B_dense = new T[dense_rows * dense_cols]();
            int64_t ldb = (layout == Layout::ColMajor) ? dense_rows : dense_cols;

            // Convert sparse to dense using utility function
            RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);

            // Use existing dense operator
            (*this)(layout, trans_AO, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);
            delete[] B_dense;

        } else {
            // A_eff is not sparse, use sparse-sparse multiplication approach
            if (trans_AO == Op::NoTrans) {
                // C := alpha * (A * Omega) * op(B_sp) + beta * C
                int64_t temp_rows = Omega_sp.n_cols;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = temp_rows;

                // Step 1: temp := Omega * op(B_sp), dimension (n_cols x n)
                RandBLAS::sparse_data::left_spmm(layout, Op::NoTrans, trans_B, temp_rows, n, k, (T)1.0, Omega_sp, 0, 0, B_sp, 0, 0, (T)0.0, temp_buffer, ldt);

                // Step 2: C := alpha * A * temp + beta * C
                A_eff(layout, Op::NoTrans, Op::NoTrans, m, n, temp_rows, alpha, temp_buffer, ldt, beta, C, ldc);
                delete[] temp_buffer;

            } else {
                // trans_AO == Op::Trans
                // C := alpha * (A * Omega)^T * op(B_sp) + beta * C
                //    = alpha * Omega^T * A^T * op(B_sp) + beta * C
                int64_t temp_rows = A_eff.n_cols;
                T* temp_buffer = new T[temp_rows * n]();
                int64_t ldt = temp_rows;

                // Step 1: Materialize op(B_sp) to dense, then apply A^T
                // Use left_spmm with an identity-like operation to convert sparse to dense
                // This handles trans_B automatically
                int64_t temp2_rows = k;
                T* temp_buffer2 = new T[temp2_rows * n]();
                int64_t ldt2 = temp2_rows;

                // Apply transpose using layout trick when converting to dense
                Layout temp_layout = layout;
                if (trans_B == Op::Trans) {
                    temp_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
                }

                // Convert sparse to dense using utility function
                RandLAPACK::util::sparse_to_dense(B_sp, temp_layout, temp_buffer2);

                // Now apply A^T to dense temp2 (trans_B already handled via layout)
                A_eff(layout, Op::Trans, Op::NoTrans, temp_rows, n, k, (T)1.0, temp_buffer2, ldt2, (T)0.0, temp_buffer, ldt);
                delete[] temp_buffer2;

                // Step 2: C := alpha * Omega^T * temp + beta * C
                RandBLAS::sparse_data::left_spmm(layout, Op::Trans, Op::NoTrans, m, n, temp_rows, alpha, Omega_sp, 0, 0, temp_buffer, ldt, beta, C, ldc);
                delete[] temp_buffer;
            }
        }
    }

    // Augmented sparse operator to allow for left and right multiplication.
    // Side::Left: C := alpha * (A*Omega) * op(B_sp) + beta * C  (delegates to default sparse operator)
    // Side::Right: C := alpha * op(B_sp) * (A*Omega) + beta * C
    template <RandBLAS::sparse_data::SparseMatrix SpMatB>
    void operator()(
        Side side,
        Layout layout,
        Op trans_AO,  // Transposition of the composite operator (A*Omega)
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
            // Left multiplication: delegate to default sparse operator
            (*this)(layout, trans_AO, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);

        } else {  // side == Side::Right
            // Right multiplication: C := alpha * op(B_sp) * op(A*Omega) + beta * C
            // We use the transpose trick: compute C^T := alpha * op(A*Omega)^T * op(B_sp)^T + beta * C^T
            // with swapped layout

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            auto [rows_AO, cols_AO] = RandBLAS::dims_before_op(k, n, trans_AO);
            randblas_require(rows_AO <= n_rows);
            randblas_require(cols_AO <= n_cols);
            randblas_require(ldc >= m);

            // Transpose the operation: swap trans_AO and use opposite layout
            auto trans_trans_AO = (trans_AO == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;

            // Now call the default sparse operator with transposed parameters
            (*this)(trans_layout, trans_trans_AO, trans_B, n, m, k, alpha, B_sp, beta, C, ldc);
        }
    }
};

/*********************************************************/
/*                                                       */
/*                      SpLinOp                          */
/*                                                       */
/*********************************************************/

// Sparse linear operator struct, supplied with a sparse-with-dense matrix multiplication operator 
// and a Frobenius norm function.
template <RandBLAS::sparse_data::SparseMatrix SpMat>
struct SpLinOp {
    using T = typename SpMat::scalar_t;
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    SpMat &A_sp;

    SpLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        SpMat &A_sp
    ) : n_rows(n_rows), n_cols(n_cols), A_sp(A_sp) {
       
    }

    T fro_nrm(
    ) {
        return blas::nrm2(A_sp.nnz, A_sp.vals, 1);
    }

    // Default operator.
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
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(n, k, trans_B);
        randblas_require(ldb >= rows_B);
        auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(m, n, trans_A);
        randblas_require(rows_submat_A <= n_rows);
        randblas_require(cols_submat_A <= n_cols);
        randblas_require(ldc >= m);

        RandBLAS::sparse_data::left_spmm(layout, trans_A, trans_B, m, n, k, alpha, A_sp, 0, 0, B, ldb, beta, C, ldc);
    }

    // Augmented operator to allow for left and right spmm.
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
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(n, k, trans_B);
            randblas_require(ldb >= rows_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(m, n, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);
            randblas_require(ldc >= m);

            RandBLAS::sparse_data::left_spmm(layout, trans_A, trans_B, m, n, k, alpha, A_sp, 0, 0, B, ldb, beta, C, ldc);
        } else {
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, n, trans_B);
            randblas_require(ldb >= rows_B);
            auto [rows_submat_A, cols_submat_A] = RandBLAS::dims_before_op(n, k, trans_A);
            randblas_require(rows_submat_A <= n_rows);
            randblas_require(cols_submat_A <= n_cols);
            randblas_require(ldc >= m);

            auto trans_trans_A = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
            left_spmm(trans_layout, trans_trans_A, trans_B, k, m, n, alpha, A_sp, 0, 0, B, ldb, beta, C, ldc);
        }
    }
};

/*********************************************************/
/*                                                       */
/*                      GenLinOp                         */
/*                                                       */
/*********************************************************/
// General linear operator struct, supplied with a general matrix multiplication operator 
// (through blas::gemm) and a Frobenius norm function.
template <typename T>
struct GenLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    const T* A_buff;
    const int64_t lda;
    const Layout buff_layout;

    GenLinOp(
        const int64_t n_rows,
        const int64_t n_cols,
        const T* A_buff,
        int64_t lda,
        Layout buff_layout
    ) : n_rows(n_rows), n_cols(n_cols), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {
        randblas_require(buff_layout == Layout::ColMajor);
    }

    T fro_nrm(
    ) {
        return lapack::lange(Norm::Fro, n_rows, n_cols, A_buff, lda);
    }

    // Comments on the use of layout in the below operators:
    // Current implementation: requires that the layout of A, B, C is column-major.
    // Intended behavior: the "layout" parameter here is interpreted for (B and C).
    // If layout conflicts with this->buff_layout then we manipulate
    // parameters to blas::gemm to reconcile the different layouts of
    // A vs (B, C).

    // Default operator
    void operator()(
        Layout layout, 
        Op trans_A, 
        Op trans_B, 
        int64_t m, 
        int64_t n, 
        int64_t k, 
        T alpha,  
        T* const B, 
        int64_t ldb, 
        T beta, 
        T* C, 
        int64_t ldc
    ) {
        randblas_require(layout == buff_layout);
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        randblas_require(ldb >= rows_B);
        auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
        randblas_require(rows_A == n_rows);
        randblas_require(cols_A == n_cols);
        randblas_require(ldc >= m);

        blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A_buff, lda, B, ldb, beta, C, ldc);
    }

    // Augmented operator to allow for left and right gemm.
    void operator()(
        Side side,
        Layout layout, 
        Op trans_A, 
        Op trans_B, 
        int64_t m, 
        int64_t n, 
        int64_t k, 
        T alpha,  
        T* const B, 
        int64_t ldb, 
        T beta, 
        T* C, 
        int64_t ldc
    ) {
        if (side == Side::Left) {
            randblas_require(layout == buff_layout);
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
            randblas_require(ldb >= rows_B);
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(m, k, trans_A);
            randblas_require(rows_A == n_rows);
            randblas_require(cols_A == n_cols);
            randblas_require(ldc >= m);

            blas::gemm(layout, trans_A, trans_B, m, n, k, alpha, A_buff, lda, B, ldb, beta, C, ldc);
        } else {
            randblas_require(layout == buff_layout);
            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            randblas_require(ldb >= rows_B);
            auto [rows_A, cols_A] = RandBLAS::dims_before_op(k, n, trans_A);
            randblas_require(rows_A == n_rows);
            randblas_require(cols_A == n_cols);
            randblas_require(ldc >= m);

            auto trans_trans_A = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
            blas::gemm(trans_layout, trans_trans_A, trans_B, m, n, k, alpha, A_buff, lda, B, ldb, beta, C, ldc);
        }
    }
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

/*********************************************************/
/*                                                       */
/*                  ExplicitSymLinOp                     */
/*                                                       */
/*********************************************************/
template <typename T>
struct ExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const Uplo uplo;
    const T* A_buff;
    const int64_t lda;
    const Layout buff_layout;

    ExplicitSymLinOp(
        int64_t dim,
        Uplo uplo,
        const T* A_buff,
        int64_t lda,
        Layout buff_layout
    ) : m(dim), dim(dim), uplo(uplo), A_buff(A_buff), lda(lda), buff_layout(buff_layout) {}

    // Note: the "layout" parameter here is interpreted for (B and C).
    // If layout conflicts with this->buff_layout then we manipulate
    // parameters to blas::symm to reconcile the different layouts of
    // A vs (B, C).
    void operator()(
        Layout layout,
        int64_t n,
        T alpha,
        T* const B,
        int64_t ldb,
        T beta,
        T* C,
        int64_t ldc
    ) {
        randblas_require(ldb >= dim);
        randblas_require(ldc >= dim);
        auto blas_call_uplo = this->uplo;
        if (layout != this->buff_layout)
            blas_call_uplo = (this->uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
        // Reading the "blas_call_uplo" triangle of "this->A_buff" in "layout" order is the same
        // as reading the "this->uplo" triangle of "this->A_buff" in "this->buff_layout" order.
        blas::symm(
            layout, Side::Left, blas_call_uplo, dim, n, alpha,
            this->A_buff, this->lda, B, ldb, beta, C, ldc
        );
    }

    inline T operator()(int64_t i, int64_t j) {
        randblas_require(this->uplo == Uplo::Upper && this->buff_layout == Layout::ColMajor);
        if (i > j) {
            return A_buff[j + i*lda];
        } else {
            return A_buff[i + j*lda];
        }
    }
};

/*********************************************************/
/*                                                       */
/*               RegExplicitSymLinOp                     */
/*                                                       */
/*********************************************************/
template <typename T>
struct RegExplicitSymLinOp {

    using scalar_t = T;
    const int64_t m;
    const int64_t dim;
    const T* A_buff;
    const int64_t lda;
    int64_t num_ops = 1;
    T* regs = nullptr;
    bool _eval_includes_reg;

    static const Uplo uplo = Uplo::Upper;
    static const Layout buff_layout = Layout::ColMajor;

    RegExplicitSymLinOp(
        int64_t dim, const T* A_buff, int64_t lda, T* arg_regs, int64_t arg_num_ops
    ) : m(dim), dim(dim), A_buff(A_buff), lda(lda) {
        randblas_require(lda >= dim);
        _eval_includes_reg = false;
        num_ops = arg_num_ops;
        num_ops = std::max(num_ops, (int64_t) 1);
        regs = new T[num_ops]{};
        std::copy(arg_regs, arg_regs + arg_num_ops, regs);
    }

    RegExplicitSymLinOp(
        int64_t dim, const T* A_buff, int64_t lda, std::vector<T> &arg_regs
    ) : RegExplicitSymLinOp<T>(dim, A_buff, lda, arg_regs.data(), static_cast<int64_t>(arg_regs.size())) {}

    ~RegExplicitSymLinOp() {
        if (regs != nullptr) delete [] regs;
    }

    void set_eval_includes_reg(bool eir) {
        _eval_includes_reg = eir;
    }

    void operator()(Layout layout, int64_t n, T alpha, T* const B, int64_t ldb, T beta, T* C, int64_t ldc) {
        randblas_require(layout == this->buff_layout);
        randblas_require(ldb >= dim);
        randblas_require(ldc >= dim);
        blas::symm(layout, blas::Side::Left, this->uplo, dim, n, alpha, this->A_buff, this->lda, B, ldb, beta, C, ldc);

        if (_eval_includes_reg) {
            if (num_ops != 1) randblas_require(n == num_ops);
            for (int64_t i = 0; i < n; ++i) {
                T coeff =  alpha * regs[std::min(i, num_ops - 1)];
                blas::axpy(dim, coeff, B + i*ldb, 1, C +  i*ldc, 1);
            }
        }
        return;
    }

    inline T operator()(int64_t i, int64_t j) {
        T val;
        if (i > j) {
            val = A_buff[j + i*lda];
        } else {
            val = A_buff[i + j*lda];
        }
        if (_eval_includes_reg && i == j) {
            randblas_require(num_ops == 1);
            val += regs[0];
        }
        return val;
    }

};

/*********************************************************/
/*                                                       */
/*                  SpectralPrecond                      */
/*                                                       */
/*********************************************************/
template<typename T>
struct SpectralPrecond {

    using scalar_t = T; 
    const int64_t m; // an alias for dim, keeping for backward compatibility reasons.
    const int64_t dim;
    int64_t dim_pre;
    int64_t num_rhs;
    T* V = nullptr;
    T* D = nullptr;
    T* W = nullptr;
    int64_t num_ops = 0;

    /* Suppose we want to precondition a positive semidefinite matrix G_mu = G + mu*I.
     *
     * Once properly preparred, this preconditioner represents a linear operator of the form
     *      P = V diag(D) V' + I.
     * The columns of V approximate the top dim_pre eigenvectors of G, while the 
     * entries of D are *functions of* the corresponding approximate eigenvalues.
     * 
     * The specific form of the entries of D are as follows. Suppose we start with
     * (V, lambda) as approximations of the top dim_pre eigenpairs of G, and define the vector
     *      D0 = (min(lambda) + mu) / (lambda + mu).
     * From a mathematical perspective, this preconditioner represents the linear operator
     *      P = V diag(D0) V' + (I - VV').
     * The action of this linear operator can be computed with two calls to GEMM
     * instead of three if we store D = D0 - 1 instead of D0 itself.
     */

    SpectralPrecond(int64_t dim) : m(dim), dim(dim), dim_pre(0), num_rhs(0) {}

    // Move constructor
    // Call as SpectralPrecond<T> spc(std::move(other)) when we want to transfer the
    // contents of "other" to "this". 
    SpectralPrecond(SpectralPrecond &&other) noexcept
        : m(other.dim), dim(other.dim), dim_pre(other.dim_pre), num_rhs(other.num_rhs), num_ops(other.num_ops)
    {
        std::swap(V, other.V);
        std::swap(D, other.D);
        std::swap(W, other.W);
    }

    // Copy constructor
    // Call as SpectralPrecond<T> spc(other) when we want to copy "other".
    SpectralPrecond(const SpectralPrecond &other)
        : m(other.dim), dim(other.dim), dim_pre(other.dim_pre), num_rhs(other.num_rhs),  num_ops(other.num_ops)
     {
        reset_owned_buffers(dim_pre, num_rhs, num_ops);
        std::copy(other.V, other.V + dim * dim_pre,        V);
        std::copy(other.D, other.D + dim_pre * num_ops, D);
     } 

    ~SpectralPrecond() {
        if (D != nullptr) delete [] D;
        if (V != nullptr) delete [] V;
        if (W != nullptr) delete [] W;
    }

    void reset_owned_buffers(int64_t arg_dim_pre, int64_t arg_num_rhs, int64_t arg_num_ops) {
        randblas_require(arg_num_rhs == arg_num_ops || arg_num_ops == 1);

        if (arg_dim_pre * arg_num_ops > dim_pre * num_ops) {
            if (D != nullptr) delete [] D;
            D = new T[arg_dim_pre * arg_num_ops]{};
        } 
        if (arg_dim_pre > dim_pre) {
            if (V != nullptr) delete [] V;
            V = new T[dim * arg_dim_pre];
        }
        if (arg_dim_pre * arg_num_rhs > dim_pre * num_rhs) {
            if (W != nullptr) delete [] W;
            W = new T[arg_dim_pre * arg_num_rhs];
        }

        dim_pre = arg_dim_pre;
        num_rhs = arg_num_rhs;
        num_ops = arg_num_ops;
    }

    void set_D_from_eigs_and_regs(T* eigvals, T* mus) {
        for (int64_t r = 0; r < num_ops; ++r) {
            T  mu_r = mus[r];
            T* D_r  = D + r*dim_pre;
            T  numerator = eigvals[dim_pre-1] + mu_r;
            for (int i = 0; i < dim_pre; ++i) {
                D_r[i] = (numerator / (eigvals[i] + mu_r)) - 1.0;
            }
        }
        return;
    }

    void prep(std::vector<T> &eigvecs, std::vector<T> &eigvals, std::vector<T> &mus, int64_t arg_num_rhs) {
        // assume eigvals are positive numbers sorted in decreasing order.
        int64_t arg_num_ops = mus.size();
        int64_t arg_dim_pre  = eigvals.size();
        reset_owned_buffers(arg_dim_pre, arg_num_rhs, arg_num_ops);
        set_D_from_eigs_and_regs(eigvals.data(), mus.data());
        std::copy(eigvecs.begin(), eigvecs.end(), V);
        return;
    }

    void operator()(
        Layout layout, int64_t n, T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc
    ) {
        randblas_require(layout == Layout::ColMajor);
        randblas_require(ldb >= dim);
        randblas_require(ldc >= dim);
        if (this->num_ops != 1) {
            randblas_require(n == num_ops);
        } else {
            randblas_require(this->num_rhs >= n);
        }
        // update C = alpha*(V diag(D) V' + I)B + beta*C
        //      Step 1: w = V'B                    with blas::gemm
        //      Step 2: w = D w                    with our own kernel
        //      Step 3: C = beta * C + alpha * B   with blas::copy or blas::scal + blas::axpy
        //      Step 4: C = alpha * V w + C        with blas::gemm
        blas::gemm(layout, blas::Op::Trans, blas::Op::NoTrans, dim_pre, n, dim, (T) 1.0, V, dim, B, ldb, (T) 0.0, W, dim_pre);
 
        // -----> start step 2
        #define mat_D(_i, _j)  ((num_ops == 1) ? D[(_i)] : D[(_i) + dim_pre*(_j)])
        #define mat_W(_i, _j)  W[(_i) + dim_pre*(_j)]
        for (int64_t j = 0; j < n; j++) {
            for (int64_t i = 0; i < dim_pre; i++) {
                mat_W(i, j) = mat_D(i, j) * mat_W(i, j);
            }
        }
        #undef mat_D
        #undef mat_W
        // <----- end step 2

        // -----> start step 3
        int64_t i;
        #define colB(_i) &B[(_i)*ldb]
        #define colC(_i) &C[(_i)*ldb]
        if (beta == (T) 0.0 && alpha == (T) 1.0) {
            for (i = 0; i < n; ++i)
                blas::copy(dim, colB(i), 1, colC(i), 1);
        } else {
            for (i = 0; i < n; ++i) {
                T* Ci = colC(i);
                blas::scal(dim, beta, Ci, 1);
                blas::axpy(dim, alpha, colB(i), 1, Ci, 1);
            }
        }
        #undef colB
        #undef colC
        // <----- end step 3
    
        blas::gemm(layout, blas::Op::NoTrans, blas::Op::NoTrans, dim, n, dim_pre, (T) 1.0, V, dim, W, dim_pre, 1.0, C, ldc);
        return;
    }
};

} // end namespace RandLAPACK::linops
