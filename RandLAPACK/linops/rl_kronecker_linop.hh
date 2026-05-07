#pragma once

// Public API: KroneckerOperator — implicit Kronecker-product linear operator
//                                 A = A2 ⊗ A1 with two small dense factors.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                  KroneckerOperator                    */
/*                                                       */
/*********************************************************/

/// @brief Linear operator representing A = A2 ⊗ A1, with small dense factors.
///
/// @details Applies the Kronecker product implicitly via the identity
///     (A2 ⊗ A1) vec(X) = vec(A1 X A2^T),  with X = reshape(x, n1, n2).
/// Each matvec is two GEMM calls (one with A1, one with A2). The full
/// operator A is m1*m2 × n1*n2 and is never materialized. Both factors are
/// stored in column-major layout with leading dimensions m1 (for A1) and
/// m2 (for A2).
///
/// Use case: tall, structured least-squares problems whose forward operator
/// has Kronecker structure — most notably 2D NMR relaxometry (PRnmr in the
/// IR Tools toolbox) and any separable-kernel Fredholm integral equation.
///
/// Storage: this operator owns its A1 and A2 buffers (allocated in the
/// constructor, freed by the destructor). The factors are typically tiny
/// (e.g., 256×128 each at default NMR scale) compared to the implicit
/// m1*m2 × n1*n2 operator.
///
/// Supported call patterns:
///   - Side::Left, NoTrans/NoTrans, dense B    — forward apply (per-col loop)
///   - Side::Left, Trans/NoTrans,   dense B    — adjoint apply (per-col loop)
///   - Side::Right, NoTrans/NoTrans, dense B   — used by Side::Right SkOp path
///   - Side::Right, NoTrans/NoTrans, sparse SkOp — sketch step (nnz-loop)
///
/// Limitations: Side::Right with general dense B uses a row-loop dispatch;
/// other configurations (trans_B != NoTrans, sketch with dense SkOp, etc.)
/// will throw via randblas_require.
template <typename T>
struct KroneckerOperator {
    using scalar_t = T;
    const int64_t n_rows;     ///< m1 * m2
    const int64_t n_cols;     ///< n1 * n2
    const int64_t m1, n1, m2, n2;
    T* A1;                    ///< m1 × n1, ColMajor, lda = m1, owned
    T* A2;                    ///< m2 × n2, ColMajor, lda = m2, owned

    /// Construct a Kronecker operator. The constructor copies A1 and A2 into
    /// internally-owned buffers (the source pointers may be freed afterwards).
    KroneckerOperator(int64_t m1_, int64_t n1_, int64_t m2_, int64_t n2_,
                      const T* A1_buff, const T* A2_buff)
        : n_rows(m1_ * m2_), n_cols(n1_ * n2_),
          m1(m1_), n1(n1_), m2(m2_), n2(n2_)
    {
        A1 = new T[m1 * n1];
        A2 = new T[m2 * n2];
        std::copy(A1_buff, A1_buff + m1 * n1, A1);
        std::copy(A2_buff, A2_buff + m2 * n2, A2);
    }

    ~KroneckerOperator() {
        delete[] A1;
        delete[] A2;
    }

    KroneckerOperator(const KroneckerOperator&) = delete;
    KroneckerOperator& operator=(const KroneckerOperator&) = delete;

    /// @brief Frobenius norm: ||A2 ⊗ A1||_F = ||A1||_F * ||A2||_F.
    T fro_nrm() {
        T n1_fro = blas::nrm2(m1 * n1, A1, 1);
        T n2_fro = blas::nrm2(m2 * n2, A2, 1);
        return n1_fro * n2_fro;
    }

    // -----------------------------------------------------------------
    // Concept-required overload (no Side; defaults to Side::Left)
    // -----------------------------------------------------------------
    void operator()(
        Layout layout, Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    // -----------------------------------------------------------------
    // Dense matmul with explicit Side
    //   Side::Left:  C = alpha * op(A) * B + beta * C
    //   Side::Right: C = alpha * B * op(A) + beta * C
    // -----------------------------------------------------------------
    void operator()(
        Side side, Layout layout, Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb, T beta, T* C, int64_t ldc)
    {
        randblas_require(layout == Layout::ColMajor);
        randblas_require(trans_B == Op::NoTrans);

        if (side == Side::Left) {
            apply_left_dense(trans_A, m, n, k, alpha, B, ldb, beta, C, ldc);
        } else {
            apply_right_dense(trans_A, m, n, k, alpha, B, ldb, beta, C, ldc);
        }
    }

    // -----------------------------------------------------------------
    // Sketching-operator overloads
    // -----------------------------------------------------------------
    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Layout layout, Op trans_A, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S, T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Right, layout, trans_A, trans_S, m, n, k, alpha, S, beta, C, ldc);
    }

    template <RandBLAS::SketchingOperator SkOp>
    void operator()(
        Side side, Layout layout, Op trans_A, Op trans_S,
        int64_t m, int64_t n, int64_t k,
        T alpha, SkOp& S, T beta, T* C, int64_t ldc)
    {
        randblas_require(layout == Layout::ColMajor);
        randblas_require(side == Side::Right);
        randblas_require(trans_A == Op::NoTrans);
        randblas_require(trans_S == Op::NoTrans);
        // Side::Right, NoTrans/NoTrans: C = alpha * S * A + beta * C
        // m = rows of C = rows of S = d
        // n = cols of C = cols of A = n1*n2
        // k = inner = cols of S = rows of A = m1*m2
        randblas_require(k == m1 * m2);
        randblas_require(n == n1 * n2);
        randblas_require(ldc >= m);

        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            // Dense SkOp: forward to the dense Side::Right path using S.buff.
            if (S.buff == nullptr) RandBLAS::fill_dense(S);
            int64_t lds = S.dist.dim_major;
            // RandBLAS DenseSkOp may be RowMajor — densely-stored SkOps store data
            // with a single layout. If S.layout != ColMajor, treat the buffer as a
            // ColMajor view of S^T (which is m × d), and apply with trans_S = Trans.
            // Currently we only need NoTrans so require matching layout.
            randblas_require(S.layout == Layout::ColMajor);
            apply_right_dense(Op::NoTrans, m, n, k, alpha, S.buff, lds, beta, C, ldc);
        } else {
            // Sparse SkOp (SASO): iterate nnz directly. No materialization.
            if (S.nnz < 0) RandBLAS::fill_sparse(S);
            auto S_coo = RandBLAS::coo_view_of_skop(S);
            apply_right_sparse_coo(m, n, alpha, beta,
                                   S_coo.nnz, S_coo.rows, S_coo.cols, S_coo.vals,
                                   C, ldc);
        }
    }

private:
    // -----------------------------------------------------------------
    // Side::Left dense apply: C = alpha * op(A) * B + beta * C
    //
    // Per-column loop. For each column j of B (length n1*n2 in NoTrans case
    // or m1*m2 in Trans case), reshape into a small matrix, apply two GEMMs.
    // Math:
    //   NoTrans: A x = vec(A1 X A2^T),       X = reshape(x, n1, n2),  output m1×m2
    //   Trans:   A^T y = vec(A1^T Y A2),     Y = reshape(y, m1, m2),  output n1×n2
    // -----------------------------------------------------------------
    void apply_left_dense(Op trans_A, int64_t m, int64_t n, int64_t k,
                          T alpha, const T* B, int64_t ldb,
                          T beta, T* C, int64_t ldc)
    {
        int64_t rows_in, cols_in, rows_out, cols_out;
        Op trans_first, trans_second;
        const T* fac1;  int64_t lda_fac1;
        const T* fac2;  int64_t lda_fac2;

        if (trans_A == Op::NoTrans) {
            // dims-before-op for op(A): (m, k) = (n_rows, n_cols)
            randblas_require(m == n_rows);
            randblas_require(k == n_cols);
            rows_in = n1;  cols_in = n2;
            rows_out = m1; cols_out = m2;
            // GEMM 1: Tmp = A1 * X        (m1 × n2) <- (m1 × n1) * (n1 × n2)
            // GEMM 2: Y = Tmp * A2^T      (m1 × m2) <- (m1 × n2) * (n2 × m2)
            trans_first = Op::NoTrans;   fac1 = A1; lda_fac1 = m1;
            trans_second = Op::Trans;    fac2 = A2; lda_fac2 = m2;
        } else {
            // dims-before-op for op(A) = A^T: (m, k) = (n_cols, n_rows)
            randblas_require(m == n_cols);
            randblas_require(k == n_rows);
            rows_in = m1;  cols_in = m2;
            rows_out = n1; cols_out = n2;
            // GEMM 1: Tmp = A1^T * Y      (n1 × m2) <- (n1 × m1) * (m1 × m2)
            // GEMM 2: Z = Tmp * A2        (n1 × n2) <- (n1 × m2) * (m2 × n2)
            trans_first = Op::Trans;     fac1 = A1; lda_fac1 = m1;
            trans_second = Op::NoTrans;  fac2 = A2; lda_fac2 = m2;
        }

        randblas_require(ldb >= rows_in * cols_in);  // B columns are length k = rows_in*cols_in
        randblas_require(ldc >= rows_out * cols_out);

        std::vector<T> Tmp(static_cast<size_t>(rows_out) * cols_in);

        for (int64_t j = 0; j < n; ++j) {
            const T* Bj = B + j * ldb;     // n1*n2 (or m1*m2) vector
            T*       Cj = C + j * ldc;     // m1*m2 (or n1*n2) vector

            // X = reshape(Bj, rows_in, cols_in)  ← view, no copy (lda = rows_in)
            // Tmp = op(fac1) * X      (rows_out × cols_in)
            blas::gemm(
                Layout::ColMajor,
                trans_first, Op::NoTrans,
                rows_out, cols_in, rows_in,
                (T)1.0,
                fac1, lda_fac1,
                Bj, rows_in,
                (T)0.0,
                Tmp.data(), rows_out
            );
            // Y = alpha * Tmp * op(fac2)^T     (rows_out × cols_out)
            // (where the "T" depends on direction: Trans for forward, NoTrans for adjoint)
            blas::gemm(
                Layout::ColMajor,
                Op::NoTrans, trans_second,
                rows_out, cols_out, cols_in,
                alpha,
                Tmp.data(), rows_out,
                fac2, lda_fac2,
                beta,
                Cj, rows_out
            );
        }
    }

    // -----------------------------------------------------------------
    // Side::Right dense apply: C = alpha * B * op(A) + beta * C
    // Implementation: row-by-row of B, delegate to Side::Left with flipped trans_A.
    //   C[i, :] = (B[i, :]) * op(A) = (op(A)^T * B[i, :]^T)^T
    // -----------------------------------------------------------------
    void apply_right_dense(Op trans_A, int64_t m, int64_t n, int64_t k,
                           T alpha, const T* B, int64_t ldb,
                           T beta, T* C, int64_t ldc)
    {
        int64_t rows_op_A, cols_op_A;
        if (trans_A == Op::NoTrans) {
            rows_op_A = n_rows; cols_op_A = n_cols;
        } else {
            rows_op_A = n_cols; cols_op_A = n_rows;
        }
        randblas_require(k == rows_op_A);
        randblas_require(n == cols_op_A);

        Op flipped = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;

        std::vector<T> b_row(k);
        std::vector<T> c_row(n);

        for (int64_t i = 0; i < m; ++i) {
            // b_row = B[i, :]  (gather across the i-th row in ColMajor B)
            for (int64_t j = 0; j < k; ++j) b_row[j] = B[i + j * ldb];

            // c_row = op(A)^T * b_row  (Side::Left with flipped trans_A, single column)
            apply_left_dense(flipped, n, /*ncols*/1, k,
                             (T)1.0, b_row.data(), k,
                             (T)0.0, c_row.data(), n);

            // C[i, :] = alpha * c_row + beta * C[i, :]
            for (int64_t j = 0; j < n; ++j) {
                T& ce = C[i + j * ldc];
                ce = alpha * c_row[j] + beta * ce;
            }
        }
    }

public:
    // -----------------------------------------------------------------
    // Sparse-COO sketch apply: C = alpha * S * A + beta * C
    // S is given as a COO view (rows[l], cols[l], vals[l]) with `nnz` entries.
    // For each nonzero (i, p, val): C[i, :] += alpha * val * A[p, :], where
    // A[p, j] = A1[p_inner, j_inner] * A2[p_outer, j_outer], with the standard
    // Kronecker convention p = p_outer*m1 + p_inner, j = j_outer*n1 + j_inner.
    //
    // Public so wrapper operators (e.g., RegularizedLinOp) can dispatch a
    // pre-filtered COO directly without re-materializing through SparseSkOp.
    // -----------------------------------------------------------------
    template <typename SInt>
    void apply_right_sparse_coo(int64_t m, int64_t n,
                                T alpha, T beta,
                                int64_t nnz, const SInt* rows, const SInt* cols, const T* vals,
                                T* C, int64_t ldc)
    {
        // Scale C by beta first.
        if (beta == (T)0) {
            for (int64_t j = 0; j < n; ++j)
                std::fill(C + j * ldc, C + j * ldc + m, (T)0);
        } else if (beta != (T)1) {
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i < m; ++i)
                    C[i + j * ldc] *= beta;
        }

        // For each nonzero, accumulate alpha * val * outer_prod(A1[p_inner,:], A2[p_outer,:])
        // into C[i, :] viewed as n1 × n2.
        for (int64_t l = 0; l < nnz; ++l) {
            int64_t i = rows[l];
            int64_t p = cols[l];
            T av = alpha * vals[l];

            int64_t p_outer = p / m1;
            int64_t p_inner = p % m1;

            // C[i, j_outer*n1 + j_inner] += av * A1[p_inner, j_inner] * A2[p_outer, j_outer]
            for (int64_t j_outer = 0; j_outer < n2; ++j_outer) {
                T scaled = av * A2[p_outer + j_outer * m2];
                if (scaled == (T)0) continue;
                T* C_col_block = C + (j_outer * n1) * ldc + i;  // start of i-th row, j_outer*n1-th col block
                for (int64_t j_inner = 0; j_inner < n1; ++j_inner) {
                    C_col_block[j_inner * ldc] += scaled * A1[p_inner + j_inner * m1];
                }
            }
        }
    }
};

} // end namespace RandLAPACK::linops
