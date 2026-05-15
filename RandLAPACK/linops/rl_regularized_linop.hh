#pragma once

// Public API: RegularizedLinOp — wraps a tall LinearOperator J as the
// augmented (m+n) × n operator
//
//                       ⎡ J  ⎤
//             A_aug  =  ⎣ λI ⎦
//
// so that QR(A_aug) succeeds even when J is highly ill-conditioned (σ_min(J)
// can be anything, but σ_min(A_aug) ≥ λ by construction). Solving
// min ||A_aug x − [b; 0]||₂² is mathematically equivalent to the Tikhonov-
// regularized LS problem  min ||J x − b||₂² + λ²||x||₂². This is the
// standard augmentation trick for handling ill-posed inverse problems.

#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <vector>


namespace RandLAPACK::linops {

/*********************************************************/
/*                                                       */
/*                  RegularizedLinOp                     */
/*                                                       */
/*********************************************************/

/// @brief Tikhonov-augmented LinearOperator wrapper: A_aug = [J; λI].
///
/// @tparam JLO  Underlying tall LinearOperator type. Must satisfy
///              `LinearOperator` and provide both forward + adjoint apply
///              and a Side::Right + SkOp overload (used by the sketch step
///              of CQRRT_linops).
template <typename JLO>
struct RegularizedLinOp {
    using scalar_t = typename JLO::scalar_t;
    using T = scalar_t;

    JLO& J;
    const T lambda;
    const int64_t m_J;        ///< J.n_rows
    const int64_t n_J;        ///< J.n_cols
    const int64_t n_rows;     ///< m_J + n_J
    const int64_t n_cols;     ///< n_J

    RegularizedLinOp(JLO& J_in, T lambda_in)
        : J(J_in), lambda(lambda_in),
          m_J(J_in.n_rows), n_J(J_in.n_cols),
          n_rows(J_in.n_rows + J_in.n_cols),
          n_cols(J_in.n_cols) {}

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
    // Dense matmul with Side
    //   Side::Left:  C = alpha * op(A_aug) * B + beta * C
    //   Side::Right: C = alpha * B * op(A_aug) + beta * C
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
    //
    // Side::Right, NoTrans/NoTrans corresponds to the CQRRT_linops sketch
    // step `A_hat = S * A_aug` where S is d × (m_J+n_J).
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
        randblas_require(k == m_J + n_J);
        randblas_require(n == n_J);
        randblas_require(ldc >= m);

        if constexpr (requires { S.buff; S.layout; S.dist; }) {
            // Dense SkOp: split S.buff column-wise. Forwards to dense Side::Right.
            if (S.buff == nullptr) RandBLAS::fill_dense(S);
            int64_t lds = S.dist.dim_major;
            randblas_require(S.layout == Layout::ColMajor);
            apply_right_dense(Op::NoTrans, m, n, k, alpha, S.buff, lds, beta, C, ldc);
        } else {
            // Sparse SASO path. Partition S's COO into J-touching nonzeros
            // (col < m_J) and I-touching nonzeros (col ≥ m_J), then dispatch
            // each subset in bulk:
            //   J-touching → one call to J.apply_right_sparse_coo (assumes
            //                JLO exposes it; KroneckerOperator does)
            //   I-touching → scalar updates C[i, p − m_J] += α·val·λ
            // This avoids the per-nonzero J-adjoint apply that scales as
            // O(nnz_J × cost(J adj)) and dominated runtime at moderate n.
            if (S.nnz < 0) RandBLAS::fill_sparse(S);
            auto S_coo = RandBLAS::coo_view_of_skop(S);
            using sint_t = std::remove_pointer_t<decltype(S_coo.rows)>;

            // First pass: count J-touching nonzeros.
            int64_t nnz_J = 0;
            for (int64_t l = 0; l < S_coo.nnz; ++l) {
                if (static_cast<int64_t>(S_coo.cols[l]) < m_J) ++nnz_J;
            }
            int64_t nnz_I = S_coo.nnz - nnz_J;

            // Second pass: copy J-touching into local arrays.
            std::vector<sint_t> J_rows; J_rows.reserve(nnz_J);
            std::vector<sint_t> J_cols; J_cols.reserve(nnz_J);
            std::vector<T>      J_vals; J_vals.reserve(nnz_J);
            // I-touching: collected into parallel arrays for the scalar pass.
            std::vector<int64_t> I_rows; I_rows.reserve(nnz_I);
            std::vector<int64_t> I_cols; I_cols.reserve(nnz_I);
            std::vector<T>       I_vals; I_vals.reserve(nnz_I);
            for (int64_t l = 0; l < S_coo.nnz; ++l) {
                int64_t p = static_cast<int64_t>(S_coo.cols[l]);
                if (p < m_J) {
                    J_rows.push_back(S_coo.rows[l]);
                    J_cols.push_back(S_coo.cols[l]);
                    J_vals.push_back(S_coo.vals[l]);
                } else {
                    I_rows.push_back(S_coo.rows[l]);
                    I_cols.push_back(p - m_J);
                    I_vals.push_back(S_coo.vals[l]);
                }
            }

            // J-touching: bulk dispatch via J's COO entry point.
            // The call writes  C ← α·(S_J · J) + β·C  in one pass.
            J.apply_right_sparse_coo(m, n_J, alpha, beta,
                                     nnz_J, J_rows.data(), J_cols.data(), J_vals.data(),
                                     C, ldc);

            // I-touching: scalar updates  C[i, col] += α·val·λ.
            // Note: the β scaling above already applied to the entire C, so
            // we just accumulate (no need to re-apply β).
            T scale = alpha * lambda;
            for (int64_t l = 0; l < nnz_I; ++l) {
                C[I_rows[l] + I_cols[l] * ldc] += scale * I_vals[l];
            }
        }
    }

private:
    // -----------------------------------------------------------------
    // Side::Left dense apply: C = alpha * op(A_aug) * B + beta * C
    //
    // NoTrans: C is (m_J + n_J) × n_blas.
    //   C_top (first m_J rows) ← alpha * J * B + beta * C_top
    //   C_bot (last  n_J rows) ← alpha * λ * B + beta * C_bot
    //
    // Trans: C is n_J × n_blas, B is (m_J + n_J) × n_blas.
    //   C ← alpha * (J^T * B_top + λ * B_bot) + beta * C
    // -----------------------------------------------------------------
    void apply_left_dense(Op trans_A, int64_t m, int64_t n, int64_t k,
                          T alpha, const T* B, int64_t ldb,
                          T beta, T* C, int64_t ldc)
    {
        if (trans_A == Op::NoTrans) {
            randblas_require(m == n_rows);
            randblas_require(k == n_cols);
            randblas_require(ldc >= n_rows);

            // Top: C[0:m_J, :] = alpha * J * B + beta * C[0:m_J, :]
            J(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
              m_J, n, n_J, alpha, B, ldb, beta, C, ldc);

            // Bottom: C[m_J:m_J+n_J, :] = alpha * λ * B + beta * C[m_J:..., :]
            for (int64_t j = 0; j < n; ++j) {
                T*       Cj_bot = C + j * ldc + m_J;
                const T* Bj     = B + j * ldb;
                T scale = alpha * lambda;
                if (beta == (T)0) {
                    for (int64_t i = 0; i < n_J; ++i) Cj_bot[i] = scale * Bj[i];
                } else if (beta == (T)1) {
                    for (int64_t i = 0; i < n_J; ++i) Cj_bot[i] += scale * Bj[i];
                } else {
                    for (int64_t i = 0; i < n_J; ++i) Cj_bot[i] = scale * Bj[i] + beta * Cj_bot[i];
                }
            }
        } else {
            // Trans: m == n_cols == n_J, k == n_rows == m_J + n_J
            randblas_require(m == n_cols);
            randblas_require(k == n_rows);
            randblas_require(ldb >= n_rows);

            // First: C = alpha * J^T * B_top + beta * C
            //   B_top is the top m_J rows of B (length-m_J vectors), stride ldb.
            J(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
              n_J, n, m_J, alpha, B, ldb, beta, C, ldc);

            // Then: C += alpha * λ * B_bot, where B_bot starts at row m_J of B.
            for (int64_t j = 0; j < n; ++j) {
                T*       Cj     = C + j * ldc;
                const T* Bj_bot = B + j * ldb + m_J;
                T scale = alpha * lambda;
                for (int64_t i = 0; i < n_J; ++i) Cj[i] += scale * Bj_bot[i];
            }
        }
    }

    // -----------------------------------------------------------------
    // Side::Right dense apply: C = alpha * B * op(A_aug) + beta * C
    //
    // NoTrans on A_aug: B is p × (m_J + n_J), output C is p × n_J.
    //   B = [B1 | B2], B1 = p × m_J, B2 = p × n_J.
    //   C = alpha * (B1 * J + λ * B2) + beta * C.
    // -----------------------------------------------------------------
    void apply_right_dense(Op trans_A, int64_t m, int64_t n, int64_t k,
                           T alpha, const T* B, int64_t ldb,
                           T beta, T* C, int64_t ldc)
    {
        randblas_require(trans_A == Op::NoTrans);  // Trans on Side::Right is uncommon
        randblas_require(k == n_rows);
        randblas_require(n == n_cols);
        randblas_require(ldb >= n_rows);
        randblas_require(ldc >= m);

        // First: C = alpha * B1 * J + beta * C   (B1 is the leftmost m_J cols of B)
        // J's Side::Right takes a (p × m_J) dense block and returns p × n_J.
        J(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          m, n, m_J, alpha, B, ldb, beta, C, ldc);

        // Then: C += alpha * λ * B2  where B2 starts at column m_J of B.
        T scale = alpha * lambda;
        for (int64_t j = 0; j < n; ++j) {
            T*       Cj     = C + j * ldc;
            const T* Bj_2   = B + (m_J + j) * ldb;
            // B2 is (m × n_J), col-major. B2[:, j] starts at B + (m_J + j)*ldb.
            for (int64_t i = 0; i < m; ++i) Cj[i] += scale * Bj_2[i];
        }
    }
};

} // end namespace RandLAPACK::linops
