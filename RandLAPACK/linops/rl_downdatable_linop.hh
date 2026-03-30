#pragma once

// Public API: DowndatableLinOp — a linear operator that implicitly represents
// A - Q * BT^T, where Q and BT accumulate columns across QB iterations.
//
// This enables linop-based QB/RSVD: QB calls operator() for all matmuls
// (including inside RS power iteration), and calls update() instead of
// explicit deflation (A = A - Q_i * B_i^T).
//
// Matvec cost: base operator matmul + O(curr_rank * n_rhs * max(m,n)).
// This is negligible when curr_rank << min(m,n).

#include "rl_concepts.hh"
#include "rl_blaspp.hh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace RandLAPACK::linops {

template <typename T, LinearOperator BaseLinOp>
struct DowndatableLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;

    // Non-copyable (owns raw buffers)
    DowndatableLinOp(const DowndatableLinOp&) = delete;
    DowndatableLinOp& operator=(const DowndatableLinOp&) = delete;

    BaseLinOp& base_op;     // The underlying A operator (read-only)

    // Deflation data: the operator represents base_op - Q * BT^T.
    // Q is m x curr_rank (ColMajor, ld m), BT is n x curr_rank (ColMajor, ld n).
    T* Q_data;
    T* BT_data;
    int64_t curr_rank;
    int64_t max_rank;

    // Scratch buffer for intermediate products (reused across calls).
    // Size: max(m, n) * max_rank — allocated lazily on first matmul.
    T* scratch;

    DowndatableLinOp(BaseLinOp& base, int64_t max_rank)
        : n_rows(base.n_rows), n_cols(base.n_cols), base_op(base),
          curr_rank(0), max_rank(max_rank)
    {
        int64_t m = base.n_rows;
        int64_t n = base.n_cols;
        Q_data  = (T*) calloc(m * max_rank, sizeof(T));
        BT_data = (T*) calloc(n * max_rank, sizeof(T));
        scratch = (T*) calloc(std::max(m, n) * max_rank, sizeof(T));
    }

    ~DowndatableLinOp() {
        free(Q_data);
        free(BT_data);
        free(scratch);
    }

    // Append b_sz new columns to Q and BT (one QB iteration's worth).
    void update(int64_t b_sz, const T* Q_new, const T* BT_new) {
        int64_t m = n_rows;
        int64_t n = n_cols;
        // Copy Q_new (m x b_sz) into Q_data columns [curr_rank, curr_rank + b_sz)
        lapack::lacpy(MatrixType::General, m, b_sz, Q_new, m,
                      &Q_data[m * curr_rank], m);
        // Copy BT_new (n x b_sz) into BT_data columns [curr_rank, curr_rank + b_sz)
        lapack::lacpy(MatrixType::General, n, b_sz, BT_new, n,
                      &BT_data[n * curr_rank], n);
        curr_rank += b_sz;
    }

    // GEMM-like operator: C := alpha * op(A - Q*BT^T) * op(B) + beta * C
    //
    // For NoTrans:
    //   C = alpha * A * op(B) + beta * C  -  alpha * Q * (BT^T * op(B))
    //
    // For Trans:
    //   C = alpha * A^T * op(B) + beta * C  -  alpha * BT * (Q^T * op(B))
    //
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
        // Step 1: C = alpha * op(base_op) * op(B) + beta * C
        base_op(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);

        // Step 2: subtract the deflation term (only if we have accumulated columns)
        if (curr_rank == 0) return;

        if (trans_A == Op::NoTrans) {
            // C -= alpha * Q * (BT^T * op(B))
            // scratch (curr_rank x n) = BT^T * op(B)
            //   BT is n_cols x curr_rank, so BT^T is curr_rank x n_cols
            blas::gemm(layout, Op::Trans, trans_B, curr_rank, n, n_cols,
                       (T)1.0, BT_data, n_cols, B, ldb, (T)0.0, scratch, curr_rank);
            // C -= alpha * Q * scratch
            //   Q is n_rows x curr_rank, scratch is curr_rank x n
            blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n, curr_rank,
                       -alpha, Q_data, n_rows, scratch, curr_rank, (T)1.0, C, ldc);
        } else {
            // C -= alpha * BT * (Q^T * op(B))
            // scratch (curr_rank x n) = Q^T * op(B)
            //   Q is n_rows x curr_rank, so Q^T is curr_rank x n_rows
            blas::gemm(layout, Op::Trans, trans_B, curr_rank, n, n_rows,
                       (T)1.0, Q_data, n_rows, B, ldb, (T)0.0, scratch, curr_rank);
            // C -= alpha * BT * scratch
            //   BT is n_cols x curr_rank, scratch is curr_rank x n
            blas::gemm(layout, Op::NoTrans, Op::NoTrans, m, n, curr_rank,
                       -alpha, BT_data, n_cols, scratch, curr_rank, (T)1.0, C, ldc);
        }
    }

    // Compute Frobenius norm of the base operator (deflation not accounted for).
    // This is used by QB for the initial norm_A computation.
    T fro_nrm() {
        return base_op.fro_nrm();
    }
};

} // end namespace RandLAPACK::linops
