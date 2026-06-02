#pragma once

// SparseLUSolverLinOp — sibling to CholSolverLinOp for general (possibly
// indefinite) sparse matrices.  Wraps Eigen::SparseLU with the factor-once /
// solve-many pattern: factorize() runs the sparse LU once, subsequent
// operator() calls apply A^{-1} (or A^{-T}) to dense right-hand-sides.
//
// Target use: the reduced-spectral application.  When omega is an interior
// shift, X = K - omega*M becomes indefinite and Cholesky no longer factors;
// SparseLU handles both the SPD and indefinite cases at a modest constant-
// factor slowdown vs SimplicialLLT.
//
// Scope of this implementation:
//   - Construction: from an in-memory Eigen::SparseMatrix (used by rspec
//     mode where X is computed at runtime via sparse_axpby).
//   - Apply: Side::Left, dense B, ColMajor, Op::NoTrans on B.  Op::Trans on
//     A is supported via Eigen's solveTransposed pattern (uses A^{-T}).
//
// Not yet supported (add if a consumer needs it):
//   - File-based construction (mirror CholSolverLinOp's filename ctor)
//   - Side::Right, RowMajor, trans_B != NoTrans, sparse B, SkOp B.
//
// Compatible with RandLAPACK::linops::CompositeOperator and PowerOp.

#include "rl_util.hh"
#include "rl_linops.hh"

#include <RandBLAS.hh>
#include <cstdint>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>


namespace RandLAPACK_extras::linops {

template <typename T>
struct SparseLUSolverLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;

    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> lu_solver;
    Eigen::SparseMatrix<T> A_eigen;
    bool factorization_done;

    /// Construct from an in-memory Eigen sparse matrix.  Factorization is lazy
    /// (deferred to first apply or explicit factorize() call).
    SparseLUSolverLinOp(Eigen::SparseMatrix<T> A)
        : n_rows(A.rows()), n_cols(A.cols()),
          A_eigen(std::move(A)),
          factorization_done(false)
    {
        randblas_require(n_rows == n_cols);   // must be square to invert
    }

private:
    using Layout = blas::Layout;
    using Op     = blas::Op;
    using Side   = blas::Side;

public:

    /// Run the sparse LU factorization (analyze + numeric).  Idempotent.
    void factorize() {
        if (factorization_done) return;

        // Eigen recommends compress() before factoring (cheap if already compressed).
        A_eigen.makeCompressed();

        lu_solver.analyzePattern(A_eigen);
        lu_solver.factorize(A_eigen);

        if (lu_solver.info() != Eigen::Success) {
            // Factorization failed: caller's omega is likely too close to an eigenvalue
            // of the (K, M) pencil, making K - omega*M (near) singular.
            randblas_require(false && "SparseLU factorization failed; X may be (near) singular.");
        }

        factorization_done = true;
    }

    /// Concept-required 12-arg overload (no Side); delegates to Side::Left.
    void operator()(
        Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        (*this)(Side::Left, layout, trans_A, trans_B,
                m, n, k, alpha, B, ldb, beta, C, ldc);
    }

    /// Dense operator, Side::Left, ColMajor, Op::NoTrans on B.
    ///   C := alpha * op(A)^{-1} * B + beta * C
    /// where op(A) = A if trans_A == NoTrans, A^T if trans_A == Trans.
    void operator()(
        Side side, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        T alpha, const T* B, int64_t ldb,
        T beta, T* C, int64_t ldc)
    {
        randblas_require(side == Side::Left);
        randblas_require(layout == Layout::ColMajor);
        randblas_require(trans_B == Op::NoTrans);
        randblas_require(m == n_rows);
        randblas_require(k == n_rows);

        if (!factorization_done) factorize();

        // Map B (m x n, ColMajor, leading dimension ldb).  The third template
        // parameter (OuterStride<>) lets us pass a runtime column stride at
        // construction; without it, the Map type defaults to Stride<0,0> and
        // the runtime-stride overload doesn't match.
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>,
                   Eigen::Unaligned, Eigen::OuterStride<>>
            B_map(B, m, n, Eigen::OuterStride<>(ldb));

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> X(m, n);

        if (trans_A == Op::NoTrans) {
            X.noalias() = lu_solver.solve(B_map);
        } else {
            // A^{-T} * B
            X.noalias() = lu_solver.transpose().solve(B_map);
        }

        if (lu_solver.info() != Eigen::Success) {
            randblas_require(false && "SparseLU solve failed.");
        }

        // C := alpha * X + beta * C  (in-place column-by-column)
        for (int64_t j = 0; j < n; ++j) {
            T* C_col = C + (size_t)j * (size_t)ldc;
            const T* X_col = X.data() + (size_t)j * (size_t)m;
            for (int64_t i = 0; i < m; ++i) {
                C_col[i] = alpha * X_col[i] + beta * C_col[i];
            }
        }
    }
};

} // namespace RandLAPACK_extras::linops
