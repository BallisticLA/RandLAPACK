#pragma once

#include "ext_solver_linop_util.hh"
#include "rl_util.hh"
#include "rl_linops.hh"
#include "../misc/ext_util.hh"

#include <RandBLAS.hh>
#include <RandBLAS/sparse_data/trsm_dispatch.hh>
#include <string>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

namespace RandLAPACK_extras::linops {

/// Linear operator representing A^{-1} (or L^{-1} P_r in half-solve mode) for a sparse
/// invertible matrix A loaded from a Matrix Market file.
///
/// Full-solve mode (default):
///   NoTrans: A^{-1} * x  (full forward + backward substitution with LU factors).
///   Trans:   A^{-T} * x  (transpose solve, different from NoTrans since A is non-symmetric).
///
/// Half-solve mode (half_solve=true):
///   NoTrans: L^{-1} P_r * x  (forward substitution with row permutation only).
///   Trans:   P_r^T L^{-T} * x.
///
/// Eigen computes the sparse LU factorization with COLAMD fill-reducing ordering:
///   P_r * A * P_c^T = L * U
///
/// The L and U factors are extracted and all subsequent triangular solves use
/// RandBLAS sparse TRSM for bulk multi-column solves.
///
/// Operator dispatch summary:
///
///   The inverse is ALWAYS applied via sparse L and U factors (RandBLAS sparse TRSM).
///   There is no dense factor path -- A is loaded from a sparse Matrix Market file.
///
///   Dense B (const T* B): core implementation with sparse TRSMs.
///   Sparse B (SpMatB& B_sp): densifies B first, then delegates to dense operator.
///   Sketching operator (SkOp& S): materializes S, then delegates to dense operator.
///
/// Fill-reducing ordering:
///   Uses COLAMD ordering to reduce fill-in in L and U.
///   All solves apply the appropriate row/column permutations around the triangular solves.
///
///   Full-solve NoTrans: A^{-1} x = P_c^T U^{-1} L^{-1} P_r x
///   Full-solve Trans:   A^{-T} x = P_r^T L^{-T} U^{-T} P_c x
///   Half-solve NoTrans: M x     = L^{-1} P_r x
///   Half-solve Trans:   M^T x   = P_r^T L^{-T} x
///
/// Compatible with RandLAPACK::linops::CompositeOperator.
template <typename T>
struct LUSolverLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    std::string matrix_file;

    /// When true, applies only L^{-1} P_r (forward substitution with row permutation)
    /// instead of A^{-1} = P_c^T U^{-1} L^{-1} P_r.
    bool half_solve;

    /// Eigen sparse LU solver with COLAMD fill-reducing ordering.
    /// Factorizes P_r * A * P_c^T = L * U.
    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> lu_solver;

    /// Lower-triangular factor L with unit diagonal, extracted from Eigen after factorization.
    Eigen::SparseMatrix<T> L_sparse;

    /// Upper-triangular factor U, extracted from Eigen after factorization.
    /// Converted to ColMajor (Eigen's matrixU() returns RowMajor).
    Eigen::SparseMatrix<T> U_sparse;

    /// Row permutation P_r: perm_r_fwd[i] = j means (P_r * x)[i] = x[j].
    std::vector<int> perm_r_fwd;
    /// Inverse row permutation P_r^T: perm_r_inv[j] = i means (P_r^T * x)[j] = x[i].
    std::vector<int> perm_r_inv;
    /// Column permutation P_c: perm_c_fwd[i] = j means (P_c * x)[i] = x[j].
    std::vector<int> perm_c_fwd;
    /// Inverse column permutation P_c^T: perm_c_inv[j] = i means (P_c^T * x)[j] = x[i].
    std::vector<int> perm_c_inv;

    bool factorization_done;

    /// Constructor: reads dimensions from the Matrix Market file header.
    /// The matrix data is NOT loaded until factorize() is called (lazy initialization).
    /// @param half_solve  If true, applies L^{-1} P_r only (forward substitution).
    ///                    If false (default), applies A^{-1}.
    LUSolverLinOp(
        const std::string& filename,
        bool half_solve = false
    ) : n_rows(solver_util::read_matrix_dimension(filename, 0)),
        n_cols(solver_util::read_matrix_dimension(filename, 1)),
        matrix_file(filename),
        half_solve(half_solve),
        factorization_done(false) {
        randblas_require(n_rows == n_cols); // Must be square for LU solve
    }

private:

    using Layout = blas::Layout;
    using Op     = blas::Op;
    using Side   = blas::Side;
    using Uplo   = blas::Uplo;
    using Diag   = blas::Diag;

    /// Create a RandBLAS CSCMatrix view wrapping L_sparse's raw CSC arrays.
    RandBLAS::sparse_data::CSCMatrix<T, int> make_L_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, int>(
            L_sparse.rows(), L_sparse.cols(), L_sparse.nonZeros(),
            L_sparse.valuePtr(), L_sparse.innerIndexPtr(), L_sparse.outerIndexPtr(),
            RandBLAS::sparse_data::IndexBase::Zero
        );
    }

    /// Create a RandBLAS CSCMatrix view wrapping U_sparse's raw CSC arrays.
    RandBLAS::sparse_data::CSCMatrix<T, int> make_U_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, int>(
            U_sparse.rows(), U_sparse.cols(), U_sparse.nonZeros(),
            U_sparse.valuePtr(), U_sparse.innerIndexPtr(), U_sparse.outerIndexPtr(),
            RandBLAS::sparse_data::IndexBase::Zero
        );
    }

public:

    /// Compute the sparse LU factorization P_r * A * P_c^T = L * U.
    ///
    /// Reads the sparse matrix from the Matrix Market file, performs LU
    /// factorization via Eigen with COLAMD ordering, and extracts L and U
    /// as compressed sparse matrices for subsequent use with RandBLAS sparse TRSM.
    ///
    /// This is called lazily on first use if not called explicitly.
    void factorize() {
        if (factorization_done) {
            return;
        }

        // Step 1: Read the sparse matrix from Matrix Market file into Eigen format.
        Eigen::SparseMatrix<T, Eigen::ColMajor> A_eigen;
        RandLAPACK_extras::eigen_sparse_from_matrix_market<T>(matrix_file, A_eigen);

        // Validate dimensions match what the constructor read from the header.
        randblas_require(A_eigen.rows() == n_rows);
        randblas_require(A_eigen.cols() == n_cols);

        // Step 2: Perform sparse LU factorization (P_r * A * P_c^T = L * U).
        lu_solver.compute(A_eigen);

        if (lu_solver.info() != Eigen::Success) {
            randblas_require(false);
        }

        // Step 3: Extract L and U as concrete sparse matrices.
        // L: lower triangular with unit diagonal, already ColMajor.
        L_sparse = lu_solver.matrixL().toSparse();
        randblas_require(L_sparse.isCompressed());

        // U: upper triangular. Eigen's matrixU() may return RowMajor internally,
        // so we explicitly convert to ColMajor via the Eigen::SparseMatrix<T> constructor.
        U_sparse = Eigen::SparseMatrix<T>(lu_solver.matrixU().toSparse());
        randblas_require(U_sparse.isCompressed());

        // Step 4: Extract permutation vectors.
        // Eigen's rowsPermutation() returns P_r such that P_r * A * P_c^T = L * U.
        // rowsPermutation().indices()[i] = j means P_r maps row i to position j.
        int64_t nn = n_rows;

        // Eigen permutation convention: P_sigma(e_i) = e_{sigma(i)}, so
        // (P * v)[j] = v[sigma^{-1}(j)].  Our apply_row_perm does a gather:
        // result[i] = source[perm[i]], which corresponds to P^{-1} when perm = sigma.
        //
        // We define perm_fwd as the array such that apply_row_perm(perm_fwd) = P.
        // Since apply_row_perm(perm) = "gather with perm" = P^{-1} when perm = sigma,
        // we need perm_fwd = sigma^{-1} (so gather with sigma^{-1} gives P).
        {
            const auto& pr_indices = lu_solver.rowsPermutation().indices();
            perm_r_fwd.resize(nn);
            perm_r_inv.resize(nn);
            for (int64_t i = 0; i < nn; ++i) {
                perm_r_fwd[pr_indices[i]] = i;  // sigma^{-1}: gather gives P_r
                perm_r_inv[i] = pr_indices[i];   // sigma:      gather gives P_r^{-1} = P_r^T
            }
        }

        {
            const auto& pc_indices = lu_solver.colsPermutation().indices();
            perm_c_fwd.resize(nn);
            perm_c_inv.resize(nn);
            for (int64_t i = 0; i < nn; ++i) {
                perm_c_fwd[pc_indices[i]] = i;  // sigma^{-1}: gather gives Q
                perm_c_inv[i] = pc_indices[i];   // sigma:      gather gives Q^{-1} = Q^T
            }
        }

        factorization_done = true;
    }

    /// Dense operator (non-Side version): C := alpha * op_A(A^{-1}) * op(B) + beta * C.
    /// Convenience overload that delegates to Side::Left.
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

    /// Dense operator (Side version):
    ///   Full-solve (half_solve=false):
    ///     Side::Left, NoTrans:   C := alpha * A^{-1} * op(B) + beta * C
    ///     Side::Left, Trans:     C := alpha * A^{-T} * op(B) + beta * C
    ///     Side::Right, NoTrans:  C := alpha * op(B) * A^{-1} + beta * C
    ///     Side::Right, Trans:    C := alpha * op(B) * A^{-T} + beta * C
    ///   Half-solve (half_solve=true), with M = L^{-1} P_r:
    ///     Side::Left, NoTrans:   C := alpha * M * op(B) + beta * C
    ///     Side::Left, Trans:     C := alpha * M^T * op(B) + beta * C
    ///     Side::Right, NoTrans:  C := alpha * op(B) * M + beta * C
    ///     Side::Right, Trans:    C := alpha * op(B) * M^T + beta * C
    ///
    /// Unlike CholSolverLinOp, trans_A matters for full-solve since A is non-symmetric.
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
        randblas_require(trans_A == Op::NoTrans || trans_A == Op::Trans);
        randblas_require(trans_B == Op::NoTrans || trans_B == Op::Trans);

        // Lazy factorization: compute L, U on first use.
        if (!factorization_done) {
            factorize();
        }

        if (side == Side::Right) {
            // Side::Right: use flipped-layout trick.
            // op(B) * op_A(A^{-1}) in caller's layout = op_A(A^{-1})^T * op(B)^T in flipped layout.
            //
            // Since flipping transposes both sides:
            //   NoTrans on A becomes Trans on the flipped left-side (A^{-T} in flipped)
            //   Trans on A becomes NoTrans on the flipped left-side (A^{-1} in flipped)

            randblas_require(k == n_rows);

            auto [rows_B, cols_B] = RandBLAS::dims_before_op(m, k, trans_B);
            if (layout == Layout::ColMajor) {
                randblas_require(ldb >= rows_B);
                randblas_require(ldc >= m);
            } else {
                randblas_require(ldb >= cols_B);
                randblas_require(ldc >= n);
            }

            // Allocate work buffer for op(B) (m x k) in the caller's layout.
            int64_t ldw = (layout == Layout::ColMajor) ? m : k;
            T* W = new T[m * k]();

            solver_util::copy_op_B(layout, trans_B, m, k, B, ldb, W, ldw);

            // Flip layout so TRSM interprets W as op(B)^T (k x m).
            auto flipped = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
            auto L_csc = make_L_csc();
            auto U_csc = make_U_csc();

            // In flipped layout, W is a k x m matrix. Permutations act on k rows.
            //
            // Full-solve NoTrans: want op(B) * A^{-1} = op(B) * P_c^T U^{-1} L^{-1} P_r
            //   In flipped: P_r^T L^{-T} U^{-T} P_c op(B)^T  (= A^{-T} applied left in flipped)
            //   → P_c on rows, TRSM(U, Trans), TRSM(L, Trans, Unit), P_r^T on rows.
            //
            // Full-solve Trans: want op(B) * A^{-T} = op(B) * P_r^T L^{-T} U^{-T} P_c
            //   In flipped: P_c^T U^{-1} L^{-1} P_r op(B)^T  (= A^{-1} applied left in flipped)
            //   → P_r on rows, TRSM(L, NoTrans, Unit), TRSM(U, NoTrans), P_c^T on rows.
            //
            // Half NoTrans: want op(B) * M = op(B) * L^{-1} P_r
            //   In flipped: P_r^T L^{-T} op(B)^T  (= M^T applied left in flipped)
            //   → TRSM(L, Trans, Unit), P_r^T on rows.
            //
            // Half Trans: want op(B) * M^T = op(B) * P_r^T L^{-T}
            //   In flipped: L^{-1} P_r op(B)^T  (= M applied left in flipped)
            //   → P_r on rows, TRSM(L, NoTrans, Unit).

            if (half_solve) {
                if (trans_A == Op::NoTrans) {
                    // op(B) * L^{-1} P_r: TRSM first, then permute.
                    RandBLAS::sparse_data::trsm(flipped, Op::Trans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, m, W, ldw);
                    solver_util::apply_row_perm(flipped, k, m, perm_r_inv.data(), W, ldw);
                } else {
                    // op(B) * P_r^T L^{-T}: permute first, then TRSM.
                    solver_util::apply_row_perm(flipped, k, m, perm_r_fwd.data(), W, ldw);
                    RandBLAS::sparse_data::trsm(flipped, Op::NoTrans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, m, W, ldw);
                }
            } else {
                if (trans_A == Op::NoTrans) {
                    // Full-solve NoTrans (flipped = A^{-T} left):
                    solver_util::apply_row_perm(flipped, k, m, perm_c_fwd.data(), W, ldw);
                    RandBLAS::sparse_data::trsm(flipped, Op::Trans, (T)1.0, U_csc,
                                                Uplo::Upper, Diag::NonUnit, m, W, ldw);
                    RandBLAS::sparse_data::trsm(flipped, Op::Trans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, m, W, ldw);
                    solver_util::apply_row_perm(flipped, k, m, perm_r_inv.data(), W, ldw);
                } else {
                    // Full-solve Trans (flipped = A^{-1} left):
                    solver_util::apply_row_perm(flipped, k, m, perm_r_fwd.data(), W, ldw);
                    RandBLAS::sparse_data::trsm(flipped, Op::NoTrans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, m, W, ldw);
                    RandBLAS::sparse_data::trsm(flipped, Op::NoTrans, (T)1.0, U_csc,
                                                Uplo::Upper, Diag::NonUnit, m, W, ldw);
                    solver_util::apply_row_perm(flipped, k, m, perm_c_inv.data(), W, ldw);
                }
            }

            // In the caller's layout, W now holds the result. Accumulate into C.
            solver_util::accumulate(layout, m, n, alpha, W, ldw, beta, C, ldc);
            delete[] W;
            return;
        }

        // Side::Left

        randblas_require(m == n_rows);
        randblas_require(k == n_cols);

        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        if (layout == Layout::ColMajor) {
            randblas_require(ldb >= rows_B);
            randblas_require(ldc >= m);
        } else {
            randblas_require(ldb >= cols_B);
            randblas_require(ldc >= n);
        }

        auto L_csc = make_L_csc();
        auto U_csc = make_U_csc();

        // Permutation math for Side::Left (P_r * A * P_c^T = L * U):
        //
        // Full NoTrans (A^{-1} = P_c^T U^{-1} L^{-1} P_r):
        //   apply P_r, L^{-1} TRSM (Unit), U^{-1} TRSM, apply P_c^T.
        //
        // Full Trans (A^{-T} = P_r^T L^{-T} U^{-T} P_c):
        //   apply P_c, U^{-T} TRSM, L^{-T} TRSM (Unit), apply P_r^T.
        //
        // Half NoTrans (M = L^{-1} P_r):
        //   apply P_r, L^{-1} TRSM (Unit).
        //
        // Half Trans (M^T = P_r^T L^{-T}):
        //   L^{-T} TRSM (Unit), apply P_r^T.

        if (beta == (T)0.0) {
            // Fast path: copy op(B) directly into C, TRSM in-place.
            solver_util::copy_op_B(layout, trans_B, k, n, B, ldb, C, ldc);

            if (half_solve) {
                if (trans_A == Op::NoTrans) {
                    // M = L^{-1} P_r: permute rows by P_r, then forward solve.
                    solver_util::apply_row_perm(layout, k, n, perm_r_fwd.data(), C, ldc);
                    RandBLAS::sparse_data::trsm(layout, Op::NoTrans, alpha, L_csc,
                                                Uplo::Lower, Diag::Unit, n, C, ldc);
                } else {
                    // M^T = P_r^T L^{-T}: backward solve, then permute rows by P_r^T.
                    RandBLAS::sparse_data::trsm(layout, Op::Trans, alpha, L_csc,
                                                Uplo::Lower, Diag::Unit, n, C, ldc);
                    solver_util::apply_row_perm(layout, k, n, perm_r_inv.data(), C, ldc);
                }
            } else {
                if (trans_A == Op::NoTrans) {
                    // A^{-1} = P_c^T U^{-1} L^{-1} P_r
                    solver_util::apply_row_perm(layout, k, n, perm_r_fwd.data(), C, ldc);
                    RandBLAS::sparse_data::trsm(layout, Op::NoTrans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, n, C, ldc);
                    RandBLAS::sparse_data::trsm(layout, Op::NoTrans, alpha, U_csc,
                                                Uplo::Upper, Diag::NonUnit, n, C, ldc);
                    solver_util::apply_row_perm(layout, k, n, perm_c_inv.data(), C, ldc);
                } else {
                    // A^{-T} = P_r^T L^{-T} U^{-T} P_c
                    solver_util::apply_row_perm(layout, k, n, perm_c_fwd.data(), C, ldc);
                    RandBLAS::sparse_data::trsm(layout, Op::Trans, (T)1.0, U_csc,
                                                Uplo::Upper, Diag::NonUnit, n, C, ldc);
                    RandBLAS::sparse_data::trsm(layout, Op::Trans, alpha, L_csc,
                                                Uplo::Lower, Diag::Unit, n, C, ldc);
                    solver_util::apply_row_perm(layout, k, n, perm_r_inv.data(), C, ldc);
                }
            }
        } else {
            // General path: allocate W, TRSM on W, accumulate into C.
            int64_t ldw = (layout == Layout::ColMajor) ? k : n;
            T* W = new T[k * n]();

            solver_util::copy_op_B(layout, trans_B, k, n, B, ldb, W, ldw);

            if (half_solve) {
                if (trans_A == Op::NoTrans) {
                    solver_util::apply_row_perm(layout, k, n, perm_r_fwd.data(), W, ldw);
                    RandBLAS::sparse_data::trsm(layout, Op::NoTrans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, n, W, ldw);
                } else {
                    RandBLAS::sparse_data::trsm(layout, Op::Trans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, n, W, ldw);
                    solver_util::apply_row_perm(layout, k, n, perm_r_inv.data(), W, ldw);
                }
            } else {
                if (trans_A == Op::NoTrans) {
                    solver_util::apply_row_perm(layout, k, n, perm_r_fwd.data(), W, ldw);
                    RandBLAS::sparse_data::trsm(layout, Op::NoTrans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, n, W, ldw);
                    RandBLAS::sparse_data::trsm(layout, Op::NoTrans, (T)1.0, U_csc,
                                                Uplo::Upper, Diag::NonUnit, n, W, ldw);
                    solver_util::apply_row_perm(layout, k, n, perm_c_inv.data(), W, ldw);
                } else {
                    solver_util::apply_row_perm(layout, k, n, perm_c_fwd.data(), W, ldw);
                    RandBLAS::sparse_data::trsm(layout, Op::Trans, (T)1.0, U_csc,
                                                Uplo::Upper, Diag::NonUnit, n, W, ldw);
                    RandBLAS::sparse_data::trsm(layout, Op::Trans, (T)1.0, L_csc,
                                                Uplo::Lower, Diag::Unit, n, W, ldw);
                    solver_util::apply_row_perm(layout, k, n, perm_r_inv.data(), W, ldw);
                }
            }

            // C := beta * C + alpha * W
            solver_util::accumulate(layout, m, n, alpha, W, ldw, beta, C, ldc);
            delete[] W;
        }
    }

    /// Sparse B operator (non-Side version): delegates to Side::Left.
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

    /// Sparse B operator (Side version).
    /// Densifies B_sp, then delegates to the dense operator.
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
        int64_t rows_B, cols_B;
        if (side == Side::Left) {
            auto [rb, cb] = RandBLAS::dims_before_op(k, n, trans_B);
            rows_B = rb;
            cols_B = cb;
        } else {
            auto [rb, cb] = RandBLAS::dims_before_op(m, k, trans_B);
            rows_B = rb;
            cols_B = cb;
        }

        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

        T* B_dense = new T[rows_B * cols_B]();
        RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);

        (*this)(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);

        delete[] B_dense;
    }

    /// Sketching operator (Side version).
    /// Materializes S to dense, then delegates to the dense operator.
    template <typename SkOp>
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
        T* S_dense = new T[S.n_rows * S.n_cols]();
        int64_t lds = (layout == Layout::ColMajor) ? S.n_rows : S.n_cols;

        if constexpr (requires { S.rows; S.cols; S.vals; S.nnz; }) {
            // SparseSkOp: scatter COO entries into dense matrix.
            T scale = S.dist.isometry_scale;
            for (int64_t i = 0; i < S.nnz; ++i) {
                int64_t r = S.rows[i];
                int64_t c = S.cols[i];
                int64_t idx = (layout == Layout::ColMajor) ? (r + c * lds) : (r * lds + c);
                S_dense[idx] += scale * S.vals[i];
            }
        } else {
            // DenseSkOp: materialize block-by-block.
            constexpr int64_t block_size = 256;
            int64_t num_blocks = (S.n_cols + block_size - 1) / block_size;

            for (int64_t b = 0; b < num_blocks; ++b) {
                int64_t col_start = b * block_size;
                int64_t col_end = std::min(col_start + block_size, S.n_cols);
                int64_t block_cols = col_end - col_start;

                T* I_block = new T[block_cols * block_cols]();
                for (int64_t i = 0; i < block_cols; ++i) {
                    I_block[i * block_cols + i] = (T)1.0;
                }

                T* S_block = S_dense + (layout == Layout::ColMajor ? col_start * lds : col_start);

                RandBLAS::sketch_general(
                    Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    S.n_rows, block_cols, block_cols,
                    (T)1.0, S, 0, col_start, I_block, block_cols,
                    (T)0.0, S_block, lds
                );

                delete[] I_block;
            }
        }

        (*this)(side, layout, trans_A, trans_S, m, n, k, alpha, S_dense, lds, beta, C, ldc);

        delete[] S_dense;
    }

    // =====================================================================
    // Block views — intentionally unsupported
    // =====================================================================

    template <typename Dummy = void>
    auto row_block(int64_t, int64_t) const {
        static_assert(!std::is_same_v<Dummy, void>,
            "LUSolverLinOp represents an implicit inverse (A^{-1}) and cannot be "
            "partitioned into blocks.");
    }

    template <typename Dummy = void>
    auto col_block(int64_t, int64_t) const {
        static_assert(!std::is_same_v<Dummy, void>,
            "LUSolverLinOp represents an implicit inverse (A^{-1}) and cannot be "
            "partitioned into blocks.");
    }

    template <typename Dummy = void>
    auto submatrix(int64_t, int64_t, int64_t, int64_t) const {
        static_assert(!std::is_same_v<Dummy, void>,
            "LUSolverLinOp represents an implicit inverse (A^{-1}) and cannot be "
            "partitioned into blocks.");
    }
};

} // namespace RandLAPACK_extras::linops
