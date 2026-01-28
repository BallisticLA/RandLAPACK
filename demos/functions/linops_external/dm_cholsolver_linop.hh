#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "../misc/dm_util.hh"

#include <RandBLAS.hh>
#include <RandBLAS/sparse_data/trsm_dispatch.hh>
#include <string>
#include <fstream>
#include <stdexcept>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace RandLAPACK_demos {

/// Linear operator representing A^{-1} for a sparse symmetric positive definite matrix A
/// loaded from a Matrix Market file.
///
/// The inverse is applied via sparse Cholesky factorization (A = L * L^T):
///   A^{-1} * x = L^{-T} * L^{-1} * x  (forward then backward substitution).
///
/// Eigen computes the Cholesky factorization; the L factor is then extracted and all
/// subsequent triangular solves use RandBLAS sparse TRSM for bulk multi-column solves.
///
/// Side::Right uses a layout-flip trick: copying op(B) in one layout and calling
/// TRSM with the opposite layout implicitly transposes B without an explicit copy.
/// This works because a ColMajor m x k matrix is the same memory as a RowMajor k x m matrix.
///
/// Operator dispatch summary:
///
///   A^{-1} is ALWAYS applied via sparse L factor (RandBLAS sparse TRSM).
///   There is no dense L path -- A is loaded from a sparse Matrix Market file.
///
///   The B input determines which operator() overload is called:
///
///   1. Dense B (const T* B):
///      Core implementation. Copies op(B) into a work buffer, applies two sparse
///      TRSMs (forward and backward substitution with L), then accumulates into C.
///
///   2. Sparse B (SpMatB& B_sp, e.g., CSCMatrix or COOMatrix):
///      Densifies B first, then delegates to the dense B operator.
///      Rationale: A^{-1} * b is dense regardless of b's sparsity.
///
///   3. Sketching operator (SkOp& S, e.g., SparseSkOp or DenseSkOp):
///      Materializes S to a dense matrix, then delegates to the dense B operator.
///
/// CQRRT compatibility:
///   Natural ordering (no permutations) is required for CQRRT to work correctly.
///   Fill-reducing permutations (AMD, COLAMD) break CQRRT's numerical stability,
///   causing non-orthogonal Q factors.
///
/// Compatible with RandLAPACK::linops::CompositeOperator.
template <typename T>
struct CholSolverLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    std::string matrix_file;

    /// Eigen sparse Cholesky solver configured with natural ordering (no fill-reducing
    /// permutations), which is required for CQRRT compatibility.
    Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower, Eigen::NaturalOrdering<int>> chol_solver;

    /// Lower-triangular Cholesky factor L, extracted from Eigen after factorization.
    /// Stored as an Eigen compressed CSC sparse matrix. Its raw CSC arrays (valuePtr,
    /// innerIndexPtr, outerIndexPtr) are wrapped by make_L_csc() for use with RandBLAS.
    Eigen::SparseMatrix<T> L_sparse;

    bool factorization_done;

    /// Constructor: reads dimensions from the Matrix Market file header.
    /// The matrix data is NOT loaded until factorize() is called (lazy initialization).
    CholSolverLinOp(
        const std::string& filename
    ) : n_rows(read_matrix_dimension(filename, 0)),
        n_cols(read_matrix_dimension(filename, 1)),
        matrix_file(filename),
        factorization_done(false) {
        randblas_require(n_rows == n_cols); // Must be square for Cholesky
    }

private:

    /// Read a single dimension (rows or cols) from a Matrix Market file header.
    /// @param dim_index  0 for rows, 1 for cols.
    static int64_t read_matrix_dimension(const std::string& filename, int dim_index) {
        std::ifstream file(filename);
        randblas_require(file.is_open());

        // Skip comment lines (start with '%')
        std::string line;
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        // First non-comment line contains: rows cols nnz
        std::istringstream iss(line);
        int64_t rows, cols, nnz;
        iss >> rows >> cols >> nnz;
        file.close();

        return (dim_index == 0) ? rows : cols;
    }

    /// Create a RandBLAS CSCMatrix view that wraps L_sparse's raw CSC arrays.
    /// Eigen uses int for sparse indices, so we use CSCMatrix<T, int> (not int64_t).
    /// The expert constructor with own_memory=false avoids copying the data.
    RandBLAS::sparse_data::CSCMatrix<T, int> make_L_csc() {
        return RandBLAS::sparse_data::CSCMatrix<T, int>(
            L_sparse.rows(), L_sparse.cols(), L_sparse.nonZeros(),
            L_sparse.valuePtr(), L_sparse.innerIndexPtr(), L_sparse.outerIndexPtr(),
            RandBLAS::sparse_data::IndexBase::Zero
        );
    }

    /// Copy op(B) into a contiguous work buffer W.
    ///
    /// Both B (or its transpose) and W use the same layout. After the copy,
    /// W contains op(B) with dimensions rows_op x cols_op and leading dimension ldw.
    ///
    /// This is needed because RandBLAS sparse TRSM overwrites its B argument in-place,
    /// and the caller's B is const.
    ///
    /// For NoTrans, lapack::lacpy handles the copy in one call (supports both layouts
    /// and different leading dimensions). For Trans, we fall back to strided blas::copy
    /// loops since lacpy does not support transpose.
    ///
    /// @param layout   Storage layout for both B and W.
    /// @param trans_B  Whether to apply B or B^T.
    /// @param rows_op  Number of rows of op(B).
    /// @param cols_op  Number of columns of op(B).
    /// @param B        Source matrix (const, not modified).
    /// @param ldb      Leading dimension of B.
    /// @param W        Destination work buffer (must be pre-allocated).
    /// @param ldw      Leading dimension of W.
    void copy_op_B(Layout layout, Op trans_B, int64_t rows_op, int64_t cols_op,
                   const T* B, int64_t ldb, T* W, int64_t ldw) {
        if (trans_B == Op::NoTrans) {
            // No transpose: B and op(B) have the same shape (rows_op x cols_op).
            // lacpy copies an m x n submatrix from one leading dimension to another.
            // It is always ColMajor, so for RowMajor we swap dimensions: a RowMajor
            // rows_op x cols_op matrix is the same memory as ColMajor cols_op x rows_op.
            if (layout == Layout::ColMajor) {
                lapack::lacpy(lapack::MatrixType::General, rows_op, cols_op, B, ldb, W, ldw);
            } else {
                lapack::lacpy(lapack::MatrixType::General, cols_op, rows_op, B, ldb, W, ldw);
            }
        } else {
            // Transpose: B is cols_op x rows_op before transpose, op(B) = B^T is rows_op x cols_op.
            // lacpy cannot transpose, so we copy column-by-column (ColMajor) or row-by-row (RowMajor).
            // Each iteration reads a strided slice of B and writes a contiguous slice of W.
            if (layout == Layout::ColMajor) {
                // Column j of op(B) = row j of B.
                // Row j of B (ColMajor cols_op x rows_op): starts at B + j, stride ldb.
                // Column j of W: contiguous at W + j*ldw.
#if defined(_OPENMP)
                #pragma omp parallel for schedule(static)
#endif
                for (int64_t j = 0; j < cols_op; ++j)
                    blas::copy(rows_op, B + j, ldb, W + j * ldw, 1);
            } else {
                // Row i of op(B) = column i of B.
                // Column i of B (RowMajor cols_op x rows_op): starts at B + i, stride ldb.
                // Row i of W: contiguous at W + i*ldw.
#if defined(_OPENMP)
                #pragma omp parallel for schedule(static)
#endif
                for (int64_t i = 0; i < rows_op; ++i)
                    blas::copy(cols_op, B + i, ldb, W + i * ldw, 1);
            }
        }
    }

    /// Accumulate: C := beta * C + alpha * W.
    ///
    /// Both C and W are m x n in the given layout. Operates column-by-column (ColMajor)
    /// or row-by-row (RowMajor) to correctly handle non-contiguous C (when ldc > m or ldc > n).
    ///
    /// @param layout  Storage layout for both C and W.
    /// @param m       Number of rows.
    /// @param n       Number of columns.
    /// @param alpha   Scalar multiplier for W.
    /// @param W       Source matrix (m x n).
    /// @param ldw     Leading dimension of W.
    /// @param beta    Scalar multiplier for C (applied before adding alpha * W).
    /// @param C       Destination matrix (m x n), modified in-place.
    /// @param ldc     Leading dimension of C.
    void accumulate(Layout layout, int64_t m, int64_t n, T alpha,
                    const T* W, int64_t ldw, T beta, T* C, int64_t ldc) {
        if (layout == Layout::ColMajor) {
            // Each column is contiguous: scale then add.
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t j = 0; j < n; ++j) {
                blas::scal(m, beta, C + j * ldc, 1);
                blas::axpy(m, alpha, W + j * ldw, 1, C + j * ldc, 1);
            }
        } else {
            // Each row is contiguous: scale then add.
#if defined(_OPENMP)
            #pragma omp parallel for schedule(static)
#endif
            for (int64_t i = 0; i < m; ++i) {
                blas::scal(n, beta, C + i * ldc, 1);
                blas::axpy(n, alpha, W + i * ldw, 1, C + i * ldc, 1);
            }
        }
    }

public:

    /// Compute the sparse Cholesky factorization A = L * L^T.
    ///
    /// Reads the sparse matrix from the Matrix Market file, performs Cholesky
    /// factorization via Eigen, and extracts L as a compressed sparse matrix
    /// for subsequent use with RandBLAS sparse TRSM.
    ///
    /// This is called lazily on first use if not called explicitly.
    void factorize() {
        if (factorization_done) {
            return;
        }

        // Step 1: Read the sparse matrix from Matrix Market file into Eigen format.
        Eigen::SparseMatrix<T, Eigen::ColMajor> A_eigen;
        RandLAPACK_demos::eigen_sparse_from_matrix_market<T>(matrix_file, A_eigen);

        // Validate dimensions match what the constructor read from the header.
        randblas_require(A_eigen.rows() == n_rows);
        randblas_require(A_eigen.cols() == n_cols);

        // Step 2: Perform sparse Cholesky factorization (A = L * L^T).
        chol_solver.compute(A_eigen);

        if (chol_solver.info() != Eigen::Success) {
            randblas_require(false);
        }

        // Step 3: Extract L as a concrete sparse matrix so we can access its raw arrays.
        // chol_solver.matrixL() returns a view; assigning materializes it.
        L_sparse = chol_solver.matrixL();
        randblas_require(L_sparse.isCompressed());

        factorization_done = true;
    }

    /// Dense operator (non-Side version): C := alpha * A^{-1} * op(B) + beta * C.
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
    ///   Side::Left:  C := alpha * A^{-1} * op(B) + beta * C
    ///   Side::Right: C := alpha * op(B) * (first n cols of A^{-1}) + beta * C
    ///
    /// For SPD matrices, A^T = A, so trans_A is accepted but has no effect.
    ///
    /// A^{-1} is applied via two sparse triangular solves with Cholesky factor L:
    ///   1. Forward substitution:  solve L * y = x      (Op::NoTrans on lower-triangular L)
    ///   2. Backward substitution: solve L^T * z = y    (Op::Trans on lower-triangular L)
    ///
    /// @param side     Left or Right multiplication by A^{-1}.
    /// @param layout   ColMajor or RowMajor storage for B and C.
    /// @param trans_A  Transpose on A^{-1} (no effect since A is SPD).
    /// @param trans_B  Whether to use B or B^T.
    /// @param m        Number of rows of C.
    /// @param n        Number of columns of C.
    /// @param k        Shared/intermediate dimension (must equal n_rows for the relevant side).
    /// @param alpha    Scalar multiplier for the product.
    /// @param B        Dense input matrix (const, not modified).
    /// @param ldb      Leading dimension of B.
    /// @param beta     Scalar multiplier for C before accumulation.
    /// @param C        Output matrix, modified in-place.
    /// @param ldc      Leading dimension of C.
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

        // Lazy factorization: compute L on first use.
        if (!factorization_done) {
            factorize();
        }

        if (side == Side::Right) {
            // C := alpha * op(B) * (first n cols of A^{-1}) + beta * C
            //
            // RandBLAS sparse TRSM only supports left-side solves (A^{-1} * X).
            // To compute X * A^{-1} we use the layout-flip trick:
            //
            //   1. Copy op(B) (m x k) into W in the caller's layout.
            //   2. Call TRSM with the OPPOSITE layout. Since a ColMajor m x k matrix
            //      is the same memory as a RowMajor k x m matrix, TRSM sees the data
            //      as op(B)^T (k x m) and solves A^{-1} * op(B)^T in-place.
            //   3. After TRSM, interpreting W back in the caller's layout gives
            //      (A^{-1} * op(B)^T)^T = op(B) * A^{-T} = op(B) * A^{-1}  (A is SPD).
            //   4. Accumulate the first n columns of W (m x k) into C (m x n).

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

            copy_op_B(layout, trans_B, m, k, B, ldb, W, ldw);

            // Flip layout so TRSM interprets W as op(B)^T (k x m).
            auto flipped = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;
            auto L_csc = make_L_csc();
            // Forward substitution: W := L^{-1} * op(B)^T  (in flipped layout)
            RandBLAS::sparse_data::trsm(flipped, Op::NoTrans, (T)1.0, L_csc,
                                        Uplo::Lower, Diag::NonUnit, m, W, ldw);
            // Backward substitution: W := L^{-T} * W = A^{-1} * op(B)^T  (in flipped layout)
            RandBLAS::sparse_data::trsm(flipped, Op::Trans, (T)1.0, L_csc,
                                        Uplo::Lower, Diag::NonUnit, m, W, ldw);

            // In the caller's layout, W now holds op(B) * A^{-1} (m x k).
            // Accumulate the first n columns into C (m x n), since n <= k.
            accumulate(layout, m, n, alpha, W, ldw, beta, C, ldc);
            delete[] W;
            return;
        }

        // Side::Left: C := alpha * A^{-1} * op(B) + beta * C
        //
        // Straightforward: copy op(B) into W, apply two TRSMs in the caller's layout,
        // then accumulate W into C.

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

        // Allocate work buffer for op(B) (k x n) in the caller's layout.
        int64_t ldw = (layout == Layout::ColMajor) ? k : n;
        T* W = new T[k * n]();

        copy_op_B(layout, trans_B, k, n, B, ldb, W, ldw);

        // Forward substitution: W := L^{-1} * op(B)
        auto L_csc = make_L_csc();
        RandBLAS::sparse_data::trsm(layout, Op::NoTrans, (T)1.0, L_csc,
                                    Uplo::Lower, Diag::NonUnit, n, W, ldw);

        // Backward substitution: W := L^{-T} * W = A^{-1} * op(B)
        RandBLAS::sparse_data::trsm(layout, Op::Trans, (T)1.0, L_csc,
                                    Uplo::Lower, Diag::NonUnit, n, W, ldw);

        // C := beta * C + alpha * W
        accumulate(layout, m, n, alpha, W, ldw, beta, C, ldc);
        delete[] W;
    }

    /// Sparse B operator (non-Side version): C := alpha * A^{-1} * op(B_sp) + beta * C.
    /// Convenience overload that delegates to Side::Left.
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
    ///
    /// Densifies B_sp into a temporary dense matrix, then delegates to the dense
    /// operator above. This is correct because A^{-1} * B is generally dense
    /// regardless of B's sparsity pattern.
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
        // Determine the dimensions of B before any transpose.
        // Side::Left:  op(B) is k x n, so B is either k x n or n x k.
        // Side::Right: op(B) is m x k, so B is either m x k or k x m.
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

        // Densify: convert sparse B to a dense matrix.
        T* B_dense = new T[rows_B * cols_B]();
        RandLAPACK::util::sparse_to_dense_summing_duplicates(B_sp, layout, B_dense);

        // Delegate to the dense operator.
        (*this)(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);

        delete[] B_dense;
    }

    /// Sketching operator (Side version).
    ///
    /// Materializes the sketching operator S into a dense matrix, then delegates
    /// to the dense operator. Supports both SparseSkOp (COO materialization in
    /// O(nnz) time) and DenseSkOp (block-wise materialization via sketch_general).
    ///
    /// @tparam SkOp  SparseSkOp or DenseSkOp type (detected via constexpr if).
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
        // Allocate dense matrix to hold the materialized sketch.
        T* S_dense = new T[S.n_rows * S.n_cols]();
        int64_t lds = (layout == Layout::ColMajor) ? S.n_rows : S.n_cols;

        // Materialize the sketch. Method depends on sketch type (detected at compile time).
        if constexpr (requires { S.rows; S.cols; S.vals; S.nnz; }) {
            // SparseSkOp: stored as COO (rows[], cols[], vals[]).
            // Scatter each nonzero entry (scaled by isometry_scale) into S_dense.
            // Uses += to handle potential duplicate COO entries at the same position.
            // Note: this loop is NOT parallelized because duplicates can cause races on S_dense[idx].
            T scale = S.dist.isometry_scale;
            for (int64_t i = 0; i < S.nnz; ++i) {
                int64_t r = S.rows[i];
                int64_t c = S.cols[i];
                int64_t idx = (layout == Layout::ColMajor) ? (r + c * lds) : (r * lds + c);
                S_dense[idx] += scale * S.vals[i];
            }
        } else {
            // DenseSkOp: materialized block-by-block using sketch_general(S * I_block).
            // Each block materializes up to block_size columns of S at a time.
            constexpr int64_t block_size = 256;
            int64_t num_blocks = (S.n_cols + block_size - 1) / block_size;

            for (int64_t b = 0; b < num_blocks; ++b) {
                int64_t col_start = b * block_size;
                int64_t col_end = std::min(col_start + block_size, S.n_cols);
                int64_t block_cols = col_end - col_start;

                // Create a block_cols x block_cols identity matrix.
                T* I_block = new T[block_cols * block_cols]();
                for (int64_t i = 0; i < block_cols; ++i) {
                    I_block[i * block_cols + i] = (T)1.0;
                }

                // S_block points to the region of S_dense where this block's columns go.
                T* S_block = S_dense + (layout == Layout::ColMajor ? col_start * lds : col_start);

                // Materialize: S_block = S[:, col_start:col_end] via S * I_block.
                RandBLAS::sketch_general(
                    Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    S.n_rows, block_cols, block_cols,
                    (T)1.0, S, col_start, 0, I_block, block_cols,
                    (T)0.0, S_block, lds
                );

                delete[] I_block;
            }
        }

        // Delegate to the dense operator.
        (*this)(side, layout, trans_A, trans_S, m, n, k, alpha, S_dense, lds, beta, C, ldc);

        delete[] S_dense;
    }
};

} // namespace RandLAPACK_demos
