#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "../misc/dm_util.hh"

#include <RandBLAS.hh>
#include <string>
#include <fstream>
#include <stdexcept>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

namespace RandLAPACK_demos {

/// Linear operator that represents the inverse of a sparse symmetric positive definite matrix
/// loaded from a Matrix Market file. The inverse is applied via sparse Cholesky factorization:
/// A = L * L^T, then A^{-1} * x is computed as L^{-T} * L^{-1} * x.
///
/// Sparsity considerations:
/// - The Cholesky factor L is stored in sparse format (using Eigen::SimplicialLLT)
/// - Applying A^{-1} involves sparse triangular solves with L and L^T
/// - However, A^{-1} itself is typically dense, so A^{-1} * b produces a dense result
///   even when b is sparse
///
/// Implementation rationale for sparse B:
/// The operator processes input matrix B column-by-column, solving A * x = b for each column.
/// When B is sparse, we densify it before processing because:
/// 1. Sparse triangular solves effectively treat input vectors as dense (must touch all entries)
/// 2. The result A^{-1} * b_sparse is dense regardless of input sparsity
/// 3. Full densification is simpler and avoids repeated column extraction overhead
///
/// CQRRT compatibility:
/// Natural ordering (no permutations) is required for CQRRT to work correctly with sparse SPD.
/// Fill-reducing permutations (AMD, COLAMD) break CQRRT's numerical stability, causing non-orthogonal Q.
/// Trade-off: Natural ordering may increase fill-in but ensures CQRRT produces machine-precision results.
///
/// This operator is compatible with RandLAPACK::linops::CompositeOperator.
template <typename T>
struct CholSolverLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    std::string matrix_file;

    // Eigen sparse Cholesky solver with natural ordering (no permutations)
    // Using NaturalOrdering to avoid fill-reducing permutations that break CQRRT
    Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower, Eigen::NaturalOrdering<int>> chol_solver;

    bool factorization_done;

    // Constructor - checks the header of the sparse matrix; does not load data until it is needed.
    CholSolverLinOp(
        const std::string& filename
    ) : n_rows(read_matrix_dimension(filename, 0)),
        n_cols(read_matrix_dimension(filename, 1)),
        matrix_file(filename),
        factorization_done(false) {
        randblas_require(n_rows == n_cols); // Must be square for Cholesky
    }

private:
    // Helper function to read matrix dimensions from Matrix Market file
    static int64_t read_matrix_dimension(const std::string& filename, int dim_index) {
        std::ifstream file(filename);
        randblas_require(file.is_open());

        // Read Matrix Market header
        std::string line;
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        // Parse dimensions
        std::istringstream iss(line);
        int64_t rows, cols, nnz;
        iss >> rows >> cols >> nnz;
        file.close();

        return (dim_index == 0) ? rows : cols;
    }

    // Solve with optional transpose: op(A) * x = b
    Eigen::Matrix<T, Eigen::Dynamic, 1> solve_with_transpose(
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& b, Op trans_A
    ) {
        if (trans_A == Op::NoTrans) {
            // Solve A * x = b via L * L^T * x = b
            // CRITICAL: Force evaluation to concrete vector
            Eigen::Matrix<T, Eigen::Dynamic, 1> x = chol_solver.solve(b);
            return x;
        } else {
            // Solve A^T * x = b, where A = L * L^T, so A^T = L^T * L
            // Step 1: L * y = b (solve for y)
            // Step 2: L^T * x = y (solve for x)
            auto L = chol_solver.matrixL();
            auto LT = chol_solver.matrixU();
            // CRITICAL: Force evaluation to concrete vector
            Eigen::Matrix<T, Eigen::Dynamic, 1> y = L.solve(b);
            Eigen::Matrix<T, Eigen::Dynamic, 1> x = LT.solve(y);
            return x;
        }
    }

    // Get pointer to column j of C and its increment
    std::pair<T*, int64_t> get_c_column(int64_t j, Layout layout, T* C, int64_t ldc) {
        if (layout == Layout::ColMajor) {
            return {C + j * ldc, 1};  // {pointer, increment}
        } else {  // RowMajor
            return {C + j, ldc};
        }
    }

    // Extract column j of op(B) into b_col
    void extract_column_from_op_B(
        int64_t j, int64_t k, Layout layout, Op trans_B,
        const T* B, int64_t ldb, Eigen::Matrix<T, Eigen::Dynamic, 1>& b_col
    ) {
        if (layout == Layout::ColMajor) {
            if (trans_B == Op::NoTrans) {
                // B is k × n, column j is contiguous
                b_col = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(B + j * ldb, k);
            } else {
                // B is n × k, need row j (not contiguous)
                for (int64_t i = 0; i < k; ++i) {
                    b_col(i) = B[j + i * ldb];
                }
            }
        } else {  // RowMajor
            if (trans_B == Op::NoTrans) {
                // B is k × n, column j has stride ldb
                b_col = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, 0, Eigen::InnerStride<>>(
                    B + j, k, Eigen::InnerStride<>(ldb));
            } else {
                // B is n × k, row j is contiguous
                b_col = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(B + j * ldb, k);
            }
        }
    }

public:

    /// Compute the sparse Cholesky factorization A = L * L^T.
    // The factorization is stored in chol_solver; A-eigen gets freed automatically.
    void factorize() {
        if (factorization_done) {
            return;
        }

        // Step 1: Read the sparse matrix from Matrix Market file directly to Eigen format
        Eigen::SparseMatrix<T, Eigen::ColMajor> A_eigen;
        RandLAPACK_demos::eigen_sparse_from_matrix_market<T>(matrix_file, A_eigen);

        // Validate dimensions match what constructor read from header
        randblas_require(A_eigen.rows() == n_rows);
        randblas_require(A_eigen.cols() == n_cols);

        // Step 2: Perform sparse Cholesky factorization
        chol_solver.compute(A_eigen);

        if (chol_solver.info() != Eigen::Success) {
            randblas_require(false);
        }

        factorization_done = true;
    }

    /// Dense matrix multiplication operator (non-Side version): C := alpha * A^{-1} * op(B) + beta * C
    /// Delegates to Side::Left version
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

    /// Dense matrix multiplication operator (Side version): C := alpha * op(A^{-1}) * op(B) + beta * C
    /// where A^{-1} is computed via sparse Cholesky solve
    ///
    /// Supported operations:
    ///   trans_A = NoTrans: A^{-1} * op(B)     (solve A*x = b via L*L^T*x = b)
    ///   trans_A = Trans:   A^{-T} * op(B)    (solve A^T*x = b via L^T*L*x = b)
    ///
    /// Side parameter:
    ///   Side::Left  - C := alpha * op(A^{-1}) * op(B) + beta * C (supported)
    ///   Side::Right - Uses transpose trick: (B * A^{-1})^T = A^{-T} * B^T
    ///                 Requires trans_A support, which is now implemented
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
        // Handle Side::Right using block-solve approach
        if (side == Side::Right) {
            // Want: C := alpha * op(B) * op(A^{-1}) + beta * C
            // Strategy: Solve op(A) * X = I_n to get the first n columns of op(A^{-1}),
            // then compute each column of C using gemv.
            //
            // We solve for ALL columns of A^{-1} at once (block solve) for better numerical
            // stability with sparse Cholesky, then apply them column-by-column using gemv.

            // Ensure factorization is computed
            if (!factorization_done) {
                factorize();
            }

            // Dimensions:
            // C is m × n
            // op(B) is m × k
            // op(A^{-1}) is k × n (first n columns/rows of A^{-1})

            // Solve op(A) * X = I_n to get X = first n columns of op(A^{-1})
            // For sparse SPD matrices, solving for multiple RHS at once is more
            // numerically stable than solving column-by-column.
            Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(k, n);
            Eigen::MatrixXd A_inv;  // Will be k × n

            if (trans_A == Op::NoTrans) {
                // Solve A * X = I_n
                A_inv = chol_solver.solve(I_n);
            } else {
                // For SPD matrices, A^T = A, so this is the same as NoTrans
                // But we implement it anyway for generality
                A_inv = chol_solver.solve(I_n);
            }

            // Scale C by beta
            if (beta != (T)1.0) {
                blas::scal(m * n, beta, C, 1);
            }

            // Now compute each column of C: C[:, j] = alpha * op(B) * A_inv[:, j] + C[:, j]
            // Use gemv for each column
            for (int64_t j = 0; j < n; ++j) {
                // Get pointer to column j of A_inv (in Eigen's column-major storage)
                const T* a_inv_col = A_inv.data() + j * k;

                // Get pointer to column j of C and its increment
                auto [c_col, inc_c] = get_c_column(j, layout, C, ldc);

                // Compute C[:, j] += alpha * op(B) * A_inv[:, j]
                // op(B) is m × k, A_inv[:, j] is k × 1, result is m × 1
                blas::gemv(layout, trans_B, m, k, alpha, B, ldb, a_inv_col, 1, (T)1.0, c_col, inc_c);
            }

            return;
        }

        // Validate parameters for Side::Left
        randblas_require(trans_A == Op::NoTrans || trans_A == Op::Trans);
        randblas_require(trans_B == Op::NoTrans || trans_B == Op::Trans);

        // Ensure factorization is computed
        if (!factorization_done) {
            factorize();
        }

        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        // Layout-aware dimension checks:
        // - ColMajor: ldb is stride between columns, must be >= rows
        // - RowMajor: ldb is stride between rows, must be >= cols
        if (layout == Layout::ColMajor) {
            randblas_require(ldb >= rows_B);
            randblas_require(ldc >= m);
        } else {  // RowMajor
            randblas_require(ldb >= cols_B);
            randblas_require(ldc >= n);
        }
        randblas_require(m == n_rows);
        randblas_require(k == n_cols);

        // Scale C by beta
        if (beta != (T)1.0) {
            blas::scal(m * n, beta, C, 1);
        }

        // Process each column of op(B)
        for (int64_t j = 0; j < n; ++j) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> b_col(k);

            // Extract column j of op(B)
            extract_column_from_op_B(j, k, layout, trans_B, B, ldb, b_col);

            // Solve op(A) * x = b_col
            auto x = solve_with_transpose(b_col, trans_A);

            // Accumulate result into column j of C
            auto [c_col, inc_c] = get_c_column(j, layout, C, ldc);
            blas::axpy(m, alpha, x.data(), 1, c_col, inc_c);
        }
    }

    /// Sparse matrix multiplication operator (non-Side version): C := alpha * A^{-1} * op(B_sp) + beta * C
    /// Delegates to Side::Left version
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

    /// Sparse matrix multiplication operator (Side version): C := alpha * op(A^{-1}) * op(B_sp) + beta * C
    /// Supports both Side::Left and Side::Right via the dense operator
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
        // For sparse B, densify and delegate to dense operator (which handles Side::Right)
        // Calculate the dimensions of B before any transpose operation
        int64_t rows_B, cols_B;
        if (side == Side::Left) {
            // Side::Left: C := alpha * op(A) * op(B) + beta * C
            // A is m × k, B is k × n after transpose
            auto [rb, cb] = RandBLAS::dims_before_op(k, n, trans_B);
            rows_B = rb;
            cols_B = cb;
        } else {
            // Side::Right: C := alpha * op(B) * op(A) + beta * C
            // A is k × n, B is m × k after transpose
            auto [rb, cb] = RandBLAS::dims_before_op(m, k, trans_B);
            rows_B = rb;
            cols_B = cb;
        }

        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

        T* B_dense = new T[rows_B * cols_B]();
        RandLAPACK::util::sparse_to_dense_summing_duplicates(B_sp, layout, B_dense);

        (*this)(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);

        delete[] B_dense;
    }

    /// Sketch multiplication operator (Side version): C := alpha * op(A^{-1}) * op(S) + beta * C
    /// or C := alpha * op(S) * op(A^{-1}) + beta * C depending on Side
    ///
    /// @tparam SkOp RandBLAS sketching operator type (DenseSkOp or SparseSkOp)
    /// @param side Multiplication side (Left or Right)
    /// @param layout Memory layout of C (ColMajor or RowMajor)
    /// @param trans_A Transpose operation for this operator A (NoTrans or Trans)
    /// @param trans_S Transpose operation for sketching operator S (NoTrans or Trans)
    /// @param m Number of rows in result matrix C
    /// @param n Number of columns in result matrix C
    /// @param k Inner dimension for the multiplication
    /// @param alpha Scalar multiplier for the product
    /// @param S Reference to sketching operator
    /// @param beta Scalar multiplier for C
    /// @param C Pointer to output matrix C (modified in-place)
    /// @param ldc Leading dimension of C (layout-dependent)
    ///
    /// @details
    /// - Side::Left:  C := alpha * op(A^{-1}) * op(S) + beta * C
    /// - Side::Right: C := alpha * op(S) * op(A^{-1}) + beta * C
    ///
    /// @note Current implementation materializes the sketch and uses dense operator.
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
        // Materialize the sketch into a dense matrix
        T* S_dense = new T[S.n_rows * S.n_cols]();
        int64_t lds = (layout == Layout::ColMajor) ? S.n_rows : S.n_cols;

        // Fill S_dense with identity and apply sketch
        // For ColMajor: each column is a unit vector, sketched to get columns of S
        if (layout == Layout::ColMajor) {
            for (int64_t j = 0; j < S.n_cols; ++j) {
                S_dense[j * lds + j] = (T)1.0;  // Only works if S.n_rows >= S.n_cols
            }
        } else {
            for (int64_t i = 0; i < S.n_rows; ++i) {
                S_dense[i * lds + i] = (T)1.0;
            }
        }
        // Actually, we need to use RandBLAS to materialize the sketch properly
        // Use sketch_general with identity matrix
        T* I_mat = new T[S.n_cols * S.n_cols]();
        for (int64_t i = 0; i < S.n_cols; ++i) {
            I_mat[i * S.n_cols + i] = (T)1.0;
        }
        // S_dense = S * I = S (materialized)
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                  S.n_rows, S.n_cols, S.n_cols,
                                  (T)1.0, S, I_mat, S.n_cols, (T)0.0, S_dense, S.n_rows);
        delete[] I_mat;

        // Now use the dense operator
        (*this)(side, layout, trans_A, trans_S, m, n, k, alpha, S_dense, lds, beta, C, ldc);

        delete[] S_dense;
    }
};

} // namespace RandLAPACK_demos
