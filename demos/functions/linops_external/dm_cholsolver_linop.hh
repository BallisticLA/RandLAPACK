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
/// Implementation rationale for sparse B
/// The operator processes input matrix B column-by-column, solving A * x = b for each column.
/// When B is sparse, we densify it before processing because:
/// 1. Sparse triangular solves effectively treat input vectors as dense (must touch all entries)
/// 2. The result A^{-1} * b_sparse is dense regardless of input sparsity
/// 3. Full densification is simpler and avoids repeated column extraction overhead
///
/// This operator is compatible with RandLAPACK::linops::CompositeOperator.
template <typename T>
struct CholSolverLinOp {
    using scalar_t = T;
    const int64_t n_rows;
    const int64_t n_cols;
    std::string matrix_file;

    // Eigen sparse Cholesky solver
    Eigen::SimplicialLLT<Eigen::SparseMatrix<T>> chol_solver;

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
        // Handle Side::Right by materializing A^{-1}
        // This is necessary because Eigen's Cholesky solver is column-oriented
        if (side == Side::Right) {
            // Want: C := alpha * op(B) * op(A^{-1}) + beta * C
            // Strategy: Materialize A^{-1}, then use standard BLAS gemm

            // Ensure factorization is computed
            if (!factorization_done) {
                factorize();
            }

            // Materialize A^{-1} as a dense n_rows × n_cols matrix
            std::vector<T> A_inv_dense(n_rows * n_cols);

            // Compute A^{-1} by solving A * X = I column-by-column
            for (int64_t j = 0; j < n_cols; ++j) {
                std::vector<T> e_j(n_cols, (T)0.0);
                e_j[j] = (T)1.0;
                Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b_vec(e_j.data(), n_cols);
                Eigen::Matrix<T, Eigen::Dynamic, 1> x;

                if (trans_A == Op::NoTrans) {
                    x = chol_solver.solve(b_vec);
                } else {
                    // Solve A^T * x = e_j, where A^T = L^T * L
                    // Step 1: L * y = e_j  (solve for y)
                    // Step 2: L^T * x = y (solve for x)
                    auto L = chol_solver.matrixL();
                    auto LT = chol_solver.matrixU();
                    Eigen::Matrix<T, Eigen::Dynamic, 1> y = L.solve(b_vec);
                    x = LT.solve(y);
                }

                // Store column j of A^{-1} (or A^{-T})
                for (int64_t i = 0; i < n_rows; ++i) {
                    A_inv_dense[i + j * n_rows] = x(i);
                }
            }

            // Now compute C := alpha * op(B) * op(A^{-1}) + beta * C using BLAS gemm
            // This is: C := alpha * op(B) * op(A_inv_dense) + beta * C
            // Note: BLAS gemm can handle any layout, so no layout restriction here
            blas::gemm(layout, trans_B, trans_A, m, n, k, alpha, B, ldb,
                      A_inv_dense.data(), n_rows, beta, C, ldc);
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

        // Solve op(A^{-1}) * op(B) for each column using sparse Cholesky solver
        // For A = L * L^T (Cholesky factorization):
        //   - A^{-1} * b: solve L * L^T * x = b
        //   - A^{-T} * b: solve L^T * L * x = b (step 1: L*y=b, step 2: L^T*x=y)

        // Define solve function based on trans_A
        auto solve_column = [&](const Eigen::Matrix<T, Eigen::Dynamic, 1>& b) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            if (trans_A == Op::NoTrans) {
                return chol_solver.solve(b);
            } else {
                auto L = chol_solver.matrixL();
                auto LT = chol_solver.matrixU();
                Eigen::Matrix<T, Eigen::Dynamic, 1> y = L.solve(b);
                return LT.solve(y);
            }
        };

        // Process each column of op(B)
        std::vector<T> b_temp(k);  // Temporary buffer for column extraction

        for (int64_t j = 0; j < n; ++j) {
            Eigen::Matrix<T, Eigen::Dynamic, 1> b_col(k);

            // Extract column j of op(B) based on layout and trans_B
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

            // Solve and accumulate result
            Eigen::Matrix<T, Eigen::Dynamic, 1> x = solve_column(b_col);

            // Write to column j of C based on layout
            if (layout == Layout::ColMajor) {
                blas::axpy(m, alpha, x.data(), 1, C + j * ldc, 1);
            } else {  // RowMajor
                blas::axpy(m, alpha, x.data(), 1, C + j, ldc);
            }
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
};

} // namespace RandLAPACK_demos
