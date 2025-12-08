#pragma once

#include "rl_util.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "dm_util.hh"

#include <RandBLAS.hh>
#include <string>
#include <fstream>
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
    int64_t n_rows;
    int64_t n_cols;
    std::string matrix_file;

    // Eigen sparse Cholesky solver
    Eigen::SimplicialLLT<Eigen::SparseMatrix<T>> chol_solver;

    bool factorization_done;

    // Constructtor - checks the header of teh sparse matrix; does not load data until it is needed.
    CholSolverLinOp(
        const std::string& filename
    ) : matrix_file(filename), factorization_done(false) {
        // Read matrix dimensions from Matrix Market file
        std::ifstream file(filename);
        randblas_require(file.is_open());

        // Read Matrix Market header
        std::string line;
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        // Parse dimensions
        std::istringstream iss(line);
        int64_t nnz;
        iss >> n_rows >> n_cols >> nnz;
        file.close();

        randblas_require(n_rows == n_cols); // Must be square for Cholesky
    }

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
            std::cerr << "CholSolverLinOp: Cholesky factorization failed!" << std::endl;
            randblas_require(false);
        }

        std::cerr << "CholSolverLinOp: Matrix loaded (" << A_eigen.nonZeros() << " nonzeros). ";
        std::cerr << "Sparse Cholesky factorization completed successfully." << std::endl;

        factorization_done = true;
    }

    /// Dense matrix multiplication operator: C := alpha * A^{-1} * op(B) + beta * C
    /// where A^{-1} is computed via sparse Cholesky solve: x = L^{-T} * L^{-1} * b
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
        randblas_require(layout == Layout::ColMajor);
        randblas_require(trans_A == Op::NoTrans); // Only A^{-1} supported, not (A^{-1})^T
        randblas_require(trans_B == Op::NoTrans || trans_B == Op::Trans);

        // Ensure factorization is computed
        if (!factorization_done) {
            factorize();
        }

        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        randblas_require(ldb >= rows_B);
        randblas_require(ldc >= m);
        randblas_require(m == n_rows);
        randblas_require(k == n_cols);

        // Scale C by beta
        // Below AXPY function does not have a beta parameter,  
        // hence we scale preemptively 
        if (beta != (T)1.0) {
            blas::scal(m * n, beta, C, 1);
        }

        // Solve A^{-1} * op(B) for each column using sparse Cholesky solver
        // This computes x such that A * x = b via: L * L^T * x = b, 
        // since Eigen's sparse solver can only accept vector rhs
        if (trans_B == Op::NoTrans) {
            // B is k × n, proceed by column
            for (int64_t j = 0; j < n; ++j) {
                // Below wrapper (Eigen::map) lets Eigen operate on external memeory without performing a copy
                Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b_col(B + j * ldb, k);
                Eigen::Matrix<T, Eigen::Dynamic, 1> x = chol_solver.solve(b_col);
                // We can access data of an Eigen matrix as a raw pointer
                blas::axpy(m, alpha, x.data(), 1, C + j * ldc, 1);
            }
        } else {
            // trans_B == Op::Trans: B is n × k, need to extract rows (which become columns of B^T)
            std::vector<T> b_row(k);
            for (int64_t j = 0; j < n; ++j) {
                // Extract row j from B (stride ldb between elements)
                for (int64_t i = 0; i < k; ++i) {
                    b_row[i] = B[j + i * ldb];
                }
                Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> b_vec(b_row.data(), k);
                Eigen::Matrix<T, Eigen::Dynamic, 1> x = chol_solver.solve(b_vec);
                blas::axpy(m, alpha, x.data(), 1, C + j * ldc, 1);
            }
        }
    }

    /// Augmented dense matrix multiplication operator with Side parameter.
    /// Side::Left:  C := alpha * A^{-1} * op(B) + beta * C
    /// Side::Right: C := alpha * op(B) * A^{-1} + beta * C
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
            // Left multiplication: delegate to default operator
            (*this)(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        } else {
            // Right multiplication: C := alpha * op(B) * A^{-1} + beta * C
            // Use transpose trick: compute C^T := alpha * A^{-T} * op(B)^T + beta * C^T
            auto trans_trans_A = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;

            (*this)(trans_layout, trans_trans_A, trans_B, n, m, k, alpha, B, ldb, beta, C, ldc);
        }
    }

    /// Sparse matrix multiplication operator: C := alpha * A^{-1} * op(B_sp) + beta * C
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
        // For sparse B, densify and delegate to dense operator
        auto [rows_B, cols_B] = RandBLAS::dims_before_op(k, n, trans_B);
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;

        std::cerr << "For now, sparse * sparse is done via densifying the rhs matrix. This is suboptimal." << std::endl;

        T* B_dense = new T[rows_B * cols_B]();
        RandLAPACK::util::sparse_to_dense(B_sp, layout, B_dense);

        (*this)(layout, trans_A, trans_B, m, n, k, alpha, B_dense, ldb, beta, C, ldc);

        delete[] B_dense;
    }

    /// Augmented sparse matrix multiplication operator with Side parameter.
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
            (*this)(layout, trans_A, trans_B, m, n, k, alpha, B_sp, beta, C, ldc);
        } else {
            auto trans_trans_A = (trans_A == Op::NoTrans) ? Op::Trans : Op::NoTrans;
            auto trans_layout = (layout == Layout::ColMajor) ? Layout::RowMajor : Layout::ColMajor;

            (*this)(trans_layout, trans_trans_A, trans_B, n, m, k, alpha, B_sp, beta, C, ldc);
        }
    }
};

} // namespace RandLAPACK_demos
