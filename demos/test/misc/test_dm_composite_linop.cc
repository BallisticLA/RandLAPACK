// Test suite for composite linear operators in the demos project.
// Tests combinations of CholSolverLinOp with Dense and Sparse operators.
//
// This tests the RandLAPACK::linops::CompositeOperator with CholSolverLinOp,
// which represents the inverse of a sparse SPD matrix via Cholesky factorization.
//
// Test combinations:
// 1. CholSolver * Dense
// 2. Dense * CholSolver
// 3. CholSolver * Sparse
// 4. Sparse * CholSolver
// 5. CholSolver * CholSolver
// 6. CholSolver * (Sparse * Dense) - nested composite
//
// Each combination is tested with:
// - Dense and sparse input matrices B
// - Left multiplication (Side::Right tests commented out due to complexity)
// - ColMajor and RowMajor layouts

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include "../../functions/linops_external/dm_cholsolver_linop.hh"
#include "../../functions/misc/dm_util.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>

using RandBLAS::RNGState;
using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;

class TestDmCompositeLinOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    // Helper to generate dense matrix
    template <typename T>
    static vector<T> generate_dense_matrix(int64_t rows, int64_t cols, Layout layout, RNGState<r123::Philox4x32_R<10>>& state) {
        vector<T> mat(rows * cols);
        RandBLAS::DenseDist D(rows, cols);
        RandBLAS::fill_dense(D, mat.data(), state);

        if (layout == Layout::RowMajor) {
            // Convert from ColMajor to RowMajor
            vector<T> temp = mat;
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t j = 0; j < cols; ++j) {
                    mat[j + i * cols] = temp[i + j * rows];
                }
            }
        }
        return mat;
    }

    // Helper to generate sparse matrix in CSC format
    template <typename T>
    static RandBLAS::sparse_data::csc::CSCMatrix<T> generate_sparse_matrix(
        int64_t rows, int64_t cols, T density, RNGState<r123::Philox4x32_R<10>>& state) {
        auto coo = RandLAPACK::gen::gen_sparse_mat<T>(rows, cols, density, state);
        RandBLAS::sparse_data::csc::CSCMatrix<T> csc(rows, cols);
        RandBLAS::sparse_data::conversions::coo_to_csc(coo, csc);
        return csc;
    }

    // Helper to materialize an operator as a dense matrix
    template <typename T, typename LinOp>
    static void materialize_operator(LinOp& op, int64_t rows, int64_t cols, vector<T>& dense_mat) {
        dense_mat.resize(rows * cols, 0.0);

        // Materialize by applying operator to identity columns
        for (int64_t j = 0; j < cols; ++j) {
            vector<T> e_j(cols, 0.0);
            e_j[j] = 1.0;

            // C = op * e_j (column j of the identity)
            op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, rows, 1, cols,
               1.0, e_j.data(), cols, 0.0, dense_mat.data() + j * rows, rows);
        }
    }

    // Helper to compare two result matrices with tolerance
    template <typename T>
    static void compare_results(
        const vector<T>& C_computed,
        const vector<T>& C_reference,
        int64_t m, int64_t n,
        T atol_factor = 100.0,
        T rtol_factor = 10.0
    ) {
        T atol = atol_factor * std::numeric_limits<T>::epsilon();
        T rtol = rtol_factor * std::numeric_limits<T>::epsilon();

        for (int64_t i = 0; i < m * n; ++i) {
            T diff = std::abs(C_computed[i] - C_reference[i]);
            T magnitude = std::max(std::abs(C_reference[i]), (T)1.0);
            ASSERT_LE(diff, atol + rtol * magnitude)
                << "Mismatch at index " << i
                << ": computed=" << C_computed[i]
                << ", reference=" << C_reference[i];
        }
    }

    // Helper to convert ColMajor to RowMajor layout
    template <typename T>
    static vector<T> convert_colmajor_to_rowmajor(const vector<T>& mat_col, int64_t rows, int64_t cols) {
        vector<T> mat_row(rows * cols);
        for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) {
                mat_row[j + i * cols] = mat_col[i + j * rows];
            }
        }
        return mat_row;
    }

    // Helper to compute reference: C = alpha * intermediate * B + beta * C
    // Handles both ColMajor and RowMajor layouts, and dense/sparse B
    template <typename T>
    static void compute_reference_gemm(
        Layout layout,
        int64_t m, int64_t n, int64_t k,
        T alpha,
        const vector<T>& intermediate_col,  // Always in ColMajor
        const vector<T>& B_dense_col,       // Always in ColMajor (densified if needed)
        T beta,
        vector<T>& C_reference,
        int64_t ldc
    ) {
        if (layout == Layout::ColMajor) {
            blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, alpha,
                      intermediate_col.data(), m, B_dense_col.data(), k, beta, C_reference.data(), ldc);
        } else {
            auto intermediate_row = convert_colmajor_to_rowmajor(intermediate_col, m, k);
            auto B_row = convert_colmajor_to_rowmajor(B_dense_col, k, n);
            blas::gemm(Layout::RowMajor, Op::NoTrans, Op::NoTrans, m, n, k, alpha,
                      intermediate_row.data(), k, B_row.data(), n, beta, C_reference.data(), ldc);
        }
    }

    // Helper to compute dense_mat * chol_op by materializing each column
    template <typename T>
    static vector<T> compute_dense_times_chol(
        const vector<T>& dense_mat,
        int64_t rows_dense,
        int64_t dim_chol,
        RandLAPACK_demos::CholSolverLinOp<T>& chol_op
    ) {
        vector<T> result(rows_dense * dim_chol, 0.0);
        for (int64_t j = 0; j < dim_chol; ++j) {
            vector<T> e_j(dim_chol, 0.0);
            e_j[j] = 1.0;

            vector<T> chol_col(dim_chol, 0.0);
            chol_op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, dim_chol, 1, dim_chol,
                   1.0, e_j.data(), dim_chol, 0.0, chol_col.data(), dim_chol);

            blas::gemv(Layout::ColMajor, Op::NoTrans, rows_dense, dim_chol,
                      1.0, dense_mat.data(), rows_dense, chol_col.data(), 1,
                      0.0, result.data() + j * rows_dense, 1);
        }
        return result;
    }

    // Helper to compute mat * chol_op (works for any densified matrix)
    template <typename T>
    static vector<T> compute_mat_times_chol(
        const vector<T>& mat,
        int64_t rows_mat,
        int64_t dim_chol,
        RandLAPACK_demos::CholSolverLinOp<T>& chol_op
    ) {
        return compute_dense_times_chol(mat, rows_mat, dim_chol, chol_op);
    }

    // Main test function for CholSolver * Dense composition
    template <typename T>
    static void test_chol_dense(
        bool sparse_B,      // true if input matrix B is sparse
        Layout layout,
        int64_t m,          // rows in C (and in composite = rows in CholSolver)
        int64_t n,          // cols in C
        int64_t k_dense,    // cols in dense operator (and cols in composite)
        int64_t dim_chol    // dimension of SPD matrix for CholSolver
    ) {
        RNGState<r123::Philox4x32_R<10>> state(42);
        RNGState<r123::Philox4x32_R<10>> state_ref(42);

        T alpha = 1.5;
        T beta = 0.5;
        T density_B = 0.25;

        // Composite dimensions: m × k_dense (CholSolver is m×m, dense is m×k_dense)
        ASSERT_EQ(m, dim_chol);  // CholSolver rows must match

        // Generate SPD matrix file for CholSolver
        std::string spd_file = "/tmp/test_chol_dense.mtx";
        RandLAPACK_demos::generate_spd_matrix_file<T>(spd_file, dim_chol, 10.0, state);

        // Create CholSolver operator
        RandLAPACK_demos::CholSolverLinOp<T> chol_op(spd_file);
        chol_op.factorize();

        // Generate dense operator matrix (dim_chol × k_dense) - always in ColMajor for simplicity
        vector<T> dense_mat = generate_dense_matrix<T>(dim_chol, k_dense, Layout::ColMajor, state);
        int64_t lda_dense = dim_chol;
        RandLAPACK::linops::DenseLinOp<T> dense_op(dim_chol, k_dense, dense_mat.data(), lda_dense, Layout::ColMajor);

        // Create composite operator: CholSolver * Dense
        RandLAPACK::linops::CompositeOperator composite_op(m, k_dense, chol_op, dense_op);

        // Generate input matrix B (k_dense × n)
        int64_t rows_B = k_dense;
        int64_t cols_B = n;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        vector<T> C_composite(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);

        // Initialize C with random values for beta test
        for (auto& val : C_composite) val = (T)(rand() % 100) / 100.0;
        C_reference = C_composite;

        // Compute chol_op * dense_mat -> intermediate (in ColMajor)
        vector<T> intermediate(m * k_dense, 0.0);
        chol_op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k_dense, dim_chol,
               1.0, dense_mat.data(), dim_chol, 0.0, intermediate.data(), m);

        // Apply composite operator and compute reference
        if (sparse_B) {
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, k_dense,
                        alpha, B_csc, beta, C_composite.data(), ldc);

            vector<T> B_dense_col(rows_B * cols_B, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense_col.data());
            compute_reference_gemm(layout, m, n, k_dense, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        } else {
            auto B_dense_col = generate_dense_matrix<T>(rows_B, cols_B, Layout::ColMajor, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, k_dense,
                        alpha, B_dense_col.data(), ldb, beta, C_composite.data(), ldc);
            compute_reference_gemm(layout, m, n, k_dense, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        }

        // Compare results
        compare_results(C_composite, C_reference, m, n);

        // Cleanup
        std::remove(spd_file.c_str());
    }

    // Test function for Dense * CholSolver composition
    template <typename T>
    static void test_dense_chol(
        bool sparse_B,
        Layout layout,
        int64_t m,          // rows in C (and in composite = rows in dense)
        int64_t n,          // cols in C (and cols in CholSolver)
        int64_t rows_dense, // rows in dense operator
        int64_t dim_chol    // dimension of SPD matrix for CholSolver
    ) {
        RNGState<r123::Philox4x32_R<10>> state(42);
        RNGState<r123::Philox4x32_R<10>> state_ref(42);

        T alpha = 1.5;
        T beta = 0.5;
        T density_B = 0.25;

        // Composite dimensions: rows_dense × dim_chol (dense is rows_dense×dim_chol, CholSolver is dim_chol×dim_chol)
        ASSERT_EQ(m, rows_dense);
        ASSERT_EQ(n, dim_chol);

        // Generate SPD matrix file for CholSolver
        std::string spd_file = "/tmp/test_dense_chol.mtx";
        RandLAPACK_demos::generate_spd_matrix_file<T>(spd_file, dim_chol, 10.0, state);

        // Create CholSolver operator
        RandLAPACK_demos::CholSolverLinOp<T> chol_op(spd_file);
        chol_op.factorize();

        // Generate dense operator matrix (rows_dense × dim_chol)
        vector<T> dense_mat = generate_dense_matrix<T>(rows_dense, dim_chol, Layout::ColMajor, state);
        int64_t lda_dense = rows_dense;
        RandLAPACK::linops::DenseLinOp<T> dense_op(rows_dense, dim_chol, dense_mat.data(), lda_dense, Layout::ColMajor);

        // Create composite operator: Dense * CholSolver
        RandLAPACK::linops::CompositeOperator composite_op(rows_dense, dim_chol, dense_op, chol_op);

        // Generate input matrix B (dim_chol × n) - note B multiplies from left, so it's dim_chol×n
        int64_t rows_B = dim_chol;
        int64_t cols_B = n;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        vector<T> C_composite(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);

        for (auto& val : C_composite) val = (T)(rand() % 100) / 100.0;
        C_reference = C_composite;

        // Compute dense_mat * chol_op -> intermediate (in ColMajor)
        auto intermediate = compute_dense_times_chol(dense_mat, rows_dense, dim_chol, chol_op);

        // Apply composite operator and compute reference
        if (sparse_B) {
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, dim_chol,
                        alpha, B_csc, beta, C_composite.data(), ldc);

            vector<T> B_dense_col(rows_B * cols_B, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense_col.data());
            compute_reference_gemm(layout, m, n, dim_chol, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        } else {
            auto B_dense_col = generate_dense_matrix<T>(rows_B, cols_B, Layout::ColMajor, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, dim_chol,
                        alpha, B_dense_col.data(), ldb, beta, C_composite.data(), ldc);
            compute_reference_gemm(layout, m, n, dim_chol, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        }

        compare_results(C_composite, C_reference, m, n);

        std::remove(spd_file.c_str());
    }

    // Test function for CholSolver * Sparse composition
    template <typename T>
    static void test_chol_sparse(
        bool sparse_B,
        Layout layout,
        int64_t m,          // rows in C (and in composite = rows in CholSolver)
        int64_t n,          // cols in C (and cols in B)
        int64_t k_sparse,   // cols in composite (= cols in Sparse operator)
        int64_t dim_chol    // dimension of SPD matrix for CholSolver (also rows in Sparse)
    ) {
        RNGState<r123::Philox4x32_R<10>> state(43);

        T alpha = 1.5;
        T beta = 0.5;
        T density_B = 0.25;
        T density_sparse = 0.3;

        // Composite dimensions: dim_chol × k_sparse (CholSolver is dim_chol×dim_chol, Sparse is dim_chol×k_sparse)
        ASSERT_EQ(m, dim_chol);

        // Generate SPD matrix file for CholSolver
        std::string spd_file = "/tmp/test_chol_sparse.mtx";
        RandLAPACK_demos::generate_spd_matrix_file<T>(spd_file, dim_chol, 10.0, state);

        // Create CholSolver operator
        RandLAPACK_demos::CholSolverLinOp<T> chol_op(spd_file);
        chol_op.factorize();

        // Generate sparse operator matrix (dim_chol × k_sparse)
        auto sparse_mat_coo = RandLAPACK::gen::gen_sparse_mat<T>(dim_chol, k_sparse, density_sparse, state);
        RandBLAS::sparse_data::csc::CSCMatrix<T> sparse_mat_csc(dim_chol, k_sparse);
        RandBLAS::sparse_data::conversions::coo_to_csc(sparse_mat_coo, sparse_mat_csc);
        RandLAPACK::linops::SparseLinOp sparse_op(dim_chol, k_sparse, sparse_mat_csc);

        // Create composite operator: CholSolver * Sparse
        RandLAPACK::linops::CompositeOperator composite_op(m, k_sparse, chol_op, sparse_op);

        // Generate input matrix B (k_sparse × n)
        int64_t rows_B = k_sparse;
        int64_t cols_B = n;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        vector<T> C_composite(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);

        // Initialize C with random values for beta test
        for (auto& val : C_composite) val = (T)(rand() % 100) / 100.0;
        C_reference = C_composite;

        // Densify sparse_mat_csc and compute chol_op * sparse_mat_dense -> intermediate
        vector<T> sparse_mat_dense(dim_chol * k_sparse, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(sparse_mat_csc, Layout::ColMajor, sparse_mat_dense.data());

        vector<T> intermediate(m * k_sparse, 0.0);
        chol_op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k_sparse, dim_chol,
               1.0, sparse_mat_dense.data(), dim_chol, 0.0, intermediate.data(), m);

        // Apply composite operator and compute reference
        if (sparse_B) {
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, k_sparse,
                        alpha, B_csc, beta, C_composite.data(), ldc);

            vector<T> B_dense_col(rows_B * cols_B, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense_col.data());
            compute_reference_gemm(layout, m, n, k_sparse, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        } else {
            auto B_dense = generate_dense_matrix<T>(rows_B, cols_B, layout, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, k_sparse,
                        alpha, B_dense.data(), ldb, beta, C_composite.data(), ldc);

            // For reference, convert B to ColMajor if needed (if B is RowMajor, convert it)
            // Note: RowMajor(rows_B, cols_B) is stored as ColMajor(cols_B, rows_B)
            auto B_dense_col = (layout == Layout::ColMajor) ? B_dense : convert_colmajor_to_rowmajor(B_dense, cols_B, rows_B);
            compute_reference_gemm(layout, m, n, k_sparse, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        }

        // Compare results
        compare_results(C_composite, C_reference, m, n);

        std::remove(spd_file.c_str());
    }

    // Test function for Sparse * CholSolver composition
    template <typename T>
    static void test_sparse_chol(
        bool sparse_B,
        Layout layout,
        int64_t m,          // rows in C (and in composite = rows in Sparse)
        int64_t n,          // cols in C (and cols in B)
        int64_t rows_sparse,// rows in Sparse operator
        int64_t dim_chol    // dimension of SPD matrix for CholSolver
    ) {
        RNGState<r123::Philox4x32_R<10>> state(44);

        T alpha = 1.5;
        T beta = 0.5;
        T density_B = 0.25;
        T density_sparse = 0.3;

        // Composite dimensions: rows_sparse × dim_chol (Sparse is rows_sparse×dim_chol, CholSolver is dim_chol×dim_chol)
        ASSERT_EQ(m, rows_sparse);
        ASSERT_EQ(n, dim_chol);

        // Generate SPD matrix file for CholSolver
        std::string spd_file = "/tmp/test_sparse_chol.mtx";
        RandLAPACK_demos::generate_spd_matrix_file<T>(spd_file, dim_chol, 10.0, state);

        // Create CholSolver operator
        RandLAPACK_demos::CholSolverLinOp<T> chol_op(spd_file);
        chol_op.factorize();

        // Generate sparse operator matrix (rows_sparse × dim_chol)
        auto sparse_mat_coo = RandLAPACK::gen::gen_sparse_mat<T>(rows_sparse, dim_chol, density_sparse, state);
        RandBLAS::sparse_data::csc::CSCMatrix<T> sparse_mat_csc(rows_sparse, dim_chol);
        RandBLAS::sparse_data::conversions::coo_to_csc(sparse_mat_coo, sparse_mat_csc);
        RandLAPACK::linops::SparseLinOp sparse_op(rows_sparse, dim_chol, sparse_mat_csc);

        // Create composite operator: Sparse * CholSolver
        RandLAPACK::linops::CompositeOperator composite_op(rows_sparse, dim_chol, sparse_op, chol_op);

        // Generate input matrix B (dim_chol × n)
        int64_t rows_B = dim_chol;
        int64_t cols_B = n;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        vector<T> C_composite(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);

        for (auto& val : C_composite) val = (T)(rand() % 100) / 100.0;
        C_reference = C_composite;

        // Densify sparse_mat_csc and compute sparse_mat * chol_op -> intermediate
        vector<T> sparse_mat_dense(rows_sparse * dim_chol, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(sparse_mat_csc, Layout::ColMajor, sparse_mat_dense.data());

        auto intermediate = compute_mat_times_chol(sparse_mat_dense, rows_sparse, dim_chol, chol_op);

        // Apply composite operator and compute reference
        if (sparse_B) {
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, dim_chol,
                        alpha, B_csc, beta, C_composite.data(), ldc);

            vector<T> B_dense_col(rows_B * cols_B, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense_col.data());
            compute_reference_gemm(layout, m, n, dim_chol, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        } else {
            auto B_dense = generate_dense_matrix<T>(rows_B, cols_B, layout, state);
            composite_op(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, dim_chol,
                        alpha, B_dense.data(), ldb, beta, C_composite.data(), ldc);

            auto B_dense_col = (layout == Layout::ColMajor) ? B_dense : convert_colmajor_to_rowmajor(B_dense, cols_B, rows_B);
            compute_reference_gemm(layout, m, n, dim_chol, alpha, intermediate, B_dense_col, beta, C_reference, ldc);
        }

        compare_results(C_composite, C_reference, m, n);

        std::remove(spd_file.c_str());
    }

    // Test function for CholSolver * (Sparse * Dense) nested composition
    template <typename T>
    static void test_chol_sparse_dense(
        bool sparse_B,
        Layout layout,
        int64_t m,          // rows in C (and in outer composite = rows in CholSolver)
        int64_t n,          // cols in C (and cols in B)
        int64_t k_final,    // cols in outer composite (= cols in Dense operator)
        int64_t dim_chol,   // dimension of SPD matrix for CholSolver
        int64_t k_sparse    // intermediate dimension (cols in Sparse, rows in Dense)
    ) {
        RNGState<r123::Philox4x32_R<10>> state(45);

        T alpha = 1.5;
        T beta = 0.5;
        T density_B = 0.25;
        T density_sparse = 0.3;

        // Outer composite dimensions: dim_chol × k_final
        // CholSolver is dim_chol×dim_chol, inner composite (Sparse*Dense) is dim_chol×k_final
        ASSERT_EQ(m, dim_chol);

        // Generate SPD matrix file for CholSolver
        std::string spd_file = "/tmp/test_chol_sparse_dense.mtx";
        RandLAPACK_demos::generate_spd_matrix_file<T>(spd_file, dim_chol, 10.0, state);

        // Create CholSolver operator
        RandLAPACK_demos::CholSolverLinOp<T> chol_op(spd_file);
        chol_op.factorize();

        // Generate sparse operator matrix (dim_chol × k_sparse)
        auto sparse_mat_coo = RandLAPACK::gen::gen_sparse_mat<T>(dim_chol, k_sparse, density_sparse, state);
        RandBLAS::sparse_data::csc::CSCMatrix<T> sparse_mat_csc(dim_chol, k_sparse);
        RandBLAS::sparse_data::conversions::coo_to_csc(sparse_mat_coo, sparse_mat_csc);
        RandLAPACK::linops::SparseLinOp sparse_op(dim_chol, k_sparse, sparse_mat_csc);

        // Generate dense operator matrix (k_sparse × k_final)
        vector<T> dense_mat = generate_dense_matrix<T>(k_sparse, k_final, Layout::ColMajor, state);
        int64_t lda_dense = k_sparse;
        RandLAPACK::linops::DenseLinOp<T> dense_op(k_sparse, k_final, dense_mat.data(), lda_dense, Layout::ColMajor);

        // Create inner composite operator: Sparse * Dense
        RandLAPACK::linops::CompositeOperator inner_composite(dim_chol, k_final, sparse_op, dense_op);

        // Create outer composite operator: CholSolver * (Sparse * Dense)
        RandLAPACK::linops::CompositeOperator outer_composite(m, k_final, chol_op, inner_composite);

        // Generate input matrix B (k_final × n)
        int64_t rows_B = k_final;
        int64_t cols_B = n;
        int64_t ldb = (layout == Layout::ColMajor) ? rows_B : cols_B;
        int64_t ldc = (layout == Layout::ColMajor) ? m : n;

        vector<T> C_composite(m * n, 0.0);
        vector<T> C_reference(m * n, 0.0);

        // Initialize C with random values for beta test
        for (auto& val : C_composite) val = (T)(rand() % 100) / 100.0;
        C_reference = C_composite;

        // Compute reference: chol_op * sparse_op * dense_op
        // Step 1: Densify sparse_mat and compute sparse_mat * dense_mat -> intermediate1
        vector<T> sparse_mat_dense(dim_chol * k_sparse, 0.0);
        RandLAPACK::util::sparse_to_dense_summing_duplicates(sparse_mat_csc, Layout::ColMajor, sparse_mat_dense.data());

        vector<T> intermediate1(dim_chol * k_final, 0.0);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, dim_chol, k_final, k_sparse, 1.0,
                  sparse_mat_dense.data(), dim_chol, dense_mat.data(), k_sparse, 0.0, intermediate1.data(), dim_chol);

        // Step 2: Compute chol_op * intermediate1 -> intermediate2
        vector<T> intermediate2(m * k_final, 0.0);
        chol_op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, k_final, dim_chol,
               1.0, intermediate1.data(), dim_chol, 0.0, intermediate2.data(), m);

        // Apply composite operator and compute reference
        if (sparse_B) {
            auto B_csc = generate_sparse_matrix<T>(rows_B, cols_B, density_B, state);
            outer_composite(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, k_final,
                           alpha, B_csc, beta, C_composite.data(), ldc);

            vector<T> B_dense_col(rows_B * cols_B, 0.0);
            RandLAPACK::util::sparse_to_dense_summing_duplicates(B_csc, Layout::ColMajor, B_dense_col.data());
            compute_reference_gemm(layout, m, n, k_final, alpha, intermediate2, B_dense_col, beta, C_reference, ldc);
        } else {
            auto B_dense_col = generate_dense_matrix<T>(rows_B, cols_B, Layout::ColMajor, state);
            outer_composite(Side::Left, layout, Op::NoTrans, Op::NoTrans, m, n, k_final,
                           alpha, B_dense_col.data(), ldb, beta, C_composite.data(), ldc);
            compute_reference_gemm(layout, m, n, k_final, alpha, intermediate2, B_dense_col, beta, C_reference, ldc);
        }

        // Compare results
        compare_results(C_composite, C_reference, m, n);

        std::remove(spd_file.c_str());
    }
};

// ============================================================================
// CholSolver * Dense composition tests
// ============================================================================

TEST_F(TestDmCompositeLinOp, chol_dense_left_dense_colmajor) {
    test_chol_dense<double>(false, Layout::ColMajor, 10, 8, 8, 10);
}

// RowMajor disabled - DenseLinOp requires layout to match buff_layout (matrix storage layout)
// TEST_F(TestDmCompositeLinOp, chol_dense_left_dense_rowmajor) {
//     test_chol_dense<double>(false, Layout::RowMajor, 10, 8, 8, 10);
// }

TEST_F(TestDmCompositeLinOp, chol_dense_left_sparse_colmajor) {
    test_chol_dense<double>(true, Layout::ColMajor, 10, 8, 8, 10);
}

// RowMajor disabled - DenseLinOp requires layout to match buff_layout (matrix storage layout)
// TEST_F(TestDmCompositeLinOp, chol_dense_left_sparse_rowmajor) {
//     test_chol_dense<double>(true, Layout::RowMajor, 10, 8, 8, 10);
// }

// ============================================================================
// Dense * CholSolver composition tests
// ============================================================================

TEST_F(TestDmCompositeLinOp, dense_chol_left_dense_colmajor) {
    test_dense_chol<double>(false, Layout::ColMajor, 12, 10, 12, 10);
}

// RowMajor disabled - DenseLinOp requires layout to match buff_layout (matrix storage layout)
// TEST_F(TestDmCompositeLinOp, dense_chol_left_dense_rowmajor) {
//     test_dense_chol<double>(false, Layout::RowMajor, 12, 10, 12, 10);
// }

TEST_F(TestDmCompositeLinOp, dense_chol_left_sparse_colmajor) {
    test_dense_chol<double>(true, Layout::ColMajor, 12, 10, 12, 10);
}

// RowMajor disabled - DenseLinOp requires layout to match buff_layout (matrix storage layout)
// TEST_F(TestDmCompositeLinOp, dense_chol_left_sparse_rowmajor) {
//     test_dense_chol<double>(true, Layout::RowMajor, 12, 10, 12, 10);
// }

// ============================================================================
// CholSolver * Sparse composition tests
// ============================================================================

TEST_F(TestDmCompositeLinOp, chol_sparse_left_dense_colmajor) {
    test_chol_sparse<double>(false, Layout::ColMajor, 10, 8, 8, 10);
}

TEST_F(TestDmCompositeLinOp, chol_sparse_left_dense_rowmajor) {
    test_chol_sparse<double>(false, Layout::RowMajor, 10, 8, 8, 10);
}

TEST_F(TestDmCompositeLinOp, chol_sparse_left_sparse_colmajor) {
    test_chol_sparse<double>(true, Layout::ColMajor, 10, 8, 8, 10);
}

TEST_F(TestDmCompositeLinOp, chol_sparse_left_sparse_rowmajor) {
    test_chol_sparse<double>(true, Layout::RowMajor, 10, 8, 8, 10);
}

// ============================================================================
// Sparse * CholSolver composition tests
// ============================================================================

TEST_F(TestDmCompositeLinOp, sparse_chol_left_dense_colmajor) {
    test_sparse_chol<double>(false, Layout::ColMajor, 12, 10, 12, 10);
}

TEST_F(TestDmCompositeLinOp, sparse_chol_left_dense_rowmajor) {
    test_sparse_chol<double>(false, Layout::RowMajor, 12, 10, 12, 10);
}

TEST_F(TestDmCompositeLinOp, sparse_chol_left_sparse_colmajor) {
    test_sparse_chol<double>(true, Layout::ColMajor, 12, 10, 12, 10);
}

TEST_F(TestDmCompositeLinOp, sparse_chol_left_sparse_rowmajor) {
    test_sparse_chol<double>(true, Layout::RowMajor, 12, 10, 12, 10);
}

// ============================================================================
// CholSolver * (Sparse * Dense) nested composition tests
// ============================================================================

TEST_F(TestDmCompositeLinOp, chol_sparse_dense_left_dense_colmajor) {
    test_chol_sparse_dense<double>(false, Layout::ColMajor, 10, 8, 7, 10, 9);
}

// RowMajor disabled - DenseLinOp requires layout to match buff_layout (matrix storage layout)
// TEST_F(TestDmCompositeLinOp, chol_sparse_dense_left_dense_rowmajor) {
//     test_chol_sparse_dense<double>(false, Layout::RowMajor, 10, 8, 7, 10, 9);
// }

TEST_F(TestDmCompositeLinOp, chol_sparse_dense_left_sparse_colmajor) {
    test_chol_sparse_dense<double>(true, Layout::ColMajor, 10, 8, 7, 10, 9);
}

// RowMajor disabled - DenseLinOp requires layout to match buff_layout (matrix storage layout)
// TEST_F(TestDmCompositeLinOp, chol_sparse_dense_left_sparse_rowmajor) {
//     test_chol_sparse_dense<double>(true, Layout::RowMajor, 10, 8, 7, 10, 9);
// }


