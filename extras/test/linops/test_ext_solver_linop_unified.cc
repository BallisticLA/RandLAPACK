// Unified templated tests for solver LinearOperator implementations
// Tests CholSolverLinOp and LUSolverLinOp using a common templated framework.
//
// The test framework uses tag types to specialize behavior:
//   - CholSolverTag<T>: SPD matrix, Cholesky factorization, symmetric inverse
//   - LUSolverTag<T>:   Invertible matrix, LU factorization, non-symmetric inverse
//
// Both solver types are tested with:
//   - Full-solve mode: operator represents A^{-1}
//   - Half-solve mode: CholSolver gives L^{-1}, LUSolver gives L^{-1} * P_r
//   - All combinations of Side, Layout, trans_A, trans_B
//   - Both dense and sparse input matrices B

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <lapack.hh>

#include "../../linops/ext_cholsolver_linop.hh"
#include "../../linops/ext_lusolver_linop.hh"
#include "../../misc/ext_util.hh"
#include "../ext_scratch_dir.hh"
#include <RandLAPACK/testing/rl_test_utils.hh>
#include <RandBLAS/testing/comparison.hh>
#include <algorithm>
#include <cmath>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::RNGState;
using namespace RandLAPACK::testing;

// ============================================================================
// Solver type tags
// ============================================================================

template <typename T>
struct CholSolverTag {
    using scalar_t = T;
    using op_type = RandLAPACK_extras::linops::CholSolverLinOp<T>;
};

template <typename T>
struct LUSolverTag {
    using scalar_t = T;
    using op_type = RandLAPACK_extras::linops::LUSolverLinOp<T>;
};

// ============================================================================
// Tag-specialized helpers: matrix file generation
// ============================================================================

template <typename Tag>
void generate_matrix_file(
    const std::string& filename, int64_t n, typename Tag::scalar_t cond_num,
    RNGState<r123::Philox4x32_R<10>>& state);

template <>
void generate_matrix_file<CholSolverTag<double>>(
    const std::string& filename, int64_t n, double cond_num,
    RNGState<r123::Philox4x32_R<10>>& state
) {
    RandLAPACK::testing::generate_spd_matrix_file<double>(filename, n, cond_num, state);
}

template <>
void generate_matrix_file<LUSolverTag<double>>(
    const std::string& filename, int64_t n, double cond_num,
    RNGState<r123::Philox4x32_R<10>>& state
) {
    RandLAPACK::testing::generate_invertible_matrix_file<double>(filename, n, cond_num, state);
}

// ============================================================================
// Tag-specialized helpers: make solver operator
// ============================================================================

template <typename Tag>
typename Tag::op_type make_solver_op(const std::string& filename, bool half_solve = false);

template <>
CholSolverTag<double>::op_type make_solver_op<CholSolverTag<double>>(
    const std::string& filename, bool half_solve
) {
    return RandLAPACK_extras::linops::CholSolverLinOp<double>(filename, half_solve);
}

template <>
LUSolverTag<double>::op_type make_solver_op<LUSolverTag<double>>(
    const std::string& filename, bool half_solve
) {
    return RandLAPACK_extras::linops::LUSolverLinOp<double>(filename, half_solve);
}

// ============================================================================
// Tag-specialized helpers: compute dense inverse for reference
// ============================================================================

// CholSolver: potrf + potri (SPD)
template <typename T>
void compute_dense_inverse_chol(int64_t n, const T* A, T* A_inv) {
    lapack::lacpy(lapack::MatrixType::General, n, n, A, n, A_inv, n);
    int64_t info = lapack::potrf(lapack::Uplo::Lower, n, A_inv, n);
    ASSERT_EQ(info, 0) << "Cholesky factorization failed in reference computation";
    info = lapack::potri(lapack::Uplo::Lower, n, A_inv, n);
    ASSERT_EQ(info, 0) << "Inverse computation failed in reference computation";
    // Fill upper triangle (potri only computes lower triangle)
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = i + 1; j < n; ++j)
            A_inv[i + j * n] = A_inv[j + i * n];
}

// LUSolver: getrf + getri (general)
template <typename T>
void compute_dense_inverse_lu(int64_t n, const T* A, T* A_inv) {
    lapack::lacpy(lapack::MatrixType::General, n, n, A, n, A_inv, n);
    vector<int64_t> ipiv(n);
    int64_t info = lapack::getrf(n, n, A_inv, n, ipiv.data());
    ASSERT_EQ(info, 0) << "LU factorization failed in reference computation";
    info = lapack::getri(n, A_inv, n, ipiv.data());
    ASSERT_EQ(info, 0) << "Inverse computation failed in reference computation";
}

template <typename Tag>
void compute_dense_inverse(int64_t n, const typename Tag::scalar_t* A, typename Tag::scalar_t* A_inv);

template <>
void compute_dense_inverse<CholSolverTag<double>>(int64_t n, const double* A, double* A_inv) {
    compute_dense_inverse_chol(n, A, A_inv);
}

template <>
void compute_dense_inverse<LUSolverTag<double>>(int64_t n, const double* A, double* A_inv) {
    compute_dense_inverse_lu(n, A, A_inv);
}

// ============================================================================
// Tag-specialized helpers: half-solve reference computation
// ============================================================================

// CholSolver half-solve: L^{-1}
template <typename T>
void compute_half_solve_ref_chol(int64_t n, const T* A, T* ref) {
    // Cholesky factorize to get L
    vector<T> L(n * n);
    lapack::lacpy(lapack::MatrixType::General, n, n, A, n, L.data(), n);
    int64_t info = lapack::potrf(lapack::Uplo::Lower, n, L.data(), n);
    ASSERT_EQ(info, 0) << "Cholesky factorization failed in half-solve reference computation";
    // Set ref = identity, then solve L * ref = I
    std::fill(ref, ref + n * n, (T)0.0);
    for (int64_t i = 0; i < n; ++i)
        ref[i + i * n] = (T)1.0;
    blas::trsm(Layout::ColMajor, Side::Left, blas::Uplo::Lower, Op::NoTrans,
               blas::Diag::NonUnit, n, n, (T)1.0, L.data(), n, ref, n);
}

// LUSolver half-solve: L^{-1} * P_r
template <typename T>
void compute_half_solve_ref_lu(int64_t n, const std::string& filename, T* ref) {
    Eigen::SparseMatrix<T, Eigen::ColMajor> A_eigen;
    RandLAPACK_extras::eigen_sparse_from_matrix_market<T>(filename, A_eigen);

    Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int>> lu;
    lu.compute(A_eigen);
    ASSERT_EQ(lu.info(), Eigen::Success) << "Eigen SparseLU failed in half-solve reference";

    // Extract L as dense
    Eigen::SparseMatrix<T> L_sp = lu.matrixL().toSparse();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> L_dense(L_sp);

    // Compute L^{-1} by solving L * X = I
    vector<T> L_inv(n * n, (T)0.0);
    for (int64_t i = 0; i < n; ++i)
        L_inv[i + i * n] = (T)1.0;
    blas::trsm(Layout::ColMajor, Side::Left, blas::Uplo::Lower, Op::NoTrans,
               blas::Diag::Unit, n, n, (T)1.0, L_dense.data(), n, L_inv.data(), n);

    // Apply P_r: (L^{-1} * P_r)[:, j] = column pr_indices[j] of L^{-1}
    const auto& pr_indices = lu.rowsPermutation().indices();
    for (int64_t j = 0; j < n; ++j) {
        int64_t src_col = pr_indices[j];
        blas::copy(n, L_inv.data() + src_col * n, 1, ref + j * n, 1);
    }
}

// ============================================================================
// Need-transpose helper: CholSolver is symmetric so trans_A doesn't matter
//                        for A_inv reference; LU needs explicit A^{-T}
// ============================================================================

template <typename Tag>
bool needs_explicit_transpose_for_trans_A();

template <> bool needs_explicit_transpose_for_trans_A<CholSolverTag<double>>() { return false; }
template <> bool needs_explicit_transpose_for_trans_A<LUSolverTag<double>>() { return true; }

// ============================================================================
// Unified test fixture
// ============================================================================

class TestExtSolverLinOpUnified : public ::testing::Test {
protected:
    virtual void SetUp() {};
    virtual void TearDown() {};

    // Full-solve test: C := alpha * op(A^{-1}) * op(B) + beta * C
    template <typename Tag>
    void test_solver_linop(
        Side side, bool sparse_B, Layout layout,
        Op trans_A, Op trans_B,
        int64_t m, int64_t n, int64_t k,
        typename Tag::scalar_t density = 0.0
    ) {
        using T = typename Tag::scalar_t;
        RNGState state(42);
        T alpha = 1.5;
        T beta = 0.5;

        // Solver dimension depends on side and transpose
        int64_t dim = (side == Side::Left) ?
                      ((trans_A == Op::NoTrans) ? m : k) :
                      ((trans_A == Op::NoTrans) ? k : n);

        RandLAPACK_extras::testing::ScratchDir scratch;
        std::string filename = scratch.file("A.mtx");

        T cond_num = 10.0;
        generate_matrix_file<Tag>(filename, dim, cond_num, state);

        // Load matrix for reference computation
        vector<T> A_mat(dim * dim);
        {
            auto A_coo = RandLAPACK_extras::coo_from_matrix_market<T>(filename);
            RandLAPACK::util::sparse_to_dense(A_coo, Layout::ColMajor, A_mat.data());
        }

        // Compute A^{-1} for reference
        vector<T> A_inv(dim * dim);
        compute_dense_inverse<Tag>(dim, A_mat.data(), A_inv.data());

        // For LU with trans_A == Trans, need A^{-T} = (A^{-1})^T
        if (needs_explicit_transpose_for_trans_A<Tag>() && trans_A == Op::Trans) {
            for (int64_t i = 0; i < dim; ++i)
                for (int64_t j = i + 1; j < dim; ++j)
                    std::swap(A_inv[i + j * dim], A_inv[j + i * dim]);
        }

        // Convert to target layout for reference GEMM
        if (layout == Layout::RowMajor) {
            convert_layout_inplace(A_inv, dim, dim, Layout::ColMajor, Layout::RowMajor);
        }

        auto dims = calculate_dimensions<T>(side, layout, trans_A, trans_B, m, n, k);

        vector<T> C_solver(m * n);
        vector<T> C_reference(m * n);
        initialize_test_buffers(C_solver, C_reference);

        auto A_inv_op = make_solver_op<Tag>(filename);

        if (sparse_B) {
            auto B_csc = RandLAPACK::gen::gen_sparse_coo<T>(dims.rows_B, dims.cols_B, density, state).as_owning_csc();

            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_csc, beta, C_solver.data(), dims.ldc);

            vector<T> B_dense(dims.rows_B * dims.cols_B);
            RandLAPACK::util::sparse_to_dense(B_csc, layout, B_dense.data());

            // For reference GEMM, use NoTrans since we already transposed A_inv above
            sided_gemm(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                       A_inv.data(), dim, B_dense.data(), dims.ldb,
                       beta, C_reference.data(), dims.ldc);
        } else {
            vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);

            A_inv_op(side, layout, trans_A, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_solver.data(), dims.ldc);

            sided_gemm(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                       A_inv.data(), dim, B_dense.data(), dims.ldb,
                       beta, C_reference.data(), dims.ldc);
        }

        T atol = 500 * std::numeric_limits<T>::epsilon() * dim;
        T rtol = 100 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_solver.data(), m,
            C_reference.data(), m, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    // Half-solve test for CholSolver: reference is L^{-1}
    template <typename T>
    void test_chol_half_solve(
        Side side, Layout layout, Op trans_B,
        int64_t m, int64_t n, int64_t k
    ) {
        RNGState state(42);
        T alpha = 1.5;
        T beta = 0.5;
        int64_t dim = (side == Side::Left) ? m : k;

        RandLAPACK_extras::testing::ScratchDir scratch;
        std::string filename = scratch.file("A.mtx");

        T cond_num = 10.0;
        RandLAPACK::testing::generate_spd_matrix_file<T>(filename, dim, cond_num, state);

        // Load SPD matrix
        vector<T> A_spd(dim * dim);
        {
            auto A_coo = RandLAPACK_extras::coo_from_matrix_market<T>(filename);
            RandLAPACK::util::sparse_to_dense(A_coo, Layout::ColMajor, A_spd.data());
        }

        // Compute L^{-1} reference
        vector<T> L_inv(dim * dim);
        compute_half_solve_ref_chol(dim, A_spd.data(), L_inv.data());

        if (layout == Layout::RowMajor) {
            // L^{-1} is NOT symmetric, need actual transpose
            for (int64_t i = 0; i < dim; ++i)
                for (int64_t j = i + 1; j < dim; ++j)
                    std::swap(L_inv[i + j * dim], L_inv[j + i * dim]);
        }

        auto dims = calculate_dimensions<T>(side, layout, Op::NoTrans, trans_B, m, n, k);
        vector<T> C_halfsv(m * n);
        vector<T> C_reference(m * n);
        initialize_test_buffers(C_halfsv, C_reference);

        RandLAPACK_extras::linops::CholSolverLinOp<T> L_inv_op(filename, /*half_solve=*/true);

        vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);
        L_inv_op(side, layout, Op::NoTrans, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_halfsv.data(), dims.ldc);

        sided_gemm(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                   L_inv.data(), dim, B_dense.data(), dims.ldb,
                   beta, C_reference.data(), dims.ldc);

        T atol = 500 * std::numeric_limits<T>::epsilon() * dim;
        T rtol = 100 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_halfsv.data(), m,
            C_reference.data(), m, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }

    // Half-solve test for LUSolver: reference is L^{-1} * P_r
    template <typename T>
    void test_lu_half_solve(
        Side side, Layout layout, Op trans_B,
        int64_t m, int64_t n, int64_t k
    ) {
        RNGState state(42);
        T alpha = 1.5;
        T beta = 0.5;
        int64_t dim = (side == Side::Left) ? m : k;

        RandLAPACK_extras::testing::ScratchDir scratch;
        std::string filename = scratch.file("A.mtx");

        T cond_num = 10.0;
        RandLAPACK::testing::generate_invertible_matrix_file<T>(filename, dim, cond_num, state);

        // Compute L^{-1} * P_r reference
        vector<T> M_ref(dim * dim);
        compute_half_solve_ref_lu(dim, filename, M_ref.data());

        if (layout == Layout::RowMajor) {
            // M is NOT symmetric, need actual transpose
            vector<T> M_tmp(dim * dim);
            for (int64_t i = 0; i < dim; ++i)
                for (int64_t j = 0; j < dim; ++j)
                    M_tmp[i * dim + j] = M_ref[i + j * dim];
            std::copy(M_tmp.begin(), M_tmp.end(), M_ref.begin());
        }

        auto dims = calculate_dimensions<T>(side, layout, Op::NoTrans, trans_B, m, n, k);
        vector<T> C_halfsv(m * n);
        vector<T> C_reference(m * n);
        initialize_test_buffers(C_halfsv, C_reference);

        RandLAPACK_extras::linops::LUSolverLinOp<T> M_op(filename, /*half_solve=*/true);

        vector<T> B_dense = generate_dense_matrix<T>(dims.rows_B, dims.cols_B, layout, state);
        M_op(side, layout, Op::NoTrans, trans_B, m, n, k, alpha, B_dense.data(), dims.ldb, beta, C_halfsv.data(), dims.ldc);

        sided_gemm(side, layout, Op::NoTrans, trans_B, m, n, k, alpha,
                   M_ref.data(), dim, B_dense.data(), dims.ldb,
                   beta, C_reference.data(), dims.ldc);

        T atol = 500 * std::numeric_limits<T>::epsilon() * dim;
        T rtol = 100 * std::numeric_limits<T>::epsilon();
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, m, n, C_halfsv.data(), m,
            C_reference.data(), m, __RANDBLAS_PRETTY_FUNCTION__, __FILE__, __LINE__,
            atol, rtol
        ), "");
    }
};

// ============================================================================
// CholSolver full-solve tests: Side::Left, dense B, ColMajor
// ============================================================================

TEST_F(TestExtSolverLinOpUnified, chol_left_dense_colmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_colmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_colmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_colmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// CholSolver: Side::Left, dense B, RowMajor
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_rowmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_rowmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_rowmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_dense_rowmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// CholSolver: Side::Right, dense B, ColMajor
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_colmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_colmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_colmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_colmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// CholSolver: Side::Right, dense B, RowMajor
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_rowmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_rowmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_rowmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_dense_rowmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// CholSolver: Side::Left, sparse B, ColMajor
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_colmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_colmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_colmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_colmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// CholSolver: Side::Left, sparse B, RowMajor
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_rowmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_rowmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_rowmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_left_sparse_rowmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// CholSolver: Side::Right, sparse B, ColMajor
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_colmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_colmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_colmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_colmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// CholSolver: Side::Right, sparse B, RowMajor
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_rowmajor_nn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_rowmajor_nt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_rowmajor_tn) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, chol_right_sparse_rowmajor_tt) {
    test_solver_linop<CholSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// ============================================================================
// CholSolver half-solve tests
// ============================================================================

TEST_F(TestExtSolverLinOpUnified, chol_halfsv_left_colmajor_n) {
    test_chol_half_solve<double>(Side::Left, Layout::ColMajor, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_left_colmajor_t) {
    test_chol_half_solve<double>(Side::Left, Layout::ColMajor, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_left_rowmajor_n) {
    test_chol_half_solve<double>(Side::Left, Layout::RowMajor, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_left_rowmajor_t) {
    test_chol_half_solve<double>(Side::Left, Layout::RowMajor, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_right_colmajor_n) {
    test_chol_half_solve<double>(Side::Right, Layout::ColMajor, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_right_colmajor_t) {
    test_chol_half_solve<double>(Side::Right, Layout::ColMajor, Op::Trans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_right_rowmajor_n) {
    test_chol_half_solve<double>(Side::Right, Layout::RowMajor, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, chol_halfsv_right_rowmajor_t) {
    test_chol_half_solve<double>(Side::Right, Layout::RowMajor, Op::Trans, 10, 8, 8);
}

// ============================================================================
// LUSolver full-solve tests: Side::Left, dense B, ColMajor
// ============================================================================

TEST_F(TestExtSolverLinOpUnified, lu_left_dense_colmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_colmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_colmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_colmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// LUSolver: Side::Left, dense B, RowMajor
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_rowmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_rowmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_rowmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_dense_rowmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10);
}

// LUSolver: Side::Right, dense B, ColMajor
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_colmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_colmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_colmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_colmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// LUSolver: Side::Right, dense B, RowMajor
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_rowmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_rowmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_rowmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_dense_rowmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, false, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8);
}

// LUSolver: Side::Left, sparse B, ColMajor
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_colmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_colmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_colmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_colmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// LUSolver: Side::Left, sparse B, RowMajor
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_rowmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_rowmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_rowmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 10, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_left_sparse_rowmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Left, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 10, 0.3);
}

// LUSolver: Side::Right, sparse B, ColMajor
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_colmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_colmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_colmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_colmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::ColMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// LUSolver: Side::Right, sparse B, RowMajor
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_rowmajor_nn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_rowmajor_nt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::NoTrans, Op::Trans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_rowmajor_tn) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::NoTrans, 10, 8, 8, 0.3);
}
TEST_F(TestExtSolverLinOpUnified, lu_right_sparse_rowmajor_tt) {
    test_solver_linop<LUSolverTag<double>>(Side::Right, true, Layout::RowMajor, Op::Trans, Op::Trans, 10, 8, 8, 0.3);
}

// ============================================================================
// LUSolver half-solve tests
// ============================================================================

TEST_F(TestExtSolverLinOpUnified, lu_halfsv_left_colmajor_n) {
    test_lu_half_solve<double>(Side::Left, Layout::ColMajor, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_left_colmajor_t) {
    test_lu_half_solve<double>(Side::Left, Layout::ColMajor, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_left_rowmajor_n) {
    test_lu_half_solve<double>(Side::Left, Layout::RowMajor, Op::NoTrans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_left_rowmajor_t) {
    test_lu_half_solve<double>(Side::Left, Layout::RowMajor, Op::Trans, 10, 8, 10);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_right_colmajor_n) {
    test_lu_half_solve<double>(Side::Right, Layout::ColMajor, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_right_colmajor_t) {
    test_lu_half_solve<double>(Side::Right, Layout::ColMajor, Op::Trans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_right_rowmajor_n) {
    test_lu_half_solve<double>(Side::Right, Layout::RowMajor, Op::NoTrans, 10, 8, 8);
}
TEST_F(TestExtSolverLinOpUnified, lu_halfsv_right_rowmajor_t) {
    test_lu_half_solve<double>(Side::Right, Layout::RowMajor, Op::Trans, 10, 8, 8);
}


// ============================================================================
// CholSolverLinOp: in-memory (RandBLAS CSR) constructor
// ----------------------------------------------------------------------------
// The cases above build CholSolverLinOp from a Matrix Market file. These build it
// from a CSR tridiagonal matrix and check results by re-applying A as a sparse product.
// ============================================================================

template <typename T>
using InMemCSR = RandBLAS::sparse_data::csr::CSRMatrix<T>;

using RandLAPACK::testing::rel_fro_error;

// AX := A * X for the n x n_rhs ColMajor block X.
template <typename T>
static vector<T> inmem_apply(InMemCSR<T>& A_csr, const T* X, int64_t n, int64_t n_rhs) {
    vector<T> AX(n * n_rhs);
    RandLAPACK::linops::SparseLinOp<InMemCSR<T>> A_op(n, n, A_csr);
    A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, n_rhs, n, (T)1, X, n, (T)0, AX.data(), n);
    return AX;
}

// Solve A X = B through the operator and check A X against B.
template <typename T>
static void inmem_check_solve(InMemCSR<T>& A_csr, RandLAPACK_extras::linops::CholSolverLinOp<T>& A_inv,
                              const vector<T>& B, int64_t n, int64_t n_rhs, T tol) {
    vector<T> X(n * n_rhs);
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, (T)1, B.data(), n, (T)0, X.data(), n);
    vector<T> AX = inmem_apply(A_csr, X.data(), n, n_rhs);
    for (int64_t j = 0; j < n_rhs; ++j)
        ASSERT_LE(rel_fro_error(AX.data() + j * n, B.data() + j * n, n), tol);
}

TEST(TestCholSolverInMemory, spd_tridiag_single_rhs_releases_the_copy) {
    const int64_t n = 8;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);
    EXPECT_GT(A_inv.A_mem.nonZeros(), 0);
    A_inv.factorize();
    EXPECT_EQ(A_inv.A_mem.data().allocatedSize(), 0) << "the in-memory copy must be freed after factorize()";
    vector<double> b(n);
    for (int64_t i = 0; i < n; ++i) b[i] = (double)(i + 1);
    inmem_check_solve<double>(A, A_inv, b, n, 1, 1e-12);
}

TEST(TestCholSolverInMemory, spd_tridiag_multi_rhs_lazy_factorize) {
    const int64_t n = 10, n_rhs = 3;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);  // factorizes on first apply
    vector<double> B(n * n_rhs);
    for (int64_t i = 0; i < n * n_rhs; ++i) B[i] = std::cos((double)i * 0.5);
    inmem_check_solve<double>(A, A_inv, B, n, n_rhs, 1e-12);
}

TEST(TestCholSolverInMemory, spd_tridiag_float) {
    const int64_t n = 10;
    auto A = RandLAPACK::gen::gen_tridiag_csr<float>(n, 2.0f, -1.0f);
    RandLAPACK_extras::linops::CholSolverLinOp<float> A_inv(A);
    vector<float> b(n);
    for (int64_t i = 0; i < n; ++i) b[i] = 1.0f + 0.1f * (float)i;
    inmem_check_solve<float>(A, A_inv, b, n, 1, 1e-4f);
}

// C := alpha A^{-1} B + beta C0, checked independently: A (C - beta C0) / alpha must equal B.
TEST(TestCholSolverInMemory, spd_tridiag_alpha_beta) {
    const int64_t n = 6, n_rhs = 3;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 4.0, -1.0);
    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);
    vector<double> B(n * n_rhs);
    for (int64_t i = 0; i < n * n_rhs; ++i) B[i] = std::sin((double)i * 0.3);
    const double alpha = 2.0, beta = 1.5;
    vector<double> C(n * n_rhs, 7.0);
    A_inv(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, n_rhs, n, alpha, B.data(), n, beta, C.data(), n);
    for (auto& c : C) c = (c - beta * 7.0) / alpha;
    vector<double> AX = inmem_apply(A, C.data(), n, n_rhs);
    ASSERT_LE(rel_fro_error(AX.data(), B.data(), n * n_rhs), 1e-12);
}

// Half solve: M = L^{-1} P satisfies M^T M = A^{-1}, so A M^T (M b) = b.
TEST(TestCholSolverInMemory, spd_tridiag_half_solve_composes_to_inverse) {
    const int64_t n = 9;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    RandLAPACK_extras::linops::CholSolverLinOp<double> M(A, /*half_solve=*/true);
    vector<double> b(n), y(n), x(n);
    for (int64_t i = 0; i < n; ++i) b[i] = 1.0 + 0.25 * (double)i;
    M(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, 1, n, 1.0, b.data(), n, 0.0, y.data(), n);
    M(Side::Left, Layout::ColMajor, Op::Trans,   Op::NoTrans, n, 1, n, 1.0, y.data(), n, 0.0, x.data(), n);
    vector<double> Ax = inmem_apply(A, x.data(), n, 1);
    ASSERT_LE(rel_fro_error(Ax.data(), b.data(), n), 1e-12);
}

// Only the lower triangle is read: garbage in the strict upper triangle must not change
// the solve.
TEST(TestCholSolverInMemory, strict_upper_triangle_is_ignored) {
    const int64_t n = 7;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    auto A_bad = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t p = A_bad.rowptr[i]; p < A_bad.rowptr[i + 1]; ++p)
            if (A_bad.colidxs[p] > i) A_bad.vals[p] = 100.0;
    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A_bad);
    vector<double> b(n);
    for (int64_t i = 0; i < n; ++i) b[i] = 1.0 - 0.1 * (double)i;
    inmem_check_solve<double>(A, A_inv, b, n, 1, 1e-12);
}

// Rows need not list their columns in order. Interior rows stored as {diag, upper, lower} must
// give the solve of the sorted matrix (a reader that stops each row at its first upper entry
// drops the subdiagonal and factors 2 I instead).
TEST(TestCholSolverInMemory, rows_in_any_column_order) {
    const int64_t n = 7;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    auto A_rot = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    for (int64_t i = 1; i + 1 < n; ++i) {   // {lower, diag, upper} -> {diag, upper, lower}
        const int64_t p0 = A_rot.rowptr[i];
        std::rotate(A_rot.colidxs + p0, A_rot.colidxs + p0 + 1, A_rot.colidxs + p0 + 3);
        std::rotate(A_rot.vals + p0, A_rot.vals + p0 + 1, A_rot.vals + p0 + 3);
    }
    ASSERT_EQ(A_rot.colidxs[A_rot.rowptr[1]], 1);
    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A_rot);
    vector<double> b(n);
    for (int64_t i = 0; i < n; ++i) b[i] = 1.0 + 0.2 * (double)i;
    inmem_check_solve<double>(A, A_inv, b, n, 1, 1e-12);
}

// An indefinite matrix must throw a message naming the SPD requirement, and keep throwing
// on a retry (the in-memory copy is released only after a successful factorization).
TEST(TestCholSolverInMemory, indefinite_matrix_throws_on_every_factorize) {
    const int64_t n = 6;
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(n, 0.5, -1.0);   // indefinite
    RandLAPACK_extras::linops::CholSolverLinOp<double> A_inv(A);
    try {
        A_inv.factorize();
        FAIL() << "factorize() accepted an indefinite matrix";
    } catch (const RandLAPACK::Error& e) {
        EXPECT_NE(std::string(e.what()).find("symmetric positive definite"), std::string::npos) << e.what();
    }
    EXPECT_THROW(A_inv.factorize(), RandLAPACK::Error);
    EXPECT_FALSE(A_inv.factorization_done);
    EXPECT_GT(A_inv.A_mem.nonZeros(), 0);
}

TEST(TestCholSolverInMemory, invalid_csr_inputs_are_rejected) {
    auto A = RandLAPACK::gen::gen_tridiag_csr<double>(4, 2.0, -1.0);
    A.index_base = RandBLAS::sparse_data::IndexBase::One;
    EXPECT_THROW(RandLAPACK_extras::linops::CholSolverLinOp<double>{A}, RandLAPACK::Error);
    InMemCSR<double> rect(3, 4);
    try {
        RandLAPACK_extras::linops::CholSolverLinOp<double> bad(rect);
        FAIL() << "a non-square CSR matrix was accepted";
    } catch (const RandLAPACK::Error& e) {
        EXPECT_NE(std::string(e.what()).find("square"), std::string::npos) << e.what();
    }
    InMemCSR<double> empty(4, 4);
    EXPECT_THROW(RandLAPACK_extras::linops::CholSolverLinOp<double>{empty}, RandLAPACK::Error);
}

// A double CSR matrix can build a float operator directly.
TEST(TestCholSolverInMemory, double_source_float_operator) {
    const int64_t n = 10;
    auto A_d = RandLAPACK::gen::gen_tridiag_csr<double>(n, 2.0, -1.0);
    auto A_f = RandLAPACK::gen::gen_tridiag_csr<float>(n, 2.0f, -1.0f);
    RandLAPACK_extras::linops::CholSolverLinOp<float> A_inv(A_d);
    vector<float> b(n);
    for (int64_t i = 0; i < n; ++i) b[i] = 1.0f - 0.05f * (float)i;
    inmem_check_solve<float>(A_f, A_inv, b, n, 1, 1e-4f);
}
