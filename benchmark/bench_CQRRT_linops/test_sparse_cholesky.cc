#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

// Focused test: Compare dense Cholesky vs sparse Cholesky on identical SPD matrices

template <typename T>
struct cholesky_comparison {
    T cond_num;
    bool dense_success;
    bool sparse_success;
    T dense_time_ms;
    T sparse_time_ms;
};

template <typename T>
cholesky_comparison<T> test_both_cholesky(int64_t n, T cond_num) {
    cholesky_comparison<T> result;
    result.cond_num = cond_num;

    // Generate SPD matrix
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);
    std::vector<T> A_data(n * n);
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A_data.data(), state);

    // Convert to dense Eigen matrix
    using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    DenseMatrix A_dense = Eigen::Map<DenseMatrix>(A_data.data(), n, n);

    // Convert to sparse Eigen matrix (even though it's dense, we want to test sparse solver)
    Eigen::SparseMatrix<T, Eigen::ColMajor> A_sparse = A_dense.sparseView();
    A_sparse.makeCompressed();

    // Test 1: Dense Cholesky (Eigen::LLT)
    auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::LLT<DenseMatrix> dense_chol(A_dense);
    auto t2 = std::chrono::high_resolution_clock::now();
    result.dense_success = (dense_chol.info() == Eigen::Success);
    result.dense_time_ms = std::chrono::duration<T, std::milli>(t2 - t1).count();

    // Test 2: Sparse Cholesky with NaturalOrdering (what CholSolver uses)
    auto t3 = std::chrono::high_resolution_clock::now();
    Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower, Eigen::NaturalOrdering<int>> sparse_chol;
    sparse_chol.compute(A_sparse);
    auto t4 = std::chrono::high_resolution_clock::now();
    result.sparse_success = (sparse_chol.info() == Eigen::Success);
    result.sparse_time_ms = std::chrono::duration<T, std::milli>(t4 - t3).count();

    return result;
}

int main() {
    using T = double;

    printf("========================================\n");
    printf("Sparse vs Dense Cholesky Comparison\n");
    printf("Testing identical SPD matrices\n");
    printf("========================================\n\n");

    printf("Matrix size: 1138 × 1138 (matching CQRRT benchmark)\n\n");

    printf("%-15s %-15s %-15s %-15s %-15s\n",
           "Cond Number", "Dense Success", "Sparse Success", "Dense Time(ms)", "Sparse Time(ms)");
    printf("%-15s %-15s %-15s %-15s %-15s\n",
           "===============", "===============", "===============", "===============", "===============");

    // Test sequence matching the conditioning study
    std::vector<T> cond_nums = {
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6,
        2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7,
        1e8, 1e10, 1e12
    };

    for (T cond : cond_nums) {
        auto result = test_both_cholesky<T>(1138, cond);

        printf("%-15.2e %-15s %-15s %-15.3f %-15.3f",
               result.cond_num,
               result.dense_success ? "YES" : "NO",
               result.sparse_success ? "YES" : "NO",
               result.dense_time_ms,
               result.sparse_time_ms);

        // Highlight failures
        if (result.dense_success && !result.sparse_success) {
            printf(" <-- SPARSE FAILS BUT DENSE WORKS!");
        } else if (!result.dense_success && !result.sparse_success) {
            printf(" <-- BOTH FAIL");
        } else if (!result.dense_success && result.sparse_success) {
            printf(" <-- DENSE FAILS BUT SPARSE WORKS (unexpected!)");
        }

        printf("\n");

        // Stop testing once both fail
        if (!result.dense_success && !result.sparse_success) {
            printf("\nBoth solvers failed - stopping tests.\n");
            break;
        }
    }

    printf("\n========================================\n");
    printf("Small matrix test (10×10)\n");
    printf("========================================\n\n");

    printf("%-15s %-15s %-15s\n", "Cond Number", "Dense Success", "Sparse Success");
    printf("%-15s %-15s %-15s\n", "===============", "===============", "===============");

    std::vector<T> small_conds = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16};

    for (T cond : small_conds) {
        auto result = test_both_cholesky<T>(10, cond);
        printf("%-15.2e %-15s %-15s\n",
               result.cond_num,
               result.dense_success ? "YES" : "NO",
               result.sparse_success ? "YES" : "NO");
    }

    printf("\n========================================\n");
    printf("Test complete\n");
    printf("========================================\n");

    return 0;
}
