#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Diagnostic utility to understand Cholesky failures

template <typename T>
void check_spd_properties(int64_t n, const T* A, const char* label) {
    using DenseMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    printf("\n=== Checking SPD properties: %s ===\n", label);

    // Copy to Eigen matrix
    DenseMatrix A_eigen = Eigen::Map<const DenseMatrix>(A, n, n);

    // 1. Check symmetry
    T sym_error = (A_eigen - A_eigen.transpose()).norm() / A_eigen.norm();
    printf("Symmetry error: %.6e\n", sym_error);

    // 2. Compute eigenvalues
    Eigen::SelfAdjointEigenSolver<DenseMatrix> es(A_eigen);
    if (es.info() != Eigen::Success) {
        printf("ERROR: Eigenvalue computation failed!\n");
        return;
    }

    auto eigenvalues = es.eigenvalues();
    T lambda_min = eigenvalues(0);
    T lambda_max = eigenvalues(n-1);
    T actual_cond = lambda_max / lambda_min;

    printf("λ_min = %.6e, λ_max = %.6e\n", lambda_min, lambda_max);
    printf("Condition number: %.6e\n", actual_cond);
    printf("Smallest eigenvalue > 0? %s\n", lambda_min > 0 ? "YES" : "NO");

    // 3. Count negative eigenvalues
    int64_t num_negative = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (eigenvalues(i) < 0) num_negative++;
    }
    printf("Number of negative eigenvalues: %lld / %lld\n", num_negative, n);

    // 4. Try dense Cholesky
    Eigen::LLT<DenseMatrix> chol(A_eigen);
    printf("Dense Cholesky factorization: %s\n",
           chol.info() == Eigen::Success ? "SUCCESS" : "FAILED");

    // 5. Print smallest/largest diagonal elements
    T diag_min = A_eigen(0, 0);
    T diag_max = A_eigen(0, 0);
    for (int64_t i = 0; i < n; ++i) {
        diag_min = std::min(diag_min, A_eigen(i, i));
        diag_max = std::max(diag_max, A_eigen(i, i));
    }
    printf("Diagonal: min=%.6e, max=%.6e\n", diag_min, diag_max);
}

template <typename T>
void test_direct_cholesky(int64_t n, T cond_num) {
    printf("\n========================================\n");
    printf("Test 1: Direct Cholesky on generated SPD matrix\n");
    printf("n=%lld, κ=%.6e\n", n, cond_num);
    printf("========================================\n");

    // Generate SPD matrix
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);
    std::vector<T> A(n * n);
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A.data(), state);

    check_spd_properties(n, A.data(), "Original generated matrix");
}

template <typename T>
void test_sketched_matrix(int64_t n, int64_t m, T cond_num, T d_factor) {
    printf("\n========================================\n");
    printf("Test 2: Matrix after sketching (what CholSolver sees)\n");
    printf("n=%lld, m=%lld, κ=%.6e, d_factor=%.1f\n", n, m, cond_num, d_factor);
    printf("========================================\n");

    // Generate SPD matrix A (n × n)
    auto state = RandBLAS::RNGState<r123::Philox4x32>(42);
    std::vector<T> A(n * n);
    RandLAPACK::gen::gen_spd_mat(n, cond_num, A.data(), state);

    check_spd_properties(n, A.data(), "Original A");

    // Generate random matrix B (n × m) to sketch
    std::vector<T> B(n * m);
    for (int64_t i = 0; i < n * m; ++i) {
        B[i] = static_cast<T>(rand()) / RAND_MAX;
    }

    // Apply SASO sketch: S = Ω^T * A where Ω is d×n
    int64_t d = (int64_t)(d_factor * m);
    RandBLAS::DenseDist D(d, n);
    auto Omega_state = RandBLAS::RNGState<r123::Philox4x32>(1337);
    std::vector<T> Omega(d * n);
    RandBLAS::fill_dense(D, Omega.data(), Omega_state);

    // Compute sketch_A = Ω * A * Ω^T (this is what goes into CholSolver)
    std::vector<T> temp(d * n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               d, n, n, 1.0, Omega.data(), d, A.data(), n, 0.0, temp.data(), d);

    std::vector<T> sketch_A(d * d);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans,
               d, d, n, 1.0, temp.data(), d, Omega.data(), d, 0.0, sketch_A.data(), d);

    check_spd_properties(d, sketch_A.data(), "Sketched matrix Ω*A*Ω^T");
}

template <typename T>
void test_small_high_condition() {
    printf("\n========================================\n");
    printf("Test 3: Small matrix (10×10) with high condition number\n");
    printf("========================================\n");

    // Test sequence: κ = 10^2, 10^4, 10^6, 10^8, 10^10, 10^12, 10^14, 10^16
    std::vector<T> cond_nums = {1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16};

    for (T cond : cond_nums) {
        test_direct_cholesky<T>(10, cond);
    }
}

int main() {
    using T = double;

    printf("=== Cholesky Diagnostic Tool ===\n");
    printf("Investigating why Cholesky fails at κ~10^6 instead of κ~10^16\n\n");

    // Test 1: Small matrix with very high condition numbers
    test_small_high_condition<T>();

    // Test 2: Direct Cholesky on medium-sized matrix
    test_direct_cholesky<T>(100, 1e6);
    test_direct_cholesky<T>(100, 1e8);
    test_direct_cholesky<T>(100, 1e10);

    // Test 3: Check what matrix CholSolver actually sees after sketching
    test_sketched_matrix<T>(100, 50, 1e6, 2.0);
    test_sketched_matrix<T>(100, 50, 1e8, 2.0);

    printf("\n========================================\n");
    printf("Diagnostic complete\n");
    printf("========================================\n");

    return 0;
}
