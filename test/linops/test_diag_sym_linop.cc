#include "RandLAPACK.hh"
#include "rl_blaspp.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace linops = RandLAPACK::linops;
using blas::Layout;

// DiagSymLinOp<T> represents diag(lambda) without ever forming the n x n
// matrix. It must be a drop-in for ExplicitSymLinOp<T> built on a densely
// stored diag(lambda): same concept, both apply overloads (dense right operand
// and RandBLAS sketching operator), same numbers to rounding. Local buffers
// use std::vector (allowed in tests).

static_assert(linops::SymmetricLinearOperator<linops::DiagSymLinOp<double>>);
static_assert(linops::SymmetricLinearOperator<linops::DiagSymLinOp<float>>);

class TestDiagSymLinOp : public ::testing::Test {
protected:
    using RNG = r123::Philox4x32;

    template <typename T>
    static std::vector<T> spectrum(int64_t n) {
        // Loguniform over three decades, deterministic.
        std::vector<T> lam(n);
        const T den = (T)std::max<int64_t>(1, n - 1);
        for (int64_t i = 0; i < n; ++i)
            lam[i] = std::pow((T)10, (T)-3 * (T)i / den);
        return lam;
    }

    template <typename T>
    static std::vector<T> dense_diag(const std::vector<T>& lam) {
        const int64_t n = (int64_t)lam.size();
        std::vector<T> A(n * n, (T)0);
        for (int64_t i = 0; i < n; ++i) A[i + i * n] = lam[i];
        return A;
    }

    template <typename T>
    static std::vector<T> randn(int64_t rows, int64_t cols, uint32_t seed) {
        std::vector<T> M(rows * cols);
        RandBLAS::RNGState<RNG> state(seed);
        RandBLAS::DenseDist D(rows, cols);
        RandBLAS::fill_dense(D, M.data(), state);
        return M;
    }

    template <typename T>
    static T max_abs_diff(const std::vector<T>& a, const std::vector<T>& b) {
        T d = 0;
        for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
        return d;
    }

    template <typename T>
    static T max_abs(const std::vector<T>& a) {
        T d = 0;
        for (T v : a) d = std::max(d, std::abs(v));
        return d;
    }

    template <typename T>
    static constexpr T tol() { return std::is_same_v<T, double> ? (T)1e-13 : (T)1e-5; }

    // Dense overload, C := alpha*A*B + beta*C, strides larger than dim, either
    // layout. With beta == 0 the input C is NaN-filled: a correct operator never
    // reads it, so the result must still be finite and match the reference.
    template <typename T>
    void run_dense(int64_t n, int64_t nb, Layout layout, T alpha, T beta) {
        auto lam = spectrum<T>(n);
        auto A   = dense_diag<T>(lam);
        const int64_t ldb = n + 3, ldc = n + 2;   // >= dim in both layouts (nb <= n)
        auto B = randn<T>(ldb * std::max(n, nb), 1, 21);
        std::vector<T> C0;
        if (beta == (T)0) {
            C0.assign(ldc * std::max(n, nb), std::numeric_limits<T>::quiet_NaN());
        } else {
            C0 = randn<T>(ldc * std::max(n, nb), 1, 22);
        }
        std::vector<T> C_ref = C0, C_op = C0;

        linops::ExplicitSymLinOp<T> ref(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
        linops::DiagSymLinOp<T>     op(n, lam.data());
        ref(layout, nb, alpha, B.data(), ldb, beta, C_ref.data(), ldc);
        op (layout, nb, alpha, B.data(), ldb, beta, C_op.data(),  ldc);

        // Compare only the written region; padding may hold NaN from C0.
        T diff = 0, scale = 0;
        for (int64_t j = 0; j < nb; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                const int64_t idx = (layout == Layout::ColMajor) ? i + j * ldc : j + i * ldc;
                ASSERT_TRUE(std::isfinite(C_op[idx])) << "NaN/Inf at (" << i << "," << j << ")";
                diff  = std::max(diff,  std::abs(C_op[idx] - C_ref[idx]));
                scale = std::max(scale, std::abs(C_ref[idx]));
            }
        }
        EXPECT_LE(diff, tol<T>() * std::max(scale, (T)1))
            << "layout=" << (layout == Layout::ColMajor ? "col" : "row")
            << " alpha=" << alpha << " beta=" << beta;
    }

    // Sparse sketching operator (the SASO the Nyström kernel draws). The same
    // SkOp object is applied through both operators; whichever goes first fills
    // it, so op_first exercises DiagSymLinOp's own fill path.
    template <typename T>
    void run_sparse_skop(int64_t n, int64_t k, int64_t vec_nnz, T alpha, T beta, bool op_first) {
        auto lam = spectrum<T>(n);
        auto A   = dense_diag<T>(lam);
        const int64_t ldc = n + 4;
        auto C0 = randn<T>(ldc * k, 1, 31);
        std::vector<T> C_ref = C0, C_op = C0;

        RandBLAS::RNGState<RNG> state(41);
        RandBLAS::SparseDist DS(n, k, vec_nnz);
        RandBLAS::SparseSkOp<T, RNG> S(DS, state);
        linops::ExplicitSymLinOp<T> ref(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
        linops::DiagSymLinOp<T>     op(n, lam.data());
        if (op_first) {
            op (Layout::ColMajor, k, alpha, S, beta, C_op.data(),  ldc);
            ref(Layout::ColMajor, k, alpha, S, beta, C_ref.data(), ldc);
        } else {
            ref(Layout::ColMajor, k, alpha, S, beta, C_ref.data(), ldc);
            op (Layout::ColMajor, k, alpha, S, beta, C_op.data(),  ldc);
        }
        EXPECT_GT(S.nnz, 0);
        EXPECT_LE(max_abs_diff(C_ref, C_op), tol<T>() * std::max(max_abs(C_ref), (T)1))
            << "vec_nnz=" << vec_nnz << " op_first=" << op_first;
    }

    // Dense sketching operator: DiagSymLinOp must take the same dense branch
    // ExplicitSymLinOp does (fill the buffer, then the dense apply).
    template <typename T>
    void run_dense_skop(int64_t n, int64_t k, T alpha, T beta) {
        auto lam = spectrum<T>(n);
        auto A   = dense_diag<T>(lam);
        const int64_t ldc = n + 1;
        auto C0 = randn<T>(ldc * k, 1, 51);
        std::vector<T> C_ref = C0, C_op = C0;

        RandBLAS::RNGState<RNG> state(61);
        RandBLAS::DenseDist D(n, k);
        RandBLAS::DenseSkOp<T, RNG> S(D, state);
        linops::ExplicitSymLinOp<T> ref(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
        linops::DiagSymLinOp<T>     op(n, lam.data());
        op (Layout::ColMajor, k, alpha, S, beta, C_op.data(),  ldc);
        ref(Layout::ColMajor, k, alpha, S, beta, C_ref.data(), ldc);
        EXPECT_LE(max_abs_diff(C_ref, C_op), tol<T>() * std::max(max_abs(C_ref), (T)1));
    }
};

TEST_F(TestDiagSymLinOp, DenseColMajorGeneralAlphaBeta) {
    run_dense<double>(37, 9, Layout::ColMajor, 1.7, -0.3);
    run_dense<float >(37, 9, Layout::ColMajor, 1.7f, -0.3f);
}

TEST_F(TestDiagSymLinOp, DenseRowMajorGeneralAlphaBeta) {
    run_dense<double>(37, 9, Layout::RowMajor, -2.5, 0.75);
    run_dense<float >(37, 9, Layout::RowMajor, -2.5f, 0.75f);
}

TEST_F(TestDiagSymLinOp, DenseBetaZeroDoesNotReadC) {
    run_dense<double>(40, 6, Layout::ColMajor, 1.0, 0.0);
    run_dense<double>(40, 6, Layout::RowMajor, 0.5, 0.0);
}

TEST_F(TestDiagSymLinOp, DenseSingleColumn) {
    run_dense<double>(25, 1, Layout::ColMajor, 1.0, 1.0);
}

TEST_F(TestDiagSymLinOp, SparseSkOpVecNnz1) {
    run_sparse_skop<double>(60, 12, 1, 1.0, 0.0, false);
    run_sparse_skop<double>(60, 12, 1, 1.0, 0.0, true);
}

TEST_F(TestDiagSymLinOp, SparseSkOpVecNnz8) {
    run_sparse_skop<double>(60, 12, 8, -1.25, 0.4, false);
    run_sparse_skop<double>(60, 12, 8, -1.25, 0.4, true);
    run_sparse_skop<float >(60, 12, 8, -1.25f, 0.4f, true);
}

TEST_F(TestDiagSymLinOp, SparseSkOpVecNnzEqualsK) {
    run_sparse_skop<double>(60, 12, 12, 2.0, 0.0, true);
}

TEST_F(TestDiagSymLinOp, DenseSkOp) {
    run_dense_skop<double>(48, 7, 1.0, 0.0);
    run_dense_skop<double>(48, 7, 0.3, -1.0);
}

TEST_F(TestDiagSymLinOp, ElementAccess) {
    const int64_t n = 8;
    auto lam = spectrum<double>(n);
    linops::DiagSymLinOp<double> op(n, lam.data());
    EXPECT_EQ(op.dim, n);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i)
            EXPECT_DOUBLE_EQ(op(i, j), (i == j) ? lam[i] : 0.0);
}

TEST_F(TestDiagSymLinOp, RejectsShortStrides) {
    const int64_t n = 10;
    auto lam = spectrum<double>(n);
    linops::DiagSymLinOp<double> op(n, lam.data());
    std::vector<double> B(n * n, 1.0), C(n * n, 0.0);
    EXPECT_THROW(op(Layout::ColMajor, 2, 1.0, B.data(), n - 1, 0.0, C.data(), n), std::exception);
    EXPECT_THROW(op(Layout::ColMajor, 2, 1.0, B.data(), n, 0.0, C.data(), n - 1), std::exception);
}

// The opt-in RANDLAPACK_PERF_GEMM switch reads the whole buffer as a general matrix. An
// ExplicitSymLinOp that stores only one triangle (the documented default) must ignore it; only an
// operator whose owner declares both triangles valid may take the gemm path.
class TestExplicitSymLinOpPerfSwitch : public ::testing::Test {
protected:
    static constexpr int64_t n = 64, k = 8;
    // Upper triangle holds a symmetric matrix; the strict lower triangle holds unrelated values.
    static std::vector<double> upper_only() {
        std::vector<double> A(n * n);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i)
                A[i + j * n] = (i <= j) ? 1.0 / (1.0 + i + j) : 1.0e3 + i;
        return A;
    }
    static std::vector<double> apply(linops::ExplicitSymLinOp<double>& op) {
        std::vector<double> B(n * k), C(n * k, 0.0);
        for (int64_t i = 0; i < n * k; ++i) B[i] = std::sin(0.1 * (double)(i + 1));
        op(Layout::ColMajor, k, 1.0, B.data(), n, 0.0, C.data(), n);
        return C;
    }
    void TearDown() override { unsetenv("RANDLAPACK_PERF_GEMM"); }
};

TEST_F(TestExplicitSymLinOpPerfSwitch, GemmSwitchIgnoredForOneTriangleStorage) {
    auto A = upper_only();
    linops::ExplicitSymLinOp<double> op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    unsetenv("RANDLAPACK_PERF_GEMM");
    auto ref = apply(op);
    setenv("RANDLAPACK_PERF_GEMM", "1", 1);
    auto got = apply(op);
    for (int64_t i = 0; i < n * k; ++i) ASSERT_EQ(got[i], ref[i]) << "entry " << i;
}

TEST_F(TestExplicitSymLinOpPerfSwitch, GemmSwitchHonouredWhenBothTrianglesDeclared) {
    auto A = upper_only();   // deliberately inconsistent lower triangle: the gemm path must read it
    linops::ExplicitSymLinOp<double> op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    op.both_triangles = true;
    unsetenv("RANDLAPACK_PERF_GEMM");
    auto ref = apply(op);
    setenv("RANDLAPACK_PERF_GEMM", "1", 1);
    auto got = apply(op);
    double diff = 0;
    for (int64_t i = 0; i < n * k; ++i) diff = std::max(diff, std::abs(got[i] - ref[i]));
    EXPECT_GT(diff, 1.0) << "with both_triangles set, the switch should use gemm on the full buffer";
}
