// Tests for PowerOp: implicit j-th power of a square LinearOperator.

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::RNGState;

class TestPowerOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    // Reference apply: y = A^j * x via dense GEMV iteration.
    template <typename T>
    void apply_power_reference(const T* A, int64_t n, int j,
                               const T* x_in, T* x_out)
    {
        vector<T> buf_a(n), buf_b(n);
        std::copy(x_in, x_in + n, buf_a.data());
        T* in = buf_a.data();
        T* out = buf_b.data();
        for (int it = 0; it < j; ++it) {
            blas::gemv(Layout::ColMajor, Op::NoTrans, n, n,
                       (T)1.0, A, n, in, 1, (T)0.0, out, 1);
            std::swap(in, out);
        }
        std::copy(in, in + n, x_out);
    }

    template <typename T>
    double rel_err(const T* a, const T* b, int64_t n) {
        double err = 0.0, ref = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double d = (double)a[i] - (double)b[i];
            err += d * d;
            ref += (double)b[i] * (double)b[i];
        }
        return (ref > 0) ? std::sqrt(err / ref) : std::sqrt(err);
    }
};

// PowerOp(A, 1) · v should equal a single base apply.
TEST_F(TestPowerOp, j_equals_1_single_apply) {
    int64_t n = 50;
    vector<double> A(n * n);
    RNGState<> state(42);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 1);

    vector<double> x(n);
    state = RNGState<>(1);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    vector<double> y_ref(n);
    apply_power_reference(A.data(), n, 1, x.data(), y_ref.data());

    ASSERT_LE(rel_err(y_pow.data(), y_ref.data(), n), 1e-12);
}

// PowerOp(A, 3) · v should equal A·A·A · v (ping-pong path).
TEST_F(TestPowerOp, j_equals_3_dense_vector) {
    int64_t n = 30;
    vector<double> A(n * n);
    RNGState<> state(7);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    // Scale so A^3 doesn't explode (spectral radius < 1)
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 3);

    vector<double> x(n);
    state = RNGState<>(2);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    vector<double> y_ref(n);
    apply_power_reference(A.data(), n, 3, x.data(), y_ref.data());

    ASSERT_LE(rel_err(y_pow.data(), y_ref.data(), n), 1e-10);
}

// PowerOp(A, 2) on a multi-column B (Side::Left path with n > 1).
TEST_F(TestPowerOp, multi_rhs) {
    int64_t n = 20, k_cols = 5;
    vector<double> A(n * n);
    RNGState<> state(13);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    vector<double> B(n * k_cols);
    state = RNGState<>(3);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, k_cols), B.data(), state);

    vector<double> C(n * k_cols, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, k_cols, n, 1.0, B.data(), n, 0.0, C.data(), n);

    // Reference: per-column apply
    vector<double> C_ref(n * k_cols);
    for (int64_t c = 0; c < k_cols; ++c) {
        apply_power_reference(A.data(), n, 2,
                              B.data() + c * n, C_ref.data() + c * n);
    }
    ASSERT_LE(rel_err(C.data(), C_ref.data(), n * k_cols), 1e-10);
}

// PowerOp(A, 2) with Op::Trans should compute (A^T)^2 · v.
TEST_F(TestPowerOp, transpose_dispatch) {
    int64_t n = 25;
    vector<double> A(n * n);
    RNGState<> state(21);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    vector<double> x(n);
    state = RNGState<>(4);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
          n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    // Reference: explicit A^T, then apply twice
    vector<double> AT(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            AT[i + j * n] = A[j + i * n];
    vector<double> y_ref(n);
    apply_power_reference(AT.data(), n, 2, x.data(), y_ref.data());

    ASSERT_LE(rel_err(y_pow.data(), y_ref.data(), n), 1e-10);
}

// alpha/beta path: C := alpha · A^2 · B + beta · C
TEST_F(TestPowerOp, alpha_beta_respected) {
    int64_t n = 15;
    vector<double> A(n * n);
    RNGState<> state(99);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    vector<double> x(n);
    state = RNGState<>(5);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    const double alpha = 2.5, beta = 1.5;
    vector<double> y(n, 7.0);   // initial C
    vector<double> y_initial = y;

    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, alpha, x.data(), n, beta, y.data(), n);

    // Reference: alpha · A^2 · x + beta · y_initial
    vector<double> A2x(n);
    apply_power_reference(A.data(), n, 2, x.data(), A2x.data());
    vector<double> y_ref(n);
    for (int64_t i = 0; i < n; ++i)
        y_ref[i] = alpha * A2x[i] + beta * y_initial[i];

    ASSERT_LE(rel_err(y.data(), y_ref.data(), n), 1e-10);
}

// PowerOp wrapping a CompositeOperator: matches the rspec usage pattern
// (C = L^T · X^{-1} · L raised to a power).  Here we use a simpler composite
// of two dense ops: PowerOp(D1 · D2, 2) should equal (D1·D2)·(D1·D2).
TEST_F(TestPowerOp, composite_inner) {
    int64_t n = 18;
    vector<double> D1(n * n), D2(n * n);
    RNGState<> state(77);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), D1.data(), state);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), D2.data(), state);
    for (auto& v : D1) v *= 0.1;
    for (auto& v : D2) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> D1_op(n, n, D1.data(), n, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> D2_op(n, n, D2.data(), n, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator<RandLAPACK::linops::DenseLinOp<double>,
                                          RandLAPACK::linops::DenseLinOp<double>>
        comp(n, n, D1_op, D2_op);
    RandLAPACK::linops::PowerOp<decltype(comp)> comp_pow(comp, 2);

    vector<double> x(n);
    state = RNGState<>(6);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    comp_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
             n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    // Reference: build A_dense = D1 * D2, then A_dense^2 · x
    vector<double> A_dense(n * n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n,
               1.0, D1.data(), n, D2.data(), n, 0.0, A_dense.data(), n);
    vector<double> y_ref(n);
    apply_power_reference(A_dense.data(), n, 2, x.data(), y_ref.data());

    ASSERT_LE(rel_err(y_pow.data(), y_ref.data(), n), 1e-9);
}

// SkOp overload: applying A^2 to a DenseSkOp must match applying A^2 to that
// same operator's explicit dense materialization (same seed, same distribution).
// Regression test for the bug where the SkOp overload always materialized S in
// ColMajor but reported a RowMajor leading dimension to RowMajor callers.
TEST_F(TestPowerOp, skop_matches_manual_materialization_colmajor) {
    int64_t n = 12, d = 4;
    vector<double> A(n * n);
    RNGState<> state(55);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    RandBLAS::DenseDist D(n, d);
    RNGState<> skop_state(9);
    RandBLAS::DenseSkOp<double> S(D, skop_state);

    vector<double> C_sk(n * d, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, d, n, 1.0, S, 0.0, C_sk.data(), n);

    // D's natural layout is ColMajor here (tall, major_axis == Long by default),
    // so this manual materialization has the same memory layout as S_dense above.
    vector<double> S_dense(n * d);
    RNGState<> skop_state2(9);
    RandBLAS::fill_dense(D, S_dense.data(), skop_state2);

    vector<double> C_manual(n * d, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, d, n, 1.0, S_dense.data(), n, 0.0, C_manual.data(), n);

    ASSERT_LE(rel_err(C_sk.data(), C_manual.data(), n * d), 1e-12);
}

// SkOp overload under RowMajor: two DenseLinOps storing the identical abstract
// matrix (one ColMajor, one RowMajor) must produce the identical abstract result
// when hit with the same DenseSkOp (same seed) under each op's own layout.
// Before the fix this failed because S was always materialized ColMajor while the
// RowMajor call read it back with a RowMajor leading dimension (transposed S).
TEST_F(TestPowerOp, skop_rowmajor_matches_colmajor) {
    int64_t n = 12, d = 4;
    vector<double> A_cm(n * n);
    RNGState<> state(56);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A_cm.data(), state);
    for (auto& v : A_cm) v *= 0.1;

    // Same abstract matrix, stored RowMajor.
    vector<double> A_rm(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            A_rm[i * n + j] = A_cm[i + j * n];

    RandLAPACK::linops::DenseLinOp<double> A_op_cm(n, n, A_cm.data(), n, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> A_op_rm(n, n, A_rm.data(), n, Layout::RowMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow_cm(A_op_cm, 2);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow_rm(A_op_rm, 2);

    RandBLAS::DenseDist D(n, d);
    RNGState<> skop_state_cm(10);
    RandBLAS::DenseSkOp<double> S_cm(D, skop_state_cm);
    RNGState<> skop_state_rm(10);
    RandBLAS::DenseSkOp<double> S_rm(D, skop_state_rm);

    vector<double> C_col(n * d, 0.0);
    A_pow_cm(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
             n, d, n, 1.0, S_cm, 0.0, C_col.data(), n);

    vector<double> C_row(n * d, 0.0);
    A_pow_rm(Side::Left, Layout::RowMajor, Op::NoTrans, Op::NoTrans,
             n, d, n, 1.0, S_rm, 0.0, C_row.data(), d);

    double err = 0.0, ref = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < d; ++j) {
            double diff = C_col[i + j * n] - C_row[j + i * d];
            err += diff * diff;
            ref += C_col[i + j * n] * C_col[i + j * n];
        }
    }
    ASSERT_LE(std::sqrt(err / ref), 1e-12);
}

// SkOp overload: a SparseSkOp under RowMajor must throw. The overload only knows
// how to materialize a SparseSkOp in ColMajor (see rl_power_linop.hh); RowMajor
// callers must be rejected rather than silently handed a transposed S.
TEST_F(TestPowerOp, skop_sparse_rowmajor_throws) {
    int64_t n = 10, d = 3;
    vector<double> A(n * n);
    RNGState<> state(64);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    RandBLAS::SparseDist SDist(n, d, 2, RandBLAS::Axis::Short);
    RNGState<> skop_state(11);
    RandBLAS::SparseSkOp<double> S(SDist, skop_state);
    RandBLAS::fill_sparse(S);

    vector<double> C(n * d, 0.0);
    ASSERT_THROW(
        A_pow(Side::Left, Layout::RowMajor, Op::NoTrans, Op::NoTrans,
              n, d, n, 1.0, S, 0.0, C.data(), d),
        RandLAPACK::Error);
}
