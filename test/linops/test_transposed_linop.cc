// Tests for TransposedOp: implicit transpose view of any LinearOperator.

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

class TestTransposedOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

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

// TransposedOp(dense A) · v should equal A^T · v.
TEST_F(TestTransposedOp, dense_inner) {
    int64_t m = 30, n = 20;
    vector<double> A(m * n);
    RNGState<> state(42);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::TransposedOp<RandLAPACK::linops::DenseLinOp<double>> AT_op(A_op);

    ASSERT_EQ(AT_op.n_rows, n);  // dims swapped
    ASSERT_EQ(AT_op.n_cols, m);

    vector<double> x(m);
    state = RNGState<>(1);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), x.data(), state);

    // Apply: y_TT (n x 1) = A^T · x  (via TransposedOp with NoTrans-self)
    vector<double> y_TT(n, 0.0);
    AT_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, m, 1.0, x.data(), m, 0.0, y_TT.data(), n);

    // Reference: explicit GEMV with Op::Trans on A
    vector<double> y_ref(n, 0.0);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n,
               1.0, A.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_err(y_TT.data(), y_ref.data(), n), 1e-12);
}

// TransposedOp called with Op::Trans should undo the transpose -> base directly.
TEST_F(TestTransposedOp, double_transpose_yields_base) {
    int64_t m = 25, n = 18;
    vector<double> A(m * n);
    RNGState<> state(7);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::TransposedOp<RandLAPACK::linops::DenseLinOp<double>> AT_op(A_op);

    vector<double> x(n);
    state = RNGState<>(2);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    // (A^T)^T · x  = A · x  via TransposedOp called with Op::Trans
    vector<double> y_TT(m, 0.0);
    AT_op(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
          m, 1, n, 1.0, x.data(), n, 0.0, y_TT.data(), m);

    // Reference: A · x
    vector<double> y_ref(m, 0.0);
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n,
               1.0, A.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_err(y_TT.data(), y_ref.data(), m), 1e-12);
}

// TransposedOp(CompositeOperator(D1, D2)) · v should equal (D1·D2)^T · v = D2^T · D1^T · v.
TEST_F(TestTransposedOp, composite_inner) {
    int64_t m = 20, p = 15, n = 12;
    vector<double> D1(m * p), D2(p * n);
    RNGState<> state(13);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, p), D1.data(), state);
    RandBLAS::fill_dense(RandBLAS::DenseDist(p, n), D2.data(), state);

    RandLAPACK::linops::DenseLinOp<double> D1_op(m, p, D1.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> D2_op(p, n, D2.data(), p, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator<RandLAPACK::linops::DenseLinOp<double>,
                                          RandLAPACK::linops::DenseLinOp<double>>
        comp(m, n, D1_op, D2_op);   // m × n composite
    RandLAPACK::linops::TransposedOp<decltype(comp)> comp_T(comp);

    ASSERT_EQ(comp_T.n_rows, n);
    ASSERT_EQ(comp_T.n_cols, m);

    vector<double> x(m);
    state = RNGState<>(3);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), x.data(), state);

    // y = (D1·D2)^T · x  via TransposedOp (NoTrans-self)
    vector<double> y_TT(n, 0.0);
    comp_T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
           n, 1, m, 1.0, x.data(), m, 0.0, y_TT.data(), n);

    // Reference: build dense A = D1·D2, then A^T · x
    vector<double> A_dense(m * n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, p,
               1.0, D1.data(), m, D2.data(), p, 0.0, A_dense.data(), m);
    vector<double> y_ref(n, 0.0);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n,
               1.0, A_dense.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_err(y_TT.data(), y_ref.data(), n), 1e-10);
}

// TransposedOp(TransposedOp(A)) · v should equal A · v (double transpose == identity).
TEST_F(TestTransposedOp, transposed_of_transposed) {
    int64_t m = 22, n = 14;
    vector<double> A(m * n);
    RNGState<> state(99);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::TransposedOp<RandLAPACK::linops::DenseLinOp<double>> AT_op(A_op);
    RandLAPACK::linops::TransposedOp<decltype(AT_op)> ATT_op(AT_op);

    ASSERT_EQ(ATT_op.n_rows, m);
    ASSERT_EQ(ATT_op.n_cols, n);

    vector<double> x(n);
    state = RNGState<>(4);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    // y = (A^T)^T · x = A · x
    vector<double> y_TT(m, 0.0);
    ATT_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
           m, 1, n, 1.0, x.data(), n, 0.0, y_TT.data(), m);

    vector<double> y_ref(m, 0.0);
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n,
               1.0, A.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_err(y_TT.data(), y_ref.data(), m), 1e-12);
}

// TransposedOp around PowerOp: (A^j)^T should equal (A^T)^j.
TEST_F(TestTransposedOp, around_power_op) {
    int64_t n = 16;
    vector<double> A(n * n);
    RNGState<> state(31);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;   // shrink so A^3 doesn't blow up

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerOp<RandLAPACK::linops::DenseLinOp<double>> A3(A_op, 3);
    RandLAPACK::linops::TransposedOp<decltype(A3)> A3_T(A3);

    vector<double> x(n);
    state = RNGState<>(5);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    // y = (A^3)^T · x
    vector<double> y_TT(n, 0.0);
    A3_T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, 1, n, 1.0, x.data(), n, 0.0, y_TT.data(), n);

    // Reference: build A^T explicitly, then apply 3 times
    vector<double> AT(n * n);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < n; ++j)
            AT[i + j * n] = A[j + i * n];
    vector<double> buf(n), out(n);
    std::copy(x.begin(), x.end(), buf.begin());
    for (int it = 0; it < 3; ++it) {
        blas::gemv(Layout::ColMajor, Op::NoTrans, n, n,
                   1.0, AT.data(), n, buf.data(), 1, 0.0, out.data(), 1);
        std::swap(buf, out);
    }
    ASSERT_LE(rel_err(y_TT.data(), buf.data(), n), 1e-10);
}
