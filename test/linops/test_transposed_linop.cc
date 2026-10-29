// Tests for TransposedLinOp: implicit transpose view of any LinearOperator.

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include "../test_helpers.hh"

#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::RNGState;
using RandLAPACK::testing::rel_fro_error;

class TestTransposedLinOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// TransposedLinOp(dense A) · v should equal A^T · v.
TEST_F(TestTransposedLinOp, dense_inner) {
    int64_t m = 30, n = 20;
    vector<double> A(m * n);
    RNGState<> state(42);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::TransposedLinOp<RandLAPACK::linops::DenseLinOp<double>> AT_op(A_op);

    ASSERT_EQ(AT_op.n_rows, n);  // dims swapped
    ASSERT_EQ(AT_op.n_cols, m);

    vector<double> x(m);
    state = RNGState<>(1);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), x.data(), state);

    // Apply: y_T (n x 1) = A^T · x  (via TransposedLinOp with trans_A = NoTrans)
    vector<double> y_T(n, 0.0);
    AT_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, m, 1.0, x.data(), m, 0.0, y_T.data(), n);

    // Reference: explicit GEMV with Op::Trans on A
    vector<double> y_ref(n, 0.0);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n,
               1.0, A.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_fro_error(y_T.data(), y_ref.data(), n), 1e-12);
}

// TransposedLinOp called with Op::Trans should undo the transpose -> base directly.
TEST_F(TestTransposedLinOp, double_transpose_yields_base) {
    int64_t m = 25, n = 18;
    vector<double> A(m * n);
    RNGState<> state(7);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::TransposedLinOp<RandLAPACK::linops::DenseLinOp<double>> AT_op(A_op);

    vector<double> x(n);
    state = RNGState<>(2);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    // (A^T)^T · x  = A · x  via TransposedLinOp called with Op::Trans
    vector<double> y_TT(m, 0.0);
    AT_op(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
          m, 1, n, 1.0, x.data(), n, 0.0, y_TT.data(), m);

    // Reference: A · x
    vector<double> y_ref(m, 0.0);
    blas::gemv(Layout::ColMajor, Op::NoTrans, m, n,
               1.0, A.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_fro_error(y_TT.data(), y_ref.data(), m), 1e-12);
}

// TransposedLinOp(CompositeOperator(D1, D2)) · v should equal (D1·D2)^T · v = D2^T · D1^T · v.
TEST_F(TestTransposedLinOp, composite_inner) {
    int64_t m = 20, p = 15, n = 12;
    vector<double> D1(m * p), D2(p * n);
    RNGState<> state(13);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(m, p), D1.data(), state);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(p, n), D2.data(), state);

    RandLAPACK::linops::DenseLinOp<double> D1_op(m, p, D1.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> D2_op(p, n, D2.data(), p, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator<RandLAPACK::linops::DenseLinOp<double>,
                                          RandLAPACK::linops::DenseLinOp<double>>
        comp(m, n, D1_op, D2_op);   // m × n composite
    RandLAPACK::linops::TransposedLinOp<decltype(comp)> comp_T(comp);

    ASSERT_EQ(comp_T.n_rows, n);
    ASSERT_EQ(comp_T.n_cols, m);

    vector<double> x(m);
    state = RNGState<>(3);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), x.data(), state);

    // y = (D1·D2)^T · x  via TransposedLinOp (trans_A = NoTrans)
    vector<double> y_T(n, 0.0);
    comp_T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
           n, 1, m, 1.0, x.data(), m, 0.0, y_T.data(), n);

    // Reference: build dense A = D1·D2, then A^T · x
    vector<double> A_dense(m * n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, p,
               1.0, D1.data(), m, D2.data(), p, 0.0, A_dense.data(), m);
    vector<double> y_ref(n, 0.0);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n,
               1.0, A_dense.data(), m, x.data(), 1, 0.0, y_ref.data(), 1);

    ASSERT_LE(rel_fro_error(y_T.data(), y_ref.data(), n), 1e-10);
}

// TransposedLinOp(TransposedLinOp(A)) · v should equal A · v (double transpose == identity).
TEST_F(TestTransposedLinOp, transposed_of_transposed) {
    int64_t m = 22, n = 14;
    vector<double> A(m * n);
    RNGState<> state(99);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::TransposedLinOp<RandLAPACK::linops::DenseLinOp<double>> AT_op(A_op);
    RandLAPACK::linops::TransposedLinOp<decltype(AT_op)> ATT_op(AT_op);

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

    ASSERT_LE(rel_fro_error(y_TT.data(), y_ref.data(), m), 1e-12);
}

// TransposedLinOp around PowerLinOp: (A^p)^T should equal (A^T)^p.
TEST_F(TestTransposedLinOp, around_power_op) {
    int64_t n = 16;
    vector<double> A(n * n);
    RNGState<> state(31);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;   // shrink so A^3 doesn't blow up

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A3(A_op, 3);
    RandLAPACK::linops::TransposedLinOp<decltype(A3)> A3_T(A3);

    vector<double> x(n);
    state = RNGState<>(5);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    // y = (A^3)^T · x
    vector<double> y_T(n, 0.0);
    A3_T(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, 1, n, 1.0, x.data(), n, 0.0, y_T.data(), n);

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
    ASSERT_LE(rel_fro_error(y_T.data(), buf.data(), n), 1e-10);
}

// Side::Right dense apply and both SkOp sides must match an explicit A^T operator.
TEST_F(TestTransposedLinOp, side_right_and_skop_match_explicit_transpose) {
    int64_t m = 14, n = 9, d = 5;
    vector<double> A(m * n);
    RNGState<> state(77);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);
    vector<double> At(n * m);
    for (int64_t i = 0; i < m; ++i)
        for (int64_t j = 0; j < n; ++j)
            At[j + i * n] = A[i + j * m];

    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> At_op(n, m, At.data(), n, Layout::ColMajor);
    RandLAPACK::linops::TransposedLinOp<RandLAPACK::linops::DenseLinOp<double>> T_op(A_op);

    // Side::Right: C (d x m) = B (d x n) * A^T.
    vector<double> B(d * n);
    RandBLAS::fill_dense(RandBLAS::DenseDist(d, n), B.data(), state);
    vector<double> C_t(d * m, 0.0), C_ref(d * m, 0.0);
    T_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, n, 1.0, B.data(), d, 0.0, C_t.data(), d);
    At_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, n, 1.0, B.data(), d, 0.0, C_ref.data(), d);
    ASSERT_LE(rel_fro_error(C_t.data(), C_ref.data(), d * m), 1e-12);

    // SkOp, Side::Right: C (d x m) = S (d x n) * A^T.
    RandBLAS::DenseDist DR(d, n);
    RandBLAS::DenseSkOp<double> S_r(DR, RNGState<>(3));
    RandBLAS::DenseSkOp<double> S_r2(DR, RNGState<>(3));
    vector<double> D_t(d * m, 0.0), D_ref(d * m, 0.0);
    T_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, n, 1.0, S_r, 0.0, D_t.data(), d);
    At_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, m, n, 1.0, S_r2, 0.0, D_ref.data(), d);
    ASSERT_LE(rel_fro_error(D_t.data(), D_ref.data(), d * m), 1e-12);

    // SkOp, Side::Left: C (n x d) = A^T * S (m x d).
    RandBLAS::DenseDist DL(m, d);
    RandBLAS::DenseSkOp<double> S_l(DL, RNGState<>(4));
    RandBLAS::DenseSkOp<double> S_l2(DL, RNGState<>(4));
    vector<double> E_t(n * d, 0.0), E_ref(n * d, 0.0);
    T_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, d, m, 1.0, S_l, 0.0, E_t.data(), n);
    At_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, d, m, 1.0, S_l2, 0.0, E_ref.data(), n);
    ASSERT_LE(rel_fro_error(E_t.data(), E_ref.data(), n * d), 1e-12);
}

// A base with only the concept-required 12-arg overload: Side::Left must compile and work,
// and Side::Right must throw instead of failing to compile.
TEST_F(TestTransposedLinOp, concept_only_base) {
    int64_t m = 8, n = 5;
    vector<double> A(m * n);
    RNGState<> state(13);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);
    test_helpers::ConceptOnlyLinOp<double> base{m, n, A.data()};
    RandLAPACK::linops::TransposedLinOp<test_helpers::ConceptOnlyLinOp<double>> T_op(base);

    vector<double> x(m), y(n, 0.0), ref(n, 0.0);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), x.data(), state);
    T_op(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, 1, m, 1.0, x.data(), m, 0.0, y.data(), n);
    blas::gemv(Layout::ColMajor, Op::Trans, m, n, 1.0, A.data(), m, x.data(), 1, 0.0, ref.data(), 1);
    ASSERT_LE(rel_fro_error(y.data(), ref.data(), n), 1e-12);

    vector<double> B(2 * n, 1.0), C(2 * m, 0.0);
    EXPECT_THROW(T_op(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                      2, m, n, 1.0, B.data(), 2, 0.0, C.data(), 2), RandLAPACK::Error);
}

// RowMajor, trans_A = Trans on both sides: op(A^T) = A.
TEST_F(TestTransposedLinOp, rowmajor_trans_both_sides) {
    int64_t m = 7, n = 4, d = 3;
    vector<double> A_cm(m * n);
    RNGState<> state(21);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A_cm.data(), state);
    vector<double> A_rm(m * n);
    RandBLAS::util::omatcopy(m, n, A_cm.data(), 1, m, A_rm.data(), n, 1);
    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A_rm.data(), n, Layout::RowMajor);
    RandLAPACK::linops::TransposedLinOp<RandLAPACK::linops::DenseLinOp<double>> T_op(A_op);

    // Side::Left: C (m x d) = A * B (n x d), RowMajor.
    vector<double> B(n * d), C(m * d, 0.0), C_ref(m * d, 0.0);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, d), B.data(), state);
    T_op(Side::Left, Layout::RowMajor, Op::Trans, Op::NoTrans, m, d, n, 1.0, B.data(), d, 0.0, C.data(), d);
    A_op(Side::Left, Layout::RowMajor, Op::NoTrans, Op::NoTrans, m, d, n, 1.0, B.data(), d, 0.0, C_ref.data(), d);
    ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), m * d), 1e-12);

    // Side::Right: E (d x n) = F (d x m) * A, RowMajor.
    vector<double> F(d * m), E(d * n, 0.0), E_ref(d * n, 0.0);
    RandBLAS::fill_dense(RandBLAS::DenseDist(d, m), F.data(), state);
    T_op(Side::Right, Layout::RowMajor, Op::Trans, Op::NoTrans, d, n, m, 1.0, F.data(), m, 0.0, E.data(), n);
    A_op(Side::Right, Layout::RowMajor, Op::NoTrans, Op::NoTrans, d, n, m, 1.0, F.data(), m, 0.0, E_ref.data(), n);
    ASSERT_LE(rel_fro_error(E.data(), E_ref.data(), d * n), 1e-12);
}
