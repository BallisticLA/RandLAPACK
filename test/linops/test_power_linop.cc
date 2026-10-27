// Tests for PowerLinOp: implicit p-th power of a square LinearOperator.

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
using RandLAPACK::testing::rel_fro_error;

class TestPowerLinOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    // Reference apply: y = A^p * x via dense GEMV iteration.
    template <typename T>
    void apply_power_reference(const T* A, int64_t n, int p,
                               const T* x_in, T* x_out)
    {
        vector<T> buf_a(n), buf_b(n);
        std::copy(x_in, x_in + n, buf_a.data());
        T* in = buf_a.data();
        T* out = buf_b.data();
        for (int it = 0; it < p; ++it) {
            blas::gemv(Layout::ColMajor, Op::NoTrans, n, n,
                       (T)1.0, A, n, in, 1, (T)0.0, out, 1);
            std::swap(in, out);
        }
        std::copy(in, in + n, x_out);
    }
};

// PowerLinOp(A, 1) · v should equal a single base apply.
TEST_F(TestPowerLinOp, power_1_single_apply) {
    int64_t n = 50;
    vector<double> A(n * n);
    RNGState<> state(42);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 1);

    vector<double> x(n);
    state = RNGState<>(1);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    vector<double> y_ref(n);
    apply_power_reference(A.data(), n, 1, x.data(), y_ref.data());

    ASSERT_LE(rel_fro_error(y_pow.data(), y_ref.data(), n), 1e-12);
}

// PowerLinOp(A, 3) · v should equal A·A·A · v (ping-pong path).
TEST_F(TestPowerLinOp, power_3_dense_vector) {
    int64_t n = 30;
    vector<double> A(n * n);
    RNGState<> state(7);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    // Scale so A^3 doesn't explode (spectral radius < 1)
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 3);

    vector<double> x(n);
    state = RNGState<>(2);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    vector<double> y_ref(n);
    apply_power_reference(A.data(), n, 3, x.data(), y_ref.data());

    ASSERT_LE(rel_fro_error(y_pow.data(), y_ref.data(), n), 1e-10);
}

// PowerLinOp(A, 2) on a multi-column B (Side::Left path with n > 1).
TEST_F(TestPowerLinOp, multi_rhs) {
    int64_t n = 20, k_cols = 5;
    vector<double> A(n * n);
    RNGState<> state(13);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

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
    ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), n * k_cols), 1e-10);
}

// PowerLinOp(A, 2) with Op::Trans should compute (A^T)^2 · v.
TEST_F(TestPowerLinOp, transpose_dispatch) {
    int64_t n = 25;
    vector<double> A(n * n);
    RNGState<> state(21);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    vector<double> x(n);
    state = RNGState<>(4);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);

    vector<double> y_pow(n, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
          n, 1, n, 1.0, x.data(), n, 0.0, y_pow.data(), n);

    // Reference: explicit A^T, then apply twice
    vector<double> AT(n * n);
    RandBLAS::util::omatcopy(n, n, A.data(), n, 1, AT.data(), 1, n);
    vector<double> y_ref(n);
    apply_power_reference(AT.data(), n, 2, x.data(), y_ref.data());

    ASSERT_LE(rel_fro_error(y_pow.data(), y_ref.data(), n), 1e-10);
}

// alpha/beta path: C := alpha · A^2 · B + beta · C
TEST_F(TestPowerLinOp, alpha_beta_respected) {
    int64_t n = 15;
    vector<double> A(n * n);
    RNGState<> state(99);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

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

    ASSERT_LE(rel_fro_error(y.data(), y_ref.data(), n), 1e-10);
}

// PowerLinOp over a CompositeOperator: PowerLinOp(D1 * D2, 2) equals (D1 * D2) * (D1 * D2).
TEST_F(TestPowerLinOp, composite_inner) {
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
    RandLAPACK::linops::PowerLinOp<decltype(comp)> comp_pow(comp, 2);

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

    ASSERT_LE(rel_fro_error(y_pow.data(), y_ref.data(), n), 1e-9);
}

// SkOp overload: applying A^2 to a DenseSkOp must match applying A^2 to that
// same operator's explicit dense materialization (same seed, same distribution).
TEST_F(TestPowerLinOp, skop_matches_manual_materialization_colmajor) {
    int64_t n = 12, d = 4;
    vector<double> A(n * n);
    RNGState<> state(55);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    RandBLAS::DenseDist D(n, d);
    RNGState<> skop_state(9);
    RandBLAS::DenseSkOp<double> S(D, skop_state);

    vector<double> C_sk(n * d, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, d, n, 1.0, S, 0.0, C_sk.data(), n);

    // fill_dense writes D in its natural layout (ColMajor for a tall Long-axis DenseDist),
    // matching the ColMajor call.
    vector<double> S_dense(n * d);
    RNGState<> skop_state2(9);
    RandBLAS::fill_dense(D, S_dense.data(), skop_state2);

    vector<double> C_manual(n * d, 0.0);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
          n, d, n, 1.0, S_dense.data(), n, 0.0, C_manual.data(), n);

    ASSERT_LE(rel_fro_error(C_sk.data(), C_manual.data(), n * d), 1e-12);
}

// SkOp overload under RowMajor: two DenseLinOps storing the identical abstract
// matrix (one ColMajor, one RowMajor) must produce the identical abstract result
// when hit with the same DenseSkOp (same seed) under each op's own layout.
TEST_F(TestPowerLinOp, skop_rowmajor_matches_colmajor) {
    int64_t n = 12, d = 4;
    vector<double> A_cm(n * n);
    RNGState<> state(56);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A_cm.data(), state);
    for (auto& v : A_cm) v *= 0.1;

    // Same abstract matrix, stored RowMajor.
    vector<double> A_rm(n * n);
    RandBLAS::util::omatcopy(n, n, A_cm.data(), 1, n, A_rm.data(), n, 1);

    RandLAPACK::linops::DenseLinOp<double> A_op_cm(n, n, A_cm.data(), n, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> A_op_rm(n, n, A_rm.data(), n, Layout::RowMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow_cm(A_op_cm, 2);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow_rm(A_op_rm, 2);

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

    vector<double> C_row_as_col(n * d);
    RandBLAS::util::omatcopy(n, d, C_row.data(), d, 1, C_row_as_col.data(), 1, n);
    ASSERT_LE(rel_fro_error(C_row_as_col.data(), C_col.data(), n * d), 1e-12);
}

// SkOp overload with a SparseSkOp, in both layouts: the result must match A^2 applied to
// the explicit S, built independently by sketching the identity.
TEST_F(TestPowerLinOp, skop_sparse_matches_explicit_both_layouts) {
    int64_t n = 10, d = 3;
    vector<double> A_cm(n * n);
    RNGState<> state(64);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A_cm.data(), state);
    for (auto& v : A_cm) v *= 0.1;
    vector<double> A_rm(n * n);
    RandBLAS::util::omatcopy(n, n, A_cm.data(), 1, n, A_rm.data(), n, 1);

    RandBLAS::SparseDist SDist(n, d, 2, RandBLAS::Axis::Short);
    RNGState<> skop_state(11);
    RandBLAS::SparseSkOp<double> S(SDist, skop_state);

    // Explicit S (n x d, ColMajor) = S * I_d.
    vector<double> eye(d * d, 0.0);
    for (int64_t i = 0; i < d; ++i) eye[i + i * d] = 1.0;
    vector<double> S_explicit(n * d, 0.0);
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, d, d,
                             1.0, S, eye.data(), d, 0.0, S_explicit.data(), n);
    vector<double> C_ref(n * d, 0.0);
    vector<double> tmp(n * d, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, d, n, 1.0, A_cm.data(), n,
               S_explicit.data(), n, 0.0, tmp.data(), n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, d, n, 1.0, A_cm.data(), n,
               tmp.data(), n, 0.0, C_ref.data(), n);

    RandLAPACK::linops::DenseLinOp<double> A_op_cm(n, n, A_cm.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow_cm(A_op_cm, 2);
    vector<double> C_col(n * d, 0.0);
    A_pow_cm(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
             n, d, n, 1.0, S, 0.0, C_col.data(), n);
    ASSERT_LE(rel_fro_error(C_col.data(), C_ref.data(), n * d), 1e-12);

    RandLAPACK::linops::DenseLinOp<double> A_op_rm(n, n, A_rm.data(), n, Layout::RowMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow_rm(A_op_rm, 2);
    vector<double> C_row(n * d, 0.0);
    A_pow_rm(Side::Left, Layout::RowMajor, Op::NoTrans, Op::NoTrans,
             n, d, n, 1.0, S, 0.0, C_row.data(), d);
    vector<double> C_row_as_col(n * d);
    RandBLAS::util::omatcopy(n, d, C_row.data(), d, 1, C_row_as_col.data(), 1, n);
    ASSERT_LE(rel_fro_error(C_row_as_col.data(), C_ref.data(), n * d), 1e-12);
}

// RowMajor, p = 3, trans_B = Trans, padded ldb/ldc, alpha and beta, against an explicit
// reference C_ref = alpha * A^3 * B^T + beta * C0 computed ColMajor.
TEST_F(TestPowerLinOp, rowmajor_power3_transB_padded_alpha_beta) {
    int64_t n = 9, nrhs = 4, ldb = n + 3, ldc = nrhs + 2;   // RowMajor: ld >= #cols
    double alpha = 0.7, beta = -1.3;
    vector<double> A_cm(n * n);
    RNGState<> state(81);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A_cm.data(), state);
    for (auto& v : A_cm) v *= 0.2;
    vector<double> A_rm(n * n);
    RandBLAS::util::omatcopy(n, n, A_cm.data(), 1, n, A_rm.data(), n, 1);

    // B is nrhs x n (op(B) = B^T is n x nrhs), RowMajor with padding.
    vector<double> B(nrhs * ldb, 99.0);
    vector<double> C(n * ldc, 99.0);
    vector<double> Bt_cm(n * nrhs), C0_cm(n * nrhs);
    for (int64_t r = 0; r < nrhs; ++r)
        for (int64_t c = 0; c < n; ++c) {
            B[r * ldb + c] = std::sin(0.3 * (double)(r * n + c));
            Bt_cm[c + r * n] = B[r * ldb + c];
        }
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < nrhs; ++j) {
            C[i * ldc + j] = std::cos(0.2 * (double)(i * nrhs + j));
            C0_cm[i + j * n] = C[i * ldc + j];
        }

    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A_rm.data(), n, Layout::RowMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 3);
    A_pow(Side::Left, Layout::RowMajor, Op::NoTrans, Op::Trans,
          n, nrhs, n, alpha, B.data(), ldb, beta, C.data(), ldc);

    vector<double> ref(n * nrhs);
    for (int64_t j = 0; j < nrhs; ++j)
        apply_power_reference(A_cm.data(), n, 3, Bt_cm.data() + j * n, ref.data() + j * n);
    for (int64_t t = 0; t < n * nrhs; ++t) ref[t] = alpha * ref[t] + beta * C0_cm[t];
    vector<double> got(n * nrhs);
    RandBLAS::util::omatcopy(n, nrhs, C.data(), ldc, 1, got.data(), 1, n);
    ASSERT_LE(rel_fro_error(got.data(), ref.data(), n * nrhs), 1e-12);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = nrhs; j < ldc; ++j)
            ASSERT_EQ(C[i * ldc + j], 99.0) << "padding of C must be untouched";
}

// SkOp overload with trans_S = Trans and alpha/beta: S is d x n, op(S) = S^T is n x d.
TEST_F(TestPowerLinOp, skop_transS_alpha_beta_matches_dense) {
    int64_t n = 10, d = 4;
    double alpha = 1.5, beta = 0.5;
    vector<double> A(n * n);
    RNGState<> state(91);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.1;
    RandLAPACK::linops::DenseLinOp<double> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>> A_pow(A_op, 2);

    RandBLAS::DenseDist D(d, n);
    RandBLAS::DenseSkOp<double> S(D, RNGState<>(12));
    RandBLAS::fill_dense(S);
    vector<double> S_cm(d * n);   // S explicitly, ColMajor d x n
    for (int64_t i = 0; i < d; ++i)
        for (int64_t j = 0; j < n; ++j)
            S_cm[i + j * d] = (S.layout == Layout::ColMajor) ? S.buff[i + j * S.dist.dim_major]
                                                             : S.buff[j + i * S.dist.dim_major];
    vector<double> C0(n * d);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, d), C0.data(), state);

    vector<double> C_sk = C0, C_ref = C0;
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::Trans, n, d, n, alpha, S, beta, C_sk.data(), n);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::Trans, n, d, n, alpha, S_cm.data(), d, beta, C_ref.data(), n);
    ASSERT_LE(rel_fro_error(C_sk.data(), C_ref.data(), n * d), 1e-12);
}

TEST_F(TestPowerLinOp, invalid_arguments_throw) {
    int64_t n = 5;
    vector<double> A(n * (n + 1), 1.0);
    RandLAPACK::linops::DenseLinOp<double> sq(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> rect(n, n + 1, A.data(), n, Layout::ColMajor);
    using P = RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<double>>;
    EXPECT_THROW(P(rect, 2), RandLAPACK::Error);
    EXPECT_THROW(P(sq, 0), RandLAPACK::Error);
    P A_pow(sq, 2);
    vector<double> B(n * n, 1.0), C(n * n, 0.0);
    EXPECT_THROW(A_pow(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, n, n, 1.0, B.data(), n, 0.0, C.data(), n), RandLAPACK::Error);
    EXPECT_THROW(A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n - 1, n, n, 1.0, B.data(), n, 0.0, C.data(), n), RandLAPACK::Error);
    EXPECT_THROW(A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, n, n - 1, 1.0, B.data(), n, 0.0, C.data(), n), RandLAPACK::Error);
    RandBLAS::DenseSkOp<double> S(RandBLAS::DenseDist(n + 1, 2), RNGState<>(1));
    EXPECT_THROW(A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, 2, n, 1.0, S, 0.0, C.data(), n), RandLAPACK::Error);
    RandBLAS::DenseSkOp<double> S_ok(RandBLAS::DenseDist(n, 2), RNGState<>(1));
    EXPECT_THROW(A_pow(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, 2, n, 1.0, S_ok, 0.0, C.data(), n), RandLAPACK::Error);
    EXPECT_THROW(A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                       n, 2, n - 1, 1.0, S_ok, 0.0, C.data(), n), RandLAPACK::Error);
}

TEST_F(TestPowerLinOp, float_power2) {
    int64_t n = 12;
    vector<float> A(n * n);
    RNGState<> state(5);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, n), A.data(), state);
    for (auto& v : A) v *= 0.2f;
    vector<float> x(n), y(n, 0.0f), ref(n);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), x.data(), state);
    RandLAPACK::linops::DenseLinOp<float> A_op(n, n, A.data(), n, Layout::ColMajor);
    RandLAPACK::linops::PowerLinOp<RandLAPACK::linops::DenseLinOp<float>> A_pow(A_op, 2);
    A_pow(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, 1, n, 1.0f, x.data(), n, 0.0f, y.data(), n);
    apply_power_reference(A.data(), n, 2, x.data(), ref.data());
    ASSERT_LE(rel_fro_error(y.data(), ref.data(), n), 1e-5);
}
