// Tests for VStackOp (vertical concatenation [Top; Bot]) and ScaledIdentityOp
// (matrix-free mu*I), including the headline use: the mu-regularized augmented
// operator A_hat = [A; mu*I] whose Cholesky-QR factor R satisfies
// R^T R = A^T A + mu^2 I.

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::RNGState;

namespace rl = RandLAPACK::linops;

class TestVStackOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    template <typename T>
    static double rel_err(const T* a, const T* b, int64_t n) {
        double err = 0.0, ref = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            double d = (double)a[i] - (double)b[i];
            err += d * d;
            ref += (double)b[i] * (double)b[i];
        }
        return (ref > 0) ? std::sqrt(err / ref) : std::sqrt(err);
    }
};

// mu*I applied to a block equals mu * that block.
TEST_F(TestVStackOp, scaled_identity_applies_mu) {
    int64_t n = 12, k = 5;
    double mu = 0.37;
    vector<double> X(n * k);
    RNGState<> state(42);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, k), X.data(), state);

    rl::ScaledIdentityOp<double> I_op(n, mu);
    ASSERT_EQ(I_op.n_rows, n);
    ASSERT_EQ(I_op.n_cols, n);

    vector<double> Y(n * k, 0.0);
    I_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, k, n, 1.0, X.data(), n, 0.0, Y.data(), n);

    vector<double> ref(n * k);
    for (int64_t i = 0; i < n * k; ++i) ref[i] = mu * X[i];
    ASSERT_LE(rel_err(Y.data(), ref.data(), n * k), 1e-14);

    // beta != 0 accumulation: Y := 2*Y + alpha*mu*X.
    vector<double> Y2 = Y;
    I_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, k, n, 1.0, X.data(), n, 2.0, Y2.data(), n);
    vector<double> ref2(n * k);
    for (int64_t i = 0; i < n * k; ++i) ref2[i] = 2.0 * Y[i] + mu * X[i];
    ASSERT_LE(rel_err(Y2.data(), ref2.data(), n * k), 1e-14);
}

// [Top; Bot] * X = [Top*X; Bot*X], stacked row-wise.
TEST_F(TestVStackOp, notrans_matches_stack) {
    int64_t mt = 17, mb = 9, n = 8, k = 6;   // Top: mt x n, Bot: mb x n
    vector<double> Tm(mt * n), Bm(mb * n), X(n * k);
    RNGState<> state(7);
    RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), Tm.data(), state);
    RandBLAS::fill_dense(RandBLAS::DenseDist(mb, n), Bm.data(), state);
    RandBLAS::fill_dense(RandBLAS::DenseDist(n, k), X.data(), state);

    rl::DenseLinOp<double> Top(mt, n, Tm.data(), mt, Layout::ColMajor);
    rl::DenseLinOp<double> Bot(mb, n, Bm.data(), mb, Layout::ColMajor);
    rl::VStackOp<rl::DenseLinOp<double>, rl::DenseLinOp<double>> S(Top, Bot);

    ASSERT_EQ(S.n_rows, mt + mb);
    ASSERT_EQ(S.n_cols, n);

    vector<double> Y((mt + mb) * k, 0.0);
    S(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
      mt + mb, k, n, 1.0, X.data(), n, 0.0, Y.data(), mt + mb);

    // Reference: top block = Tm*X, bottom block = Bm*X.
    vector<double> Yt(mt * k), Yb(mb * k);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, mt, k, n,
               1.0, Tm.data(), mt, X.data(), n, 0.0, Yt.data(), mt);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, mb, k, n,
               1.0, Bm.data(), mb, X.data(), n, 0.0, Yb.data(), mb);
    vector<double> ref((mt + mb) * k);
    for (int64_t j = 0; j < k; ++j) {
        for (int64_t i = 0; i < mt; ++i) ref[i + j * (mt + mb)] = Yt[i + j * mt];
        for (int64_t i = 0; i < mb; ++i) ref[mt + i + j * (mt + mb)] = Yb[i + j * mb];
    }
    ASSERT_LE(rel_err(Y.data(), ref.data(), (mt + mb) * k), 1e-12);
}

// [Top; Bot]^T * Y = Top^T * Y_top + Bot^T * Y_bot.
TEST_F(TestVStackOp, trans_matches_sum) {
    int64_t mt = 14, mb = 11, n = 7, k = 4;
    vector<double> Tm(mt * n), Bm(mb * n), Y((mt + mb) * k);
    RNGState<> state(99);
    RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), Tm.data(), state);
    RandBLAS::fill_dense(RandBLAS::DenseDist(mb, n), Bm.data(), state);
    RandBLAS::fill_dense(RandBLAS::DenseDist(mt + mb, k), Y.data(), state);

    rl::DenseLinOp<double> Top(mt, n, Tm.data(), mt, Layout::ColMajor);
    rl::DenseLinOp<double> Bot(mb, n, Bm.data(), mb, Layout::ColMajor);
    rl::VStackOp<rl::DenseLinOp<double>, rl::DenseLinOp<double>> S(Top, Bot);

    vector<double> Z(n * k, 0.0);
    S(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
      n, k, mt + mb, 1.0, Y.data(), mt + mb, 0.0, Z.data(), n);

    // Reference: Top^T * Y_top + Bot^T * Y_bot.
    vector<double> Yt(mt * k), Yb(mb * k);
    for (int64_t j = 0; j < k; ++j) {
        for (int64_t i = 0; i < mt; ++i) Yt[i + j * mt] = Y[i + j * (mt + mb)];
        for (int64_t i = 0; i < mb; ++i) Yb[i + j * mb] = Y[mt + i + j * (mt + mb)];
    }
    vector<double> ref(n * k, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, mt,
               1.0, Tm.data(), mt, Yt.data(), mt, 0.0, ref.data(), n);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, k, mb,
               1.0, Bm.data(), mb, Yb.data(), mb, 1.0, ref.data(), n);
    ASSERT_LE(rel_err(Z.data(), ref.data(), n * k), 1e-12);
}

// Headline: Cholesky-QR of A_hat = [A; mu*I] yields R with R^T R = A^T A + mu^2 I.
TEST_F(TestVStackOp, augmented_gram_is_regularized) {
    int64_t m = 60, n = 24;
    double mu = 1e-2;
    vector<double> A(m * n);
    RNGState<> state(2024);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    rl::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    rl::ScaledIdentityOp<double> mu_op(n, mu);
    rl::VStackOp<rl::DenseLinOp<double>, rl::ScaledIdentityOp<double>> A_hat(A_op, mu_op);

    ASSERT_EQ(A_hat.n_rows, m + n);
    ASSERT_EQ(A_hat.n_cols, n);

    vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> qr(/*timing=*/false,
                                         std::pow(std::numeric_limits<double>::epsilon(), 0.85));
    int status = qr.call(A_hat, R.data(), n);
    ASSERT_EQ(status, 0);

    // Zero the strict lower triangle of R (potrf leaves it untouched).
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            R[i + j * n] = 0.0;

    // R^T R should equal A^T A + mu^2 I.
    vector<double> RtR(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n,
               1.0, R.data(), n, R.data(), n, 0.0, RtR.data(), n);

    vector<double> G(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m,
               1.0, A.data(), m, A.data(), m, 0.0, G.data(), n);
    for (int64_t i = 0; i < n; ++i) G[i + i * n] += mu * mu;

    ASSERT_LE(rel_err(RtR.data(), G.data(), n * n), 1e-10);
}

// CQRRT handed the augmented operator A_hat = [A; mu*I] directly: it sketches
// A_hat via VStack's blocked-sketch overload and Grams A_hat, so R^T R = A^T A +
// mu^2 I -- with no CQRRT source changes. (block_size < n exercises the blocking.)
TEST_F(TestVStackOp, cqrrt_augmented_gram) {
    int64_t m = 90, n = 20;
    double mu = 1e-2;
    vector<double> A(m * n);
    RNGState<> state(2025);
    RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    rl::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    rl::ScaledIdentityOp<double> mu_op(n, mu);
    rl::VStackOp<rl::DenseLinOp<double>, rl::ScaledIdentityOp<double>> A_hat(A_op, mu_op);
    A_hat.block_size = 8;

    vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRT_linops<double> qr(/*timing=*/false,
                                        std::pow(std::numeric_limits<double>::epsilon(), 0.85));
    qr.nnz = 4;
    int status = qr.call(A_hat, R.data(), n, /*d_factor=*/2.0, state);
    ASSERT_EQ(status, 0);

    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            R[i + j * n] = 0.0;

    vector<double> RtR(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n,
               1.0, R.data(), n, R.data(), n, 0.0, RtR.data(), n);

    vector<double> G(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m,
               1.0, A.data(), m, A.data(), m, 0.0, G.data(), n);
    for (int64_t i = 0; i < n; ++i) G[i + i * n] += mu * mu;

    ASSERT_LE(rel_err(RtR.data(), G.data(), n * n), 1e-8);
}

