// Tests for VStackLinOp (vertical concatenation [Top; Bot]) and ScaledIdentityLinOp
// (matrix-free mu*I), including the headline use: the mu-regularized augmented
// operator A_hat = [A; mu*I] whose Cholesky-QR factor R satisfies
// R^T R = A^T A + mu^2 I.

#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include "../test_helpers.hh"
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using blas::Side;
using RandBLAS::RNGState;

namespace rl = RandLAPACK::linops;

using RandLAPACK::testing::rel_fro_error;

// R from a QR of [A; mu I] (its strict lower triangle ignored) must satisfy
// R^T R = A^T A + mu^2 I to within tol.
static void check_regularized_gram(vector<double> R, const vector<double>& A, int64_t m, int64_t n,
                                   double mu, double tol) {
    RandLAPACK::util::get_U(n, n, R.data(), n);
    vector<double> RtR(n * n, 0.0), G(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, n,
               1.0, R.data(), n, R.data(), n, 0.0, RtR.data(), n);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, m,
               1.0, A.data(), m, A.data(), m, 0.0, G.data(), n);
    for (int64_t i = 0; i < n; ++i) G[i + i * n] += mu * mu;
    ASSERT_LE(rel_fro_error(RtR.data(), G.data(), n * n), tol);
}

class TestVStackLinOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}
};

// mu*I applied to a block equals mu * that block.
TEST_F(TestVStackLinOp, scaled_identity_applies_mu) {
    int64_t n = 12, k = 5;
    double mu = 0.37;
    vector<double> X(n * k);
    RNGState<> state(42);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(n, k), X.data(), state);

    rl::ScaledIdentityLinOp<double> I_op(n, mu);
    ASSERT_EQ(I_op.n_rows, n);
    ASSERT_EQ(I_op.n_cols, n);

    vector<double> Y(n * k, 0.0);
    I_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, k, n, 1.5, X.data(), n, 0.0, Y.data(), n);

    vector<double> ref(n * k);
    for (int64_t i = 0; i < n * k; ++i) ref[i] = 1.5 * mu * X[i];
    ASSERT_LE(rel_fro_error(Y.data(), ref.data(), n * k), 1e-14);

    // beta != 0 accumulation: Y := 2*Y + alpha*mu*X.
    vector<double> Y2 = Y;
    I_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         n, k, n, -0.5, X.data(), n, 2.0, Y2.data(), n);
    vector<double> ref2(n * k);
    for (int64_t i = 0; i < n * k; ++i) ref2[i] = 2.0 * Y[i] - 0.5 * mu * X[i];
    ASSERT_LE(rel_fro_error(Y2.data(), ref2.data(), n * k), 1e-14);
}

// ScaledIdentityLinOp rejects a call whose output-row dim m does not match n_rows,
// even when the contracted dim k == m.
TEST_F(TestVStackLinOp, scaled_identity_wrong_m_throws) {
    int64_t n_rows = 5, m = 4;   // m != n_rows, k == m below
    double mu = 1.1;
    rl::ScaledIdentityLinOp<double> I_op(n_rows, mu);

    vector<double> B(m * m, 0.0), C(m * m, 0.0);
    ASSERT_THROW(
        I_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
             m, m, /*k=*/m, 1.0, B.data(), m, 0.0, C.data(), m),
        RandLAPACK::Error);
}

// [Top; Bot] * X = [Top*X; Bot*X], stacked row-wise.
TEST_F(TestVStackLinOp, notrans_matches_stack) {
    int64_t mt = 17, mb = 9, n = 8, k = 6;   // Top: mt x n, Bot: mb x n
    vector<double> Tm(mt * n), Bm(mb * n), X(n * k);
    RNGState<> state(7);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), Tm.data(), state);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mb, n), Bm.data(), state);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(n, k), X.data(), state);

    rl::DenseLinOp<double> Top(mt, n, Tm.data(), mt, Layout::ColMajor);
    rl::DenseLinOp<double> Bot(mb, n, Bm.data(), mb, Layout::ColMajor);
    rl::VStackLinOp<rl::DenseLinOp<double>, rl::DenseLinOp<double>> S(Top, Bot);

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
    ASSERT_LE(rel_fro_error(Y.data(), ref.data(), (mt + mb) * k), 1e-12);
}

// [Top; Bot]^T * Y = Top^T * Y_top + Bot^T * Y_bot.
TEST_F(TestVStackLinOp, trans_matches_sum) {
    int64_t mt = 14, mb = 11, n = 7, k = 4;
    vector<double> Tm(mt * n), Bm(mb * n), Y((mt + mb) * k);
    RNGState<> state(99);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), Tm.data(), state);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mb, n), Bm.data(), state);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mt + mb, k), Y.data(), state);

    rl::DenseLinOp<double> Top(mt, n, Tm.data(), mt, Layout::ColMajor);
    rl::DenseLinOp<double> Bot(mb, n, Bm.data(), mb, Layout::ColMajor);
    rl::VStackLinOp<rl::DenseLinOp<double>, rl::DenseLinOp<double>> S(Top, Bot);

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
    ASSERT_LE(rel_fro_error(Z.data(), ref.data(), n * k), 1e-12);
}

// Headline: Cholesky-QR of A_hat = [A; mu*I] yields R with R^T R = A^T A + mu^2 I.
TEST_F(TestVStackLinOp, augmented_gram_is_regularized) {
    int64_t m = 60, n = 24;
    double mu = 1e-2;
    vector<double> A(m * n);
    RNGState<> state(2024);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    rl::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    rl::ScaledIdentityLinOp<double> mu_op(n, mu);
    rl::VStackLinOp<rl::DenseLinOp<double>, rl::ScaledIdentityLinOp<double>> A_hat(A_op, mu_op);

    ASSERT_EQ(A_hat.n_rows, m + n);
    ASSERT_EQ(A_hat.n_cols, n);

    vector<double> R(n * n, 0.0);
    RandLAPACK::CholQR_linops<double> qr(/*timing=*/false,
                                         std::pow(std::numeric_limits<double>::epsilon(), 0.85));
    int status = qr.call(A_hat, R.data(), n);
    ASSERT_EQ(status, 0);

    check_regularized_gram(R, A, m, n, mu, 1e-10);
}

// CQRRTO_linops handed A_hat = [A; mu*I] directly: it sketches A_hat through VStackLinOp's
// blocked sketch overload and forms its Gram, so R^T R = A^T A + mu^2 I.
// (block_size < n exercises the blocking.)
TEST_F(TestVStackLinOp, cqrrto_linops_augmented_gram) {
    int64_t m = 90, n = 20;
    double mu = 1e-2;
    vector<double> A(m * n);
    RNGState<> state(2025);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(m, n), A.data(), state);

    rl::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);
    rl::ScaledIdentityLinOp<double> mu_op(n, mu);
    rl::VStackLinOp<rl::DenseLinOp<double>, rl::ScaledIdentityLinOp<double>> A_hat(A_op, mu_op);
    A_hat.block_size = 8;

    vector<double> R(n * n, 0.0);
    RandLAPACK::CQRRTO_linops<double> qr(/*timing=*/false,
                                         std::pow(std::numeric_limits<double>::epsilon(), 0.85));
    qr.nnz = 4;
    int status = qr.call(A_hat, R.data(), n, /*d_factor=*/2.0, state);
    ASSERT_EQ(status, 0);

    check_regularized_gram(R, A, m, n, mu, 1e-8);
}


// The blocked sketch overload must equal sketching the explicit stacked matrix, for sparse
// and dense sketches, with a partial last block and beta != 0; the top block is a DenseLinOp
// stored ColMajor, then RowMajor with a padded lda whose padding (NaN) must not be read.
TEST_F(TestVStackLinOp, sketch_matches_explicit_stack) {
    int64_t mt = 23, n = 11, d = 9;
    double mu = 0.3, alpha = 1.5, beta = -0.5;
    vector<double> A(mt * n);
    RNGState<> state(99);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), A.data(), state);
    const int64_t lda_row = n + 3;
    vector<double> A_row(mt * lda_row, std::numeric_limits<double>::quiet_NaN());
    for (int64_t i = 0; i < mt; ++i)
        for (int64_t j = 0; j < n; ++j) A_row[i * lda_row + j] = A[i + j * mt];
    rl::DenseLinOp<double> A_op(mt, n, A.data(), mt, Layout::ColMajor);
    rl::DenseLinOp<double> A_row_op(mt, n, A_row.data(), lda_row, Layout::RowMajor);
    rl::ScaledIdentityLinOp<double> mu_op(n, mu);
    rl::VStackLinOp<rl::DenseLinOp<double>, rl::ScaledIdentityLinOp<double>> A_hat(A_op, mu_op), A_hat_row(A_row_op, mu_op);
    A_hat.block_size = A_hat_row.block_size = 4;

    const int64_t mr = mt + n;
    vector<double> Ahat_explicit(mr * n, 0.0);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < mt; ++i) Ahat_explicit[i + j * mr] = A[i + j * mt];
        Ahat_explicit[mt + j + j * mr] = mu;
    }
    const int64_t ldc = d + 3;
    vector<double> C0(ldc * n);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(ldc, n), C0.data(), state);

    auto check = [&](auto& stack, auto& S, auto& S_ref) {
        vector<double> C = C0, C_ref = C0;
        stack(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr, alpha, S, beta, C.data(), ldc);
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr,
                                 alpha, S_ref, Ahat_explicit.data(), mr, beta, C_ref.data(), ldc);
        ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), ldc * n), 1e-12);
    };
    for (auto* stack : {&A_hat, &A_hat_row}) {
        RandBLAS::SparseDist SD(d, mr, 3);
        RandBLAS::SparseSkOp<double> S_sp(SD, RNGState<>(5)), S_sp_ref(SD, RNGState<>(5));
        check(*stack, S_sp, S_sp_ref);
        RandBLAS::DenseDist DD(d, mr);
        RandBLAS::DenseSkOp<double> S_de(DD, RNGState<>(6)), S_de_ref(DD, RNGState<>(6));
        check(*stack, S_de, S_de_ref);
    }
}

// materialize_cols of mu*I writes exact columns at any offset and width, leaves the padding
// rows of buf alone, and checks its range and ldb.
TEST_F(TestVStackLinOp, scaled_identity_materialize_cols) {
    const int64_t n = 9, ldb = n + 3;
    const double mu = 0.7;
    rl::ScaledIdentityLinOp<double> I_op(n, mu);
    static_assert(rl::HasMaterializeCols<rl::ScaledIdentityLinOp<double>>);
    for (auto [c0, cnt] : {std::pair<int64_t, int64_t>{0, n}, {2, 4}, {7, 2}, {5, 0}}) {
        vector<double> buf(ldb * n, -42.0);
        I_op.materialize_cols(c0, cnt, buf.data(), ldb);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < ldb; ++i) {
                const double want = (j >= cnt || i >= n) ? -42.0 : (i == c0 + j ? mu : 0.0);
                ASSERT_EQ(buf[i + j * ldb], want) << "cols " << c0 << "+" << cnt << " at " << i << "," << j;
            }
    }
    vector<double> buf(ldb * n);
    EXPECT_THROW(I_op.materialize_cols(-1, 2, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(I_op.materialize_cols(0, -1, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(I_op.materialize_cols(6, 4, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(I_op.materialize_cols(0, 2, buf.data(), n - 1), RandLAPACK::Error);
    // The range check cannot overflow: start + count is never formed out of range.
    const int64_t big = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(I_op.materialize_cols(big, 1, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(I_op.materialize_cols(1, big, buf.data(), ldb), RandLAPACK::Error);
}

// A stack has materialize_cols only when both blocks do; it writes the explicit stacked columns
// through the blocks' own materialize_cols, with no product.
TEST_F(TestVStackLinOp, materialize_cols_forwards_to_both_blocks) {
    const int64_t mt = 13, n = 8, mr = mt + n, ldb = mr + 2;
    const double mu = 0.4;
    vector<double> A(mt * n);
    RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), A.data(), RNGState<>(17));
    test_helpers::MaterializeColsProbe<double> top{mt, n, A.data()};
    rl::ScaledIdentityLinOp<double> bot(n, mu);
    using Probe = test_helpers::MaterializeColsProbe<double>;
    using SI = rl::ScaledIdentityLinOp<double>;
    rl::VStackLinOp<Probe, SI> A_hat(top, bot);
    static_assert(rl::HasMaterializeCols<rl::VStackLinOp<Probe, SI>>);
    static_assert(rl::HasMaterializeCols<rl::VStackLinOp<rl::DenseLinOp<double>, SI>>);
    static_assert(!rl::HasMaterializeCols<rl::VStackLinOp<test_helpers::ConceptOnlyLinOp<double>, SI>>);

    const int64_t c0 = 3, cnt = 5;
    vector<double> buf(ldb * cnt, -42.0);
    A_hat.materialize_cols(c0, cnt, buf.data(), ldb);
    for (int64_t j = 0; j < cnt; ++j)
        for (int64_t i = 0; i < ldb; ++i) {
            const double want = (i < mt) ? A[i + (c0 + j) * mt] : (i < mr) ? (i - mt == c0 + j ? mu : 0.0) : -42.0;
            ASSERT_EQ(buf[i + j * ldb], want) << i << "," << j;
        }
    EXPECT_EQ(top.direct_cols, cnt);
    EXPECT_EQ(top.product_cols, 0);
    EXPECT_THROW(A_hat.materialize_cols(0, 2, buf.data(), mr - 1), RandLAPACK::Error);
    EXPECT_THROW(A_hat.materialize_cols(c0, n, buf.data(), ldb), RandLAPACK::Error);
    const int64_t big = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(A_hat.materialize_cols(big, 1, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(A_hat.materialize_cols(1, big, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_EQ(top.direct_cols, cnt);   // every failed call stopped before the blocks
    const auto& A_hat_const = A_hat;
    A_hat_const.materialize_cols(c0, cnt, buf.data(), ldb);
    EXPECT_EQ(top.direct_cols, 2 * cnt);
}

// The sketch overload of a stack whose blocks have materialize_cols reads its column blocks
// through it (the top block is never applied) and equals the product path of a stack whose top
// block lacks it, and the explicit stacked matrix, for sparse and dense sketches with a partial
// last block.
TEST_F(TestVStackLinOp, sketch_reads_columns_directly) {
    const int64_t mt = 23, n = 11, d = 9, mr = mt + n, ldc = d + 2;
    const double mu = 0.3, alpha = 1.5, beta = -0.5;
    vector<double> A(mt * n);
    RNGState<> state(99);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), A.data(), state);
    test_helpers::MaterializeColsProbe<double> top{mt, n, A.data()};
    test_helpers::ConceptOnlyLinOp<double> top_products{mt, n, A.data()};
    rl::ScaledIdentityLinOp<double> bot(n, mu);
    rl::VStackLinOp<test_helpers::MaterializeColsProbe<double>, rl::ScaledIdentityLinOp<double>> direct(top, bot);
    rl::VStackLinOp<test_helpers::ConceptOnlyLinOp<double>, rl::ScaledIdentityLinOp<double>> products(top_products, bot);
    direct.block_size = products.block_size = 4;

    vector<double> Ahat_explicit(mr * n, 0.0);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < mt; ++i) Ahat_explicit[i + j * mr] = A[i + j * mt];
        Ahat_explicit[mt + j + j * mr] = mu;
    }
    vector<double> C0(ldc * n);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(ldc, n), C0.data(), state);
    auto check = [&](auto& S, auto& S_prod, auto& S_ref) {
        vector<double> C = C0, C_prod = C0, C_ref = C0;
        direct(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr, alpha, S, beta, C.data(), ldc);
        products(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr, alpha, S_prod, beta, C_prod.data(), ldc);
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr,
                                 alpha, S_ref, Ahat_explicit.data(), mr, beta, C_ref.data(), ldc);
        // Identity columns make the product path exact, so the two paths agree to rounding of
        // the sketch alone.
        EXPECT_LE(rel_fro_error(C.data(), C_prod.data(), ldc * n), 4 * std::numeric_limits<double>::epsilon());
        EXPECT_LE(rel_fro_error(C.data(), C_ref.data(), ldc * n), 1e-12);
    };
    RandBLAS::SparseDist SD(d, mr, 3);
    RandBLAS::SparseSkOp<double> S_sp(SD, RNGState<>(5)), S_sp2(SD, RNGState<>(5)), S_sp_ref(SD, RNGState<>(5));
    check(S_sp, S_sp2, S_sp_ref);
    RandBLAS::DenseDist DD(d, mr);
    RandBLAS::DenseSkOp<double> S_de(DD, RNGState<>(6)), S_de2(DD, RNGState<>(6)), S_de_ref(DD, RNGState<>(6));
    check(S_de, S_de2, S_de_ref);
    EXPECT_EQ(top.product_cols, 0);
    EXPECT_EQ(top.direct_cols, 2 * n);
}

// [Top; mu I] in both layouts, NoTrans and Trans, padded leading dimensions, alpha and beta,
// against gemm on the explicit stacked matrix; instantiated for float and double.
template <typename T>
static void check_vstack_against_explicit(Layout layout) {
    const int64_t mt = 13, n = 6, k = 5, mr = mt + n;
    const T mu = (T)0.4, alpha = (T)1.5, beta = (T)-0.5;
    const T tol = (std::is_same_v<T, float>) ? (T)1e-5 : (T)1e-12;
    const bool col = (layout == Layout::ColMajor);
    RNGState<> state(31);

    // Explicit Top and A_hat in the requested layout.
    vector<T> Tcm(mt * n);
    state = RandBLAS::fill_dense(RandBLAS::DenseDist(mt, n), Tcm.data(), state);
    auto at = [&](int64_t ld, int64_t i, int64_t j) { return col ? i + j * ld : j + i * ld; };
    const int64_t ld_top = col ? mt : n, ld_hat = col ? mr : n;
    vector<T> Top_buf(mt * n), Ahat(mr * n, (T)0);
    for (int64_t i = 0; i < mt; ++i)
        for (int64_t j = 0; j < n; ++j) {
            Top_buf[at(ld_top, i, j)] = Tcm[i + j * mt];
            Ahat[at(ld_hat, i, j)] = Tcm[i + j * mt];
        }
    for (int64_t j = 0; j < n; ++j) Ahat[at(ld_hat, mt + j, j)] = mu;

    rl::DenseLinOp<T> Top(mt, n, Top_buf.data(), ld_top, layout);
    rl::ScaledIdentityLinOp<T> Bot(n, mu);
    rl::VStackLinOp<rl::DenseLinOp<T>, rl::ScaledIdentityLinOp<T>> S(Top, Bot);

    // NoTrans: C (mr x k) = alpha * A_hat * X (n x k) + beta * C, for beta == 0 and != 0.
    for (T b_used : {(T)0, beta}) {
        const int64_t ldx = (col ? n : k) + 2, ldc = (col ? mr : k) + 3;
        vector<T> X((col ? k : n) * ldx), C((col ? k : mr) * ldc);
        state = RandBLAS::fill_dense(RandBLAS::DenseDist(X.size(), 1), X.data(), state);
        state = RandBLAS::fill_dense(RandBLAS::DenseDist(C.size(), 1), C.data(), state);
        vector<T> C_ref = C;
        S(Side::Left, layout, Op::NoTrans, Op::NoTrans, mr, k, n, alpha, X.data(), ldx, b_used, C.data(), ldc);
        blas::gemm(layout, Op::NoTrans, Op::NoTrans, mr, k, n, alpha, Ahat.data(), ld_hat,
                   X.data(), ldx, b_used, C_ref.data(), ldc);
        ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), (int64_t)C.size()), tol);
    }
    // Trans: Z (n x k) = alpha * A_hat^T * Y (mr x k) + beta * Z.
    {
        const int64_t ldy = (col ? mr : k) + 4, ldz = (col ? n : k) + 1;
        vector<T> Y((col ? k : mr) * ldy), Z((col ? k : n) * ldz);
        state = RandBLAS::fill_dense(RandBLAS::DenseDist(Y.size(), 1), Y.data(), state);
        state = RandBLAS::fill_dense(RandBLAS::DenseDist(Z.size(), 1), Z.data(), state);
        vector<T> Z_ref = Z;
        S(Side::Left, layout, Op::Trans, Op::NoTrans, n, k, mr, alpha, Y.data(), ldy, beta, Z.data(), ldz);
        blas::gemm(layout, Op::Trans, Op::NoTrans, n, k, mr, alpha, Ahat.data(), ld_hat,
                   Y.data(), ldy, beta, Z_ref.data(), ldz);
        ASSERT_LE(rel_fro_error(Z.data(), Z_ref.data(), (int64_t)Z.size()), tol);
    }
}

TEST_F(TestVStackLinOp, explicit_stack_colmajor_double) { check_vstack_against_explicit<double>(Layout::ColMajor); }
TEST_F(TestVStackLinOp, explicit_stack_rowmajor_double) { check_vstack_against_explicit<double>(Layout::RowMajor); }
TEST_F(TestVStackLinOp, explicit_stack_colmajor_float)  { check_vstack_against_explicit<float>(Layout::ColMajor); }
TEST_F(TestVStackLinOp, explicit_stack_rowmajor_float)  { check_vstack_against_explicit<float>(Layout::RowMajor); }

TEST_F(TestVStackLinOp, invalid_arguments_throw) {
    int64_t mt = 6, n = 4, mr = mt + n;
    vector<double> Tm(mt * n, 1.0), Wide(mt * (n + 1), 1.0);
    rl::DenseLinOp<double> Top(mt, n, Tm.data(), mt, Layout::ColMajor);
    rl::DenseLinOp<double> TopWide(mt, n + 1, Wide.data(), mt, Layout::ColMajor);
    rl::ScaledIdentityLinOp<double> Bot(n, 2.0);
    using V = rl::VStackLinOp<rl::DenseLinOp<double>, rl::ScaledIdentityLinOp<double>>;
    EXPECT_THROW(V(TopWide, Bot), RandLAPACK::Error);

    V S(Top, Bot);
    vector<double> B(mr * mr, 1.0), C(mr * mr, 0.0);
    EXPECT_THROW(S(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mr, n, n, 1.0, B.data(), mr, 0.0, C.data(), mr), RandLAPACK::Error);
    EXPECT_THROW(S(Side::Left, Layout::ColMajor, Op::NoTrans, Op::Trans, mr, n, n, 1.0, B.data(), mr, 0.0, C.data(), mr), RandLAPACK::Error);
    EXPECT_THROW(S(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mr - 1, n, n, 1.0, B.data(), mr, 0.0, C.data(), mr), RandLAPACK::Error);
    EXPECT_THROW(S(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, mr, n, n, 1.0, B.data(), mr, 0.0, C.data(), mr - 1), RandLAPACK::Error);
    EXPECT_THROW(S(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans, n, n, mr, 1.0, B.data(), mr - 1, 0.0, C.data(), n), RandLAPACK::Error);
    EXPECT_THROW(Bot(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, 2, n - 1, 1.0, B.data(), n, 0.0, C.data(), n), RandLAPACK::Error);
    EXPECT_THROW(Bot(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, 2, n, 1.0, B.data(), n - 1, 0.0, C.data(), n), RandLAPACK::Error);

    RandBLAS::DenseSkOp<double> S_bad(RandBLAS::DenseDist(3, mr + 1), RNGState<>(1));
    EXPECT_THROW(S(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 3, n, mr, 1.0, S_bad, 0.0, C.data(), 3), RandLAPACK::Error);
    RandBLAS::DenseSkOp<double> S_ok(RandBLAS::DenseDist(3, mr), RNGState<>(1));
    EXPECT_THROW(S(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, 3, n, mr, 1.0, S_ok, 0.0, C.data(), 2), RandLAPACK::Error);
}
