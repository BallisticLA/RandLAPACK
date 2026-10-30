#include <blas.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>
#include <math.h>
#include <lapack.hh>
#include <RandBLAS/testing/comparison.hh>
#include "../test_helpers.hh"
#include <concepts>
#include <limits>
#include <type_traits>
#include <utility>


using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::DenseDist;
using RandBLAS::SparseDist;
using RandBLAS::RNGState;


/**
 * Note: a few implicit linear operators are tested implicitly (ha) in
 * test_determiter.cc. It's important to have tests for these things
 * since bugs in their implementation can be hard to track down. 
 */


class TestSpectralPrecondLinearOperator: public ::testing::Test {

    protected:
    
    virtual void SetUp() {};

    virtual void TearDown() {};

    // Run on a diagonal matrix with an optimal rank-k preconditioner.
    template <typename T>
    void run_diag(int64_t n, int64_t k, T mu) {
        int64_t i;
        vector<T> alleigs(n);
        vector<T> allV(n*n, 0.0);
        for (i = 0; i < n; ++i) {
            alleigs[i] = std::pow((T)i + (T)1.0, (T) -3.0);
            allV[i + i*n] = 1.0;
        }

        vector<T> G_mu(n*n, 0.0);
        for (i = 0; i < n; ++i) {
            G_mu[i + i*n] = alleigs[i] + mu;
        }

        vector<T> pceigs(k);
        vector<T> pcV(n*k, 0.0);
        for (i = 0; i < k; ++i) {
            pceigs[i] = alleigs[i];
            pcV[i + i*n] = 1.0;
        }
        vector<T> G_mu_pre_expect(n*n, 0.0);
        T scale_on_precond_subspace = alleigs[k-1] + mu;
        for (i = 0; i < n; ++i) {
            if (i < k) {
                G_mu_pre_expect[i + i*n] = scale_on_precond_subspace;
            } else {
                G_mu_pre_expect[i + i*n] = alleigs[i] + mu;
            }
        }
        RandLAPACK::linops::SpectralPrecond<T> invP_operator(n);
        vector<T> mus(1, mu);
        invP_operator.prep(pcV, pceigs, mus, n);
        vector<T> G_mu_pre_actual(n*n, 0.0);
        invP_operator(blas::Layout::ColMajor, n, (T) 1.0,  G_mu.data(), n, (T)0.0, G_mu_pre_actual.data(), n);
        EXPECT_EQ(RandBLAS::testing::matrices_approx_equal(
            Layout::ColMajor, Op::NoTrans, n, n, G_mu_pre_actual.data(), n,
            G_mu_pre_expect.data(), n, __RANDBLAS_PRETTY_FUNCTION__, 
            __FILE__, __LINE__
        ), "");
        return;
    }
};

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n3_k1) {
    run_diag<float>(3, 1, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n3_k2) {
    run_diag<float>(3, 2, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n4_k1) {
    run_diag<float>(4, 1, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n4_k2) {
    run_diag<float>(4, 2, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n4_k3) {
    run_diag<float>(4, 3, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k1) {
    run_diag<float>(5, 1, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k2) {
    run_diag<float>(5, 2, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k3) {
    run_diag<float>(5, 3, 0.1);
}

TEST_F(TestSpectralPrecondLinearOperator, test_diag_n5_k4) {
    run_diag<float>(5, 4, 0.1);
}

// The generic materialize fallback allocates an n by n identity behind the
// caller's back; that hidden allocation is capped.
namespace {
struct IdentityFallbackProbeOp {
    using scalar_t = double;
    void operator()(Layout, Op, Op,
                    int64_t, int64_t, int64_t, double,
                    const double*, int64_t, double,
                    double*, int64_t) {}
};
}

TEST(TestMaterializeGuard, generic_fallback_refuses_huge_identity) {
    IdentityFallbackProbeOp A;
    int64_t n_over = RandLAPACK::MATERIALIZE_IDENTITY_MAX_DIM + 1;
    double stub = 0.0;
    // The guard fires before the output buffer is touched, so a stub is safe.
    EXPECT_THROW(RandLAPACK::materialize(A, n_over, n_over, &stub, n_over), RandLAPACK::Error);

    // Below the cap the generic path still works.
    int64_t n_small = 4;
    std::vector<double> buf(n_small * n_small, 1.0);
    RandLAPACK::materialize(A, n_small, n_small, buf.data(), n_small);
    // The probe writes nothing, so only the zeroing of the buffer happened.
    for (auto v : buf)
        ASSERT_EQ(v, 0.0);
}

// The CompositeOperator overload multiplies materialized operands with one
// gemm: no identity, no cap, no call through the composite's operator().
TEST(TestMaterializeComposite, matches_reference_product) {
    int64_t m = 7;
    int64_t k = 4;
    int64_t n = 5;
    int64_t ldb = m + 2;

    // Deterministic operand fills.
    vector<double> L_buf(m * k), R_buf(k * n);
    for (int64_t i = 0; i < m * k; ++i)
        L_buf[i] = 0.5 + (double)((3 * i) % 11);
    for (int64_t i = 0; i < k * n; ++i)
        R_buf[i] = -1.0 + (double)((5 * i) % 7);

    RandLAPACK::linops::DenseLinOp<double> L_op(m, k, L_buf.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> R_op(k, n, R_buf.data(), k, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

    // Sentinel-fill past-the-column entries to check ldb handling.
    vector<double> buf(ldb * n, -42.0);
    RandLAPACK::materialize(comp, m, n, buf.data(), ldb);

    vector<double> ref(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k,
               1.0, L_buf.data(), m, R_buf.data(), k, 0.0, ref.data(), m);

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i)
            ASSERT_DOUBLE_EQ(buf[i + j * ldb], ref[i + j * m]);
        for (int64_t i = m; i < ldb; ++i)
            ASSERT_EQ(buf[i + j * ldb], -42.0) << "materialize wrote past column height";
    }
}

// An operator with materialize_cols is materialized through it: no product, no identity and no
// cap (wider than the generic fallback allows), and the padding rows of buf are left alone.
TEST(TestMaterializeCols, operator_with_materialize_cols) {
    const int64_t m = 3, n = RandLAPACK::MATERIALIZE_IDENTITY_MAX_DIM + 1, ldb = m + 1;
    vector<double> A(m * n);
    for (int64_t i = 0; i < m * n; ++i) A[i] = (double)(i % 13) - 6.0;
    test_helpers::MaterializeColsProbe<double> probe{m, n, A.data()};
    vector<double> buf(ldb * n, -42.0);
    RandLAPACK::materialize(probe, m, n, buf.data(), ldb);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < ldb; ++i)
            ASSERT_EQ(buf[i + j * ldb], i < m ? A[i + j * m] : -42.0) << i << "," << j;
    EXPECT_EQ(probe.product_cols, 0);
    EXPECT_EQ(probe.direct_cols, n);
    // Shape and ldb are checked before the hook runs (the probe checks neither).
    EXPECT_THROW(RandLAPACK::materialize(probe, m + 1, n, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::materialize(probe, m, n - 1, buf.data(), ldb), RandLAPACK::Error);
    EXPECT_THROW(RandLAPACK::materialize(probe, m, n, buf.data(), m - 1), RandLAPACK::Error);
    EXPECT_EQ(probe.direct_cols, n);
}

namespace {
// Forwards every product the borrowed operator has, counting the calls, and with Cols also its
// materialize_cols, counting the columns, so a test can tell which path a caller took.
template <typename LinOp, bool Cols>
struct CountingLinOp {
    using scalar_t = typename LinOp::scalar_t;
    const int64_t n_rows;
    const int64_t n_cols;
    LinOp& op;
    int64_t product_calls = 0;
    int64_t direct_cols = 0;
    explicit CountingLinOp(LinOp& A) : n_rows(A.n_rows), n_cols(A.n_cols), op(A) {}
    template <typename... Args>
        requires std::invocable<LinOp&, Args...>
    void operator()(Args&&... args) { ++product_calls; op(std::forward<Args>(args)...); }
    void materialize_cols(int64_t col_start, int64_t col_count, scalar_t* buf, int64_t ldb)
        requires Cols
    {
        direct_cols += col_count;
        op.materialize_cols(col_start, col_count, buf, ldb);
    }
};
} // namespace

using DenseProductsOnly = RandLAPACK::testing::ProductsOnly<RandLAPACK::linops::DenseLinOp<double>>;
using ConceptProductsOnly = RandLAPACK::testing::ProductsOnly<test_helpers::ConceptOnlyLinOp<double>>;
static_assert(RandLAPACK::linops::LinearOperator<DenseProductsOnly>);
static_assert(!RandLAPACK::linops::HasMaterializeCols<DenseProductsOnly>);
// ProductsOnly accepts exactly the calls its operator does: DenseLinOp's Side-taking product, but
// not one on a concept-only operator, so TransposedLinOp's detection of that overload still
// holds (instantiating it compiles its Side::Right branch, which then throws).
template <typename F>
constexpr bool takes_sided_product = std::invocable<F&, blas::Side, Layout, Op, Op, int64_t, int64_t,
                                                    int64_t, double, const double*, int64_t, double,
                                                    double*, int64_t>;
static_assert(takes_sided_product<DenseProductsOnly> && !takes_sided_product<ConceptProductsOnly>);
template struct RandLAPACK::linops::TransposedLinOp<ConceptProductsOnly>;

// CompositeOperator's blocked sketch (block_size > 0) copies the columns of a right operand that
// has materialize_cols (DenseLinOp, SparseLinOp): exactly n columns, and the operand is never
// applied, where an operand without the hook is applied once per column block. Both equal the
// sketch of the explicit product L * R, and each other to rounding of the sketch alone, for
// sparse and dense sketches with a partial last block and beta != 0.
TEST(TestCompositeSketch, blocked_sketch_copies_right_operand_columns) {
    namespace rl = RandLAPACK::linops;
    using RandLAPACK::testing::rel_fro_error;
    const int64_t m = 31, k = 12, n = 10, d = 8, ldc = d + 2, b = 4, n_blocks = 3;
    const double alpha = 1.5, beta = -0.5;
    auto L = test_helpers::random_matrix<double>(m, k, 3);
    auto R = test_helpers::random_matrix<double>(k, n, 4);
    RNGState<> state(5);
    auto R_coo = RandLAPACK::gen::gen_sparse_coo<double>(k, n, 0.4, state);
    auto R_csr = R_coo.as_owning_csr();
    vector<double> R_sp(k * n, 0.0);
    RandLAPACK::util::sparse_to_dense(R_csr, Layout::ColMajor, R_sp.data());
    rl::DenseLinOp<double> L_op(m, k, L.data(), m, Layout::ColMajor);
    rl::DenseLinOp<double> R_dense_op(k, n, R.data(), k, Layout::ColMajor);
    rl::SparseLinOp<RandBLAS::sparse_data::CSRMatrix<double>> R_sparse_op(k, n, R_csr);
    vector<double> C0(ldc * n);
    RandBLAS::fill_dense(DenseDist(ldc, n), C0.data(), RNGState<>(6));

    auto check = [&](auto& R_op, const vector<double>& R_explicit) {
        using ROp = std::remove_reference_t<decltype(R_op)>;
        CountingLinOp<ROp, true> R_cols(R_op);
        CountingLinOp<ROp, false> R_prod(R_op);
        static_assert(rl::HasMaterializeCols<decltype(R_cols)> && !rl::HasMaterializeCols<decltype(R_prod)>);
        rl::CompositeOperator direct(m, n, L_op, R_cols);
        rl::CompositeOperator products(m, n, L_op, R_prod);
        static_assert(!rl::HasMaterializeCols<decltype(direct)>);
        direct.block_size = products.block_size = b;
        vector<double> LR(m * n);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, L.data(), m,
                   R_explicit.data(), k, 0.0, LR.data(), m);
        auto run = [&](auto& S, auto& S_prod, auto& S_ref) {
            vector<double> C = C0, C_prod = C0, C_ref = C0;
            R_cols.product_calls = R_cols.direct_cols = R_prod.product_calls = R_prod.direct_cols = 0;
            direct(blas::Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                   alpha, S, beta, C.data(), ldc);
            products(blas::Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                     alpha, S_prod, beta, C_prod.data(), ldc);
            EXPECT_EQ(R_cols.direct_cols, n);
            EXPECT_EQ(R_cols.product_calls, 0);
            EXPECT_EQ(R_prod.product_calls, n_blocks);
            RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, m,
                                     alpha, S_ref, LR.data(), m, beta, C_ref.data(), ldc);
            // Identity columns make the product path's V block exact.
            const double eps = std::numeric_limits<double>::epsilon();
            EXPECT_LE(rel_fro_error(C.data(), C_prod.data(), ldc * n), 4 * eps);
            EXPECT_LE(rel_fro_error(C.data(), C_ref.data(), ldc * n), 1e-12);
        };
        RandBLAS::SparseDist SD(d, m, 3);
        RandBLAS::SparseSkOp<double> S_sp(SD, RNGState<>(7)), S_sp2(SD, RNGState<>(7)),
                                     S_sp_ref(SD, RNGState<>(7));
        run(S_sp, S_sp2, S_sp_ref);
        DenseDist DD(d, m);
        RandBLAS::DenseSkOp<double> S_de(DD, RNGState<>(8)), S_de2(DD, RNGState<>(8)),
                                    S_de_ref(DD, RNGState<>(8));
        run(S_de, S_de2, S_de_ref);
    };
    check(R_dense_op, R);
    check(R_sparse_op, R_sp);
}

TEST(TestMaterializeComposite, wide_composite_is_exempt_from_identity_cap) {
    // Wider than the generic fallback allows; the operand-product path handles
    // it in m*k + k*n scratch.
    int64_t m = 3;
    int64_t k = 2;
    int64_t n = RandLAPACK::MATERIALIZE_IDENTITY_MAX_DIM + 1;

    vector<double> L_buf(m * k, 1.0), R_buf(k * n);
    for (int64_t i = 0; i < k * n; ++i)
        R_buf[i] = (double)(i % 5);

    RandLAPACK::linops::DenseLinOp<double> L_op(m, k, L_buf.data(), m, Layout::ColMajor);
    RandLAPACK::linops::DenseLinOp<double> R_op(k, n, R_buf.data(), k, Layout::ColMajor);
    RandLAPACK::linops::CompositeOperator comp(m, n, L_op, R_op);

    vector<double> buf(m * n, 0.0);
    RandLAPACK::materialize(comp, m, n, buf.data(), m);

    // With every row of L equal to ones, each column j of the product holds
    // the column sum of R in every entry.
    for (int64_t j = 0; j < n; ++j) {
        double col_sum = R_buf[j * k] + R_buf[j * k + 1];
        for (int64_t i = 0; i < m; ++i)
            ASSERT_DOUBLE_EQ(buf[i + j * m], col_sum);
    }
}
