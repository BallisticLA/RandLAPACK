// Tests for RandLAPACK_extras::linops::ToeplitzLinOp: forward and adjoint FFT
// applies checked against an explicitly formed dense toeplitz(c, r) matrix.

#include "../../linops/ext_toeplitz_linop.hh"
#include <RandLAPACK.hh>
#include <RandBLAS.hh>

#include <blas.hh>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using std::vector;
using blas::Layout;
using blas::Op;
using RandBLAS::RNGState;
using RandLAPACK::testing::rel_fro_error;

namespace ext = RandLAPACK_extras::linops;

class TestToeplitzLinOp : public ::testing::Test {
protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    // Dense toeplitz(c, r): T(i,j) = c[i-j] for i >= j, r[j-i] for j > i.
    // ColMajor, leading dimension m.
    template <typename T>
    static void build_dense(const T* c, int64_t m, const T* r, int64_t n, vector<T>& Tdense) {
        Tdense.assign((size_t)m * (size_t)n, (T)0);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                Tdense[i + j * m] = (i >= j) ? c[i - j] : r[j - i];
    }

    struct Case {
        int64_t m, n;
        vector<double> c, r, Tdense;
    };

    // c[0] == r[0] is required by the operator; the rest is arbitrary Gaussian data.
    static Case make_case(int64_t m, int64_t n, uint32_t seed, bool with_dense = true) {
        Case cs;
        cs.m = m; cs.n = n;
        cs.c.resize(m);
        cs.r.resize(n);
        RNGState<> state(seed);
        state = RandBLAS::fill_dense(RandBLAS::DenseDist(m, 1), cs.c.data(), state);
        RandBLAS::fill_dense(RandBLAS::DenseDist(n, 1), cs.r.data(), state);
        cs.r[0] = cs.c[0];
        if (with_dense) build_dense(cs.c.data(), m, cs.r.data(), n, cs.Tdense);
        return cs;
    }

    // The operator against blas::gemm on the dense matrix, to `tol` relative Frobenius error.
    // poison_c (beta == 0 only) fills C with NaN; the reference starts from a clean zero buffer.
    static void run_case(ext::ToeplitzLinOp<double>& Top, const Case& cs,
                          Op trans_A, int64_t nrhs, double alpha, double beta,
                          bool poison_c, uint32_t seed, double tol = 1e-13) {
        int64_t in_rows  = (trans_A == Op::NoTrans) ? cs.n : cs.m;
        int64_t out_rows = (trans_A == Op::NoTrans) ? cs.m : cs.n;

        vector<double> B(in_rows * nrhs);
        RNGState<> bstate(seed);
        RandBLAS::fill_dense(RandBLAS::DenseDist(in_rows, nrhs), B.data(), bstate);

        vector<double> C(out_rows * nrhs);
        vector<double> C_ref(out_rows * nrhs, 0.0);

        if (poison_c) {
            ASSERT_EQ(beta, 0.0) << "poison_c is only meaningful with beta == 0";
            std::fill(C.begin(), C.end(), std::numeric_limits<double>::quiet_NaN());
            // C_ref stays a clean zero buffer: reference is alpha*op(T)*B only.
            blas::gemm(Layout::ColMajor, trans_A, Op::NoTrans, out_rows, nrhs, in_rows,
                       alpha, cs.Tdense.data(), cs.m, B.data(), in_rows, 0.0, C_ref.data(), out_rows);
        } else {
            RNGState<> cstate(seed + 1);
            RandBLAS::fill_dense(RandBLAS::DenseDist(out_rows, nrhs), C.data(), cstate);
            C_ref = C;
            blas::gemm(Layout::ColMajor, trans_A, Op::NoTrans, out_rows, nrhs, in_rows,
                       alpha, cs.Tdense.data(), cs.m, B.data(), in_rows, beta, C_ref.data(), out_rows);
        }

        Top(Layout::ColMajor, trans_A, Op::NoTrans,
            out_rows, nrhs, in_rows, alpha, B.data(), in_rows, beta, C.data(), out_rows);

        for (double v : C) ASSERT_FALSE(std::isnan(v)) << "C retained an unwritten (NaN) entry";
        ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), out_rows * nrhs), tol);
    }
};

// One dense comparison per case: shape and data seed, direction, one column (the one-column
// descriptors) or kFFTBatch + 1 (crosses the batched recommit), alpha, beta, a NaN-filled C.
struct DenseCase {
    const char* name;
    int64_t m, n;
    uint32_t case_seed;
    Op trans;
    bool multicolumn;
    double alpha, beta;
    bool poison_c;
    uint32_t seed;
};

class TestToeplitzVsDense : public TestToeplitzLinOp, public ::testing::WithParamInterface<DenseCase> {};

TEST_P(TestToeplitzVsDense, matches_dense) {
    const DenseCase& p = GetParam();
    Case cs = make_case(p.m, p.n, p.case_seed);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    ASSERT_EQ(Top.n_rows, cs.m);
    ASSERT_EQ(Top.n_cols, cs.n);
    const int64_t nrhs = p.multicolumn ? ext::ToeplitzLinOp<double>::kFFTBatch + 1 : 1;
    run_case(Top, cs, p.trans, nrhs, p.alpha, p.beta, p.poison_c, p.seed);
}

INSTANTIATE_TEST_SUITE_P(Cases, TestToeplitzVsDense, ::testing::Values(
    DenseCase{"notrans_single_column_matches_dense",           43, 37, 11, Op::NoTrans, false,  1.0,   0.0,  false, 100},
    DenseCase{"notrans_multicolumn_crosses_batch_boundary",    43, 37, 12, Op::NoTrans, true,   1.0,   0.0,  false, 101},
    DenseCase{"trans_single_column_matches_dense",             41, 39, 13, Op::Trans,   false,  1.0,   0.0,  false, 102},
    DenseCase{"trans_multicolumn_crosses_batch_boundary",      41, 39, 14, Op::Trans,   true,   1.0,   0.0,  false, 103},
    DenseCase{"notrans_alpha_beta_accumulate_single_column",   43, 37, 15, Op::NoTrans, false,  2.5,   1.5,  false, 104},
    DenseCase{"notrans_alpha_beta_accumulate_multicolumn",     43, 37, 16, Op::NoTrans, true,   2.5,  -0.75, false, 105},
    DenseCase{"trans_alpha_beta_accumulate_multicolumn",       41, 39, 17, Op::Trans,   true,  -1.25,  0.5,  false, 106},
    DenseCase{"beta_zero_does_not_read_garbage_single_column", 43, 37, 18, Op::NoTrans, false,  1.0,   0.0,  true,  107},
    DenseCase{"beta_zero_does_not_read_garbage_multicolumn",   43, 37, 19, Op::NoTrans, true,   1.0,   0.0,  true,  108},
    DenseCase{"beta_zero_does_not_read_garbage_trans",         41, 39, 20, Op::Trans,   true,   1.0,   0.0,  true,  109}),
    [](const ::testing::TestParamInfo<DenseCase>& info) { return std::string(info.param.name); });

// A single column goes through the one-column descriptors, two columns through the batched
// ones. They must agree: same operator, same input, different descriptors. Also at larger L,
// where FFT rounding is larger.
TEST_F(TestToeplitzLinOp, single_column_matches_batched_path_at_scale) {
    for (auto [m, n] : std::vector<std::pair<int64_t,int64_t>>{{2000,400},{8000,1000}}) {
        Case cs = make_case(m, n, 99, /*with_dense=*/false);
        ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);

        for (Op ts : {Op::NoTrans, Op::Trans}) {
            const int64_t out_rows = (ts == Op::NoTrans) ? cs.m : cs.n;
            const int64_t in_rows  = (ts == Op::NoTrans) ? cs.n : cs.m;

            vector<double> B(in_rows);
            RNGState<> st(7);
            RandBLAS::fill_dense(RandBLAS::DenseDist(in_rows, 1), B.data(), st);

            // Single column: one-column descriptors.
            vector<double> c_single(out_rows, 0.0);
            Top(Layout::ColMajor, ts, Op::NoTrans, out_rows, 1, in_rows,
                1.0, B.data(), in_rows, 0.0, c_single.data(), out_rows);

            // Two columns with the same data in column 0: batched descriptors.
            vector<double> B2(in_rows * 2);
            std::copy(B.begin(), B.end(), B2.begin());
            std::copy(B.begin(), B.end(), B2.begin() + in_rows);
            vector<double> c_batched(out_rows * 2, 0.0);
            Top(Layout::ColMajor, ts, Op::NoTrans, out_rows, 2, in_rows,
                1.0, B2.data(), in_rows, 0.0, c_batched.data(), out_rows);

            double err = rel_fro_error(c_single.data(), c_batched.data(), out_rows);
            EXPECT_LE(err, 1e-13) << "m=" << m << " n=" << n
                                  << " trans=" << (ts == Op::NoTrans ? "N" : "T")
                                  << " rel_fro=" << err;
        }
    }
}

TEST_F(TestToeplitzLinOp, padded_leading_dimensions) {
    Case cs = make_case(37, 21, 211);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    const int64_t nrhs = 3, ldb = cs.n + 5, ldc = cs.m + 4;
    vector<double> B(ldb * nrhs), C(ldc * nrhs, 7.0);
    RNGState<> st(5);
    RandBLAS::fill_dense(RandBLAS::DenseDist(ldb, nrhs), B.data(), st);
    vector<double> C_ref = C;
    Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, nrhs, cs.n, 1.0, B.data(), ldb, 0.5, C.data(), ldc);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, nrhs, cs.n, 1.0, cs.Tdense.data(), cs.m,
               B.data(), ldb, 0.5, C_ref.data(), ldc);
    ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), ldc * nrhs), 1e-13);
}

TEST_F(TestToeplitzLinOp, float_matches_dense) {
    const int64_t m = 40, n = 25;
    Case cs = make_case(m, n, 301);
    vector<float> c(cs.c.begin(), cs.c.end()), r(cs.r.begin(), cs.r.end());
    vector<float> Td(cs.Tdense.begin(), cs.Tdense.end());
    ext::ToeplitzLinOp<float> Top(c.data(), m, r.data(), n);
    for (int64_t nrhs : {int64_t(1), int64_t(3)}) {
        vector<float> B(n * nrhs), C(m * nrhs, 0.0f), C_ref(m * nrhs, 0.0f);
        RNGState<> st(9);
        RandBLAS::fill_dense(RandBLAS::DenseDist(n, nrhs), B.data(), st);
        Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, nrhs, n, 1.0f, B.data(), n, 0.0f, C.data(), m);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, nrhs, n, 1.0f, Td.data(), m,
                   B.data(), n, 0.0f, C_ref.data(), m);
        ASSERT_LE(rel_fro_error(C.data(), C_ref.data(), m * nrhs), 1e-5);
    }
}

TEST_F(TestToeplitzLinOp, invalid_arguments_throw) {
    Case cs = make_case(8, 5, 401);
    vector<double> r_bad = cs.r;
    r_bad[0] = std::nextafter(r_bad[0], 2.0 * r_bad[0] + 1.0);   // one ulp off
    EXPECT_THROW(ext::ToeplitzLinOp<double>(cs.c.data(), cs.m, r_bad.data(), cs.n), RandLAPACK::Error);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    vector<double> B(64, 1.0), C(64, 0.0);
    EXPECT_THROW(Top(blas::Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, 1, cs.n, 1.0, B.data(), cs.n, 0.0, C.data(), cs.m), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::RowMajor, Op::NoTrans, Op::NoTrans, cs.m, 1, cs.n, 1.0, B.data(), 1, 0.0, C.data(), 1), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::ColMajor, Op::NoTrans, Op::Trans, cs.m, 1, cs.n, 1.0, B.data(), 1, 0.0, C.data(), cs.m), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, 1, cs.n + 1, 1.0, B.data(), cs.n + 1, 0.0, C.data(), cs.m), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, 2, cs.n, 1.0, B.data(), cs.n - 1, 0.0, C.data(), cs.m), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m + 1, 1, cs.n, 1.0, B.data(), cs.n, 0.0, C.data(), cs.m + 1), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, -1, cs.n, 1.0, B.data(), cs.n, 0.0, C.data(), cs.m), RandLAPACK::Error);
    EXPECT_THROW(Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, 2, cs.n, 1.0, B.data(), cs.n, 0.0, C.data(), cs.m - 1), RandLAPACK::Error);
    EXPECT_THROW(ext::ToeplitzLinOp<double>(cs.c.data(), 0, cs.r.data(), cs.n), RandLAPACK::Error);
}

static_assert(RandLAPACK::linops::LinearOperator<ext::ToeplitzLinOp<double>>);
static_assert(RandLAPACK::linops::LinearOperator<ext::ToeplitzLinOp<float>>);

// One operator, the batch width growing between calls: 2 columns, then kFFTBatch + 1 (a
// full batch and a tail), then 2 again.
TEST_F(TestToeplitzLinOp, batch_width_changes_on_one_operator) {
    Case cs = make_case(30, 18, 503);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    for (int64_t nrhs : {int64_t(2), ext::ToeplitzLinOp<double>::kFFTBatch + 1, int64_t(2)})
        for (Op ts : {Op::NoTrans, Op::Trans})
            run_case(Top, cs, ts, nrhs, 1.0, 0.0, false, 600 + (uint32_t)nrhs);
}

// Inside a SolveWidthScope the FFT runs at the narrowed width; the result must not change.
TEST_F(TestToeplitzLinOp, apply_inside_solve_width_scope) {
    Case cs = make_case(40, 30, 509);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    RandLAPACK::SolveWidthScope scope(10);
    run_case(Top, cs, Op::NoTrans, 1, 1.0, 0.0, false, 701);
    run_case(Top, cs, Op::Trans, 3, 1.0, 0.0, false, 702);
}

// Degenerate shapes: 1 x 1, a single row, a single column.
TEST_F(TestToeplitzLinOp, degenerate_shapes) {
    for (auto [m, n] : std::vector<std::pair<int64_t,int64_t>>{{1, 1}, {1, 7}, {7, 1}}) {
        Case cs = make_case(m, n, 521);
        ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
        for (Op ts : {Op::NoTrans, Op::Trans}) {
            run_case(Top, cs, ts, 1, 1.0, 0.0, false, 801);
            run_case(Top, cs, ts, 2, 1.0, 0.5, false, 802);
        }
    }
}

// The circulant length is the smallest even 2^a 3^b 5^c >= m + n - 1.
TEST_F(TestToeplitzLinOp, fft_length_is_the_smallest_even_5_smooth) {
    for (auto [target, L] : std::vector<std::pair<int64_t,int64_t>>{
             {1, 2}, {2, 2}, {3, 4}, {7, 8}, {11, 12}, {59, 60}, {1025, 1080},
             {76220, 76800}, {83721, 86400}, {304882, 307200}})
        EXPECT_EQ(ext::toeplitz_fft_length(target), L) << "target " << target;
    Case cs = make_case(43, 37, 11, /*with_dense=*/false);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    EXPECT_EQ(Top.L, ext::toeplitz_fft_length(43 + 37 - 1));
    EXPECT_EQ(Top.L, 80);
}

// At L >= kParallelMinLength the elementwise passes run on OpenMP threads; single and
// batched applies, both directions, must still match the dense product.
TEST_F(TestToeplitzLinOp, threaded_elementwise_passes_match_dense) {
    Case cs = make_case(40000, 50, 541);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    ASSERT_GE(Top.L, ext::ToeplitzLinOp<double>::kParallelMinLength);
    for (Op ts : {Op::NoTrans, Op::Trans}) {
        run_case(Top, cs, ts, 1, 1.0, 0.0, false, 901, 1e-12);
        run_case(Top, cs, ts, ext::ToeplitzLinOp<double>::kFFTBatch + 1, -0.5, 0.25, false, 902, 1e-12);
    }
}

// Zero right-hand-side columns is a no-op.
TEST_F(TestToeplitzLinOp, zero_columns_is_a_noop) {
    Case cs = make_case(9, 4, 531);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
    double B = 1.0, C = 3.0;
    Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, cs.m, 0, cs.n, 1.0, &B, cs.n, 0.0, &C, cs.m);
    EXPECT_EQ(C, 3.0);
}

namespace {
// The operator's products only, hiding materialize_cols, so callers take their product path.
template <typename LinOp>
struct ProductsOnly {
    using scalar_t = typename LinOp::scalar_t;
    const int64_t n_rows;
    const int64_t n_cols;
    LinOp& op;
    void operator()(Layout layout, Op trans_A, Op trans_B, int64_t m, int64_t n, int64_t k, scalar_t alpha,
                    const scalar_t* B, int64_t ldb, scalar_t beta, scalar_t* C, int64_t ldc) {
        op(layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
    }
};
} // namespace

static_assert(RandLAPACK::linops::HasMaterializeCols<ext::ToeplitzLinOp<double>>);
static_assert(!RandLAPACK::linops::HasMaterializeCols<ProductsOnly<ext::ToeplitzLinOp<double>>>);

// materialize_cols copies T's columns from c and r: exactly toeplitz(c, r)'s entries at any
// offset and width (tall, and wide with columns past row m), leaving padding rows alone; and,
// to the 1e-13 every FFT apply above meets, T applied to the same identity columns.
TEST_F(TestToeplitzLinOp, materialize_cols_reads_the_generators) {
    for (auto [m, n] : std::vector<std::pair<int64_t,int64_t>>{{37, 12}, {9, 20}, {1, 1}}) {
        Case cs = make_case(m, n, 611);
        ext::ToeplitzLinOp<double> Top(cs.c.data(), cs.m, cs.r.data(), cs.n);
        const int64_t ldb = m + 3;
        for (auto [c0, cnt] : std::vector<std::pair<int64_t,int64_t>>{{0, n}, {n / 3, n / 2}, {n - 1, 1}, {n, 0}}) {
            vector<double> buf(ldb * std::max<int64_t>(cnt, 1), -42.0);
            Top.materialize_cols(c0, cnt, buf.data(), ldb);
            for (int64_t j = 0; j < std::max<int64_t>(cnt, 1); ++j)
                for (int64_t i = 0; i < ldb; ++i)
                    ASSERT_EQ(buf[i + j * ldb], (i < m && j < cnt) ? cs.Tdense[i + (c0 + j) * m] : -42.0)
                        << m << "x" << n << " cols " << c0 << "+" << cnt << " at " << i << "," << j;
            if (cnt == 0) continue;
            vector<double> eye(n * cnt, 0.0), prod(m * cnt, 0.0), direct(m * cnt);
            for (int64_t j = 0; j < cnt; ++j) eye[(c0 + j) + j * n] = 1.0;
            Top(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, cnt, n, 1.0, eye.data(), n, 0.0, prod.data(), m);
            lapack::lacpy(lapack::MatrixType::General, m, cnt, buf.data(), ldb, direct.data(), m);
            EXPECT_LE(rel_fro_error(prod.data(), direct.data(), m * cnt), 1e-13) << m << "x" << n << " cols " << c0;
        }
    }
}

// Columns of a float operator, the threaded copy at m * count >= kParallelMinLength, and the
// argument checks.
TEST_F(TestToeplitzLinOp, materialize_cols_float_threaded_and_invalid) {
    {
        const int64_t m = 12, n = 7, ldb = m + 1;
        Case cs = make_case(m, n, 631);
        vector<float> c(cs.c.begin(), cs.c.end()), r(cs.r.begin(), cs.r.end());
        ext::ToeplitzLinOp<float> Top(c.data(), m, r.data(), n);
        vector<float> buf(ldb * 4, -42.0f);
        Top.materialize_cols(2, 4, buf.data(), ldb);
        for (int64_t j = 0; j < 4; ++j)
            for (int64_t i = 0; i < m; ++i)
                ASSERT_EQ(buf[i + j * ldb], (float)cs.Tdense[i + (2 + j) * m]) << i << "," << j;
    }
    const int64_t m = 40000, n = 50;
    Case cs = make_case(m, n, 641, /*with_dense=*/false);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), m, cs.r.data(), n);
    ASSERT_GE(m * n, ext::ToeplitzLinOp<double>::kParallelMinLength);
    vector<double> buf(m * n);
    Top.materialize_cols(0, n, buf.data(), m);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            ASSERT_EQ(buf[i + j * m], i >= j ? cs.c[i - j] : cs.r[j - i]) << i << "," << j;
    EXPECT_THROW(Top.materialize_cols(-1, 2, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(Top.materialize_cols(0, -1, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(Top.materialize_cols(n - 1, 2, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(Top.materialize_cols(0, 1, buf.data(), m - 1), RandLAPACK::Error);
    // The range check cannot overflow: start + count is never formed out of range.
    const int64_t big = std::numeric_limits<int64_t>::max();
    EXPECT_THROW(Top.materialize_cols(big, 1, buf.data(), m), RandLAPACK::Error);
    EXPECT_THROW(Top.materialize_cols(1, big, buf.data(), m), RandLAPACK::Error);
    // A single column (no parallel region) at the threaded size.
    vector<double> col(m);
    Top.materialize_cols(n - 1, 1, col.data(), m);
    for (int64_t i = 0; i < m; ++i)
        ASSERT_EQ(col[i], i >= n - 1 ? cs.c[i - (n - 1)] : cs.r[(n - 1) - i]) << i;
}

// [T; mu I] forwards materialize_cols to both blocks, so its sketch overload reads exact
// columns: it matches the sketch of the explicit stack to the sketch's own rounding, and the
// FFT product path of a stack whose top hides materialize_cols to the FFT's 1e-13.
TEST_F(TestToeplitzLinOp, vstack_reads_columns_directly) {
    namespace rll = RandLAPACK::linops;
    const int64_t m = 43, n = 17, d = 34, mr = m + n;
    const double mu = 0.5;
    Case cs = make_case(m, n, 621);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), m, cs.r.data(), n);
    ProductsOnly<ext::ToeplitzLinOp<double>> Top_prod{m, n, Top};
    rll::ScaledIdentityLinOp<double> mu_I(n, mu);
    rll::VStackLinOp<ext::ToeplitzLinOp<double>, rll::ScaledIdentityLinOp<double>> A_hat(Top, mu_I);
    rll::VStackLinOp<ProductsOnly<ext::ToeplitzLinOp<double>>, rll::ScaledIdentityLinOp<double>> A_prod(Top_prod, mu_I);
    static_assert(rll::HasMaterializeCols<decltype(A_hat)>);
    static_assert(!rll::HasMaterializeCols<decltype(A_prod)>);
    A_hat.block_size = A_prod.block_size = 5;

    vector<double> explicit_stack(mr * n, 0.0), W(mr * n);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) explicit_stack[i + j * mr] = cs.Tdense[i + j * m];
        explicit_stack[m + j + j * mr] = mu;
    }
    A_hat.materialize_cols(0, n, W.data(), mr);
    for (int64_t i = 0; i < mr * n; ++i) ASSERT_EQ(W[i], explicit_stack[i]) << i;

    RandBLAS::SparseDist SD(d, mr, 4);
    RandBLAS::SparseSkOp<double> S(SD, RNGState<>(7)), S_prod(SD, RNGState<>(7)), S_ref(SD, RNGState<>(7));
    vector<double> C(d * n), C_prod(d * n), C_ref(d * n);
    A_hat(blas::Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr, 1.0, S, 0.0, C.data(), d);
    A_prod(blas::Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr, 1.0, S_prod, 0.0, C_prod.data(), d);
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans, d, n, mr, 1.0, S_ref,
                             explicit_stack.data(), mr, 0.0, C_ref.data(), d);
    EXPECT_LE(rel_fro_error(C.data(), C_ref.data(), d * n), 4 * std::numeric_limits<double>::epsilon());
    EXPECT_LE(rel_fro_error(C.data(), C_prod.data(), d * n), 1e-13);
}

namespace {
// R and the test-mode Q of one driver on [T; mu I] (columns from materialize_cols) and on the
// same stack with T's FFT products only.
template <typename QR, typename OpHat, typename OpProd>
void compare_column_paths(OpHat& A_hat, OpProd& A_prod, double tol) {
    const int64_t m = A_hat.n_rows, n = A_hat.n_cols;
    vector<double> R(n * n, 0.0), R_prod(n * n, 0.0);
    QR qr(false, 0.0, /*test_mode=*/true), qr_prod(false, 0.0, /*test_mode=*/true);
    if constexpr (requires { qr.block_size; }) qr.block_size = qr_prod.block_size = 8;
    ASSERT_EQ(qr.call(A_hat, R.data(), n), 0);
    ASSERT_EQ(qr_prod.call(A_prod, R_prod.data(), n), 0);
    EXPECT_LE(rel_fro_error(R.data(), R_prod.data(), n * n), tol);
    EXPECT_LE(rel_fro_error(qr.Q, qr_prod.Q, m * n), tol);
}
} // namespace

// Where a driver applied [T; mu I] to identity columns (the first Gram, sCholQR3_basic's first
// product, the test-mode Q, materialize), it now reads exact columns. Those differ from the FFT
// products by the FFT's rounding (under the 1e-13 relative every apply above meets), and with
// mu = 5 here cond([T; mu I]) is small, so R and Q agree with the FFT path to 1e-12.
TEST_F(TestToeplitzLinOp, cholqr_family_reads_columns_directly) {
    namespace rll = RandLAPACK::linops;
    const int64_t m = 200, n = 30, mr = m + n;
    const double mu = 5.0, tol = 1e-12;
    Case cs = make_case(m, n, 651);
    ext::ToeplitzLinOp<double> Top(cs.c.data(), m, cs.r.data(), n);
    ProductsOnly<ext::ToeplitzLinOp<double>> Top_prod{m, n, Top};
    rll::ScaledIdentityLinOp<double> mu_I(n, mu);
    rll::VStackLinOp<ext::ToeplitzLinOp<double>, rll::ScaledIdentityLinOp<double>> A_hat(Top, mu_I);
    rll::VStackLinOp<ProductsOnly<ext::ToeplitzLinOp<double>>, rll::ScaledIdentityLinOp<double>> A_prod(Top_prod, mu_I);
    compare_column_paths<RandLAPACK::CholQR_linops<double>>(A_hat, A_prod, tol);
    compare_column_paths<RandLAPACK::CholQR2_linops<double>>(A_hat, A_prod, tol);
    compare_column_paths<RandLAPACK::sCholQR3_linops<double>>(A_hat, A_prod, tol);
    compare_column_paths<RandLAPACK::sCholQR3_linops_basic<double>>(A_hat, A_prod, tol);

    vector<double> buf(mr * n);
    RandLAPACK::materialize(A_hat, mr, n, buf.data(), mr);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < mr; ++i)
            ASSERT_EQ(buf[i + j * mr], i < m ? cs.Tdense[i + j * m] : (i - m == j ? mu : 0.0)) << i << "," << j;
}
