#include "rl_downdatable_linop.hh"
#include "rl_dense_linop.hh"
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace {

using RandLAPACK::linops::DenseLinOp;
using RandLAPACK::linops::DowndatableLinOp;

// Store logical entries in either layout, with one padding entry per row/column.
struct PaddedMatrix {
    Layout layout;
    int64_t rows;
    int64_t cols;
    int64_t ld;
    std::vector<double> values;

    PaddedMatrix(Layout layout, int64_t rows, int64_t cols, double initial = 0)
        : layout(layout), rows(rows), cols(cols),
          ld((layout == Layout::ColMajor ? rows : cols) + 1),
          values(ld * (layout == Layout::ColMajor ? cols : rows), initial) {}

    double& operator()(int64_t row, int64_t col) {
        return values[layout == Layout::ColMajor ? row + ld * col : col + ld * row];
    }
};

constexpr int64_t base_rows = 3;
constexpr int64_t base_cols = 2;
constexpr double base_entries[base_rows][base_cols] = {{2, 7}, {3, 11}, {5, 13}};

// E = [1 -1; 2 0; 3 2], F = [2 1; -1 3]. Neither has orthogonal columns.
// Thus A - E F^T = [1 11; -1 13; -3 10]. Keep this reference independent
// of the factor multiplication performed by DowndatableLinOp.
constexpr std::array<double, 6> E = {1, 2, 3, -1, 0, 2};
constexpr std::array<double, 4> F = {2, -1, 1, 3};
constexpr double residual_entries[base_rows][base_cols] = {{1, 11}, {-1, 13}, {-3, 10}};

PaddedMatrix make_base_matrix(Layout layout) {
    PaddedMatrix matrix(layout, base_rows, base_cols);
    for (int64_t row = 0; row < base_rows; ++row)
        for (int64_t col = 0; col < base_cols; ++col)
            matrix(row, col) = base_entries[row][col];
    return matrix;
}

double rhs_entry(int64_t row, int64_t col) {
    return row + 2 * col + 1;
}

PaddedMatrix make_rhs(Layout layout, Op trans_b, int64_t inner_dim, int64_t nrhs) {
    const bool transpose = trans_b == Op::Trans;
    PaddedMatrix rhs(layout, transpose ? nrhs : inner_dim,
                     transpose ? inner_dim : nrhs);
    for (int64_t row = 0; row < inner_dim; ++row) {
        for (int64_t col = 0; col < nrhs; ++col) {
            if (transpose)
                rhs(col, row) = rhs_entry(row, col);
            else
                rhs(row, col) = rhs_entry(row, col);
        }
    }
    return rhs;
}

PaddedMatrix reference_result(Op trans_a, int64_t inner_dim,
                             double alpha, double beta, PaddedMatrix output) {
    for (int64_t row = 0; row < output.rows; ++row) {
        for (int64_t col = 0; col < output.cols; ++col) {
            double product = 0;
            for (int64_t inner = 0; inner < inner_dim; ++inner) {
                const double residual_entry = trans_a == Op::NoTrans
                    ? residual_entries[row][inner] : residual_entries[inner][row];
                product += residual_entry * rhs_entry(inner, col);
            }
            output(row, col) = alpha * product + beta * output(row, col);
        }
    }
    return output;
}

using ApplicationCase = std::tuple<Layout, Op, Op, int64_t>;

std::string application_case_name(const ::testing::TestParamInfo<ApplicationCase>& info) {
    const auto [layout, trans_a, trans_b, nrhs] = info.param;
    std::string name = layout == Layout::ColMajor ? "ColMajor" : "RowMajor";
    name += trans_a == Op::NoTrans ? "ANoTrans" : "ATrans";
    name += trans_b == Op::NoTrans ? "BNoTrans" : "BTrans";
    name += nrhs == 1 ? "SingleRhs" : "WideRhs";
    return name;
}

class TestDowndatableApplication : public ::testing::TestWithParam<ApplicationCase> {};

TEST_P(TestDowndatableApplication, MatchesExplicitResidualAndPreservesInputs) {
    const auto [layout, trans_a, trans_b, nrhs] = GetParam();
    const int64_t output_rows = trans_a == Op::NoTrans ? base_rows : base_cols;
    const int64_t inner_dim = trans_a == Op::NoTrans ? base_cols : base_rows;
    // Nonunit coefficients detect applying beta twice during the correction.
    constexpr double alpha = 2.0;
    constexpr double beta = -0.5;
    constexpr double initial_output = 3.0;

    auto stored_base = make_base_matrix(layout);
    auto rhs = make_rhs(layout, trans_b, inner_dim, nrhs);
    PaddedMatrix output(layout, output_rows, nrhs, initial_output);
    const auto saved_base = stored_base.values;
    const auto saved_rhs = rhs.values;
    const auto expected = reference_result(trans_a, inner_dim, alpha, beta, output);

    DenseLinOp<double> base(base_rows, base_cols, stored_base.values.data(),
                            stored_base.ld, layout);
    DowndatableLinOp<double, decltype(base)> residual(base, 2);
    residual.update(1, E.data(), F.data());
    residual.update(1, E.data() + base_rows, F.data() + base_cols);
    residual(layout, trans_a, trans_b, output_rows, nrhs, inner_dim,
             alpha, rhs.values.data(), rhs.ld, beta, output.values.data(), output.ld);

    // Include output padding in the comparison to catch writes outside C.
    for (std::size_t index = 0; index < output.values.size(); ++index)
        EXPECT_NEAR(output.values[index], expected.values[index], 1e-12)
            << "output buffer index " << index;
    EXPECT_EQ(stored_base.values, saved_base);
    EXPECT_EQ(rhs.values, saved_rhs);
}

INSTANTIATE_TEST_SUITE_P(
    LayoutsTransposesAndRhsWidths, TestDowndatableApplication,
    ::testing::Combine(
        ::testing::Values(Layout::ColMajor, Layout::RowMajor),
        ::testing::Values(Op::NoTrans, Op::Trans),
        ::testing::Values(Op::NoTrans, Op::Trans),
        // Seven RHS columns exceed both base dimensions and the old scratch capacity.
        ::testing::Values(int64_t(1), int64_t(7))),
    application_case_name);

TEST(TestDowndatableLinOp, RejectsRankOverflowBeforeUpdating) {
    const double identity[] = {1, 0, 0, 1};
    const double first_column[] = {1, 0};
    DenseLinOp<double> base(2, 2, identity, 2, Layout::ColMajor);
    DowndatableLinOp<double, decltype(base)> residual(base, 1);
    residual.update(1, first_column, first_column);

    EXPECT_THROW(residual.update(1, first_column, first_column), RandLAPACK::Error);

    // The rejected update must leave I - e_1 e_1^T unchanged.
    double output[2];
    residual(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 1, 2,
             1.0, first_column, 2, 0.0, output, 2);
    EXPECT_DOUBLE_EQ(output[0], 0);
    EXPECT_DOUBLE_EQ(output[1], 0);
}

TEST(TestDowndatableLinOp, RejectsBufferSizeOverflow) {
    const double zero_matrix[16] = {};
    DenseLinOp<double> base(4, 4, zero_matrix, 4, Layout::ColMajor);
    using Residual = DowndatableLinOp<double, decltype(base)>;

    // Four times this rank wraps to zero in a signed 64-bit product.
    constexpr int64_t excessive_rank = int64_t(1) << 62;
    EXPECT_THROW(Residual(base, excessive_rank), RandLAPACK::Error);
}

TEST(TestDowndatableLinOp, RejectsScratchOverflowBeforeApplyingBase) {
    const double zero_matrix[16] = {};
    const double zero_factors[8] = {};
    double output = 17;
    DenseLinOp<double> base(4, 4, zero_matrix, 4, Layout::ColMajor);
    DowndatableLinOp<double, decltype(base)> residual(base, 2);
    residual.update(2, zero_factors, zero_factors);

    // Reject before forwarding this oversized RHS to BLAS with the small buffers.
    constexpr int64_t excessive_nrhs = int64_t(1) << 62;
    EXPECT_THROW(residual(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                         4, excessive_nrhs, 4, 1.0, zero_factors, 4, 0.0, &output, 4),
                 RandLAPACK::Error);
    EXPECT_DOUBLE_EQ(output, 17);
}

} // namespace
