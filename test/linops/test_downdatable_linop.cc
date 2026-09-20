#include "rl_downdatable_linop.hh"
#include "rl_dense_linop.hh"
#include <gtest/gtest.h>

#include <vector>

// Independent scalar reference for all storage/transposition combinations.
// Nonunit alpha/beta catch applying beta twice during the low-rank correction.
TEST(TestDowndatableLinOp, MatchesExplicitResidual) {
    using namespace RandLAPACK::linops;
    constexpr int64_t rows = 3, cols = 2;
    const double a[] = {2, 3, 5, 7, 11, 13};
    const double q[] = {1, 2, 3, -1, 0, 2};
    const double bt[] = {2, -1, 1, 3};
    for (auto layout : {Layout::ColMajor, Layout::RowMajor}) {
        std::vector<double> stored(rows * cols);
        for (int64_t j = 0; j < cols; ++j)
            for (int64_t i = 0; i < rows; ++i)
                stored[layout == Layout::ColMajor ? i + rows*j : j + cols*i] = a[i + rows*j];
        DenseLinOp<double> base(rows, cols, stored.data(),
                                layout == Layout::ColMajor ? rows : cols, layout);
        const auto saved = stored;
        DowndatableLinOp<double, decltype(base)> residual(base, 2);
        residual.update(1, q, bt);
        residual.update(1, q + rows, bt + cols);
        for (auto trans_a : {Op::NoTrans, Op::Trans}) {
            const int64_t m = trans_a == Op::NoTrans ? rows : cols;
            const int64_t k = trans_a == Op::NoTrans ? cols : rows;
            // Seven RHS columns exceed both base dimensions and the old scratch capacity.
            for (const int64_t nrhs : {int64_t(1), int64_t(7)}) {
                for (auto trans_b : {Op::NoTrans, Op::Trans}) {
                    const int64_t br = trans_b == Op::NoTrans ? k : nrhs;
                    const int64_t bc = trans_b == Op::NoTrans ? nrhs : k;
                    const int64_t ldb = (layout == Layout::ColMajor ? br : bc) + 1;
                    const int64_t ldc = (layout == Layout::ColMajor ? m : nrhs) + 1;
                    std::vector<double> b(ldb * (layout == Layout::ColMajor ? bc : br), 0);
                    std::vector<double> c(ldc * (layout == Layout::ColMajor ? nrhs : m), 3);
                    auto index = [layout](int64_t i, int64_t j, int64_t ld) {
                        return layout == Layout::ColMajor ? i + ld*j : j + ld*i;
                    };
                    for (int64_t j = 0; j < nrhs; ++j)
                        for (int64_t i = 0; i < k; ++i)
                            b[trans_b == Op::NoTrans ? index(i,j,ldb) : index(j,i,ldb)] = i + 2*j + 1;
                    residual(layout, trans_a, trans_b, m, nrhs, k,
                             2.0, b.data(), ldb, -0.5, c.data(), ldc);
                    for (int64_t j = 0; j < nrhs; ++j) {
                        for (int64_t i = 0; i < m; ++i) {
                            double expected = -1.5;
                            for (int64_t l = 0; l < k; ++l) {
                                const int64_t ar = trans_a == Op::NoTrans ? i : l;
                                const int64_t ac = trans_a == Op::NoTrans ? l : i;
                                const double entry = a[ar + rows*ac] - q[ar]*bt[ac]
                                                   - q[ar + rows]*bt[ac + cols];
                                expected += 2 * entry * (l + 2*j + 1);
                            }
                            EXPECT_NEAR(c[index(i,j,ldc)], expected, 1e-12);
                        }
                    }
                }
            }
        }
        EXPECT_EQ(stored, saved);
    }
}

TEST(TestDowndatableLinOp, RejectsRankOverflowBeforeUpdating) {
    double a[] = {1, 0, 0, 1}, column[] = {1, 0};
    RandLAPACK::linops::DenseLinOp<double> base(2, 2, a, 2, Layout::ColMajor);
    RandLAPACK::linops::DowndatableLinOp<double, decltype(base)> residual(base, 1);
    residual.update(1, column, column);
    EXPECT_THROW(residual.update(1, column, column), RandLAPACK::Error);
    double output[2];
    residual(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 2, 1, 2,
             1.0, column, 2, 0.0, output, 2);
    EXPECT_DOUBLE_EQ(output[0], 0);
    EXPECT_DOUBLE_EQ(output[1], 0);
}

TEST(TestDowndatableLinOp, RejectsBufferSizeOverflow) {
    double a[16] = {};
    RandLAPACK::linops::DenseLinOp<double> base(4, 4, a, 4, Layout::ColMajor);
    using Residual = RandLAPACK::linops::DowndatableLinOp<double, decltype(base)>;
    // Four times this rank wraps to zero in a signed 64-bit product.
    EXPECT_THROW(Residual(base, int64_t(1) << 62), RandLAPACK::Error);
}

TEST(TestDowndatableLinOp, RejectsScratchOverflowBeforeApplyingBase) {
    double a[16] = {}, q[8] = {}, output = 17;
    RandLAPACK::linops::DenseLinOp<double> base(4, 4, a, 4, Layout::ColMajor);
    RandLAPACK::linops::DowndatableLinOp<double, decltype(base)> residual(base, 2);
    residual.update(2, q, q);
    // Reject before forwarding this oversized RHS to BLAS with the small buffers.
    EXPECT_THROW(residual(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                         4, int64_t(1) << 62, 4, 1.0, q, 4, 0.0, &output, 4),
                 RandLAPACK::Error);
    EXPECT_DOUBLE_EQ(output, 17);
}
