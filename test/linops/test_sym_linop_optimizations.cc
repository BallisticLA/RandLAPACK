#include "rl_sym_linops.hh"
#include "rl_blaspp.hh"

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {

namespace linops = RandLAPACK::linops;
using blas::Layout;

// The opt-in RANDLAPACK_PERF_GEMM switch reads the whole buffer as a general matrix. An
// ExplicitSymLinOp that stores only one triangle (the documented default) must ignore it; only an
// operator whose owner declares both triangles valid may take the gemm path.
class TestExplicitSymLinOpPerfSwitch : public ::testing::Test {
protected:
    static constexpr int64_t n = 64, k = 8;
    static int set_gemm_switch(bool enabled) {
#ifdef _WIN32
        return _putenv_s("RANDLAPACK_PERF_GEMM", enabled ? "1" : "");
#else
        return enabled ? setenv("RANDLAPACK_PERF_GEMM", "1", 1)
                       : unsetenv("RANDLAPACK_PERF_GEMM");
#endif
    }
    // Upper triangle holds a symmetric matrix; the strict lower triangle holds unrelated values.
    static std::vector<double> upper_only() {
        std::vector<double> A(n * n);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < n; ++i)
                A[i + j * n] = (i <= j) ? 1.0 / (1.0 + i + j) : 1.0e3 + i;
        return A;
    }
    static std::vector<double> apply(linops::ExplicitSymLinOp<double>& op) {
        std::vector<double> B(n * k), C(n * k, 0.0);
        for (int64_t i = 0; i < n * k; ++i) B[i] = std::sin(0.1 * (double)(i + 1));
        op(Layout::ColMajor, k, 1.0, B.data(), n, 0.0, C.data(), n);
        return C;
    }
    void TearDown() override { EXPECT_EQ(set_gemm_switch(false), 0); }
};

TEST_F(TestExplicitSymLinOpPerfSwitch, GemmSwitchIgnoredForOneTriangleStorage) {
    auto A = upper_only();
    linops::ExplicitSymLinOp<double> op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    ASSERT_EQ(set_gemm_switch(false), 0);
    auto ref = apply(op);
    ASSERT_EQ(set_gemm_switch(true), 0);
    auto got = apply(op);
    for (int64_t i = 0; i < n * k; ++i) ASSERT_EQ(got[i], ref[i]) << "entry " << i;
}

TEST_F(TestExplicitSymLinOpPerfSwitch, GemmSwitchHonouredWhenBothTrianglesDeclared) {
    auto A = upper_only();   // deliberately inconsistent lower triangle: the gemm path must read it
    linops::ExplicitSymLinOp<double> op(n, blas::Uplo::Upper, A.data(), n, Layout::ColMajor);
    op.both_triangles = true;
    ASSERT_EQ(set_gemm_switch(false), 0);
    auto ref = apply(op);
    ASSERT_EQ(set_gemm_switch(true), 0);
    auto got = apply(op);
    double diff = 0;
    for (int64_t i = 0; i < n * k; ++i) diff = std::max(diff, std::abs(got[i] - ref[i]));
    EXPECT_GT(diff, 1.0) << "with both_triangles set, the switch should use gemm on the full buffer";
}

} // namespace
