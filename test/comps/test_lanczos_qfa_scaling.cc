#include "rl_lanczos_qfa.hh"

#include <gtest/gtest.h>
#include <cmath>

TEST(TestLanczosQFAScaling, TinyPsdOperatorDoesNotCertifyFalseBreakdown) {
    const double eigenvalues[] = {1e-200, 4e-200};
    const double b[] = {1.0, 1.0};
    RandLAPACK::linops::DiagSymLinOp<double> A(2, eigenvalues);
    RandLAPACK::LanczosQFA<double> qfa;
    qfa.adaptive = true;
    qfa.adaptive_rtol = 1e-6;
    double out = 0.0;
    qfa.call(A, b, 2, 1, [](double x) { return std::sqrt(x); }, 2, &out);

    // The two-node Gauss rule is exact: sqrt(1e-200) + sqrt(4e-200).
    EXPECT_EQ(qfa.d_used, 2);
    EXPECT_NEAR(out / 3e-100, 1.0, 1e-12);
    // T_Radau / 1e-200 = [[2.5, 1.5], [1.5, 0.9]], with nodes 0, 3.4.
    const double radau = (5.0 / std::sqrt(3.4)) * 1e-100;
    EXPECT_NEAR(qfa.radau_val[0] / radau, 1.0, 1e-7);
    EXPECT_FALSE(qfa.all_certified);
}

TEST(TestLanczosQFAScaling, AdaptiveLargeInputKeepsFiniteQuadraticForm) {
    const double eigenvalues[] = {1e-40, 4e-40, 9e-40};
    const double b[] = {1e155, 1e155, 1e155};
    RandLAPACK::linops::DiagSymLinOp<double> A(3, eigenvalues);
    RandLAPACK::LanczosQFA<double> qfa;
    qfa.adaptive = true;
    double out = 0.0;
    qfa.call(A, b, 3, 1, [](double x) { return std::sqrt(x); }, 2, &out);

    // Independently evaluated two-node Gauss value; the exact form is 6e290.
    ASSERT_TRUE(std::isfinite(out));
    EXPECT_NEAR(out / 6.0501329353679946e290, 1.0, 1e-12);
    EXPECT_TRUE(std::isfinite(qfa.radau_val[0]));
    EXPECT_LE(qfa.radau_val[0], 6e290);
    EXPECT_GE(out, 6e290);
}
