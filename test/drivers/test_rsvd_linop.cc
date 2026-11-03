#include "rl_rs.hh"
#include "rl_rsvd.hh"
#include "rl_sparse_linop.hh"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
using RNG = RandBLAS::DefaultRNG;
using RSVD = RandLAPACK::RSVD<double, RNG>;

template <typename T>
void check_sparse_rsvd() {
    // An exactly rank-three sparse matrix. A missing factor or zero result
    // cannot pass reconstruction; block size two also exercises a short tail.
    constexpr int64_t m = 8, n = 6, rank = 3;
    std::vector<T> a(m*n, 0);
    a[0] = 4; a[1 + m] = 2; a[2 + 2*m] = 1;
    RandBLAS::sparse_data::CSCMatrix<T> csc(m,n);
    RandBLAS::sparse_data::csc::dense_to_csc<T>(Layout::ColMajor, a.data(), T(0), csc);
    const std::vector<T> saved_values(csc.vals, csc.vals + csc.nnz);
    const std::vector<int64_t> saved_rows(csc.rowidxs, csc.rowidxs + csc.nnz);
    const std::vector<int64_t> saved_columns(csc.colptr, csc.colptr + n + 1);
    RandLAPACK::linops::SparseLinOp<decltype(csc)> op(m,n,csc);
    const T norm = std::sqrt(T(21));
    const T bound = 200 * std::numeric_limits<T>::epsilon();
    for (int64_t passes : {0, 1, 2, 3}) {
        RandLAPACK::HQRQ<T> orth(false,false);
        RandLAPACK::RS<T,RNG> rs(orth,passes,1,false,false);
        RandLAPACK::RF<T,RNG> rf(rs,orth,false,false);
        RandLAPACK::QB<T,RNG> qb(rf,orth,false,true);
        RandLAPACK::RSVD<T,RNG> rsvd(qb,2);
        auto state = RandBLAS::RNGState<RNG>();
        int64_t k = rank;
        T *u = nullptr, *s = nullptr, *v = nullptr;
        ASSERT_EQ(rsvd.call(op,norm,k,bound,u,s,v,state),0);
        ASSERT_EQ(k,rank);
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                T reconstructed = 0;
                for (int64_t l = 0; l < k; ++l)
                    reconstructed += u[i + m*l] * s[l] * v[j + n*l];
                EXPECT_NEAR(reconstructed,a[i + m*j],bound);
            }
        }
        EXPECT_TRUE(std::equal(saved_values.begin(), saved_values.end(), csc.vals));
        EXPECT_TRUE(std::equal(saved_rows.begin(), saved_rows.end(), csc.rowidxs));
        EXPECT_TRUE(std::equal(saved_columns.begin(), saved_columns.end(), csc.colptr));
        std::free(u); std::free(s); std::free(v);
    }
}

struct OtherRows : RandLAPACK::RowSketcher<double,RNG> {
    int call(int64_t, int64_t, const double*&, int64_t,
             double*&, RandBLAS::RNGState<RNG>&) override { return 1; }
};
struct OtherRangeFinder : RandLAPACK::RangeFinder<double,RNG> {
    int call(int64_t, int64_t, const double*, int64_t,
             double*, RandBLAS::RNGState<RNG>&) override { return 1; }
};
struct OtherQB : RandLAPACK::QBalg<double,RNG> {
    int call(int64_t, int64_t, double*, int64_t&, int64_t, double,
             double*&, double*&, RandBLAS::RNGState<RNG>&) override { return 6; }
};
struct FailingOrth : RandLAPACK::Stabilization<double> {
    int call(int64_t,int64_t,double*) override { return 1; }
};

// These stabilizers deliberately return unusable bases while reporting success,
// so the real QB orthogonality checks must detect the numerical failure.
struct ZeroOrth : RandLAPACK::Stabilization<double> {
    int call(int64_t m, int64_t n, double* q) override {
        std::fill(q, q + m*n, 0.0);
        return 0;
    }
};
struct FirstCoordinateOrth : RandLAPACK::Stabilization<double> {
    int call(int64_t m, int64_t n, double* q) override {
        std::fill(q, q + m*n, 0.0);
        q[0] = 1.0;
        return 0;
    }
};
struct CyclingCoordinateOrth : RandLAPACK::Stabilization<double> {
    int64_t next_coordinate = 0;
    int call(int64_t m, int64_t n, double* q) override {
        std::fill(q, q + m*n, 0.0);
        q[next_coordinate++ % m] = 1.0;
        return 0;
    }
};
struct ScaledOrth : RandLAPACK::Stabilization<double> {
    int call(int64_t m, int64_t n, double* q) override {
        for (int64_t i = 0; i < m*n; ++i) q[i] *= 4.0;
        return 0;
    }
};

template <typename Exception>
struct ThrowingOrth : RandLAPACK::Stabilization<double> {
    int call(int64_t, int64_t, double*) override { throw Exception(); }
};
struct OperatorFailure {};
struct ThrowingOperator {
    using scalar_t = double;
    const int64_t n_rows = 2, n_cols = 2;
    void operator()(Layout, Op, Op, int64_t, int64_t, int64_t, double,
                    const double*, int64_t, double, double*, int64_t) {
        throw OperatorFailure();
    }
};

// Nonnull inputs also check that failure neither frees nor writes caller data.
struct OutputSentinels {
    double values[3] = {11.0, 22.0, 33.0};
    double *u = values, *s = values + 1, *v = values + 2;
    void expect_unchanged() const {
        EXPECT_EQ(u, values);
        EXPECT_EQ(s, values + 1);
        EXPECT_EQ(v, values + 2);
        EXPECT_EQ(values[0], 11.0);
        EXPECT_EQ(values[1], 22.0);
        EXPECT_EQ(values[2], 33.0);
    }
    ~OutputSentinels() {
        if (u != values) std::free(u);
        if (s != values + 1) std::free(s);
        if (v != values + 2) std::free(v);
    }
};
} // namespace

TEST(TestRSVDLinearOperator, SparseFloat) { check_sparse_rsvd<float>(); }
TEST(TestRSVDLinearOperator, SparseDouble) { check_sparse_rsvd<double>(); }

TEST(TestRSVDLinearOperator, RejectsUnsupportedAlgorithmObjects) {
    double a[] = {2, 0, 0, 1}, q[2];
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    RandLAPACK::HQRQ<double> orth(false,false);
    auto state = RandBLAS::RNGState<RNG>();
    OtherRows rows;
    RandLAPACK::RF<double,RNG> rf(rows,orth,false,false);
    EXPECT_THROW(RandLAPACK::rf_linop(rf,op,1,q,state), RandLAPACK::Error);
    OtherRangeFinder range_finder;
    // Custom dense algorithms still satisfy the original constructor API.
    RandLAPACK::QB<double,RNG> qb(range_finder,orth,false,false);
    int64_t k = 1;
    double *u = nullptr, *s = nullptr, *v = nullptr;
    EXPECT_THROW(qb.call(op,k,1,1e-8,std::sqrt(5.0),u,v,state), RandLAPACK::Error);
    EXPECT_EQ(u,nullptr);
    EXPECT_EQ(v,nullptr);
    OtherQB other_qb;
    RandLAPACK::RSVD<double,RNG> rsvd(other_qb,1);
    EXPECT_THROW(rsvd.call(op,std::sqrt(5.0),k,1e-8,u,s,v,state), RandLAPACK::Error);
}

TEST(TestRSVDLinearOperator, ReturnsStabilizationFailureWithoutCallingSVD) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    FailingOrth orth;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,false);
    RandLAPACK::RSVD<double,RNG> rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 1;
    double *u = nullptr, *s = nullptr, *v = nullptr;
    EXPECT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,u,s,v,state),RSVD::QBSubroutineFailure);
    EXPECT_EQ(k,0);
    EXPECT_EQ(u,nullptr);
    EXPECT_EQ(s,nullptr);
    EXPECT_EQ(v,nullptr);
}

TEST(TestRSVDLinearOperator, RejectsInvalidInputsBeforeAllocating) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    RandLAPACK::HQRQ<double> orth(false,false);
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,false);
    RandLAPACK::RSVD<double,RNG> rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 0;
    double *u = nullptr, *s = nullptr, *v = nullptr;
    EXPECT_THROW(rsvd.call(op,1.0,k,1e-8,u,s,v,state), RandLAPACK::Error);
    k = 3;
    EXPECT_THROW(rsvd.call(op,1.0,k,1e-8,u,s,v,state), RandLAPACK::Error);
    k = 1;
    EXPECT_THROW(rsvd.call(op,0.0,k,1e-8,u,s,v,state), RandLAPACK::Error);
    EXPECT_THROW(qb.call(op,k,0,1e-8,1.0,u,v,state), RandLAPACK::Error);
    rs.passes_per_stab = 0;
    EXPECT_THROW(rsvd.call(op,1.0,k,1e-8,u,s,v,state), RandLAPACK::Error);
    EXPECT_EQ(u,nullptr);
    EXPECT_EQ(s,nullptr);
    EXPECT_EQ(v,nullptr);
}

TEST(TestRSVDLinearOperator, PreservesOutputsOnBlockOrthogonalityFailure) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    ZeroOrth orth;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,true);
    RSVD rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 1;
    OutputSentinels out;
    EXPECT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,out.u,out.s,out.v,state),
              RSVD::BlockOrthogonalityFailure);
    EXPECT_EQ(k,0);
    out.expect_unchanged();
}

TEST(TestRSVDLinearOperator, PreservesOutputsOnBasisOrthogonalityFailure) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    FirstCoordinateOrth orth;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,true);
    RSVD rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 2;
    OutputSentinels out;
    EXPECT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,out.u,out.s,out.v,state),
              RSVD::BasisOrthogonalityFailure);
    EXPECT_EQ(k,1);
    out.expect_unchanged();
}

TEST(TestRSVDLinearOperator, PreservesOutputsOnReorthogonalizationFailure) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    RandLAPACK::HQRQ<double> orth(false,false);
    FailingOrth fail;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,fail,false,true);
    RSVD rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 2;
    OutputSentinels out;
    EXPECT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,out.u,out.s,out.v,state),
              RSVD::QBSubroutineFailure);
    EXPECT_EQ(k,1);
    out.expect_unchanged();
}

TEST(TestRSVDLinearOperator, ReturnsUsableApproximationWhenRankLimitIsReached) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    FirstCoordinateOrth orth;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,true);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 1;
    double *q = nullptr, *bt = nullptr;
    ASSERT_EQ(qb.call(op,k,1,1e-8,std::sqrt(5.0),q,bt,state),3);
    std::free(q); std::free(bt);
    RSVD rsvd(qb,1);
    double *u = nullptr, *s = nullptr, *v = nullptr;
    ASSERT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,u,s,v,state),RSVD::Success);
    ASSERT_EQ(k,1);
    EXPECT_DOUBLE_EQ(s[0],2.0);
    EXPECT_DOUBLE_EQ(u[0]*s[0]*v[0],2.0);
    EXPECT_DOUBLE_EQ(u[0]*s[0]*v[1],0.0);
    EXPECT_DOUBLE_EQ(u[1]*s[0]*v[0],0.0);
    EXPECT_DOUBLE_EQ(u[1]*s[0]*v[1],0.0);
    std::free(u); std::free(s); std::free(v);
}

TEST(TestRSVDLinearOperator, ReturnsUsablePrefixWhenQBErrorEstimateGrows) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    CyclingCoordinateOrth orth;
    ScaledOrth unstable_orth;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,unstable_orth,false,false);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 2;
    double *q = nullptr, *bt = nullptr;
    // The first block is e_1. An unstable reorthogonalization scales e_2 by
    // four, making the error estimate grow; QB must retain the first block.
    ASSERT_EQ(qb.call(op,k,1,1e-8,std::sqrt(5.0),q,bt,state),2);
    ASSERT_EQ(k,1);
    std::free(q); std::free(bt);
    k = 2;
    RSVD rsvd(qb,1);
    double *u = nullptr, *s = nullptr, *v = nullptr;
    ASSERT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,u,s,v,state),RSVD::Success);
    ASSERT_EQ(k,1);
    EXPECT_DOUBLE_EQ(s[0],2.0);
    EXPECT_DOUBLE_EQ(u[0]*s[0]*v[0],2.0);
    EXPECT_DOUBLE_EQ(u[0]*s[0]*v[1],0.0);
    EXPECT_DOUBLE_EQ(u[1]*s[0]*v[0],0.0);
    EXPECT_DOUBLE_EQ(u[1]*s[0]*v[1],0.0);
    std::free(u); std::free(s); std::free(v);
}

TEST(TestRSVDLinearOperator, PreservesOutputsOnInvalidUse) {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    RandLAPACK::HQRQ<double> orth(false,false);
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,true);
    RSVD rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 1;
    OutputSentinels out;
    EXPECT_THROW(rsvd.call(op,0.0,k,1e-8,out.u,out.s,out.v,state),RandLAPACK::Error);
    EXPECT_THROW(rsvd.call(op,1.0,k,-1.0,out.u,out.s,out.v,state),RandLAPACK::Error);
    EXPECT_THROW(rsvd.call(op,1.0,k,std::numeric_limits<double>::quiet_NaN(),
                           out.u,out.s,out.v,state),RandLAPACK::Error);
    OtherQB other_qb;
    RSVD unsupported(other_qb,1);
    EXPECT_THROW(unsupported.call(op,1.0,k,1e-8,out.u,out.s,out.v,state),RandLAPACK::Error);
    out.expect_unchanged();
}

TEST(TestRSVDLinearOperator, PreservesOutputsAndPropagatesOperatorException) {
    ThrowingOperator op;
    RandLAPACK::HQRQ<double> orth(false,false);
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,true);
    RSVD rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 1;
    OutputSentinels out;
    EXPECT_THROW(rsvd.call(op,1.0,k,1e-8,out.u,out.s,out.v,state),OperatorFailure);
    out.expect_unchanged();
}

template <typename Exception>
void check_component_exception() {
    double a[] = {2, 0, 0, 1};
    RandLAPACK::linops::DenseLinOp<double> op(2,2,a,2,Layout::ColMajor);
    ThrowingOrth<Exception> orth;
    RandLAPACK::RS<double,RNG> rs(orth,0,1,false,false);
    RandLAPACK::RF<double,RNG> rf(rs,orth,false,false);
    RandLAPACK::QB<double,RNG> qb(rf,orth,false,true);
    RSVD rsvd(qb,1);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = 1;
    OutputSentinels out;
    EXPECT_THROW(rsvd.call(op,std::sqrt(5.0),k,1e-8,out.u,out.s,out.v,state),Exception);
    out.expect_unchanged();
}

TEST(TestRSVDLinearOperator, PreservesOutputsAndPropagatesAllocationException) {
    check_component_exception<std::bad_alloc>();
}

TEST(TestRSVDLinearOperator, PreservesOutputsAndPropagatesLAPACKException) {
    check_component_exception<lapack::Error>();
}

namespace {
// A = diag(r, r-1, ..., 1, 0, ..., 0), an exactly rank-r n x n matrix; returns ||A||_F.
double diagonal_low_rank(int64_t n, int64_t r, std::vector<double>& a) {
    a.assign(n * n, 0.0);
    double norm_sq = 0;
    for (int64_t i = 0; i < r; ++i) {
        a[i + n * i] = double(r - i);
        norm_sq += a[i + n * i] * a[i + n * i];
    }
    return std::sqrt(norm_sq);
}

double max_reconstruction_error(int64_t n, int64_t k, const std::vector<double>& a,
                                const double* q, const double* bt) {
    double err = 0;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < n; ++i) {
            double qb = 0;
            for (int64_t l = 0; l < k; ++l)
                qb += q[i + n * l] * bt[j + n * l];
            const double d = std::abs(qb - a[i + n * j]);
            if (std::isnan(d))
                return d;
            err = std::max(err, d);
        }
    return err;
}
} // namespace

// QB's storage follows the rank it reaches, not the rank bound it is given.
TEST(TestQBLinearOperator, StorageScalesWithTheReachedRank) {
    constexpr int64_t n = 200, rank = 6, block = 2;
    std::vector<double> a;
    const double norm = diagonal_low_rank(n, rank, a);
    RandLAPACK::linops::DenseLinOp<double> op(n, n, a.data(), n, Layout::ColMajor);
    RandLAPACK::HQRQ<double> orth(false, false);
    RandLAPACK::RS<double,RNG> rs(orth, 1, 1, false, false);
    RandLAPACK::RF<double,RNG> rf(rs, orth, false, false);
    RandLAPACK::QB<double,RNG> qb(rf, orth, false, true);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = n;
    double *q = nullptr, *bt = nullptr;
    ASSERT_EQ(qb.call(op, k, block, 1e-6, norm, q, bt, state), 0);
    EXPECT_EQ(k, rank);
    EXPECT_LE(qb.result_capacity, std::max(block, 2 * k));
    EXPECT_LT(max_reconstruction_error(n, k, a, q, bt), 1e-10);
    std::free(q); std::free(bt);
}

// One-column blocks grow the storage to 1, 2 and 4 columns; the fourth block fits without growing.
TEST(TestQBLinearOperator, AcceptsBlocksAcrossCapacityDoublings) {
    constexpr int64_t n = 50, rank = 4;
    std::vector<double> a;
    const double norm = diagonal_low_rank(n, rank, a);
    RandLAPACK::linops::DenseLinOp<double> op(n, n, a.data(), n, Layout::ColMajor);
    RandLAPACK::HQRQ<double> orth(false, false);
    RandLAPACK::RS<double,RNG> rs(orth, 2, 1, false, false);
    RandLAPACK::RF<double,RNG> rf(rs, orth, false, false);
    RandLAPACK::QB<double,RNG> qb(rf, orth, false, true);
    auto state = RandBLAS::RNGState<RNG>();
    int64_t k = n;
    double *q = nullptr, *bt = nullptr;
    ASSERT_EQ(qb.call(op, k, 1, 1e-6, norm, q, bt, state), 0);
    EXPECT_EQ(k, rank);
    EXPECT_EQ(qb.result_capacity, 4);
    EXPECT_LT(max_reconstruction_error(n, k, a, q, bt), 1e-10);
    std::free(q); std::free(bt);
}

// Operator QB deflates through borrowed views of its growing factors; on a matrix whose spectrum decays it
// reaches the accuracy of dense QB, which deflates an explicit copy of A.
TEST(TestQBLinearOperator, MatchesDenseQBOnADecayingSpectrum) {
    constexpr int64_t m = 120, n = 80, block = 4;
    std::vector<double> a(m * n);
    RandBLAS::DenseDist dist(m, n);
    RandBLAS::RNGState<RNG> seed(11);
    RandBLAS::fill_dense(dist, a.data(), seed);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            a[i + m * j] *= std::pow(0.7, double(j));
    const double norm = lapack::lange(lapack::Norm::Fro, m, n, a.data(), m);
    auto relative_error = [&](int64_t k, const double* q, const double* bt) {
        std::vector<double> r(a);
        blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, m, n, k,
                   -1.0, q, m, bt, n, 1.0, r.data(), m);
        return lapack::lange(lapack::Norm::Fro, m, n, r.data(), m) / norm;
    };
    RandLAPACK::HQRQ<double> orth(false, false);
    RandLAPACK::RS<double,RNG> rs(orth, 1, 1, false, false);
    RandLAPACK::RF<double,RNG> rf(rs, orth, false, false);
    RandLAPACK::QB<double,RNG> qb(rf, orth, false, true);

    RandLAPACK::linops::DenseLinOp<double> op(m, n, a.data(), m, Layout::ColMajor);
    auto state = RandBLAS::RNGState<RNG>(3);
    int64_t k_op = n;
    double *q_op = nullptr, *bt_op = nullptr;
    ASSERT_EQ(qb.call(op, k_op, block, 1e-6, norm, q_op, bt_op, state), 0);

    std::vector<double> a_dense(a);
    state = RandBLAS::RNGState<RNG>(3);
    int64_t k_dense = n;
    double *q_dense = nullptr, *bt_dense = nullptr;
    ASSERT_EQ(qb.call(m, n, a_dense.data(), k_dense, block, 1e-6, q_dense, bt_dense, state), 0);

    EXPECT_LE(std::abs(k_op - k_dense), block);
    EXPECT_LT(relative_error(k_op, q_op, bt_op), 1e-5);
    EXPECT_LT(relative_error(k_dense, q_dense, bt_dense), 1e-5);
    std::free(q_op); std::free(bt_op); std::free(q_dense); std::free(bt_dense);
}
