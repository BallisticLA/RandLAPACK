#include "rl_rs.hh"
#include "rl_rsvd.hh"
#include "rl_sparse_linop.hh"
#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

namespace {
using RNG = r123::Philox4x32;

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
    EXPECT_EQ(rsvd.call(op,std::sqrt(5.0),k,1e-8,u,s,v,state),6);
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
