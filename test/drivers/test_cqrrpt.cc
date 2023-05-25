#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <gtest/gtest.h>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestCQRRPT : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct CQRRPTTestData {
        std::vector<T> A;
        std::vector<T> R;
        std::vector<int64_t> J;
        std::vector<T> A_cpy;

        CQRRPTTestData(int64_t m, int64_t n) :
        A(m * n, 0.0), 
        J(n, 0.0),  
        A_cpy(m * n, 0.0) 
        {}
    };

    template <typename T, typename RNG>
    static void computational_helper(int64_t m, int64_t n, CQRRPTTestData<T>& all_data) {
        lapack::lacpy(MatrixType::General, m, n, all_data.A.data(), m, all_data.A_cpy.data(), m);
    }

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T, typename RNG>
    static void test_CQRRPT_general(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, RandBLAS::base::RNGState<RNG> state, uint64_t no_hqrrp, CQRRPTTestData<T>& all_data) {

        RandLAPACK::CQRRPT<T, RNG> CQRRPT(false, false, state, tol);
        CQRRPT.nnz = nnz;
        CQRRPT.num_threads = 4;
        CQRRPT.no_hqrrp = no_hqrrp;

        CQRRPT.call(m, n, all_data.A, d, all_data.R, all_data.J);

        T* A_dat = all_data.A.data();
        T* A_cpy_dat = all_data.A_cpy.data();
        T* R_dat = all_data.R.data();
        k = CQRRPT.rank;

        RandLAPACK::util::col_swap(m, n, n, all_data.A_cpy, all_data.J);

        // AP - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_cpy_dat, m);

        T norm_test = lapack::lange(Norm::Fro, m, n, A_cpy_dat, m);
        printf("FRO NORM OF AP - QR:  %e\n", norm_test);
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, CQRRPT_full_rank_no_hqrrp)
{
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 200;
    int64_t d = 400;
    int64_t nnz = 2;
    uint64_t no_hqrrp = 1; 
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();
    CQRRPTTestData<double> all_data(m, n);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    computational_helper<double, r123::Philox4x32>(m, n, all_data);
    test_CQRRPT_general<double, r123::Philox4x32>(m, n, k, d, nnz, tol, state, no_hqrrp, all_data);
}

TEST_F(TestCQRRPT, CQRRPT_low_rank_with_hqrrp)
{
    int64_t m = 10000;
    int64_t n = 200;
    int64_t k = 100;
    int64_t d = 400;
    int64_t nnz = 2;
    uint64_t no_hqrrp = 0; 
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.75);
    auto state = RandBLAS::base::RNGState();
    CQRRPTTestData<double> all_data(m, n);

    RandLAPACK::util::gen_mat_type<double, r123::Philox4x32>(m, n, all_data.A, k, state, std::make_tuple(0, 2, false));
    computational_helper<double, r123::Philox4x32>(m, n, all_data);
    test_CQRRPT_general<double, r123::Philox4x32>(m, n, k, d, nnz, tol, state, no_hqrrp, all_data);
}