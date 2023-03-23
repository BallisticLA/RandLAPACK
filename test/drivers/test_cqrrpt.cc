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

    /// General test for CQRRPT:
    /// Computes QR factorzation, and computes A[:, J] - QR.
    template <typename T>
    static void test_CQRRPT_general(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed, uint64_t no_hqrrp) {

        printf("|================================TEST CQRRPT GENERAL BEGIN===============================|\n");

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);

        std::vector<T> R;
        std::vector<int64_t> J(n, 0);

        RandLAPACK::util::gen_mat_type(m, n, A, k, seed, mat_type);

        std::vector<T> A_hat(size, 0.0);
        std::copy(A.data(), A.data() + size, A_hat.data());

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();

        RandLAPACK::CQRRPT<T> CQRRPT(false, false, seed, tol);
        CQRRPT.nnz = nnz;
        CQRRPT.num_threads = 4;
        CQRRPT.no_hqrrp = no_hqrrp;

        CQRRPT.call(m, n, A, d, R, J);

        A_dat = A.data();
        T* R_dat = R.data();
        k = CQRRPT.rank;

        RandLAPACK::util::col_swap(m, n, n, A_hat, J);

        // AP - QR
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_hat_dat, m);

        T norm_test = lapack::lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF AP - QR:  %e\n", norm_test);

        printf("|=================================TEST CQRRPT GENERAL END================================|\n");
    }
};

// Note: If Subprocess killed exception -> reload vscode
TEST_F(TestCQRRPT, SimpleTest)
{
    test_CQRRPT_general<double>(10000, 200, 200, 400, 2, std::pow(std::numeric_limits<double>::epsilon(), 0.75), std::make_tuple(0, 2, false), 2, 1);
    test_CQRRPT_general<double>(10000, 200, 100, 400, 2, std::pow(std::numeric_limits<double>::epsilon(), 0.75), std::make_tuple(0, 2, false), 2, 0);
}
