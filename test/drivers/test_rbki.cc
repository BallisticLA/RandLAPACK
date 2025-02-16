#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include "../RandLAPACK/RandBLAS/test/test_datastructures/test_spmats/common.hh"
#include <fstream>
#include <gtest/gtest.h>


class TestRBKI : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct RBKITestData {
        int64_t row;
        int64_t col;
        T* A;
        T* A_buff;
        T* U;
        T* VT; // RBKI returns V'
        T* Sigma;
        T* U_cpy;
        T* VT_cpy;

        RBKITestData(int64_t m, int64_t n)
        {
            A      = new T[m * n]();
            A_buff = new T[m * n]();
            U      = new T[m * n]();
            VT     = new T[n * n]();
            Sigma  = new T[n]();
            U_cpy  = new T[m * n]();
            VT_cpy = new T[n * n]();
            row   = m;
            col   = n;
        }

        ~RBKITestData() {
            delete[] A;
            delete[] A_buff;
            delete[] U;
            delete[] VT;
            delete[] Sigma;
            delete[] U_cpy;
            delete[] VT_cpy;
        }
    };

    template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
    struct RBKITestDataSparse {
        int64_t row;
        int64_t col;
        SpMat A;
        T*  A_buff;
        T*  U;
        T*  VT; // RBKI returns V'
        T*  Sigma;
        T*  U_cpy;
        T*  VT_cpy;

        RBKITestDataSparse(int64_t m, int64_t n) :
        A(m, n)
        {
            A_buff = new T[m * n]();
            U      = new T[m * n]();
            VT     = new T[n * n]();
            Sigma  = new T[n]();
            U_cpy  = new T[m * n]();
            VT_cpy = new T[n * n]();
            row    = m;
            col    = n;
        }

        ~RBKITestDataSparse() {
            delete[] A_buff;
            delete[] U;
            delete[] VT;
            delete[] Sigma;
            delete[] U_cpy;
            delete[] VT_cpy;
        }
    };

    // This routine computes the residual norm error, consisting of two parts (one of which) vanishes
    // in exact precision. Target_rank defines size of U, V as returned by RBKI; custom_rank <= target_rank.
    template <typename T, typename TestData>
    static T
    residual_error_comp(TestData &all_data, int64_t custom_rank) {
        auto m = all_data.row;
        auto n = all_data.col;

        T* U_cpy_dat = all_data.U_cpy;
        T* VT_cpy_dat = all_data.VT_cpy;

        lapack::lacpy(MatrixType::General, m, n, all_data.U, m, U_cpy_dat, m);
        lapack::lacpy(MatrixType::General, n, n, all_data.VT, n, VT_cpy_dat, n);

        // AV - US
        // Scale columns of U by S
        for (int i = 0; i < custom_rank; ++i)
            blas::scal(m, all_data.Sigma[i], &U_cpy_dat[m * i], 1);

        // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, m, custom_rank, n, 1.0, all_data.A_buff, m, all_data.VT, n, -1.0, U_cpy_dat, m);


        // A'U - VS
        // Scale columns of V by S
        // Since we have VT, we will be scaling its rows
        // The data is, however, stored in a column-major format, so it is a bit weird.
        for (int i = 0; i < custom_rank; ++i)
            blas::scal(n, all_data.Sigma[i], &VT_cpy_dat[i], n);
        // Compute A'U(:, 1:custom_rank) - VS(1:custom_rank).
        // We will actually have to perform U' * A - Sigma * VT.
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, custom_rank, n, m, 1.0, all_data.U, m, all_data.A_buff, m, -1.0, VT_cpy_dat, n);

        T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, U_cpy_dat, m);
        T nrm2 = lapack::lange(Norm::Fro, custom_rank, n, VT_cpy_dat, n);

        return std::hypot(nrm1, nrm2);
    }


    template <typename T, typename RNG, typename TestData, typename alg_type>
    static void test_RBKI_general(
        int64_t b_sz,
        int64_t target_rank,
        int64_t custom_rank,
        TestData &all_data,
        alg_type &RBKI,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        RBKI.max_krylov_iters = (int) ((target_rank * 2) / b_sz);

        RBKI.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.VT, all_data.Sigma, state);
        // Compute singular values via a deterministic method

        T residual_err_custom = residual_error_comp<T>(all_data, custom_rank);
        printf("residual_err_custom %e\n", residual_err_custom);
        ASSERT_LE(residual_err_custom, 10 * std::pow(std::numeric_limits<T>::epsilon(), 0.825));
    }
};

TEST_F(TestRBKI, RBKI_basic) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    RBKITestData<double> all_data(m, n);
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, false, tol);
    RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();
    RBKI.num_threads_min = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);

    test_RBKI_general<double>(b_sz, target_rank, custom_rank, all_data, RBKI, state);
}

TEST_F(TestRBKI, RBKI_sparse_csc) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    RBKITestDataSparse<double, RandBLAS::sparse_data::CSCMatrix<double>> all_data(m, n);
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, false, tol);
    RBKI.num_threads_min = 1;
    RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_RBKI_general<double>(b_sz, target_rank, custom_rank, all_data, RBKI, state);
}

TEST_F(TestRBKI, RBKI_sparse_csr) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    RBKITestDataSparse<double, RandBLAS::sparse_data::CSRMatrix<double>> all_data(m, n);
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, false, tol);
    RBKI.num_threads_min = 1;
    RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::csr::dense_to_csr<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_RBKI_general<double>(b_sz, target_rank, custom_rank, all_data, RBKI, state);
}

TEST_F(TestRBKI, RBKI_sparse_coo) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    RBKITestDataSparse<double, RandBLAS::sparse_data::COOMatrix<double>> all_data(m, n);
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, false, tol);
    RBKI.num_threads_min = 1;
    RBKI.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::coo::dense_to_coo<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_RBKI_general<double>(b_sz, target_rank, custom_rank, all_data, RBKI, state);
}
