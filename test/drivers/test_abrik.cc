#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include "../RandLAPACK/RandBLAS/test/test_datastructures/test_spmats/common.hh"
#include <fstream>
#include <gtest/gtest.h>

using Subroutines = RandLAPACK::ABRIKSubroutines;

class TestABRIK : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    struct ABRIKTestData {
        int64_t row;
        int64_t col;
        T* A;
        T* A_buff;
        T* U;
        T* V; 
        T* Sigma;
        T* U_cpy;
        T* V_cpy;

        ABRIKTestData(int64_t m, int64_t n)
        {
            A      = new T[m * n]();
            A_buff = new T[m * n]();
            U      = nullptr;
            V      = nullptr;
            Sigma  = nullptr;
            U_cpy  = nullptr;
            V_cpy  = nullptr;
            row    = m;
            col    = n;
        }

        ~ABRIKTestData() {
            delete[] A;
            delete[] A_buff;
            delete[] U;
            delete[] V;
            delete[] Sigma;
            delete[] U_cpy;
            delete[] V_cpy;
        }
    };

    template <typename T, RandBLAS::sparse_data::SparseMatrix SpMat>
    struct ABRIKTestDataSparse {
        int64_t row;
        int64_t col;
        SpMat A;
        T*  A_buff;
        T*  U;
        T*  V; 
        T*  Sigma;
        T*  U_cpy;
        T*  V_cpy;

        ABRIKTestDataSparse(int64_t m, int64_t n) :
        A(m, n)
        {
            A_buff = new T[m * n]();
            U      = nullptr;
            V      = nullptr;
            Sigma  = nullptr;
            U_cpy  = nullptr;
            V_cpy  = nullptr;
            row    = m;
            col    = n;
        }

        ~ABRIKTestDataSparse() {
            delete[] A_buff;
            delete[] U;
            delete[] V;
            delete[] Sigma;
            delete[] U_cpy;
            delete[] V_cpy;
        }
    };

    // This routine computes the residual norm error, consisting of two parts (one of which) vanishes
    // in exact precision. Target_rank defines size of U, V as returned by ABRIK; custom_rank <= target_rank.
    template <typename T, typename TestData>
    static T
    residual_error_comp(TestData &all_data, int64_t custom_rank) {
        auto m = all_data.row;
        auto n = all_data.col;

        all_data.U_cpy = new T[m * custom_rank]();
        all_data.V_cpy = new T[n * custom_rank]();

        lapack::lacpy(MatrixType::General, m, custom_rank, all_data.U, m, all_data.U_cpy, m);
        lapack::lacpy(MatrixType::General, n, custom_rank, all_data.V, n, all_data.V_cpy, n);

        // AV - US
        // Scale columns of U by S
        for (int i = 0; i < custom_rank; ++i)
            blas::scal(m, all_data.Sigma[i], &all_data.U_cpy[m * i], 1);

        // Compute AV(:, 1:custom_rank) - SU(1:custom_rank)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, custom_rank, n, 1.0, all_data.A_buff, m, all_data.V, n, -1.0, all_data.U_cpy, m);

        // A'U - VS
        // Scale columns of V by S
        for (int i = 0; i < custom_rank; ++i)
            blas::scal(n, all_data.Sigma[i], &all_data.V_cpy[i * n], 1);
        // Compute A'U(:, 1:custom_rank) - VS(1:custom_rank).
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, custom_rank, m, 1.0, all_data.A_buff, m, all_data.U, m, -1.0, all_data.V_cpy, n);

        T nrm1 = lapack::lange(Norm::Fro, m, custom_rank, all_data.U_cpy, m);
        T nrm2 = lapack::lange(Norm::Fro, n, custom_rank, all_data.V_cpy, n);

        return std::hypot(nrm1, nrm2);
    }


    template <typename T, typename RNG, typename TestData, typename alg_type>
    static void test_ABRIK_general(
        int64_t b_sz,
        int64_t target_rank,
        int64_t custom_rank,
        TestData &all_data,
        alg_type &ABRIK,
        RandBLAS::RNGState<RNG> &state) {

        auto m = all_data.row;
        auto n = all_data.col;
        ABRIK.max_krylov_iters = (int) ((target_rank * 2) / b_sz);

        ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state);
        
        T residual_err_custom = residual_error_comp<T>(all_data, custom_rank);
        std::cout << "residual_err_custom " << std::scientific << residual_err_custom << "\n";
        ASSERT_LE(residual_err_custom, 10 * std::pow(std::numeric_limits<T>::epsilon(), 0.825));
    }
};


TEST_F(TestABRIK, ABRIK_basic1) {
    int64_t m           = 10;
    int64_t n           = 5;
    int64_t b_sz        = 1;
    int64_t target_rank = 5;
    int64_t custom_rank = 3;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestData<double> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();
    ABRIK.num_threads_min = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}

TEST_F(TestABRIK, ABRIK_basic) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestData<double> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();
    ABRIK.num_threads_min = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}

TEST_F(TestABRIK, ABRIK_sparse_csc) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestDataSparse<double, RandBLAS::sparse_data::CSCMatrix<double>> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.num_threads_min = 1;
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::csc::dense_to_csc<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}

TEST_F(TestABRIK, ABRIK_sparse_csr) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestDataSparse<double, RandBLAS::sparse_data::CSRMatrix<double>> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.num_threads_min = 1;
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::csr::dense_to_csr<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}

TEST_F(TestABRIK, ABRIK_sparse_coo) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestDataSparse<double, RandBLAS::sparse_data::COOMatrix<double>> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.num_threads_min = 1;
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::coo::dense_to_coo<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}

TEST_F(TestABRIK, ABRIK_sparse_coo_cqrrt) {
    int64_t m           = 400;
    int64_t n           = 200;
    int64_t b_sz        = 10;
    int64_t target_rank = 200;
    int64_t custom_rank = 100;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestDataSparse<double, RandBLAS::sparse_data::COOMatrix<double>> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.num_threads_min = 1;
    ABRIK.qr_exp = Subroutines::QR_explicit::cqrrt;
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::coo::dense_to_coo<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}
