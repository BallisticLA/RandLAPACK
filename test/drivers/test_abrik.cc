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
        printf("residual_err_custom %e\n", residual_err_custom);
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

    ABRIK.qr_exp = Subroutines::QR_explicit::cqrrt;


    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    test::test_datastructures::test_spmats::iid_sparsify_random_dense<double, r123::Philox4x32>(m, n, Layout::ColMajor, all_data.A_buff, 0.9, 0);
    RandBLAS::sparse_data::coo::dense_to_coo<double>(Layout::ColMajor, all_data.A_buff, 0.0, all_data.A);

    test_ABRIK_general<double>(b_sz, target_rank, custom_rank, all_data, ABRIK, state);
}

// ========== Adaptive mode tests ==========

// Adaptive mode converges from a small initial max_krylov_iters.
TEST_F(TestABRIK, ABRIK_adaptive_converges) {
    int64_t m    = 200;
    int64_t n    = 100;
    int64_t b_sz = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    ABRIKTestData<double> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.adaptive = true;
    ABRIK.max_krylov_iters = 4; // Start with few iterations

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);

    ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state);

    auto k = ABRIK.singular_triplets_found;
    double residual = residual_error_comp<double>(all_data, k);
    printf("adaptive_converges: residual %e, k=%ld, iters=%d\n", residual, k, ABRIK.num_krylov_iters);
    ASSERT_LE(residual, 10 * std::pow(std::numeric_limits<double>::epsilon(), 0.825));
    ASSERT_GT(ABRIK.num_krylov_iters, 4); // Should have extended beyond initial
}

// Adaptive mode with unreasonable tolerance — BK norm converges, ABRIK stops gracefully.
TEST_F(TestABRIK, ABRIK_adaptive_norm_converged) {
    int64_t m    = 200;
    int64_t n    = 100;
    int64_t b_sz = 10;
    double tol = 1e-20; // Unreachable in double precision
    auto state = RandBLAS::RNGState();

    ABRIKTestData<double> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.adaptive = true;
    ABRIK.max_krylov_iters = 4;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);

    ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state);

    // Should terminate gracefully despite unreasonable tolerance.
    auto k = ABRIK.singular_triplets_found;
    printf("adaptive_norm_converged: iters=%d, k=%ld\n", ABRIK.num_krylov_iters, k);
    ASSERT_GT(k, (int64_t)0);
    // Result should still be reasonable even though tol wasn't met.
    double residual = residual_error_comp<double>(all_data, std::min(k, (int64_t)50));
    printf("adaptive_norm_converged: residual %e\n", residual);
    ASSERT_LE(residual, 10 * std::pow(std::numeric_limits<double>::epsilon(), 0.825));
}

// Adaptive mode with a rank-deficient matrix — BK detects rank deficiency, ABRIK stops.
TEST_F(TestABRIK, ABRIK_adaptive_rank_deficient) {
    int64_t m    = 100;
    int64_t n    = 50;
    int64_t b_sz = 10;
    int64_t true_rank = 5;
    double tol = 1e-20; // Unreachable
    auto state = RandBLAS::RNGState();

    ABRIKTestData<double> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.adaptive = true;
    ABRIK.max_krylov_iters = 4;

    // Create a rank-5 matrix: A = L * R
    double* L     = new double[m * true_rank]();
    double* R_mat = new double[true_rank * n]();
    RandBLAS::DenseDist DL(m, true_rank);
    state = RandBLAS::fill_dense(DL, L, state);
    RandBLAS::DenseDist DR(true_rank, n);
    state = RandBLAS::fill_dense(DR, R_mat, state);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, true_rank,
               1.0, L, m, R_mat, true_rank, 0.0, all_data.A, m);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);
    delete[] L;
    delete[] R_mat;

    ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state);

    auto k = ABRIK.singular_triplets_found;
    printf("adaptive_rank_deficient: iters=%d, k=%ld\n", ABRIK.num_krylov_iters, k);
    ASSERT_GT(k, (int64_t)0);
}

// Adaptive mode with max_retries=1 — verifies the retry limit is respected.
TEST_F(TestABRIK, ABRIK_adaptive_max_retries) {
    int64_t m    = 200;
    int64_t n    = 100;
    int64_t b_sz = 10;
    double tol = 1e-20; // Unreachable
    auto state = RandBLAS::RNGState();

    ABRIKTestData<double> all_data(m, n);
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, false, tol);
    ABRIK.adaptive = true;
    ABRIK.max_krylov_iters = 4;
    ABRIK.adaptive_max_retries = 1;

    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    lapack::lacpy(MatrixType::General, m, n, all_data.A, m, all_data.A_buff, m);

    ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state);

    printf("adaptive_max_retries: iters=%d, k=%ld\n", ABRIK.num_krylov_iters, ABRIK.singular_triplets_found);
    // Initial call: 4 iters. After 1 retry: 4 more iters = 8 total.
    ASSERT_GT(ABRIK.num_krylov_iters, 4);
    ASSERT_LE(ABRIK.num_krylov_iters, 8);
    ASSERT_GT(ABRIK.singular_triplets_found, (int64_t)0);
}

// Adaptive mode produces comparable quality to non-adaptive with enough iterations.
TEST_F(TestABRIK, ABRIK_adaptive_matches_nonadaptive) {
    int64_t m    = 200;
    int64_t n    = 100;
    int64_t b_sz = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Generate the matrix once.
    ABRIKTestData<double> data1(m, n);
    auto state = RandBLAS::RNGState();
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, data1.A, state);
    lapack::lacpy(MatrixType::General, m, n, data1.A, m, data1.A_buff, m);

    // Copy for second run.
    ABRIKTestData<double> data2(m, n);
    lapack::lacpy(MatrixType::General, m, n, data1.A_buff, m, data2.A, m);
    lapack::lacpy(MatrixType::General, m, n, data1.A_buff, m, data2.A_buff, m);

    // Run 1: non-adaptive with generous iterations.
    auto state1 = RandBLAS::RNGState();
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK1(false, false, tol);
    ABRIK1.max_krylov_iters = 20;
    ABRIK1.call(m, n, data1.A, m, b_sz, data1.U, data1.V, data1.Sigma, state1);

    auto k1 = ABRIK1.singular_triplets_found;
    double residual1 = residual_error_comp<double>(data1, std::min(k1, (int64_t)50));

    // Run 2: adaptive with small initial iterations.
    auto state2 = RandBLAS::RNGState();
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK2(false, false, tol);
    ABRIK2.adaptive = true;
    ABRIK2.max_krylov_iters = 4;
    ABRIK2.call(m, n, data2.A, m, b_sz, data2.U, data2.V, data2.Sigma, state2);

    auto k2 = ABRIK2.singular_triplets_found;
    double residual2 = residual_error_comp<double>(data2, std::min(k2, (int64_t)50));

    printf("non-adaptive: residual %e, k=%ld, iters=%d\n", residual1, k1, ABRIK1.num_krylov_iters);
    printf("adaptive:     residual %e, k=%ld, iters=%d\n", residual2, k2, ABRIK2.num_krylov_iters);

    // Both should achieve good quality.
    ASSERT_LE(residual1, 10 * std::pow(std::numeric_limits<double>::epsilon(), 0.825));
    ASSERT_LE(residual2, 10 * std::pow(std::numeric_limits<double>::epsilon(), 0.825));
}