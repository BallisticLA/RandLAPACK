#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

using namespace RandLAPACK::comps::util;
using namespace RandLAPACK::comps::orth;
using namespace RandLAPACK::drivers::cholqrcp;


/*
Beware of improper rank k defining.
*/
class TestCholQRCP : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void test_CholQRCP1_general(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|================================TEST CholQRCP GENERAL BEGIN===============================|\n");

        using namespace blas;
        using namespace lapack;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);

        std::vector<T> R;
        std::vector<int64_t> J(n, 0);

        gen_mat_type<T>(m, n, A, k, seed, mat_type);

        std::vector<T> A_hat(size, 0.0);
        std::copy(A.data(), A.data() + size, A_hat.data());

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();

        CholQRCP<T> CholQRCP(false, false, seed, tol, use_cholqrcp1);
        CholQRCP.nnz = nnz;
        CholQRCP.num_threads = 32;

        CholQRCP.call(m, n, A, d, R, J);

        A_dat = A.data();
        T* R_dat = R.data();
        int64_t* J_dat = J.data();
        k = CholQRCP.rank;

        col_swap(m, n, n, A_hat, J);

        // AP - QR
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_hat_dat, m);

        T norm_test = lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF AP - QR:  %e\n", norm_test);
        
        printf("|=================================TEST CholQRCP GENERAL END================================|\n");
    }


    template <typename T>
    static void test_CholQRCP1_piv_dist(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|================================TEST CholQRCP BENCH PARAM BEGIN===============================|\n");

        using namespace blas;
        using namespace lapack;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);

        std::vector<T> R;
        std::vector<int64_t> J(n, 0);

        // Random Gaussian test matrix
        //RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        gen_mat_type<T>(m, n, A, k, seed, mat_type);

        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_1(size, 0.0);
        std::vector<int64_t> J_1(n, 0.0);
        std::vector<T> tau_1(n, 0.0);

        std::copy(A.data(), A.data() + size, A_hat.data());
        std::copy(A.data(), A.data() + size, A_1.data());

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();

        CholQRCP<T> CholQRCP(false, false, seed, tol, use_cholqrcp1);
        CholQRCP.nnz = nnz;
        CholQRCP.num_threads = 32;

        CholQRCP.call(m, n, A, d, R, J);

        A_dat = A.data();
        T* R_dat = R.data();
        int64_t* J_dat = J.data();
        k = CholQRCP.rank;

        col_swap(m, n, n, A_hat, J);

        // AP - QR
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_hat_dat, m);

        T norm_test = lange(Norm::Fro, m, n, A_hat_dat, m);

        // Deterministic alg
        geqp3(m, n, A_1.data(), m, J_1.data(), tau_1.data());

        printf("\nMatrix size: %ld by %ld.\n", m, n);
        printf("Embedding size: %ld.\n", d);
        printf("Number of nonzeros per column in SASO: %ld\n", nnz);
        printf("FRO NORM OF AP - QR: %e\n", norm_test);
        
        // Something potentially wrong with this
        printf("Levenshtein Distance of permutation vectors: %d\n", levenstein_dist(n, J, J_1));

        for(int i = 0; i < n; ++i)
        {
            printf("%ld, %ld\n", J[i], J_1[i]);
        }
        
        printf("|=================================TEST CholQRCP BENCH PARAM END================================|\n");
    }

    template <typename T>
    static void test_CholQRCP1_R_dist(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|================================TEST CholQRCP BENCH PARAM BEGIN===============================|\n");

        using namespace blas;
        using namespace lapack;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);

        std::vector<T> R;
        std::vector<int64_t> J(n, 0);

        // Random Gaussian test matrix
        //RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        gen_mat_type<T>(m, n, A, k, seed, mat_type);

        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_1(size, 0.0);
        std::vector<T> A_2(size, 0.0);

        // For QRP
        std::vector<int64_t> J_1(n, 0.0);
        std::vector<T> tau_1(n, 0.0);

        // For SVD
        std::vector<T> s(n, 0.0);
        std::vector<T> U(m * n, 0.0);
        std::vector<T> VT(n * n, 0.0);

        // Diag buffers
        std::vector<T> r(n, 0.0);
        std::vector<T> r_1(n, 0.0);

        std::copy(A.data(), A.data() + size, A_hat.data());
        std::copy(A.data(), A.data() + size, A_1.data());
        std::copy(A.data(), A.data() + size, A_2.data());

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();

        CholQRCP<T> CholQRCP(false, false, seed, tol, use_cholqrcp1);
        CholQRCP.nnz = nnz;
        CholQRCP.num_threads = 32;

        CholQRCP.call(m, n, A, d, R, J);

        A_dat = A.data();
        T* R_dat = R.data();
        int64_t* J_dat = J.data();
        k = CholQRCP.rank;

        col_swap(m, n, n, A_hat, J);

        // AP - QR
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_hat_dat, m);

        T norm_test = lange(Norm::Fro, m, n, A_hat_dat, m);

        // Deterministic QRP
        geqp3(m, n, A_1.data(), m, J_1.data(), tau_1.data());

        // Deterministic SVD
        gesdd(Job::SomeVec, m, n, A_2.data(), m, s.data(), U.data(), m, VT.data(), n);

        extract_diag(k, n, k, R, r);
        extract_diag(m, n, n, A_1, r_1);

        printf("\nMatrix size: %ld by %ld.\n", m, n);
        printf("Embedding size: %ld.\n", d);
        printf("Number of nonzeros per column in SASO: %ld\n", nnz);
        printf("FRO NORM OF AP - QR: %e\n", norm_test);

        printf("Ratios of diagonal R to true singular values:\n");
        printf("CHOLQRCP GEQP3\n");



        std::ofstream ofs;
        ofs.open("../../testing/r_ratios.dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();


        std::fstream file("../../testing/r_ratios.dat", std::fstream::app);

        for(int i = 0; i < n; ++i)
        {
            file << std::abs(r[i] / s[i]) << "  " << std::abs(r_1[i] / s[i]) << "\n";
            //file << s[i] << "  " << s[i] << "\n";
        }

        printf("|=================================TEST CholQRCP BENCH PARAM END================================|\n");
    }

    
    template <typename T>
    static void test_CholQRCP1_lowrank_dist(int64_t m, int64_t n, int64_t k, int64_t d, int64_t nnz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|================================TEST CholQRCP BENCH PARAM BEGIN===============================|\n");

        using namespace blas;
        using namespace lapack;

        int64_t size = m * n;
        std::vector<T> A(size, 0.0);

        std::vector<T> R;
        std::vector<int64_t> J(n, 0);

        // Random Gaussian test matrix
        //RandBLAS::dense_op::gen_rmat_norm<T>(m, n, A.data(), seed);
        gen_mat_type<T>(m, n, A, k, seed, mat_type);

        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_1(size, 0.0);
        std::vector<T> A_2(size, 0.0);

        // For QRP
        std::vector<int64_t> J_1(n, 0.0);
        std::vector<T> tau_1(n, 0.0);
        std::vector<T> R_1(n * n, 0.0);


        std::copy(A.data(), A.data() + size, A_hat.data());
        std::copy(A.data(), A.data() + size, A_1.data());
        std::copy(A.data(), A.data() + size, A_2.data());

        T* A_dat = A.data();
        T* A_hat_dat = A_hat.data();

        CholQRCP<T> CholQRCP(false, false, seed, tol, use_cholqrcp1);
        CholQRCP.nnz = nnz;
        CholQRCP.num_threads = 32;

        CholQRCP.call(m, n, A, d, R, J);

        A_dat = A.data();
        T* R_dat = R.data();
        int64_t* J_dat = J.data();
        k = CholQRCP.rank;

        col_swap(m, n, n, A_hat, J);

        // AP - QR
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, A_dat, m, R_dat, k, -1.0, A_hat_dat, m);

        T norm_test = lange(Norm::Fro, m, n, A_hat_dat, m);

        // Deterministic QRP, explicit extraction of R
        geqp3(m, n, A_1.data(), m, J_1.data(), tau_1.data());
        get_U(m, n, A_1, R_1);


        printf("\nMatrix size: %ld by %ld.\n", m, n);
        printf("Embedding size: %ld.\n", d);
        printf("Number of nonzeros per column in SASO: %ld\n", nnz);
        printf("FRO NORM OF AP - QR: %e\n", norm_test);

        printf("Ratios of diagonal R to true singular values:\n");
        printf("CHOLQRCP GEQP3\n");



        std::ofstream ofs;
        ofs.open("../../testing/norm_r_ratios.dat", std::ofstream::out | std::ofstream::trunc);
        ofs.close();


        std::fstream file("../../testing/norm_r_ratios.dat", std::fstream::app);

        //std::vector<T> data(n, 0.0);

        /*
        for(int i = 1; i < n; ++i)
        {
            row_resize(k, n, A_1, k - i);
            row_resize(k, n, R, k - i);

            T norm_geqp3 = lange(Norm::Fro, k - i, k - i, A_1.data(), k - i);
            T norm_cholqrcp = lange(Norm::Fro, k - i, k - i, R.data(), k - i);

            data[i] = norm_geqp3 / norm_cholqrcp;
        }

        for(int i = 1; i < n; ++i)
        {
            file << data[n - i] << "\n";
        }
        */
        std::vector<T> z_buf(k, 0.0);
        T* R_1_dat = R_1.data();
        T* z_buf_dat = z_buf.data();
        R_dat = R.data();

        
        // || R_qp3[k:, :] || / || R_cqrcp[k:, :] ||
        for(int i = 1; i < n - 1; ++i)
        {

            for(int j = 0; j < n; ++j)
            {
                copy(i, &z_buf_dat[0], 1, &R_1_dat[n * j], 1);
                copy(i, &z_buf_dat[0], 1, &R_dat[k * j], 1);
            }

            //char name[] = "R";
            //RandBLAS::util::print_colmaj(k, n, R.data(), name);

            //char name1[] = "R_1";
            //RandBLAS::util::print_colmaj(n, n, R_1.data(), name1);

            T norm_geqp3 = lange(Norm::Fro, n, n, R_1.data(), n);
            T norm_cholqrcp = lange(Norm::Fro, k, n, R.data(), k);

            file << norm_geqp3 / norm_cholqrcp << "\n";
        }

        printf("|=================================TEST CholQRCP BENCH PARAM END================================|\n");
    }
};

// New test - compare diag entries of R matrices

/*Subprocess killed exception - reload vscode*/
TEST_F(TestCholQRCP, SimpleTest)
{ 
    for (uint32_t seed : {2})//, 1, 2})
    {
        //test_CholQRCP1_general<double>(10000, 200, 200, 400, 2, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);
        //test_CholQRCP1_general<double>(10000, 200, 100, 400, 2, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);

        // Tests copying sizes used in benchmarks
        //test_CholQRCP1_piv_dist<double>(5, 4, 4, 4, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);
        //test_CholQRCP1_piv_dist<double>(131072, 2000, 2000, 2000, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);
        //test_CholQRCP1_piv_dist<double>(131072, 2000, 2000, 4000, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);
        //test_CholQRCP1_piv_dist<double>(131072, 2000, 2000, 6000, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);
        //test_CholQRCP1_piv_dist<double>(131072, 2000, 2000, 20000, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);
        //test_CholQRCP1_piv_dist<double>(131072, 5000, 5000, 20000, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2, false), seed);


        //test_CholQRCP1_R_dist<double>(131072, 2000, 2000, 2000, 4, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2.5, false), seed);
        test_CholQRCP1_lowrank_dist<double>(131072, 2000, 2000, 2000, 1, std::pow(1.0e-16, 0.75), std::make_tuple(0, 2.5, false), seed);
    }
}
