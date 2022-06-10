#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void test_QB2_general(int64_t m, int64_t n, int64_t k, int64_t block_sz, T tol, int64_t mat_type, uint32_t seed) {

        printf("\n|========================TEST QB2 GENERAL BEGIN========================|\n");

        using namespace blas;
        using namespace lapack;
        
        int64_t size = m * n;

        // For running QB
        std::vector<T> A(size, 0.0);

        std::vector<T> Q(m * k, 0.0);
        std::vector<T> B(k * n, 0.0);
        std::vector<T> B_cpy(k * n, 0.0);

        // For results comparison
        std::vector<T> A_hat(size, 0.0);
        std::vector<T> A_k(size, 0.0);
        std::vector<T> A_cpy (m * n, 0.0);
        std::vector<T> A_cpy_2 (m * n, 0.0);

        // For low-rank SVD
        std::vector<T> s(n, 0.0);
        std::vector<T> S(n * n, 0.0);
        std::vector<T> U(m * n, 0.0);
        std::vector<T> VT(n * n, 0.0);

        T* A_dat = A.data();
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* B_cpy_dat = B_cpy.data();

        T* A_hat_dat = A_hat.data();
        T* A_k_dat = A_k.data();
        T* A_cpy_dat = A_cpy.data();
        T* A_cpy_2_dat = A_cpy_2.data();

        T* U_dat = U.data();
        T* s_dat = s.data();
        T* S_dat = S.data();
        T* VT_dat = VT.data();

        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);
        
        //char name1[] = "A";
        //RandBLAS::util::print_colmaj(m, n, A.data(), name1);

        // Create a copy of the original matrix
        std::copy(A_dat, A_dat + size, A_cpy_dat);
        std::copy(A_dat, A_dat + size, A_cpy_2_dat);

        int termination = RandLAPACK::comps::qb::qb2_safe<T>(
        m,
        n,
        A,
        k, // Here, serves as a backup termination criteria
        block_sz,
        tol,
        2,
        1,
        Q, // m by k
        B, // k by n
        seed
        );

        if (termination == 1)
        {
            printf("Input matrix of zero entries.\n");
            EXPECT_TRUE(true);
            return;
        }
        else if (termination == 2)
        {
            printf("Early termination due to unexpected error accumulation.\n");
        }
        else if (termination == 3)
        {
            printf("Reached the expected rank without achieving the specified tolerance.\n");
        }
        else if (termination == 0)
        {
            printf("Expected tolerance reached.\n");
        }

        printf("Inner dimension of QB: %-25ld\n", k);

        std::vector<T> Ident(k * k, 0.0);
        T* Ident_dat = Ident.data();
        // Generate a reference identity
        RandLAPACK::comps::util::eye<T>(k, k, Ident); 

        // Buffer for testing B
        std::copy(B_dat, B_dat + (k * n), B_cpy_dat);
        
        //char name_3[] = "QB1  output Q";
        //RandBLAS::util::print_colmaj(m, k, Q.data(), name_3);

        //char name_4[] = "QB1  output B";
        //RandBLAS::util::print_colmaj(k, n, B.data(), name_4);

        // A_hat = Q * B
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, 1.0, Q_dat, m, B_dat, k, 0.0, A_hat_dat, m);
        // TEST 1: A = A - Q * B = 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k, -1.0, Q_dat, m, B_dat, k, 1.0, A_dat, m);

        // TEST 2: B - Q'A = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, n, m, -1.0, Q_dat, m, A_cpy_2_dat, m, 1.0, B_cpy_dat, k);

        // TEST 3: Q'Q = I = 0
        gemm<T>(Layout::ColMajor, Op::Trans, Op::NoTrans, k, k, m, -1.0, Q_dat, m, Q_dat, m, 1.0, Ident_dat, k);

        // Get low-rank SVD
        gesdd(Job::SomeVec, m, n, A_cpy_dat, m, s_dat, U_dat, m, VT_dat, n);
        // buffer zero vector
        std::vector<T> z_buf(n, 0.0);
        T* z_buf_dat = z_buf.data();
        // zero out the trailing singular values
        std::copy(z_buf_dat, z_buf_dat + (n - k), s_dat + k);
        RandLAPACK::comps::util::diag(n, n, s, S);

        //char name_u[] = "U";
        //RandBLAS::util::print_colmaj(m, n, U.data(), name_u);

        //char name_s[] = "s";
        //RandBLAS::util::print_colmaj(n, 1, s.data(), name_s);

        //char name_vt[] = "VT";
        //RandBLAS::util::print_colmaj(n, n, VT.data(), name_vt);

        //char name_S[] = "S";
        //RandBLAS::util::print_colmaj(n, n, S.data(), name_S);

        // TEST 4: Below is A_k - A_hat = A_k - QB
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, U_dat, m, S_dat, n, 1.0, A_k_dat, m);
        // A_k * VT -  A_hat == 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, n, 1.0, A_k_dat, m, VT_dat, n, -1.0, A_hat_dat, m);

        // Test 1 Output
        T norm_test_1 = lange(Norm::Fro, m, n, A_dat, m);
        printf("FRO NORM OF A - QB:    %.12f\n", norm_test_1);
        //ASSERT_NEAR(norm_test_1, 0, 1e-10);

        // Test 2 Output
        T norm_test_2 = lange(Norm::Fro, k, n, B_cpy_dat, k);
        printf("FRO NORM OF B - Q'A:   %.12f\n", norm_test_2);
        //ASSERT_NEAR(norm_test_2, 0, 1e-10);

        // Test 3 Output
        T norm_test_3 = lapack::lange(lapack::Norm::Fro, k, k, Ident_dat, k);
        printf("FRO NORM OF Q'Q - I:   %.12f\n", norm_test_3);
        //ASSERT_NEAR(norm_test_3, 0, 1e-10);

        // Test 4 Output
        T norm_test_4 = lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB:  %.12f\n", norm_test_4);
        //ASSERT_NEAR(norm_test_4, 0, 1e-10);
    }

//Varying tol, k = min(m, n)
template <typename T>
    static void test_QB2_overest_k(int64_t m, int64_t n, int64_t k, int64_t block_sz, T tol, int64_t mat_type, uint32_t seed) {
        
        printf("\n|========================TEST QB2 OVEREST K BEGIN========================|\n");

        using namespace blas;
        using namespace lapack;
        
        int64_t size = m * n;
        int64_t k_est = std::min(m, n);

        // For running QB
        std::vector<T> A(size, 0.0);
        std::vector<T> Q(m * k_est, 0.0);
        std::vector<T> B(k_est * n, 0.0);
        // For results comparison
        std::vector<T> A_hat(size, 0.0);

        T* A_dat = A.data();
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* A_hat_dat = A_hat.data();

        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);
        
        //char name1[] = "A";
        //RandBLAS::util::print_colmaj(m, n, A_dat, name1);

        // pre-compute norm
        T norm_A = lange(Norm::Fro, m, n, A_dat, m);

        // Immediate termination criteria
        if(norm_A == 0.0)
        {
            return;
        }

        int termination = RandLAPACK::comps::qb::qb2_safe<T>(
        m,
        n,
        A,
        k_est, // Here, serves as a backup termination criteria
        block_sz,
        tol,
        2,
        1,
        Q, // m by k
        B, // k by n
        seed
        );

        if (termination == 1)
        {
            printf("Input matrix of zero entries.\n");
            //Terminate
            EXPECT_TRUE(true);
        }
        else if (termination == 2)
        {
            printf("Early termination due to unexpected error accumulation.\n");
        }
        else if (termination == 3)
        {
            printf("Reached the expected rank without achieving the specified tolerance.\n");
        }
        else if (termination == 0)
        {
            printf("Expected tolerance reached.\n");
        }

        printf("Inner dimension of QB: %ld\n", k_est);
        
        //char name_3[] = "QB1  output Q";
        //RandBLAS::util::print_colmaj(m, k_est, Q_dat, name_3);

        //char name_4[] = "QB1  output B";
        //RandBLAS::util::print_colmaj(k_est, n, B_dat, name_4);

        // A_hat = Q * B
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_est, 1.0, Q_dat, m, B_dat, k_est, 0.0, A_hat_dat, m);
        // TEST 1: A = A - Q * B = 0
        gemm<T>(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, n, k_est, -1.0, Q_dat, m, B_dat, k_est, 1.0, A_dat, m);

        T norm_test_1 = lange(Norm::Fro, m, n, A_dat, m);
        if(tol == 0.0)
        {
            // Test Zero Tol Output
            printf("FRO NORM OF A - QB:    %-23.12f\n", norm_test_1);
            ASSERT_NEAR(norm_test_1, 0, 1e-12);
        }
        else
        {
            // Test Nonzero Tol Output
            printf("FRO NORM OF A - QB:    %-23.12f\n", norm_test_1);
            printf("FRO NORM OF A:    %-23.12f\n", norm_A);
            EXPECT_TRUE(norm_test_1 <= (tol * norm_A));
        }
    }
};


TEST_F(TestQB, SimpleTest)
{
    //for (uint32_t seed : {0, 1, 2})
    //{
        // Testing rank-deficient matrices with exp decay, increasing clock size, normal termination
        //test_QB2<double>(1000, 1000, 100, 1, 0.0000000001, 1, 0);
        test_QB2_general<double>(100, 100, 50, 5, 0.0000000001, 1, 0);
        test_QB2_general<double>(100, 100, 50, 5, 0.0000000001, 4, 0);
        //test_QB2<double>(1000, 1000, 100, 10, 0.0000000001, 1, 0);
        //test_QB2<double>(1000, 1000, 100, 20, 0.0000000001, 1, 0);
        //test_QB2<double>(1000, 1000, 100, 50, 0.0000000001, 1, 0);
        // 74 is when conditioning becomes an issue
        //test_QB2_general<double>(1000, 1000, 100, 73, 0.0000000001, 1, 0);
        //test_QB2<double>(5000, 1000, 500, 10, 0.0000000001, 1, 0);

        // test zero tol
        test_QB2_overest_k<double>(100, 100, 10, 5, 0.0, 1, 0);
        // test nonzero tol
        test_QB2_overest_k<double>(100, 100, 10, 5, 0.1, 1, 0);
    //}
}