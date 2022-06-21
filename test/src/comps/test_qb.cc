#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

//#include "gnuplot-iostream.h"
#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

    template <typename T>
    static void test_QB2_general(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|========================TEST QB2 GENERAL BEGIN========================|\n");
        using namespace blas;
        using namespace lapack;
        
        // For running QB
        std::vector<T> A(m * n, 0.0);
        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);

        int64_t size = m * n;
        // Adjust the expected rank
        if(k == 0)
        {
            k = std::min(m, n);
        }

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
        p,
        1,
        Q, // m by k
        B, // k by n
        seed
        );

        switch(termination)
        {
            case 1:
                printf("\nTERMINATED VIA: Input matrix of zero entries.\n");
                EXPECT_TRUE(true);
                return;
                break;
            case 2:
                printf("\nTERMINATED VIA: Early termination due to unexpected error accumulation.\n");
                break;
            case 3:
                printf("\nTERMINATED VIA: Reached the expected rank without achieving the specified tolerance.\n");
                break;
            case 5:
                printf("\nTERMINATED VIA: Lost orthonormality of Q.\n");
                //EXPECT_TRUE(true);
                //return;
                break;
            case 0:
               printf("\nTERMINATED VIA: Expected tolerance reached.\n");
                break;
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
        RandLAPACK::comps::util::diag(n, n, s, n, S);

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
        printf("FRO NORM OF A - QB:    %e\n", norm_test_1);
        //ASSERT_NEAR(norm_test_1, 0, 1e-10);

        // Test 2 Output
        T norm_test_2 = lange(Norm::Fro, k, n, B_cpy_dat, k);
        printf("FRO NORM OF B - Q'A:   %e\n", norm_test_2);
        //ASSERT_NEAR(norm_test_2, 0, 1e-10);

        // Test 3 Output
        T norm_test_3 = lapack::lange(lapack::Norm::Fro, k, k, Ident_dat, k);
        printf("FRO NORM OF Q'Q - I:   %e\n", norm_test_3);
        //ASSERT_NEAR(norm_test_3, 0, 1e-10);

        // Test 4 Output
        T norm_test_4 = lange(Norm::Fro, m, n, A_hat_dat, m);
        printf("FRO NORM OF A_k - QB:  %e\n", norm_test_4);
        //ASSERT_NEAR(norm_test_4, 0, 1e-10);
        printf("|=========================TEST QB2 GENERAL END=========================|\n");
    }

//Varying tol, k = min(m, n)
template <typename T>
    static void test_QB2_k_eq_min(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|=====================TEST QB2 K = min(M, N) BEGIN=====================|\n");

        using namespace blas;
        using namespace lapack;
        
        // For running QB
        std::vector<T> A(m * n, 0.0);
        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);

        int64_t size = m * n;
        // Adjust the expected rank
        if(k == 0)
        {
            k = std::min(m, n);
        }
        
        int64_t k_est = std::min(m, n);

        std::vector<T> Q(m * k_est, 0.0);
        std::vector<T> B(k_est * n, 0.0);
        // For results comparison
        std::vector<T> A_hat(size, 0.0);

        T* A_dat = A.data();
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* A_hat_dat = A_hat.data();

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
        p,
        1,
        Q, // m by k
        B, // k by n
        seed
        );
        
        switch(termination)
        {
            case 1:
                printf("\nTERMINATED VIA: Input matrix of zero entries.\n");
                EXPECT_TRUE(true);
                return;
                break;
            case 2:
                printf("\nTERMINATED VIA: Early termination due to unexpected error accumulation.\n");
                break;
            case 3:
                printf("\nTERMINATED VIA: Reached the expected rank without achieving the specified tolerance.\n");
                break;
            case 4:
                printf("\nTERMINATED VIA: Lost orthonormality of Q_i.\n");
                EXPECT_TRUE(true);
                return;
                break;
            case 5:
                printf("\nTERMINATED VIA: Lost orthonormality of Q.\n");
                EXPECT_TRUE(true);
                return;
                break;
            case 0:
                printf("\nTERMINATED VIA: Expected tolerance reached.\n");
                break;
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
            printf("FRO NORM OF A - QB:    %e\n", norm_test_1);
            ASSERT_NEAR(norm_test_1, 0, 1e-12);
        }
        else
        {
            // Test Nonzero Tol Output
            printf("FRO NORM OF A - QB:    %e\n", norm_test_1);
            printf("FRO NORM OF A:         %e\n", norm_A);
            EXPECT_TRUE(norm_test_1 <= (tol * norm_A));
        }
        printf("|======================TEST QB2 K = min(M, N) END======================|\n");
    }

//Varying tol, k = min(m, n)
template <typename T>
    static std::vector<T> test_QB2_plot_helper(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {

        using namespace blas;
        using namespace lapack;
        
        // For running QB
        std::vector<T> A(m * n, 0.0);
        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);

        int64_t size = m * n;
        // Adjust the expected rank
        if(k == 0)
        {
            k = std::min(m, n);
        }

        int64_t k_est = std::min(m, n);

        std::vector<T> Q(m * k_est, 0.0);
        std::vector<T> B(k_est * n, 0.0);
        // For results comparison
        std::vector<T> A_hat(size, 0.0);
        //WARNING: block_sz may change.
        std::vector<T> cond_nums(k / block_sz);

        T* A_dat = A.data();
        T* Q_dat = Q.data();
        T* B_dat = B.data();
        T* A_hat_dat = A_hat.data();

        //RandLAPACK::comps::util::disp_diag(m, n, k, A);

        int termination = RandLAPACK::comps::qb::qb2<T>(
        m,
        n,
        A,
        k_est, // Here, serves as a backup termination criteria
        block_sz,
        tol,
        p,
        1,
        Q, // m by k
        B, // k by n
        seed,
        cond_nums
        );
        return cond_nums;
    }

template <typename T>
    static void test_QB2_plot(int64_t max_k, int64_t max_b_sz, int64_t max_p, int mat_type, T decay, bool diagon)
    {
        printf("|===========================TEST QB2 K PLOT===========================|\n");
        using namespace blas; 
        int32_t seed = 0;
        // Number of repeated runs of the same test
        int runs = 5;

        // varying matrix size
        for (int64_t k = 4096; k <= max_k; k *= 2)
        {
            // varying block size
            for (int64_t block_sz = 16; block_sz <= max_b_sz; block_sz *= 4)
            {
                int64_t v_sz = k / block_sz;  
                std::vector<T> all_vecs(v_sz * (runs + 1));
                T* all_vecs_dat = all_vecs.data();

                // fill the 1st coumn with iteration indexes
                int cnt = 0;
                std::for_each(all_vecs_dat, all_vecs_dat + v_sz,
                        // Lambda expression begins
                        [&cnt](T& entry)
                        {
                                entry = ++cnt;
                        }
                );

                // varying power iters
                for (int64_t p = 0; p <= max_p; p += 2)
                {
                    for (int i = 1; i < (runs + 1); ++i)
                    {
                        // Grab the vetcor of condition numbers
                        //std::vector<T>cond_nums = test_QB2_plot_helper<T>(k, k, k, p, block_sz, 0, std::make_tuple(0, 2, true), ++seed);
                        std::vector<T>cond_nums = test_QB2_plot_helper<T>(k, k, k, p, block_sz, 0, std::make_tuple(mat_type, decay, diagon), ++seed);
                        copy<T, T>(v_sz, cond_nums.data(), 1, all_vecs_dat + (v_sz * i), 1);
                    }
                    
                    // Save array as .dat file
                    std::ofstream file("../../build/test_plots/raw_data/test_" + std::to_string(k) + "_" + std::to_string(block_sz) + "_" + std::to_string(p) + "_" + std::to_string(int(decay)) + ".dat");
                    //unfortunately, cant do below with foreach
                    for (int i = 0; i < v_sz; ++ i)
                    {
                        T* entry = all_vecs_dat + i;
                        // how to simplify this expression?
                        file << *(entry) << "  " << *(entry + v_sz) << "  " << *(entry + (2 * v_sz)) << "  " << *(entry + (3 * v_sz)) << "  " << *(entry + (4 * v_sz)) << "  " << *(entry + (5 * v_sz)) << "\n";
                    }
                }
            }
        }
        printf("|============================TEST QB2 PLOT============================|\n");
    }

};
/*
TEST_F(TestQB, SimpleTest)
{ 
    for (uint32_t seed : {2})//, 1, 2})
    {
        
        // Fast polynomial decay test
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(0, 2, false), seed);
        // Slow polynomial decay test
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(0, 0.5, false), seed);
        // Superfast exponential decay test
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(1, 0.5, false), seed);
        // S-shaped decay matrix test 
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(2, 0, false), seed);
        // A = [A A]
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(3, 0, false), seed);
        // A = 0
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(4, 0, false), seed);
        // Random diagonal matrix test
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(5, 0, false), seed);
        // A = diag(sigma), where sigma_1 = ... = sigma_l > sigma_{l + 1} = ... = sigma_n
        test_QB2_general<double>(1000, 1000, 0, 5, 2, 1.0e-9, std::make_tuple(6, 0, false), seed);
        

        // test zero tol
        test_QB2_k_eq_min<double>(1000, 1000, 10, 5, 2, 0.0, std::make_tuple(1, 0, false), seed);
        // test nonzero tol
        test_QB2_k_eq_min<double>(1000, 1000, 10, 5, 2, 0.1, std::make_tuple(1, 0, false), seed);
    }
}
*/
// Testing with full-rank square diagonal matrices with polynomial decay of varying speed.
TEST_F(TestQB, PlotTest)
{ 
    // Fast decay
    //test_QB2_plot<double>(4096, 256, 2, 0, 2, true);
    // Slow decay
    //test_QB2_plot<double>(4096, 256, 0, 0, 0.5, true);
}

TEST_F(TestQB, RS_OOTest)
{ 
    /*
    int64_t m = 10;
    int64_t n = 10;
    int64_t k = 10;
    int64_t p = 2;
    int64_t q = 1;
    int32_t seed = 0;

    std::vector<double> A(m * n, 0.0);
    std::vector<double> Omega(n * k, 0.0);

    RandBLAS::dense_op::gen_rmat_norm<double>(m, n, A.data(), seed);

    RandLAPACK::comps::rs::RowSketcher<double> RS
    (
        seed, 
        p, 
        q,
        &RandLAPACK::comps::orth::stab_LU<double>
    );

    RS.RS1(m, n, A, k, Omega);
    */
}

TEST_F(TestQB, RF_OOTest)
{ 
    /*
    // Make a subclass object by calling a constructor
    RandLAPACK::comps::rs::One<double> One(1);

    RandLAPACK::comps::rf::Two<double> Two(2, One);

    Two.do_more_stuff(1.1);
    */

    // BASIC PARAMS
    int64_t m = 10;
    int64_t n = 10;
    int64_t k = 10;

    // PARAMS FOR RS
    int64_t p = 2;
    int64_t q = 1;
    int32_t seed = 0;

    // PARAMS FOR RF
    bool use_qr = true;

    std::vector<double> A(m * n, 0.0);
    std::vector<double> Q(m * k, 0.0);
    std::vector<double> Omega(n * k, 0.0);

    // FILL THE MATRIX
    RandBLAS::dense_op::gen_rmat_norm<double>(m, n, A.data(), seed);

    // DECLARE A RS
    RandLAPACK::comps::rs::RowSketcher<double> RS
    (
        seed, p, q, &RandLAPACK::comps::orth::stab_LU<double>
    );

    // DECLARE A RF
    RandLAPACK::comps::rf::RangeFinder<double> RF
    (
        RS, NULL, true, true
    );

    RF.RF1(m, n, A, k, Q, use_qr);
}
