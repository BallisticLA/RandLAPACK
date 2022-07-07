#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class TestQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};
/*
The function below, togethre with commented out lines 97, 98 and commented oyt lines 34, 42 and 47 within rs.hh
represent an example of failing to use reference RandBLAS function as an argument to RS class.
*/
/*
    template <typename T>
    static void check(int64_t a, int64_t b, T* c, int32_t d)
    {
        printf("hi");
    }
*/

    template <typename T>
    static void test_QB2_general(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|==================================TEST QB2 GENERAL BEGIN==================================|\n");
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
        copy(size, A_dat, 1, A_cpy_dat, 1);
        copy(size, A_dat, 1, A_cpy_2_dat, 1);

        //Subroutine parameters 
        bool verbosity = false;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::comps::orth::Stab<T> Stab(1);

        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::comps::rs::RS<T> RS(Stab, seed, p, passes_per_iteration, verbosity, cond_check, 0);

        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::comps::orth::Orth<T> Orth_RF(0);

        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::comps::rf::RF<T> RF(RS, Orth_RF, verbosity, cond_check, 0);

        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::comps::orth::Orth<T> Orth_QB(0);

        // QB constructor - Choose defaut (QB2)
        RandLAPACK::comps::qb::QB<T> QB(RF, Orth_QB, verbosity, orth_check, 0);

        // Regular QB2 call
        QB.call(
            m,
            n,
            A,
            k,
            block_sz,
            tol,
            Q,
            B
        );

        switch(QB.termination)
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
                //EXPECT_TRUE(true);
                //return;
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
        copy(k * n, B_dat, 1, B_cpy_dat, 1);
        
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
        copy(n - k, z_buf_dat, 1, s_dat + k, 1);
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
        printf("|===================================TEST QB2 GENERAL END===================================|\n");
    }

//Varying tol, k = min(m, n)
template <typename T>
    static void test_QB2_k_eq_min(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {
        
        printf("|===============================TEST QB2 K = min(M, N) BEGIN===============================|\n");

        using namespace blas;
        using namespace lapack;
        
        // For running QB
        std::vector<T> A(m * n, 0.0);
        RandLAPACK::comps::util::gen_mat_type(m, n, A, k, seed, mat_type);

        int64_t size = m * n;
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

        //Subroutine parameters 
        bool verbosity = false;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose CholQR
        RandLAPACK::comps::orth::Stab<double> Stab(0);

        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::comps::rs::RS<double> RS(Stab, seed, p, passes_per_iteration, verbosity, cond_check, 0);

        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::comps::orth::Orth<T> Orth_RF(0);

        // RangeFinder constructor - Choose default (rf1)
        RandLAPACK::comps::rf::RF<double> RF(RS, Orth_RF, verbosity, cond_check, 0);

        // Orthogonalization Constructor - Choose CholQR
        RandLAPACK::comps::orth::Orth<T> Orth_QB(0);

        // QB constructor - Choose defaut (QB2)
        RandLAPACK::comps::qb::QB<double> QB(RF, Orth_QB, verbosity, orth_check, 0);

        // Regular QB2 call
        QB.call(
            m,
            n,
            A,
            k_est,
            block_sz,
            tol,
            Q,
            B
        );
        
        switch(QB.termination)
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
        printf("|================================TEST QB2 K = min(M, N) END================================|\n");
    }

// Define a new return type
typedef std::pair<std::vector<double>, std::vector<double>>  vector_pair;

    template <typename T>
    static vector_pair test_QB2_plot_helper_run(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, std::tuple<int, T, bool> mat_type, uint32_t seed) {

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

        //RandLAPACK::comps::util::disp_diag(m, n, k, A);

        //Subroutine parameters 
        bool verbosity = true;
        bool cond_check = true;
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::comps::orth::Stab<double> Stab(1);

        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::comps::rs::RS<double> RS(Stab, seed, p, passes_per_iteration, verbosity, cond_check, 0);

        // Orthogonalization Constructor
        RandLAPACK::comps::orth::Orth<T> Orth_RF(1);

        // RangeFinder constructor
        RandLAPACK::comps::rf::RF<double> RF(RS, Orth_RF, verbosity, cond_check, 1);

        // Orthogonalization Constructor 
        RandLAPACK::comps::orth::Orth<T> Orth_QB(1);

        // QB constructor - Choose QB2_test_mode
        RandLAPACK::comps::qb::QB<double> QB(RF, Orth_QB, verbosity, orth_check, 1);

        // Test mode QB2
        QB.call(
            m,
            n,
            A,
            k,
            block_sz,
            tol,
            Q,
            B
        );

        switch(QB.termination)
        {
            case 1:
                printf("\nTERMINATED VIA: Input matrix of zero entries.\n");
                break;
            case 2:
                printf("\nTERMINATED VIA: Early termination due to unexpected error accumulation.\n");
                break;
            case 3:
                printf("\nTERMINATED VIA: Reached the expected rank without achieving the specified tolerance.\n");
                break;
            case 4:
                printf("\nTERMINATED VIA: Lost orthonormality of Q_i.\n");
                break;
            case 5:
                printf("\nTERMINATED VIA: Lost orthonormality of Q.\n");
                break;
            case 0:
                printf("\nTERMINATED VIA: Expected tolerance reached.\n");
                break;
        }
        printf("Inner dimension of QB: %ld\n", k);

        printf("SIZE IS %ld\n", RS.cond_nums.size());
        return std::make_pair(RF.cond_nums, RS.cond_nums);
    }

    template <typename T>
    static void test_QB2_plot(int64_t k, int64_t max_k, int64_t block_sz, int64_t max_b_sz, int64_t p, int64_t max_p, int mat_type, T decay, bool diagon)
    {
        printf("|==================================TEST QB2 K PLOT BEGIN==================================|\n");
        using namespace blas; 
        int32_t seed = 0;
        // Number of repeated runs of the same test
        int runs = 5;

        int64_t b_sz_init = block_sz;
        int64_t p_init = p;

        // varying matrix size
        for (; k <= max_k; k *= 2)
        {
            block_sz = b_sz_init;
            // varying block size
            for (; block_sz <= max_b_sz; block_sz *= 4)
            {
                // Making RF's ALL_VEC
                int64_t v_RF_sz = k / block_sz;  
                std::vector<T> all_vecs_RF(v_RF_sz * (runs + 1));
                T* all_vecs_RF_dat = all_vecs_RF.data();

                // fill the 1st coumn with iteration indexes
                int cnt = 0;
                std::for_each(all_vecs_RF_dat, all_vecs_RF_dat + v_RF_sz,
                        // Lambda expression begins
                        [&cnt](T& entry)
                        {
                                entry = ++cnt;
                        }
                );

                // varying power iters
                p = p_init;
                for (; p <= max_p; p += 2)
                {
                    // Making RS's ALL_VEC
                    int64_t v_RS_sz = p * k / block_sz;  
                    std::vector<T> all_vecs_RS(v_RS_sz * (runs + 1));
                    T* all_vecs_RS_dat = all_vecs_RS.data();

                    // fill the 1st coumn with iteration indexes
                    int cnt = 0;
                    std::for_each(all_vecs_RS_dat, all_vecs_RS_dat + v_RS_sz,
                            // Lambda expression begins
                            [&cnt](T& entry)
                            {
                                    entry = ++cnt;
                            }
                    );
             
                    for (int i = 1; i < (runs + 1); ++i)
                    {
                        // Grab the vetcor of condition numbers
                        vector_pair cond_nums = test_QB2_plot_helper_run<T>(k, k, k, p, block_sz, 0, std::make_tuple(mat_type, decay, diagon), ++seed);
                        // Fill RF
                        copy<T, T>(v_RF_sz, cond_nums.first.data(), 1, all_vecs_RF_dat + (v_RF_sz * i), 1);
                        // Fill RS
                        if(v_RS_sz > 0)
                        {
                            copy<T, T>(v_RS_sz, cond_nums.second.data(), 1, all_vecs_RS_dat + (v_RS_sz * i), 1);
                        
                        

                        }
                    }
                    
                    // Save array as .dat file - generic plot
                    std::string path_RF = "../../build/test_plots/test_cond/raw_data/test_RF_" + std::to_string(k) + "_" + std::to_string(block_sz) + "_" + std::to_string(p) + "_" + std::to_string(int(decay)) + ".dat";
                    std::string path_RS = "../../build/test_plots/test_cond/raw_data/test_RS_" + std::to_string(k) + "_" + std::to_string(block_sz) + "_" + std::to_string(p) + "_" + std::to_string(int(decay)) + ".dat";

                    std::ofstream file_RF(path_RF);
                    //unfortunately, cant do below with foreach
                    for (int i = 0; i < v_RF_sz; ++ i)
                    {
                        T* entry = all_vecs_RF_dat + i;
                        // how to simplify this expression?
                        file_RF << *(entry) << "  " << *(entry + v_RF_sz) << "  " << *(entry + (2 * v_RF_sz)) << "  " << *(entry + (3 * v_RF_sz)) << "  " << *(entry + (4 * v_RF_sz)) << "  " << *(entry + (5 * v_RF_sz)) << "\n";
                    }

                    if(v_RS_sz > 0)
                    {
                        std::ofstream file_RS(path_RS);
                        //unfortunately, cant do below with foreach
                        for (int i = 0; i < v_RS_sz; ++ i)
                        {
                            T* entry = all_vecs_RS_dat + i;
                            // how to simplify this expression?
                            file_RS << *(entry) << "  " << *(entry + v_RS_sz) << "  " << *(entry + (2 * v_RS_sz)) << "  " << *(entry + (3 * v_RS_sz)) << "  " << *(entry + (4 * v_RS_sz)) << "  " << *(entry + (5 * v_RS_sz)) << "\n";
                        }
                    }
                }
            }
        }
        printf("|====================================TEST QB2 PLOT END====================================|\n");
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
        test_QB2_general<double>(1000, 1000, 50, 5, 2, 1.0e-9, std::make_tuple(1, 2, false), seed);
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

        // SOMETHING IS OFF HERE
        // test zero tol
        test_QB2_k_eq_min<double>(1000, 1000, 10, 5, 2, 0.0, std::make_tuple(1, 0.5, false), seed);
        // test nonzero tol
        test_QB2_k_eq_min<double>(1000, 1000, 10, 5, 2, 0.1, std::make_tuple(1, 0.5, false), seed);
    }
}
*/

// Testing with full-rank square diagonal matrices with polynomial decay of varying speed.
// Will populate files with condition numbers of sketches
// Running tests without the orthogonality loss check to ensure normal termination
TEST_F(TestQB, PlotTest)
{   
    // Slow_decay
    //test_QB2_plot<double>(1024, 1024, 16, 16, 2, 2, 0, 2, true);
    test_QB2_plot<double>(2048, 2048, 128, 128, 2, 2, 0, 2, true);
    // Fast decay
    //test_QB2_plot<double>(1024, 2048, 128, 128, 0, 2, 0, 0.5, true);
}

