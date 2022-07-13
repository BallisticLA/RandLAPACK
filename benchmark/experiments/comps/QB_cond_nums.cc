#include <gtest/gtest.h>
#include <blas.hh>
#include <lapack.hh>
#include <RandBLAS.hh>
#include <RandLAPACK.hh>

#include <fstream>

#define RELDTOL 1e-10;
#define ABSDTOL 1e-12;

class BenchmarkQB : public ::testing::Test
{
    protected:

    virtual void SetUp() {};

    virtual void TearDown() {};

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
        bool cond_check = true; // MUST BE TRUE, OR REGFAULT
        bool orth_check = true;
        int64_t passes_per_iteration = 1;

        // Make subroutine objects
        // Stabilization Constructor - Choose PLU
        RandLAPACK::comps::orth::Stab<double> Stab(0);

        // RowSketcher constructor - Choose default (rs1)
        RandLAPACK::comps::rs::RS<double> RS(Stab, seed, p, passes_per_iteration, verbosity, cond_check, 0);

        // Orthogonalization Constructor - use HQR
        RandLAPACK::comps::orth::Orth<T> Orth_RF(0);

        // RangeFinder constructor
        RandLAPACK::comps::rf::RF<double> RF(RS, Orth_RF, verbosity, cond_check, 0);

        // Orthogonalization Constructor - use HQR
        RandLAPACK::comps::orth::Orth<T> Orth_QB(0);

        // QB constructor - Choose QB2_test_mode
        RandLAPACK::comps::qb::QB<double> QB(RF, Orth_QB, verbosity, orth_check, 0);

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

// Testing with full-rank square diagonal matrices with polynomial decay of varying speed.
// Will populate files with condition numbers of sketches
// Running tests without the orthogonality loss check to ensure normal termination
TEST_F(BenchmarkQB, PlotTest)
{   
    //test_QB2_plot<double>(10, 10, 2, 2, 2, 2, 0, 2, true);
    //test_QB2_plot_helper_run<double>(10, 10, 10, 2, 2, 0, std::make_tuple(0, 2, true), 0);
    
    
    
    // Slow_decay
    //test_QB2_plot<double>(1024, 1024, 16, 16, 2, 2, 0, 2, true);
    //test_QB2_plot<double>(2048, 2048, 128, 128, 2, 2, 0, 2, true);
    // Fast decay
    //test_QB2_plot<double>(1024, 2048, 128, 128, 0, 2, 0, 0.5, true);
}
