#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"

#include <RandBLAS.hh>
#include <math.h>
#include <numeric>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <fstream>

/*
Note: this benchmark attempts to save files into a specific location.
If the required folder structure does not exist, the files will not be saved.
*/

// Define a new return type
typedef std::pair<std::vector<double>, std::vector<double>>  vector_pair;

template <typename T, typename RNG>
static vector_pair test_QB2_plot_helper_run(int64_t m, int64_t n, int64_t k, int64_t p, int64_t block_sz, T tol, const std::tuple<int, T, bool>& mat_type, RandBLAS::base::RNGState<RNG> state) {

    // For running QB
    std::vector<T> A(m * n, 0.0);
    RandLAPACK::util::gen_mat_type(m, n, A, k, state, mat_type);

    int64_t size = m * n;
    // Adjust the expected rank
    if(k == 0) {
        k = std::min(m, n);
    }
    int64_t k_est = std::min(m, n);

    std::vector<T> Q(m * k_est, 0.0);
    std::vector<T> B(k_est * n, 0.0);
    // For results comparison
    std::vector<T> A_hat(size, 0.0);

    //RandLAPACK::comps::util::disp_diag(m, n, k, A);

    //Subroutine parameters 
    bool verbosity = true;
    bool cond_check = true; // MUST BE TRUE, OR REGFAULT
    bool orth_check = true;
    int64_t passes_per_iteration = 1;

    // Make subroutine objects
    // Stabilization Constructor - Choose PLU
    RandLAPACK::PLUL<T> Stab(cond_check, verbosity);
    // RowSketcher constructor - Choose default (rs1)
    RandLAPACK::RS<T, RNG> RS(Stab, state, p, passes_per_iteration, verbosity, cond_check);
    // Orthogonalization Constructor - use HQR
    RandLAPACK::CholQRQ<T> Orth_RF(cond_check, verbosity);
    // RangeFinder constructor
    RandLAPACK::RF<T> RF(RS, Orth_RF, verbosity, cond_check);
    // Orthogonalization Constructor - use HQR
    RandLAPACK::CholQRQ<T> Orth_QB(cond_check, verbosity);
    // QB constructor - Choose default QB2
     RandLAPACK::QB<T> QB(RF, Orth_QB, verbosity, orth_check);

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

    printf("Inner dimension of QB: %ld\n", k);

    printf("SIZE IS %ld\n", RS.cond_nums.size());
    return std::make_pair(RF.cond_nums, RS.cond_nums);
}

template <typename T, typename RNG>
static void test_QB2_plot(int64_t k, int64_t max_k, int64_t block_sz, int64_t max_b_sz, int64_t p, int64_t max_p, int mat_type, T decay, bool diagon, std::string path_RF, std::string path_RS, RandBLAS::base::RNGState<RNG> state)
{
    printf("|==================================TEST QB2 K PLOT BEGIN==================================|\n");

    // Number of repeated runs of the same test
    int runs = 5;

    int64_t b_sz_init = block_sz;
    int64_t p_init = p;

    // varying matrix size
    for (; k <= max_k; k *= 2) {
        block_sz = b_sz_init;
        // varying block size
        for (; block_sz <= max_b_sz; block_sz *= 4) {
            // Making RF's ALL_VEC
            int64_t v_RF_sz = k / block_sz;  
            std::vector<T> all_vecs_RF(v_RF_sz * (runs + 1));
            T* all_vecs_RF_dat = all_vecs_RF.data();

            // fill the 1st coumn with iteration indexes
            int cnt = 0;
            std::for_each(all_vecs_RF_dat, all_vecs_RF_dat + v_RF_sz,
                    // Lambda expression begins
                    [&cnt](T& entry) {
                            entry = ++cnt;
                    }
            );

            // varying power iters
            p = p_init;
            for (; p <= max_p; p += 2) {
                // Making RS's ALL_VEC
                int64_t v_RS_sz = p * k / block_sz;  
                std::vector<T> all_vecs_RS(v_RS_sz * (runs + 1));
                T* all_vecs_RS_dat = all_vecs_RS.data();

                // fill the 1st coumn with iteration indexes
                int cnt = 0;
                std::for_each(all_vecs_RS_dat, all_vecs_RS_dat + v_RS_sz,
                        // Lambda expression begins
                        [&cnt](T& entry) {
                                entry = ++cnt;
                        }
                );
            
                for (int i = 1; i < (runs + 1); ++i) {
                    // Grab the vetcor of condition numbers
                    vector_pair cond_nums = test_QB2_plot_helper_run<T>(k, k, k, p, block_sz, 0, std::make_tuple(mat_type, decay, diagon), state);
                    // Fill RF
                    blas::copy(v_RF_sz, cond_nums.first.data(), 1, all_vecs_RF_dat + (v_RF_sz * i), 1);
                    // Fill RS
                    if(v_RS_sz > 0) {
                        blas::copy(v_RS_sz, cond_nums.second.data(), 1, all_vecs_RS_dat + (v_RS_sz * i), 1);
                    }
                }
                
                // Save array as .dat file - generic plot
                std::string full_path_RF = path_RF + "test_RF_" + std::to_string(k) + "_" + std::to_string(block_sz) + "_" + std::to_string(p) + "_" + std::to_string(int(decay)) + ".dat";
                std::string full_path_RS = path_RS + "test_RS_" + std::to_string(k) + "_" + std::to_string(block_sz) + "_" + std::to_string(p) + "_" + std::to_string(int(decay)) + ".dat";

                std::ofstream file_RF(full_path_RF);
                //unfortunately, cant do below with foreach
                for (int i = 0; i < v_RF_sz; ++ i) {
                    T* entry = all_vecs_RF_dat + i;
                    // how to simplify this expression?
                    file_RF << *(entry) << "  " << *(entry + v_RF_sz) << "  " << *(entry + (2 * v_RF_sz)) << "  " << *(entry + (3 * v_RF_sz)) << "  " << *(entry + (4 * v_RF_sz)) << "  " << *(entry + (5 * v_RF_sz)) << "\n";
                }

                if(v_RS_sz > 0) {
                    std::ofstream file_RS(full_path_RS);
                    //unfortunately, cant do below with foreach
                    for (int i = 0; i < v_RS_sz; ++ i) {
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

// Testing with full-rank square diagonal matrices with polynomial decay of varying speed.
// Will populate files with condition numbers of sketches
// Running tests without the orthogonality loss check to ensure normal termination

int main() 
{   
    auto state = RandBLAS::base::RNGState();
    // Slow_decay
    test_QB2_plot<double, r123::Philox4x32>(2048, 2048, 256, 256, 2, 2, 0, 2, true, "../", "../", state);
    // Fast decay
    test_QB2_plot<double, r123::Philox4x32>(1024, 2048, 256, 256, 0, 2, 0, 0.5, true, "../", "../", state);
}
