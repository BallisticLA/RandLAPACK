#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct blas_benchmark_data {
    int64_t dim;
    std::vector<T> A;
    std::vector<T> B;
    std::vector<T> C;
    std::vector<T> a;
    std::vector<T> b;

    blas_benchmark_data(int64_t n) :
    A(n * n, 0.0),
    B(n * n, 0.0),
    C(n * n, 0.0),
    a(n * n, 0.0),
    b(n * n, 0.0)
    {
        dim = n;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        blas_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state,
                                        int alg_type) {

    auto state_const = state;
    switch (alg_type) {
        case 3: {
            RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state_const);
            state_const = state;
            RandLAPACK::gen::mat_gen(m_info, all_data.B.data(), state_const);
            std::fill(all_data.C.begin(), all_data.C.end(), 0.0);
            break;
            }
        case 2: {
            RandBLAS::DenseDist D1(1, m_info.cols);
            RandBLAS::fill_dense(D1, all_data.a.data(), state_const);
            state_const = state;
            RandBLAS::fill_dense(D1, all_data.b.data(), state_const);
            state_const = state;
            RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state_const);
            break;
            }
        case 1: {
            RandBLAS::DenseDist D2(1, m_info.cols);
            RandBLAS::fill_dense(D2, all_data.a.data(), state_const);
            state_const = state;
            RandBLAS::fill_dense(D2, all_data.b.data(), state_const);
            break;
        }
        default: {
            break;
        }
    }
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t n,
    blas_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    // timing vars
    long dur_blas1  = 0;
    long dur_blas2  = 0;
    long dur_blas3  = 0;
    
    // Making sure the states are unchanged
    auto state_gen = state;

    for (int i = 0; i < numruns; ++i) {
        std::cout << "ITERATION " << i << ", DIM " << n << "\n";
        // Testing BLAS3
        auto start_blas3 = steady_clock::now();
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, n, n, n, 1.0, all_data.A.data(), n, all_data.B.data(), n, 0.0, all_data.C.data(), n);
        auto stop_blas3 = steady_clock::now();
        dur_blas3 = duration_cast<microseconds>(stop_blas3 - start_blas3).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen, 3);

        // Testing BLAS2
        auto start_blas2 = steady_clock::now();
        blas::gemv(Layout::ColMajor, Op::NoTrans, n, n, 1.0, all_data.A.data(), n, all_data.a.data(), 1, 1.0, all_data.b.data(), 1);
        auto stop_blas2 = steady_clock::now();
        dur_blas2 = duration_cast<microseconds>(stop_blas2 - start_blas2).count();
    
        state_gen = state;
        data_regen(m_info, all_data, state_gen, 2);

        // Testing BLAS1
        auto start_blas1 = steady_clock::now();
        blas::axpy(n, -1.0, all_data.a.data(), 1, all_data.b.data(), 1);
        auto stop_blas1 = steady_clock::now();
        dur_blas1 = duration_cast<microseconds>(stop_blas1 - start_blas1).count();

        state_gen = state;
        data_regen(m_info, all_data, state_gen, 2);

        std::ofstream file(output_filename, std::ios::app);
        file << n << ",  " << dur_blas1 << ",  " << dur_blas2 << ",  " << dur_blas3 << ",\n";
    }
}

int main() {
    // Declare parameters
    int64_t n_start     = std::pow(2, 10);
    int64_t n_stop      = std::pow(2, 14);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    blas_benchmark_data<double> all_data(n_stop);
    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(n_stop, n_stop, RandLAPACK::gen::gaussian);
    data_regen(m_info, all_data, state, 3);
    data_regen(m_info, all_data, state, 2);

    // Declare a data file
    std::string output_filename = "BLAS_performance_comp_col_start_"    
                                                        + std::to_string(n_start)
                                      + "_col_stop_"    + std::to_string(n_stop)
                                      + ".dat"; 

    for (;n_start <= n_stop; n_start *= 2) {
        call_all_algs(m_info, numruns, n_start, all_data, state_constant, output_filename);
    }
}
