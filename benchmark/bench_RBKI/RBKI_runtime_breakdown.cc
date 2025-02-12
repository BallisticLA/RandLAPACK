/*
RBKI runtime breakdown benchmark - assesses the time taken by each subcomponent of RBKI.
Records all, data, not just the best.
There are 10 things that we time:
                1.Allocate and free time.
                2.Time to acquire the SVD factors.
                3.UNGQR time.
                4.Reorthogonalization time.
                5.QR time.
                6.GEMM A time.
                7.Sketching time.
                8.R_ii cpy time.
                9.S_ii cpy time.
                10.Norm R time.
*/
#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>

template <typename T>
struct RBKI_benchmark_data {
    int64_t row;
    int64_t col;
    int64_t rank; // has to be modifiable
    T tolerance;
    std::vector<T> A;
    std::vector<T> U;
    std::vector<T> V;
    std::vector<T> Sigma;
    std::vector<T> Sigma_cpy_RBKI;
    std::vector<T> Sigma_cpy_SVD;
    std::vector<T> Sigma_cpy_Other;

    RBKI_benchmark_data(int64_t m, int64_t n, int64_t k, T tol) :
    A(m * n, 0.0),
    U(m * n, 0.0),
    V(n * n, 0.0),
    Sigma(n, 0.0),
    Sigma_cpy_RBKI(n, 0.0),
    Sigma_cpy_SVD(n, 0.0),
    Sigma_cpy_Other(n, 0.0)
    {
        row = m;
        col = n;
        rank = k;
        tolerance = tol;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        RBKI_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int overwrite_A) {

    if (overwrite_A)
        RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);
    std::fill(all_data.U.begin(), all_data.U.end(), 0.0);
    std::fill(all_data.V.begin(), all_data.V.end(), 0.0);
    std::fill(all_data.Sigma.begin(), all_data.Sigma.end(), 0.0);
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t k,
    int64_t num_krylov_iters,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m   = all_data.row;
    auto n   = all_data. col;
    auto tol = all_data.tolerance;
    bool time_subroutines = true;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, time_subroutines, tol);
    RBKI.max_krylov_iters = num_krylov_iters;
    RBKI.num_threads_min = 4;
    RBKI.num_threads_max = util::get_omp_threads();

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    // Timing vars
    std::vector<long> inner_timing;

    for (int i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        RBKI.call(all_data.A.data(), m, n, m, k, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state_alg);
        
        // Update timing vector
        inner_timing = RBKI.times;
        // Add info about the run
        inner_timing.insert (inner_timing.begin(), k);
        inner_timing.insert (inner_timing.begin(), num_krylov_iters);

        std::ofstream file(output_filename, std::ios::app);
        std::copy(inner_timing.begin(), inner_timing.end(), std::ostream_iterator<long>(file, ", "));
        file << "\n";

        // Clear and re-generate data
        data_regen(m_info, all_data, state_gen, 0);
        state_gen = state;
        state_alg = state;
    }
}

int main(int argc, char *argv[]) {

    printf("Function begin\n");

    if(argc <= 1) {
        printf("No input provided\n");
        return 0;
    }

    int64_t m                      = 0;
    int64_t n                      = 0;
    int64_t k_start                = 0;
    int64_t k_stop                 = 0;
    int64_t num_krylov_iters_start = 2;
    int64_t num_krylov_iters_curr  = num_krylov_iters_start;
    int64_t num_krylov_iters_stop  = 64;
    double tol                     = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state                     = RandBLAS::RNGState<r123::Philox4x32>();
    auto state_constant            = state;
    int numruns                    = 5;
    std::vector<long> res;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[1];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    k_start = 2;//std::max((int64_t) 1, n / 10);
    k_stop  = 256;//std::max((int64_t) 1, n / 10);

    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen(m_info, all_data.A.data(), state);

    printf("Finished data preparation\n");

    // Declare a data file
    std::string output_filename = "RBKI_runtime_breakdown_m_"      + std::to_string(m)
                                      + "_n_"                      + std::to_string(n)
                                      + "_k_start_"                + std::to_string(k_start)
                                      + "_k_stop_"                 + std::to_string(k_stop)
                                      + "_num_krylov_iters_start_" + std::to_string(num_krylov_iters_start)
                                      + "_num_krylov_iters_stop_"  + std::to_string(num_krylov_iters_stop)
                                      + ".dat"; 

    for (;k_start <= k_stop; k_start *=2) {
        for (;num_krylov_iters_curr <= num_krylov_iters_stop; num_krylov_iters_curr *=2) {
            call_all_algs(m_info, numruns, k_start, num_krylov_iters_curr, all_data, state_constant, output_filename);
        }
        num_krylov_iters_curr = num_krylov_iters_start;
    }
}
