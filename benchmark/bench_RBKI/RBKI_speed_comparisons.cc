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
    std::vector<T> A_cpy;
    std::vector<T> Sigma_exact;

    RBKI_benchmark_data(int64_t m, int64_t n, int64_t k, T tol) :
    A(m * n, 0.0),
    U(m * n, 0.0),
    V(n * n, 0.0),
    Sigma(n, 0.0),
    A_cpy(m * n, 0.0),
    Sigma_exact(n, 0.0)
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
                                        RandBLAS::RNGState<RNG> &state) {

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);
    std::fill(all_data.U.begin(), all_data.U.end(), 0.0);
    std::fill(all_data.V.begin(), all_data.V.end(), 0.0);
    std::fill(all_data.Sigma.begin(), all_data.Sigma.end(), 0.0);
}

template <typename T, typename RNG>
static std::vector<long> call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t numruns,
    int64_t k,
    RBKI_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state) {

    auto m        = all_data.row;
    auto n        = all_data.col;
    auto tol      = all_data.tolerance;

    // Additional params setup.
    RandLAPACK::RBKI<double, r123::Philox4x32> RBKI(false, false, tol);

    // timing vars
    long dur_rbki     = 0;
    long dur_other    = 0;
    long t_rbki_best  = 0;
    long t_other_best = 0;

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    for (int i = 0; i < numruns; ++i) {
        printf("Iteration %d start.\n", i);
        
        // Testing RBKI
        auto start_rbki = high_resolution_clock::now();
        RBKI.call(m, n, all_data.A.data(), m, k, all_data.U.data(), all_data.V.data(), all_data.Sigma.data(), state);
        auto stop_rbki = high_resolution_clock::now();
        dur_rbki = duration_cast<microseconds>(stop_rbki - start_rbki).count();

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);

        // Testing Other
        auto start_other = high_resolution_clock::now();
        /// RIVAL ALGORITHM CALL
        auto stop_other = high_resolution_clock::now();
        dur_other = duration_cast<microseconds>(stop_other - start_other).count();

        state_gen = state;
        data_regen<T, RNG>(m_info, all_data, state_gen);
    
        i == 0 ? t_rbki_best  = dur_rbki  : (dur_rbki < t_rbki_best)   ? t_rbki_best = dur_rbki   : NULL;
        i == 0 ? t_other_best = dur_other : (dur_other < t_other_best) ? t_other_best = dur_other : NULL;
    }

    std::vector<long> res{t_rbki_best, t_other_best};

    return res;
}

/*
int main() {
    // Declare parameters
    int64_t m           = std::pow(10, 3);
    int64_t n           = std::pow(10, 3);
    int64_t k_start     = 100;
    int64_t k_stop      = 100;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);

    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    // Declare a data file
    std::fstream file("RBKI_speed_comp_m_"          + std::to_string(m)
                                      + "_n_"       + std::to_string(n)
                                      + "_k_start_" + std::to_string(k_start)
                                      + "_k_stop_"  + std::to_string(k_stop)
                                      + ".dat", std::fstream::app);

    for (;k_start <= k_stop; k_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, k_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",\n";
    }
}
*/

/*

int main(int argc, char *argv[]) {
    // Declare parameters
    int64_t m           = std::pow(10, 3);
    int64_t n           = std::pow(10, 3);
    int64_t k_start     = 100;
    int64_t k_stop      = 100;
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    // Timing results
    std::vector<long> res;
    // Number of algorithm runs. We only record best times.
    int64_t numruns = 5;

    // Allocate basic workspace
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);

    // Generate the input matrix - gaussian suffices for performance tests.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);

    m_info.filename = argv[1];

    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A, state);

    printf("rows %ld, cols %ld\n", m_info.rows, m_info.cols);

    // Declare a data file
    std::fstream file("RBKI_speed_comp_m_"          + std::to_string(m)
                                      + "_n_"       + std::to_string(n)
                                      + "_k_start_" + std::to_string(k_start)
                                      + "_k_stop_"  + std::to_string(k_stop)
                                      + ".dat", std::fstream::app); 
    for (;k_start <= k_stop; k_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, k_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",\n";
    }
}
*/


int main(int argc, char *argv[]) {

    int64_t m       = 0;
    int64_t n       = 0;
    int64_t k_start = 0;
    int64_t k_stop  = 0;
    double tol      = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state      = RandBLAS::RNGState();

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[1];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    k_start = std::max((int64_t) 1, n / 100);
    k_stop  = n;

    // Allocate basic workspace.
    RBKI_benchmark_data<double> all_data(m, n, k_stop, tol);
  
  
    printf("%d\n", all_data.A.size());
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen<double, r123::Philox4x32>(m_info, all_data.A.data(), state);

    printf("rows %ld, cols %ld\n", m_info.rows, m_info.cols);
    printf("%e\n", *(all_data.A.data() + 1));
/*
    // Declare a data file
    std::fstream file("RBKI_speed_comp_m_"          + std::to_string(m)
                                      + "_n_"       + std::to_string(n)
                                      + "_k_start_" + std::to_string(k_start)
                                      + "_k_stop_"  + std::to_string(k_stop)
                                      + ".dat", std::fstream::app); 


    for (;k_start <= k_stop; k_start *= 2) {
        res = call_all_algs<double, r123::Philox4x32>(m_info, numruns, k_start, all_data, state_constant);
        file << res[0]  << ",  " << res[1]  << ",\n";
    }
*/
}