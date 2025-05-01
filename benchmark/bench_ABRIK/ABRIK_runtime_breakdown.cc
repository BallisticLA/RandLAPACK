/*
ABRIK runtime breakdown benchmark - assesses the time taken by each subcomponent of ABRIK.
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
struct ABRIK_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    T* A;
    T* U;
    T* V;
    T* Sigma;

    ABRIK_benchmark_data(int64_t m, int64_t n, T tol)
    {
        row       = m;
        col       = n;
        tolerance = tol;
        A         = new T[m * n]();
        U         = nullptr;
        V         = nullptr;
        Sigma     = nullptr;
    }

    ~ABRIK_benchmark_data(){
        delete[] A;
        delete[] U;
        delete[] V;
        delete[] Sigma;
    }
};

// Re-generate and clear data
template <typename T, typename RNG>
static void data_regen(RandLAPACK::gen::mat_gen_info<T> m_info, 
                                        ABRIK_benchmark_data<T> &all_data, 
                                        RandBLAS::RNGState<RNG> &state, int overwrite_A) {
    auto m = all_data.row;
    auto n = all_data. col;

    if (overwrite_A) {
        RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    }
    delete[] all_data.U;
    delete[] all_data.V;
    delete[] all_data.Sigma;
    all_data.U     = nullptr;
    all_data.V     = nullptr;
    all_data.Sigma = nullptr;
}

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    int64_t k,
    int64_t num_krylov_iters,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::string output_filename) {

    auto m   = all_data.row;
    auto n   = all_data. col;
    auto tol = all_data.tolerance;
    bool time_subroutines = true;

    // Additional params setup.
    RandLAPACK::ABRIK<double, r123::Philox4x32> ABRIK(false, time_subroutines, tol);
    ABRIK.max_krylov_iters = num_krylov_iters;
    ABRIK.num_threads_min = 4;
    ABRIK.num_threads_max = RandLAPACK::util::get_omp_threads();

    // Making sure the states are unchanged
    auto state_gen = state;
    auto state_alg = state;

    // Timing vars
    std::vector<long> inner_timing;

    for (int i = 0; i < num_runs; ++i) {
        printf("Iteration %d start.\n", i);
        ABRIK.call(m, n, all_data.A, m, k, all_data.U, all_data.V, all_data.Sigma, state_alg);
        
        // Update timing vector
        inner_timing = ABRIK.times;
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

    if (argc < 11) {
        // Expected input into this benchmark.
        std::cerr << "Usage: " << argv[0] << " <output_directory_path> <input_matrix_path> <num_runs> <num_rows> <num_cols> <custom_rank> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return 1;
    }

    int num_runs        = std::stol(argv[3]);
    int64_t m_expected  = std::stol(argv[5]);
    int64_t n_expected  = std::stol(argv[6]);
    int64_t custom_rank = std::stol(argv[6]);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[7]); ++i)
        b_sz.push_back(std::stoi(argv[i + 9]));
    // Save elements in string for logging purposes
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[8]); ++i)
        matmuls.push_back(std::stoi(argv[i + 9 + std::stol(argv[7])]));
    // Save elements in string for logging purposes
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    double tol          = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state          = RandBLAS::RNGState();
    auto state_constant = state;
    int64_t m = 0, n = 0;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[1];
    m_info.workspace_query_mod = 1;
    // Workspace query;
    RandLAPACK::gen::mat_gen<double>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected input size did not matrch actual input size. Aborting." << std::endl;
        return 1;
    }

    // Allocate basic workspace.
    ABRIK_benchmark_data<double> all_data(m, n, tol);
  
    // Fill the data matrix;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    printf("Finished data preparation\n");
    // Declare a data file
    std::string output_filename = "_ABRIK_runtime_breakdown_num_info_lines_" + std::to_string(6) + ".txt";
    std::string path;
    if (std::string(argv[1]) != ".") {
        path = argv[1] + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Writing important data into file
    file << "Description: Results from the ABRIK runtime breakdown benchmark, recording the time it takes to perform every subroutine in ABRIK."
              "\nFile format: 13 data columns, each corresponding to a given ABRIK subroutine: allocation_t_dur, get_factors_t_dur, ungqr_t_dur, reorth_t_dur, qr_t_dur, gemm_A_t_dur, main_loop_t_dur, sketching_t_dur, r_cpy_t_dur, s_cpy_t_dur, norm_t_dur, t_rest, total_t_dur"
              "               rows correspond to ABRIK runs with block sizes varying as specified, with numruns repititions of each block size"
              "\nInput type:"       + std::string(argv[2]) +
              "\nInput size:"       + std::to_string(m) + " by "             + std::to_string(n) +
              "\nAdditional parameters: Krylov block sizes "                 + b_sz_string +
                                        " matmuls: "                         + matmuls_string +
                                        " num runs per size "                + std::to_string(num_runs) +
                                        " num singular values and vectors approximated " + std::to_string(custom_rank) +
              "\n";
    file.flush();

    size_t i, j = 0;
    for (;i < b_sz.size(); ++i) {
        for (;j < matmuls.size(); ++j) {
            call_all_algs(m_info, num_runs, b_sz[i], matmuls[j], all_data, state_constant, path);
        }
        j = 0;
    }
}
