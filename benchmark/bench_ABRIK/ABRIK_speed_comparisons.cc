/*
ABRIK speed comparison benchmark - runs ABRIK, RSVD, SVDS (Spectra), and optionally full SVD (GESDD).
Precision (float or double) is specified as the first CLI argument.
The user provides a matrix file, numbers of Krylov iterations, block sizes, and a target rank.

Output: CSV file with '#'-prefixed metadata header, column names, then data rows.
Each algorithm section manages its own output allocation/deallocation:
  - ABRIK allocates with new[] internally   -> cleanup with delete[]
  - RSVD  allocates with calloc internally  -> cleanup with free()
  - SVDS/SVD are allocated by the benchmark -> cleanup with delete[]

Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F).
Timings in microseconds.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>

// External libs includes
#include <Eigen/Dense>
#include <Spectra/contrib/PartialSVDSolver.h>

// Traits struct mapping scalar type T to Eigen matrix/vector types.
template <typename T> struct EigenTypes;
template <> struct EigenTypes<double> {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
};
template <> struct EigenTypes<float> {
    using Matrix = Eigen::MatrixXf;
    using Vector = Eigen::VectorXf;
};

template <typename T>
struct ABRIK_benchmark_data {
    int64_t row;
    int64_t col;
    T tolerance;
    T* A;
    T* U;
    T* VT;
    T* V;
    T* Sigma;
    typename EigenTypes<T>::Matrix A_spectra;

    ABRIK_benchmark_data(int64_t m, int64_t n, T tol) :
    A_spectra(m, n)
    {
        A     = new T[m * n]();
        U     = nullptr;
        VT    = nullptr;
        V     = nullptr;
        Sigma = nullptr;
        row   = m;
        col   = n;
        tolerance = tol;
    }

    ~ABRIK_benchmark_data() {
        delete[] A;
    }
};

template <typename T, typename RNG>
struct ABRIK_algorithm_objects {
    RandLAPACK::PLUL<T> Stab;
    RandLAPACK::RS<T, RNG> RS;
    RandLAPACK::CholQRQ<T> Orth_RF;
    RandLAPACK::RF<T, RNG> RF;
    RandLAPACK::CholQRQ<T> Orth_QB;
    RandLAPACK::QB<T, RNG> QB;
    RandLAPACK::RSVD<T, RNG> RSVD;
    RandLAPACK::ABRIK<T, RNG> ABRIK;

    ABRIK_algorithm_objects(
        bool verbosity,
        bool cond_check,
        bool orth_check,
        bool time_subroutines,
        int64_t p,
        int64_t passes_per_iteration,
        int64_t block_sz,
        T tol
    ) :
        Stab(cond_check, verbosity),
        RS(Stab, p, passes_per_iteration, verbosity, cond_check),
        Orth_RF(cond_check, verbosity),
        RF(RS, Orth_RF, verbosity, cond_check),
        Orth_QB(cond_check, verbosity),
        QB(RF, Orth_QB, verbosity, orth_check),
        RSVD(QB, block_sz),
        ABRIK(verbosity, time_subroutines, tol)
        {}
};

// Re-generate input matrix A and its Eigen copy A_spectra.
template <typename T, typename RNG>
static void regen_input(RandLAPACK::gen::mat_gen_info<T> m_info,
                        ABRIK_benchmark_data<T> &all_data,
                        RandBLAS::RNGState<RNG> state) {
    using EMatrix = typename EigenTypes<T>::Matrix;
    auto m = all_data.row;
    auto n = all_data.col;
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);
    Eigen::Map<EMatrix>(all_data.A_spectra.data(), m, n) = Eigen::Map<const EMatrix>(all_data.A, m, n);
}

// Computes the residual norm error: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F).
// Scratch buffers are allocated and freed locally.
template <typename T>
static T
residual_error_comp(T* A, int64_t m, int64_t n,
                    T* U, T* V, T* Sigma, int64_t target_rank) {

    T* U_cpy = new T[m * target_rank]();
    T* V_cpy = new T[n * target_rank]();

    // S^{-1}AV - U
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, target_rank, n,
               (T)1, A, m, V, n, (T)0, U_cpy, m);
    for (int i = 0; i < target_rank; ++i)
        blas::scal(m, (T)1 / Sigma[i], &U_cpy[m * i], 1);
    blas::axpy(m * target_rank, (T)-1, U, 1, U_cpy, 1);

    // (A'U)S^{-1} - V
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, n, target_rank, m,
               (T)1, A, m, U, m, (T)0, V_cpy, n);
    for (int i = 0; i < target_rank; ++i)
        blas::scal(n, (T)1 / Sigma[i], &V_cpy[i * n], 1);
    blas::axpy(n * target_rank, (T)-1, V, 1, V_cpy, 1);

    T nrm1 = lapack::lange(Norm::Fro, m, target_rank, U_cpy, m);
    T nrm2 = lapack::lange(Norm::Fro, n, target_rank, V_cpy, n);

    delete[] U_cpy;
    delete[] V_cpy;

    return std::hypot(nrm1, nrm2);
}

// ---- ABRIK sweep ----
// Loops over (b_sz, num_matmuls, run). ABRIK allocates with new[] internally.
template <typename T, typename RNG>
static void run_abrik_sweep(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    std::vector<int64_t> &b_sz_vec,
    std::vector<int64_t> &matmuls_vec,
    int64_t target_rank,
    ABRIK_algorithm_objects<T, RNG> &all_algs,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    auto m = all_data.row;
    auto n = all_data.col;

    for (auto b_sz : b_sz_vec) {
        for (auto num_matmuls : matmuls_vec) {
            all_algs.RSVD.block_sz = b_sz;
            all_algs.ABRIK.max_krylov_iters = (int) num_matmuls;

            for (int i = 0; i < num_runs; ++i) {
                auto state_alg = state;
                printf("\nABRIK: b_sz=%ld, mm=%ld, run %d\n", b_sz, num_matmuls, i);

                auto start = steady_clock::now();
                all_algs.ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
                auto stop = steady_clock::now();
                long dur = duration_cast<microseconds>(stop - start).count();

                int64_t k_found = std::min(target_rank, all_algs.ABRIK.singular_triplets_found);
                T err = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, k_found);
                printf("  err=%.16e, time=%ld us\n", err, dur);

                outfile << "ABRIK, " << b_sz << ", " << num_matmuls << ", 0, "
                        << target_rank << ", " << err << ", " << dur << "\n";
                outfile.flush();

                delete[] all_data.U;     all_data.U     = nullptr;
                delete[] all_data.V;     all_data.V     = nullptr;
                delete[] all_data.Sigma; all_data.Sigma = nullptr;
                regen_input(m_info, all_data, state);
            }
        }
    }
}

// ---- ABRIK adaptive sweep ----
// Runs ABRIK in adaptive mode: starts with initial_iters Krylov iterations,
// then resumes (adding adaptive_increment per retry) until residual <= tol.
// Reports the total iterations used and the final accuracy.
template <typename T, typename RNG>
static void run_abrik_adaptive_sweep(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    std::vector<int64_t> &b_sz_vec,
    int64_t initial_iters,
    int64_t adaptive_increment,
    int64_t target_rank,
    ABRIK_algorithm_objects<T, RNG> &all_algs,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    auto m = all_data.row;
    auto n = all_data.col;

    for (auto b_sz : b_sz_vec) {
        all_algs.RSVD.block_sz = b_sz;
        all_algs.ABRIK.adaptive = true;
        all_algs.ABRIK.adaptive_increment = (int) adaptive_increment;
        all_algs.ABRIK.adaptive_max_retries = 50;

        for (int i = 0; i < num_runs; ++i) {
            auto state_alg = state;
            all_algs.ABRIK.max_krylov_iters = (int) initial_iters;

            printf("\nABRIK_adaptive: b_sz=%ld, init_iters=%ld, increment=%ld, run %d\n",
                   b_sz, initial_iters, adaptive_increment, i);

            auto start = steady_clock::now();
            all_algs.ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
            auto stop = steady_clock::now();
            long dur = duration_cast<microseconds>(stop - start).count();

            int64_t k_found = std::min(target_rank, all_algs.ABRIK.singular_triplets_found);
            int total_iters = all_algs.ABRIK.num_krylov_iters;
            T err = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, k_found);
            printf("  err=%.16e, time=%ld us, total_iters=%d\n", err, dur, total_iters);

            // Write as ABRIK_adaptive; use num_matmuls column for total_iters
            outfile << "ABRIK_adaptive, " << b_sz << ", " << total_iters << ", 0, "
                    << target_rank << ", " << err << ", " << dur << "\n";
            outfile.flush();

            delete[] all_data.U;     all_data.U     = nullptr;
            delete[] all_data.V;     all_data.V     = nullptr;
            delete[] all_data.Sigma; all_data.Sigma = nullptr;
            regen_input(m_info, all_data, state);
        }
    }

    // Restore non-adaptive mode for subsequent sweeps
    all_algs.ABRIK.adaptive = false;
}

// ---- RSVD sweep ----
// Loops over (p_value, run). Sets passes_over_data at runtime.
// RSVD allocates with calloc internally — must use free().
// k is passed by reference and may be reduced by QB — use local copy per call.
template <typename T, typename RNG>
static void run_rsvd_sweep(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    std::vector<int64_t> &p_values,
    int64_t target_rank,
    ABRIK_algorithm_objects<T, RNG> &all_algs,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;

    for (auto p_val : p_values) {
        all_algs.RS.passes_over_data = p_val;
        all_algs.RSVD.block_sz = target_rank;

        for (int i = 0; i < num_runs; ++i) {
            auto state_alg = state;
            int64_t k = target_rank;  // local copy — QB may reduce
            printf("\nRSVD: p=%ld, run %d\n", p_val, i);

            auto start = steady_clock::now();
            all_algs.RSVD.call(m, n, all_data.A, k, tol, all_data.U, all_data.Sigma, all_data.V, state_alg);
            auto stop = steady_clock::now();
            long dur = duration_cast<microseconds>(stop - start).count();

            int64_t k_eval = std::min(target_rank, k);
            T err = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, k_eval);
            printf("  k=%ld, err=%.16e, time=%ld us\n", k, err, dur);

            outfile << "RSVD, 0, 0, " << p_val << ", "
                    << target_rank << ", " << err << ", " << dur << "\n";
            outfile.flush();

            free(all_data.U);     all_data.U     = nullptr;
            free(all_data.V);     all_data.V     = nullptr;
            free(all_data.Sigma); all_data.Sigma = nullptr;
            regen_input(m_info, all_data, state);
        }
    }
}

// ---- SVDS sweep (Spectra) ----
// Runs num_runs times with nev = target_rank.
// Spectra operates on the Eigen copy (A_spectra), not the raw A pointer.
template <typename T, typename RNG>
static void run_svds_sweep(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    int64_t target_rank,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    using EMatrix = typename EigenTypes<T>::Matrix;
    using EVector = typename EigenTypes<T>::Vector;

    auto m = all_data.row;
    auto n = all_data.col;
    int64_t nev = target_rank;
    int64_t ncv = std::min(2 * nev + 1, n - 1);

    for (int i = 0; i < num_runs; ++i) {
        printf("\nSVDS: nev=%ld, ncv=%ld, run %d\n", nev, ncv, i);

        auto start = steady_clock::now();
        Spectra::PartialSVDSolver<EMatrix> svds(all_data.A_spectra, nev, ncv);
        svds.compute(1000, all_data.tolerance);
        auto stop = steady_clock::now();
        long dur = duration_cast<microseconds>(stop - start).count();

        EMatrix U_spectra = svds.matrix_U(nev);
        EMatrix V_spectra = svds.matrix_V(nev);
        EVector S_spectra = svds.singular_values();

        all_data.U     = new T[m * nev]();
        all_data.V     = new T[n * nev]();
        all_data.Sigma = new T[nev]();

        Eigen::Map<EMatrix>(all_data.U, m, nev) = U_spectra;
        Eigen::Map<EMatrix>(all_data.V, n, nev) = V_spectra;
        Eigen::Map<EVector>(all_data.Sigma, nev) = S_spectra;

        T err = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, nev);
        printf("  err=%.16e, time=%ld us\n", err, dur);

        outfile << "SVDS, 0, 0, 0, "
                << target_rank << ", " << err << ", " << dur << "\n";
        outfile.flush();

        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;
    }
}

// ---- GESDD benchmark ----
// Single deterministic run of full SVD.
template <typename T, typename RNG>
static void run_gesdd_benchmark(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t target_rank,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    auto m = all_data.row;
    auto n = all_data.col;
    printf("\nGESDD: full SVD\n");

    all_data.U     = new T[m * n]();
    all_data.Sigma = new T[n]();
    all_data.VT    = new T[n * n]();
    all_data.V     = new T[n * n]();

    auto start = steady_clock::now();
    lapack::gesdd(Job::SomeVec, m, n, all_data.A, m, all_data.Sigma, all_data.U, m, all_data.VT, n);
    auto stop = steady_clock::now();
    long dur = duration_cast<microseconds>(stop - start).count();
    printf("  time=%ld us\n", dur);

    // GESDD destroys A, re-read before residual computation.
    regen_input(m_info, all_data, state);
    RandLAPACK::util::transposition(n, n, all_data.VT, n, all_data.V, n, 0);

    T err = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, target_rank);
    printf("  err=%.16e\n", err);

    outfile << "GESDD, 0, 0, 0, "
            << target_rank << ", " << err << ", " << dur << "\n";
    outfile.flush();

    delete[] all_data.U;     all_data.U     = nullptr;
    delete[] all_data.VT;    all_data.VT    = nullptr;
    delete[] all_data.V;     all_data.V     = nullptr;
    delete[] all_data.Sigma; all_data.Sigma = nullptr;
    regen_input(m_info, all_data, state);
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {
    using EMatrix = typename EigenTypes<T>::Matrix;

    // New CLI format:
    // <precision> <output_dir> <input> <num_runs> <num_rows> <num_cols> <target_rank>
    // <run_gesdd> <num_block_sizes> <num_matmul_sizes> <num_p_values>
    // <block_sizes...> <matmul_sizes...> <p_values...>
    if (argc < 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_matrix_path> <num_runs>"
                  << " <num_rows> <num_cols> <target_rank> <run_gesdd>"
                  << " <num_block_sizes> <num_matmul_sizes> <num_p_values>"
                  << " <block_sizes...> <matmul_sizes...> <p_values...>" << std::endl;
        return;
    }

    int num_runs              = std::stol(argv[4]);
    int64_t m_expected        = std::stol(argv[5]);
    int64_t n_expected        = std::stol(argv[6]);
    int64_t target_rank       = std::stol(argv[7]);
    bool run_gesdd            = (std::stoi(argv[8]) != 0);
    int64_t num_block_sizes   = std::stol(argv[9]);
    int64_t num_matmul_sizes  = std::stol(argv[10]);
    int64_t num_p_values      = std::stol(argv[11]);

    int64_t expected_argc = 12 + num_block_sizes + num_matmul_sizes + num_p_values;
    if (argc < expected_argc) {
        std::cerr << "Error: expected " << expected_argc << " arguments, got " << argc << std::endl;
        return;
    }

    // Parse block sizes (for ABRIK)
    int64_t offset = 12;
    std::vector<int64_t> b_sz;
    for (int64_t i = 0; i < num_block_sizes; ++i)
        b_sz.push_back(std::stol(argv[offset + i]));
    offset += num_block_sizes;

    // Parse matmul counts (for ABRIK)
    std::vector<int64_t> matmuls;
    for (int64_t i = 0; i < num_matmul_sizes; ++i)
        matmuls.push_back(std::stol(argv[offset + i]));
    offset += num_matmul_sizes;

    // Parse RSVD power iteration values
    std::vector<int64_t> p_values;
    for (int64_t i = 0; i < num_p_values; ++i)
        p_values.push_back(std::stol(argv[offset + i]));

    // Build display strings for metadata
    auto vec_to_string = [](const std::vector<int64_t> &v) {
        std::ostringstream oss;
        for (const auto &val : v) oss << val << ", ";
        return oss.str();
    };
    std::string b_sz_string    = vec_to_string(b_sz);
    std::string matmuls_string = vec_to_string(matmuls);
    std::string p_val_string   = vec_to_string(p_values);

    T tol              = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state         = RandBLAS::RNGState();
    auto state_constant = state;
    int64_t m = 0, n = 0;

    // Read the input matrix (workspace query first, then actual read).
    RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[3];
    m_info.workspace_query_mod = 1;
    RandLAPACK::gen::mat_gen<T>(m_info, NULL, state);

    m = m_info.rows;
    n = m_info.cols;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected input size (" << m_expected << ", " << n_expected
                  << ") did not match actual input size (" << m << ", " << n << "). Aborting." << std::endl;
        return;
    }

    ABRIK_benchmark_data<T> all_data(m, n, tol);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Initial p=2, passes_per_iteration=1 — p gets overridden per RSVD call.
    int64_t p_init = 2;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 0;
    ABRIK_algorithm_objects<T, r123::Philox4x32> all_algs(false, false, false, false, p_init, passes_per_iteration, block_sz, tol);

    Eigen::Map<EMatrix>(all_data.A_spectra.data(), m, n) = Eigen::Map<const EMatrix>(all_data.A, m, n);
    printf("Finished data preparation\n");

    // Generate date/time prefix for output filename
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string output_filename = std::string(date_prefix) + "ABRIK_speed_comparisons.csv";
    std::string path;
    if (std::string(argv[2]) != ".") {
        path = std::string(argv[2]) + "/" + output_filename;
    } else {
        path = output_filename;
    }
    std::ofstream file(path, std::ios::out | std::ios::app);

    // Write metadata header (prefixed with # for easy parsing)
    file << "# ABRIK Speed Comparison Benchmark\n"
         << "# Precision: " << argv[1] << "\n"
         << "# Input matrix: " << argv[3] << "\n"
         << "# Input size: " << m << " x " << n << "\n"
         << "# Target rank: " << target_rank << "\n"
         << "# Krylov block sizes: " << b_sz_string << "\n"
         << "# Matmul counts: " << matmuls_string << "\n"
         << "# RSVD p values: " << p_val_string << "\n"
         << "# Runs per configuration: " << num_runs << "\n"
         << "# Tolerance: " << tol << "\n"
         << "# Run GESDD: " << (run_gesdd ? "yes" : "no") << "\n"
         << "# Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F)\n"
         << "# Timings in microseconds\n";

    // Write CSV column header
    file << "algorithm, b_sz, num_matmuls, p, target_rank, error, duration_us\n";
    file.flush();

    // Run each algorithm sweep independently
    printf("\n=== ABRIK sweep (%zu block sizes x %zu matmul counts x %d runs) ===\n",
           b_sz.size(), matmuls.size(), num_runs);
    run_abrik_sweep(m_info, num_runs, b_sz, matmuls, target_rank, all_algs, all_data, state_constant, file);

    printf("\n=== ABRIK adaptive sweep (%zu block sizes x %d runs, init=4, incr=4) ===\n",
           b_sz.size(), num_runs);
    run_abrik_adaptive_sweep(m_info, num_runs, b_sz, /*initial_iters=*/4, /*adaptive_increment=*/4,
                             target_rank, all_algs, all_data, state_constant, file);

    printf("\n=== RSVD sweep (%zu p values x %d runs) ===\n",
           p_values.size(), num_runs);
    run_rsvd_sweep(m_info, num_runs, p_values, target_rank, all_algs, all_data, state_constant, file);

    printf("\n=== SVDS sweep (%d runs, nev=%ld) ===\n", num_runs, target_rank);
    run_svds_sweep(m_info, num_runs, target_rank, all_data, state_constant, file);

    if (run_gesdd) {
        printf("\n=== GESDD (single run) ===\n");
        run_gesdd_benchmark(m_info, target_rank, all_data, state_constant, file);
    }

    printf("\nBenchmark complete. Output: %s\n", path.c_str());
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_matrix_path> <num_runs>"
                  << " <num_rows> <num_cols> <target_rank> <run_gesdd>"
                  << " <num_block_sizes> <num_matmul_sizes> <num_p_values>"
                  << " <block_sizes...> <matmul_sizes...> <p_values...>" << std::endl;
        return 1;
    }

    std::string precision = argv[1];
    if (precision == "double") {
        run_benchmark<double>(argc, argv);
    } else if (precision == "float") {
        run_benchmark<float>(argc, argv);
    } else {
        std::cerr << "Error: precision must be 'double' or 'float', got '" << precision << "'" << std::endl;
        return 1;
    }
    return 0;
}
