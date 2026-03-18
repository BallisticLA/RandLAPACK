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

template <typename T, typename RNG>
static void call_all_algs(
    RandLAPACK::gen::mat_gen_info<T> m_info,
    int64_t num_runs,
    int64_t b_sz,
    int64_t num_matmuls,
    int64_t target_rank,
    bool run_gesdd,
    ABRIK_algorithm_objects<T, RNG> &all_algs,
    ABRIK_benchmark_data<T> &all_data,
    RandBLAS::RNGState<RNG> &state,
    std::ofstream &outfile) {

    using EMatrix = typename EigenTypes<T>::Matrix;
    using EVector = typename EigenTypes<T>::Vector;

    auto m   = all_data.row;
    auto n   = all_data.col;
    auto tol = all_data.tolerance;

    // Additional params setup.
    all_algs.RSVD.block_sz = b_sz;
    // Instead of computing max_krylov_iters from target_rank, we pre-specify
    // the maximum number of Krylov iters via num_matmuls.
    all_algs.ABRIK.max_krylov_iters = (int) num_matmuls;

    // timing vars
    long dur_ABRIK = 0;
    long dur_rsvd = 0;
    long dur_svds = 0;
    long dur_svd  = 0;

    auto state_alg = state;

    T residual_err_custom_SVD   = 0;
    T residual_err_custom_ABRIK = 0;
    T residual_err_custom_RSVD  = 0;
    T residual_err_custom_SVDS  = 0;

    int64_t singular_triplets_target_ABRIK = 0;
    int64_t singular_triplets_found_RSVD  = 0;
    int64_t singular_triplets_target_RSVD = 0;
    int64_t singular_triplets_found_SVDS  = 0;
    int64_t singular_triplets_target_SVDS = 0;

    for (int i = 0; i < num_runs; ++i) {
        printf("\nBlock size %ld, num matmuls %ld. Iteration %d start.\n", b_sz, num_matmuls, i);

        // ---- ABRIK ----
        // ABRIK allocates U, V, Sigma with new[] internally.
        auto start_ABRIK = steady_clock::now();
        all_algs.ABRIK.call(m, n, all_data.A, m, b_sz, all_data.U, all_data.V, all_data.Sigma, state_alg);
        auto stop_ABRIK = steady_clock::now();
        dur_ABRIK = duration_cast<microseconds>(stop_ABRIK - start_ABRIK).count();
        printf("TOTAL TIME FOR ABRIK %ld\n", dur_ABRIK);

        singular_triplets_target_ABRIK = std::min(target_rank, all_algs.ABRIK.singular_triplets_found);
        residual_err_custom_ABRIK = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, singular_triplets_target_ABRIK);
        printf("ABRIK residual error: %.16e\n", residual_err_custom_ABRIK);

        // Cleanup ABRIK outputs (new[])
        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;

        state_alg = state;
        regen_input(m_info, all_data, state);

        // ---- RSVD ----
        // RSVD allocates U, V, Sigma with calloc internally (via QB realloc chain).
        // Do NOT pre-allocate — RSVD overwrites the pointers.
        singular_triplets_found_RSVD = (int64_t) (b_sz * num_matmuls / 2);

        auto start_rsvd = steady_clock::now();
        all_algs.RSVD.call(m, n, all_data.A, singular_triplets_found_RSVD, tol, all_data.U, all_data.Sigma, all_data.V, state_alg);
        auto stop_rsvd = steady_clock::now();
        dur_rsvd = duration_cast<microseconds>(stop_rsvd - start_rsvd).count();
        printf("TOTAL TIME FOR RSVD %ld\n", dur_rsvd);

        singular_triplets_target_RSVD = std::min(target_rank, singular_triplets_found_RSVD);
        residual_err_custom_RSVD = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, singular_triplets_target_RSVD);
        printf("RSVD residual error: %.16e\n", residual_err_custom_RSVD);

        // Cleanup RSVD outputs (calloc)
        free(all_data.U);     all_data.U     = nullptr;
        free(all_data.V);     all_data.V     = nullptr;
        free(all_data.Sigma); all_data.Sigma = nullptr;

        state_alg = state;
        regen_input(m_info, all_data, state);

        // ---- SVDS (Spectra) ----
        // Despite my earlier expectations, estimating a larger number of
        // singular triplets via SVDS does improve the quality of the first singular triplets.
        // As such, aiming for just the "target rank" would be unfair.
        singular_triplets_found_SVDS = std::min((int64_t) (b_sz * num_matmuls / 2), n - 2);

        auto start_svds = steady_clock::now();
        Spectra::PartialSVDSolver<EMatrix> svds(all_data.A_spectra, singular_triplets_found_SVDS, std::min(2 * singular_triplets_found_SVDS, n - 1));
        svds.compute();
        auto stop_svds = steady_clock::now();
        dur_svds = duration_cast<microseconds>(stop_svds - start_svds).count();
        printf("TOTAL TIME FOR SVDS %ld\n", dur_svds);

        // Copy data from Spectra (Eigen) format to raw arrays.
        EMatrix U_spectra = svds.matrix_U(singular_triplets_found_SVDS);
        EMatrix V_spectra = svds.matrix_V(singular_triplets_found_SVDS);
        EVector S_spectra = svds.singular_values();

        all_data.U     = new T[m * singular_triplets_found_SVDS]();
        all_data.V     = new T[n * singular_triplets_found_SVDS]();
        all_data.Sigma = new T[singular_triplets_found_SVDS]();

        Eigen::Map<EMatrix>(all_data.U, m, singular_triplets_found_SVDS)  = U_spectra;
        Eigen::Map<EMatrix>(all_data.V, n, singular_triplets_found_SVDS)  = V_spectra;
        Eigen::Map<EVector>(all_data.Sigma, singular_triplets_found_SVDS) = S_spectra;

        singular_triplets_target_SVDS = std::min(target_rank, singular_triplets_found_SVDS);
        residual_err_custom_SVDS = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, singular_triplets_target_SVDS);
        printf("SVDS residual error: %.16e\n", residual_err_custom_SVDS);

        // Cleanup SVDS outputs (new[])
        delete[] all_data.U;     all_data.U     = nullptr;
        delete[] all_data.V;     all_data.V     = nullptr;
        delete[] all_data.Sigma; all_data.Sigma = nullptr;

        state_alg = state;
        regen_input(m_info, all_data, state);

        // ---- SVD (GESDD) ----
        // There is no reason to run SVD many times, as it always outputs the same result.
        if (run_gesdd && (i == 0)) {
            auto start_svd = steady_clock::now();
            all_data.U     = new T[m * n]();
            all_data.Sigma = new T[n]();
            all_data.VT    = new T[n * n]();
            all_data.V     = new T[n * n]();
            lapack::gesdd(Job::SomeVec, m, n, all_data.A, m, all_data.Sigma, all_data.U, m, all_data.VT, n);
            auto stop_svd = steady_clock::now();
            dur_svd = duration_cast<microseconds>(stop_svd - start_svd).count();
            printf("TOTAL TIME FOR SVD %ld\n", dur_svd);

            // GESDD destroys A, re-read before residual computation.
            regen_input(m_info, all_data, state);
            RandLAPACK::util::transposition(n, n, all_data.VT, n, all_data.V, n, 0);

            residual_err_custom_SVD = residual_error_comp<T>(all_data.A, m, n, all_data.U, all_data.V, all_data.Sigma, target_rank);
            printf("SVD residual error: %.16e\n", residual_err_custom_SVD);

            // Cleanup SVD outputs (new[])
            delete[] all_data.U;     all_data.U     = nullptr;
            delete[] all_data.VT;    all_data.VT    = nullptr;
            delete[] all_data.V;     all_data.V     = nullptr;
            delete[] all_data.Sigma; all_data.Sigma = nullptr;

            state_alg = state;
            regen_input(m_info, all_data, state);
        }

        // Write CSV data row
        outfile << b_sz << ", " << all_algs.ABRIK.max_krylov_iters << ", " << target_rank << ", "
                << residual_err_custom_ABRIK << ", " << dur_ABRIK << ", "
                << residual_err_custom_RSVD  << ", " << dur_rsvd  << ", "
                << residual_err_custom_SVDS  << ", " << dur_svds  << ", "
                << residual_err_custom_SVD   << ", " << dur_svd   << "\n";
        outfile.flush();
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {
    using EMatrix = typename EigenTypes<T>::Matrix;

    if (argc < 12) {
        std::cerr << "Usage: " << argv[0] << " <precision> <output_directory_path> <input_matrix_path> <num_runs> <num_rows> <num_cols> <target_rank> <run_gesdd> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
        return;
    }

    int num_runs              = std::stol(argv[4]);
    int64_t m_expected        = std::stol(argv[5]);
    int64_t n_expected        = std::stol(argv[6]);
    int64_t target_rank       = std::stol(argv[7]);
    bool run_gesdd            = (std::stoi(argv[8]) != 0);
    std::vector<int64_t> b_sz;
    for (int i = 0; i < std::stol(argv[9]); ++i)
        b_sz.push_back(std::stoi(argv[i + 11]));
    std::ostringstream oss1;
    for (const auto &val : b_sz)
        oss1 << val << ", ";
    std::string b_sz_string = oss1.str();
    std::vector<int64_t> matmuls;
    for (int i = 0; i < std::stol(argv[10]); ++i)
        matmuls.push_back(std::stoi(argv[i + 11 + std::stol(argv[9])]));
    std::ostringstream oss2;
    for (const auto &val : matmuls)
        oss2 << val << ", ";
    std::string matmuls_string = oss2.str();
    T tol                     = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state                = RandBLAS::RNGState();
    auto state_constant       = state;
    int64_t m = 0, n = 0;

    // Generate the input matrix.
    RandLAPACK::gen::mat_gen_info<T> m_info(m, n, RandLAPACK::gen::custom_input);
    m_info.filename = argv[3];
    m_info.workspace_query_mod = 1;
    RandLAPACK::gen::mat_gen<T>(m_info, NULL, state);

    // Update basic params.
    m = m_info.rows;
    n = m_info.cols;
    if (m_expected != m || n_expected != n) {
        std::cerr << "Expected input size (" << m_expected << ", " << n_expected << ") did not match actual input size (" << m << ", " << n << "). Aborting." << std::endl;
        return;
    }

    // Allocate basic workspace.
    ABRIK_benchmark_data<T> all_data(m, n, tol);
    RandLAPACK::gen::mat_gen(m_info, all_data.A, state);

    // Declare objects for RSVD and ABRIK
    int64_t p = 2;
    int64_t passes_per_iteration = 1;
    int64_t block_sz = 0;
    ABRIK_algorithm_objects<T, r123::Philox4x32> all_algs(false, false, false, false, p, passes_per_iteration, block_sz, tol);

    // Copying input data into a Spectra (Eigen) matrix object
    Eigen::Map<EMatrix>(all_data.A_spectra.data(), m, n) = Eigen::Map<const EMatrix>(all_data.A, m, n);

    printf("Finished data preparation\n");

    // Generate date/time prefix
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    // Build output file path
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
         << "# Runs per configuration: " << num_runs << "\n"
         << "# Tolerance: " << tol << "\n"
         << "# Run GESDD: " << (run_gesdd ? "yes" : "no") << "\n"
         << "# Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F)\n"
         << "# Timings in microseconds\n";
    // Write CSV column header
    file << "b_sz, num_matmuls, target_rank, "
         << "err_ABRIK, dur_ABRIK, "
         << "err_RSVD, dur_RSVD, "
         << "err_SVDS, dur_SVDS, "
         << "err_SVD, dur_SVD\n";
    file.flush();

    size_t i = 0, j = 0;
    for (; i < b_sz.size(); ++i) {
        for (; j < matmuls.size(); ++j) {
            call_all_algs(m_info, num_runs, b_sz[i], matmuls[j], target_rank, run_gesdd, all_algs, all_data, state_constant, file);
        }
        j = 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <precision: double|float> <output_directory_path> <input_matrix_path> <num_runs> <num_rows> <num_cols> <target_rank> <run_gesdd> <num_block_sizes> <num_matmul_sizes> <block_sizes> <mat_sizes>" << std::endl;
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
