/*
Unified ABRIK speed comparison benchmark — budgeted checkpointing mode.

Reads input from .mtx (Matrix Market) or .txt/.bin (dense) files.
Runs ABRIK (4 block sizes), Spectra, and RSVD (largest block size).

All algorithms run to a fixed total matvec budget. ABRIK uses BK call()/resume()
so no work is repeated; a convergence curve is produced in a single pass.

Output CSV (long format, one data point per row):
  method, b_sz, total_matvecs, err, elapsed_us

  method      = ABRIK | Spectra | RSVD | GESDD
  b_sz        = block size (0 for Spectra/GESDD)
  total_matvecs = matrix-vector products consumed
  err         = SVD residual: sqrt(||S^{-1}AV-U||^2 + ||(A'U)S^{-1}-V||^2)
  elapsed_us  = wall-clock microseconds
                ABRIK:   cumulative BK + SVD extraction (not residual check)
                Spectra/RSVD: wall clock for that independent call

Usage:
  ABRIK_speed_comparisons <precision> <output_dir> <input_file>
                          <target_rank> <run_gesdd> <budget>
                          <num_block_sizes> <block_sizes...>
                          [sub_ratio] [use_cqrrt]

  budget       = total matvec budget (e.g. 4096)
  block_sizes  = block sizes for ABRIK (e.g. 4 8 16 32)
                 RSVD uses the largest; Spectra uses a fixed ncv schedule.
  run_gesdd    = 1 to run GESDD once (dense input only)
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_svd_residual.hh"
#include "ext_matrix_io.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>
#include <algorithm>

// External libs
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Spectra/contrib/PartialSVDSolver.h>
#include "ext_budgeted_svd_solver.hh"

#include <execinfo.h>
#include <exception>
#include <unistd.h>

static void abrik_terminate_handler() {
    void* trace[64];
    int n = backtrace(trace, 64);

    fprintf(stderr, "\n=== UNCAUGHT EXCEPTION -- TERMINATE ===\n");
    auto eptr = std::current_exception();
    if (eptr) {
        try { std::rethrow_exception(eptr); }
        catch (const std::exception& e) {
            fprintf(stderr, "what(): %s\n", e.what());
        }
        catch (...) {
            fprintf(stderr, "(non-std::exception)\n");
        }
    }

    fprintf(stderr, "Backtrace (%d frames):\n", n);
    backtrace_symbols_fd(trace, n, STDERR_FILENO);
    fprintf(stderr, "Tip: c++filt for names, addr2line -e <binary> <addr> for source:line\n");
    fflush(stderr);

    std::abort();
}

// Eigen type traits
template <typename T> struct EigenTypes;
template <> struct EigenTypes<double> {
    using Matrix   = Eigen::MatrixXd;
    using Vector   = Eigen::VectorXd;
    using SpMatrix = Eigen::SparseMatrix<double>;
};
template <> struct EigenTypes<float> {
    using Matrix   = Eigen::MatrixXf;
    using Vector   = Eigen::VectorXf;
    using SpMatrix = Eigen::SparseMatrix<float>;
};

// Algorithm objects bundle
template <typename T, typename RNG>
struct AlgorithmObjects {
    RandLAPACK::PLUL<T> Stab;
    RandLAPACK::RS<T, RNG> RS;
    RandLAPACK::CholQRQ<T> Orth_RF;
    RandLAPACK::RF<T, RNG> RF;
    RandLAPACK::CholQRQ<T> Orth_QB;
    RandLAPACK::QB<T, RNG> QB;
    RandLAPACK::RSVD<T, RNG> RSVD;
    RandLAPACK::ABRIK<T, RNG> ABRIK;

    AlgorithmObjects(int64_t block_sz, T tol)
        : Stab(false, false),
          RS(Stab, 2, 1, false, false),
          Orth_RF(false, false),
          RF(RS, Orth_RF, false, false),
          Orth_QB(false, false),
          QB(RF, Orth_QB, false, false),
          RSVD(QB, block_sz),
          ABRIK(false, false, tol)
    {}
};

// LinOp-based residual
template <typename T, RandLAPACK::linops::LinearOperator LinOp>
static T residual_via_linop(LinOp& A_op, T* U, T* V, T* Sigma, int64_t k) {
    return RandLAPACK::linops::svd_residual<T>(A_op, U, V, Sigma, k);
}

// Run Spectra with a fixed total matvec budget.
template <typename T, typename EigenMatType, RandLAPACK::linops::LinearOperator LinOp>
static T run_svds(const EigenMatType& A_eigen, LinOp& A_op,
                  int64_t budget_mv, int64_t target_rank, long& dur_svds) {
    using EMatrix = typename EigenTypes<T>::Matrix;
    using EVector = typename EigenTypes<T>::Vector;

    int64_t m = A_op.n_rows;
    int64_t n = A_op.n_cols;
    int64_t nev = target_rank;
    int64_t ncv_default = std::min(2 * nev + 1, n - 1);
    int64_t ncv = BenchmarkUtil::effective_ncv(budget_mv, nev, ncv_default);
    int64_t max_restarts = BenchmarkUtil::budget_to_restarts(budget_mv, nev, ncv);

    auto t0 = steady_clock::now();
    BenchmarkUtil::BudgetedPartialSVDSolver<EigenMatType> svds(A_eigen, nev, ncv);
    svds.compute(max_restarts);
    dur_svds = duration_cast<microseconds>(steady_clock::now() - t0).count();

    EMatrix U_sp = svds.matrix_U(nev);
    EMatrix V_sp = svds.matrix_V(nev);
    EVector S_sp = svds.singular_values();

    T* U_s = new T[m * nev](); T* V_s = new T[n * nev](); T* S_s = new T[nev]();
    Eigen::Map<EMatrix>(U_s, m, nev) = U_sp;
    Eigen::Map<EMatrix>(V_s, n, nev) = V_sp;
    Eigen::Map<EVector>(S_s, nev)    = S_sp;

    T err = residual_via_linop(A_op, U_s, V_s, S_s, nev);
    delete[] U_s; delete[] V_s; delete[] S_s;
    return err;
}

// Generate checkpoint_matvecs as powers of 2 × step, up to budget.
// step = smallest block size (= 1 Krylov iteration × smallest b_sz).
static std::vector<int64_t> make_checkpoint_matvecs(int64_t step, int64_t budget) {
    std::vector<int64_t> cps;
    for (int64_t mv = step; mv < budget; mv *= 2)
        cps.push_back(mv);
    cps.push_back(budget);  // always include the full budget as the last checkpoint
    return cps;
}

// Core benchmark: runs all algorithms with checkpointing.
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp, typename SvdsFn>
static void run_with_budget(
    LinOp& A_op,
    SvdsFn svds_fn,
    T norm_A,
    int64_t target_rank,
    bool run_gesdd,
    bool use_cqrrt,
    T* A_dense_buf,
    std::vector<int64_t>& block_sizes,
    int64_t budget,
    AlgorithmObjects<T, RNG>& algs,
    RandBLAS::RNGState<RNG>& state,
    std::ofstream& outfile)
{
    int64_t m = A_op.n_rows;
    int64_t n = A_op.n_cols;
    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);

    int64_t min_b = *std::min_element(block_sizes.begin(), block_sizes.end());
    int64_t max_b = *std::max_element(block_sizes.begin(), block_sizes.end());
    std::vector<int64_t> checkpoint_matvecs = make_checkpoint_matvecs(min_b, budget);

    if (use_cqrrt)
        algs.ABRIK.qr_exp = RandLAPACK::ABRIKSubroutines::QR_explicit::cqrrt;

    // ---- ABRIK: one call_with_checkpoints per block size ----
    for (auto b_sz : block_sizes) {
        printf("\n=== ABRIK b=%ld ===\n", b_sz);
        // Krylov iteration counts for this block size: all checkpoints reachable in >= 1 iter
        std::vector<int64_t> cp_iters;
        for (auto mv : checkpoint_matvecs)
            if (mv >= b_sz)
                cp_iters.push_back(mv / b_sz);

        auto state_alg = state;
        algs.ABRIK.call_with_checkpoints(A_op, b_sz, target_rank, cp_iters,
            [&](int64_t total_mv, long elapsed_us, T residual) {
                outfile << "ABRIK, " << b_sz << ", " << total_mv << ", "
                        << residual << ", " << elapsed_us << "\n";
                outfile.flush();
                printf("  mv=%ld  err=%e  t=%ld us\n", total_mv, residual, elapsed_us);
            }, state_alg);
    }

    // ---- Spectra: one independent call per checkpoint budget ----
    printf("\n=== Spectra ===\n");
    for (auto budget_mv : checkpoint_matvecs) {
        long dur_svds = 0;
        T err_svds = svds_fn(budget_mv, dur_svds);
        outfile << "Spectra, 0, " << budget_mv << ", " << err_svds << ", " << dur_svds << "\n";
        outfile.flush();
        printf("  mv=%ld  err=%e  t=%ld us\n", budget_mv, err_svds, dur_svds);
    }

    // ---- RSVD: one independent call per checkpoint budget, largest block size ----
    printf("\n=== RSVD b=%ld ===\n", max_b);
    algs.RSVD.block_sz = max_b;
    for (auto budget_mv : checkpoint_matvecs) {
        int64_t k_r = std::max((int64_t)1, budget_mv / 2);
        T* U_r = nullptr, *V_r = nullptr, *S_r = nullptr;
        auto state_rsvd = state;  // fresh state for each independent RSVD call
        auto t0 = steady_clock::now();
        algs.RSVD.call(A_op, norm_A, k_r, tol, U_r, S_r, V_r, state_rsvd);
        long dur_rsvd = duration_cast<microseconds>(steady_clock::now() - t0).count();
        int64_t k_r_target = std::min(target_rank, k_r);
        T err_rsvd = residual_via_linop(A_op, U_r, V_r, S_r, k_r_target);
        free(U_r); free(V_r); free(S_r);
        outfile << "RSVD, " << max_b << ", " << budget_mv << ", " << err_rsvd << ", " << dur_rsvd << "\n";
        outfile.flush();
        printf("  mv=%ld  k_r=%ld  err=%e  t=%ld us\n", budget_mv, k_r, err_rsvd, dur_rsvd);
    }

    // ---- GESDD: dense only, once ----
    if (run_gesdd && A_dense_buf) {
        printf("\n=== GESDD ===\n");
        T* A_svd = new T[m * n];
        lapack::lacpy(MatrixType::General, m, n, A_dense_buf, m, A_svd, m);
        T* U_g = new T[m * n]();
        T* S_g = new T[n]();
        T* VT_g = new T[n * n]();
        T* V_g = new T[n * n]();

        auto t0 = steady_clock::now();
        lapack::gesdd(Job::SomeVec, m, n, A_svd, m, S_g, U_g, m, VT_g, n);
        long dur_svd = duration_cast<microseconds>(steady_clock::now() - t0).count();

        RandLAPACK::util::transposition(n, n, VT_g, n, V_g, n, 0);
        T err_SVD = residual_via_linop(A_op, U_g, V_g, S_g, target_rank);
        printf("  err=%e  t=%ld us\n", err_SVD, dur_svd);

        outfile << "GESDD, 0, 0, " << err_SVD << ", " << dur_svd << "\n";
        outfile.flush();

        delete[] A_svd; delete[] U_g; delete[] S_g; delete[] VT_g; delete[] V_g;
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {
    using EMatrix = typename EigenTypes<T>::Matrix;

    if (argc < 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_file> <target_rank> <run_gesdd>"
                  << " <budget> <num_block_sizes> <block_sizes...>"
                  << " [sub_ratio] [use_cqrrt]\n";
        return;
    }

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int64_t target_rank    = std::stol(argv[4]);
    bool run_gesdd         = (std::stoi(argv[5]) != 0);
    int64_t budget         = std::stol(argv[6]);
    int num_b_sz           = std::stoi(argv[7]);

    std::vector<int64_t> block_sizes;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stol(argv[8 + i]));

    int args_consumed = 8 + num_b_sz;
    double sub_ratio  = (argc > args_consumed)     ? std::stod(argv[args_consumed])     : 1.0;
    bool cli_cqrrt    = (argc > args_consumed + 1) ? (std::stoi(argv[args_consumed + 1]) != 0) : false;

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state = RandBLAS::RNGState();

    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;

    AlgorithmObjects<T, r123::Philox4x32> algs(0, tol);

    // Open output CSV
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string out_filename = std::string(date_prefix) + "ABRIK_speed_comparisons.csv";
    std::string out_path = (output_dir != ".") ? output_dir + "/" + out_filename : out_filename;
    std::ofstream outfile(out_path);

    std::ostringstream oss_b;
    for (auto v : block_sizes) oss_b << v << ", ";

    outfile << "# ABRIK Speed Comparison Benchmark (budgeted checkpointing)\n"
            << "# Precision: " << argv[1] << "\n"
            << "# Input matrix: " << input_path << "\n"
            << "# Input size: " << m << " x " << n << "\n"
            << "# Format: " << (mat.is_sparse ? "sparse" : "dense") << "\n"
            << "# Target rank: " << target_rank << "\n"
            << "# Budget (total matvecs): " << budget << "\n"
            << "# Block sizes: " << oss_b.str() << "\n"
            << "# RSVD uses largest block size with k_r = budget_mv / 2\n"
            << "# Tolerance: " << tol << "\n"
            << "# ABRIK elapsed = cumulative BK + SVD extraction (not residual eval)\n"
            << "# Spectra/RSVD elapsed = wall clock for that independent call\n"
            << "# Residual: sqrt(||S^{-1}AV-U||^2_F + ||(A'U)S^{-1}-V||^2_F)\n";
    outfile << "method, b_sz, total_matvecs, err, elapsed_us\n";
    outfile.flush();

    auto t_total = steady_clock::now();

    if (mat.is_sparse) {
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>>
            A_op(m, n, *mat.csc);
        T norm_A = A_op.fro_nrm();

        auto svds_fn = [&](int64_t budget_mv, long& dur) -> T {
            return run_svds<T>(*mat.eigen_sparse, A_op, budget_mv, target_rank, dur);
        };

        run_with_budget<T>(A_op, svds_fn, norm_A, target_rank,
                           false, cli_cqrrt, nullptr,
                           block_sizes, budget, algs, state, outfile);
    } else {
        T* A_dense = mat.data();
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A_dense, m, Layout::ColMajor);
        T norm_A = A_op.fro_nrm();

        Eigen::Map<const EMatrix> A_eigen(A_dense, m, n);
        auto svds_fn = [&](int64_t budget_mv, long& dur) -> T {
            return run_svds<T, EMatrix>(A_eigen, A_op, budget_mv, target_rank, dur);
        };

        run_with_budget<T>(A_op, svds_fn, norm_A, target_rank,
                           run_gesdd, false, A_dense,
                           block_sizes, budget, algs, state, outfile);
    }

    long total_us = duration_cast<microseconds>(steady_clock::now() - t_total).count();
    printf("\nTOTAL BENCHMARK TIME: %.2f seconds\n", total_us / 1e6);
    printf("Results: %s\n", out_path.c_str());
}

int main(int argc, char *argv[]) {
    std::set_terminate(abrik_terminate_handler);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision: double|float> <output_dir> <input_file>"
                  << " <target_rank> <run_gesdd> <budget>"
                  << " <num_block_sizes> <block_sizes...>"
                  << " [sub_ratio] [use_cqrrt]\n";
        return 1;
    }
    std::string precision = argv[1];
    if (precision == "double")     run_benchmark<double>(argc, argv);
    else if (precision == "float") run_benchmark<float>(argc, argv);
    else {
        std::cerr << "Error: precision must be 'double' or 'float'\n";
        return 1;
    }
    return 0;
}
