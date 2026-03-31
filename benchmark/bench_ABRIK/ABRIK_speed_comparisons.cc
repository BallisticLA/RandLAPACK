/*
Unified ABRIK speed comparison benchmark.
Reads input from .mtx (Matrix Market) or .txt (whitespace-delimited) files.
Runs ABRIK, RSVD, SVDS (Spectra), and optionally GESDD (dense only).

All algorithms use the LinearOperator abstraction:
  - ABRIK and RSVD operate through LinOp (no matrix copy needed)
  - SVDS uses Eigen (dense Map or sparse matrix natively)
  - GESDD requires dense buffer (dense input only)
  - Residuals computed through LinOp (no materialization)

Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F).
Timings in microseconds.

Usage:
  ABRIK_speed_comparisons <precision> <output_dir> <input_file> <target_rank> <run_gesdd>
                          <num_block_sizes> <num_matmul_sizes> <block_sizes...> <matmul_sizes...>

Input file: .mtx (Matrix Market, dense or sparse) or .txt (whitespace-delimited dense).
GESDD is skipped for sparse input regardless of the run_gesdd flag.
*/

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_linops.hh"
#include "rl_svd_residual.hh"
#include "bench_matrix_io.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>

// External libs
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Spectra/contrib/PartialSVDSolver.h>
#include "ext_budgeted_svd_solver.hh"

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

// LinOp-based residual: sqrt(||S^{-1}AV - U||^2 + ||(A'U)S^{-1} - V||^2)
// Uses LinOp for matvecs — works for both dense and sparse.
template <typename T, RandLAPACK::linops::LinearOperator LinOp>
static T residual_via_linop(LinOp& A_op, T* U, T* V, T* Sigma, int64_t k) {
    return RandLAPACK::linops::svd_residual<T>(A_op, U, V, Sigma, k);
}

// Runs SVDS (Spectra) for one configuration. Templated on Eigen matrix type
// so it works with both dense (Eigen::MatrixXd) and sparse (Eigen::SparseMatrix).
template <typename T, typename EigenMatType, RandLAPACK::linops::LinearOperator LinOp>
static T run_svds(const EigenMatType& A_eigen, LinOp& A_op,
                  int64_t b_sz, int64_t num_matmuls, int64_t target_rank, long& dur_svds) {
    using EMatrix = typename EigenTypes<T>::Matrix;
    using EVector = typename EigenTypes<T>::Vector;

    int64_t m = A_op.n_rows;
    int64_t n = A_op.n_cols;
    int64_t nev = target_rank;
    int64_t ncv_default = std::min(2 * nev + 1, n - 1);
    int64_t budget = b_sz * num_matmuls;
    int64_t ncv = BenchmarkUtil::effective_ncv(budget, nev, ncv_default);
    int64_t max_restarts = BenchmarkUtil::budget_to_restarts(budget, nev, ncv);

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
    Eigen::Map<EVector>(S_s, nev) = S_sp;

    T err = residual_via_linop(A_op, U_s, V_s, S_s, nev);
    delete[] U_s; delete[] V_s; delete[] S_s;
    return err;
}

// Core benchmark loop — templated on LinOp type.
// svds_fn is a callable that runs SVDS for a given (b_sz, num_matmuls) and returns (error, duration).
template <typename T, typename RNG, RandLAPACK::linops::LinearOperator LinOp, typename SvdsFn>
static void run_all_configs(
    LinOp& A_op,
    SvdsFn svds_fn,
    T norm_A,
    int64_t target_rank,
    bool run_gesdd,
    bool use_cqrrt,
    T* A_dense_buf,                // Dense buffer for GESDD (nullptr for sparse input)
    std::vector<int64_t>& block_sizes,
    std::vector<int64_t>& matmul_counts,
    AlgorithmObjects<T, RNG>& algs,
    RandBLAS::RNGState<RNG>& state,
    std::ofstream& outfile)
{
    using EMatrix = typename EigenTypes<T>::Matrix;
    using EVector = typename EigenTypes<T>::Vector;

    int64_t m = A_op.n_rows;
    int64_t n = A_op.n_cols;
    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);

    for (auto b_sz : block_sizes) {
        for (auto num_matmuls : matmul_counts) {
            printf("\nBlock size %ld, num matmuls %ld.\n", b_sz, num_matmuls);

            auto state_alg = state;
            algs.RSVD.block_sz = b_sz;
            algs.ABRIK.max_krylov_iters = (int) num_matmuls;
            if (use_cqrrt)
                algs.ABRIK.qr_exp = RandLAPACK::ABRIKSubroutines::QR_explicit::cqrrt;

            long dur_ABRIK = 0, dur_rsvd = 0, dur_svds = 0, dur_svd = 0;
            T err_ABRIK = 0, err_RSVD = 0, err_SVDS = 0, err_SVD = 0;

            // ---- ABRIK (LinOp) ----
            T* U_a = nullptr; T* V_a = nullptr; T* S_a = nullptr;
            auto t0 = steady_clock::now();
            algs.ABRIK.call(A_op, b_sz, U_a, V_a, S_a, state_alg);
            dur_ABRIK = duration_cast<microseconds>(steady_clock::now() - t0).count();

            int64_t k_a = std::min(target_rank, algs.ABRIK.singular_triplets_found);
            err_ABRIK = residual_via_linop(A_op, U_a, V_a, S_a, k_a);
            printf("ABRIK: err=%e, time=%ld us\n", err_ABRIK, dur_ABRIK);
            delete[] U_a; delete[] V_a; delete[] S_a;

            // ---- RSVD (LinOp, no copy needed) ----
            state_alg = state;
            T* U_r = nullptr; T* V_r = nullptr; T* S_r = nullptr;
            int64_t k_r = (int64_t)(b_sz * num_matmuls / 2);

            t0 = steady_clock::now();
            algs.RSVD.call(A_op, norm_A, k_r, tol, U_r, S_r, V_r, state_alg);
            dur_rsvd = duration_cast<microseconds>(steady_clock::now() - t0).count();

            int64_t k_r_target = std::min(target_rank, k_r);
            err_RSVD = residual_via_linop(A_op, U_r, V_r, S_r, k_r_target);
            printf("RSVD:  err=%e, time=%ld us\n", err_RSVD, dur_rsvd);
            free(U_r); free(V_r); free(S_r);

            // ---- SVDS (Spectra, budgeted) — dispatched via callback ----
            err_SVDS = svds_fn(b_sz, num_matmuls, dur_svds);
            printf("SVDS:  err=%e, time=%ld us\n", err_SVDS, dur_svds);

            // ---- GESDD (dense only, optional) ----
            if (run_gesdd && A_dense_buf) {
                T* A_svd = new T[m * n];
                lapack::lacpy(MatrixType::General, m, n, A_dense_buf, m, A_svd, m);
                T* U_g = new T[m * n]();
                T* S_g = new T[n]();
                T* VT_g = new T[n * n]();
                T* V_g = new T[n * n]();

                t0 = steady_clock::now();
                lapack::gesdd(Job::SomeVec, m, n, A_svd, m, S_g, U_g, m, VT_g, n);
                dur_svd = duration_cast<microseconds>(steady_clock::now() - t0).count();

                RandLAPACK::util::transposition(n, n, VT_g, n, V_g, n, 0);
                err_SVD = residual_via_linop(A_op, U_g, V_g, S_g, target_rank);
                printf("GESDD: err=%e, time=%ld us\n", err_SVD, dur_svd);

                delete[] A_svd; delete[] U_g; delete[] S_g; delete[] VT_g; delete[] V_g;
                run_gesdd = false;  // Only run once
            }

            // Write CSV row
            outfile << b_sz << ", " << num_matmuls << ", " << target_rank << ", "
                    << err_ABRIK << ", " << dur_ABRIK << ", "
                    << err_RSVD  << ", " << dur_rsvd  << ", "
                    << err_SVDS  << ", " << dur_svds  << ", "
                    << err_SVD   << ", " << dur_svd   << "\n";
            outfile.flush();
        }
    }
}

template <typename T>
static void run_benchmark(int argc, char *argv[]) {
    using EMatrix  = typename EigenTypes<T>::Matrix;
    using SpMatrix = typename EigenTypes<T>::SpMatrix;

    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision> <output_dir> <input_file> <target_rank> <run_gesdd>"
                  << " <num_block_sizes> <num_matmul_sizes> <block_sizes...> <matmul_sizes...>"
                  << std::endl;
        return;
    }

    std::string output_dir = argv[2];
    std::string input_path = argv[3];
    int64_t target_rank    = std::stol(argv[4]);
    bool run_gesdd         = (std::stoi(argv[5]) != 0);
    int num_b_sz           = std::stoi(argv[6]);
    int num_mm             = std::stoi(argv[7]);

    std::vector<int64_t> block_sizes, matmul_counts;
    for (int i = 0; i < num_b_sz; ++i)
        block_sizes.push_back(std::stol(argv[8 + i]));
    for (int i = 0; i < num_mm; ++i)
        matmul_counts.push_back(std::stol(argv[8 + num_b_sz + i]));

    // Optional trailing args: [sub_ratio] [use_cqrrt]
    int args_consumed = 8 + num_b_sz + num_mm;
    double sub_ratio = (argc > args_consumed) ? std::stod(argv[args_consumed]) : 1.0;
    bool cli_cqrrt = (argc > args_consumed + 1) ? (std::stoi(argv[args_consumed + 1]) != 0) : false;

    T tol = std::pow(std::numeric_limits<T>::epsilon(), (T)0.85);
    auto state = RandBLAS::RNGState();

    // --- Load matrix (auto-detects .mtx vs .txt) ---
    auto mat = BenchIO::load_matrix<T>(input_path, sub_ratio);
    int64_t m = mat.m;
    int64_t n = mat.n;

    // --- Set up algorithm objects ---
    AlgorithmObjects<T, r123::Philox4x32> algs(0, tol);

    // --- Open output CSV ---
    std::time_t now = std::time(nullptr);
    char date_prefix[20];
    std::strftime(date_prefix, sizeof(date_prefix), "%Y%m%d_%H%M%S_", std::localtime(&now));

    std::string out_filename = std::string(date_prefix) + "ABRIK_speed_comparisons.csv";
    std::string out_path = (output_dir != ".") ? output_dir + "/" + out_filename : out_filename;
    std::ofstream outfile(out_path);

    std::ostringstream oss_b, oss_m;
    for (auto v : block_sizes) oss_b << v << ", ";
    for (auto v : matmul_counts) oss_m << v << ", ";

    outfile << "# ABRIK Speed Comparison Benchmark (unified dense/sparse)\n"
            << "# Precision: " << argv[1] << "\n"
            << "# Input matrix: " << input_path << "\n"
            << "# Input size: " << m << " x " << n << "\n"
            << "# Format: " << (mat.is_sparse ? "sparse" : "dense") << "\n"
            << "# Target rank: " << target_rank << "\n"
            << "# Block sizes: " << oss_b.str() << "\n"
            << "# Matmul counts: " << oss_m.str() << "\n"
            << "# Tolerance: " << tol << "\n"
            << "# Run GESDD: " << (run_gesdd && !mat.is_sparse ? "yes" : "no") << "\n"
            << "# Residual metric: sqrt(||S^{-1}AV - U||^2_F + ||(A'U)S^{-1} - V||^2_F)\n"
            << "# Timings in microseconds\n";
    outfile << "b_sz, num_matmuls, target_rank, "
            << "err_ABRIK, dur_ABRIK, "
            << "err_RSVD, dur_RSVD, "
            << "err_SVDS, dur_SVDS, "
            << "err_SVD, dur_SVD\n";
    outfile.flush();

    auto t_total = steady_clock::now();

    if (mat.is_sparse) {
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::CSCMatrix<T>>
            A_op(m, n, *mat.csc);
        T norm_A = A_op.fro_nrm();

        // SVDS via Eigen sparse matrix (no materialization)
        auto svds_fn = [&](int64_t b_sz, int64_t num_matmuls, long& dur) -> T {
            return run_svds<T>(*mat.eigen_sparse, A_op, b_sz, num_matmuls, target_rank, dur);
        };

        run_all_configs<T>(A_op, svds_fn, norm_A, target_rank,
                          false, cli_cqrrt,  // no GESDD for sparse
                          nullptr,
                          block_sizes, matmul_counts, algs, state, outfile);
    } else {
        T* A_dense = mat.data();
        RandLAPACK::linops::DenseLinOp<T> A_op(m, n, A_dense, m, Layout::ColMajor);
        T norm_A = A_op.fro_nrm();

        // SVDS via Eigen dense matrix (Map wrapping existing buffer, no copy).
        // Template type is EMatrix (not Map) so BudgetedPartialSVDSolver's Ref works.
        Eigen::Map<const EMatrix> A_eigen(A_dense, m, n);
        auto svds_fn = [&](int64_t b_sz, int64_t num_matmuls, long& dur) -> T {
            return run_svds<T, EMatrix>(A_eigen, A_op, b_sz, num_matmuls, target_rank, dur);
        };

        run_all_configs<T>(A_op, svds_fn, norm_A, target_rank,
                          run_gesdd, false, // no cqrrt for dense
                          A_dense,
                          block_sizes, matmul_counts, algs, state, outfile);
    }

    long total_us = duration_cast<microseconds>(steady_clock::now() - t_total).count();
    printf("\nTOTAL BENCHMARK TIME: %.2f seconds\n", total_us / 1e6);
    printf("Results: %s\n", out_path.c_str());
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <precision: double|float> <output_dir> <input_file>"
                  << " <target_rank> <run_gesdd>"
                  << " <num_block_sizes> <num_matmul_sizes> <block_sizes...> <matmul_sizes...>"
                  << std::endl;
        return 1;
    }
    std::string precision = argv[1];
    if (precision == "double")     run_benchmark<double>(argc, argv);
    else if (precision == "float") run_benchmark<float>(argc, argv);
    else {
        std::cerr << "Error: precision must be 'double' or 'float'" << std::endl;
        return 1;
    }
    return 0;
}
