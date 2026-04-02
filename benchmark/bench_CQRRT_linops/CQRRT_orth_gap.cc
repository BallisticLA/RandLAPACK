// CQRRT orthogonality gap benchmark: CQRRT_linop vs CQRRT_expl
//
// Compares the two CQRRT implementations on the same matrix and reports
// the orthogonality gap.  Five modes:
//
//   generate   — synthetic sparse matrix with controlled condition number
//   file       — read a Matrix Market file from disk
//   composite  — composite operator from K.mtx + V.mtx → L^{-1}V
//   diag       — step-by-step diagnostic on a Matrix Market file
//   diag_gen   — step-by-step diagnostic on a synthetic matrix
//
// The diagnostic modes manually execute both algorithms step by step,
// comparing intermediate results (sketch, QR, preconditioned product, Gram,
// Cholesky, final R) to pinpoint where numerical divergence occurs.
// All modes write CSV output to the current working directory.
//
// This benchmark lives on the paper branch (spring-2026-wip) and is not
// intended for inclusion in the main RandLAPACK library.
//
// Usage:
//   ./CQRRT_orth_gap <prec> generate <m> <n> <cond> <density> <d_factor> <runs> [nnz] [block_size]
//   ./CQRRT_orth_gap <prec> file <mtx_path> <d_factor> <runs> [nnz] [block_size] [compute_cond]
//   ./CQRRT_orth_gap <prec> composite <K.mtx> <V.mtx> <d_factor> <runs> [nnz] [block_size]
//   ./CQRRT_orth_gap <prec> diag <mtx_path> <d_factor> [nnz]
//   ./CQRRT_orth_gap <prec> diag_gen <m> <n> <cond> <density> <d_factor> [nnz]

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <fstream>
#include <iomanip>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// Extras utilities for Matrix Market I/O and CholSolver
#include "../../extras/misc/ext_util.hh"
#include "../../extras/linops/ext_cholsolver_linop.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"

// Linops algorithms
#include "rl_cqrrt_linops.hh"
#include "rl_composite_linop.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

// Compute Q = A * R^{-1} via materialize + trsm (backward stable).
template <typename T, typename GLO>
static void compute_Q_from_R(
    GLO& A_op, T* R, int64_t ldr,
    T* Q_out, int64_t m, int64_t n) {
    T* Eye = new T[n * n]();
    RandLAPACK::util::eye(n, n, Eye);
    A_op(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
         m, n, n, (T)1.0, Eye, n, (T)0.0, Q_out, m);
    delete[] Eye;
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, R, ldr, Q_out, m);
}

// Compute condition number via SVD.
template <typename T>
static T compute_condition_number(T* A, int64_t m, int64_t n) {
    std::vector<T> A_copy(m * n);
    std::copy(A, A + m * n, A_copy.data());
    std::vector<T> sigma(n);
    lapack::gesdd(lapack::Job::NoVec, m, n, A_copy.data(), m, sigma.data(),
                  nullptr, 1, nullptr, 1);
    return sigma[0] / sigma[n - 1];
}

struct RunResult {
    double linop_orth;
    double expl_orth;
    double gap_ratio;
    long linop_time_us;
    long expl_time_us;
};

// Relative difference between two buffers: ||A - B||_F / ||A||_F
template <typename T>
static T rel_diff(const T* A, const T* B, int64_t len) {
    T norm_diff = 0, norm_A = 0;
    for (int64_t i = 0; i < len; ++i) {
        T d = A[i] - B[i];
        norm_diff += d * d;
        norm_A += A[i] * A[i];
    }
    return std::sqrt(norm_diff) / std::sqrt(norm_A);
}

// Step-by-step diagnostic: manually runs both algorithms and compares intermediates.
template <typename T, typename RNG, typename GLO>
static void run_diagnostic(
    GLO& A_linop,
    int64_t m, int64_t n, T d_factor, int64_t sketch_nnz,
    RandBLAS::RNGState<RNG>& state) {

    using RandBLAS::RNGState;
    int64_t d = (int64_t)std::ceil(d_factor * n);

    printf("\n=== Step-by-step Diagnostic ===\n");
    printf("  m=%ld, n=%ld, d=%ld, sketch_nnz=%ld\n\n", m, n, d, sketch_nnz);

    // ----- Step 0: Materialize operator -----
    std::vector<T> A_dense(m * n, 0.0);
    {
        T* Eye = new T[n * n]();
        RandLAPACK::util::eye(n, n, Eye);
        A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                m, n, n, (T)1.0, Eye, n, (T)0.0, A_dense.data(), m);
        delete[] Eye;
    }

    // ----- Step 1: Sketch S*A -----
    // Both use same RNG state → same S
    auto state_linop = state;
    auto state_expl  = state;

    // Linop sketch: A_linop(Side::Right, S) → spgemm
    RandBLAS::SparseDist Dl(d, m, sketch_nnz, RandBLAS::Axis::Short);
    RandBLAS::SparseSkOp<T> S_linop(Dl, state_linop);
    std::vector<T> Ahat_linop(d * n, 0.0);
    A_linop(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            d, n, m, (T)1.0, S_linop, (T)0.0, Ahat_linop.data(), d);

    // Expl sketch: sketch_general(S, A_dense) → left_spmm
    RandBLAS::SparseDist De(d, m, sketch_nnz, RandBLAS::Axis::Short);
    RandBLAS::SparseSkOp<T> S_expl(De, state_expl);
    std::vector<T> Ahat_expl(d * n, 0.0);
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                              d, n, m, (T)1.0, S_expl, A_dense.data(), m,
                              (T)0.0, Ahat_expl.data(), d);

    double rd_sketch = (double)rel_diff(Ahat_linop.data(), Ahat_expl.data(), d * n);
    printf("  %-40s  %12.3e\n", "Step 1: Sketch ||Ahat_l - Ahat_e||/||Ahat_l||", rd_sketch);

    // ----- Step 2: QR of sketch → R_sk -----
    std::vector<T> R_sk_linop(n * n, 0.0);
    std::vector<T> R_sk_expl(n * n, 0.0);
    {
        // QR on linop sketch
        std::vector<T> tau(n);
        lapack::geqrf(d, n, Ahat_linop.data(), d, tau.data());
        // Extract R
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i <= j; ++i)
                R_sk_linop[i + j * n] = Ahat_linop[i + j * d];
    }
    {
        // QR on expl sketch
        std::vector<T> tau(n);
        lapack::geqrf(d, n, Ahat_expl.data(), d, tau.data());
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i <= j; ++i)
                R_sk_expl[i + j * n] = Ahat_expl[i + j * d];
    }

    double rd_qr = (double)rel_diff(R_sk_linop.data(), R_sk_expl.data(), n * n);
    printf("  %-40s  %12.3e\n", "Step 2: QR ||R_sk_l - R_sk_e||/||R_sk_l||", rd_qr);

    // ----- Step 3 (linop only): Explicit inverse R_pre = R_sk^{-1} -----
    std::vector<T> R_pre(n * n, 0.0);
    RandLAPACK::util::eye(n, n, R_pre.data());
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, n, n, (T)1.0, R_sk_linop.data(), n, R_pre.data(), n);

    // ----- Step 3: A_pre = A * R_pre (linop) vs A * R_sk^{-1} via TRSM (expl) -----
    // Linop: A_pre_l = A * R_pre (using the linop's explicit inverse)
    std::vector<T> A_pre_linop(m * n, 0.0);
    A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
            m, n, n, (T)1.0, R_pre.data(), n, (T)0.0, A_pre_linop.data(), m);

    // Expl: A_pre_e = A * R_sk^{-1} via in-place TRSM (backward stable, no explicit inverse)
    std::vector<T> A_pre_expl(A_dense.begin(), A_dense.end());  // copy
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, R_sk_expl.data(), n, A_pre_expl.data(), m);

    double rd_apre = (double)rel_diff(A_pre_linop.data(), A_pre_expl.data(), m * n);
    printf("  %-40s  %12.3e\n", "Step 3: A_pre ||Apre_l - Apre_e||/||Apre_l||", rd_apre);

    // ----- Step 5a (linop): Gram via A^T * A_pre then R_pre^T * ... -----
    std::vector<T> G_linop(n * n, 0.0);
    A_linop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
            n, n, m, (T)1.0, A_pre_linop.data(), m, (T)0.0, G_linop.data(), n);
    // Apply R_pre^T on left: G = R_pre^T * (A^T * A_pre)
    blas::trmm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans,
               Diag::NonUnit, n, n, (T)1.0, R_pre.data(), n, G_linop.data(), n);
    std::vector<T> G_linop_final(G_linop.begin(), G_linop.end());

    // Compute A^T * A_pre for BOTH paths via dense GEMM (apples-to-apples, no SYRK vs GEMM difference)
    std::vector<T> AtApre_linop(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
               n, n, m, (T)1.0, A_dense.data(), m, A_pre_linop.data(), m,
               (T)0.0, AtApre_linop.data(), n);
    std::vector<T> AtApre_expl(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
               n, n, m, (T)1.0, A_dense.data(), m, A_pre_expl.data(), m,
               (T)0.0, AtApre_expl.data(), n);

    double rd_atapre = (double)rel_diff(AtApre_linop.data(), AtApre_expl.data(), n * n);
    printf("  %-40s  %12.3e\n", "Step 4a: A^T*Apre ||l - e||/||l|| (GEMM)", rd_atapre);

    // ----- Step 5b: Full Gram: linop uses GEMM+TRMM, expl uses SYRK -----
    std::vector<T> G_expl(n * n, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
               n, m, (T)1.0, A_pre_expl.data(), m, (T)0.0, G_expl.data(), n);
    // Fill lower triangle
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i)
            G_expl[i + j * n] = G_expl[j + i * n];

    // Also compare SYRK vs GEMM on the SAME A_pre_expl buffer (isolate SYRK effect)
    std::vector<T> G_expl_gemm(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
               n, n, m, (T)1.0, A_pre_expl.data(), m, A_pre_expl.data(), m,
               (T)0.0, G_expl_gemm.data(), n);

    double rd_gram = (double)rel_diff(G_linop_final.data(), G_expl.data(), n * n);
    double rd_syrk_gemm = (double)rel_diff(G_expl.data(), G_expl_gemm.data(), n * n);
    printf("  %-40s  %12.3e\n", "Step 4b: Gram ||G_l - G_e||/||G_l||", rd_gram);
    printf("  %-40s  %12.3e\n", "Step 4c: SYRK vs GEMM on same Apre_e", rd_syrk_gemm);

    // ----- Step 5: Cholesky -----
    std::vector<T> R_chol_linop(G_linop_final.begin(), G_linop_final.end());
    std::vector<T> R_chol_expl(G_expl.begin(), G_expl.end());
    lapack::potrf(Uplo::Upper, n, R_chol_linop.data(), n);
    lapack::potrf(Uplo::Upper, n, R_chol_expl.data(), n);
    // Zero below diagonal
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) {
            R_chol_linop[i + j * n] = 0;
            R_chol_expl[i + j * n] = 0;
        }

    double rd_chol = (double)rel_diff(R_chol_linop.data(), R_chol_expl.data(), n * n);
    printf("  %-40s  %12.3e\n", "Step 5: Cholesky ||Rc_l - Rc_e||/||Rc_l||", rd_chol);

    // ----- Step 6: Final R = R_chol * R_sk -----
    std::vector<T> R_final_linop(n * n, 0.0);
    std::vector<T> R_final_expl(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               n, n, n, (T)1.0, R_chol_linop.data(), n, R_sk_linop.data(), n,
               (T)0.0, R_final_linop.data(), n);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               n, n, n, (T)1.0, R_chol_expl.data(), n, R_sk_expl.data(), n,
               (T)0.0, R_final_expl.data(), n);

    double rd_finalR = (double)rel_diff(R_final_linop.data(), R_final_expl.data(), n * n);
    printf("  %-40s  %12.3e\n", "Step 6: Final R ||R_l - R_e||/||R_l||", rd_finalR);

    // ----- Step 7: Orthogonality of Q = A * R^{-1} -----
    std::vector<T> Q(m * n);
    compute_Q_from_R(A_linop, R_final_linop.data(), n, Q.data(), m, n);
    T orth_linop = RandLAPACK::testing::orthogonality_error<T>(Q.data(), m, n);

    compute_Q_from_R(A_linop, R_final_expl.data(), n, Q.data(), m, n);
    T orth_expl = RandLAPACK::testing::orthogonality_error<T>(Q.data(), m, n);

    printf("  %-40s  linop=%.3e, expl=%.3e, gap=%.1fx\n",
           "Step 7: Orthogonality",
           (double)orth_linop, (double)orth_expl, (double)(orth_linop / orth_expl));
    printf("\n");

    // Write diagnostic CSV
    // Filename: diag_<timestamp>.csv in current directory
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

    std::string csv_name = std::string("orth_gap_diag_") + ts + ".csv";
    std::ofstream csv(csv_name);
    if (csv.is_open()) {
        std::string prec_name = (sizeof(T) == 8) ? "double" : "float";
        csv << "# CQRRT orth gap diagnostic\n";
        csv << "# precision=" << prec_name << ", m=" << m << ", n=" << n << ", d=" << d << ", sketch_nnz=" << sketch_nnz << "\n";
        csv << "step,description,rel_diff\n";
        csv << std::scientific << std::setprecision(6);
        csv << "1,sketch," << rd_sketch << "\n";
        csv << "2,qr_R_sk," << rd_qr << "\n";
        csv << "3,A_pre," << rd_apre << "\n";
        csv << "4,gram," << rd_gram << "\n";
        csv << "5,cholesky," << rd_chol << "\n";
        csv << "6,final_R," << rd_finalR << "\n";
        csv << "7,orth_linop," << (double)orth_linop << "\n";
        csv << "7,orth_expl," << (double)orth_expl << "\n";
        csv << "7,orth_gap," << (double)(orth_linop / orth_expl) << "\n";
        csv.close();
        printf("  Diagnostic CSV: %s\n\n", csv_name.c_str());
    }
}

template <typename T, typename RNG, typename GLO>
static RunResult run_comparison(
    GLO& A_linop,
    int64_t m, int64_t n, T d_factor, int64_t sketch_nnz, int64_t block_size,
    RandBLAS::RNGState<RNG>& state) {

    T tol = std::pow(std::numeric_limits<T>::epsilon(), 0.85);
    std::vector<T> Q(m * n);
    RunResult res;

    // --- CQRRT_linop ---
    {
        std::vector<T> R(n * n, 0.0);
        auto state_copy = state;
        RandLAPACK::CQRRT_linops<T, RNG> algo(true, tol, false);
        algo.nnz = sketch_nnz;
        algo.block_size = block_size;

        auto t0 = steady_clock::now();
        algo.call(A_linop, R.data(), n, d_factor, state_copy);
        auto t1 = steady_clock::now();
        res.linop_time_us = duration_cast<microseconds>(t1 - t0).count();

        compute_Q_from_R(A_linop, R.data(), n, Q.data(), m, n);
        res.linop_orth = RandLAPACK::testing::orthogonality_error<T>(Q.data(), m, n);
    }

    // --- CQRRT_expl (materialize + dense CQRRT) ---
    {
        // Materialize the operator
        T* Eye = new T[n * n]();
        RandLAPACK::util::eye(n, n, Eye);
        T* A_dense = new T[m * n]();
        A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                m, n, n, (T)1.0, Eye, n, (T)0.0, A_dense, m);
        delete[] Eye;

        std::vector<T> R(n * n, 0.0);
        auto state_copy = state;
        RandLAPACK::CQRRT<T, RNG> algo(true, tol);
        algo.compute_Q = false;
        algo.orthogonalization = false;
        algo.nnz = sketch_nnz;

        auto t0 = steady_clock::now();
        algo.call(m, n, A_dense, m, R.data(), n, d_factor, state_copy);
        auto t1 = steady_clock::now();
        res.expl_time_us = duration_cast<microseconds>(t1 - t0).count();

        delete[] A_dense;

        compute_Q_from_R(A_linop, R.data(), n, Q.data(), m, n);
        res.expl_orth = RandLAPACK::testing::orthogonality_error<T>(Q.data(), m, n);
    }

    res.gap_ratio = res.linop_orth / res.expl_orth;
    return res;
}

template <typename T>
static int run_generate_mode(int argc, char* argv[]) {
    // ./CQRRT_orth_gap <prec> generate <m> <n> <cond_num> <density> <d_factor> <num_runs> [sketch_nnz] [block_size]
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0]
                  << " <prec> generate <m> <n> <cond_num> <density> <d_factor> <num_runs> [sketch_nnz] [block_size]\n";
        return 1;
    }
    int64_t m         = std::stol(argv[3]);
    int64_t n         = std::stol(argv[4]);
    double cond_num   = std::stod(argv[5]);
    double density    = std::stod(argv[6]);
    double d_factor   = std::stod(argv[7]);
    int64_t num_runs  = std::stol(argv[8]);
    int64_t sketch_nnz = (argc >= 10) ? std::stol(argv[9]) : 4;
    int64_t block_size = (argc >= 11) ? std::stol(argv[10]) : 0;

    printf("\n=== CQRRT Orthogonality Gap Benchmark (generate mode) ===\n");
    printf("  Matrix: %ld x %ld, kappa=%.2e, density=%.3f\n", m, n, cond_num, density);
    printf("  d_factor=%.2f, sketch_nnz=%ld, block_size=%ld, runs=%ld\n",
           d_factor, sketch_nnz, block_size, num_runs);
#ifdef _OPENMP
    printf("  OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("=========================================================\n\n");

    auto state = RandBLAS::RNGState<r123::Philox4x32>();
    auto A_coo = RandLAPACK::gen::gen_sparse_cond_coo<T>(m, n, (T)cond_num, state, (T)density);
    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    // Timestamp for CSV filenames
    auto now_gen = std::chrono::system_clock::now();
    auto t_gen = std::chrono::system_clock::to_time_t(now_gen);
    char ts_gen[32];
    std::strftime(ts_gen, sizeof(ts_gen), "%Y%m%d_%H%M%S", std::localtime(&t_gen));
    std::string csv_name = std::string("orth_gap_gen_") + ts_gen + ".csv";
    std::ofstream csv(csv_name);
    if (csv.is_open()) {
        std::string prec_name = (sizeof(T) == 8) ? "double" : "float";
        csv << "# CQRRT orth gap benchmark (generate mode)\n";
        csv << "# precision=" << prec_name << ", m=" << m << ", n=" << n << ", kappa=" << cond_num
            << ", density=" << density << ", d_factor=" << d_factor
            << ", sketch_nnz=" << sketch_nnz << ", block_size=" << block_size << "\n";
        csv << "run,linop_orth,expl_orth,gap_ratio,linop_time_us,expl_time_us\n";
    }

    printf("  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
           "Run", "CQRRT_linop", "CQRRT_expl", "Gap (x)", "Linop (us)", "Expl (us)");
    printf("  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
           "---", "-----------", "----------", "-------", "----------", "---------");

    for (int64_t r = 0; r < num_runs; ++r) {
        auto run_state = state;
        auto res = run_comparison<T>(A_linop, m, n, (T)d_factor, sketch_nnz, block_size, run_state);
        printf("  %-6ld  %-14.6e  %-14.6e  %-10.1f  %-12ld  %-12ld\n",
               r, res.linop_orth, res.expl_orth, res.gap_ratio, res.linop_time_us, res.expl_time_us);
        if (csv.is_open()) {
            csv << r << "," << std::scientific << std::setprecision(6)
                << res.linop_orth << "," << res.expl_orth << ","
                << res.gap_ratio << "," << res.linop_time_us << "," << res.expl_time_us << "\n";
        }
        state = RandBLAS::RNGState<r123::Philox4x32>(state.key.incr());
    }
    if (csv.is_open()) {
        csv.close();
        printf("  Results CSV: %s\n", csv_name.c_str());
    }
    printf("\n");
    return 0;
}

template <typename T>
static int run_file_mode(int argc, char* argv[]) {
    // ./CQRRT_orth_gap <prec> file <mtx_path> <d_factor> <num_runs> [sketch_nnz] [block_size] [compute_cond]
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <prec> file <mtx_path> <d_factor> <num_runs> [sketch_nnz] [block_size] [compute_cond]\n";
        return 1;
    }
    std::string mtx_path = argv[3];
    double d_factor       = std::stod(argv[4]);
    int64_t num_runs      = std::stol(argv[5]);
    int64_t sketch_nnz    = (argc >= 7) ? std::stol(argv[6]) : 4;
    int64_t block_size    = (argc >= 8) ? std::stol(argv[7]) : 0;
    bool compute_cond     = (argc >= 9) ? (std::stol(argv[8]) != 0) : false;

    printf("\n=== CQRRT Orthogonality Gap Benchmark (file mode) ===\n");
    printf("  File: %s\n", mtx_path.c_str());
    printf("  d_factor=%.2f, sketch_nnz=%ld, block_size=%ld, runs=%ld\n",
           d_factor, sketch_nnz, block_size, num_runs);
#ifdef _OPENMP
    printf("  OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("=====================================================\n\n");

    // Read Matrix Market file
    printf("Loading %s...", mtx_path.c_str());
    fflush(stdout);
    auto A_coo = RandLAPACK_extras::coo_from_matrix_market<T>(mtx_path);
    int64_t m = A_coo.n_rows;
    int64_t n = A_coo.n_cols;
    printf(" done (%ld x %ld, nnz=%ld)\n", m, n, A_coo.nnz);

    RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

    if (compute_cond) {
        printf("Computing condition number via SVD (%ld x %ld)...", m, n);
        fflush(stdout);
        std::vector<T> A_dense(m * n, 0.0);
        T* Eye = new T[n * n]();
        RandLAPACK::util::eye(n, n, Eye);
        A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                m, n, n, (T)1.0, Eye, n, (T)0.0, A_dense.data(), m);
        delete[] Eye;
        T kappa = compute_condition_number<T>(A_dense.data(), m, n);
        printf(" kappa = %.6e\n", (double)kappa);
    }

    auto state = RandBLAS::RNGState<r123::Philox4x32>();

    // Extract matrix name from path for CSV
    std::string mtx_name = mtx_path;
    auto slash = mtx_name.rfind('/');
    if (slash != std::string::npos) mtx_name = mtx_name.substr(slash + 1);
    auto dot = mtx_name.rfind('.');
    if (dot != std::string::npos) mtx_name = mtx_name.substr(0, dot);

    // Timestamp for CSV
    auto now_file = std::chrono::system_clock::now();
    auto t_file = std::chrono::system_clock::to_time_t(now_file);
    char ts_file[32];
    std::strftime(ts_file, sizeof(ts_file), "%Y%m%d_%H%M%S", std::localtime(&t_file));
    std::string csv_name = std::string("orth_gap_") + mtx_name + "_" + ts_file + ".csv";
    std::ofstream csv(csv_name);
    if (csv.is_open()) {
        std::string prec_name = (sizeof(T) == 8) ? "double" : "float";
        csv << "# CQRRT orth gap benchmark (file mode)\n";
        csv << "# precision=" << prec_name << ", file=" << mtx_path << ", m=" << m << ", n=" << n
            << ", d_factor=" << d_factor << ", sketch_nnz=" << sketch_nnz
            << ", block_size=" << block_size << "\n";
        csv << "run,linop_orth,expl_orth,gap_ratio,linop_time_us,expl_time_us\n";
    }

    printf("\n  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
           "Run", "CQRRT_linop", "CQRRT_expl", "Gap (x)", "Linop (us)", "Expl (us)");
    printf("  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
           "---", "-----------", "----------", "-------", "----------", "---------");

    for (int64_t r = 0; r < num_runs; ++r) {
        auto run_state = state;
        auto res = run_comparison<T>(A_linop, m, n, (T)d_factor, sketch_nnz, block_size, run_state);
        printf("  %-6ld  %-14.6e  %-14.6e  %-10.1f  %-12ld  %-12ld\n",
               r, res.linop_orth, res.expl_orth, res.gap_ratio, res.linop_time_us, res.expl_time_us);
        if (csv.is_open()) {
            csv << r << "," << std::scientific << std::setprecision(6)
                << res.linop_orth << "," << res.expl_orth << ","
                << res.gap_ratio << "," << res.linop_time_us << "," << res.expl_time_us << "\n";
        }
        state = RandBLAS::RNGState<r123::Philox4x32>(state.key.incr());
    }
    if (csv.is_open()) {
        csv.close();
        printf("  Results CSV: %s\n", csv_name.c_str());
    }
    printf("\n");
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <prec> generate <m> <n> <cond_num> <density> <d_factor> <num_runs> [sketch_nnz] [block_size]\n"
                  << "  " << argv[0] << " <prec> file <mtx_path> <d_factor> <num_runs> [sketch_nnz] [block_size] [compute_cond]\n"
                  << "  " << argv[0] << " <prec> composite <K.mtx> <V.mtx> <d_factor> <num_runs> [sketch_nnz] [block_size]\n"
                  << "  " << argv[0] << " <prec> diag <mtx_path> <d_factor> [sketch_nnz]\n"
                  << "  " << argv[0] << " <prec> diag_gen <m> <n> <cond_num> <density> <d_factor> [sketch_nnz]\n";
        return 1;
    }

    std::string precision = argv[1];
    std::string mode = argv[2];

    // Diagnostic mode: step-by-step comparison on a file
    if (mode == "diag") {
        // ./CQRRT_orth_gap <prec> diag <mtx_path> <d_factor> [sketch_nnz]
        if (argc < 5) {
            std::cerr << "Usage: " << argv[0] << " <prec> diag <mtx_path> <d_factor> [sketch_nnz]\n";
            return 1;
        }
        std::string mtx_path = argv[3];
        double d_factor = std::stod(argv[4]);
        int64_t sketch_nnz = (argc >= 6) ? std::stol(argv[5]) : 4;

        printf("Loading %s...", mtx_path.c_str()); fflush(stdout);
        if (precision == "double") {
            auto A_coo = RandLAPACK_extras::coo_from_matrix_market<double>(mtx_path);
            int64_t m = A_coo.n_rows, n = A_coo.n_cols;
            printf(" done (%ld x %ld, nnz=%ld)\n", m, n, A_coo.nnz);
            RandBLAS::sparse_data::csr::CSRMatrix<double> A_csr(m, n);
            RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
            RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<double>> A_linop(m, n, A_csr);
            auto state = RandBLAS::RNGState<r123::Philox4x32>();
            run_diagnostic<double>(A_linop, m, n, d_factor, sketch_nnz, state);
        } else {
            auto A_coo = RandLAPACK_extras::coo_from_matrix_market<float>(mtx_path);
            int64_t m = A_coo.n_rows, n = A_coo.n_cols;
            printf(" done (%ld x %ld, nnz=%ld)\n", m, n, A_coo.nnz);
            RandBLAS::sparse_data::csr::CSRMatrix<float> A_csr(m, n);
            RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
            RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<float>> A_linop(m, n, A_csr);
            auto state = RandBLAS::RNGState<r123::Philox4x32>();
            run_diagnostic<float>(A_linop, m, n, (float)d_factor, sketch_nnz, state);
        }
        return 0;
    }

    // Diagnostic mode on synthetic matrix
    if (mode == "diag_gen") {
        // ./CQRRT_orth_gap <prec> diag_gen <m> <n> <cond_num> <density> <d_factor> [sketch_nnz]
        if (argc < 8) {
            std::cerr << "Usage: " << argv[0] << " <prec> diag_gen <m> <n> <cond_num> <density> <d_factor> [sketch_nnz]\n";
            return 1;
        }
        int64_t m = std::stol(argv[3]);
        int64_t n = std::stol(argv[4]);
        double cond_num = std::stod(argv[5]);
        double density = std::stod(argv[6]);
        double d_factor = std::stod(argv[7]);
        int64_t sketch_nnz = (argc >= 9) ? std::stol(argv[8]) : 4;

        printf("Generating %ld x %ld matrix, kappa=%.2e, density=%.3f...", m, n, cond_num, density);
        fflush(stdout);
        auto state = RandBLAS::RNGState<r123::Philox4x32>();
        if (precision == "double") {
            auto A_coo = RandLAPACK::gen::gen_sparse_cond_coo<double>(m, n, cond_num, state, density);
            printf(" done (nnz=%ld)\n", A_coo.nnz);
            RandBLAS::sparse_data::csr::CSRMatrix<double> A_csr(m, n);
            RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
            RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<double>> A_linop(m, n, A_csr);
            run_diagnostic<double>(A_linop, m, n, d_factor, sketch_nnz, state);
        } else {
            auto A_coo = RandLAPACK::gen::gen_sparse_cond_coo<float>(m, n, (float)cond_num, state, (float)density);
            printf(" done (nnz=%ld)\n", A_coo.nnz);
            RandBLAS::sparse_data::csr::CSRMatrix<float> A_csr(m, n);
            RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
            RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<float>> A_linop(m, n, A_csr);
            run_diagnostic<float>(A_linop, m, n, (float)d_factor, sketch_nnz, state);
        }
        return 0;
    }

    // Composite mode: K.mtx + V.mtx → CholSolver(K, half_solve) ∘ Sparse(V) = L^{-1}V
    if (mode == "composite") {
        // ./CQRRT_orth_gap <prec> composite <K.mtx> <V.mtx> <d_factor> <runs> [nnz] [block_size]
        if (argc < 7) {
            std::cerr << "Usage: " << argv[0]
                      << " <prec> composite <K.mtx> <V.mtx> <d_factor> <num_runs> [sketch_nnz] [block_size]\n";
            return 1;
        }
        std::string K_file = argv[3];
        std::string V_file = argv[4];
        double d_factor    = std::stod(argv[5]);
        int64_t num_runs   = std::stol(argv[6]);
        int64_t sketch_nnz = (argc >= 8) ? std::stol(argv[7]) : 4;
        int64_t block_size = (argc >= 9) ? std::stol(argv[8]) : 256;

        if (precision == "double") {
            using T = double;

            printf("\n=== CQRRT Orthogonality Gap Benchmark (composite mode) ===\n");
            printf("  K file: %s\n  V file: %s\n", K_file.c_str(), V_file.c_str());
            printf("  d_factor=%.2f, sketch_nnz=%ld, block_size=%ld, runs=%ld\n",
                   d_factor, sketch_nnz, block_size, num_runs);
#ifdef _OPENMP
            printf("  OpenMP threads: %d\n", omp_get_max_threads());
#endif
            printf("==========================================================\n\n");

            // Load V
            printf("Loading V from %s...", V_file.c_str()); fflush(stdout);
            auto V_coo = RandLAPACK_extras::coo_from_matrix_market<T>(V_file);
            int64_t m = V_coo.n_rows, n = V_coo.n_cols;
            printf(" done (%ld x %ld, nnz=%ld)\n", m, n, V_coo.nnz);

            RandBLAS::sparse_data::csr::CSRMatrix<T> V_csr(m, n);
            RandBLAS::sparse_data::conversions::coo_to_csr(V_coo, V_csr);
            RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> V_linop(m, n, V_csr);

            // Build CholSolver for K (half_solve = true → L^{-1})
            printf("Factorizing K = LL^T from %s...", K_file.c_str()); fflush(stdout);
            RandLAPACK_extras::linops::CholSolverLinOp<T> L_inv_op(K_file, /*half_solve=*/true);
            L_inv_op.factorize();
            printf(" done\n");

            // Composite operator: L^{-1} * V (CTAD deduces template args)
            RandLAPACK::linops::CompositeOperator LiV_op(m, n, L_inv_op, V_linop);

            printf("Composite operator L^{-1}V: %ld x %ld\n\n", m, n);

            auto state = RandBLAS::RNGState<r123::Philox4x32>();

            // CSV output
            auto now = std::chrono::system_clock::now();
            auto t = std::chrono::system_clock::to_time_t(now);
            char ts[32];
            std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
            std::string csv_name = std::string("orth_gap_composite_") + ts + ".csv";
            std::ofstream csv(csv_name);
            if (csv.is_open()) {
                csv << "# CQRRT orth gap benchmark (composite mode)\n";
                csv << "# K_file=" << K_file << ", V_file=" << V_file
                    << ", m=" << m << ", n=" << n
                    << ", d_factor=" << d_factor << ", sketch_nnz=" << sketch_nnz
                    << ", block_size=" << block_size << "\n";
                csv << "run,linop_orth,expl_orth,gap_ratio,linop_time_us,expl_time_us\n";
            }

            printf("  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
                   "Run", "CQRRT_linop", "CQRRT_expl", "Gap (x)", "Linop (us)", "Expl (us)");
            printf("  %-6s  %-14s  %-14s  %-10s  %-12s  %-12s\n",
                   "---", "-----------", "----------", "-------", "----------", "---------");

            for (int64_t r = 0; r < num_runs; ++r) {
                auto run_state = state;
                auto res = run_comparison<T>(LiV_op, m, n, (T)d_factor, sketch_nnz, block_size, run_state);
                printf("  %-6ld  %-14.6e  %-14.6e  %-10.1f  %-12ld  %-12ld\n",
                       r, res.linop_orth, res.expl_orth, res.gap_ratio, res.linop_time_us, res.expl_time_us);
                if (csv.is_open()) {
                    csv << r << "," << std::scientific << std::setprecision(6)
                        << res.linop_orth << "," << res.expl_orth << ","
                        << res.gap_ratio << "," << res.linop_time_us << "," << res.expl_time_us << "\n";
                }
                state = RandBLAS::RNGState<r123::Philox4x32>(state.key.incr());
            }
            if (csv.is_open()) { csv.close(); printf("  Results CSV: %s\n", csv_name.c_str()); }
            printf("\n");
        } else {
            std::cerr << "Composite mode only supports double precision\n";
            return 1;
        }
        return 0;
    }

    if (precision == "double") {
        if (mode == "generate") return run_generate_mode<double>(argc, argv);
        if (mode == "file")     return run_file_mode<double>(argc, argv);
    } else if (precision == "float") {
        if (mode == "generate") return run_generate_mode<float>(argc, argv);
        if (mode == "file")     return run_file_mode<float>(argc, argv);
    }

    std::cerr << "Unknown precision '" << precision << "' or mode '" << mode << "'\n";
    return 1;
}
