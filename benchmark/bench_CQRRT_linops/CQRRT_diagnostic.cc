// CQRRT preconditioner comparison benchmark
//
// Isolates the effect of different methods for forming R_sk^{-1} on the final
// orthogonality quality of CQRRT. Tests four paths:
//
//   [1] expl_trsm:      DTRSM_R(A, R_sk) in-place                   ← CQRRT_expl path
//   [2] expl_inv_trsm:  TRSM(I, R_sk) → R_inv; DGEMM(A, R_inv)       (TRSM on identity)
//   [3] expl_inv_trtri: TRTRI(R_sk) → R_inv;   DGEMM(A, R_inv)       (LAPACK trtri)
//   [4] expl_inv_geqp3: GEQP3(R_sk) = Q*R_buf*P^T;
//                       R_inv = P * TRTRI(R_buf) * Q^T; DGEMM(A, R_inv)
//
//   Path [1] never forms R_sk^{-1} explicitly (backward stable).
//   Paths [2]-[4] all form R_sk^{-1} explicitly via different methods.
//   Path [4] uses a rank-revealing QR to invert R_sk; the Q factor makes
//   the inversion well-conditioned even when R_sk itself is ill-conditioned.
//
//   All four paths use the same sketch (same RNG state).
//
//   Per-path metrics:
//     cond(A_pre)             — condition number of the preconditioned matrix
//     cond(G = A_pre^T A_pre) — condition number of the Gram matrix (input to Cholesky)
//     orth_error(Q)           — full-pipeline: G=SYRK(A_pre), R_chol=chol(G),
//                               R_final=R_chol*R_sk, Q=A_orig*R_final^{-1}
//
//   Cross-path relative differences of A_pre (reference = path [1]):
//     rd_12 = ||A_pre[1] - A_pre[2]|| / ||A_pre[1]||
//     rd_13 = ||A_pre[1] - A_pre[3]|| / ||A_pre[1]||
//     rd_14 = ||A_pre[1] - A_pre[4]|| / ||A_pre[1]||
//
//   Step-by-step pipeline divergence between paths [1] and [2] (CQRRT_expl vs CQRRT_linop):
//   Reproduces the "step-by-step divergence" plot from CQRRT_orth_gap diag mode.
//   Since all paths share the same sketch, M^sk and R^sk diffs are 0 by construction
//   (the shared-sketch design cleanly eliminates sketch method as a variable).
//     rd_G_12     = ||G1 - G2|| / ||G1||          (Gram matrix:  G = A_pre^T A_pre)
//     rd_Rchol_12 = ||Rchol1 - Rchol2|| / ||Rchol1||  (Cholesky factor)
//     rd_Rfinal_12= ||Rfinal1 - Rfinal2|| / ||Rfinal1|| (final R = R_chol * R_sk)
//
//   Sketch diagnostic:
//     cond(R_sk)
//
// Usage:
//   ./CQRRT_diagnostic <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <ctime>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../extras/misc/ext_util.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using blas::Layout;
using blas::Op;
using blas::Side;
using blas::Uplo;
using blas::Diag;

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
static T orth_error(const T* Q, int64_t m, int64_t n) {
    return RandLAPACK::testing::orthogonality_error<T>(Q, m, n);
}

template <typename T>
static T rel_diff(const T* A, const T* B, int64_t len) {
    T nd = 0, na = 0;
    for (int64_t i = 0; i < len; ++i) {
        T d = A[i] - B[i];
        nd += d * d; na += A[i] * A[i];
    }
    return (na > 0) ? std::sqrt(nd / na) : std::sqrt(nd);
}

template <typename T>
static T condition_number(const T* A, int64_t m, int64_t n) {
    std::vector<T> tmp(A, A + m * n);
    std::vector<T> s(n);
    lapack::gesdd(lapack::Job::NoVec, m, n, tmp.data(), m,
                  s.data(), nullptr, 1, nullptr, 1);
    return (s[n-1] > 0) ? s[0] / s[n-1] : std::numeric_limits<T>::infinity();
}

// Condition number of the Gram matrix G = A_pre^T A_pre
template <typename T>
static T gram_condition_number(const T* A_pre, int64_t m, int64_t n) {
    std::vector<T> G(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
               n, n, m, (T)1.0, A_pre, m, A_pre, m, (T)0.0, G.data(), n);
    return condition_number(G.data(), n, n);
}

// Full CQRRT pipeline (matching V2 benchmark orth_error computation):
//   G = A_pre^T A_pre  (SYRK)
//   R_chol = chol(G)   (POTRF, overwrites G with upper triangular factor)
//   R_final = R_chol * R_sketch  (TRMM)
//   Q = A_orig * R_final^{-1}   (TRSM on copy of A_orig)
//   return orth_error(Q)
// Does NOT modify A_pre or A_orig.
template <typename T>
static T cholqr_orth_error(const std::vector<T>& A_pre, const T* A_orig,
                            int64_t m, int64_t n, const T* R_sketch) {
    std::vector<T> G(n * n, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
               n, m, (T)1.0, A_pre.data(), m, (T)0.0, G.data(), n);
    if (lapack::potrf(Uplo::Upper, n, G.data(), n))
        return std::numeric_limits<T>::infinity();
    blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, n, n, (T)1.0, R_sketch, n, G.data(), n);
    std::vector<T> Q(A_orig, A_orig + m * n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, G.data(), n, Q.data(), m);
    return orth_error(Q.data(), m, n);
}

// ============================================================================
// One trial: three paths, same sketch
// ============================================================================

template <typename T>
struct TrialResult {
    // Per-path metrics
    double cond_Apre[4];
    double cond_G[4];
    double orth_Q[4];
    // Cross-path relative differences of A_pre (reference = path [1])
    double rd_Apre_12;  // TRSM in-place vs TRSM-on-identity
    double rd_Apre_13;  // TRSM in-place vs trtri
    double rd_Apre_14;  // TRSM in-place vs geqp3
    // Step-by-step pipeline divergence: paths [1] vs [2] (CQRRT_expl vs CQRRT_linop)
    double rd_G_12;      // Gram matrix:      G = A_pre^T A_pre
    double rd_Rchol_12;  // Cholesky factor:  R_chol = chol(G)
    double rd_Rfinal_12; // Final R:          R_final = R_chol * R_sk
    // Sketch diagnostic
    double cond_Rsk;
};

template <typename T, typename RNG>
static TrialResult<T> run_trial(
    const T* A_dense,
    int64_t m, int64_t n,
    T d_factor, int64_t sketch_nnz,
    RandBLAS::RNGState<RNG>& state)
{
    TrialResult<T> res{};
    int64_t d = (int64_t)std::ceil(d_factor * n);

    // ----------------------------------------------------------------
    // Sketch A → Ahat  (d × n), then QR → R_sk
    // ----------------------------------------------------------------
    RandBLAS::SparseDist Ds(d, m, sketch_nnz, RandBLAS::Axis::Short);
    RandBLAS::SparseSkOp<T> S(Ds, state);
    std::vector<T> Ahat(d * n, 0.0);
    RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                              d, n, m, (T)1.0, S, A_dense, m,
                              (T)0.0, Ahat.data(), d);

    std::vector<T> R_sk(n * n, 0.0);
    {
        std::vector<T> tau(n);
        lapack::geqrf(d, n, Ahat.data(), d, tau.data());
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i <= j; ++i)
                R_sk[i + j*n] = Ahat[i + j*d];
    }
    res.cond_Rsk = (double)condition_number(R_sk.data(), n, n);

    // ----------------------------------------------------------------
    // Explicit inverses of R_sk via two methods
    // ----------------------------------------------------------------

    // Method A: TRSM on identity  (path [2])
    std::vector<T> R_inv_trsm(n * n, 0.0);
    RandLAPACK::util::eye(n, n, R_inv_trsm.data());
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, n, n, (T)1.0,
               R_sk.data(), n, R_inv_trsm.data(), n);

    // Method B: LAPACK trtri  (path [3])
    std::vector<T> R_inv_trtri(R_sk.begin(), R_sk.end());
    lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_inv_trtri.data(), n);

    // Method C: GEQP3 factorization of R_sk  (path [4])
    //   R_sk * P = Q_buf * R_buf   (GEQP3)
    //   R_sk^{-1} = P * R_buf^{-1} * Q_buf^T
    //   Computed via Option A: ungqr (explicit Q) + TRSM (cheaper than trtri + ormqr)
    //     1. ungqr  -> Q_buf explicit (~4n^3/3 flops)
    //     2. W = Q_buf^T (explicit transpose, O(n^2))
    //     3. TRSM(Left, R_buf, W) -> W = R_buf^{-1} * Q_buf^T (~n^3/2 flops)
    //     4. scatter W by jpiv -> R_sk^{-1} = P * W
    //   Total: ~11n^3/6.  Alternative (trtri + ormqr) costs ~7n^3/3 — see dev log.
    std::vector<T> R_inv_geqp3(n * n, 0.0);
    {
        std::vector<T> R_copy(R_sk.begin(), R_sk.end());
        std::vector<int64_t> jpiv(n, 0);
        std::vector<T> tau_qr(n);
        lapack::geqp3(n, n, R_copy.data(), n, jpiv.data(), tau_qr.data());

        // Extract upper triangular R_buf before overwriting with Q
        std::vector<T> R_buf(n * n, 0.0);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i <= j; ++i)
                R_buf[i + j*n] = R_copy[i + j*n];

        // Expand Q_buf from Householder reflectors (overwrites R_copy)
        lapack::ungqr(n, n, n, R_copy.data(), n, tau_qr.data());

        // W = R_buf^{-1} * Q_buf^T via TRSM: initialize W as Q_buf^T, then solve in-place
        std::vector<T> W(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                W[i + j*n] = R_copy[j + i*n];  // W := Q_buf^T (col-major)
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, n, n, (T)1.0,
                   R_buf.data(), n, W.data(), n);

        // R_sk^{-1} = P * W: row (jpiv[k]-1) of R_inv gets row k of W
        for (int64_t k = 0; k < n; ++k)
            for (int64_t j = 0; j < n; ++j)
                R_inv_geqp3[(jpiv[k]-1) + j*n] = W[k + j*n];
    }

    // ----------------------------------------------------------------
    // Compute all four A_pre matrices
    // ----------------------------------------------------------------

    // Path [1]: TRSM in-place on A   ← CQRRT_expl
    std::vector<T> Apre1(A_dense, A_dense + m*n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0,
               R_sk.data(), n, Apre1.data(), m);

    // Path [2]: explicit inverse via TRSM-on-I + DGEMM
    std::vector<T> Apre2(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               m, n, n, (T)1.0,
               A_dense, m, R_inv_trsm.data(), n,
               (T)0.0, Apre2.data(), m);

    // Path [3]: explicit inverse via trtri + DGEMM
    std::vector<T> Apre3(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               m, n, n, (T)1.0,
               A_dense, m, R_inv_trtri.data(), n,
               (T)0.0, Apre3.data(), m);

    // Path [4]: explicit inverse via geqp3 + trtri + Q^T + permutation + DGEMM
    std::vector<T> Apre4(m * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
               m, n, n, (T)1.0,
               A_dense, m, R_inv_geqp3.data(), n,
               (T)0.0, Apre4.data(), m);

    // ----------------------------------------------------------------
    // Cross-path relative differences (reference = path [1])
    // ----------------------------------------------------------------
    res.rd_Apre_12 = (double)rel_diff(Apre1.data(), Apre2.data(), m*n);
    res.rd_Apre_13 = (double)rel_diff(Apre1.data(), Apre3.data(), m*n);
    res.rd_Apre_14 = (double)rel_diff(Apre1.data(), Apre4.data(), m*n);

    // ----------------------------------------------------------------
    // Step-by-step pipeline intermediates: paths [1] vs [2]
    // G = A_pre^T A_pre (upper triangle via SYRK)
    // R_chol = chol(G)  (POTRF in-place on upper triangle)
    // R_final = R_chol * R_sk  (TRMM)
    // ----------------------------------------------------------------
    auto make_G = [&](const std::vector<T>& Apre) {
        std::vector<T> G(n*n, 0.0);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                   n, m, (T)1.0, Apre.data(), m, (T)0.0, G.data(), n);
        return G;
    };
    auto make_Rchol = [&](std::vector<T> G) {
        lapack::potrf(Uplo::Upper, n, G.data(), n);
        return G;
    };
    auto make_Rfinal = [&](std::vector<T> Rchol) {
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, n, n, (T)1.0, R_sk.data(), n, Rchol.data(), n);
        return Rchol;
    };

    auto G1      = make_G(Apre1);
    auto G2      = make_G(Apre2);
    auto Rchol1  = make_Rchol(G1);
    auto Rchol2  = make_Rchol(G2);
    auto Rfinal1 = make_Rfinal(Rchol1);
    auto Rfinal2 = make_Rfinal(Rchol2);

    res.rd_G_12      = (double)rel_diff(G1.data(),      G2.data(),      n*n);
    res.rd_Rchol_12  = (double)rel_diff(Rchol1.data(),  Rchol2.data(),  n*n);
    res.rd_Rfinal_12 = (double)rel_diff(Rfinal1.data(), Rfinal2.data(), n*n);

    // ----------------------------------------------------------------
    // Per-path metrics
    // ----------------------------------------------------------------
    res.cond_Apre[0] = (double)condition_number(Apre1.data(), m, n);
    res.cond_Apre[1] = (double)condition_number(Apre2.data(), m, n);
    res.cond_Apre[2] = (double)condition_number(Apre3.data(), m, n);
    res.cond_Apre[3] = (double)condition_number(Apre4.data(), m, n);

    res.cond_G[0] = (double)gram_condition_number(Apre1.data(), m, n);
    res.cond_G[1] = (double)gram_condition_number(Apre2.data(), m, n);
    res.cond_G[2] = (double)gram_condition_number(Apre3.data(), m, n);
    res.cond_G[3] = (double)gram_condition_number(Apre4.data(), m, n);

    res.orth_Q[0] = (double)cholqr_orth_error(Apre1, A_dense, m, n, R_sk.data());
    res.orth_Q[1] = (double)cholqr_orth_error(Apre2, A_dense, m, n, R_sk.data());
    res.orth_Q[2] = (double)cholqr_orth_error(Apre3, A_dense, m, n, R_sk.data());
    res.orth_Q[3] = (double)cholqr_orth_error(Apre4, A_dense, m, n, R_sk.data());

    return res;
}

// ============================================================================
// Main benchmark
// ============================================================================

template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n";
        return 1;
    }

    std::string output_dir = argv[2];
    std::string mtx_path  = argv[3];
    T d_factor            = (T)std::stod(argv[4]);
    int64_t num_runs      = std::stol(argv[5]);
    int64_t sketch_nnz    = (argc >= 7) ? std::stol(argv[6]) : 4;

    // ----------------------------------------------------------------
    // Load matrix and materialize as dense
    // ----------------------------------------------------------------
    auto coo = RandLAPACK_extras::coo_from_matrix_market<T>(mtx_path);
    int64_t m = coo.n_rows, n = coo.n_cols;

    RandBLAS::sparse_data::csr::CSRMatrix<T> csr(m, n);
    RandBLAS::sparse_data::conversions::coo_to_csr(coo, csr);
    RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>>
        A_linop(m, n, csr);

    std::vector<T> A_dense(m * n, 0.0);
    {
        T* Eye = new T[n * n]();
        RandLAPACK::util::eye(n, n, Eye);
        A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                m, n, n, (T)1.0, Eye, n, (T)0.0, A_dense.data(), m);
        delete[] Eye;
    }

    T cond_A = condition_number(A_dense.data(), m, n);
    int64_t d = (int64_t)std::ceil(d_factor * n);

    std::cout << "\n=== CQRRT Preconditioner Comparison ===\n";
    std::cout << "  Matrix:     " << mtx_path << "\n";
    std::cout << "  Size:       " << m << " x " << n << "  (nnz=" << coo.nnz << ")\n";
    std::cout << "  d_factor:   " << d_factor << "  (d=" << d << ")\n";
    std::cout << "  sketch_nnz: " << sketch_nnz << "\n";
    std::cout << "  runs:       " << num_runs << "\n";
    std::cout << "  cond(A):    " << std::scientific << std::setprecision(3) << cond_A << "\n";
#ifdef _OPENMP
    std::cout << "  OMP threads: " << omp_get_max_threads() << "\n";
#endif
    std::cout << "\n";

    // ----------------------------------------------------------------
    // CSV setup
    // ----------------------------------------------------------------
    char time_buf[64];
    time_t now = time(nullptr);
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&now));
    std::string csv_path = output_dir + "/diagnostic_" + time_buf + ".csv";
    std::ofstream csv(csv_path);
    csv << "# CQRRT Preconditioner Comparison\n";
    csv << "# Date: " << ctime(&now);
    csv << "# Matrix: " << mtx_path << "\n";
    csv << "# m=" << m << " n=" << n << " d_factor=" << d_factor
        << " sketch_nnz=" << sketch_nnz << "\n";
    csv << "# cond_A=" << cond_A << "\n";
    csv << "run,"
        << "orth_Q1,orth_Q2,orth_Q3,orth_Q4,"
        << "cond_Apre1,cond_Apre2,cond_Apre3,cond_Apre4,"
        << "cond_G1,cond_G2,cond_G3,cond_G4,"
        << "rd_Apre_12,rd_Apre_13,rd_Apre_14,"
        << "rd_G_12,rd_Rchol_12,rd_Rfinal_12,"
        << "cond_Rsk\n";

    // ----------------------------------------------------------------
    // Runs
    // ----------------------------------------------------------------
    RandBLAS::RNGState<RNG> base_state(42);
    for (int64_t r = 0; r < num_runs; ++r) {
        auto state = base_state;
        if (r > 0) state.key.incr(r);

        auto res = run_trial<T, RNG>(A_dense.data(), m, n, d_factor, sketch_nnz, state);

        printf("  run %ld  orth_error(Q = A * R_final^{-1}):\n", r);
        printf("    [1] expl_trsm:       %12.3e\n", res.orth_Q[0]);
        printf("    [2] expl_inv_trsm:   %12.3e\n", res.orth_Q[1]);
        printf("    [3] expl_inv_trtri:  %12.3e\n", res.orth_Q[2]);
        printf("    [4] expl_inv_geqp3:  %12.3e\n", res.orth_Q[3]);

        printf("  run %ld  cond(A_pre):\n", r);
        printf("    [1] expl_trsm:       %12.3e\n", res.cond_Apre[0]);
        printf("    [2] expl_inv_trsm:   %12.3e\n", res.cond_Apre[1]);
        printf("    [3] expl_inv_trtri:  %12.3e\n", res.cond_Apre[2]);
        printf("    [4] expl_inv_geqp3:  %12.3e\n", res.cond_Apre[3]);

        printf("  run %ld  cond(G = A_pre^T A_pre):\n", r);
        printf("    [1] expl_trsm:       %12.3e\n", res.cond_G[0]);
        printf("    [2] expl_inv_trsm:   %12.3e\n", res.cond_G[1]);
        printf("    [3] expl_inv_trtri:  %12.3e\n", res.cond_G[2]);
        printf("    [4] expl_inv_geqp3:  %12.3e\n", res.cond_G[3]);

        printf("  run %ld  rel_diff(A_pre) vs [1]:\n", r);
        printf("    rd_12 (trsm-on-I):   %12.3e\n", res.rd_Apre_12);
        printf("    rd_13 (trtri):       %12.3e\n", res.rd_Apre_13);
        printf("    rd_14 (geqp3):       %12.3e\n", res.rd_Apre_14);

        printf("  run %ld  step-by-step divergence [1] vs [2] (expl vs linop):\n", r);
        printf("    M^sk:    %12.3e  (0 by construction — shared sketch)\n", 0.0);
        printf("    R^sk:    %12.3e  (0 by construction — shared sketch)\n", 0.0);
        printf("    MR^pre:  %12.3e\n", res.rd_Apre_12);
        printf("    G:       %12.3e\n", res.rd_G_12);
        printf("    R^chol:  %12.3e\n", res.rd_Rchol_12);
        printf("    R:       %12.3e\n", res.rd_Rfinal_12);

        printf("  run %ld  cond(R_sk): %9.3e\n\n", r, res.cond_Rsk);

        csv << r << ","
            << std::scientific << std::setprecision(6)
            << res.orth_Q[0]    << "," << res.orth_Q[1]    << ","
            << res.orth_Q[2]    << "," << res.orth_Q[3]    << ","
            << res.cond_Apre[0] << "," << res.cond_Apre[1] << ","
            << res.cond_Apre[2] << "," << res.cond_Apre[3] << ","
            << res.cond_G[0]    << "," << res.cond_G[1]    << ","
            << res.cond_G[2]    << "," << res.cond_G[3]    << ","
            << res.rd_Apre_12    << "," << res.rd_Apre_13    << ","
            << res.rd_Apre_14   << ","
            << res.rd_G_12      << "," << res.rd_Rchol_12  << ","
            << res.rd_Rfinal_12 << ","
            << res.cond_Rsk     << "\n";
    }
    csv.close();

    std::cout << "  Legend:\n";
    std::cout << "    [1] expl_trsm:      DTRSM_R(A, R_sk) in-place                    <- CQRRT_expl\n";
    std::cout << "    [2] expl_inv_trsm:  TRSM(I, R_sk)->R_inv; DGEMM(A, R_inv)\n";
    std::cout << "    [3] expl_inv_trtri: TRTRI(R_sk)->R_inv;   DGEMM(A, R_inv)\n";
    std::cout << "    [4] expl_inv_geqp3: GEQP3(R_sk)=Q*R_buf*P^T; R_inv=P*TRTRI(R_buf)*Q^T; DGEMM(A, R_inv)\n";
    std::cout << "\n  CSV written to: " << csv_path << "\n";

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n";
        return 1;
    }
    std::string prec = argv[1];
    if (prec == "double") return run_benchmark<double>(argc, argv);
    if (prec == "float")  return run_benchmark<float>(argc, argv);
    std::cerr << "Unknown precision '" << prec << "' (use double or float)\n";
    return 1;
}
