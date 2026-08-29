// CQRRT preconditioner comparison benchmark
//
// Isolates the effect of different methods for forming R_sk^{-1} on the final
// orthogonality quality of CQRRT. Tests seven paths:
//
//   [1] expl_trsm:           DTRSM_R(A, R_sk) in-place                <- CQRRT_expl path
//   [2] expl_inv_trsm_left:  solve R_sk * X = I (TRSM Side::Left) -> R_inv; DGEMM(A, R_inv)
//                            (column-ordered solve; the RandLAPACK default,
//                             PCholQRPrecondMethod::TRSM_IDENTITY)
//   [3] expl_inv_trsm_right: solve X * R_sk = I (TRSM Side::Right) -> R_inv; DGEMM(A, R_inv)
//                            (reversed, row-ordered solve; kept to document the
//                             solve-ordering effect, NOT shipped)
//   [4] expl_inv_trtri:      TRTRI(R_sk) -> R_inv;   DGEMM(A, R_inv)   (LAPACK trtri)
//   [5] expl_inv_geqp3:      GEQP3(R_sk) = Q*R_buf*P^T;
//                            R_inv = P * TRSM(R_buf, Q^T); DGEMM(A, R_inv)
//   [6] expl_inv_svd:        GESDD(R_sk) = U*S*Vt;
//                            R_inv = V * diag(1/S) * U^T; DGEMM(A, R_inv)
//   [7] expl_inv_bqrrp:      BQRRP(R_sk)=Q*R_buf*P^T; R_inv=P*TRSM(R_buf,Q^T); DGEMM(A, R_inv)
//
//   Path [1] never forms R_sk^{-1} explicitly (backward stable).
//   Paths [2]-[7] all form R_sk^{-1} explicitly via different methods.
//   Paths [2] and [3] differ ONLY in how the triangular system is posed;
//   the ordering governs the error of the composite product M * R_inv.
//   Path [5] uses a rank-revealing QR to invert R_sk; the Q factor makes
//   the inversion well-conditioned even when R_sk itself is ill-conditioned.
//   Path [6] uses the SVD (gold standard for stability).
//   Path [7] uses BQRRP (blocked randomized QRCP), the randomized
//   counterpart of path [5]'s GEQP3.
//
//   All seven paths use the same sketch (same RNG state).
//
//   Per-path metrics:
//     cond(A_pre)             : condition number of the preconditioned matrix
//     cond(G = A_pre^T A_pre) : condition number of the Gram matrix (input to Cholesky)
//     orth_error(Q)           : full-pipeline: G=SYRK(A_pre), R_chol=chol(G),
//                               R_final=R_chol*R_sk, Q=A_orig*R_final^{-1}
//
//   Cross-path relative differences of A_pre (reference = path [1]):
//     rd_1p = ||A_pre[1] - A_pre[p]|| / ||A_pre[1]||   for p = 2..7
//
//   Step-by-step pipeline divergence between paths [1] and [2] (CQRRT_expl vs CQRRT_linop):
//   Each path uses the same RNG seed but a different sketch code path, mirroring actual impls.
//   Path [1] (CQRRT_expl):   sketch via sketch_general(S, A_dense); TRSM in-place; SYRK
//   Path [2] (CQRRT_linop):  sketch via A_linop(Side::Right, S) [SpGEMM];
//                             TRSM_IDENTITY → R_inv; A_linop fwd/adj;
//                             TRSM(Left,Trans) Gram completion on R_sk; TRMM R_final
//     rd_Msk_12   = ||Ahat1 - Ahat2|| / ||Ahat1||      (raw sketch S*A, different code paths)
//     rd_Rsk_12   = ||R_sk1 - R_sk2|| / ||R_sk1||      (QR of above)
//     rd_G_12     = ||G1 - G2|| / ||G1||                (Gram matrix)
//     rd_Rchol_12 = ||Rchol1 - Rchol2|| / ||Rchol1||   (Cholesky factor)
//     rd_Rfinal_12= ||Rfinal1 - Rfinal2|| / ||Rfinal1|| (final R = R_chol * R_sk)
//
//   Sketch diagnostic:
//     cond(R_sk)
//
// Usage (file mode):
//   ./CQRRT_diagnostic <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]
//
// Usage (generate mode):
//   ./CQRRT_diagnostic <prec> <output_dir> gen <m> <n> <kappa> <density> <d_factor> <runs> [sketch_nnz]
//
// NOTE: fidelity gaps vs the shipped drivers (cholqr_primitive, comps/rl_cholqr.hh):
//   - No adaptive Cholesky-shift retry is modeled here; a potrf failure in Part B
//     is left unretried, so failures are expected in the high-kappa regime this
//     tool probes.
//   - The RANDLAPACK_GRAM_LEFT=gemm per-block-GEMM Gram completion arm is not
//     modeled; this tool always completes the Gram with a single TRSM.
//   - R_final is formed via TRMM(Side::Right, R_sk) here, versus the primitive's
//     TRMM(Side::Left, R_chol); same mathematical product, different rounding.
//   - This tool computes the sketch dimension d = ceil(d_factor * n), while the
//     shipped drivers (rl_cqrrt.hh) truncate: d = (int64_t)(d_factor * n). The
//     two agree only when d_factor * n is exactly representable; otherwise this
//     tool's sketch is one row taller than the driver it is meant to model.

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <array>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../extras/misc/ext_util.hh"
#include "RandLAPACK/testing/rl_test_utils.hh"
#include "cqrrt_bench_common.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using blas::Layout;
using blas::Op;
using blas::Side;
using blas::Uplo;
using blas::Diag;
using RandLAPACK::bench::quote_join_argv;
using RandLAPACK::bench::get_hostname;

// ============================================================================
// Path constants
// ============================================================================

static constexpr int N_PATHS = 7;

static constexpr const char* PATH_NAMES[N_PATHS] = {
    "expl_trsm",
    "expl_inv_trsm_left",
    "expl_inv_trsm_right",
    "expl_inv_trtri",
    "expl_inv_geqp3",
    "expl_inv_svd",
    "expl_inv_bqrrp",
};

static constexpr const char* PATH_DESCS[N_PATHS] = {
    "DTRSM_R(A, R_sk) in-place                              <- CQRRT_expl",
    "solve R_sk*X=I (Side::Left)->R_inv; DGEMM(A, R_inv)    <- RandLAPACK default",
    "solve X*R_sk=I (Side::Right)->R_inv; DGEMM(A, R_inv)   (reversed ordering, NOT shipped)",
    "TRTRI(R_sk)->R_inv;   DGEMM(A, R_inv)",
    "GEQP3(R_sk)=Q*R_buf*P^T; R_inv=P*TRSM(R_buf,Q^T); DGEMM(A, R_inv)",
    "GESDD(R_sk)=U*S*Vt; R_inv=V*diag(1/S)*U^T; DGEMM(A, R_inv)",
    "BQRRP(R_sk)=Q*R_buf*P^T; R_inv=P*TRSM(R_buf,Q^T); DGEMM(A, R_inv)",
};

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
static T rel_diff(const T* A, const T* B, int64_t len) {
    T nd = 0, na = 0;
    for (int64_t i = 0; i < len; ++i) {
        T d = A[i] - B[i];
        nd += d * d; na += A[i] * A[i];
    }
    return (na > 0) ? std::sqrt(nd / na) : std::sqrt(nd);
}

// Condition number of the Gram matrix G = A_pre^T A_pre. Uses syrk
// (upper triangle) + symmetrize so gesdd inside cond_num_check sees a
// full symmetric matrix.
template <typename T>
static T gram_condition_number(const T* A_pre, int64_t m, int64_t n) {
    std::vector<T> G(n * n, 0.0);
    blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
               n, m, (T)1.0, A_pre, m, (T)0.0, G.data(), n);
    RandBLAS::symmetrize(Layout::ColMajor, Uplo::Upper, n, G.data(), n);
    return RandLAPACK::util::cond_num_check<T>(n, n, G.data(), /*verbose=*/false);
}

// Full CQRRT pipeline (matching CQRRT_linops Gram computation):
//   G = A_orig^T * A_pre         (GEMM, full n×n, not exploiting symmetry)
//   G = (R_sketch)^{-T} * G     (TRSM Left, backward-stable left factor on original R^sk)
//   Zero lower triangle of G    (POTRF/TRMM only use upper triangle; lower has TRSM output)
//   R_chol = chol(G)             (POTRF, upper triangle only)
//   R_final = R_chol * R_sketch  (TRMM)
//   Q = A_orig * R_final^{-1}   (TRSM on copy of A_orig)
//   return orth_error(Q)
// Does NOT modify A_pre or A_orig.
template <typename T>
static T cholqr_orth_error(const std::vector<T>& A_pre, const T* A_orig,
                            int64_t m, int64_t n, const T* R_sketch) {
    std::vector<T> G(n * n, 0.0);
    blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans,
               n, n, m, (T)1.0, A_orig, m, A_pre.data(), m, (T)0.0, G.data(), n);
    blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit,
               n, n, (T)1.0, R_sketch, n, G.data(), n);
    // Zero strictly lower triangle: TRSM fills the full n×n matrix; the subsequent
    // TRMM(Right,Upper) reads lower-triangle entries of G when computing upper-triangle
    // output entries and would produce corrupted R_chol if left non-zero.
    if (n > 1)
        lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, &G.data()[1], n);
    if (lapack::potrf(Uplo::Upper, n, G.data(), n))
        return std::numeric_limits<T>::infinity();
    blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, n, n, (T)1.0, R_sketch, n, G.data(), n);
    std::vector<T> Q(A_orig, A_orig + m * n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, G.data(), n, Q.data(), m);
    return RandLAPACK::testing::orthogonality_error<T>(Q.data(), m, n);
}

// ============================================================================
// One trial: N_PATHS-path orth comparison (shared sketch) +
//            independent path [1] vs [2] step-by-step divergence
// ============================================================================

template <typename T>
struct TrialResult {
    // Per-path metrics (shared sketch, all N_PATHS paths)
    double cond_Apre[N_PATHS];
    double cond_G[N_PATHS];
    double orth_Q[N_PATHS];
    // Cross-path relative differences of A_pre (reference = path [1], shared
    // sketch): entry p holds rel_diff(Apre[0], Apre[p]); entry 0 is unused.
    double rd_Apre_vs1[N_PATHS];
    // Step-by-step pipeline divergence: paths [1] vs [2], faithful to actual implementations
    //   Path [1] (CQRRT_expl):  sketch via sketch_general(S, A_dense); TRSM in-place; SYRK
    //   Path [2] (CQRRT_linop): sketch via A_linop(Side::Right, S) [SpGEMM];
    //                            TRSM_IDENTITY → R_inv; A_linop fwd/adj;
    //                            TRSM(Left,Trans) Gram completion on R_sk; TRMM R_final
    // -1 = not computed: the potrf that would have produced this quantity failed.
    double rd_Msk_12;       // M^sk:  raw sketch Ahat = S*A (different code paths)
    double rd_Rsk_12;       // R^sk:  QR factor of the above
    double rd_Apre_12_step; // MR^pre: TRSM in-place vs A_linop(fwd, R_inv)
    double rd_G_12;         // Gram:  SYRK(A_pre) vs A_linop(adj, A_pre)+TRSM(R_sk) completion
    double rd_Rchol_12;     // R^chol: Cholesky factor
    double rd_Rfinal_12;    // R:     R_final = R_chol * R_sk
    // Sketch diagnostic
    double cond_Rsk;
};

template <typename T, typename RNG, typename LinOpT>
static TrialResult<T> run_trial(
    LinOpT& A_linop,
    const T* A_dense,
    int64_t m, int64_t n,
    T d_factor, int64_t sketch_nnz,
    RandBLAS::RNGState<RNG>& state)
{
    TrialResult<T> res{};
    // Part B early-returns on a Cholesky failure before every rd_* field is
    // computed; a value-initialized 0.0 would then read as "paths agree
    // exactly," exactly in the ill-conditioned regime this tool exists to
    // probe. Seed the suite-wide -1 sentinel ("not computed, factorization
    // failed") and let each Part B step overwrite it on success.
    res.rd_Msk_12 = res.rd_Rsk_12 = res.rd_Apre_12_step = -1.0;
    res.rd_G_12 = res.rd_Rchol_12 = res.rd_Rfinal_12 = -1.0;
    int64_t d = (int64_t)std::ceil(d_factor * n);

    // Save RNG state before any sketching; Part B (step-by-step) uses this
    // to compute independent sketches for paths [1] and [2].
    auto initial_state = state;

    // ----------------------------------------------------------------
    // Part A: Shared sketch → R_sk, N_PATHS-path orth comparison
    // ----------------------------------------------------------------
    RandBLAS::SparseDist Ds(d, m, sketch_nnz, RandBLAS::Axis::Short);
    RandBLAS::SparseSkOp<T> S(Ds, state);
    // Advance state past what S consumed: the BQRRP path below (Method E)
    // also draws from `state` and must not reuse this sketch's seed material.
    state = S.next_state;
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
    res.cond_Rsk = (double)RandLAPACK::util::cond_num_check<T>(n, n, R_sk.data(), /*verbose=*/false);

    // ----------------------------------------------------------------
    // Explicit inverses of R_sk via two methods
    // ----------------------------------------------------------------

    // Method A1: TRSM on identity, column-ordered solve  (path [2])
    //   Solve R_sk * X = I (Side::Left): each column of X is an independent
    //   backward-stable solve R_sk * x_j = e_j. This is the RandLAPACK default
    //   (PCholQRPrecondMethod::TRSM_IDENTITY, comps/rl_cholqr.hh), including
    //   the trailing lower-triangle laset the primitive performs.
    std::vector<T> R_inv_trsm_left(n * n, T(0));
    RandLAPACK::util::eye(n, n, R_inv_trsm_left.data());
    blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, n, n, (T)1.0,
               R_sk.data(), n, R_inv_trsm_left.data(), n);
    if (n > 1)
        lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, R_inv_trsm_left.data() + 1, n);

    // Method A2: TRSM on identity, reversed row-ordered solve  (path [3])
    //   Solve X * R_sk = I (Side::Right): rows of X carry independent
    //   perturbations of R_sk, and the error of the composite M * X scales
    //   with kappa(R_sk). Kept only to document the solve-ordering effect.
    std::vector<T> R_inv_trsm_right(n * n, T(0));
    RandLAPACK::util::eye(n, n, R_inv_trsm_right.data());
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, n, n, (T)1.0,
               R_sk.data(), n, R_inv_trsm_right.data(), n);
    if (n > 1)
        lapack::laset(MatrixType::Lower, n-1, n-1, (T)0.0, (T)0.0, R_inv_trsm_right.data() + 1, n);

    // Method B: LAPACK trtri  (path [4])
    std::vector<T> R_inv_trtri(R_sk.begin(), R_sk.end());
    lapack::trtri(Uplo::Upper, Diag::NonUnit, n, R_inv_trtri.data(), n);

    // Method C: GEQP3 factorization of R_sk  (path [5])
    //   R_sk * P = Q_buf * R_buf   (GEQP3)
    //   R_sk^{-1} = P * R_buf^{-1} * Q_buf^T
    //   Computed via Option A: ungqr (explicit Q) + TRSM (cheaper than trtri + ormqr)
    //     1. ungqr  -> Q_buf explicit (~4n^3/3 flops)
    //     2. W = Q_buf^T (explicit transpose, O(n^2))
    //     3. TRSM(Left, R_buf, W) -> W = R_buf^{-1} * Q_buf^T (~n^3/2 flops)
    //     4. scatter W by jpiv -> R_sk^{-1} = P * W
    //   Total: ~11n^3/6.  Alternative (trtri + ormqr) costs ~7n^3/3.
    std::vector<T> R_inv_geqp3(n * n, 0.0);
    {
        std::vector<T> R_copy(R_sk.begin(), R_sk.end());
        std::vector<int64_t> jpiv(n, 0);
        std::vector<T> tau_qr(n);
        lapack::geqp3(n, n, R_copy.data(), n, jpiv.data(), tau_qr.data());

        // Extract upper triangular R_buf before overwriting with Q
        std::vector<T> R_buf(n * n, 0.0);
        lapack::lacpy(MatrixType::Upper, n, n, R_copy.data(), n, R_buf.data(), n);

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

    // Method D: SVD of R_sk  (path [6])
    //   R_sk = U * diag(s) * Vt
    //   R_sk^{-1} = V * diag(1/s) * U^T = Vt^T * diag(1/s) * U^T
    std::vector<T> R_inv_svd(n * n, 0.0);
    {
        std::vector<T> R_copy(R_sk.begin(), R_sk.end());
        std::vector<T> U(n * n, 0.0), Vt(n * n, 0.0), s(n);
        lapack::gesdd(lapack::Job::AllVec, n, n, R_copy.data(), n,
                      s.data(), U.data(), n, Vt.data(), n);
        // Scale row k of Vt by 1/s[k]: Vt[k + j*n] is row k, col j (col-major)
        for (int64_t k = 0; k < n; ++k)
            for (int64_t j = 0; j < n; ++j)
                Vt[k + j*n] /= s[k];
        // R_inv = scaled_Vt^T * U^T
        blas::gemm(Layout::ColMajor, Op::Trans, Op::Trans,
                   n, n, n, (T)1.0, Vt.data(), n, U.data(), n,
                   (T)0.0, R_inv_svd.data(), n);
    }

    // Method E: BQRRP factorization of R_sk  (path [7])
    //   Same output format as GEQP3: R_sk * P = Q_buf * R_buf, then
    //   R_sk^{-1} = P * R_buf^{-1} * Q_buf^T.
    //   BQRRP uses blocked randomized QRCP; block size matches CQRRT_linops
    //   adaptive heuristic (1.0 for n <= 2000, 0.5 for n <= 8000, 1/32 else).
    std::vector<T> R_inv_bqrrp(n * n, 0.0);
    {
        std::vector<T> R_copy(R_sk.begin(), R_sk.end());
        std::vector<int64_t> jpiv(n, 0);
        std::vector<T> tau_qr(n);

        T block_ratio;
        if (n <= 2000)      block_ratio = (T)1.0;
        else if (n <= 8000) block_ratio = (T)0.5;
        else                block_ratio = (T)1.0 / (T)32;
        int64_t bqrrp_block = std::max<int64_t>(1, (int64_t)(n * block_ratio));
        RandLAPACK::BQRRP<T, RNG> bqrrp(false, bqrrp_block);
        bqrrp.call(n, n, R_copy.data(), n, (T)1.0, tau_qr.data(), jpiv.data(), state);

        std::vector<T> R_buf(n * n, 0.0);
        lapack::lacpy(MatrixType::Upper, n, n, R_copy.data(), n, R_buf.data(), n);

        lapack::ungqr(n, n, n, R_copy.data(), n, tau_qr.data());

        std::vector<T> W(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                W[i + j*n] = R_copy[j + i*n];  // W := Q_buf^T (col-major)
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, n, n, (T)1.0,
                   R_buf.data(), n, W.data(), n);

        for (int64_t k = 0; k < n; ++k)
            for (int64_t j = 0; j < n; ++j)
                R_inv_bqrrp[(jpiv[k]-1) + j*n] = W[k + j*n];
    }

    // ----------------------------------------------------------------
    // Compute all N_PATHS preconditioned matrices:
    //   Apre[0]: TRSM in-place                (path [1], CQRRT_expl)
    //   Apre[p]: GEMM + R_invs[p-1], p >= 1   (paths [2]..[7])
    // ----------------------------------------------------------------
    std::array<std::vector<T>, N_PATHS> Apre;
    for (auto& a : Apre) a.resize(m * n, T(0));

    Apre[0].assign(A_dense, A_dense + m*n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, R_sk.data(), n, Apre[0].data(), m);

    const T* R_invs[N_PATHS - 1] = {
        R_inv_trsm_left.data(), R_inv_trsm_right.data(), R_inv_trtri.data(),
        R_inv_geqp3.data(), R_inv_svd.data(), R_inv_bqrrp.data()
    };
    for (int p = 1; p < N_PATHS; ++p)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   m, n, n, (T)1.0, A_dense, m, R_invs[p-1], n, (T)0.0, Apre[p].data(), m);

    // ----------------------------------------------------------------
    // Cross-path relative differences (reference = path [1] = Apre[0])
    // ----------------------------------------------------------------
    for (int p = 1; p < N_PATHS; ++p)
        res.rd_Apre_vs1[p] = (double)rel_diff(Apre[0].data(), Apre[p].data(), m*n);

    // ----------------------------------------------------------------
    // Part B: Independent step-by-step divergence, paths [1] vs [2]
    //
    // Both paths compute their own sketch and R_sk from initial_state
    // (same seed → same result, but as separate objects).
    //
    // Path [1] (CQRRT_expl): TRSM in-place on A; Gram via SYRK.
    // Path [2] (CQRRT_linop, block_size=0):
    //   R_inv = TRSM_IDENTITY(R_sk);
    //   A_pre = GEMM(A, R_inv)             [fwd linop call]
    //   G     = GEMM(A^T, A_pre)           [adj linop call]
    //   G     = TRSM(Left,Trans,R_sk,G)    [complete Gram: (R_sk)^{-T} * A^T * A * R_inv,
    //                                        backward-stable solve on the original R_sk]
    // ----------------------------------------------------------------

    // Run Part B as a lambda so early returns on Cholesky failure are clean.
    [&]() {
        // ---- Step 1: Sketch ----
        // Path [1] (CQRRT_expl):  sketch_general(S, A_dense), left SPMM on dense copy
        RandBLAS::SparseDist Ds_1(d, m, sketch_nnz, RandBLAS::Axis::Short);
        RandBLAS::SparseSkOp<T> S_1(Ds_1, initial_state);
        std::vector<T> Ahat_1(d * n, 0.0);
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                  d, n, m, (T)1.0, S_1, A_dense, m,
                                  (T)0.0, Ahat_1.data(), d);

        // Path [2] (CQRRT_linop): A_linop(Side::Right, S), SpGEMM on sparse CSR matrix
        RandBLAS::SparseDist Ds_2(d, m, sketch_nnz, RandBLAS::Axis::Short);
        RandBLAS::SparseSkOp<T> S_2(Ds_2, initial_state);  // same seed → same S
        std::vector<T> Ahat_2(d * n, 0.0);
        A_linop(Side::Right, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                d, n, m, (T)1.0, S_2, (T)0.0, Ahat_2.data(), d);

        // M^sk diff: before geqrf overwrites the sketch buffers
        res.rd_Msk_12 = (double)rel_diff(Ahat_1.data(), Ahat_2.data(), d*n);

        // ---- Step 2: QR → R_sk ----
        std::vector<T> R_sk_1(n * n, 0.0);
        {
            std::vector<T> tau_1(n);
            lapack::geqrf(d, n, Ahat_1.data(), d, tau_1.data());
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i <= j; ++i)
                    R_sk_1[i + j*n] = Ahat_1[i + j*d];
        }
        std::vector<T> R_sk_2(n * n, 0.0);
        {
            std::vector<T> tau_2(n);
            lapack::geqrf(d, n, Ahat_2.data(), d, tau_2.data());
            for (int64_t j = 0; j < n; ++j)
                for (int64_t i = 0; i <= j; ++i)
                    R_sk_2[i + j*n] = Ahat_2[i + j*d];
        }
        res.rd_Rsk_12 = (double)rel_diff(R_sk_1.data(), R_sk_2.data(), n*n);

        // ---- Path [1]: CQRRT_expl, TRSM in-place, SYRK ----
        std::vector<T> Apre_1(A_dense, A_dense + m*n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, m, n, (T)1.0, R_sk_1.data(), n, Apre_1.data(), m);

        std::vector<T> G_1(n*n, 0.0);
        blas::syrk(Layout::ColMajor, Uplo::Upper, Op::Trans,
                   n, m, (T)1.0, Apre_1.data(), m, (T)0.0, G_1.data(), n);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j+1; i < n; ++i)
                G_1[i + j*n] = G_1[j + i*n];

        std::vector<T> Rchol_1(G_1);
        if (lapack::potrf(Uplo::Upper, n, Rchol_1.data(), n)) return;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j+1; i < n; ++i)
                Rchol_1[i + j*n] = (T)0.0;

        std::vector<T> Rfinal_1(Rchol_1);
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, n, n, (T)1.0, R_sk_1.data(), n, Rfinal_1.data(), n);

        // ---- Path [2]: CQRRT_linop, TRSM_IDENTITY, linop fwd/adj, TRSM Gram completion ----
        // R_inv via TRSM_IDENTITY: solve R_sk_2 * X = I (Side::Left), matching
        // cholqr_primitive's shipping default (comps/rl_cholqr.hh).
        std::vector<T> R_inv_2(n * n, T(0));
        RandLAPACK::util::eye(n, n, R_inv_2.data());
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, n, n, (T)1.0, R_sk_2.data(), n, R_inv_2.data(), n);
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j+1; i < n; ++i)
                R_inv_2[i + j*n] = (T)0.0;

        // fwd: Apre_2 = A * R_inv_2  via A_linop(Side::Left, NoTrans)
        std::vector<T> Apre_2(m * n, 0.0);
        A_linop(Side::Left, Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                m, n, n, (T)1.0, R_inv_2.data(), n, (T)0.0, Apre_2.data(), m);
        res.rd_Apre_12_step = (double)rel_diff(Apre_1.data(), Apre_2.data(), m*n);

        // adj: G_2 = A^T * Apre_2  via A_linop(Side::Left, Trans)
        std::vector<T> G_2(n * n, 0.0);
        A_linop(Side::Left, Layout::ColMajor, Op::Trans, Op::NoTrans,
                n, n, m, (T)1.0, Apre_2.data(), m, (T)0.0, G_2.data(), n);
        // Complete Gram: G_2 = (R_sk_2)^{-T} * G_2  (backward-stable TRSM on original R_sk)
        blas::trsm(Layout::ColMajor, Side::Left, Uplo::Upper, Op::Trans, Diag::NonUnit,
                   n, n, (T)1.0, R_sk_2.data(), n, G_2.data(), n);
        res.rd_G_12 = (double)rel_diff(G_1.data(), G_2.data(), n*n);

        std::vector<T> Rchol_2(G_2);
        if (lapack::potrf(Uplo::Upper, n, Rchol_2.data(), n)) return;
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j+1; i < n; ++i)
                Rchol_2[i + j*n] = (T)0.0;
        res.rd_Rchol_12 = (double)rel_diff(Rchol_1.data(), Rchol_2.data(), n*n);

        std::vector<T> Rfinal_2(Rchol_2);
        blas::trmm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
                   Diag::NonUnit, n, n, (T)1.0, R_sk_2.data(), n, Rfinal_2.data(), n);
        res.rd_Rfinal_12 = (double)rel_diff(Rfinal_1.data(), Rfinal_2.data(), n*n);
    }();

    // ----------------------------------------------------------------
    // Per-path metrics
    // ----------------------------------------------------------------
    for (int p = 0; p < N_PATHS; ++p) {
        res.cond_Apre[p] = (double)RandLAPACK::util::cond_num_check<T>(m, n, Apre[p].data(), /*verbose=*/false);
        res.cond_G[p]    = (double)gram_condition_number(Apre[p].data(), m, n);
        res.orth_Q[p]    = (double)cholqr_orth_error(Apre[p], A_dense, m, n, R_sk.data());
    }

    return res;
}

// ============================================================================
// Shared: write CSV header and run trials given a dense matrix
// ============================================================================

template <typename T, typename LinOpT, typename RNG = r123::Philox4x32>
static void write_csv_and_run(
    LinOpT& A_linop,
    const std::vector<T>& A_dense,
    int64_t m, int64_t n,
    T d_factor, int64_t sketch_nnz, int64_t num_runs,
    T cond_A, double kappa_target,   // kappa_target < 0 means "from file"
    const std::string& matrix_label,
    const std::string& output_dir,
    const std::string& argv_line)
{
    std::string ts = make_run_timestamp();
    std::string csv_path = output_dir + "/diagnostic_" + ts + ".csv";
    std::ofstream csv(csv_path);

    csv << "# CQRRT Preconditioner Comparison\n";
    csv << "# Date: " << ts << "\n";
    csv << "# host: " << get_hostname() << "\n";
    csv << "# argv: " << argv_line << "\n";
    csv << "# RANDLAPACK_GIT_COMMIT=" << RandLAPACK::bench::env_or("RANDLAPACK_GIT_COMMIT") << "\n";
    csv << "# This tool hand-rolls its own CQRRT pipeline rather than calling\n";
    csv << "#   cholqr_primitive, so the RANDLAPACK_GRAM_LEFT / RANDLAPACK_SCHOLQR3_SHIFT /\n";
    csv << "#   RANDLAPACK_BLAS2_THREADS / RANDLAPACK_FFT_THREADS / RANDLAPACK_SOLVE_FFT_MATCH\n";
    csv << "#   knobs that steer the shipped drivers do not affect any column below.\n";
    csv << "# Matrix: " << matrix_label << "\n";
    csv << "# m=" << m << " n=" << n << " d_factor=" << d_factor
        << " sketch_nnz=" << sketch_nnz << "\n";
    csv << "# cond_A=" << std::scientific << std::setprecision(6) << cond_A;
    if (kappa_target > 0)
        csv << " kappa_target=" << std::scientific << std::setprecision(6) << kappa_target;
    csv << "\n";
    for (int p = 0; p < N_PATHS; ++p)
        csv << "# path " << (p+1) << ": " << PATH_NAMES[p] << "\n";
    csv << "# rd_Msk_12/rd_Rsk_12/rd_Apre_12_step/rd_G_12/rd_Rchol_12/rd_Rfinal_12:\n";
    csv << "#   -1 = not computed; Part B's Cholesky factorization failed before this\n";
    csv << "#   quantity was formed (no adaptive-shift retry is modeled here, so this is\n";
    csv << "#   expected in the high-kappa regime this tool probes).\n";
    csv << "run,";
    for (int p = 1; p <= N_PATHS; ++p) csv << "orth_Q" << p << ",";
    for (int p = 1; p <= N_PATHS; ++p) csv << "cond_Apre" << p << ",";
    for (int p = 1; p <= N_PATHS; ++p) csv << "cond_G" << p << ",";
    for (int p = 2; p <= N_PATHS; ++p) csv << "rd_Apre_1" << p << ",";
    csv << "rd_Msk_12,rd_Rsk_12,rd_Apre_12_step,rd_G_12,rd_Rchol_12,rd_Rfinal_12,"
        << "cond_Rsk\n";

    RandBLAS::RNGState<RNG> base_state(42);
    for (int64_t r = 0; r < num_runs; ++r) {
        auto state = base_state;
        if (r > 0) state.key.incr(r);

        auto res = run_trial<T, RNG>(A_linop, A_dense.data(), m, n, d_factor, sketch_nnz, state);

        printf("  run %lld  orth_error(Q = A * R_final^{-1}):\n", (long long)r);
        for (int p = 0; p < N_PATHS; ++p)
            printf("    [%d] %-18s %12.3e\n", p+1, PATH_NAMES[p], res.orth_Q[p]);

        printf("  run %lld  cond(MR^pre):\n", (long long)r);
        for (int p = 0; p < N_PATHS; ++p)
            printf("    [%d] %-18s %12.3e\n", p+1, PATH_NAMES[p], res.cond_Apre[p]);

        printf("  run %lld  cond(G = MR^pre^T MR^pre):\n", (long long)r);
        for (int p = 0; p < N_PATHS; ++p)
            printf("    [%d] %-18s %12.3e\n", p+1, PATH_NAMES[p], res.cond_G[p]);

        printf("  run %lld  rel_diff(MR^pre) vs [1]:\n", (long long)r);
        for (int p = 1; p < N_PATHS; ++p)
            printf("    rd_1%d (%-19s): %12.3e\n", p+1, PATH_NAMES[p], res.rd_Apre_vs1[p]);

        printf("  run %lld  step-by-step divergence [1] vs [2] (expl: sketch_general; linop: SpGEMM):\n", (long long)r);
        printf("    M^sk:    %12.3e\n", res.rd_Msk_12);
        printf("    R^sk:    %12.3e\n", res.rd_Rsk_12);
        printf("    MR^pre:  %12.3e\n", res.rd_Apre_12_step);
        printf("    G:       %12.3e\n", res.rd_G_12);
        printf("    R^chol:  %12.3e\n", res.rd_Rchol_12);
        printf("    R:       %12.3e\n", res.rd_Rfinal_12);

        printf("  run %lld  cond(R_sk): %9.3e\n\n", (long long)r, res.cond_Rsk);

        csv << r << "," << std::scientific << std::setprecision(6);
        for (int p = 0; p < N_PATHS; ++p) csv << res.orth_Q[p]    << ",";
        for (int p = 0; p < N_PATHS; ++p) csv << res.cond_Apre[p] << ",";
        for (int p = 0; p < N_PATHS; ++p) csv << res.cond_G[p]    << ",";
        for (int p = 1; p < N_PATHS; ++p) csv << res.rd_Apre_vs1[p] << ",";
        csv << res.rd_Msk_12 << "," << res.rd_Rsk_12 << "," << res.rd_Apre_12_step << ","
            << res.rd_G_12 << "," << res.rd_Rchol_12 << "," << res.rd_Rfinal_12 << ","
            << res.cond_Rsk << "\n";
    }
    csv.close();

    std::cout << "  Legend:\n";
    for (int p = 0; p < N_PATHS; ++p)
        printf("    [%d] %-18s %s\n", p+1, PATH_NAMES[p], PATH_DESCS[p]);
    std::cout << "\n  CSV written to: " << csv_path << "\n";
}

// ============================================================================
// Main benchmark
// ============================================================================

template <typename T, typename RNG = r123::Philox4x32>
int run_benchmark(int argc, char* argv[]) {
    // argc < 3 (not < 2): output_dir is read from argv[2] unconditionally
    // right below, and argc==2 (e.g. just <prec>) would read past argv's end.
    if (argc < 3) {
        std::cerr << "Usage (file mode):     " << argv[0]
                  << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n"
                  << "Usage (generate mode): " << argv[0]
                  << " <prec> <output_dir> gen <m> <n> <kappa> <density> <d_factor> <runs> [sketch_nnz]\n";
        return 1;
    }

    std::string output_dir = argv[2];
    std::string argv_line = quote_join_argv(argc, argv);
    bool is_generate = (argc >= 4 && std::string(argv[3]) == "gen");

    if (is_generate) {
        // generate mode: prec output_dir gen m n kappa density d_factor runs [sketch_nnz]
        // Required form is argv[0..9] (10 tokens); argc < 10 rejects it. The
        // trailing [sketch_nnz] is argv[10] and requires argc >= 11.
        if (argc < 10) {
            std::cerr << "Usage (generate mode): " << argv[0]
                      << " <prec> <output_dir> gen <m> <n> <kappa> <density> <d_factor> <runs> [sketch_nnz]\n";
            return 1;
        }
        int64_t m         = std::stoll(argv[4]);
        int64_t n         = std::stoll(argv[5]);
        T kappa           = (T)std::stod(argv[6]);
        T density         = (T)std::stod(argv[7]);
        T d_factor        = (T)std::stod(argv[8]);
        int64_t num_runs  = std::stoll(argv[9]);
        int64_t sketch_nnz = (argc >= 11) ? std::stoll(argv[10]) : 4;

        if (num_runs < 1) {
            std::cerr << "Error: runs must be >= 1 (got " << num_runs << ")\n";
            return 1;
        }

        std::cout << "\n=== CQRRT Preconditioner Comparison (generate mode) ===\n";
        std::cout << "  Size:       " << m << " x " << n << "\n";
        std::cout << "  kappa:      " << std::scientific << std::setprecision(3) << (double)kappa << "\n";
        std::cout << "  density:    " << density << "\n";
        std::cout << "  d_factor:   " << d_factor << "\n";
        std::cout << "  sketch_nnz: " << sketch_nnz << "\n";
        std::cout << "  runs:       " << num_runs << "\n";
#ifdef _OPENMP
        std::cout << "  OMP threads: " << omp_get_max_threads() << "\n";
#endif

        RandBLAS::RNGState<RNG> gen_state(0);
        auto A_coo = RandLAPACK::gen::gen_sparse_cond_coo<T>(m, n, kappa, gen_state, density);
        RandBLAS::sparse_data::csr::CSRMatrix<T> A_csr(m, n);
        RandBLAS::sparse_data::conversions::coo_to_csr(A_coo, A_csr);
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, A_csr);

        std::vector<T> A_dense(m * n, 0.0);
        {
            std::vector<T> Eye(n * n, T(0));
            RandLAPACK::util::eye(n, n, Eye.data());
            A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_dense.data(), m);
        }
        T cond_A = RandLAPACK::util::cond_num_check<T>(m, n, A_dense.data(), /*verbose=*/false);
        std::cout << "  cond(A):    " << std::scientific << std::setprecision(3) << (double)cond_A << "\n\n";

        std::string label = "gen_" + std::to_string(m) + "x" + std::to_string(n)
                          + "_kappa" + std::to_string((int)std::round(std::log10((double)kappa)));
        write_csv_and_run<T, decltype(A_linop), RNG>(A_linop, A_dense, m, n, d_factor, sketch_nnz, num_runs,
                                  cond_A, (double)kappa, label, output_dir, argv_line);
    } else {
        // file mode: prec output_dir mtx_path d_factor runs [sketch_nnz]
        if (argc < 6) {
            std::cerr << "Usage (file mode): " << argv[0]
                      << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n";
            return 1;
        }
        std::string mtx_path  = argv[3];
        T d_factor            = (T)std::stod(argv[4]);
        int64_t num_runs      = std::stoll(argv[5]);
        int64_t sketch_nnz    = (argc >= 7) ? std::stoll(argv[6]) : 4;

        if (num_runs < 1) {
            std::cerr << "Error: runs must be >= 1 (got " << num_runs << ")\n";
            return 1;
        }

        int64_t m, n, nnz;
        auto csr = load_csr<T>(mtx_path, m, n, nnz);
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, csr);

        std::vector<T> A_dense(m * n, 0.0);
        {
            std::vector<T> Eye(n * n, T(0));
            RandLAPACK::util::eye(n, n, Eye.data());
            A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_dense.data(), m);
        }
        T cond_A = RandLAPACK::util::cond_num_check<T>(m, n, A_dense.data(), /*verbose=*/false);
        int64_t d = (int64_t)std::ceil(d_factor * n);

        std::cout << "\n=== CQRRT Preconditioner Comparison ===\n";
        std::cout << "  Matrix:     " << mtx_path << "\n";
        std::cout << "  Size:       " << m << " x " << n << "  (nnz=" << nnz << ")\n";
        std::cout << "  d_factor:   " << d_factor << "  (d=" << d << ")\n";
        std::cout << "  sketch_nnz: " << sketch_nnz << "\n";
        std::cout << "  runs:       " << num_runs << "\n";
        std::cout << "  cond(A):    " << std::scientific << std::setprecision(3) << (double)cond_A << "\n";
#ifdef _OPENMP
        std::cout << "  OMP threads: " << omp_get_max_threads() << "\n";
#endif
        std::cout << "\n";

        write_csv_and_run<T, decltype(A_linop), RNG>(A_linop, A_dense, m, n, d_factor, sketch_nnz, num_runs,
                                  cond_A, -1.0, mtx_path, output_dir, argv_line);
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage (file mode):     " << argv[0]
                  << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n"
                  << "Usage (generate mode): " << argv[0]
                  << " <prec> <output_dir> gen <m> <n> <kappa> <density> <d_factor> <runs> [sketch_nnz]\n";
        return 1;
    }
    std::string prec = argv[1];
    if (prec == "double") return run_benchmark<double>(argc, argv);
    if (prec == "float")  return run_benchmark<float>(argc, argv);
    std::cerr << "Unknown precision '" << prec << "' (use double or float)\n";
    return 1;
}
