// CQRRT preconditioner comparison benchmark
//
// Isolates the effect of different methods for forming R_sk^{-1} on the final
// orthogonality quality of CQRRT. Tests four paths:
//
//   [1] expl_trsm:      DTRSM_R(A, R_sk) in-place                   ← CQRRT_expl path
//   [2] expl_inv_trsm:  TRSM(I, R_sk) → R_inv; DGEMM(A, R_inv)       (TRSM on identity)
//   [3] expl_inv_trtri: TRTRI(R_sk) → R_inv;   DGEMM(A, R_inv)       (LAPACK trtri)
//   [4] expl_inv_geqp3: GEQP3(R_sk) = Q*R_buf*P^T;
//                       R_inv = P * TRSM(R_buf, Q^T); DGEMM(A, R_inv)
//   [5] expl_inv_svd:   GESDD(R_sk) = U*S*Vt;
//                       R_inv = V * diag(1/S) * U^T; DGEMM(A, R_inv)
//   [6] expl_inv_getri: GETRF(R_sk) then GETRI; DGEMM(A, R_inv)
//
//   Path [1] never forms R_sk^{-1} explicitly (backward stable).
//   Paths [2]-[6] all form R_sk^{-1} explicitly via different methods.
//   Path [4] uses a rank-revealing QR to invert R_sk; the Q factor makes
//   the inversion well-conditioned even when R_sk itself is ill-conditioned.
//   Path [5] uses the SVD (gold standard for stability).
//   Path [6] uses general LU with partial pivoting; for upper-triangular R_sk
//   this is expected to behave similarly to paths [2]-[3].
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
//     rd_15 = ||A_pre[1] - A_pre[5]|| / ||A_pre[1]||
//     rd_16 = ||A_pre[1] - A_pre[6]|| / ||A_pre[1]||
//
//   Step-by-step pipeline divergence between paths [1] and [2] (CQRRT_expl vs CQRRT_linop):
//   Each path uses the same RNG seed but a different sketch code path — mirrors actual impls.
//   Path [1] (CQRRT_expl):   sketch via sketch_general(S, A_dense); TRSM in-place; SYRK
//   Path [2] (CQRRT_linop):  sketch via A_linop(Side::Right, S) [SpGEMM];
//                             TRSM_IDENTITY → R_inv; A_linop fwd/adj + TRMM
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

#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"
#include "rl_cqrrt_linops.hh"

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
#include "cqrrt_bench_common.hh"

using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using blas::Layout;
using blas::Op;
using blas::Side;
using blas::Uplo;
using blas::Diag;

// ============================================================================
// Path constants
// ============================================================================

static constexpr int N_PATHS = 6;

static constexpr const char* PATH_NAMES[N_PATHS] = {
    "expl_trsm",
    "expl_inv_trsm",
    "expl_inv_trtri",
    "expl_inv_geqp3",
    "expl_inv_svd",
    "expl_inv_getri",
};

static constexpr const char* PATH_DESCS[N_PATHS] = {
    "DTRSM_R(A, R_sk) in-place                              <- CQRRT_expl",
    "TRSM(I, R_sk)->R_inv; DGEMM(A, R_inv)",
    "TRTRI(R_sk)->R_inv;   DGEMM(A, R_inv)",
    "GEQP3(R_sk)=Q*R_buf*P^T; R_inv=P*TRSM(R_buf,Q^T); DGEMM(A, R_inv)",
    "GESDD(R_sk)=U*S*Vt; R_inv=V*diag(1/S)*U^T; DGEMM(A, R_inv)",
    "GETRF+GETRI(R_sk)->R_inv; DGEMM(A, R_inv)",
};

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

// Full CQRRT pipeline (matching CQRRT_linops Gram computation):
//   G = A_orig^T * A_pre         (GEMM — full n×n, not exploiting symmetry)
//   G = (R_sketch)^{-T} * G     (TRSM Left — backward-stable left factor on original R^sk)
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
    return orth_error(Q.data(), m, n);
}

// ============================================================================
// One trial: 4-path orth comparison (shared sketch) +
//            independent path [1] vs [2] step-by-step divergence
// ============================================================================

template <typename T>
struct TrialResult {
    // Per-path metrics (shared sketch, 4 paths)
    double cond_Apre[N_PATHS];
    double cond_G[N_PATHS];
    double orth_Q[N_PATHS];
    // Cross-path relative differences of A_pre (reference = path [1], shared sketch)
    double rd_Apre_12;  // TRSM in-place vs TRSM-on-identity
    double rd_Apre_13;  // TRSM in-place vs trtri
    double rd_Apre_14;  // TRSM in-place vs geqp3
    double rd_Apre_15;  // TRSM in-place vs svd
    double rd_Apre_16;  // TRSM in-place vs getri
    // Step-by-step pipeline divergence: paths [1] vs [2], faithful to actual implementations
    //   Path [1] (CQRRT_expl):  sketch via sketch_general(S, A_dense); TRSM in-place; SYRK
    //   Path [2] (CQRRT_linop): sketch via A_linop(Side::Right, S) [SpGEMM];
    //                            TRSM_IDENTITY → R_inv; A_linop fwd/adj + TRMM
    double rd_Msk_12;       // M^sk:  raw sketch Ahat = S*A (different code paths)
    double rd_Rsk_12;       // R^sk:  QR factor of the above
    double rd_Apre_12_step; // MR^pre: TRSM in-place vs A_linop(fwd, R_inv)
    double rd_G_12;         // Gram:  SYRK(A_pre) vs A_linop(adj, A_pre)+TRMM
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
    int64_t d = (int64_t)std::ceil(d_factor * n);

    // Save RNG state before any sketching; Part B (step-by-step) uses this
    // to compute independent sketches for paths [1] and [2].
    auto initial_state = state;

    // ----------------------------------------------------------------
    // Part A: Shared sketch → R_sk, 4-path orth comparison
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
    auto R_inv_trsm = make_eye<T>(n);
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

    // Method D: SVD of R_sk  (path [5])
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

    // Method E: GETRF + GETRI (general LU with partial pivoting)  (path [6])
    std::vector<T> R_inv_getri(R_sk.begin(), R_sk.end());
    {
        std::vector<int64_t> ipiv(n);
        lapack::getrf(n, n, R_inv_getri.data(), n, ipiv.data());
        lapack::getri(n, R_inv_getri.data(), n, ipiv.data());
    }

    // ----------------------------------------------------------------
    // Compute all N_PATHS preconditioned matrices:
    //   Apre[0]: TRSM in-place   (path [1], CQRRT_expl)
    //   Apre[1]: GEMM + R_inv_trsm  (path [2])
    //   Apre[2]: GEMM + R_inv_trtri (path [3])
    //   Apre[3]: GEMM + R_inv_geqp3 (path [4])
    // ----------------------------------------------------------------
    std::array<std::vector<T>, N_PATHS> Apre;
    for (auto& a : Apre) a.resize(m * n, T(0));

    Apre[0].assign(A_dense, A_dense + m*n);
    blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
               Diag::NonUnit, m, n, (T)1.0, R_sk.data(), n, Apre[0].data(), m);

    const T* R_invs[N_PATHS - 1] = {
        R_inv_trsm.data(), R_inv_trtri.data(), R_inv_geqp3.data(),
        R_inv_svd.data(), R_inv_getri.data()
    };
    for (int p = 1; p < N_PATHS; ++p)
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                   m, n, n, (T)1.0, A_dense, m, R_invs[p-1], n, (T)0.0, Apre[p].data(), m);

    // ----------------------------------------------------------------
    // Cross-path relative differences (reference = path [1] = Apre[0])
    // ----------------------------------------------------------------
    res.rd_Apre_12 = (double)rel_diff(Apre[0].data(), Apre[1].data(), m*n);
    res.rd_Apre_13 = (double)rel_diff(Apre[0].data(), Apre[2].data(), m*n);
    res.rd_Apre_14 = (double)rel_diff(Apre[0].data(), Apre[3].data(), m*n);
    res.rd_Apre_15 = (double)rel_diff(Apre[0].data(), Apre[4].data(), m*n);
    res.rd_Apre_16 = (double)rel_diff(Apre[0].data(), Apre[5].data(), m*n);

    // ----------------------------------------------------------------
    // Part B: Independent step-by-step divergence, paths [1] vs [2]
    //
    // Both paths compute their own sketch and R_sk from initial_state
    // (same seed → same result, but as separate objects).
    //
    // Path [1] (CQRRT_expl): TRSM in-place on A; Gram via SYRK.
    // Path [2] (CQRRT_linop, block_size=0):
    //   R_inv = TRSM_IDENTITY(R_sk);
    //   A_pre = GEMM(A, R_inv)         [fwd linop call]
    //   G     = GEMM(A^T, A_pre)        [adj linop call]
    //   G     = TRMM(R_inv^T, G)        [complete Gram: R_inv^T * A^T * A * R_inv]
    // ----------------------------------------------------------------

    // Run Part B as a lambda so early returns on Cholesky failure are clean.
    [&]() {
        // ---- Step 1: Sketch ----
        // Path [1] (CQRRT_expl):  sketch_general(S, A_dense) — left SPMM on dense copy
        RandBLAS::SparseDist Ds_1(d, m, sketch_nnz, RandBLAS::Axis::Short);
        RandBLAS::SparseSkOp<T> S_1(Ds_1, initial_state);
        std::vector<T> Ahat_1(d * n, 0.0);
        RandBLAS::sketch_general(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                                  d, n, m, (T)1.0, S_1, A_dense, m,
                                  (T)0.0, Ahat_1.data(), d);

        // Path [2] (CQRRT_linop): A_linop(Side::Right, S) — SpGEMM on sparse CSR matrix
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

        // ---- Path [1]: CQRRT_expl — TRSM in-place, SYRK ----
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

        // ---- Path [2]: CQRRT_linop — TRSM_IDENTITY, linop fwd/adj, TRMM ----
        // R_inv via TRSM_IDENTITY: solve X * R_sk_2 = I  (upper triangular result)
        auto R_inv_2 = make_eye<T>(n);
        blas::trsm(Layout::ColMajor, Side::Right, Uplo::Upper, Op::NoTrans,
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
        res.cond_Apre[p] = (double)condition_number(Apre[p].data(), m, n);
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
    const std::string& output_dir)
{
    char time_buf[64];
    time_t now = time(nullptr);
    strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", localtime(&now));
    std::string csv_path = output_dir + "/diagnostic_" + time_buf + ".csv";
    std::ofstream csv(csv_path);

    csv << "# CQRRT Preconditioner Comparison\n";
    csv << "# Date: " << ctime(&now);
    csv << "# Matrix: " << matrix_label << "\n";
    csv << "# m=" << m << " n=" << n << " d_factor=" << d_factor
        << " sketch_nnz=" << sketch_nnz << "\n";
    csv << "# cond_A=" << std::scientific << std::setprecision(6) << cond_A;
    if (kappa_target > 0)
        csv << " kappa_target=" << std::scientific << std::setprecision(6) << kappa_target;
    csv << "\n";
    csv << "run,"
        << "orth_Q1,orth_Q2,orth_Q3,orth_Q4,orth_Q5,orth_Q6,"
        << "cond_Apre1,cond_Apre2,cond_Apre3,cond_Apre4,cond_Apre5,cond_Apre6,"
        << "cond_G1,cond_G2,cond_G3,cond_G4,cond_G5,cond_G6,"
        << "rd_Apre_12,rd_Apre_13,rd_Apre_14,rd_Apre_15,rd_Apre_16,"
        << "rd_Msk_12,rd_Rsk_12,rd_Apre_12_step,rd_G_12,rd_Rchol_12,rd_Rfinal_12,"
        << "cond_Rsk\n";

    RandBLAS::RNGState<RNG> base_state(42);
    for (int64_t r = 0; r < num_runs; ++r) {
        auto state = base_state;
        if (r > 0) state.key.incr(r);

        auto res = run_trial<T, RNG>(A_linop, A_dense.data(), m, n, d_factor, sketch_nnz, state);

        printf("  run %ld  orth_error(Q = A * R_final^{-1}):\n", r);
        for (int p = 0; p < N_PATHS; ++p)
            printf("    [%d] %-18s %12.3e\n", p+1, PATH_NAMES[p], res.orth_Q[p]);

        printf("  run %ld  cond(MR^pre):\n", r);
        for (int p = 0; p < N_PATHS; ++p)
            printf("    [%d] %-18s %12.3e\n", p+1, PATH_NAMES[p], res.cond_Apre[p]);

        printf("  run %ld  cond(G = MR^pre^T MR^pre):\n", r);
        for (int p = 0; p < N_PATHS; ++p)
            printf("    [%d] %-18s %12.3e\n", p+1, PATH_NAMES[p], res.cond_G[p]);

        printf("  run %ld  rel_diff(MR^pre) vs [1]:\n", r);
        printf("    rd_12 (trsm-on-I):   %12.3e\n", res.rd_Apre_12);
        printf("    rd_13 (trtri):       %12.3e\n", res.rd_Apre_13);
        printf("    rd_14 (geqp3):       %12.3e\n", res.rd_Apre_14);
        printf("    rd_15 (svd):         %12.3e\n", res.rd_Apre_15);
        printf("    rd_16 (getri):       %12.3e\n", res.rd_Apre_16);

        printf("  run %ld  step-by-step divergence [1] vs [2] (expl: sketch_general; linop: SpGEMM):\n", r);
        printf("    M^sk:    %12.3e\n", res.rd_Msk_12);
        printf("    R^sk:    %12.3e\n", res.rd_Rsk_12);
        printf("    MR^pre:  %12.3e\n", res.rd_Apre_12_step);
        printf("    G:       %12.3e\n", res.rd_G_12);
        printf("    R^chol:  %12.3e\n", res.rd_Rchol_12);
        printf("    R:       %12.3e\n", res.rd_Rfinal_12);

        printf("  run %ld  cond(R_sk): %9.3e\n\n", r, res.cond_Rsk);

        csv << r << "," << std::scientific << std::setprecision(6);
        for (int p = 0; p < N_PATHS; ++p) csv << res.orth_Q[p]    << ",";
        for (int p = 0; p < N_PATHS; ++p) csv << res.cond_Apre[p] << ",";
        for (int p = 0; p < N_PATHS; ++p) csv << res.cond_G[p]    << ",";
        csv << res.rd_Apre_12 << "," << res.rd_Apre_13 << "," << res.rd_Apre_14 << ","
            << res.rd_Apre_15 << "," << res.rd_Apre_16 << ","
            << res.rd_Msk_12 << "," << res.rd_Rsk_12 << "," << res.rd_Apre_12_step << ","
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
    if (argc < 2) {
        std::cerr << "Usage (file mode):     " << argv[0]
                  << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n"
                  << "Usage (generate mode): " << argv[0]
                  << " <prec> <output_dir> gen <m> <n> <kappa> <density> <d_factor> <runs> [sketch_nnz]\n";
        return 1;
    }

    std::string output_dir = argv[2];
    bool is_generate = (argc >= 4 && std::string(argv[3]) == "gen");

    if (is_generate) {
        // generate mode: prec output_dir gen m n kappa density d_factor runs [sketch_nnz]
        if (argc < 11) {
            std::cerr << "Usage (generate mode): " << argv[0]
                      << " <prec> <output_dir> gen <m> <n> <kappa> <density> <d_factor> <runs> [sketch_nnz]\n";
            return 1;
        }
        int64_t m         = std::stol(argv[4]);
        int64_t n         = std::stol(argv[5]);
        T kappa           = (T)std::stod(argv[6]);
        T density         = (T)std::stod(argv[7]);
        T d_factor        = (T)std::stod(argv[8]);
        int64_t num_runs  = std::stol(argv[9]);
        int64_t sketch_nnz = (argc >= 11) ? std::stol(argv[10]) : 4;

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
            auto Eye = make_eye<T>(n);
            A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_dense.data(), m);
        }
        T cond_A = condition_number(A_dense.data(), m, n);
        std::cout << "  cond(A):    " << std::scientific << std::setprecision(3) << (double)cond_A << "\n\n";

        std::string label = "gen_" + std::to_string(m) + "x" + std::to_string(n)
                          + "_kappa" + std::to_string((int)std::round(std::log10((double)kappa)));
        write_csv_and_run<T, decltype(A_linop), RNG>(A_linop, A_dense, m, n, d_factor, sketch_nnz, num_runs,
                                  cond_A, (double)kappa, label, output_dir);
    } else {
        // file mode: prec output_dir mtx_path d_factor runs [sketch_nnz]
        if (argc < 6) {
            std::cerr << "Usage (file mode): " << argv[0]
                      << " <prec> <output_dir> <mtx_path> <d_factor> <runs> [sketch_nnz]\n";
            return 1;
        }
        std::string mtx_path  = argv[3];
        T d_factor            = (T)std::stod(argv[4]);
        int64_t num_runs      = std::stol(argv[5]);
        int64_t sketch_nnz    = (argc >= 7) ? std::stol(argv[6]) : 4;

        int64_t m, n, nnz;
        auto csr = load_csr<T>(mtx_path, m, n, nnz);
        RandLAPACK::linops::SparseLinOp<RandBLAS::sparse_data::csr::CSRMatrix<T>> A_linop(m, n, csr);

        std::vector<T> A_dense(m * n, 0.0);
        {
            auto Eye = make_eye<T>(n);
            A_linop(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                    m, n, n, (T)1.0, Eye.data(), n, (T)0.0, A_dense.data(), m);
        }
        T cond_A = condition_number(A_dense.data(), m, n);
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
                                  cond_A, -1.0, mtx_path, output_dir);
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
