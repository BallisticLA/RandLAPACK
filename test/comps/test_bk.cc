#include "RandLAPACK.hh"
#include "rl_blaspp.hh"
#include "rl_lapackpp.hh"
#include "rl_gen.hh"

#include <RandBLAS.hh>
#include <gtest/gtest.h>

// Direct tests for BK (RandLAPACK/comps/rl_bk.hh), the block Krylov component underneath
// the ABRIK driver. Before this file BK had no direct coverage at all: it was exercised
// only through ABRIK, which hides end_rows/end_cols, the band buffers, and BKTermination.
// Everything that made the 2026-07-29 rank-deficiency attempt hard to debug lives at this
// level, so this is where the structural invariants belong.

class TestBK : public ::testing::Test
{
    protected:

    virtual void SetUp() {};
    virtual void TearDown() {};

    // Owns the four buffers BK returns. BK allocates them with calloc and documents that
    // the caller must free() them, so a RAII holder keeps the asan-enabled Debug CI job
    // clean even when an assertion aborts a test mid-way.
    template <typename T>
    struct BKOut {
        T* X_ev = nullptr;
        T* Y_od = nullptr;
        T* R    = nullptr;
        T* S    = nullptr;
        int64_t end_rows = 0;
        int64_t end_cols = 0;
        bool final_iter_is_odd = false;

        ~BKOut() { free(X_ev); free(Y_od); free(R); free(S); }
    };

    /// The band identity: band == X_ev(:,1:end_rows)' * A * Y_od(:,1:end_cols).
    ///
    /// This is the strongest and cheapest assertion available at the BK level. Two GEMMs on
    /// a small matrix, and it simultaneously catches a transpose slip, a permutation that
    /// was applied to a basis but not folded into the band, replacement columns missing
    /// from the band, and truncated columns left unaccounted for.
    ///
    /// It also settled a documentation ambiguity, since corrected. rl_bk.hh used to call R
    /// an "Upper band matrix (stored transposed)" while ABRIK hands the stored buffer
    /// straight to gesdd, so the comments did not pin down whether the stored orientation
    /// or its transpose was the true band. Both orientations are measured and
    /// reported; the assertion accepts whichever the code actually means, and the printed
    /// pair records which one that is.
    template <typename T>
    static void check_band_identity(
        int64_t m, int64_t n, int64_t k, const T* A, BKOut<T> &out, T rtol
    ) {
        const int64_t er = out.end_rows;
        const int64_t ec = out.end_cols;
        ASSERT_GT(er, 0);
        ASSERT_GT(ec, 0);

        // P = X_ev(:,1:er)' * A * Y_od(:,1:ec), formed as (A*Y) then X'(AY).
        T* AY = new T[m * ec]();
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, ec, n,
                   (T)1.0, A, m, out.Y_od, n, (T)0.0, AY, m);
        T* P = new T[er * ec]();
        blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, er, ec, m,
                   (T)1.0, out.X_ev, m, AY, m, (T)0.0, P, er);
        T norm_P = lapack::lange(Norm::Fro, er, ec, P, er);

        // Band sparsity. Measured first with a block-norm map, then asserted; the map showed
        // a clean lower block-bidiagonal support at k = 10, er = 110, ec = 100:
        //
        //   R (odd terminal, ld n)            S (even terminal, ld n+k)
        //   diagonal blocks   2.2e-01           diagonal blocks   3.8e-01 .. 2.2e-01
        //   subdiagonal       1.3e-01           subdiagonal       1.7e-01 .. 6.7e-02
        //   ABOVE  diagonal   EXACTLY 0         ABOVE  diagonal   ~5e-17, computed
        //   BELOW  subdiag    ~3.5e-17          BELOW  subdiag    EXACTLY 0
        //
        // The two are structural mirrors, and that asymmetry is the point: the odd branch
        // fills a ROW strip of R (rows of the current X block, all previous Y columns) while
        // the even branch fills a COLUMN strip of S. So each band has one side that is merely
        // at roundoff and one side that was never touched at all, and they are opposite sides.
        // Asserting a single rule for both is wrong; the first version of this check did that
        // and failed on R at entry (20,0), which is below R's subdiagonal and therefore
        // computed rather than exact.
        //
        // The two assertions below are stated in ENTRY indices rather than block indices, so
        // they stay valid once the rank criterion narrows a block: widths only ever shrink, so
        // the true support only tightens and these bounds stay conservative.
        //
        // This catches what the identity alone cannot. A band that is dense but happens to
        // equal X'AY would pass the identity; loss of the block-bidiagonal structure through
        // reorthogonalisation drift is exactly the failure this sees and the identity does not.
        {
            const T* bnd = out.final_iter_is_odd ? out.R : out.S;
            const int64_t ld = out.final_iter_is_odd ? n : (n + k);
            T nrm = 0;
            for (int64_t j = 0; j < ec; ++j)
                for (int64_t i = 0; i < er; ++i) nrm += bnd[i + j*ld] * bnd[i + j*ld];
            nrm = std::sqrt(nrm);

            for (int64_t j = 0; j < ec; ++j) {
                for (int64_t i = 0; i < er; ++i) {
                    const T v = bnd[i + j * ld];
                    if (j >= i + k) {
                        // Strictly above the diagonal band.
                        if (out.final_iter_is_odd)
                            ASSERT_EQ(v, T(0)) << "R entry (" << i << "," << j
                                << ") is above the band and is never written";
                        else
                            ASSERT_LE(std::abs(v), rtol * nrm) << "S entry (" << i << "," << j
                                << ") is above the band and should be at roundoff";
                    } else if (i >= j + 2 * k) {
                        // Strictly below the subdiagonal band.
                        if (out.final_iter_is_odd)
                            ASSERT_LE(std::abs(v), rtol * nrm) << "R entry (" << i << "," << j
                                << ") is below the band and should be at roundoff";
                        else
                            ASSERT_EQ(v, T(0)) << "S entry (" << i << "," << j
                                << ") is below the band and is never written";
                    }
                }
            }
        }

        // S's leading dimension is n + k, always. It must NOT be derived as n + (er - ec):
        // that difference is k only when the terminal block is full width, and it is the
        // last accepted width once the rank criterion truncates, which reads the buffer at
        // the wrong stride and reports a spurious band-identity failure.
        const T* band = out.final_iter_is_odd ? out.R : out.S;
        const int64_t ldb = out.final_iter_is_odd ? n : (n + k);

        // Orientation 1: band(i,j) as stored.
        T* D1 = new T[er * ec]();
        for (int64_t j = 0; j < ec; ++j)
            for (int64_t i = 0; i < er; ++i)
                D1[i + j * er] = P[i + j * er] - band[i + j * ldb];
        T e1 = lapack::lange(Norm::Fro, er, ec, D1, er);

        // Orientation 2: the transpose of the stored buffer, only meaningful when square.
        T e2 = std::numeric_limits<T>::infinity();
        if (er == ec) {
            T* D2 = new T[er * ec]();
            for (int64_t j = 0; j < ec; ++j)
                for (int64_t i = 0; i < er; ++i)
                    D2[i + j * er] = P[i + j * er] - band[j + i * ldb];
            e2 = lapack::lange(Norm::Fro, er, ec, D2, er);
            delete[] D2;
        }

        printf("BAND er=%ld ec=%ld odd=%d ||P||=%.3e  as-stored=%.3e  transposed=%.3e\n",
               (long)er, (long)ec, (int)out.final_iter_is_odd, (double)norm_P,
               (double)(e1 / norm_P), (double)(e2 / norm_P));
        fflush(stdout);

        delete[] D1;
        delete[] P;
        delete[] AY;

        T best = std::min(e1, e2) / norm_P;
        ASSERT_LE(best, rtol) << "neither orientation of the band reproduces X' A Y";
    }

    /// Orthonormality of a basis, ||Q'Q - I||_F / sqrt(cols).
    template <typename T>
    static T orth_err(const T* Q, int64_t rows, int64_t cols) {
        return RandLAPACK::testing::orthogonality_error<T>(Q, rows, cols);
    }
    /// A = U diag(s) V' with Haar-like factors, drawn from a fresh generator state so that
    /// every test asking for the same spectrum gets the same matrix. The pattern of
    /// build_from_spectrum in test_abrik.cc; trailing singular values are exactly zero.
    static void build_from_spectrum(int64_t m, int64_t n, std::vector<double> s, double* A) {
        const int64_t r = (int64_t) s.size();
        std::vector<double> S(r * r, 0.0);
        RandLAPACK::util::diag(r, r, s.data(), r, S.data());
        auto gs = RandBLAS::RNGState();
        RandLAPACK::gen::gen_singvec<double>(m, n, A, r, S.data(), gs);
    }

    /// r singular values decaying over three decades: the exact-rank matrices of this file.
    static std::vector<double> decaying_spectrum(int64_t r) {
        std::vector<double> s(r);
        for (int i = 0; i < r; ++i) s[i] = std::pow(10.0, -3.0 * i / (r - 1));
        return s;
    }

    /// What the checkpoint leg of check_resume_equals_single_shot saw: the refill state at the
    /// checkpoint, and refills_exhausted once resume() has run.
    struct ResumeCheckpoint {
        int64_t refilled_blocks       = 0;
        int64_t saved_pending_refills = 0;
        bool    refills_exhausted     = false;
    };

    /// call(p) must equal call(p1) followed by resume(p), bitwise. Shared by the full-width case
    /// and the narrowed case; `expect_narrowed` demands that the checkpoint leg actually crossed
    /// a narrowing, so the narrowed variant cannot silently degrade into a second copy of the
    /// full-width one. A non-null `ckpt` receives the checkpoint's refill state.
    static void check_resume_equals_single_shot(
        int64_t m, int64_t n, int64_t k, int p1, int p, const double* A_src, bool expect_narrowed,
        ResumeCheckpoint* ckpt = nullptr
    ) {
        double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
        std::vector<double> A(A_src, A_src + m * n);
        RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A.data(), m, Layout::ColMajor);

        BKOut<double> one;
        {
            auto state = RandBLAS::RNGState();
            RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
            bk.max_krylov_iters = p;
            ASSERT_EQ(bk.call(A_op, k, one.X_ev, one.Y_od, one.R, one.S,
                              one.end_rows, one.end_cols, one.final_iter_is_odd, state), 0);
        }

        BKOut<double> two;
        {
            auto state = RandBLAS::RNGState();
            RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
            bk.max_krylov_iters = p1;
            ASSERT_EQ(bk.call(A_op, k, two.X_ev, two.Y_od, two.R, two.S,
                              two.end_rows, two.end_cols, two.final_iter_is_odd, state), 0);
            ASSERT_EQ(bk.termination_reason, RandLAPACK::BKTermination::max_iters_reached)
                << "resume is only defined after max_iters_reached; the premise of this test";
            if (expect_narrowed) {
                ASSERT_GE(bk.narrowed_blocks, (int64_t) 1)
                    << "this variant exists to resume ACROSS a narrowing, but none occurred "
                       "before the checkpoint; the configuration no longer tests what it claims";
            }
            if (ckpt) {
                ckpt->refilled_blocks       = bk.refilled_blocks;
                ckpt->saved_pending_refills = bk.saved_pending_refills;
            }
            bk.max_krylov_iters = p;
            ASSERT_EQ(bk.resume(A_op, k, two.X_ev, two.Y_od, two.R, two.S,
                                two.end_rows, two.end_cols, two.final_iter_is_odd, state), 0);
            if (ckpt)
                ckpt->refills_exhausted = bk.refills_exhausted;
        }

        printf("RESUME%s single(%d): rows=%ld cols=%ld | %d then resume(%d): rows=%ld cols=%ld\n",
               expect_narrowed ? "-NARROWED" : "",
               p, (long)one.end_rows, (long)one.end_cols,
               p1, p, (long)two.end_rows, (long)two.end_cols);
        fflush(stdout);

        ASSERT_EQ(one.end_rows, two.end_rows);
        ASSERT_EQ(one.end_cols, two.end_cols);
        ASSERT_EQ(one.final_iter_is_odd, two.final_iter_is_odd);
        for (int64_t i = 0; i < m * one.end_rows; ++i)
            ASSERT_EQ(one.X_ev[i], two.X_ev[i]) << "X_ev differs at " << i;
        for (int64_t i = 0; i < n * one.end_cols; ++i)
            ASSERT_EQ(one.Y_od[i], two.Y_od[i]) << "Y_od differs at " << i;
    }

};


TEST_F(TestBK, BK_norm_convergence_at_iteration_budget_is_terminal) {
    const int64_t m = 8, n = 6, k = 2;
    std::vector<double> A(m * n, 0.0);
    A[0] = 4.0;
    A[1 + m] = 2.0;

    // The first odd step captures this rank-two matrix. A loose norm tolerance
    // makes that terminal decision independent of rounding in the factorization.
    for (int budget : {1, 3}) {
        SCOPED_TRACE(budget);
        auto state = RandBLAS::RNGState();
        BKOut<double> out;
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, 0.25);
        bk.max_krylov_iters = budget;
        ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                          out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
        EXPECT_EQ(bk.termination_reason, RandLAPACK::BKTermination::norm_converged);
        EXPECT_EQ(bk.num_krylov_iters, 1);
        EXPECT_TRUE(out.final_iter_is_odd);
        EXPECT_EQ(out.end_rows, k);
        EXPECT_EQ(out.end_cols, k);
        check_band_identity<double>(m, n, k, A.data(), out, 1e-12);
    }
}


// Does the band actually equal X' A Y, at the point where norm_converged now stops?
//
// Phase 0.2 restored norm_converged (rl_bk.hh:716 was measuring the wrong triangle) and
// fixed a latent miscount in that exit. With both fixed, five ABRIK tests stop one
// iteration earlier -- at correctly detected full saturation -- and their residuals move
// from ~1e-13 to ~1e-8. Two explanations were open: the basis has lost orthogonality across
// a wide block Krylov run (a numerical fact, in which case the old tolerances were only
// achievable because the criterion was dead), or the R extraction path is itself wrong (a
// bug). This test discriminates: if the band reconstructs but the basis is not orthonormal,
// it is the former.
// Exactly the ABRIK_basic configuration (m=400, n=200, b_sz=10, budget 40), which is the
// one that regressed from 7.8e-13 to 2.7e-08. Reproduced at BK level so the band, the two
// bases and the termination state are all directly visible.
TEST_F(TestBK, BK_band_equals_XtAY_abrik_basic_config) {
    int64_t m = 400;
    int64_t n = 200;
    int64_t k = 10;
    double  tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    double* A = new double[m * n]();
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, A, state);

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = 40;   // what test_ABRIK_general derives: (target_rank*2)/b_sz

    int status = bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,
                         out.end_rows, out.end_cols, out.final_iter_is_odd, state);
    ASSERT_EQ(status, 0);

    printf("BK iters=%d reason=%d end_rows=%ld end_cols=%ld odd=%d\n",
           bk.num_krylov_iters, (int)bk.termination_reason,
           (long)out.end_rows, (long)out.end_cols, (int)out.final_iter_is_odd);

    double oX = orth_err<double>(out.X_ev, m, out.end_rows);
    double oY = orth_err<double>(out.Y_od, n, out.end_cols);
    printf("BASIS orth: ||X'X-I||/sqrt=%.3e  ||Y'Y-I||/sqrt=%.3e\n", oX, oY);
    printf("BASIS max orthonormal prefix: X=%ld of %ld, Y=%ld of %ld\n",
           (long)RandLAPACK::testing::max_orthonormal_cols<double>(out.X_ev, m, out.end_rows),
           (long)out.end_rows,
           (long)RandLAPACK::testing::max_orthonormal_cols<double>(out.Y_od, n, out.end_cols),
           (long)out.end_cols);
    fflush(stdout);

    check_band_identity<double>(m, n, k, A, out, 1e-10);

    // BK's output is exact here (band identity 7e-16, both bases orthonormal). If ABRIK's
    // residual is nonetheless ~1e-8, the loss is downstream. Reproduce ABRIK's own
    // reconstruction (rl_abrik.hh:340-353) on this band and measure it directly, which
    // localizes the error to either this arithmetic or something else in the driver.
    {
        const int64_t er = out.end_rows, ec = out.end_cols;
        double* band_cpy = new double[er * ec]();
        lapack::lacpy(MatrixType::General, er, ec, out.R, n, band_cpy, er);

        double* Sigma  = new double[std::min(er, ec)]();
        double* U_hat  = new double[er * ec]();
        double* VT_hat = new double[ec * ec]();
        lapack::gesdd(Job::SomeVec, er, ec, band_cpy, er, Sigma, U_hat, er, VT_hat, ec);

        double* U = new double[m * ec]();
        double* V = new double[n * ec]();
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, m, ec, er,
                   1.0, out.X_ev, m, U_hat, er, 0.0, U, m);
        blas::gemm(Layout::ColMajor, Op::NoTrans, Op::Trans, n, ec, ec,
                   1.0, out.Y_od, n, VT_hat, ec, 0.0, V, n);

        RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A, m, Layout::ColMajor);
        int64_t lead = std::min<int64_t>(100, ec);
        double res_lead = RandLAPACK::linops::svd_residual<double>(A_op, U, V, Sigma, lead);
        int64_t cert = RandLAPACK::linops::svd_triplets_certified<double>(
            A_op, U, V, Sigma, ec, 1e-8);
        printf("RECONSTRUCT lead-%ld normalized residual=%.3e   certified=%ld of %ld\n",
               (long)lead, res_lead, (long)cert, (long)ec);
        fflush(stdout);

        delete[] band_cpy; delete[] Sigma; delete[] U_hat;
        delete[] VT_hat;   delete[] U;     delete[] V;
    }

    delete[] A;
}

// ---------------------------------------------------------------------------------------
// Liveness. BK must terminate, and stay inside its buffers, with max_krylov_iters left at
// its default.
//
// That default is INT_MAX (rl_bk.hh:67), and EVERY pre-existing test overrides it, so the
// default path had no coverage at all. In that mode `iter >= max_iters` never fires, which
// leaves exactly two exits. One of them, norm_converged, was broken until Phase 0.2. So
// rank_deficient was carrying termination single-handedly -- and that is the exit the
// rank-deficiency work exists to change. A relative threshold tau*||A|| is identically zero
// for a zero matrix, so the natural fix would hang on one; these tests are what stops that
// reaching main.
//
// The band-bounds assertions matter for a reason a sanitizer cannot help with: past
// saturation the band writes land in the NEXT allocated column rather than off the end of
// the allocation, so they corrupt silently.
// ---------------------------------------------------------------------------------------

// Shared body: run BK at the default budget and assert it stops inside its bounds.
#define BK_LIVENESS_BODY(NAME, MK_MATRIX)                                                  \
TEST_F(TestBK, NAME) {                                                                     \
    int64_t m = 120, n = 60, k = 10;                                                       \
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);                   \
    auto state = RandBLAS::RNGState();                                                     \
    double* A = new double[m * n]();                                                       \
    MK_MATRIX                                                                              \
    BKOut<double> out;                                                                     \
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);                        \
    /* max_krylov_iters deliberately left at its INT_MAX default. */                       \
    int status = bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,                  \
                         out.end_rows, out.end_cols, out.final_iter_is_odd, state);        \
    printf("LIVENESS %-28s status=%d iters=%d reason=%d end_rows=%ld end_cols=%ld\n",      \
           #NAME, status, bk.num_krylov_iters, (int)bk.termination_reason,                 \
           (long)out.end_rows, (long)out.end_cols);                                        \
    fflush(stdout);                                                                        \
    ASSERT_EQ(status, 0);                                                                  \
    EXPECT_LE(out.end_cols, n);                                                            \
    EXPECT_LE(out.end_rows, n + k);                                                        \
    EXPECT_LE(out.end_cols + k, n + k);                                                    \
    EXPECT_LE(bk.num_krylov_iters, 2 * ((n + k - 1) / k) + 2);                             \
    delete[] A;                                                                            \
}

// A zero matrix: norm_A == 0, so the relative threshold tau*||A|| is zero, and a block of exact
// zeros still meets it. Iteration 1 is dead and refilled; iteration 2 finds the refills' images
// dead, retracts them and stops rank_deficient. Measured 2 iterations (1 without refilling).
BK_LIVENESS_BODY(BK_terminates_on_zero_matrix, /* A stays all zeros */)

// The identity, padded to m x n: the Krylov space is span(Omega) and closes after one block.
// Refilling reopens it and the run grows until the saturation guard fires. The X-side refills
// are Gaussian in R^m, so they also carry components outside the range of A, which is how
// end_rows reaches n + k. Measured 12 iterations, end_rows = 70, end_cols = n = 60 (2 iterations
// without refilling).
BK_LIVENESS_BODY(BK_terminates_on_identity,
    for (int64_t i = 0; i < std::min(m, n); ++i) A[i + i * m] = 1.0;)

// Denormal scaling: ||A|| is representable but tau*||A|| underflows toward zero, and so do the
// squared entries the rank test sums, so the first block reads as dead. It is refilled, the next
// iteration finds the refills dead too and retracts them: 2 iterations (1 without refilling).
BK_LIVENESS_BODY(BK_terminates_on_denormal_scaled,
    RandLAPACK::gen::mat_gen_info<double> mi(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(mi, A, state);
    for (int64_t i = 0; i < m * n; ++i) A[i] *= 1e-300;)

// Rank 1: the space stops growing after the first block.
BK_LIVENESS_BODY(BK_terminates_on_rank_one,
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            A[i + j * m] = (double)(i + 1) * (double)(j + 1);)

// Full rank, the ordinary case, to confirm the guard does not fire early.
BK_LIVENESS_BODY(BK_terminates_on_full_rank,
    RandLAPACK::gen::mat_gen_info<double> mi(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(mi, A, state);)

// k > min(m, n) is an out-of-bounds read without a precondition (see rl_bk.hh). There is no
// other EXPECT_THROW in this repo's test tree; randlapack_require is not NDEBUG-gated, so
// this holds in Release builds too.
TEST_F(TestBK, BK_rejects_block_size_exceeding_min_dimension) {
    int64_t m = 10, n = 5, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();
    double* A = new double[m * n]();
    RandLAPACK::gen::mat_gen_info<double> mi(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(mi, A, state);

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = 4;
    EXPECT_THROW(
        bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,
                out.end_rows, out.end_cols, out.final_iter_is_odd, state),
        RandLAPACK::Error);
    delete[] A;
}

// ---------------------------------------------------------------------------------------
// Determinism and resume equivalence.
//
// Added before refills existed, deliberately: refills consume RNG, which shifts every
// downstream stream, so both properties had to be pinned first for an intended change to be
// told apart from an accidental one. BK_refills_are_bitwise_deterministic and
// BK_resume_equals_single_shot_across_a_refill extend them to runs that draw refills.
//
// Resume equivalence was also the guard for a narrowed block reaching resume(): the resume
// path once reconstructed its state as pure k-arithmetic (curr_X_cols = (1+iter_ev)*k,
// curr_Y_cols = iter_od*k), which is silently wrong once a block is not exactly k wide. It now
// restores saved counters; BK_resume_equals_single_shot_across_a_narrowing exercises that.
// ---------------------------------------------------------------------------------------

TEST_F(TestBK, BK_is_bitwise_deterministic_for_a_fixed_seed) {
    int64_t m = 120, n = 60, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    double* A = new double[m * n]();
    {
        auto gs = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen_info<double> mi(m, n, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen(mi, A, gs);
    }

    BKOut<double> o1, o2;
    for (int rep = 0; rep < 2; ++rep) {
        BKOut<double> &o = (rep == 0) ? o1 : o2;
        auto state = RandBLAS::RNGState();          // identical seed both times
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = 8;
        ASSERT_EQ(bk.call(m, n, A, m, k, o.X_ev, o.Y_od, o.R, o.S,
                          o.end_rows, o.end_cols, o.final_iter_is_odd, state), 0);
    }

    ASSERT_EQ(o1.end_rows, o2.end_rows);
    ASSERT_EQ(o1.end_cols, o2.end_cols);
    ASSERT_EQ(o1.final_iter_is_odd, o2.final_iter_is_odd);
    for (int64_t i = 0; i < m * o1.end_rows; ++i)
        ASSERT_EQ(o1.X_ev[i], o2.X_ev[i]) << "X_ev differs at " << i;
    for (int64_t i = 0; i < n * o1.end_cols; ++i)
        ASSERT_EQ(o1.Y_od[i], o2.Y_od[i]) << "Y_od differs at " << i;

    delete[] A;
}

// call(p) must equal call(p1) followed by resume(p), bitwise, on the geqrf path.
TEST_F(TestBK, BK_resume_equals_single_shot) {
    int64_t m = 120, n = 60, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    const int p1 = 4, p = 8;

    double* A = new double[m * n]();
    {
        auto gs = RandBLAS::RNGState();
        RandLAPACK::gen::mat_gen_info<double> mi(m, n, RandLAPACK::gen::gaussian);
        RandLAPACK::gen::mat_gen(mi, A, gs);
    }
    RandLAPACK::linops::DenseLinOp<double> A_op(m, n, A, m, Layout::ColMajor);

    // Single shot to p.
    BKOut<double> one;
    {
        auto state = RandBLAS::RNGState();
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = p;
        ASSERT_EQ(bk.call(A_op, k, one.X_ev, one.Y_od, one.R, one.S,
                          one.end_rows, one.end_cols, one.final_iter_is_odd, state), 0);
    }

    // p1, then resume to p.
    BKOut<double> two;
    {
        auto state = RandBLAS::RNGState();
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = p1;
        ASSERT_EQ(bk.call(A_op, k, two.X_ev, two.Y_od, two.R, two.S,
                          two.end_rows, two.end_cols, two.final_iter_is_odd, state), 0);
        ASSERT_EQ(bk.termination_reason, RandLAPACK::BKTermination::max_iters_reached)
            << "resume is only defined after max_iters_reached; the premise of this test";
        bk.max_krylov_iters = p;
        ASSERT_EQ(bk.resume(A_op, k, two.X_ev, two.Y_od, two.R, two.S,
                            two.end_rows, two.end_cols, two.final_iter_is_odd, state), 0);
    }

    printf("RESUME single(%d): rows=%ld cols=%ld | %d then resume(%d): rows=%ld cols=%ld\n",
           p, (long)one.end_rows, (long)one.end_cols,
           p1, p, (long)two.end_rows, (long)two.end_cols);
    fflush(stdout);

    ASSERT_EQ(one.end_rows, two.end_rows);
    ASSERT_EQ(one.end_cols, two.end_cols);
    ASSERT_EQ(one.final_iter_is_odd, two.final_iter_is_odd);
    for (int64_t i = 0; i < m * one.end_rows; ++i)
        ASSERT_EQ(one.X_ev[i], two.X_ev[i]) << "X_ev differs at " << i;
    for (int64_t i = 0; i < n * one.end_cols; ++i)
        ASSERT_EQ(one.Y_od[i], two.Y_od[i]) << "Y_od differs at " << i;

    delete[] A;
}

// The even/S path, for contrast: a budget that ends on an even iteration.
TEST_F(TestBK, BK_band_equals_XtAY_even_final_iteration) {
    int64_t m = 200;
    int64_t n = 100;
    int64_t k = 10;
    double  tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    double* A = new double[m * n]();
    RandLAPACK::gen::mat_gen_info<double> m_info(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(m_info, A, state);

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = (int)((2 * n) / k);

    int status = bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,
                         out.end_rows, out.end_cols, out.final_iter_is_odd, state);
    ASSERT_EQ(status, 0);

    printf("BK iters=%d reason=%d end_rows=%ld end_cols=%ld odd=%d\n",
           bk.num_krylov_iters, (int)bk.termination_reason,
           (long)out.end_rows, (long)out.end_cols, (int)out.final_iter_is_odd);
    double oX = orth_err<double>(out.X_ev, m, out.end_rows);
    double oY = orth_err<double>(out.Y_od, n, out.end_cols);
    printf("BASIS orth: ||X'X-I||/sqrt=%.3e  ||Y'Y-I||/sqrt=%.3e\n", oX, oY);
    fflush(stdout);

    check_band_identity<double>(m, n, k, A, out, 1e-10);

    delete[] A;
}

// Stop at the narrowed even block to exercise S with leading dimension n + k
// and non-square geometry. A longer run can stop through either norm convergence
// or a zero-width rank probe, depending on roundoff in the BLAS/LAPACK backend.
TEST_F(TestBK, BK_even_terminal_band_identity_cqrrt) {
    int64_t m = 200, n = 200, k = 10, r = 25;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    std::vector<double> s(r);
    for (int i = 0; i < r; ++i) s[i] = std::pow(10.0, -3.0 * i / (r - 1));
    std::vector<double> S(r * r, 0.0);
    RandLAPACK::util::diag(r, r, s.data(), r, S.data());
    double* A = new double[m * n]();
    RandLAPACK::gen::gen_singvec<double>(m, n, A, r, S.data(), state);

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.qr_exp = RandLAPACK::BKSubroutines::QR_explicit::cqrrt;
    bk.max_krylov_iters = 4;
    ASSERT_EQ(bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,
                      out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);

    EXPECT_EQ(bk.termination_reason, RandLAPACK::BKTermination::max_iters_reached);
    EXPECT_FALSE(out.final_iter_is_odd) << "expected an even terminal iteration";
    EXPECT_EQ(bk.final_block_width, (int64_t) 5);
    EXPECT_GE(bk.narrowed_blocks, (int64_t) 1) << "continuation must actually have fired";
    EXPECT_EQ(out.end_rows, (int64_t) 25);
    EXPECT_EQ(out.end_cols, (int64_t) 20);

    check_band_identity<double>(m, n, k, A, out, 1e-10);
    delete[] A;
}

TEST_F(TestBK, BK_diagnose_exact_rank_25) {
    int64_t m = 200, n = 200, k = 10, r = 25;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();

    std::vector<double> s(r);
    for (int i = 0; i < r; ++i) s[i] = std::pow(10.0, -3.0 * i / (r - 1));
    std::vector<double> S(r * r, 0.0);
    RandLAPACK::util::diag(r, r, s.data(), r, S.data());
    double* A = new double[m * n]();
    RandLAPACK::gen::gen_singvec<double>(m, n, A, r, S.data(), state);

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = 40;
    ASSERT_EQ(bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,
                      out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
    printf("T2DIAG iters=%d reason=%d width=%ld end_rows=%ld end_cols=%ld odd=%d\n",
           bk.num_krylov_iters, (int)bk.termination_reason, (long)bk.final_block_width,
           (long)out.end_rows, (long)out.end_cols, (int)out.final_iter_is_odd);
    fflush(stdout);
    delete[] A;
}



/// Rank 39 at block size 10 is the one rank in 20..40 that does not certify r of r under the
/// DEFAULT tau, and this test records why: it is threshold sensitivity in tau, not the
/// stranding mechanism that prune-and-narrow fixes.
///
/// With tau defaulting to n*eps (4.44e-14 at n = 200), the tenth column of the iteration-6
/// block carries a reorthogonalisation residual just above tau*||A||, so the block is
/// accepted at full width and the left basis reaches 40 columns for a rank-39 matrix. The
/// run then exits through norm_converged on an ODD terminal with end_rows = 40 > end_cols =
/// 39, and the single junk left column prevents any triplet from certifying.
///
/// Raising tau to 1e-12 makes it behave exactly like ranks 37, 38 and 40: the block narrows,
/// the run continues, and it stops at iteration 8 with end_rows = end_cols = 39.
///
/// This is NOT a regression from continuation: rank 39 measured claimed 39 / certified 0
/// before continuation as well. The default tau is deliberately left alone, because the
/// ill-conditioned regime (kappa 1e10, 155 of 200 certified) depends on tau being small
/// enough not to discard genuine trailing directions. That trade-off is exactly what the
/// user-facing tau knob exists for.
TEST_F(TestBK, BK_rank_39_is_a_tau_sensitivity_not_a_shortfall) {
    int64_t m = 200, n = 200, k = 10, r = 39;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    auto run = [&](double tau_val, int64_t &er, int64_t &ec, int &reason, int64_t &narrowed) {
        auto state = RandBLAS::RNGState();
        std::vector<double> s(r);
        for (int i = 0; i < r; ++i) s[i] = std::pow(10.0, -3.0 * i / (r - 1));
        std::vector<double> S(r * r, 0.0);
        RandLAPACK::util::diag(r, r, s.data(), r, S.data());
        double* A = new double[m * n]();
        RandLAPACK::gen::gen_singvec<double>(m, n, A, r, S.data(), state);
        BKOut<double> out;
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = 40;
        bk.tau = tau_val;
        ASSERT_EQ(bk.call(m, n, A, m, k, out.X_ev, out.Y_od, out.R, out.S,
                          out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
        er = out.end_rows; ec = out.end_cols;
        reason = (int) bk.termination_reason; narrowed = bk.narrowed_blocks;
        delete[] A;
    };

    int64_t er = 0, ec = 0, narrowed = 0; int reason = 0;

    // Default tau: over-accepts the left basis by one column.
    run(0.0, er, ec, reason, narrowed);
    EXPECT_EQ(er, (int64_t) 40) << "default tau accepts a junk 40th left column";
    EXPECT_EQ(ec, (int64_t) 39);

    // A slightly larger tau rejects it and the run completes symmetrically.
    run(1e-12, er, ec, reason, narrowed);
    EXPECT_EQ(er, (int64_t) 39);
    EXPECT_EQ(ec, (int64_t) 39);
    EXPECT_GE(narrowed, (int64_t) 1) << "continuation must have fired";
}

/// Resume ACROSS a narrowing. This is the guard for the persisted resume state, and it is the
/// case the old code could not have survived.
///
/// Before continuation, resume() reconstructed curr_X_cols = (1 + iter_ev) * k and
/// curr_Y_cols = iter_od * k from the iteration count, which is correct only while every
/// block is exactly k wide. It was latent rather than broken because narrowing terminated the
/// loop, so a narrowed state could never reach resume(). Continuation makes it reachable: a
/// narrowed run now ends at max_iters_reached, which is precisely the state ABRIK resumes
/// from. Saving the counters instead of recomputing them is what makes this test pass;
/// with the old arithmetic the two legs would diverge from the first post-checkpoint block.
///
/// Exact rank 25 at block size 10 narrows the left block to 5 at iteration 4, so a checkpoint
/// at 4 sits after the narrowing and before possible norm convergence at 5.
/// The helper asserts that the checkpoint is both narrowed and resumable.
TEST_F(TestBK, BK_resume_equals_single_shot_across_a_narrowing) {
    int64_t m = 200, n = 200, k = 10, r = 25;

    std::vector<double> sv(r);
    for (int i = 0; i < r; ++i) sv[i] = std::pow(10.0, -3.0 * i / (r - 1));
    std::vector<double> S(r * r, 0.0);
    RandLAPACK::util::diag(r, r, sv.data(), r, S.data());
    double* A = new double[m * n]();
    {
        auto gs = RandBLAS::RNGState();
        RandLAPACK::gen::gen_singvec<double>(m, n, A, r, S.data(), gs);
    }

    check_resume_equals_single_shot(m, n, k, /*p1=*/4, /*p=*/8, A, /*expect_narrowed=*/true);
    delete[] A;
}


/// Matvec accounting. BK applies the operator in exactly three places: once in the prologue
/// (NoTrans), once per odd iteration (Trans, forming Y from X), and once per even iteration
/// (NoTrans, forming X from Y). One application per iteration, no more, plus one for the
/// prologue, which runs before iteration 1. Nothing checked that until now, so a stray extra
/// apply, or a reorthogonalisation pass silently routed through the operator, would have cost
/// matvecs without any test noticing. The counts are exact integers, so there is no tolerance
/// here.
class CountingLinOp {
    public:
        using scalar_t = double;
        const int64_t n_rows;
        const int64_t n_cols;
        RandLAPACK::linops::DenseLinOp<double> inner;
        mutable int64_t n_notrans = 0;
        mutable int64_t n_trans   = 0;

        CountingLinOp(int64_t m, int64_t n, const double* A, int64_t lda)
            : n_rows(m), n_cols(n), inner(m, n, A, lda, Layout::ColMajor) {}

        // BK needs the Frobenius norm to anchor the rank criterion. Pass-through, and
        // deliberately NOT counted: it is not an operator application.
        double fro_nrm() { return inner.fro_nrm(); }

        // The 12-argument form required by the LinearOperator concept.
        void operator()(Layout layout, Op trans_A, Op trans_B,
                        int64_t m, int64_t n, int64_t k, double alpha,
                        const double* B, int64_t ldb, double beta, double* C, int64_t ldc) {
            (*this)(Side::Left, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        }

        // The 13-argument form BK actually calls.
        void operator()(Side side, Layout layout, Op trans_A, Op trans_B,
                        int64_t m, int64_t n, int64_t k, double alpha,
                        const double* B, int64_t ldb, double beta, double* C, int64_t ldc) {
            if (trans_A == Op::Trans) ++n_trans; else ++n_notrans;
            inner(side, layout, trans_A, trans_B, m, n, k, alpha, B, ldb, beta, C, ldc);
        }
};

/// File-scope RAII holder for BK's four calloc'd outputs. Mirrors the fixture's BKOut, which
/// is a protected nested type and so not visible to a free function.
struct BKOutFree {
    double* X_ev = nullptr;
    double* Y_od = nullptr;
    double* R    = nullptr;
    double* S    = nullptr;
    int64_t end_rows = 0;
    int64_t end_cols = 0;
    bool final_iter_is_odd = false;
    ~BKOutFree() { free(X_ev); free(Y_od); free(R); free(S); }
};

static void check_matvec_accounting(int64_t m, int64_t n, int64_t k, int budget,
                                    const double* A_src, const char* label) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    CountingLinOp op(m, n, A_src, m);
    BKOutFree out;
    auto state = RandBLAS::RNGState();
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = budget;
    ASSERT_EQ(bk.call(op, k, out.X_ev, out.Y_od, out.R, out.S,
                      out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);

    const int64_t iters = bk.num_krylov_iters;
    printf("MATVEC %-22s iters=%2ld notrans=%2ld trans=%2ld reason=%d\n",
           label, (long)iters, (long)op.n_notrans, (long)op.n_trans,
           (int)bk.termination_reason);
    fflush(stdout);

    // iters + 1, not iters. The prologue applies A once (X = A*Omega) BEFORE iter is
    // incremented to 1, so it is not counted by num_krylov_iters; the loop then applies A
    // exactly once per iteration 1..iters. Measured 41 at iters=40 and 7 at iters=6.
    EXPECT_EQ(op.n_notrans + op.n_trans, iters + 1)
        << "one operator application per iteration, plus the prologue, and no more";
    // Parity is pinned: the prologue is NoTrans, odd iterations are Trans (Y from X) and
    // even iterations are NoTrans (X from Y).
    EXPECT_EQ(op.n_trans,   (iters + 1) / 2) << "one per odd iteration";
    EXPECT_EQ(op.n_notrans, iters / 2 + 1)   << "one per even iteration, plus the prologue";
}

TEST_F(TestBK, BK_matvec_count_matches_iterations) {
    int64_t m = 400, n = 200, k = 10;
    double* A = new double[m * n]();
    auto gs = RandBLAS::RNGState();
    RandLAPACK::gen::mat_gen_info<double> mi(m, n, RandLAPACK::gen::gaussian);
    RandLAPACK::gen::mat_gen(mi, A, gs);
    check_matvec_accounting(m, n, k, 40, A, "gaussian budget40");
    delete[] A;
}

/// Second case terminates EARLY, through the rank criterion rather than the budget, so the
/// count is checked against work actually done rather than against a budget the run exhausts.
TEST_F(TestBK, BK_matvec_count_on_early_termination) {
    int64_t m = 200, n = 200, k = 10, r = 25;
    std::vector<double> sv(r);
    for (int i = 0; i < r; ++i) sv[i] = std::pow(10.0, -3.0 * i / (r - 1));
    std::vector<double> S(r * r, 0.0);
    RandLAPACK::util::diag(r, r, sv.data(), r, S.data());
    double* A = new double[m * n]();
    auto gs = RandBLAS::RNGState();
    RandLAPACK::gen::gen_singvec<double>(m, n, A, r, S.data(), gs);
    check_matvec_accounting(m, n, k, 40, A, "rank25 early-exit");
    delete[] A;
}

// ---------------------------------------------------------------------------------------
// Refilling dead block columns.
//
// When the rank test rejects columns of a new block, those slots are refilled with random
// directions orthogonal to the basis, so the run continues where the Krylov space closed early.
// The first and third tests below pin that behaviour and fail without it; the second passed
// before refilling existed and guards that refills never touch the operator.
// ---------------------------------------------------------------------------------------

/// The identity closes its Krylov space after one block, so every column past k must come
/// from a refill. Fails without refilling: prune-and-narrow stops at iteration 2 with 10 columns.
TEST_F(TestBK, BK_identity_refills_to_saturation) {
    const int64_t m = 200, n = 200, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();
    std::vector<double> A(m * n, 0.0);
    for (int64_t i = 0; i < n; ++i) A[i + i * m] = 1.0;

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = 40;
    ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                      out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
    printf("REFILL identity iters=%d reason=%d end_rows=%ld end_cols=%ld odd=%d\n",
           bk.num_krylov_iters, (int)bk.termination_reason,
           (long)out.end_rows, (long)out.end_cols, (int)out.final_iter_is_odd);
    fflush(stdout);

    EXPECT_EQ(out.end_rows, (int64_t) 200);
    EXPECT_EQ(out.end_cols, (int64_t) 200);
    // Whether norm_R clears sqrt(1 - tol^2)||A||_F at iteration 39 is decided by rounding.
    const auto reason = bk.termination_reason;
    EXPECT_TRUE(reason == RandLAPACK::BKTermination::saturated ||
                reason == RandLAPACK::BKTermination::norm_converged)
        << "termination_reason=" << (int)reason;
    EXPECT_TRUE(bk.num_krylov_iters == 39 || bk.num_krylov_iters == 40)
        << "num_krylov_iters=" << bk.num_krylov_iters;

    // orth_err divides by sqrt(cols); the bound here is on the unnormalized ||Q^T Q - I||_F.
    EXPECT_LT(orth_err<double>(out.X_ev, m, out.end_rows) * std::sqrt((double)out.end_rows), 1e-13);
    EXPECT_LT(orth_err<double>(out.Y_od, n, out.end_cols) * std::sqrt((double)out.end_cols), 1e-13);

    check_band_identity<double>(m, n, k, A.data(), out, 1e-10);

    // X' I Y is orthogonal when both bases span R^n, so every singular value of the band is 1.
    const int64_t er = out.end_rows, ec = out.end_cols;
    const double* band = out.final_iter_is_odd ? out.R : out.S;
    const int64_t ldb = out.final_iter_is_odd ? n : (n + k);
    std::vector<double> band_cpy(er * ec), sv(std::min(er, ec));
    lapack::lacpy(MatrixType::General, er, ec, band, ldb, band_cpy.data(), er);
    ASSERT_EQ(lapack::gesdd(Job::NoVec, er, ec, band_cpy.data(), er, sv.data(),
                            static_cast<double*>(nullptr), 1,
                            static_cast<double*>(nullptr), 1), 0);
    for (size_t i = 0; i < sv.size(); ++i)
        EXPECT_NEAR(sv[i], 1.0, 1e-13) << "band singular value " << i;
}

/// Refills are random draws and never touch the operator, so the padded identity must still
/// cost one application per iteration plus the prologue. Without refilling the run stops at
/// iteration 2; with it the space is refilled and the whole budget of 8 is spent.
TEST_F(TestBK, BK_refill_matvec_accounting) {
    int64_t m = 400, n = 200, k = 10;
    double* A = new double[m * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * m] = 1.0;
    check_matvec_accounting(m, n, k, 8, A, "identity refill budget8");
    delete[] A;
}

/// A = [I_20 0] has a 20-dimensional range, so the left basis must stop at 20 columns: the
/// room for X-side refills is min(m, n) - x_cols, never n - x_cols. Fails without refilling,
/// stopping at iteration 2 with 10 columns, because A A^T = I closes the space after one block.
/// The end counts alone do not pin the bound. With room from n, iteration 4 would refill 10
/// junk columns, iteration 5 would probe them dead and retract them, and the run would end
/// rank_deficient at iteration 5 with the same end_rows = end_cols = 20; saturated at
/// iteration 4 is what separates the two.
TEST_F(TestBK, BK_refill_room_uses_min_of_m_and_n) {
    const int64_t m = 20, n = 40, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    auto state = RandBLAS::RNGState();
    std::vector<double> A(m * n, 0.0);
    for (int64_t i = 0; i < m; ++i) A[i + i * m] = 1.0;

    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = 40;
    ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                      out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
    printf("REFILL room iters=%d reason=%d end_rows=%ld end_cols=%ld odd=%d\n",
           bk.num_krylov_iters, (int)bk.termination_reason,
           (long)out.end_rows, (long)out.end_cols, (int)out.final_iter_is_odd);
    fflush(stdout);

    EXPECT_EQ(out.end_rows, (int64_t) 20);
    EXPECT_EQ(out.end_cols, (int64_t) 20);
    EXPECT_LT(orth_err<double>(out.X_ev, m, out.end_rows) * std::sqrt((double)out.end_rows), 1e-13);
    EXPECT_EQ(bk.termination_reason, RandLAPACK::BKTermination::saturated);
    EXPECT_EQ(bk.num_krylov_iters, 4);
}

/// Exact rank 2 at k = 10, singular values 1 and 1e-10. The first Y block keeps 2 columns and
/// refills 8. Everything M has is captured at that point, so the run can end in two ways, and
/// which one happens depends on rounding in ||R||_F against ||M||_F (the content threshold is
/// exactly ||M||_F in double for any small tol): MKL continues to iteration 2, where the probe
/// finds the 8 refill images dead, retracts them and stops as rank_deficient; OpenBLAS and
/// Accelerate stop at iteration 1 as norm_converged with the 8 refills unprobed. Both must give
/// the same reported result. The second pass forces the iteration-1 exit with a loose tol.
TEST_F(TestBK, BK_exact_rank_two_refills_once_then_probe_switches_off) {
    const int64_t m = 200, n = 200, k = 10;
    std::vector<double> A(m * n, 0.0);
    build_from_spectrum(m, n, {1.0, 1e-10}, A.data());

    for (double tol : {std::pow(std::numeric_limits<double>::epsilon(), 0.85), 0.25}) {
        SCOPED_TRACE(tol);
        auto state = RandBLAS::RNGState();
        BKOut<double> out;
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = 40;
        ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                          out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
        printf("RANK2 tol=%.1e iters=%d reason=%d refilled=%ld exhausted=%d pending=%ld narrowed=%ld end_rows=%ld end_cols=%ld odd=%d\n",
               tol, bk.num_krylov_iters, (int)bk.termination_reason, (long)bk.refilled_blocks,
               (int)bk.refills_exhausted, (long)bk.saved_pending_refills, (long)bk.narrowed_blocks,
               (long)out.end_rows, (long)out.end_cols, (int)out.final_iter_is_odd);
        fflush(stdout);

        // Independent of the exit taken.
        EXPECT_EQ(bk.refilled_blocks, (int64_t) 1);
        EXPECT_EQ(bk.narrowed_blocks, (int64_t) 1) << "only the first Y block narrows";
        ASSERT_EQ(out.end_rows, (int64_t) 10);
        ASSERT_EQ(out.end_cols, (int64_t) 2) << "the 8 refills are never reported, probed or not";

        // The exit taken, and what each one implies.
        ASSERT_TRUE(bk.num_krylov_iters == 1 || bk.num_krylov_iters == 2);
        if (tol == 0.25)
            EXPECT_EQ(bk.num_krylov_iters, 1) << "a loose tol must take the content exit at once";
        if (bk.num_krylov_iters == 2) {
            EXPECT_EQ(bk.termination_reason, RandLAPACK::BKTermination::rank_deficient);
            EXPECT_TRUE(bk.refills_exhausted) << "the probe found the refill images dead";
            EXPECT_EQ(bk.saved_pending_refills, (int64_t) 0);
            EXPECT_FALSE(out.final_iter_is_odd);
        } else {
            EXPECT_EQ(bk.termination_reason, RandLAPACK::BKTermination::norm_converged);
            EXPECT_FALSE(bk.refills_exhausted) << "no probe ran";
            EXPECT_EQ(bk.saved_pending_refills, (int64_t) 8) << "the refills stay in the resume state";
            EXPECT_TRUE(out.final_iter_is_odd);
        }

        // The reported band carries both singular values, the small one to absolute roundoff.
        const int64_t er = out.end_rows, ec = out.end_cols;
        const double* band = out.final_iter_is_odd ? out.R : out.S;
        const int64_t ldb = out.final_iter_is_odd ? n : (n + k);
        std::vector<double> band_cpy(er * ec), sv(std::min(er, ec));
        lapack::lacpy(MatrixType::General, er, ec, band, ldb, band_cpy.data(), er);
        ASSERT_EQ(lapack::gesdd(Job::NoVec, er, ec, band_cpy.data(), er, sv.data(),
                                static_cast<double*>(nullptr), 1,
                                static_cast<double*>(nullptr), 1), 0);
        printf("RANK2 band sv = %.17e %.17e\n", sv[0], sv[1]);
        fflush(stdout);
        EXPECT_LE(std::abs(sv[0] - 1.0),   1e-13);
        EXPECT_LE(std::abs(sv[1] - 1e-10), 1e-14);

        // The reported prefix of Y is orthonormal; the refills sit past column 2.
        EXPECT_LT(orth_err<double>(out.Y_od, n, 2) * std::sqrt(2.0), 1e-13);
    }
}

/// The odd-parity exit with pending refills. On the matrix above, a budget of 1 stops right
/// after the Y block keeps 2 columns and refills 8, before any iteration has probed them. The
/// refills stay in y_cols and in the resume state, but end_cols must report only the 2 probed
/// columns: end_cols = y_cols - pending_refills after an odd final iteration.
TEST_F(TestBK, BK_odd_budget_exit_does_not_report_pending_refills) {
    const int64_t m = 200, n = 200, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);
    std::vector<double> A(m * n, 0.0);
    build_from_spectrum(m, n, {1.0, 1e-10}, A.data());

    auto state = RandBLAS::RNGState();
    BKOut<double> out;
    RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
    bk.max_krylov_iters = 1;
    ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                      out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
    printf("RANK2-BUDGET1 iters=%d reason=%d refilled=%ld pending=%ld end_rows=%ld end_cols=%ld odd=%d\n",
           bk.num_krylov_iters, (int)bk.termination_reason, (long)bk.refilled_blocks,
           (long)bk.saved_pending_refills, (long)out.end_rows, (long)out.end_cols,
           (int)out.final_iter_is_odd);
    fflush(stdout);

    EXPECT_EQ(bk.num_krylov_iters, 1);
    // The budget and the content test both end this run at iteration 1; which one fires first
    // depends on rounding in ||R||_F against ||M||_F (see BK_exact_rank_two_refills_once...).
    EXPECT_TRUE(bk.termination_reason == RandLAPACK::BKTermination::max_iters_reached ||
                bk.termination_reason == RandLAPACK::BKTermination::norm_converged)
        << "reason=" << (int)bk.termination_reason;
    EXPECT_TRUE(out.final_iter_is_odd);
    EXPECT_EQ(bk.refilled_blocks, (int64_t) 1);
    EXPECT_EQ(bk.saved_pending_refills, (int64_t) 8) << "the exit must be taken with refills pending";
    EXPECT_EQ(out.end_rows, (int64_t) 10);
    EXPECT_EQ(out.end_cols, (int64_t) 2) << "the 8 unprobed Y refills must not be reported";
}

/// Resume across a refill. The checkpoint must hold refills that no iteration has probed yet,
/// so that resume() restores and probes them exactly as the single shot does.
/// Rank 25 (the matrix of BK_resume_equals_single_shot_across_a_narrowing): iteration 4 keeps 5
/// X columns and refills 5, and iteration 5 probes them dead, which exhausts refilling.
/// The identity: every even block is entirely old and refilled in full, so the even checkpoint
/// at 4 holds k pending refills; they are never probed dead, so refilling stays on.
/// Exact rank 2 (the matrix of BK_exact_rank_two_refills_once_then_probe_switches_off): the odd
/// checkpoint at 1 holds the 8 Y-side refills, which iteration 2 probes dead and retracts. This
/// leg runs only where iteration 1 ends on the budget rather than on the content test.
TEST_F(TestBK, BK_resume_equals_single_shot_across_a_refill) {
    {
        SCOPED_TRACE("rank 25");
        const int64_t m = 200, n = 200, k = 10;
        std::vector<double> A(m * n, 0.0);
        build_from_spectrum(m, n, decaying_spectrum(25), A.data());
        ResumeCheckpoint ck;
        check_resume_equals_single_shot(m, n, k, /*p1=*/4, /*p=*/8, A.data(),
                                        /*expect_narrowed=*/true, &ck);
        if (HasFatalFailure()) return;
        printf("RESUME-REFILL rank25 refilled=%ld pending=%ld exhausted_after=%d\n",
               (long)ck.refilled_blocks, (long)ck.saved_pending_refills, (int)ck.refills_exhausted);
        fflush(stdout);
        EXPECT_GE(ck.refilled_blocks, (int64_t) 1);
        EXPECT_EQ(ck.saved_pending_refills, (int64_t) 5);
        EXPECT_TRUE(ck.refills_exhausted) << "iteration 5 must probe the 5 refills and find them dead";
    }
    {
        SCOPED_TRACE("identity 60");
        const int64_t m = 60, n = 60, k = 10;
        std::vector<double> A(m * n, 0.0);
        for (int64_t i = 0; i < n; ++i) A[i + i * m] = 1.0;
        ResumeCheckpoint ck;
        check_resume_equals_single_shot(m, n, k, /*p1=*/4, /*p=*/9, A.data(),
                                        /*expect_narrowed=*/true, &ck);
        if (HasFatalFailure()) return;
        printf("RESUME-REFILL identity60 refilled=%ld pending=%ld exhausted_after=%d\n",
               (long)ck.refilled_blocks, (long)ck.saved_pending_refills, (int)ck.refills_exhausted);
        fflush(stdout);
        EXPECT_GE(ck.refilled_blocks, (int64_t) 1);
        EXPECT_EQ(ck.saved_pending_refills, k) << "an even checkpoint must hold the refilled X block";
        EXPECT_FALSE(ck.refills_exhausted) << "identity refills are confirmed at full width";
    }
    {
        SCOPED_TRACE("rank 2, odd checkpoint");
        const int64_t m = 200, n = 200, k = 10;
        std::vector<double> A(m * n, 0.0);
        build_from_spectrum(m, n, {1.0, 1e-10}, A.data());
        // Resume is defined only after max_iters_reached. On this matrix the content test can
        // fire at iteration 1 instead, depending on the BLAS (see the rank-2 test above); then
        // there is nothing to resume and this leg does not apply. The odd-parity exclusion of
        // the pending refills is pinned by BK_odd_budget_exit_does_not_report_pending_refills.
        {
            auto state = RandBLAS::RNGState();
            BKOut<double> probe;
            RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, std::pow(std::numeric_limits<double>::epsilon(), 0.85));
            bk.max_krylov_iters = 1;
            ASSERT_EQ(bk.call(m, n, A.data(), m, k, probe.X_ev, probe.Y_od, probe.R, probe.S,
                              probe.end_rows, probe.end_cols, probe.final_iter_is_odd, state), 0);
            if (bk.termination_reason == RandLAPACK::BKTermination::norm_converged) {
                printf("RESUME-REFILL rank2: content exit at iteration 1 on this BLAS, leg skipped\n");
                fflush(stdout);
                return;
            }
        }
        ResumeCheckpoint ck;
        check_resume_equals_single_shot(m, n, k, /*p1=*/1, /*p=*/2, A.data(),
                                        /*expect_narrowed=*/true, &ck);
        if (HasFatalFailure()) return;
        printf("RESUME-REFILL rank2 refilled=%ld pending=%ld exhausted_after=%d\n",
               (long)ck.refilled_blocks, (long)ck.saved_pending_refills, (int)ck.refills_exhausted);
        fflush(stdout);
        EXPECT_EQ(ck.refilled_blocks, (int64_t) 1);
        EXPECT_EQ(ck.saved_pending_refills, (int64_t) 8) << "an odd checkpoint must hold the Y refills";
        EXPECT_TRUE(ck.refills_exhausted) << "iteration 2 must probe the 8 refills and find them dead";
    }
}

/// Refills are random draws from the RNG state BK threads through the run, so two runs from
/// equal seeds must agree bitwise, exactly as without refills. Both inputs refill before the
/// budget of 8 runs out, which the refilled_blocks check guarantees.
TEST_F(TestBK, BK_refills_are_bitwise_deterministic) {
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    auto check = [&](int64_t m, int64_t n, int64_t k, const double* A_src, const char* label) {
        SCOPED_TRACE(label);
        std::vector<double> A(A_src, A_src + m * n);
        BKOut<double> o1, o2;
        for (int rep = 0; rep < 2; ++rep) {
            BKOut<double> &o = (rep == 0) ? o1 : o2;
            auto state = RandBLAS::RNGState();          // identical seed both times
            RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
            bk.max_krylov_iters = 8;
            ASSERT_EQ(bk.call(m, n, A.data(), m, k, o.X_ev, o.Y_od, o.R, o.S,
                              o.end_rows, o.end_cols, o.final_iter_is_odd, state), 0);
            ASSERT_GE(bk.refilled_blocks, (int64_t) 1)
                << "the run must cross a refill, or this is a copy of the no-refill test";
        }
        printf("DETERMINISM-REFILL %-12s end_rows=%ld end_cols=%ld odd=%d\n",
               label, (long)o1.end_rows, (long)o1.end_cols, (int)o1.final_iter_is_odd);
        fflush(stdout);

        ASSERT_EQ(o1.end_rows, o2.end_rows);
        ASSERT_EQ(o1.end_cols, o2.end_cols);
        ASSERT_EQ(o1.final_iter_is_odd, o2.final_iter_is_odd);
        for (int64_t i = 0; i < m * o1.end_rows; ++i)
            ASSERT_EQ(o1.X_ev[i], o2.X_ev[i]) << "X_ev differs at " << i;
        for (int64_t i = 0; i < n * o1.end_cols; ++i)
            ASSERT_EQ(o1.Y_od[i], o2.Y_od[i]) << "Y_od differs at " << i;
    };

    {
        const int64_t n = 60;
        std::vector<double> A(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i) A[i + i * n] = 1.0;
        check(n, n, 10, A.data(), "identity 60");
    }
    if (HasFatalFailure()) return;
    {
        const int64_t m = 200, n = 200;
        std::vector<double> A(m * n, 0.0);
        build_from_spectrum(m, n, decaying_spectrum(25), A.data());
        check(m, n, 10, A.data(), "rank 25");
    }
}

/// The band still equals X' A Y once refills have entered the basis, been probed, and been
/// retracted. Refill slots in the band hold zeros whose true value is bounded by tau*||A||_F,
/// so the identity holds to the same tolerance as without refills.
TEST_F(TestBK, BK_band_identity_after_refills) {
    const int64_t m = 200, n = 200, k = 10;
    double tol = std::pow(std::numeric_limits<double>::epsilon(), 0.85);

    // Rank 25 at budget 8: refilled at iteration 4, probed dead and retracted at iteration 5.
    {
        SCOPED_TRACE("rank 25");
        std::vector<double> A(m * n, 0.0);
        build_from_spectrum(m, n, decaying_spectrum(25), A.data());
        auto state = RandBLAS::RNGState();
        BKOut<double> out;
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = 8;
        ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                          out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
        printf("BAND-REFILL rank25 iters=%d reason=%d refilled=%ld exhausted=%d end_rows=%ld end_cols=%ld\n",
               bk.num_krylov_iters, (int)bk.termination_reason, (long)bk.refilled_blocks,
               (int)bk.refills_exhausted, (long)out.end_rows, (long)out.end_cols);
        fflush(stdout);
        EXPECT_GE(bk.refilled_blocks, (int64_t) 1);
        EXPECT_TRUE(bk.refills_exhausted);
        EXPECT_EQ(out.end_rows, (int64_t) 25);
        EXPECT_EQ(out.end_cols, (int64_t) 25);
        check_band_identity<double>(m, n, k, A.data(), out, 1e-10);
    }
    if (HasFatalFailure()) return;

    // Singular values 1 and 1e-10: eight Y refills probed dead and retracted at iteration 2.
    {
        SCOPED_TRACE("(1, 1e-10)");
        std::vector<double> A(m * n, 0.0);
        build_from_spectrum(m, n, {1.0, 1e-10}, A.data());
        auto state = RandBLAS::RNGState();
        BKOut<double> out;
        RandLAPACK::BK<double, r123::Philox4x32> bk(false, false, tol);
        bk.max_krylov_iters = 40;
        ASSERT_EQ(bk.call(m, n, A.data(), m, k, out.X_ev, out.Y_od, out.R, out.S,
                          out.end_rows, out.end_cols, out.final_iter_is_odd, state), 0);
        EXPECT_GE(bk.refilled_blocks, (int64_t) 1);
        check_band_identity<double>(m, n, k, A.data(), out, 1e-10);
    }
}
