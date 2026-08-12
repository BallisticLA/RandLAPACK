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
    /// It also settles a documentation ambiguity. rl_bk.hh:96 calls R an "Upper band matrix
    /// (stored transposed)", but ABRIK hands the stored buffer straight to gesdd
    /// (rl_abrik.hh:340), so whether the stored orientation or its transpose is the true
    /// band is not something the comments pin down. Both orientations are measured and
    /// reported; the assertion accepts whichever the code actually means, and the printed
    /// pair records which one that is.
    template <typename T>
    static void check_band_identity(
        int64_t m, int64_t n, const T* A, BKOut<T> &out, T rtol
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

        // The band as stored: R (ld n) on an odd final iteration, S (ld n+k) on an even one.
        const T* band = out.final_iter_is_odd ? out.R : out.S;
        const int64_t ldb = out.final_iter_is_odd ? n : (n + (er - ec));

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
};


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

    check_band_identity<double>(m, n, A, out, 1e-10);

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
// L0: the rank criterion in isolation.
//
// block_numerical_rank is a free function precisely so it can be tested here, with no
// Krylov iteration, no BLAS and no RNG -- microseconds, and fully deterministic. Each case
// below is a property the old inline test
//     std::abs(R_ii[(n + 1) * (k - 1)]) < std::sqrt(eps)
// got wrong.
// ---------------------------------------------------------------------------------------

namespace {
    // Build a k x k lower-triangular block with the given diagonal, ld = k.
    template <typename T>
    std::vector<T> tri_block(const std::vector<T>& diag) {
        int64_t k = (int64_t)diag.size();
        std::vector<T> B(k * k, (T)0);
        for (int64_t i = 0; i < k; ++i) B[i + i * k] = diag[i];
        return B;
    }
}

TEST_F(TestBK, criterion_healthy_block_keeps_every_column) {
    std::vector<double> d(8, 1.0);
    auto B = tri_block(d);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(8, B.data(), 8, 1.0, 1e-12), 8);
}

// The case that killed a block-relative criterion. Every diagonal is ~1e-16, so the ratio
// between them is ~1 and any rule anchored to the block's OWN scale sees a perfectly
// conditioned block and flags nothing. Anchored to ||A|| the answer is 0, correctly.
TEST_F(TestBK, criterion_uniformly_dead_block_keeps_nothing) {
    std::vector<double> d(8, 1e-16);
    auto B = tri_block(d);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(8, B.data(), 8, 1.0, 1e-12), 0);
}

TEST_F(TestBK, criterion_finds_the_healthy_prefix) {
    std::vector<double> d = {1.0, 1.0, 1.0, 1e-16, 1e-16};
    auto B = tri_block(d);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(5, B.data(), 5, 1.0, 1e-12), 3);
}

// A small diagonal in the MIDDLE of an unpivoted block does NOT truncate, and that is the
// desired behaviour rather than a limitation.
//
// The criterion asks "is everything from column r onward negligible?", not "is column r
// healthy?". With a tiny entry at position 1 but full-size entries at 2 and 3, every
// trailing sub-block still carries content, so the answer is the full width.
//
// This matters because the factorization is unpivoted: neither geqrf nor CQRRT reorders
// columns, so a small interior diagonal genuinely does not imply the columns after it are
// junk. A per-column diagonal scan would truncate here and throw away two good directions.
// Testing the trailing BLOCK is what makes the unpivoted case safe, and it is why pivoting
// turned out to be a refinement rather than a prerequisite.
TEST_F(TestBK, criterion_does_not_truncate_on_an_interior_dip) {
    std::vector<double> d = {1.0, 1e-16, 1.0, 1.0};
    auto B = tri_block(d);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(4, B.data(), 4, 1.0, 1e-12), 4);

    // The tail genuinely going dead is still caught.
    std::vector<double> tail = {1.0, 1e-16, 1e-16, 1e-16};
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(4, tri_block(tail).data(), 4, 1.0, 1e-12), 1);
}

// THE test that would have caught the original bug on its own. Scaling a matrix does not
// change its rank, so the same block at 1e-8 and 1e+8 must give the same answer. An
// absolute sqrt(eps) threshold fails this outright.
TEST_F(TestBK, criterion_is_scale_invariant) {
    std::vector<double> d = {1.0, 1.0, 1.0, 1e-16, 1e-16};
    const double tau = 1e-12;
    int64_t ref = RandLAPACK::block_numerical_rank<double>(5, tri_block(d).data(), 5, 1.0, tau);
    for (double s : {1e-8, 1e-4, 1e+4, 1e+8}) {
        std::vector<double> ds(d.size());
        for (size_t i = 0; i < d.size(); ++i) ds[i] = d[i] * s;
        auto Bs = tri_block(ds);
        EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(5, Bs.data(), 5, s, tau), ref)
            << "scale " << s << " changed the rank decision";
    }
}

// Either side of the threshold, never the exact tie: tie behaviour of a floating comparison
// is a spec-lock that any FMA or vectorization change can flip, with no defect-finding power.
TEST_F(TestBK, criterion_brackets_the_threshold) {
    const double norm_A = 2.0, tau = 1e-10;
    const double thresh = tau * norm_A;
    std::vector<double> below = {1.0, thresh * 0.5};
    std::vector<double> above = {1.0, thresh * 2.0};
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(2, tri_block(below).data(), 2, norm_A, tau), 1);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(2, tri_block(above).data(), 2, norm_A, tau), 2);
}

// Degenerate ||A||. A zero matrix gives a zero threshold, so nothing is retained; that is
// safe only because termination no longer depends on this function.
TEST_F(TestBK, criterion_handles_degenerate_norms) {
    std::vector<double> d(4, 0.0);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(4, tri_block(d).data(), 4, 0.0, 1e-12), 0);
    std::vector<double> h(4, 1.0);
    EXPECT_EQ(RandLAPACK::block_numerical_rank<double>(4, tri_block(h).data(), 4, 0.0, 1e-12), 4);
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

// A zero matrix: norm_A == 0, so any threshold of the form tau*||A|| is zero and can never
// fire. Today the absolute sqrt(eps) test catches this by accident; after Phase 2 only the
// saturation guard will.
BK_LIVENESS_BODY(BK_terminates_on_zero_matrix, /* A stays all zeros */)

// The identity, padded to m x n: the Krylov space is span(Omega) and never grows.
BK_LIVENESS_BODY(BK_terminates_on_identity,
    for (int64_t i = 0; i < std::min(m, n); ++i) A[i + i * m] = 1.0;)

// Denormal scaling: ||A|| is representable but tau*||A|| underflows toward zero.
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
// Recorded now, deliberately, BEFORE Phase 3 adds replacement draws and an operator probe.
// Both consume RNG, which shifts every downstream stream; once that lands there is no
// baseline left to record and no way to tell an intended change from an accidental one.
//
// Resume equivalence is also the assertion that will fail loudly if a narrowed block ever
// reaches resume(): the resume path reconstructs its state as pure k-arithmetic
// (curr_X_cols = (1+iter_ev)*k, curr_Y_cols = iter_od*k), which is silently wrong once a
// block is not exactly k wide.
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

    check_band_identity<double>(m, n, A, out, 1e-10);

    delete[] A;
}

// Diagnostic for the T2 regime (exact rank 25, b_sz 10): BK-level view of why the run
// stops where it does, since the driver reports only "not_adaptive" in non-adaptive mode.
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
