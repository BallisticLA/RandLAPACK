#include "rl_lanczos_qfa.hh"
#include "rl_lanczos_qfa_block.hh"
#include "lanczos_test_support.hh"

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace linops = RandLAPACK::linops;

class TestLanczosOptimizations : public RandLAPACK::testing::LanczosTestSupport {};

// ===== First-row implicit QL evaluates the same quadrature as stevd =========
// use_first_row_ql swaps the certificate's eigensolver (full-eigenvector stevd)
// for implicit QL that rotates only the first eigenvector row. Nothing else
// changes, so on the same matrix and probes the two modes must certify the
// same columns at the same depths and return the same Gauss/Radau values to
// roundoff - in adaptive mode (checks along the whole ladder, then the at-cap
// re-check) and in fixed-depth mode (one evaluation at t = d). The matrix has
// a 1e8 spread of eigenvalues so the tridiagonals carry ghost clusters, the
// regime where a wrong eigensolver shows (stevr is off by 3-6x there).
TEST_F(TestLanczosOptimizations, ScalarQFAfirstRowQLmatchesStevd) {
    using T = double;
    const int64_t n = 400, s = 6;
    // A = Q diag(lambda) Q^T, lambda geometric over [1, spread], Q from QR of a Gaussian.
    auto build = [&](T spread, int seed) {
        T *A = new T[n * n];
        T *G = randn<T>(n, n, seed);
        T *tau = new T[n];
        lapack::geqrf(n, n, G, n, tau);
        lapack::orgqr(n, n, n, G, n, tau);
        T *Gs = new T[n * n];
        for (int64_t j = 0; j < n; ++j) {
            const T lj = std::pow(spread, (T)1 - (T)j / (T)n);
            for (int64_t i = 0; i < n; ++i) Gs[i + j * n] = G[i + j * n] * lj;
        }
        blas::gemm(Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   n, n, n, (T)1, Gs, n, G, n, (T)0, A, n);
        delete[] G; delete[] tau; delete[] Gs;
        return A;
    };
    T *A_mild = build((T)1e3, /*seed=*/89);   // certifies at heterogeneous depths
    T *A_hard = build((T)1e8, /*seed=*/91);   // ghost clusters; never certifies at n = 400
    linops::ExplicitSymLinOp<T> op_mild(n, blas::Uplo::Upper, A_mild, n, Layout::ColMajor);
    linops::ExplicitSymLinOp<T> op_hard(n, blas::Uplo::Upper, A_hard, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/97);
    auto fscalar = [](T x) { return std::log1p(x); };

    // Three regimes: adaptive on the mild matrix (every column certifies well
    // before the cap, on the ladder), adaptive on the hard matrix (every column
    // runs to the cap and the at-cap re-check fires), and fixed depth on the
    // hard matrix (one evaluation at t = d).
    struct Cfg { linops::ExplicitSymLinOp<T>* op; bool adaptive; int64_t d; bool expect_cert; };
    for (const Cfg cfg : {Cfg{&op_mild, true, n, true}, Cfg{&op_hard, true, n, false}, Cfg{&op_hard, false, 250, false}}) {
        const bool adaptive = cfg.adaptive;
        const int64_t d = cfg.d;
        RandLAPACK::LanczosQFA<T> ref, ql;
        ref.adaptive = ql.adaptive = adaptive;
        ref.adaptive_rtol = ql.adaptive_rtol = (T)1e-6;
        ql.use_first_row_ql = true;
        T *out_ref = new T[s]; T *out_ql = new T[s];
        ref.call(*cfg.op, Bmat, n, s, fscalar, d, out_ref);
        ql.call(*cfg.op, Bmat, n, s, fscalar, d, out_ql);
        EXPECT_EQ(ref.matvecs, ql.matvecs) << "adaptive=" << adaptive;
        EXPECT_EQ(ref.d_used, ql.d_used) << "adaptive=" << adaptive;
        EXPECT_EQ(ref.all_certified, ql.all_certified) << "adaptive=" << adaptive;
        T maxrel = 0;
        int64_t tmin = n, tmax = 0;
        for (int64_t j = 0; j < s; ++j) {
            EXPECT_EQ(ref.t_used[j], ql.t_used[j]) << "col " << j;
            EXPECT_EQ(ref.certified[j], ql.certified[j]) << "col " << j;
            tmin = std::min(tmin, ql.t_used[j]); tmax = std::max(tmax, ql.t_used[j]);
            maxrel = std::max(maxrel, std::abs(out_ref[j] - out_ql[j]) / std::abs(out_ref[j]));
            if (adaptive && ref.certified[j])
                maxrel = std::max(maxrel, std::abs(ref.radau_val[j] - ql.radau_val[j]) / std::abs(ref.radau_val[j]));
        }
        std::printf("first-row QL vs stevd (adaptive=%d, %s): t_used in [%ld, %ld] matvecs=%ld certified=%d  max rel diff=%.2e\n",
                    (int)adaptive, cfg.op == &op_mild ? "mild" : "hard", (long)tmin, (long)tmax,
                    (long)ql.matvecs, (int)ql.all_certified, maxrel);
        if (cfg.expect_cert) { EXPECT_TRUE(ql.all_certified); EXPECT_LT(tmax, d); }
        EXPECT_LT(maxrel, 1e-10);
        delete[] out_ref; delete[] out_ql;
    }
    delete[] A_mild; delete[] A_hard; delete[] Bmat;
}

// ===== OPT-1: banded eigensolve route vs dense (A/B equivalence, gate 2) ====
// The 2026-09-02 banded-eigensolver optimization plan's gate 2 (revised per
// its pass-2 numerics review): dual-evaluate BOTH routes on the IDENTICAL
// in-memory T_blk and compare the ASSEMBLED quantities only (M / tr_U /
// tr_L), never eigenvalues directly - Radau pins a node at ~0 (infinite
// relative error between two equally-correct routes there) and clustered
// Ritz weights are only meaningful cluster-summed. The plan's flat f-split
// numbers (1e-11 rel log1p, 1e-8 rel sqrt) hold for the GAUSS side
// (compute_M) at every spectrum tried here (measured 1e-13 to 1e-16, no
// scaling needed). The RADAU side (compute_M_radau) needs an extra
// condition-number scale on the "hard" (kappa=1e6) spectrum: sbtrd+stedc's
// similarity-transform path and syevd's direct path round the near-zero
// pinned corner eigenvalue differently, and sqrt's f'(0)=infinity amplifies
// that roundoff-level disagreement - measured up to ~1.4e-7 on hard/sqrt,
// which the flat 1e-8 does not cover. This is the SAME phenomenon (not a
// new one) BlockQFAradauS1MatchesScalar already needed 3e-8 (not the flat
// 1e-8) for, comparing two DIFFERENT algorithms (scalar vs block) on an
// easier spectrum - here it is two different algorithms (dense vs banded)
// on a harder one, so the scale factor is applied explicitly rather than
// silently baked into one extra tolerance constant: tol_radau = tol *
// max(1, sqrt(kappa)), kappa = 1 (easy, no scaling) or 1e6 (hard, matching
// its construction exactly) - a direct instance of the plan's own guidance
// to bound the pinned node via an ||T||/kappa-scaled quantity rather than a
// spectrum-blind flat relative number.
//
// "Identical in-memory T_blk": fa.run_lanczos() is called ONCE per config
// (its behavior never reads use_banded_eig - only reduce_fT_to_M does), then
// the SAME qfa instance's public compute_M/compute_M_radau (preserve_T=true,
// so T_blk is never touched) are called twice per ladder depth, flipping
// use_banded_eig between calls - not two separately-run recurrences that
// merely happen to agree.
TEST_F(TestLanczosOptimizations, BlockQFABandedVsDenseAssembledEquivalence) {
    using T = double;
    struct Cfg { const char* spectrum; const char* fname; int64_t s; };
    // 2 spectra x 2 f x 2 s = 8 configs (>= the plan's 6-config floor).
    const std::vector<Cfg> configs = {
        {"easy", "sqrt",  4}, {"easy", "sqrt",  16},
        {"easy", "log1p", 4}, {"easy", "log1p", 16},
        {"hard", "sqrt",  4}, {"hard", "sqrt",  16},
        {"hard", "log1p", 4}, {"hard", "log1p", 16},
    };
    const int64_t n = 640;   // headroom: d*s <= n for every (s, d) pair below

    for (const auto &cfg : configs) {
        const bool easy = (std::string(cfg.spectrum) == "easy");
        T *A = easy ? build_easy_psd<T>(n, 601) : build_hard_psd<T>(n, (T)1e6);
        linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
        T *Bmat = randn<T>(n, cfg.s, 617);
        const bool is_sqrt = (std::string(cfg.fname) == "sqrt");
        std::function<T(T)> fscalar = is_sqrt
            ? std::function<T(T)>([](T x) { return std::sqrt(std::max(x, (T)0)); })
            : std::function<T(T)>([](T x) { return std::log1p(std::max(x, (T)0)); });
        const T tol = is_sqrt ? (T)1e-8 : (T)1e-11;
        const T kappa = easy ? (T)1 : (T)1e6;   // hard's construction: kappa exactly
        const T tol_radau = tol * std::max((T)1, std::sqrt(kappa));

        RandLAPACK::BlockLanczosQFA<T> qfa;   // one shared recurrence + T_blk
        const int64_t d = 24;                 // d*16 = 384 <= 640; d*4 = 96 <= 640
        qfa.fa.run_lanczos(A_op, Bmat, n, cfg.s, d);

        T *M_dense  = new T[cfg.s * cfg.s];
        T *M_banded = new T[cfg.s * cfg.s];

        for (int64_t kdepth : {(int64_t)2, (int64_t)4, (int64_t)8, d}) {
            // ---- Gauss side (compute_M) ----
            qfa.use_banded_eig = false;
            qfa.compute_M(fscalar, cfg.s, kdepth, d, M_dense, /*preserve_T=*/true);
            qfa.use_banded_eig = true;
            qfa.compute_M(fscalar, cfg.s, kdepth, d, M_banded, /*preserve_T=*/true);

            T maxdiff = 0, scale = 0, trD = 0, trB = 0;
            for (int64_t e = 0; e < cfg.s * cfg.s; ++e) {
                maxdiff = std::max(maxdiff, std::abs(M_dense[e] - M_banded[e]));
                scale   = std::max(scale, std::abs(M_dense[e]));
            }
            for (int64_t i = 0; i < cfg.s; ++i) { trD += M_dense[i + i * cfg.s]; trB += M_banded[i + i * cfg.s]; }
            const T relmat = maxdiff / std::max(scale, std::numeric_limits<T>::min());
            const T reltr  = std::abs(trD - trB) / std::max(std::abs(trD), std::numeric_limits<T>::min());
            EXPECT_LT(relmat, tol) << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s
                                    << " kdepth=" << kdepth << " (Gauss, compute_M)";
            EXPECT_LT(reltr, tol)  << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s
                                    << " kdepth=" << kdepth << " (Gauss, compute_M)";

            // ---- Radau side (compute_M_radau) - only valid for kdepth >= 2 ----
            // (no guard needed here: the ladder above is {2, 4, 8, d}, so
            // kdepth is always >= 2 by construction, unlike a caller that
            // ran with d = 1. The banded route's kd = min(s, m-1) degenerate
            // clamp at kdepth = 1 is therefore still untested end-to-end by
            // this file - it is reachable only via a fixed-depth d = 1 call,
            // which is out of scope for this equivalence ladder.)
            // D_last at this exact kdepth: run a SEPARATE cap'd instance
            // sharing nothing with the dual-eval instance above (D_buf is
            // only ever valid at the depth update_pivot last advanced to;
            // reading it off the primary `qfa` here would require it to have
            // been advanced adaptively to exactly `kdepth`, which it was
            // not - it only ran the plain, non-adaptive recurrence). This
            // does not reintroduce the "two separate recurrences" concern
            // for the Radau comparison itself: both banded/dense calls below
            // still read the ONE T_blk this pivot-instance produced.
            RandLAPACK::BlockLanczosQFA<T> qfa_piv;
            qfa_piv.adaptive = true;
            qfa_piv.stop_rule = RandLAPACK::BlockQFAStop::Radau;
            qfa_piv.adaptive_rtol = std::numeric_limits<T>::min();
            // Zero rounded gap can satisfy any tolerance. Check only at the
            // requested depth so the pivot belongs to the matrix being tested.
            qfa_piv.check_every = kdepth;
            T *dummy = new T[cfg.s * cfg.s];
            qfa_piv.call(A_op, Bmat, n, cfg.s, fscalar, kdepth, dummy);
            ASSERT_EQ(qfa_piv.d_used, kdepth);
            delete[] dummy;

            T *ML_dense  = new T[cfg.s * cfg.s];
            T *ML_banded = new T[cfg.s * cfg.s];
            qfa_piv.use_banded_eig = false;
            qfa_piv.compute_M_radau(fscalar, cfg.s, kdepth, kdepth, qfa_piv.D_buf, ML_dense);
            qfa_piv.use_banded_eig = true;
            qfa_piv.compute_M_radau(fscalar, cfg.s, kdepth, kdepth, qfa_piv.D_buf, ML_banded);

            T maxdiffL = 0, scaleL = 0, trDL = 0, trBL = 0;
            for (int64_t e = 0; e < cfg.s * cfg.s; ++e) {
                maxdiffL = std::max(maxdiffL, std::abs(ML_dense[e] - ML_banded[e]));
                scaleL   = std::max(scaleL, std::abs(ML_dense[e]));
            }
            for (int64_t i = 0; i < cfg.s; ++i) { trDL += ML_dense[i + i * cfg.s]; trBL += ML_banded[i + i * cfg.s]; }
            const T relmatL = maxdiffL / std::max(scaleL, std::numeric_limits<T>::min());
            const T reltrL  = std::abs(trDL - trBL) / std::max(std::abs(trDL), std::numeric_limits<T>::min());
            std::printf("banded-vs-dense %s %s s=%ld kdepth=%ld: Gauss relmat=%.3e reltr=%.3e "
                        "Radau relmat=%.3e reltr=%.3e (tol_radau=%.3e)\n",
                        cfg.spectrum, cfg.fname, (long)cfg.s, (long)kdepth,
                        (double)relmat, (double)reltr, (double)relmatL, (double)reltrL, (double)tol_radau);
            EXPECT_LT(relmatL, tol_radau) << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s
                                     << " kdepth=" << kdepth << " (Radau, compute_M_radau)";
            EXPECT_LT(reltrL, tol_radau)  << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s
                                     << " kdepth=" << kdepth << " (Radau, compute_M_radau)";
            delete[] ML_dense; delete[] ML_banded;
        }
        delete[] M_dense; delete[] M_banded;
        delete[] A; delete[] Bmat;
    }
}

// ===== OPT-1: d_used / certified flip check, fixed AND certified, ladder ===
// Same >= 6 (here 8) configs as the assembled-equivalence test above, but at
// the full BlockLanczosQFA::call() level: two INDEPENDENT instances differing
// ONLY in use_banded_eig must agree EXACTLY on which depth a certified run
// stops at (d_used, certified) - the plan's "comparability regression" this
// gate exists to catch before the full qfa_micro census (gate 3) - and their
// traces must agree to the same f-split tolerance as above. Also covers the
// FIXED (non-adaptive, preserve_T=false) call path, which
// BlockQFABandedVsDenseAssembledEquivalence never exercises (it only calls
// compute_M/compute_M_radau with preserve_T=true).
TEST_F(TestLanczosOptimizations, BlockQFABandedVsDenseFixedAndCertifiedMatch) {
    using T = double;
    struct Cfg { const char* spectrum; const char* fname; int64_t s; };
    const std::vector<Cfg> configs = {
        {"easy", "sqrt",  4}, {"easy", "sqrt",  16},
        {"easy", "log1p", 4}, {"easy", "log1p", 16},
        {"hard", "sqrt",  4}, {"hard", "sqrt",  16},
        {"hard", "log1p", 4}, {"hard", "log1p", 16},
    };
    const int64_t n = 640;

    for (const auto &cfg : configs) {
        const bool easy = (std::string(cfg.spectrum) == "easy");
        T *A = easy ? build_easy_psd<T>(n, 701) : build_hard_psd<T>(n, (T)1e6);
        linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
        T *Bmat = randn<T>(n, cfg.s, 719);
        const bool is_sqrt = (std::string(cfg.fname) == "sqrt");
        std::function<T(T)> fscalar = is_sqrt
            ? std::function<T(T)>([](T x) { return std::sqrt(std::max(x, (T)0)); })
            : std::function<T(T)>([](T x) { return std::log1p(std::max(x, (T)0)); });
        const T tol = is_sqrt ? (T)1e-8 : (T)1e-11;
        // tr_L (Radau side) needs the same kappa-scaled softening as
        // BlockQFABandedVsDenseAssembledEquivalence, for the identical
        // reason (see that test's header comment) - tr_U (Gauss side) does
        // not.
        const T kappa = easy ? (T)1 : (T)1e6;
        const T tol_radau = tol * std::max((T)1, std::sqrt(kappa));
        const int64_t d_cap = 24;
        T *M_dense = new T[cfg.s * cfg.s];
        T *M_band  = new T[cfg.s * cfg.s];

        // ---- FIXED depth (non-adaptive; exercises preserve_T = false) ----
        for (int64_t depth : {(int64_t)3, (int64_t)8, d_cap}) {
            RandLAPACK::BlockLanczosQFA<T> q_dense; q_dense.use_banded_eig = false;
            RandLAPACK::BlockLanczosQFA<T> q_band;  q_band.use_banded_eig  = true;
            q_dense.call(A_op, Bmat, n, cfg.s, fscalar, depth, M_dense);
            q_band.call(A_op, Bmat, n, cfg.s, fscalar, depth, M_band);
            T trD = 0, trB = 0;
            for (int64_t i = 0; i < cfg.s; ++i) { trD += M_dense[i + i * cfg.s]; trB += M_band[i + i * cfg.s]; }
            const T reltr = std::abs(trD - trB) / std::max(std::abs(trD), std::numeric_limits<T>::min());
            EXPECT_LT(reltr, tol) << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s
                                   << " fixed depth=" << depth;
        }

        // ---- CERTIFIED (adaptive Radau): d_used/certified must NOT flip ----
        for (T eps : {(T)1e-2, (T)1e-6}) {
            RandLAPACK::BlockLanczosQFA<T> q_dense;
            q_dense.use_banded_eig = false;
            q_dense.adaptive = true; q_dense.stop_rule = RandLAPACK::BlockQFAStop::Radau;
            q_dense.adaptive_rtol = eps;
            RandLAPACK::BlockLanczosQFA<T> q_band;
            q_band.use_banded_eig = true;
            q_band.adaptive = true; q_band.stop_rule = RandLAPACK::BlockQFAStop::Radau;
            q_band.adaptive_rtol = eps;

            q_dense.call(A_op, Bmat, n, cfg.s, fscalar, d_cap, M_dense);
            q_band.call(A_op, Bmat, n, cfg.s, fscalar, d_cap, M_band);

            std::printf("fixed/cert flip-check %s %s s=%ld eps=%.0e: "
                        "d_used dense=%ld banded=%ld certified dense=%d banded=%d\n",
                        cfg.spectrum, cfg.fname, (long)cfg.s, (double)eps,
                        (long)q_dense.d_used, (long)q_band.d_used,
                        (int)q_dense.certified, (int)q_band.certified);
            EXPECT_EQ(q_dense.d_used, q_band.d_used)
                << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s << " eps=" << eps
                << " - d_used FLIPPED between dense and banded routes";
            EXPECT_EQ(q_dense.certified, q_band.certified)
                << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s << " eps=" << eps
                << " - certified FLIPPED between dense and banded routes";
            const T reltrU = std::abs(q_dense.tr_U - q_band.tr_U)
                              / std::max(std::abs(q_dense.tr_U), std::numeric_limits<T>::min());
            const T reltrL = std::abs(q_dense.tr_L - q_band.tr_L)
                              / std::max(std::abs(q_dense.tr_L), std::numeric_limits<T>::min());
            EXPECT_LT(reltrU, tol)       << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s << " eps=" << eps << " tr_U";
            EXPECT_LT(reltrL, tol_radau) << cfg.spectrum << " " << cfg.fname << " s=" << cfg.s << " eps=" << eps << " tr_L";
        }
        delete[] M_dense; delete[] M_band;
        delete[] A; delete[] Bmat;
    }
}

// ===== OPT-1: float32 instantiation, float-scaled tolerances ===============
TEST_F(TestLanczosOptimizations, FloatBlockQFABandedVsDenseEquivalence) {
    using T = float;
    const int64_t n = 500, s = 8, d_cap = 20;   // d*s = 160 <= n
    T *G0 = randn<T>(n, n, /*seed=*/821);
    T *A  = new T[n * n];
    blas::syrk(Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
               n, n, (T)1, G0, n, (T)0, A, n);
    for (int64_t i = 0; i < n; ++i) A[i + i * n] += (T)n;
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = j + 1; i < n; ++i) A[i + j * n] = A[j + i * n];
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/823);
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    RandLAPACK::BlockLanczosQFA<T> q_dense;
    q_dense.use_banded_eig = false;
    q_dense.adaptive = true; q_dense.stop_rule = RandLAPACK::BlockQFAStop::Radau;
    q_dense.adaptive_rtol = (T)1e-3;
    RandLAPACK::BlockLanczosQFA<T> q_band;
    q_band.use_banded_eig = true;
    q_band.adaptive = true; q_band.stop_rule = RandLAPACK::BlockQFAStop::Radau;
    q_band.adaptive_rtol = (T)1e-3;

    T *M_dense = new T[s * s], *M_band = new T[s * s];
    q_dense.call(A_op, Bmat, n, s, fscalar, d_cap, M_dense);
    q_band.call(A_op, Bmat, n, s, fscalar, d_cap, M_band);

    std::printf("f32 banded-vs-dense: d_used dense=%ld banded=%ld certified dense=%d banded=%d "
                "tr_U dense=%.6e banded=%.6e\n",
                (long)q_dense.d_used, (long)q_band.d_used,
                (int)q_dense.certified, (int)q_band.certified,
                (double)q_dense.tr_U, (double)q_band.tr_U);
    // Exact EXPECT_EQ (not a tolerance) is intentional: this is the census
    // d_used/certified identity itself, not a numeric comparison, so it must
    // stay exact even though it is the most backend-sensitive assertion in
    // this file (float bracket margins near a ladder depth can straddle the
    // rtol boundary on a non-MKL BLAS).
    EXPECT_EQ(q_dense.d_used, q_band.d_used);
    EXPECT_EQ(q_dense.certified, q_band.certified);
    const T reltrU = std::abs(q_dense.tr_U - q_band.tr_U)
                      / std::max(std::abs(q_dense.tr_U), std::numeric_limits<T>::min());
    EXPECT_TRUE(std::isfinite(reltrU));
    EXPECT_LT(reltrU, (T)1e-4);   // float-scaled analog of the sqrt-family bound above
    delete[] G0; delete[] A; delete[] Bmat; delete[] M_dense; delete[] M_band;
}

// ===== OPT-1: clamp preservation - negative pinned Ritz through the banded
// route =======================================================================
// The Radau construction pins quadrature nodes at 0 +/- roundoff; with f =
// sqrt a node that lands SLIGHTLY NEGATIVE must still be clamped to 0 before
// f is applied (reduce_fT_to_M's assemble_M_from_eig: f(max(eig,0))) or sqrt
// produces NaN. This must hold identically through the banded route (the
// clamp lives in the SHARED assembly tail, but the banded route's eigenvalue
// ORDERING/sign near the pinned node comes from a different algorithm --
// stedc on a similarity-transformed tridiagonal, not syevd directly on V --
// so this is not automatically free). s = 1 diagonal spectrum reliably
// produces a near-zero pinned node (same construction as
// BlockQFAradauS1MatchesScalar, whose own doc explains why sqrt's node is
// only reproducible to ~sqrt(roundoff) across algebraically equivalent
// corner computations).
TEST_F(TestLanczosOptimizations, BlockQFABandedRouteClampsNegativePinnedRitz) {
    using T = double;
    const int64_t n = 80;
    T *A = new T[n * n]();
    for (int64_t i = 0; i < n; ++i) A[i + i * n] = (T)(i + 1);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *b = randn<T>(n, 1, /*seed=*/827);
    auto f_sqrt = [](T x) { return std::sqrt(x); };

    for (int64_t depth : {4, 8, 16}) {
        RandLAPACK::BlockLanczosQFA<T> q_dense;
        q_dense.use_banded_eig = false;
        q_dense.adaptive = true; q_dense.stop_rule = RandLAPACK::BlockQFAStop::Radau;
        q_dense.adaptive_rtol = std::numeric_limits<T>::min();   // never fires
        RandLAPACK::BlockLanczosQFA<T> q_band;
        q_band.use_banded_eig = true;
        q_band.adaptive = true; q_band.stop_rule = RandLAPACK::BlockQFAStop::Radau;
        q_band.adaptive_rtol = std::numeric_limits<T>::min();

        T M_dense, M_band;
        q_dense.call(A_op, b, n, 1, f_sqrt, depth, &M_dense);
        q_band.call(A_op, b, n, 1, f_sqrt, depth, &M_band);

        ASSERT_TRUE(std::isfinite(M_dense)) << "dense route produced non-finite M at depth " << depth;
        ASSERT_TRUE(std::isfinite(M_band))  << "banded route produced non-finite M at depth " << depth
                                             << " - clamp not preserved through sbtrd/stedc";
        ASSERT_TRUE(std::isfinite(q_dense.tr_L));
        ASSERT_TRUE(std::isfinite(q_band.tr_L)) << "banded tr_L non-finite at depth " << depth;

        const T reltrL = std::abs(q_dense.tr_L - q_band.tr_L)
                          / std::max(std::abs(q_dense.tr_L), std::numeric_limits<T>::min());
        std::printf("clamp-preservation sqrt s=1 depth=%ld: tr_L dense=%.10e banded=%.10e reltrL=%.3e\n",
                    (long)depth, q_dense.tr_L, q_band.tr_L, (double)reltrL);
        // Same ~sqrt(roundoff) bound BlockQFAradauS1MatchesScalar documents
        // for the pinned-node sqrt family (f'(0) = infinity amplifies a
        // roundoff-level shift in the near-zero Ritz value by ~sqrt(shift)).
        EXPECT_LT(reltrL, 3e-8) << "depth " << depth;
    }
    delete[] A; delete[] b;
}

// ===== OPT-1: Window rule coverage (banded route reached through the legacy
// heuristic's compute_M call, not just the Radau certificate) ===============
TEST_F(TestLanczosOptimizations, BlockQFABandedVsDenseWindowRule) {
    using T = double;
    const int64_t n = 80, s = 6, d_max = 70;
    T *A = build_easy_psd<T>(n, 831);
    linops::ExplicitSymLinOp<T> A_op(n, blas::Uplo::Upper, A, n, Layout::ColMajor);
    T *Bmat = randn<T>(n, s, /*seed=*/833);
    auto fscalar = [](T x) { return std::sqrt(x); };

    RandLAPACK::BlockLanczosQFA<T> q_dense;
    q_dense.use_banded_eig = false;
    q_dense.adaptive = true; q_dense.stop_rule = RandLAPACK::BlockQFAStop::Window;
    q_dense.adaptive_rtol = (T)1e-3;
    RandLAPACK::BlockLanczosQFA<T> q_band;
    q_band.use_banded_eig = true;
    q_band.adaptive = true; q_band.stop_rule = RandLAPACK::BlockQFAStop::Window;
    q_band.adaptive_rtol = (T)1e-3;

    T *M_dense = new T[s * s], *M_band = new T[s * s];
    q_dense.call(A_op, Bmat, n, s, fscalar, d_max, M_dense);
    q_band.call(A_op, Bmat, n, s, fscalar, d_max, M_band);

    std::printf("Window rule banded-vs-dense: d_used dense=%ld banded=%ld\n",
                (long)q_dense.d_used, (long)q_band.d_used);
    EXPECT_EQ(q_dense.d_used, q_band.d_used)
        << "Window rule d_used flip between dense and banded eigensolve routes";
    T trD = 0, trB = 0;
    for (int64_t i = 0; i < s; ++i) { trD += M_dense[i + i * s]; trB += M_band[i + i * s]; }
    const T reltr = std::abs(trD - trB) / std::max(std::abs(trD), std::numeric_limits<T>::min());
    EXPECT_LT(reltr, 1e-8);
    delete[] A; delete[] Bmat; delete[] M_dense; delete[] M_band;
}

// ===== OPT-2: reorth kernel accuracy A/B - CGS2 vs block-MGS ===============
// Plan gate 2 (2026-09-02-blockqfa-optimization-plan.md): CGS2
// (use_cgs2_reorth) CHANGES ROUNDOFF relative to block-MGS (not
// bit-identical), so acceptance is accuracy-EQUIVALENCE, not bit-identity.
// The rl_lanczos_fa_block.hh history warning (~:248-251, quoted in the class
// doc comment above use_cgs2_reorth) records that a SINGLE-pass block-CGS was
// already tried in this exact file and reverted for catastrophic
// orthogonality loss (||Q'Q-I|| ~ 1) - CGS2 (two passes) is the textbook fix,
// but that history makes this accuracy gate load-bearing, not a formality.
//
// Method: for >= 6 configs (easy/hard spectra x s in {2,4,8}), sweep depth
// AT and PAST the cliff (n chosen small enough that d*s reaches and exceeds
// n within the depth list - the plan's "d*s near and beyond n at small n").
// At each depth, run 8 trials (independent random B seeds) through BOTH
// kernels (paired per trial) and compare relative error against the exact
// f(A)B oracle (make_exact_fa_oracle). Gate: at every depth, (a) every new
// (CGS2) trial's error stays within the old (MGS) 8-trial envelope (with a
// 2x safety margin, exempting both sides once they are below a roundoff
// floor where relative comparisons are noise-dominated), and (b) the
// two-sided geomean(new/old) error ratio over the 8 trials is <= 2.
TEST_F(TestLanczosOptimizations, BlockFACGS2VsMGSAccuracyEnvelope) {
    using T = double;
    struct Cfg { const char* label; bool hard; int64_t n; int64_t s; std::vector<int64_t> depths; };
    // n/s chosen so the depth list runs up to and past the cliff (d*s >= n).
    const std::vector<Cfg> configs = {
        {"easy_s4", false, 60, 4, {4, 8, 12, 14, 15, 16, 18}},   // n/s = 15
        {"hard_s4", true,  60, 4, {4, 8, 12, 14, 15, 16, 18}},
        {"easy_s8", false, 96, 8, {4, 8, 10, 11, 12, 14}},       // n/s = 12
        {"hard_s8", true,  96, 8, {4, 8, 10, 11, 12, 14}},
        {"easy_s2", false, 40, 2, {6, 12, 18, 20, 22, 24}},      // n/s = 20
        {"hard_s2", true,  40, 2, {6, 12, 18, 20, 22, 24}},
    };
    const int NTRIALS = 8;
    const T roundoff_floor = (T)1e-13;
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    for (const auto &cfg : configs) {
        T *A = cfg.hard ? build_hard_psd<T>(cfg.n, (T)1e6) : build_easy_psd<T>(cfg.n, 941);
        linops::ExplicitSymLinOp<T> A_op(cfg.n, blas::Uplo::Upper, A, cfg.n, Layout::ColMajor);
        auto fAfun = RandLAPACK::testing::make_exact_fa_oracle<T>(cfg.n, A, fscalar);

        for (int64_t depth : cfg.depths) {
            std::vector<T> old_err(NTRIALS), new_err(NTRIALS);
            for (int t = 0; t < NTRIALS; ++t) {
                T *Bmat = randn<T>(cfg.n, cfg.s, /*seed=*/(uint32_t)(90000 + 1000 * depth + t));
                T *target = new T[cfg.n * cfg.s];
                fAfun(cfg.n, cfg.s, Bmat, target);
                const T tnorm = blas::nrm2(cfg.n * cfg.s, target, 1);

                RandLAPACK::BlockLanczosFA<T> fa_mgs;
                fa_mgs.reorth = true; fa_mgs.use_cgs2_reorth = false;
                T *out_mgs = new T[cfg.n * cfg.s];
                fa_mgs.call(A_op, Bmat, cfg.n, cfg.s, fscalar, depth, out_mgs);

                RandLAPACK::BlockLanczosFA<T> fa_cgs2;
                fa_cgs2.reorth = true; fa_cgs2.use_cgs2_reorth = true;
                T *out_cgs2 = new T[cfg.n * cfg.s];
                fa_cgs2.call(A_op, Bmat, cfg.n, cfg.s, fscalar, depth, out_cgs2);

                T diff_mgs = 0, diff_cgs2 = 0;
                for (int64_t e = 0; e < cfg.n * cfg.s; ++e) {
                    const T dm = out_mgs[e]  - target[e]; diff_mgs  += dm * dm;
                    const T dc = out_cgs2[e] - target[e]; diff_cgs2 += dc * dc;
                }
                old_err[t] = std::sqrt(diff_mgs)  / std::max(tnorm, std::numeric_limits<T>::min());
                new_err[t] = std::sqrt(diff_cgs2) / std::max(tnorm, std::numeric_limits<T>::min());

                delete[] Bmat; delete[] target; delete[] out_mgs; delete[] out_cgs2;
            }

            const T old_max = *std::max_element(old_err.begin(), old_err.end());
            for (int t = 0; t < NTRIALS; ++t) {
                // Roundoff-floor exempt: once EITHER side of a pair is
                // indistinguishable from roundoff noise, a further ratio
                // between them is not a meaningful accuracy comparison (one
                // near-zero value against a tiny-but-nonzero one can swing
                // the ratio arbitrarily without either route actually being
                // worse) - exempt on either side dipping below the floor,
                // not only when both do.
                if (new_err[t] < roundoff_floor || old_max < roundoff_floor) continue;
                EXPECT_LT(new_err[t], std::max(old_max * (T)2, roundoff_floor))
                    << cfg.label << " depth=" << depth << " trial=" << t
                    << " CGS2 error exceeds the MGS 8-trial envelope (old_max=" << old_max << ")";
            }
            double log_ratio_sum = 0; int n_counted = 0;
            for (int t = 0; t < NTRIALS; ++t) {
                if (old_err[t] < roundoff_floor || new_err[t] < roundoff_floor) continue;
                log_ratio_sum += std::log((double)new_err[t] / std::max((double)old_err[t], 1e-300));
                ++n_counted;
            }
            if (n_counted > 0) {
                const double geomean_ratio = std::exp(log_ratio_sum / n_counted);
                std::printf("CGS2-vs-MGS %s depth=%ld d*s=%ld/n=%ld: old_max=%.3e geomean_ratio(new/old)=%.3f\n",
                            cfg.label, (long)depth, (long)(depth * cfg.s), (long)cfg.n,
                            (double)old_max, geomean_ratio);
                EXPECT_LT(geomean_ratio, 2.0) << cfg.label << " depth=" << depth << " (ratio too high)";
                EXPECT_GT(geomean_ratio, 0.5) << cfg.label << " depth=" << depth << " (ratio too low, two-sided)";
            }
        }
        delete[] A;
    }
}

// ===== OPT-2: direct ||I - Q'Q|| orthogonality probe, CGS2 vs MGS, AT/PAST
// the cliff ====================================================================
// Plan gate 2's non-negotiable guard (per the reverted-single-pass-CGS
// history warning): compare the CGS2 basis's departure from orthogonality
// against the MGS basis's, at depths at and past the d*s >= n cliff, and
// require them to stay the SAME ORDER OF MAGNITUDE. This is the direct
// structural check the accuracy-envelope test above cannot see on its own
// (a badly non-orthogonal basis can still, by cancellation, produce a
// deceptively small f(A)B error at one particular depth).
TEST_F(TestLanczosOptimizations, BlockFACGS2OrthogonalityAtCliff) {
    using T = double;
    struct Cfg { const char* label; int64_t n; int64_t s; int64_t depth; };
    const std::vector<Cfg> configs = {
        {"n60_s4_d15_atcliff",   60, 4, 15},   // d*s == n
        {"n60_s4_d18_pastcliff", 60, 4, 18},   // d*s > n
        {"n96_s8_d12_atcliff",   96, 8, 12},
        {"n96_s8_d14_pastcliff", 96, 8, 14},
        {"n40_s2_d20_atcliff",   40, 2, 20},
        {"n40_s2_d24_pastcliff", 40, 2, 24},
    };
    for (const auto &cfg : configs) {
        T *A = build_easy_psd<T>(cfg.n, 951);
        linops::ExplicitSymLinOp<T> A_op(cfg.n, blas::Uplo::Upper, A, cfg.n, Layout::ColMajor);
        T *Bmat = randn<T>(cfg.n, cfg.s, /*seed=*/953);

        RandLAPACK::BlockLanczosFA<T> fa_mgs;
        fa_mgs.reorth = true; fa_mgs.use_cgs2_reorth = false;
        fa_mgs.run_lanczos(A_op, Bmat, cfg.n, cfg.s, cfg.depth);
        RandLAPACK::BlockLanczosFA<T> fa_cgs2;
        fa_cgs2.reorth = true; fa_cgs2.use_cgs2_reorth = true;
        fa_cgs2.run_lanczos(A_op, Bmat, cfg.n, cfg.s, cfg.depth);

        auto orth_err = [&](RandLAPACK::BlockLanczosFA<T>& fa) -> T {
            const int64_t m = fa.steps_run * cfg.s;   // Q = K_big[0..m-1], n x m, ld = n
            T *G = new T[m * m];
            blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, m, m, cfg.n,
                       (T)1, fa.K_big, cfg.n, fa.K_big, cfg.n, (T)0, G, m);
            for (int64_t i = 0; i < m; ++i) G[i + i * m] -= (T)1;
            const T fro = blas::nrm2(m * m, G, 1);
            delete[] G;
            return fro;
        };
        const T err_mgs  = orth_err(fa_mgs);
        const T err_cgs2 = orth_err(fa_cgs2);
        std::printf("orth-probe %s: ||I-Q'Q||_F  MGS=%.3e (m=%ld)  CGS2=%.3e (m=%ld)\n",
                    cfg.label, (double)err_mgs, (long)(fa_mgs.steps_run * cfg.s),
                    (double)err_cgs2, (long)(fa_cgs2.steps_run * cfg.s));
        // "Same order of magnitude": guard against the reverted single-pass
        // failure mode (||Q'Q-I|| ~ 1 there, vs a well-behaved MGS/CGS2
        // basis typically at or near roundoff-times-condition scale). A
        // factor-of-10 band around the MGS reference (floored at 1e-10 so a
        // near-zero MGS reference cannot make the bound vacuously tight)
        // catches an order-of-magnitude blowup while tolerating CGS2's
        // expected (not necessarily smaller) roundoff.
        EXPECT_LT(err_cgs2, std::max(err_mgs * (T)10, (T)1e-10))
            << cfg.label << ": CGS2 orthogonality loss is not the same order of magnitude as MGS's";
        delete[] A; delete[] Bmat;
    }
}

// ===== OPT-2: float32 instantiation, float-scaled tolerances ===============
// Mirrors OPT-1's FloatBlockQFABandedVsDenseEquivalence pattern (a single
// float32 A/B test with tolerances widened by roughly the double-to-float
// roundoff-floor ratio, ~1e4, rather than a full float re-sweep of the
// double-precision grid above). Combines both of this file's CGS2 gates
// (assembled-accuracy envelope + the non-negotiable ||I-Q'Q|| orthogonality
// probe) into one test at AND past the d*s >= n cliff, since both gates are
// cheap at these small float sizes and the point is coverage, not volume.
TEST_F(TestLanczosOptimizations, FloatBlockFACGS2OrthogonalityAndAccuracy) {
    using T = float;
    struct Cfg { const char* label; int64_t n; int64_t s; int64_t d_cliff; int64_t d_past; };
    const std::vector<Cfg> configs = {
        {"f32_n60_s4",  60, 4, 15, 18},   // d*s = 60 (== n), 72 (> n)
        {"f32_n40_s2",  40, 2, 20, 24},   // d*s = 40 (== n), 48 (> n)
    };
    const int NTRIALS = 4;
    // Double used a 1e-13 roundoff floor / 1e-10 orthogonality floor; float's
    // machine epsilon is ~1e4x coarser (1.19e-7 vs 2.2e-16), so both floors
    // are widened by the same ~1e4 factor, matching OPT-1's float-scaled
    // pattern (1e-8 double -> 1e-4 float in FloatBlockQFABandedVsDenseEquivalence).
    const T roundoff_floor = (T)1e-9;
    const T orth_floor     = (T)1e-6;
    auto fscalar = [](T x) { return std::sqrt(std::max(x, (T)0)); };

    for (const auto &cfg : configs) {
        T *A = build_easy_psd<T>(cfg.n, 977);
        linops::ExplicitSymLinOp<T> A_op(cfg.n, blas::Uplo::Upper, A, cfg.n, Layout::ColMajor);
        auto fAfun = RandLAPACK::testing::make_exact_fa_oracle<T>(cfg.n, A, fscalar);

        for (int64_t depth : {cfg.d_cliff, cfg.d_past}) {
            // --- Orthogonality probe (direct ||I - Q'Q||, no oracle needed) ---
            T *Bmat_o = randn<T>(cfg.n, cfg.s, /*seed=*/(uint32_t)(978 + depth));
            RandLAPACK::BlockLanczosFA<T> fa_mgs_o;
            fa_mgs_o.reorth = true; fa_mgs_o.use_cgs2_reorth = false;
            fa_mgs_o.run_lanczos(A_op, Bmat_o, cfg.n, cfg.s, depth);
            RandLAPACK::BlockLanczosFA<T> fa_cgs2_o;
            fa_cgs2_o.reorth = true; fa_cgs2_o.use_cgs2_reorth = true;
            fa_cgs2_o.run_lanczos(A_op, Bmat_o, cfg.n, cfg.s, depth);

            auto orth_err = [&](RandLAPACK::BlockLanczosFA<T>& fa) -> T {
                const int64_t m = fa.steps_run * cfg.s;
                T *G = new T[m * m];
                blas::gemm(Layout::ColMajor, Op::Trans, Op::NoTrans, m, m, cfg.n,
                           (T)1, fa.K_big, cfg.n, fa.K_big, cfg.n, (T)0, G, m);
                for (int64_t i = 0; i < m; ++i) G[i + i * m] -= (T)1;
                const T fro = blas::nrm2(m * m, G, 1);
                delete[] G;
                return fro;
            };
            const T err_mgs_o  = orth_err(fa_mgs_o);
            const T err_cgs2_o = orth_err(fa_cgs2_o);
            std::printf("f32 orth-probe %s depth=%ld: MGS=%.3e CGS2=%.3e\n",
                        cfg.label, (long)depth, (double)err_mgs_o, (double)err_cgs2_o);
            EXPECT_LT(err_cgs2_o, std::max(err_mgs_o * (T)10, orth_floor))
                << cfg.label << " depth=" << depth << ": f32 CGS2 orthogonality loss not "
                << "the same order of magnitude as MGS's";
            delete[] Bmat_o;

            // --- Assembled-accuracy envelope (paired trials vs exact oracle) ---
            std::vector<T> old_err(NTRIALS), new_err(NTRIALS);
            for (int t = 0; t < NTRIALS; ++t) {
                T *Bmat = randn<T>(cfg.n, cfg.s, /*seed=*/(uint32_t)(90500 + 1000 * depth + t));
                T *target = new T[cfg.n * cfg.s];
                fAfun(cfg.n, cfg.s, Bmat, target);
                const T tnorm = blas::nrm2(cfg.n * cfg.s, target, 1);

                RandLAPACK::BlockLanczosFA<T> fa_mgs;
                fa_mgs.reorth = true; fa_mgs.use_cgs2_reorth = false;
                T *out_mgs = new T[cfg.n * cfg.s];
                fa_mgs.call(A_op, Bmat, cfg.n, cfg.s, fscalar, depth, out_mgs);

                RandLAPACK::BlockLanczosFA<T> fa_cgs2;
                fa_cgs2.reorth = true; fa_cgs2.use_cgs2_reorth = true;
                T *out_cgs2 = new T[cfg.n * cfg.s];
                fa_cgs2.call(A_op, Bmat, cfg.n, cfg.s, fscalar, depth, out_cgs2);

                T diff_mgs = 0, diff_cgs2 = 0;
                for (int64_t e = 0; e < cfg.n * cfg.s; ++e) {
                    const T dm = out_mgs[e]  - target[e]; diff_mgs  += dm * dm;
                    const T dc = out_cgs2[e] - target[e]; diff_cgs2 += dc * dc;
                }
                old_err[t] = std::sqrt(diff_mgs)  / std::max(tnorm, std::numeric_limits<T>::min());
                new_err[t] = std::sqrt(diff_cgs2) / std::max(tnorm, std::numeric_limits<T>::min());

                delete[] Bmat; delete[] target; delete[] out_mgs; delete[] out_cgs2;
            }
            const T old_max = *std::max_element(old_err.begin(), old_err.end());
            for (int t = 0; t < NTRIALS; ++t) {
                if (new_err[t] < roundoff_floor || old_max < roundoff_floor) continue;
                EXPECT_LT(new_err[t], std::max(old_max * (T)2, roundoff_floor))
                    << cfg.label << " depth=" << depth << " trial=" << t
                    << " f32 CGS2 error exceeds the MGS trial envelope (old_max=" << old_max << ")";
            }
        }
        delete[] A;
    }
}

TEST_F(TestLanczosOptimizations, ScalarQFAFirstRowQLRetainsFloatSolver) {
    using T = float;
    const int64_t n = 128, s = 4;
    std::vector<T> lambda(n), reference(s), candidate(s);
    for (int64_t i = 0; i < n; ++i)
        lambda[i] = i < 8 ? T(1) : T(1e-8);
    linops::DiagSymLinOp<T> op(n, lambda.data());
    T* B = randn<T>(n, s, 346001);
    RandLAPACK::LanczosQFA<T> ref, ql;
    ql.use_first_row_ql = true;
    for (bool adaptive : {false, true}) {
        ref.adaptive = ql.adaptive = adaptive;
        auto f = [](T x) { return std::sqrt(x); };
        ref.call(op, B, n, s, f, 64, reference.data());
        ql.call(op, B, n, s, f, 64, candidate.data());
        EXPECT_TRUE(ql.first_row_ql_fallback);
        EXPECT_EQ(ref.matvecs, ql.matvecs);
        for (int64_t j = 0; j < s; ++j) {
            EXPECT_EQ(reference[j], candidate[j]);
            EXPECT_EQ(ref.t_used[j], ql.t_used[j]);
            EXPECT_EQ(ref.certified[j], ql.certified[j]);
        }
    }
    delete[] B;
}

TEST_F(TestLanczosOptimizations, ScalarQFAFirstRowQLPropagatesParallelEvaluationException) {
    const int64_t n = 64, s = 4;
    std::vector<double> lambda(n), out(s);
    for (int64_t i = 0; i < n; ++i) lambda[i] = 1. + i;
    linops::DiagSymLinOp<double> op(n, lambda.data());
    double* B = randn<double>(n, s, 346002);
    RandLAPACK::LanczosQFA<double> q;
    for (bool adaptive : {false, true}) {
        q.adaptive = adaptive; q.use_first_row_ql = true;
        auto fails = [](double) -> double { throw std::runtime_error("fixture: failed evaluation"); };
        EXPECT_THROW(q.call(op, B, n, s, fails, 8, out.data()), std::runtime_error);
        EXPECT_LE(q.matvecs, s * 8);
        auto f = [](double x) { return std::log1p(x); };
        EXPECT_NO_THROW(q.call(op, B, n, s, f, 8, out.data()));
        for (double x : out) EXPECT_TRUE(std::isfinite(x));
    }
    delete[] B;
}

// Preserve the optional-QL branch of the scaling regression.
TEST_F(TestLanczosOptimizations, FirstRowQLTinyPsdOperatorDoesNotCertifyFalseBreakdown) {
    const double eigenvalues[] = {1e-200, 4e-200};
    const double b[] = {1.0, 1.0};
    RandLAPACK::linops::DiagSymLinOp<double> A(2, eigenvalues);
    RandLAPACK::LanczosQFA<double> qfa;
    qfa.adaptive = true;
    qfa.adaptive_rtol = 1e-6;
    qfa.use_first_row_ql = true;
    double out = 0.0;
    qfa.call(A, b, 2, 1, [](double x) { return std::sqrt(x); }, 2, &out);

    // The two-node Gauss rule is exact: sqrt(1e-200) + sqrt(4e-200).
    EXPECT_EQ(qfa.d_used, 2);
    EXPECT_NEAR(out / 3e-100, 1.0, 1e-12);
    // T_Radau / 1e-200 = [[2.5, 1.5], [1.5, 0.9]], with nodes 0, 3.4.
    const double radau = (5.0 / std::sqrt(3.4)) * 1e-100;
    EXPECT_NEAR(qfa.radau_val[0] / radau, 1.0, 1e-7);
    EXPECT_FALSE(qfa.all_certified);
}

// Preserve the optional-QL branch of the scaling regression.
TEST_F(TestLanczosOptimizations, FirstRowQLAdaptiveLargeInputKeepsFiniteQuadraticForm) {
    const double eigenvalues[] = {1e-40, 4e-40, 9e-40};
    const double b[] = {1e155, 1e155, 1e155};
    RandLAPACK::linops::DiagSymLinOp<double> A(3, eigenvalues);
    RandLAPACK::LanczosQFA<double> qfa;
    qfa.adaptive = true;
    qfa.use_first_row_ql = true;
    double out = 0.0;
    qfa.call(A, b, 3, 1, [](double x) { return std::sqrt(x); }, 2, &out);

    // Independently evaluated two-node Gauss value; the exact form is 6e290.
    ASSERT_TRUE(std::isfinite(out));
    EXPECT_NEAR(out / 6.0501329353679946e290, 1.0, 1e-12);
    EXPECT_TRUE(std::isfinite(qfa.radau_val[0]));
    EXPECT_LE(qfa.radau_val[0], 6e290);
    EXPECT_GE(out, 6e290);
}

} // namespace
