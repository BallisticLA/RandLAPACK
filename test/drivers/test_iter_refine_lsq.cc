// Tests for RandLAPACK::IterRefineLSQ, the member-configured adapter over restarted_pcg_ne.
// A small dense tall A is wrapped as a DenseLinOp, R comes from the QR of (a perturbation of)
// A, and solutions are checked against lapack::gels. The engine's own exits are tested in
// comps/test_restarted_pcg_ne.cc.

#include <RandBLAS.hh>
#include <RandLAPACK.hh>
#include <gtest/gtest.h>

#include "../lsq_test_helpers.hh"

#include <algorithm>
#include <utility>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <vector>


using RandLAPACK::IterRefineLSQ;
using RandLAPACK::linops::DenseLinOp;
using RandLAPACK::PCGInnerTermination;
using Term = RandLAPACK::RestartedPCGNETermination;
using blas::Layout;
using lsq_test::fill_random;
using lsq_test::consistent_problem;
using lsq_test::gels_reference;
using lsq_test::rel_err;


class TestIterRefineLSQ : public ::testing::Test {};


// Well-conditioned problem with the exact R: M is numerically the identity, so CG converges
// in about one iteration and the solution matches gels.
TEST_F(TestIterRefineLSQ, dense_well_conditioned) {
    using T = double;
    auto p = consistent_problem<T>(80, 12, 42, 99);
    auto x_ref = gels_reference(p.A, p.b, p.m, p.n);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*abs_tol=*/1e-12, /*max_inner=*/50, /*n_steps=*/2);
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-10);
    EXPECT_LE(ir.report.cg_iters.front(), 5);
}


// Least squares with a real residual (b not in range(A)).
TEST_F(TestIterRefineLSQ, dense_with_residual) {
    using T = double;
    auto p = consistent_problem<T>(100, 7, 7, 17);
    std::vector<T> noise(p.m);
    fill_random(noise, 27, 0.05);
    for (int64_t i = 0; i < p.m; ++i) p.b[i] += noise[i];
    auto x_ref = gels_reference(p.A, p.b, p.m, p.n);

    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 50, 2);
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-10);
}


// An imperfect R (QR of a perturbed A) needs more rounds; outer_tol stops the loop once the
// solution matches gels.
TEST_F(TestIterRefineLSQ, imperfect_preconditioner) {
    using T = double;
    auto p = consistent_problem<T>(60, 10, 3, 13, 99, (T)0.1);
    auto x_ref = gels_reference(p.A, p.b, p.m, p.n);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 100, /*n_steps=*/20);
    ir.outer_tol = (T)1e-12;
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_LT(rel_err(x, x_ref), 1e-9);
}


// A capped inner solve is distinguishable from a converged one: a cap of 2 against an
// unreachable round target forces hit_cap, then a generous budget reports converged. The
// round target is a 1e-14 drop with no absolute floor.
TEST_F(TestIterRefineLSQ, capped_solve_is_reported) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 4242, 777, 31337, (T)0.30);   // weak R
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    {
        IterRefineLSQ<T> ir(/*abs_tol=*/0, /*max_inner=*/2, /*n_steps=*/2);
        ir.round_drop = (T)1e-14;
        std::vector<T> x(p.n, 0);
        EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);   // a cap is not an error
        ASSERT_EQ(ir.report.cg_status.size(), 2u);
        EXPECT_EQ(ir.report.cg_status.front(), PCGInnerTermination::hit_cap);
        EXPECT_EQ(ir.report.cg_iters.front(), 2);
        EXPECT_GT(ir.report.cg_relres.front(), 1e-14);
        EXPECT_EQ(ir.report.cg_best_iter.front(), 2);   // still descending at the cap
    }
    {
        IterRefineLSQ<T> ir(/*abs_tol=*/0, /*max_inner=*/200, /*n_steps=*/2);
        ir.round_drop = (T)1e-14;
        std::vector<T> x(p.n, 0);
        EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
        ASSERT_EQ(ir.report.cg_status.size(), 2u);
        EXPECT_EQ(ir.report.cg_status.front(), PCGInnerTermination::converged);
        EXPECT_LT(ir.report.cg_iters.front(), 200);
        EXPECT_LE(ir.report.cg_relres.front(), 1e-14);
    }
}


// Stagnation exit: a flatlined inner CG stops and returns its best iterate. The window is
// driven directly: stag_rel_improve = 1 demands an impossible 100% drop, so the exit must fire
// at exactly inner_stag_window iterations.
TEST_F(TestIterRefineLSQ, stagnating_solve_exits_early_with_best_iterate) {
    using T = double;
    // A weak R on purpose: with the exact R, CG converges in one iteration and the contrast
    // case below has nothing to measure.
    auto p = consistent_problem<T>(120, 20, 4242, 777, 31337, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const int kCap = 400, kWindow = 3;

    std::vector<T> x_stag(p.n, 0);
    {
        IterRefineLSQ<T> ir(/*abs_tol=*/1e-14, /*max_inner=*/kCap, /*n_steps=*/2);
        ir.inner_stag_window = kWindow;
        ir.stag_rel_improve  = 1.0;   // no drop counts as progress
        EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x_stag.data(), p.n), 0);
        ASSERT_EQ(ir.report.cg_status.size(), 2u);
        EXPECT_EQ(ir.report.cg_status.front(), PCGInnerTermination::stagnated);
        EXPECT_EQ(ir.report.cg_iters.front(), kWindow);
        EXPECT_EQ(ir.report.cg_relres.front(), ir.report.cg_best_relres.front());
        EXPECT_LE(ir.report.cg_best_iter.front(), ir.report.cg_iters.front());
        for (int64_t i = 0; i < p.n; ++i) EXPECT_TRUE(std::isfinite(x_stag[i]));
    }
    // Contrast: with the exit disabled the same round runs a deep solve (a 1e-14 drop, no
    // absolute floor) to convergence, taking more iterations than the window did.
    {
        IterRefineLSQ<T> ir(/*abs_tol=*/0, /*max_inner=*/kCap, /*n_steps=*/2);
        ir.inner_stag_window = 0;
        ir.round_drop = (T)1e-14;
        std::vector<T> x(p.n, 0);
        EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
        EXPECT_EQ(ir.report.cg_status.front(), PCGInnerTermination::converged);
        EXPECT_GT(ir.report.cg_iters.front(), kWindow);
        for (int64_t i = 0; i < p.n; ++i) EXPECT_NEAR(x[i], p.x_true[i], 1e-8);
    }
}


// Paced rounds: each round stops after a round_drop residual drop and the next restarts from
// the true residual. A weak R needs several rounds but converges well before the 20-round cap,
// every round stays shallow, and the solution matches x_true.
TEST_F(TestIterRefineLSQ, paced_rounds_converge_with_early_exit) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 4242, 777, 31337, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*abs_tol=*/1e-14, /*max_inner=*/200, /*n_steps=*/20);
    ir.outer_tol = (T)1e-10;
    std::vector<T> x(p.n, 0);
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_GT(ir.report.rounds_done, 1);
    EXPECT_LT(ir.report.rounds_done, 20);
    EXPECT_LE(ir.report.final_relres, (T)1e-10);
    for (auto it : ir.report.cg_iters) EXPECT_LT(it, 200);
    for (int64_t i = 0; i < p.n; ++i) EXPECT_NEAR(x[i], p.x_true[i], 1e-8);
}


// The adapter maps its members onto restarted_pcg_ne's controls: the same problem must give the
// identical iterate sequence through either interface.
TEST_F(TestIterRefineLSQ, delegation_matches_restarted_pcg_ne_exactly) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 4242, 777, 31337, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T tol_outer = (T)1e-10, tol_abs = (T)1e-14, drop = (T)1e-4;
    const int cap = 200, rounds_max = 20;

    IterRefineLSQ<T> ir(tol_abs, cap, rounds_max);
    ir.outer_tol = tol_outer;
    std::vector<T> x_ir(p.n, 0);
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x_ir.data(), p.n), 0);

    RandLAPACK::RestartedPCGNEControls<T> ctl;
    ctl.tol = tol_outer;
    ctl.max_iters = cap * rounds_max;
    ctl.max_iters_per_round = cap;
    ctl.round_drop = drop;
    ctl.max_restarts = rounds_max - 1;
    ctl.inner_abs_tol = tol_abs;
    RandLAPACK::RestartedPCGNEReport<T> rep;
    std::vector<T> x_eng(p.n, 0);
    ASSERT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x_eng.data(), ctl, rep),
              Term::converged);
    EXPECT_EQ(ir.termination_reason, Term::converged);
    EXPECT_EQ(ir.report.rounds_done, rep.rounds_done);
    EXPECT_EQ(ir.report.iters_done, rep.iters_done);
    EXPECT_EQ(x_ir, x_eng);
}


// A warm start reaches the engine: with the same controls and x0, the adapter and
// restarted_pcg_ne agree bit for bit, oracle polls included.
TEST_F(TestIterRefineLSQ, warm_start_matches_restarted_pcg_ne_exactly) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 4243, 778, 31338, (T)0.30);
    for (int64_t i = 0; i < p.m; ++i) p.b[i] += (T)0.01 * std::sin((T)i);   // inconsistent
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    std::vector<T> x0 = gels_reference(p.A, p.b, p.m, p.n);
    for (int64_t j = 0; j < p.n; ++j) x0[j] *= (T)1 + (T)1e-3 * (T)(j % 5);
    const T tol_outer = (T)1e-12, tol_abs = (T)1e-14, drop = (T)1e-4;
    const int cap = 50, rounds_max = 10;
    RandLAPACK::BackwardErrorOracle<T> oracle = [](const T*, const T*, const T*) { return (T)1; };

    IterRefineLSQ<T> ir(tol_abs, cap, rounds_max);
    ir.outer_tol = tol_outer;
    ir.round_drop = drop;
    ir.oracle = oracle;
    ir.be_tol = (T)0;
    ir.be_poll_every = 3;
    std::vector<T> x_ir(p.n, 0);
    ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x_ir.data(), p.n, x0.data());

    RandLAPACK::RestartedPCGNEControls<T> ctl;
    ctl.tol = tol_outer;
    ctl.max_iters = cap * rounds_max;
    ctl.max_iters_per_round = cap;
    ctl.round_drop = drop;
    ctl.max_restarts = rounds_max - 1;
    ctl.inner_abs_tol = tol_abs;
    ctl.oracle = oracle;
    ctl.be_tol = (T)0;
    ctl.be_poll_every = 3;
    RandLAPACK::RestartedPCGNEReport<T> rep;
    std::vector<T> x_eng(p.n, 0);
    Term t = RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x_eng.data(),
                                             ctl, rep, x0.data());
    EXPECT_EQ(ir.termination_reason, t);
    EXPECT_EQ(ir.report.rounds_done, rep.rounds_done);
    EXPECT_EQ(ir.report.iters_done, rep.iters_done);
    EXPECT_EQ(ir.report.polls, rep.polls);
    EXPECT_EQ(ir.report.be_x0, rep.be_x0);
    EXPECT_EQ(ir.report.final_relres, rep.final_relres);
    EXPECT_EQ(x_ir, x_eng);
    EXPECT_GE(rep.be_x0, (T)0);   // the oracle saw x0
}


// A converging solve is untouched by the stagnation logic: every round reports converged.
TEST_F(TestIterRefineLSQ, stagnation_exit_does_not_disturb_a_converging_solve) {
    using T = double;
    auto p = consistent_problem<T>(150, 25, 9091, 1234);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*abs_tol=*/1e-12, /*max_inner=*/200, /*n_steps=*/2);
    ir.verbose = true;   // one line per round
    std::vector<T> x(p.n, 0);
    testing::internal::CaptureStdout();
    EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    std::string out = testing::internal::GetCapturedStdout();
    EXPECT_NE(out.find("inner CG converged"), std::string::npos) << out;
    for (size_t s = 0; s < ir.report.cg_status.size(); ++s)
        EXPECT_EQ(ir.report.cg_status[s], PCGInnerTermination::converged) << "round " << s;
    for (int64_t i = 0; i < p.n; ++i) EXPECT_NEAR(x[i], p.x_true[i], 1e-8);
}


// outer_tol > 0 stops the loop as soon as the true residual meets it: with the exact R one
// round reaches far below 1e-8, so an 8-round budget is cut short.
TEST_F(TestIterRefineLSQ, outer_tol_stops_refinement_early) {
    using T = double;
    auto p = consistent_problem<T>(80, 12, 71, 72);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*abs_tol=*/1e-12, /*max_inner=*/50, /*n_steps=*/8);
    ir.outer_tol = (T)1e-8;
    std::vector<T> x(p.n, 0);
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_LT(ir.report.rounds_done, 8);
    EXPECT_GT(ir.report.rounds_done, 0);
    EXPECT_LE(ir.report.final_relres, (T)1e-8);
    EXPECT_EQ(ir.termination_reason, Term::converged);
}


// The adapter forwards inner_abs_tol: with the floor, the second round starts near it and
// stops far sooner than the unfloored round.
TEST_F(TestIterRefineLSQ, forwards_inner_abs_tol) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 4243, 778, 31338, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    auto run = [&](T abs_tol) {
        IterRefineLSQ<T> ir(abs_tol, /*max_inner=*/200, /*n_steps=*/2);
        std::vector<T> x(p.n, 0);
        EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
        return ir.report.cg_iters;
    };
    auto base = run((T)0), floored = run((T)1e-3);
    ASSERT_EQ(base.size(), 2u);
    ASSERT_EQ(floored.size(), 2u);
    EXPECT_LT(floored[1], base[1]);
}


// The adapter forwards the oracle; call() still returns 0 (only a breakdown is an error) and
// the oracle exit shows in termination_reason.
TEST_F(TestIterRefineLSQ, forwards_the_oracle) {
    using T = double;
    auto p = consistent_problem<T>(120, 20, 545, 546, 547, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    IterRefineLSQ<T> ir(/*abs_tol=*/(T)0, /*max_inner=*/200, /*n_steps=*/20);
    ir.outer_stag_window = 0;
    ir.oracle = [&](const T*, const T* r, const T*) -> T { return blas::nrm2(p.m, r, 1) / b_norm; };
    ir.be_tol = (T)1e-6;
    std::vector<T> x(p.n, 0);
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_EQ(ir.termination_reason, Term::oracle_met);
    ASSERT_EQ(ir.report.be.size(), (size_t)ir.report.rounds_done);
    EXPECT_LE(ir.report.be.back(), (T)1e-6);
}


// The adapter forwards be_poll_every; its result is bit-identical to the direct engine call.
TEST_F(TestIterRefineLSQ, forwards_be_poll_every) {
    using T = double;
    auto p = consistent_problem<T>(300, 60, 641, 642, 643, (T)0.60);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    const T b_norm = blas::nrm2(p.m, p.b.data(), 1);
    RandLAPACK::BackwardErrorOracle<T> oracle = [&](const T*, const T* r, const T*) -> T {
        return blas::nrm2(p.m, r, 1) / b_norm;
    };

    IterRefineLSQ<T> ir(/*abs_tol=*/(T)0, /*max_inner=*/400, /*n_steps=*/10, /*timing=*/true);
    ir.round_drop = (T)1e-12; ir.outer_stag_window = 0;
    ir.oracle = oracle; ir.be_tol = (T)1e-6; ir.be_poll_every = 5;
    std::vector<T> x_ir(p.n, 0);
    ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x_ir.data(), p.n);
    EXPECT_EQ(ir.termination_reason, Term::oracle_met);
    int polls = 0; for (int k : ir.report.polls) polls += k;
    EXPECT_GT(polls, 0);

    RandLAPACK::RestartedPCGNEControls<T> ctl;
    ctl.max_iters = 400 * 10; ctl.max_iters_per_round = 400; ctl.round_drop = (T)1e-12;
    ctl.max_restarts = 9; ctl.outer_stag_window = 0;
    ctl.oracle = oracle; ctl.be_tol = (T)1e-6; ctl.be_poll_every = 5;
    RandLAPACK::RestartedPCGNEReport<T> rep;
    std::vector<T> x_eng(p.n, 0);
    EXPECT_EQ(RandLAPACK::restarted_pcg_ne<T>(J, p.m, p.n, p.R.data(), p.n, p.b.data(), x_eng.data(), ctl, rep),
              Term::oracle_met);
    EXPECT_EQ(ir.report.rounds_done, rep.rounds_done);
    EXPECT_EQ(rep.polls, ir.report.polls);
    EXPECT_EQ(x_ir, x_eng);
}


// An engine breakdown is the adapter's one error return.
TEST_F(TestIterRefineLSQ, breakdown_returns_1) {
    using T = double;
    auto p = consistent_problem<T>(90, 12, 585, 586);
    std::vector<T> negA(p.A.size());
    for (size_t i = 0; i < p.A.size(); ++i) negA[i] = -p.A[i];
    lsq_test::TwoFacedOp<T> J(p.m, p.n, p.A.data(), negA.data());
    IterRefineLSQ<T> ir(1e-12, 50, 2);
    std::vector<T> x(p.n, 0);
    EXPECT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 1);
    EXPECT_EQ(ir.termination_reason, Term::breakdown);
    ASSERT_EQ(ir.report.cg_status.size(), 1u);
    EXPECT_EQ(ir.report.cg_status[0], PCGInnerTermination::breakdown);
}


// With timing on, times holds [total, inner CG, trsv, fwd, adj, other], all nonnegative,
// with other = total - inner CG - the operator time outside the inner CG. Off, it is empty.
TEST_F(TestIterRefineLSQ, timing_breakdown_is_consistent) {
    using T = double;
    auto p = consistent_problem<T>(200, 30, 811, 812, 813, (T)0.30);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(1e-12, 200, 5, /*timing=*/true);
    ir.outer_tol = (T)1e-10;
    std::vector<T> x(p.n, 0);
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    ASSERT_EQ(ir.times.size(), 6u);
    for (long t : ir.times) EXPECT_GE(t, 0L);
    const auto& h = ir.report;
    EXPECT_EQ(ir.times[0], h.times[3]);
    EXPECT_EQ(ir.times[1], h.t_inner_us);
    EXPECT_EQ(ir.times[2], h.times[2]);
    EXPECT_EQ(ir.times[3], h.times[0]);
    EXPECT_EQ(ir.times[4], h.times[1]);
    long op_outer = (h.times[0] - h.t_fwd_inner_us) + (h.times[1] - h.t_adj_inner_us)
                  + (h.times[2] - h.t_trsv_inner_us);
    EXPECT_EQ(ir.times[5], std::max(0L, ir.times[0] - ir.times[1] - op_outer));
    EXPECT_GE(ir.times[0], ir.times[1]);

    ir.timing = false;
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_TRUE(ir.times.empty());
}


TEST_F(TestIterRefineLSQ, float_solve) {
    using T = float;
    auto p = consistent_problem<T>(150, 15, 821, 822, 823, (T)0.1);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    IterRefineLSQ<T> ir(/*abs_tol=*/0, /*max_inner=*/100, /*n_steps=*/10);
    ir.outer_tol = (T)1e-5;
    std::vector<T> x(p.n, 0);
    ASSERT_EQ(ir.call(J, p.R.data(), p.n, p.b.data(), p.m, x.data(), p.n), 0);
    EXPECT_EQ(ir.termination_reason, Term::converged);
    EXPECT_LT(rel_err(x, p.x_true), 1e-3f);
}


TEST_F(TestIterRefineLSQ, invalid_arguments_throw) {
    using T = double;
    auto p = consistent_problem<T>(40, 5, 831, 832);
    DenseLinOp<T> J(p.m, p.n, p.A.data(), p.m, Layout::ColMajor);
    std::vector<T> x(p.n, 0);
    auto call = [&](IterRefineLSQ<T>& ir, int64_t m, int64_t n) {
        return ir.call(J, p.R.data(), p.n, p.b.data(), m, x.data(), n);
    };
    { IterRefineLSQ<T> ir; ir.n_refine_steps = 0;                         EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.max_inner_iters = 0;                        EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.max_inner_iters = INT_MAX; ir.n_refine_steps = 2; EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir;                                                EXPECT_THROW(call(ir, p.m - 1, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir;                                                EXPECT_THROW(call(ir, p.m, p.n + 1), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.round_drop = 0;                             EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.outer_tol = -1e-3;                          EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.outer_tol = std::numeric_limits<T>::quiet_NaN(); EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.outer_tol = std::numeric_limits<T>::infinity();  EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir; ir.stag_rel_improve = 2;                       EXPECT_THROW(call(ir, p.m, p.n), RandLAPACK::Error); }
    { IterRefineLSQ<T> ir;                                                EXPECT_THROW(ir.call(J, p.R.data(), p.n - 1, p.b.data(), p.m, x.data(), p.n), RandLAPACK::Error); }
}


// The adapter forwards outer_stag_window: on a problem whose tight tolerance is below the
// floor, the default window ends the run at the floor, and 0 runs it to the round budget.
TEST_F(TestIterRefineLSQ, forwards_outer_stag_window) {
    using T = double;
    int64_t m = 200, n = 20;
    std::vector<T> A, b;
    lsq_test::floor_problem(m, n, A, b);
    DenseLinOp<T> J(m, n, A.data(), m, Layout::ColMajor);
    auto run = [&](int window) {
        IterRefineLSQ<T> ir(/*abs_tol=*/0, /*max_inner=*/200, /*n_steps=*/4);
        ir.round_drop = (T)1e-30;
        ir.outer_tol = (T)1e-15;
        ir.outer_stag_window = window;
        std::vector<T> x(n, 0);
        EXPECT_EQ(ir.call(J, nullptr, 0, b.data(), m, x.data(), n), 0);
        return std::make_pair(ir.termination_reason, ir.report.rounds_done);
    };
    auto [floor_reason, floor_rounds] = run(2);
    EXPECT_EQ(floor_reason, Term::ls_floor);
    EXPECT_LT(floor_rounds, 4);
    auto [budget_reason, budget_rounds] = run(0);
    EXPECT_EQ(budget_reason, Term::round_budget);
    EXPECT_EQ(budget_rounds, 4);
}

// One IterRefineLSQ reused for two problems: the second call's outputs equal a fresh object's.
TEST_F(TestIterRefineLSQ, reuse_across_calls_resets_outputs) {
    using T = double;
    auto p1 = consistent_problem<T>(120, 20, 881, 882, 883, (T)0.30);
    auto p2 = consistent_problem<T>(90, 9, 884, 885, 886, (T)0.20);
    DenseLinOp<T> J1(p1.m, p1.n, p1.A.data(), p1.m, Layout::ColMajor);
    DenseLinOp<T> J2(p2.m, p2.n, p2.A.data(), p2.m, Layout::ColMajor);
    IterRefineLSQ<T> reused(1e-12, 200, 6, /*timing=*/true), fresh(1e-12, 200, 6, /*timing=*/true);
    reused.outer_tol = fresh.outer_tol = (T)1e-10;
    std::vector<T> x1(p1.n, 0), x2(p2.n, 0), x2f(p2.n, 0);
    ASSERT_EQ(reused.call(J1, p1.R.data(), p1.n, p1.b.data(), p1.m, x1.data(), p1.n), 0);
    ASSERT_EQ(reused.call(J2, p2.R.data(), p2.n, p2.b.data(), p2.m, x2.data(), p2.n), 0);
    ASSERT_EQ(fresh.call(J2, p2.R.data(), p2.n, p2.b.data(), p2.m, x2f.data(), p2.n), 0);
    EXPECT_EQ(reused.termination_reason, fresh.termination_reason);
    EXPECT_EQ(reused.report.rounds_done, fresh.report.rounds_done);
    EXPECT_EQ(reused.report.cg_iters, fresh.report.cg_iters);
    EXPECT_EQ(reused.report.ls_relres, fresh.report.ls_relres);
    EXPECT_EQ(reused.times.size(), 6u);
    EXPECT_EQ(x2, x2f);
}
