# QFA micro-benchmark: block vs scalar, adaptive vs fixed (LOCAL)

Plan: `agent-workspace/randnla/project-plans/2026-09-02-qfa-micro-benchmark-plan.md`
(pass 10, FROZEN). Driver: `qfa_micro.cc`. Report/figures: `qfa_micro_report.m`.

This LOCAL micro-benchmark isolates the quadratic-form-family oracle (no Nystrom
phase, no estimator plumbing) to answer two questions the 3k_v3 campaign cannot
separate on its own: whether the joint block Krylov subspace buys accuracy over
independent per-column scalar recurrences at matched probes and matched depth,
and whether that block advantage extends to the certified stopping rule (does
the paper's joint trace certificate fire earlier than the scalar per-column
certificates, a strictly stronger requirement). Ground truth is exact per trial:
A = Q diag(lambda) Q' for a Haar-random rotation Q and a closed-form synthetic
spectrum (geo1e3, geo1e6, logu1e6), so `tr(B' f(A) B)` is known in closed form
rather than estimated. Five arm-variants share one paired probe block B per
trial across a 2x2 design (block vs scalar, fixed vs certified) plus a
`block-fixed-reorth0` control that isolates the block structure's own effect
from its reorthogonalization cost. The headline slice throughout is s=4 (s=16
is reported as a second, explicitly labeled cell, never folded into the
headline claim), and every metric is computed per (matrix, f, s, [tol], trial)
before being aggregated as a median with 25/75 spread across trials.

**Verdicts (full 3440-row sweep of 2026-09-02; all 48 reachability cells YES for
both certified arms, zero exclusions, zero certificate violations):**
- **Block vs scalar accuracy-per-depth (F1):** with reorthogonalization the block
  oracle matches scalar at shallow depth, then separates decisively and finally
  falls off a cliff to near-exact as d*s approaches n (the joint subspace
  exhausts R^n — a finite-n effect). The `block-fixed-reorth0` control sits ON
  the scalar curve almost everywhere (and never reaches 1e-4 for log1p within
  its grid): **the block advantage at these depths IS reorthogonalization**,
  which the basis-free scalar QFA structurally cannot perform, compounded by the
  exhaustion cliff. Pre-cliff, matched-reorth-free separation is modest.
- **Block certificate fires earlier than scalar (F3-right, metric 5):** YES, in
  every comparable trial at s=4 the block joint certificate stops at or before
  the slowest scalar column — in the deep (tight-tol sqrt/log1p) cells at
  d_stop ~ 140-210 where the slowest scalar column ran 470-1024 (3-5x). The
  win-count is 8/8 in every headline cell.
- **Certificate overshoot (F3-left, metric 3):** both certificates are
  reasonably tight: block 1.29-2.1x, scalar 1.45-2.5x over the interpolated
  fixed-depth oracle, with block consistently tighter at tol=1e-4. Grid-bracket
  ranges are wide where the crossing sits on the cliff (steep curve, coarse
  grid) — read the point estimates with the bracket, per the plan.
- **Scalar retirement savings (metric 4):** small — median 0-8% at the headline
  slice, at most ~25% in any trial. Per-column retirement is not a significant
  cost lever on these spectra: columns certify at nearly the same depth.
- **Go/no-go on block deflation: NO-GO.** Deflation's entire value proposition
  is recovering what retirement gives scalar — measured at ~0-25% and usually
  ~0 — while the block certificate already fires 3-5x earlier than the slowest
  scalar column without it. Not worth building for these regimes.
- **The cost the matvec axis hides:** the block certificate pays a dense
  (d*s)-sized eigensolve per ladder check — measured locally at ~26 s vs ~6 s
  per certified run (calibration cell). Matvec-cheap but flop-heavy: the block
  oracle's advantage is real where matvecs dominate (huge or expensive
  operators), and illusory where they don't. Wall-clock is reported nowhere in
  this benchmark by design; this note is from the calibration stage only.

A reader gets these verdicts, and which cells were reachable at all, from the
**a-priori reachability table** (printed first by `qfa_micro_report.m`, before
any certified-arm aggregate — it is a pure post-hoc scan of the fixed-arm sweep
and does not gate or filter any other table), then the **summary table** (half
a page, scoped to the headline slice matrix=geo1e6, s=4, one sub-table per
(f, tol)), then the three figures: **F1** (`F1_error_vs_depth.png`,
error-vs-depth for the three fixed arms, matrix=geo1e6 only, faceted by f and
s), **F2** (`F2_matvecs_to_tol.png`, matvecs-to-tol by arm across all
(matrix, s) at tol=1e-4, faceted by f), and **F3** (`F3_certificate_story.png`,
left: certificate overshoot at s=4; right: block d_stop vs scalar's slowest
column, colored by scalar retirement savings % — the figure that carries
metric 5's go/no-go claim). `rel_err_midpoint` is populated ONLY on
block-certified rows with `certified == true` (a live Radau pair survived);
every fixed-arm row and every uncertified block-certified row carries
`rel_err_midpoint = NaN` by design — a pivot-chain-death or wall-hit run
degenerates `tr_L == tr_U` exactly like the fixed arms' non-informative
stand-in, and the schema never stores that degenerate value as if it were
informative, so a NaN here is never a missing measurement, only "not
applicable." At n=8 trials per cell, medians and 25/75 spreads reported
throughout are directional/qualitative summaries, not precise confidence
intervals, and win-counts are reported alongside medians wherever a claim is
fundamentally about "how often" a count is honest where a percentile is not;
further, all trial-to-trial spread reflects PROBE (B) variance only —
`matrix_seed` fixes one Q rotation and spectrum draw per (spectrum, kappa)
cell, so realization-to-realization variance in A itself is never sampled and
is not represented in any reported spread, median, or win-count. This LOCAL
micro-benchmark characterizes oracle-level block-vs-scalar and adaptive-vs-fixed
behavior at n=1500 only; wall-reachability and orthogonality-loss depth scaling
are not verified to transfer to the 3k_v3 campaign's n=3000 or the Sia
matrices' n=10k-25k, and the verdicts here should be read as mechanism
evidence, not as a scale-extrapolated claim.
