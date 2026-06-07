# FULL_ABLATION_SPEC.md — Pareto-dominance of two state-readouts (v0.5.0, A-fix-off)

> **Status: FROZEN.** Supersedes v0.4.1. First verdict-bearing run. Builder
> implements exactly this, minimal diff. Contradiction → STOP and ask.

## AMENDMENT (v0.4.1 → v0.5.0) — primary criterion changed, documented (not silent)

**Strict super-additivity SYN over `neither` is REMOVED as the primary criterion.**
Reason: `neither` is memoryless — the only paths from state to output are the two
axes, and no neutral memory exists outside them, so with both axes off the unit is a
pure function of the current token. On memory tasks it sticks at chance (empirical:
34 `neither/sc` runs, test ∈ [0.118, 0.129] at **every** width and lr). Hence
`params@Q[neither]` is undefined and the §2.3 interaction `SYN = log2(p_lin·p_gate /
p_both) − log2(p_n)` is **uncomputable** (the `log2(p_n)` term does not cancel).

**New primary criterion: `margin` of `both` over the best single readout** (the
`log2(p_n)` terms cancel → well-posed; this was v0.4.1's exploratory metric). The
strict interaction/emergence question is NOT discarded — it is structurally
unmeasurable by *this* design (memoryless base) and is moved to a separate branch
(see `docs/EMERGENCE_BRANCH.md`). This rocade is documented, not silent.

**Terminology is now strict (binding on all A-fix-off conclusions):** the verdict is
phrased as *"both **Pareto-dominates** the best single readout by parameters, margin
X"* — NEVER "synergy" or "emergence". `margin` proves a practical benefit of
combining; it is NOT an interaction effect. The words "synergy"/"emergence" are
reserved for the EMERGENCE_BRANCH and forbidden here.

Other v0.4.1→v0.5.0: AR finalized to **n_pairs=2** (GRU solves 0.996; n≥4 unsolvable
even by GRU at this scale).

## 0. Question (A-fix-off)
Two independent ways to read one FIXED (non-selective) state, in one unit. Is the
combined unit (`both`) **more parameter-efficient at equal quality** than the better
of the two readouts alone? Goal = optimality, NOT "smarter" (HARKing forbidden).

## 1. Four points (2×2), one class — UNCHANGED from v0.4.1
`AblationUnit(d_model, d_state, linear_readout, gate_readout)`, fixed `alpha=σ(alpha_raw)`:

    s_t     = alpha·s_{t-1} + (1-alpha)·(x_t·W_s)
    gate_in = x_t·W_g + (W_h·s_t if gate_readout else 0)        # axis 2 (gate)
    y_t     = (x_t·W_v)·σ(gate_in) + (C·s_t if linear_readout else 0)   # axis 1 (linear)

| point | linear | gate | role |
|---|---|---|---|
| **neither** | off | off | **control only** (see Rule 1) — memoryless |
| **linear**  | on  | off | single readout (Mamba-style linear) |
| **gate**    | off | on  | single readout == rich (pilots 1-2); §5 anchor |
| **both**    | on  | on  | combined unit (verdict subject) |

`C` only if linear, `W_h` only if gate (param integrity). `ablate_state` zeros+freezes `W_h`.

## 2. Verdict metric — three binding rules

**Rule 1 — `neither` is control/sanity ONLY.** It is run, computed and reported, but
is the base of NO verdict metric. Its job: confirm the task truly needs memory (a
memoryless unit must sit at chance). **If `neither` is NOT at chance → the task is
solvable without memory (broken) → investigate before any verdict.**

**Rule 2 — primary verdict uses ONLY linear / gate / both.** In log2-params, with
`save(X) = −log2(params@Q[X])`:

    margin(Q) = save(both) − max(save(linear), save(gate))
              = log2(params@Q[best_single]) − log2(params@Q[both])

`best_single` = the single axis with the better (cheaper) params@Q, taken from its
**reliability-stable (median-over-seed) curve** (§3), never a lucky seed. Threshold
`MARGIN_MIN` is pre-registered (PREREG_FULL) before reading any margin.
`margin ≥ MARGIN_MIN` ⇒ `both` Pareto-dominates the best single by parameters.

**Rule 3 — inconclusive, not fitting.** If any of linear/gate/both fails to reach the
chosen Q at any width → the verdict for THAT task is **inconclusive**. No post-hoc
rescue with a different Q/band. Inconclusive is a valid, recorded outcome.

### Quality-band (pre-registered rule; resolved by calibration, not pilot memory)
The band must sit where the curves are distinguishable (different params@Q), away
from floor (trivially reached by all) and ceiling (saturation collapses differences).
- **Primary Q** = median of the Q-grid levels (step 0.05) reachable by **all three**
  of linear/gate/both at some width. **Neighbors** = primary ± 0.05 (robustness).
- Verdict on the primary band. If the **sign of margin flips** on a neighbor → soften
  the verdict to **"band-dependent"**. Bands fixed before reading margins.
- Calibration uses linear/gate/both (NOT `neither`, which is at chance).

## 3. Reliability — UNCHANGED
Identical tuning budget for all 4 points (lr grid, seeds, early-stop, init). `reliability`
= fraction of seeds with test ≥ 0.7, reported beside every curve point. lr* by median
val (equal rule for all points). "best_single" (Rule 2) uses median-stable curves.

## 4. Tasks, grids, budget — UNCHANGED except AR
- Both tasks: selective copy AND **associative recall (n_pairs=2, finalized)**. Verdict
  needs both to agree; disagreement = "task-dependent". Per task: margin within task.
- Width grid d_model {16,32,48,64}, d_state **{16}** (deviation, PREREG). Extend UP
  (d96 for ALL) only if no point reaches the band.
- lr {1e-3,3e-3,1e-2}, lr* by median val; 5 seeds; early-stop patience 5×200,
  min_delta 0.005, cap 8000, eval 200; metric test_at_best.

## 5. A1 + controls — UNCHANGED (only "synergy" wording dropped)
A1 (zero W_h, 5 seeds) on gate/both at d64/s16. §5 anchor: `gate` at d64/s16 must
reproduce pilot 2's rich (~0.80 test, A1 Δ~0.67); else the stand drifted. A1 checks
gate-path fictitiousness; it does NOT decide Pareto-dominance (the curves do). Log
fixed `alpha` per point (collapse to 0/1 = degenerate).

## 6. B2/GRU anchor — UNCHANGED. d32,d64, 3 seeds, both tasks; sanity, not in margin.

## 7. Files
    rich_unit/PREREG_FULL.md, models/ablation_unit_v0_4_1.py, full_ablation_v0_4_1.py
    rich_unit/report_full_v0_5_0.py        # margin verdict (replaces report_full_v0_4_1)
    docs/EMERGENCE_BRANCH.md               # deferred strict-interaction question (stub)

## 8. Definition of Done
Verdict per Rule 2 on both tasks (margin vs MARGIN_MIN at the pre-registered band,
+ reliability + A1), any sign, phrased as Pareto-dominance (never synergy). Failure =
only dishonesty: post-hoc MARGIN_MIN/band, asymmetric tuning, dropped seeds,
dominance claimed without A1, verdict on one task instead of two, or "synergy" wording.
