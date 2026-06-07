# PREREG_FULL.md — pre-registration for the FULL ablation (v0.5.0, A-fix-off)

> Committed BEFORE reading any margin. Spec: docs/FULL_ABLATION_SPEC_v0_5_0.md.
> Values fixed here are not changed post-hoc; a forced change is logged below as a
> dated deviation with a reason (RESEARCH_MAP §0).

## Primary verdict criterion (v0.5.0) — registered
`save(X) = −log2(params@Q[X])`;  `margin(Q) = save(both) − max(save(linear), save(gate))`
= `log2(params@Q[best_single]) − log2(params@Q[both])`  (the `neither` term cancels).

| constant | value | meaning |
|---|---|---|
| `MARGIN_MIN` | **0.5 bits** (≈1.4× fewer params) | `margin ≥ MARGIN_MIN` ⇒ `both` Pareto-dominates the best single by parameters. Justification: above the ~0.3-bit interpolation-noise floor of the width grid; practically significant; consistent with the project's earlier DELTA_MIN=30% (≈0.5 bit). Engineering proposal, not a law of nature. |
| `RELIABILITY_BAR` | **0.7** | a seed "succeeds" if test ≥ 0.7; reliability = fraction succeeding; reported beside every curve point. `best_single` uses the median-stable curve, never a lucky seed. |
| seeds | **5** (0..4) | per (point, task, width, lr) |
| lr grid | **{1e-3, 3e-3, 1e-2}** | per-cell; lr* by median val; identical for all 4 points |
| early-stop | patience 5×200, min_delta 0.005 vs best-so-far, cap 8000, eval 200 | identical for all points |
| metric | **test_at_best** | test at the best-val step (held-out test seeds) |
| width grid | d_model {16,32,48,64} × d_state **{16}** | curve domain |

### Quality-band — pre-registered RULE (resolved by calibration, before reading margins)
- **Primary Q** = median of the Q-grid levels (step 0.05) reachable by **all three** of
  linear/gate/both at some width (NOT calibrated on `neither`, which is at chance).
- **Neighbors** = primary ± 0.05 (robustness). Verdict on the primary band; if the
  **sign of margin flips** on a neighbor → soften to **"band-dependent"**.
- Bands are fixed by this rule once the sweep is in; not hand-picked per result.

## Three rules (spec §2)
1. **`neither` = control only.** Reported; base of no verdict metric. If `neither` is
   NOT at chance → task solvable without memory (broken) → investigate before verdict.
2. **Primary verdict on linear/gate/both only**, via `margin` ≥ `MARGIN_MIN` at the
   primary band.
3. **Inconclusive, not fitting.** If any of linear/gate/both never reaches the chosen
   Q → that task's verdict = inconclusive (no post-hoc Q/band rescue).

## Terminology (binding)
Verdict phrased as **"both Pareto-dominates the best single readout by parameters,
margin X"**. The words "synergy"/"emergence" are FORBIDDEN in A-fix-off conclusions
(reserved for docs/EMERGENCE_BRANCH.md). margin proves practical benefit, not interaction.

## Controls
- A1 (zero W_h, 5 seeds) on gate/both at d64/s16; fictitious gate-path → win annulled.
- §5 anchor: `gate` at d64/s16 reproduces pilot 2 rich (~0.80 test, A1 Δ~0.67); else stand drifted.
- Tasks: selective copy + associative recall (n_pairs=2, GRU-validated 0.996). Both must agree.

## Deviations log
- **2026-06-06 — d_state narrowed {8,16} → {16}.** CPU/ephemeral-container budget;
  sanctioned by spec §4 (narrow d_state first, not seeds/lr). Cost halved.
- **2026-06-06 — strict SYN over `neither` demoted from primary to deferred branch.**
  Reason: `neither` memoryless → params@Q[neither] undefined → SYN uncomputable
  (empirical: 34 neither/sc runs at chance [0.118,0.129]). Primary = margin (well-posed).
  Full amendment in spec v0.5.0; strict question parked in EMERGENCE_BRANCH.md.
- **2026-06-06 — AR n_pairs 8 → 2.** The v0.2.0 proposal (8) is unsolvable even by GRU
  at this scale; n_pairs=2 is GRU-validated (0.996) and a genuine memory+association test.
