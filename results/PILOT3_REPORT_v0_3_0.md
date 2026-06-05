# PILOT 3 mini REPORT — do the axes collapse? (v0.3.0)

> FROZEN spec: PILOT3_MINI_SPEC.md. Reconnaissance, NOT the H0 test, **NO synergy
> verdict**. One question (§1): with selectivity ON (input-dependent `alpha_t`,
> axis 1), does the readout `W_h·s` (axis 2) still do work, or become redundant?
> Raw: `pilot3_selective_copy.csv`, `pilot3_curves.jsonl`, `pilot3_summary.json`.

## Result: **GREEN — the axes do NOT collapse.**
With selectivity active, zeroing the readout still collapses the model to chance.
The readout is at least as load-bearing as in plain rich — if anything, more.

### Numbers (RichSel, d_model=64, d_state=16; metric = test_at_best)
- lr by median val: 0.001→0.416, 0.003→0.416, **0.01→0.978** ⇒ **lr\* = 0.01**.
- normal (per seed): 0.978 / 0.890 / 0.972 / 0.973 / **0.417** → **median 0.972**
- ablated W_h=0 (per seed): 0.131 / 0.128 / 0.118 / 0.128 / 0.128 → **median 0.128** (≈ chance)
- **per-seed Δ (paired, same init except W_h):** 0.846 / 0.761 / 0.854 / 0.844 / 0.288 → median **0.844**
- **Δ_richsel (median−median) = 0.843** ≥ threshold **0.10** ✓ — and **larger** than
  Δ_rich = 0.674 (pilot 2, same config). Selectivity did NOT make the readout
  redundant; the readout matters *more* when alpha is input-dependent.

### Decision (§5, pre-registered): GREEN → proceed to the full 4-point ablation.

## Trainability controls (§4) — checked, result is interpretable
1. **alpha is NOT degenerate.** Trained `alpha_t` over a test batch: mean 0.943,
   std-over-(batch,time) 0.105 (0% pinned at 0). Selectivity genuinely varies by
   input/position — the axis-1 path is really on, so the test is valid.
2. **Model is trained (better than rich).** normal median test 0.972 ≥ the rich
   sanity level (~0.80). The §5 "undertrained → uninterpretable" branch does not
   apply.
3. **Stability tail flagged.** 1 of 5 seeds (seed 4) stuck at 0.417 in the normal
   run (and lr 0.001/0.003 were broadly unstable). Consistent with §4's warning
   that selectivity hardens optimisation. The median rule absorbs it, and even the
   stuck seed shows a positive paired Δ (0.288), so it does not change the verdict —
   but rich+sel has a real train-stability tail to keep in mind for the full ablation.

## What this means (and does NOT)
- **Means:** the necessary condition for synergy holds — axes 1 (selectivity) and 2
  (readout) are not interchangeable; the readout keeps doing work when selectivity
  is present. The cheap gate is passed → the expensive 4-point ablation is warranted.
- **Does NOT mean synergy / super-additivity** — the mini-pilot does not test that
  (§6). No claim that combining both axes beats the sum of parts. That is exactly
  what the full ablation must measure.

## Observation, explicitly NOT the question (no HARKing)
At lr=0.01, rich+sel reaches **~0.97** test on Selective Copy — far above plain
rich's ~0.80 (pilot 2) and near GRU's 1.0. So input-dependent forgetting helps a
lot *here*. This is **not** the mini-pilot's question and **not** a verdict; if we
want to claim it, it is a separate pre-registered comparison (rich vs rich+sel at
equal params). Noted only as context: Selective Copy discriminates these variants
more than pilot 2's rich-vs-B1 plateau suggested.

## Next
Green light for the full 4-point ablation (rich / rich+sel / sel-only / neither) to
test super-additivity — with the train-stability tail handled (more seeds and/or
init/step-budget attention for rich+sel), and still no H0 verdict.
