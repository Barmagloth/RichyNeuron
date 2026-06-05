# PILOT 2 REPORT — does the gap survive convergence + fair lr? (v0.2.0)

> **Reconnaissance, NOT the H0 test.** No H0 verdict, PREREG stays unfrozen. One
> task (Selective Copy). Metric = `test_at_best` (held-out test at the best-VAL
> step); lr chosen per cell by **median val**; 5 seeds; early-stop on a val
> plateau (patience 5×200 steps, min_delta 0.005 vs best-so-far, cap 8000).
> Read `results/PILOT2_ANALYSIS_RUBRIC.md` first. Raw: `pilot2_selective_copy.csv`,
> `pilot2_summary.json`.

## The question
Pilot 1 measured a large rich-vs-B1 gap in an **under-trained** regime (fixed 1200
steps, single lr) — the least trustworthy place to compare. Pilot 2 lets both
models converge under a fair, identical budget and a per-cell lr sweep, and asks:
**does the gap survive, or was it an artifact?**

## Answer: the gap SHRANK sharply but did NOT fully collapse.
A small, seed-robust rich edge persists on the wider configs; the parameter
advantage is **~2.5×**, down from pilot 1's ~11×.

### Converged comparison (median test over 5 seeds; [min–max])
| dm/ds | rich (params) | B1 (params) | gap |
|--:|--|--|--:|
| 16/8  | 0.559 [.46–.61] (1138) | 0.532 [.43–.58] (1386) | +0.026 |
| 16/16 | 0.642 [.47–.65] (1402) | **0.686** [.67–.70] (1658) | −0.044 |
| 32/8  | 0.639 [.61–.70] (3282) | 0.560 [.53–.65] (4282) | +0.079 |
| 32/16 | 0.744 [.74–.75] (3802) | 0.695 [.67–.74] (4810) | +0.049 |
| 48/8  | 0.634 [.61–.71] (6450) | 0.562 [.53–.67] (8714) | +0.072 |
| 48/16 | 0.785 [.74–.79] (7226) | 0.738 [.70–.75] (9498) | +0.046 |
| 64/8  | 0.666 [.63–.72] (10642)| 0.572 [.47–.61] (14682)| +0.095 |
| 64/16 | **0.793** [.78–.82] (11674) | 0.738 [.69–.75] (15722) | +0.055 |

- **Ceilings:** rich 0.793 (64/16, 11674 params); B1 0.738.
- **Param efficiency:** B1 reaches its ceiling (0.738) at its **cheapest** such
  config = 48/16 = **9498** params; rich matches 0.738 at **3802** (32/16) →
  **2.5×**. *(Correction: an earlier mid-run note said ~4.1× — that wrongly used
  B1's more expensive tied ceiling cell 64/16=15722. The honest ratio vs B1's
  cheapest ceiling is 2.5×.)*

### How much the gap moved (same cells, pilot 1 → pilot 2)
- rich 32/16 vs B1: **+0.120 → +0.049**; rich 16/16 vs B1: **+0.116 → −0.044**.
- Param advantage: **~11× → ~2.5×**. The pilot-1 lead was mostly an
  under-training + lr-fairness artifact, exactly as suspected.

### Where the edge is real vs noise
- **Real (seed-robust):** wide ds16 configs. At 64/16 the seed ranges **do not
  overlap** (rich [.78–.82] vs B1 [.69–.75]); 48/16, 32/16 barely overlap.
- **Noise:** narrow configs. rich 16/16 spans [.47–.65] (0.18!); B1's lone win
  there is inside that spread. ds8 gaps are larger (+.07–.095) but also noisier.

## A1 — is the state load-bearing on the CONVERGED model? **PASS (strong).**
Winning config rich 64/16 @ lr0.01:
- normal: median test **0.803** (seeds .78–.82)
- `W_h=0` frozen: median test **0.128** (≈ chance 0.125)
- **Δ = 0.674.** Δ *grew* vs pilot 1 (0.47 → 0.67): the converged model relies on
  the state path MORE, not less. Decisive point (rubric #3): the ablated model
  sits **at chance**, so the state is load-bearing regardless of Δ. The rich
  edge, such as it is, is **not** a dead-feature/extra-params artifact.

## A2 — alpha collapse? **No trivial collapse; pronounced long-timescale skew (flag).**
α ∈ [0.647, 0.996], mean 0.941; **0%** near 0, **71%** above 0.95 (was 58% in
pilot 1). No unit ignores history and values aren't pinned at exactly 1, but the
skew toward very long memory is strong — flagged per SPEC §6 A2. Expected for a
64-step memory task; watch for saturation.

## Convergence speed — NO advantage (rubric #2).
Median steps_to_best: rich **4400**, B1 **4700** — effectively the same. There is
no "rich converges faster" story; and per rubric #2, steps_to_best under early-stop
is loaded (mixes real speed with when patience fired), so even this near-tie is not
claimed as a result. (Full val curves were not persisted, so curve-shape
verification is unavailable — do not assert a speed difference.)

## Hard caveats (bound the strength of the conclusion)
1. **Low-accuracy regime.** Both rich (~0.79) and B1 (~0.74) sit far below
   **B2/GRU = 1.0** (anchor: dm32→0.996, dm64→1.000). The comparison is in a
   regime where neither solves the task — so this is "rich slightly better at a
   *partial* solution", not an at-target efficiency win. Whether the 2.5× holds at
   a real ACC_TARGET is unknown (neither reaches one here).
2. **2.5× is "match B1's ceiling", not the pre-registered "min params @
   ACC_TARGET with DELTA_MIN".** This is reconnaissance, not H0.
3. **One task.** Selective Copy only.

## Net read & implications for run FULL
A **real but modest** signal survives convergence: ~2.5× param efficiency to match
B1's ceiling, a seed-robust +0.05 edge on wide configs, with the state mechanism
confirmed functional (A1) — but in a low-accuracy regime, with no convergence-speed
edge, on one task. This is far weaker than pilot 1 implied and does **not** justify
inflated FULL expectations; it does justify a properly-controlled FULL (both tasks,
pre-registered ACC_TARGET/DELTA_MIN, per-cell lr sweep) with calibrated-down hopes.

- **ACC_TARGET calibration (SPEC §5.1):** B1's converged achievable on Selective
  Copy ≈ **0.74**. But because GRU hits 1.0 while rich/B1 cap ~0.74–0.79, a target
  near B1's ceiling sits in a regime neither architecture solves — consider
  whether Selective Copy at this size is the right separating task, or whether the
  readout/budget needs revisiting, before freezing PREREG.
