# PILOT REPORT — reconnaissance (v0.2.0)

> **This is a PILOT, not the H0 test.** No verdict on H0 is issued (SPEC §5 is
> untouched, PREREG is not frozen). The pilot only observes achievable quality
> and answers three reconnaissance questions to inform run 2. Numbers below are
> medians over **3 seeds**, Selective Copy only, budget **1200 steps**, AdamW,
> CPU. Raw rows: `results/pilot_selective_copy.csv`; full dump: `pilot_summary.json`.

## Setup actually used
- Task: Selective Copy (SPEC §3.1), token-level accuracy on the k=4 answer slots.
- Grid: `d_model ∈ {16,32,48,64,96,128}`, `d_state ∈ {4,8,16}`, seeds `{0,1,2}`.
- **Per-model lr, calibrated from a shared grid {3e-3, 1e-2}** at the reference
  config (64/8): RichUnit→**0.01**, B1→**0.003**, B2/GRU→**0.003**.
  This is the *fair* reading of "each model at its best lr from the same sweep".
- Chance level ≈ 1/8 = 0.125 (4 answer slots over 8 data symbols).
- Wall time ≈ 63 min.

## Headline numbers (median best val-acc over seeds)
| model | best median | at config | params | notes |
|---|---|---|---|---|
| **B2 / GRU** | **1.000** | d≥48 | — | sanity: task is well-posed & solvable |
| **RichUnit** | **0.709** | d128 / s16 | 39 706 | still rising with width |
| **B1 (stack)** | **0.519** | d64 / s16 | 15 722 | plateaus ~0.5, unstable at large width |

RichUnit vs B1 across width (d_state=16 column, strongest for both):

| d_model | RichUnit | params | B1 | params |
|--:|--:|--:|--:|--:|
| 16 | **0.522** | 1 402 | 0.406 | 1 658 |
| 32 | **0.603** | 3 802 | 0.483 | 4 810 |
| 48 | **0.625** | 7 226 | 0.512 | 9 498 |
| 64 | **0.658** | 11 674 | 0.519 | 15 722 |
| 96 | **0.686** | 23 642 | 0.484 | 32 778 |
| 128 | **0.709** | 39 706 | 0.350 | 55 978 |

## (a) Is RichUnit no worse than B1 by minimal width?
**Yes — and favourably, on this pilot.** At every width RichUnit ≥ B1, and it
reaches quality bands (0.6–0.7) that B1 never reaches at any width in this budget.
- To match **B1's own ceiling (~0.52)**, RichUnit needs only **d16/s16 = 1 402
  params**, vs B1's **d64/s16 = 15 722 params** — **≈11× fewer parameters**.
- RichUnit is also cheaper *per width* (its 4 matrices are lighter than B1's
  SwiGLU + temporal), so it wins on both quality and param count here.
- `b1_min_width = None`: B1 never even reaches the reconnaissance bar (0.531)
  that RichUnit clears at d32/s16.

## (b) A1 — is the state actually used? **PASS (strong).**
RichUnit at the min-width config **d32/s16**:
- normal: median **0.603** (seeds 0.646 / 0.574 / 0.603)
- `W_h = 0` frozen (state→gate path removed): median **0.130** (≈ chance)
- **Δ = 0.472.** Removing the state path collapses the model to chance → the
  "richness" carries the task, it is **not** an artifact of extra parameters.

## (c) A2 — has alpha collapsed? **No trivial collapse; long-timescale skew (flag).**
Trained α over the A1 config: range **[0.709, 0.994]**, mean **0.933**;
**0%** of mass near 0, **58%** above 0.95.
- No degenerate collapse (no unit ignores history; values are spread, not pinned
  at exactly 1). Appropriate for a 64-step memory task.
- But the distribution skews to long timescales (majority > 0.95) — **flagged**
  per SPEC §6 A2; watch for saturation in run 2, not disqualifying here.

## Caveats that shape run 2 (do NOT read this pilot as an H0 result)
1. **Low-accuracy regime.** Neither model is near GRU's 1.0; 1200 steps is short.
   The RichUnit≥B1 trend is real *as a trend*, not an at-target comparison.
2. **Baseline fairness (A3 risk).** A single shared lr is unfair — B1 diverges to
   chance at lr=1e-2 (RichUnit tolerates it). The pilot used per-model lr, but B1
   still destabilises at large width (d128/s16 → 0.350), suggesting its single
   calibrated lr=3e-3 is not optimal across the whole grid. **Run 2 must do a
   per-cell lr sweep**, or B1's edge-of-grid collapses will inflate RichUnit's win.
3. **Target calibration.** B1's stable achievable here is ~0.52 at this budget; a
   real ACC_TARGET (SPEC §5.1) should be calibrated on B1 with a longer budget +
   per-cell lr before run 2 freezes PREREG.

## What the pilot does NOT claim
- No H0 verdict. No DELTA_MIN comparison. No "RichUnit wins."
- The favourable (a) + passing A1 + clean GRU sanity say the direction is worth a
  properly-controlled run 2 — nothing more.
