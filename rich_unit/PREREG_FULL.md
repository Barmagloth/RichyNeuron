# PREREG_FULL.md — pre-registration for the FULL ablation (v0.4.1, A-fix-off)

> Committed BEFORE the main sweep. Spec: docs/FULL_ABLATION_SPEC_v0_4_1.md.
> Values fixed here are not changed post-hoc; a forced change is logged below as a
> dated deviation with a reason (RESEARCH_MAP §0).

## Fixed before the run
| constant | value | meaning |
|---|---|---|
| `SYN_MIN` | **0.3 bits** | synergy threshold; `median SYN(Q) ≥ SYN_MIN` on ≥ half the achievable Q (both tasks) ⇒ H0 refuted |
| `RELIABILITY_BAR` | **0.7** | a seed "succeeds" if test ≥ 0.7; reliability = fraction of seeds succeeding |
| seeds | **5** (0..4) | per (point, task, width, lr) |
| lr grid | **{1e-3, 3e-3, 1e-2}** | per-cell; lr* by median val; identical for all 4 points |
| early-stop | patience 5×200, min_delta 0.005 vs best-so-far, cap 8000, eval 200 | identical for all points |
| metric | **test_at_best** | test at the best-val step (held-out test seeds) |
| width grid | d_model {16,32,48,64} × d_state **{16}** | curve domain |

## Verdict rule (§2.4) — registered
- log2-param savings vs `neither`; `SYN(Q)=save(both)−[save(linear)+save(gate)]`.
- **Synergy (H0 refuted):** median SYN(Q) ≥ 0.3 on BOTH tasks, ≥ half achievable Q.
- **Partial:** one task → task-dependent, not a win.
- **H0 holds:** |SYN| < 0.3 → additive (full, publishable negative result).
- **Anti-synergy:** SYN ≤ −0.3 → combining hurts.
- A1: if zeroing W_h does not drop `gate`/`both` accuracy → that win is annulled.

## Q grids — TO FILL after calibration on `neither`, BEFORE the main sweep
Calibrate `neither` at a medium width on each task; pick Q inside the achievable
median-test range (not at floor/ceiling). May differ per task. Calibrated on
`neither`, never on `both`.

- selective copy: _TODO (after calibration)_
- associative recall: _TODO (after calibration)_

## Deviations log
- **2026-06-06 — d_state narrowed {8,16} → {16}.** Reason: CPU/ephemeral-container
  budget (full grid ≈ 1000 runs / ~30h+ is infeasible with container suspends).
  Sanctioned by spec §4 ("if budget jams, narrow d_state FIRST, not seeds/lr").
  Cost halved (~500 runs). May extend to d_state=8 later if curves are too thin.
