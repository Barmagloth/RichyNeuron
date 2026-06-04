# PREREG.md — Pre-registration (SPEC §5.0)

> **STATUS: NOT FILLED, NOT FROZEN.**
> The values below are **TODO** and must be consciously decided, filled in, and
> committed **before the first `sweep` run** (SPEC §5.0, §8). They are deliberately
> left blank: the SPEC's numbers (e.g. 0.95, 0.20, median) are *proposals in the
> spec text*, NOT pre-approved values — copying them here unread would defeat the
> point of pre-registration. Decide each one explicitly, then fill it.
>
> Once a sweep has started these may not change; any forced change is recorded
> below as a dated **deviation + reason**, never a silent edit (SPEC §0, §5.4).

## Registered thresholds — TO FILL BEFORE FIRST SWEEP

| Constant | Value | Meaning | SPEC proposal (for reference, not pre-approved) | Frozen? |
|---|---|---|---|---|
| `ACC_TARGET` | _TODO_ | target token-level accuracy; **calibrated on B1 only** (§5.1), then frozen | §5.0 suggests 0.95 | ☐ |
| `DELTA_MIN` | _TODO_ | min relative param reduction counted as significant; a win `< DELTA_MIN` = H0 **not** refuted | §5.0 suggests 0.20 (20%) | ☐ |
| `N_SEEDS` | _TODO_ | seeds per point | §5.0: min 3, suggests 5 | ☐ |
| Aggregation rule | _TODO_ | how a point is judged to hit `ACC_TARGET` | §5.0 suggests median-over-seed (not max, not "any seed") | ☐ |
| Budget | _TODO_ | `max_steps`, optimizer, lr-sweep — must be IDENTICAL for ALL models | §5.0: AdamW, same for RichUnit/B1/B2 | ☐ |

## Outcome definitions (SPEC §5.4) — confirm before first sweep

- **H0 refuted on a task:** `params(RichUnit) ≤ (1 − DELTA_MIN) × params(B1)`
  with BOTH meeting `ACC_TARGET` under the registered aggregation rule.
- **H1 fully supported:** H0 refuted on **both** tasks.
- **Partial:** exactly one task — "weak/unstable signal, needs replication", NOT a success.
- **H0 not refuted (incl. win < DELTA_MIN):** a full, **publishable negative result**, NOT a failure.

## Ablation gate (SPEC §6)

A negative **A1** (state path `W_h=0` does not hurt accuracy) or **A2** (alpha
collapsed to ~0 / ~1) **annuls** any claimed profit even if §5.4 formally shows a
win. **A3** requires B1/B2 trained with the same budget/optimizer/lr-sweep, with
B1 convergence curves attached.

## Deviations log

_(none yet — append dated entries here if a registered value must change)_
