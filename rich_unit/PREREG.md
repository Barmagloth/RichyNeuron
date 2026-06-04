# PREREG.md — Pre-registration (SPEC §5.0)

> **STATUS: TEMPLATE — NOT YET FROZEN.**
> These are the SPEC's *proposed* values. They MUST be filled, confirmed and
> committed **before the first `sweep` run** (SPEC §5.0, §8). Once a sweep has
> been started they may not be changed; any forced change is recorded below as a
> dated **deviation + reason**, never a silent edit (SPEC §0, §5.4).

## Registered thresholds

| Constant | Proposed value | Meaning | Frozen? |
|---|---|---|---|
| `ACC_TARGET` | `0.95` | target token-level accuracy; **calibrated on B1 only** (§5.1), then frozen | ☐ |
| `DELTA_MIN` | `0.20` (20%) | min relative param reduction counted as significant; a win `< DELTA_MIN` = H0 **not** refuted | ☐ |
| `N_SEEDS` | `5` | seeds per point (min 3) | ☐ |
| Aggregation rule | `median` | a point hits `ACC_TARGET` only if **median over seed ≥ target** (not max, not "any seed") | ☐ |
| Budget | `max_steps`, AdamW, lr-sweep | IDENTICAL for ALL models (RichUnit, B1, B2) | ☐ |

## Outcome definitions (SPEC §5.4) — registered ahead of time

- **H0 refuted on a task:** `params(RichUnit) ≤ (1 − DELTA_MIN) × params(B1)`
  with BOTH meeting `ACC_TARGET` under the median rule.
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
