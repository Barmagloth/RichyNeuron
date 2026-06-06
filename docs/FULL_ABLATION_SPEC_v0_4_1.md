# FULL_ABLATION_SPEC.md — Synergy of two state-READOUT axes (v0.4.1, A-fix-off)

> **Status: FROZEN.** First verdict-bearing run (not reconnaissance). Builder
> implements exactly this, minimal diff. Contradiction → STOP and ask. Read with
> RESEARCH_MAP §0 (anti-bias) — all its rules hold.
>
> **v0.4.1 supersedes the uploaded v0.4.0.** The v0.4.0 `sel` point was degenerate
> (with the only state→output path being the gate, `selectivity on / readout off`
> collapsed to `neither`). Resolution (author-approved): the 2×2 now toggles two
> independent ways to READ the state; `alpha` is FIXED in all points (this run =
> **A-fix-off**); selectivity/`W_alpha` is a separate write-state factor, deferred
> to a later **A-fix-on** run (§7). The "sel = mamba-min" label is withdrawn.

## 0. Question
Two independent ways to read one fixed (non-selective) state, combined in one unit:
- **Axis 1 — linear readout:** `y_t += C·s_t` (the Mamba *readout* mechanism; not all of Mamba).
- **Axis 2 — gate readout:** `g = σ(W_g·x + W_h·s_t)` (the rich mechanism; pilots 1-2).

**H1 (central):** combining both readouts in one unit is **super-additive** — saves
more parameters than the sum of each readout alone. **H0 (default, against us):**
the axes are additive. Goal = optimality (params at equal quality), NOT "smarter"
(HARKing forbidden, RESEARCH_MAP §0).

## 1. Four points (2×2), one class
`AblationUnit(d_model, d_state, linear_readout: bool, gate_readout: bool)`:

    alpha   = σ(alpha_raw)                                   # FIXED, all points
    s_t     = alpha·s_{t-1} + (1-alpha)·(x_t·W_s)
    gate_in = x_t·W_g + (W_h·s_t if gate_readout else 0)         # axis 2
    g       = σ(gate_in)
    y_t     = (x_t·W_v)·g + (C·s_t if linear_readout else 0)     # axis 1

| point | linear | gate | == past |
|---|---|---|---|
| **neither** | off | off | — |
| **linear**  | on  | off | — (Mamba-style linear readout, fixed alpha) |
| **gate**    | off | on  | **rich (pilots 1-2)** — §5 reproducibility anchor |
| **both**    | on  | on  | — (NOT pilot 3; pilot 3 had selectivity) |

- All 4 are ONE unit with two flags → only the two axes differ, nothing drifts.
- `C` exists only if `linear_readout`; `W_h` only if `gate_readout` (no dangling
  params, else params@Q comparison breaks).
- `ablate_state` (A1): zero+freeze `W_h` (only meaningful for gate/both).
- B1/B2 do NOT participate in the synergy test (B2 anchor only, §6).

## 2. Synergy metric — curves params→quality, not single points
Accuracy saturates at 1.0, so fixed-width Δacc is mechanically sub-additive near
the ceiling. Work on curves instead.

- **2.1** For each point × task: sweep width → set of (params, median-test). 
- **2.2** Quality grid `Q` (calibrated, §4). `params@Q` = min params with
  median-test ≥ Q (log-linear interpolation along the curve allowed).
- **2.3** In **log2-params** (param savings are multiplicative):
  `save(X) = log2(params@Q[neither]) − log2(params@Q[X])`
  `SYN(Q) = save(both) − [save(linear) + save(gate)]`
  SYN>0 super-additive (synergy); ≈0 additive (H0); <0 anti-synergy.
- **2.4 Decision (pre-registered, PREREG_FULL):**
  - `SYN_MIN` = **0.3 bits** (≈ both ~1.23× cheaper than the sum).
  - **H0 refuted (synergy):** median SYN(Q) ≥ SYN_MIN on **both** tasks, on ≥ half
    the achievable Q levels.
  - **Partial:** one task only → "task-dependent", not a win.
  - **H0 holds:** |SYN| < SYN_MIN → additive (full negative result).
  - **Anti-synergy:** SYN ≤ −SYN_MIN → combining hurts (substantive result).

## 3. Reliability — trainability is part of the result
- **Identical tuning budget for all 4 points** (same lr grid, seeds, early-stop,
  init protocol). No point gets extra lr/seeds/steps. Asymmetry = corruption.
- `reliability ∈ [0,1]` per (point, width, lr*) = fraction of seeds reaching a
  quality bar (≥0.7). Reported ALONGSIDE every curve point; unstable points flagged.
- lr* by median val; if reliability is low at the best-median lr, prefer another lr
  with the same median but higher reliability — applied EQUALLY to all points.

## 4. Tasks, grids, budget
- **Both tasks:** selective copy AND associative recall. Verdict only if both agree
  (else "task-dependent"). SYN computed within a task.
- **Q calibration (before main sweep):** run ONLY `neither` at a medium width on
  each task; pick the `Q` grid inside its achievable median-test range (not at
  floor/ceiling). Tasks may have different Q grids. Calibrated on `neither`, never
  on `both` (anti-pro-H1 gaming).
- **Width grid:** `d_model ∈ {16,32,48,64}`, **`d_state ∈ {16}`** (see deviation).
  Extend UP (add d96 for ALL points) only if no point reaches top Q.
- **lr sweep per-cell:** `{1e-3, 3e-3, 1e-2}`, lr* by median val.
- **seed = 5.** early-stop: patience 5 evals, min_delta 0.005 vs best-so-far, cap
  8000, eval_every 200. Metric **test_at_best** (test at the best-val step).
- **DEVIATION (sanctioned by v0.4.0 §4):** d_state narrowed to {16} (from {8,16})
  to fit the CPU/ephemeral-container budget — halves cost. Documented in PREREG_FULL.
- **Checkpoint/resume** keyed by (point, task, d_model, d_state, lr, seed) + jsonl
  sidecar for curves; container reclaim safe.

## 5. A1 + state-mechanism control
- A1 (zero W_h, 5 seeds) on `gate` and `both` at their winning width. If zeroing
  does NOT drop accuracy → gate readout fictitious → that point's win annulled.
- **§5 reproducibility anchor:** with fixed alpha, `gate` at d64/s16... = at the
  comparable config **must reproduce pilot 2's rich** (test ~0.80, A1 Δ~0.67). If
  not → the stand drifted; investigate BEFORE any verdict. (The "both ≡ pilot 3"
  check is removed — pilot 3 had selectivity, irrelevant to A-fix-off.)
- redundancy vs synergy is decided by the SYN curves, NOT by A1.
- Log the fixed `alpha` distribution per point (collapse to 0/1 = degenerate state).

## 6. Reproducibility anchor (cheap, outside synergy)
B2/GRU on d32,d64 (known lr, 3 seeds) on BOTH tasks — confirm tasks solvable and
the stand has not degraded. NOT part of SYN.

## 7. Files
    rich_unit/
      PREREG_FULL.md                  # SYN_MIN, Q grids (post-calib), reliability bar
      models/ablation_unit_v0_4_1.py  # AblationUnit(linear_readout, gate_readout)
      full_ablation_v0_4_1.py         # calib + 4 points × 2 tasks × grid + A1 + anchor
      report_full_v0_4_1.py           # curves, params@Q, SYN(Q), reliability, verdict
      results/full_*.csv, full_curves.jsonl, full_summary.json
      tests/test_ablation_unit_v0_4_1.py

### Next run (NOT this one): A-fix-on
Selectivity returns as a flag `selectivity` (alpha_t = σ(x·W_alpha)); separate
PREREG; tests whether selectivity amplifies any readout synergy found here. One
factor at a time — not mixed into this verdict run.

## 8. Definition of Done
Verdict per §2.4 (SYN on both tasks, with reliability + A1 control), any sign.
Failure = only dishonesty: post-hoc SYN_MIN/Q, asymmetric tuning, dropped seeds,
synergy without passing A1, verdict on one task instead of two.
