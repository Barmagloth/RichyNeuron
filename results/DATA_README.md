# Reconnaissance data — README for external analysis

Two reconnaissance pilots on the "rich unit" question (does fusing temporal+channel
into one unit beat a split stack at equal parameters). **Both are reconnaissance,
not the H0 test** — no pre-registered verdict. Task: Selective Copy (synthetic,
seeded). CPU/PyTorch. All numbers are per-run; aggregate yourself if you prefer
other statistics.

## Models
- `rich`  — RichUnitLayer: `s_t=α·s_{t-1}+(1-α)·(x_tW_s)`, `g=σ(x_tW_g+s_tW_h)`, `y_t=(x_tW_v)·g`.
- `B1`    — stacked baseline: linear diagonal recurrent (no gate) → SwiGLU (stateless).
- `B2`    — single `nn.GRU` layer (sanity anchor).
All wrapped identically: `embed → core → linear head`. `params` column = trainable
params of the whole wrapped model.

## Task (Selective Copy)
T=64; alphabet: blank=0, data symbols 1..8, marker=9 (vocab=10). k=4 data tokens at
random positions; after the marker the model must emit them in order. Accuracy =
token-level over the k answer slots only (rest of the sequence is ignore_index).
Seed namespaces (disjoint): train `1000 + seed*100000 + step`; val `2_000_000+i`;
test `3_000_000+i` (8 batches each, batch 64). Same val/test seeds for all models.

## Files
| file | what |
|---|---|
| `pilot_selective_copy.csv`  | PILOT 1 raw runs (126) |
| `pilot_summary.json`        | PILOT 1 aggregates + A1/A2 |
| `PILOT_REPORT_v0_2_0.md`    | PILOT 1 writeup (interpretation) |
| `pilot_run.log`             | PILOT 1 stdout |
| `pilot2_selective_copy.csv` | PILOT 2 raw runs (246) |
| `pilot2_summary.json`       | PILOT 2 aggregates + A1/A2 |
| `PILOT2_REPORT_v0_2_0.md`   | PILOT 2 writeup (interpretation) |
| `PILOT2_ANALYSIS_RUBRIC.md` | anti-optimism interpretation rules (fixed pre-results) |
| `pilot2_run.log`            | PILOT 2 stdout |
| `code/`                     | exact source that produced the data (rich_unit/) |

## CSV schemas
**Pilot 1** `model,d_model,d_state,lr,seed,params,best_val_acc,steps_to_best`
- grid: d_model{16,32,48,64,96,128} × d_state{4,8,16} (B2: no d_state), 3 seeds.
- **single lr per model** (rich=0.01, B1=0.003, B2=0.003), **fixed 1200 steps**.
- `best_val_acc` = best val token-acc over the run (val = best-of-evals). No test split.

**Pilot 2** `model,d_model,d_state,lr,seed,params,best_val,test_at_best,steps_to_best,stopped_step`
- grid: d_model{16,32,48,64} × d_state{8,16}, **lr swept {1e-3,3e-3,1e-2}**, 5 seeds.
  (B2 anchor: d_model{32,64}, lr=3e-3, 3 seeds.)
- **early-stop**: stop when val has not improved by > `min_delta=0.005` over
  best-so-far for `patience=5` evals (eval every 200 steps); cap `max_steps=8000`.
- `best_val` = best val token-acc over the run. `test_at_best` = **held-out test
  token-acc at the step where val was best** (NOT argmax over test). `steps_to_best`
  = step of best val. `stopped_step` = where training stopped (early-stop or cap).

## Aggregation used in the reports (reproduce or replace)
Per cell (model,d_model,d_state): **choose lr by median val over seeds**, then report
**median `test_at_best`** of that lr's seeds. (lr-selection on val, reporting on test.)
Pilot 1 used median `best_val_acc` over its 3 seeds at the single lr.

## Ablations (in `*_summary.json` under `A1A2` / `A1`)
- A1: re-train the winning rich config with `W_h=0` frozen; compare accuracy.
- A2: distribution of trained `alpha` (=σ(alpha_raw)); fraction near 0 / near 1.

## Reproduce
```
pip install torch numpy
python -m rich_unit.pilot_v0_2_0     # pilot 1
python -m rich_unit.pilot2_v0_2_0    # pilot 2 (resumable; RICH_UNIT_CHECKPOINT_PUSH=0 to disable git pushes)
```

## Caveats the analyst should know
- Low-accuracy regime: rich/B1 cap ~0.74–0.79 while B2/GRU solves at ~1.0.
- Numbers are val (pilot 1) vs held-out test (pilot 2) — not identical metrics.
- Seed variance is large on narrow configs; treat small gaps as noise (use the
  per-seed rows, not just medians).
